#!/usr/bin/env python3
"""
gpu_eval.py -- GPU-parallel batch evaluator for torso decompositions.

The CPU fill-in evaluator (core.evaluate / cmaes_torso.eval_fitness) is the
throughput bottleneck on the dense large-graph: each ordering's elimination is
~O(n * deg * n/word) sequential bitword work. This module evaluates a WHOLE
POPULATION of orderings in parallel on the GPU -- one ordering per CUDA thread --
so a search can score tens of thousands of candidates per generation instead of
a few dozen. This is the throughput step-change THESIS.md s9 identifies as the
only path past the CPU evaluation wall.

Semantics mirror cmaes_torso.eval_fitness EXACTLY (validated bit-for-bit by
tools/validate_gpu.py against core.evaluate):

  * one fill-in elimination pass produces deg[i] for every step i,
  * the cap (MAX_TW=500) is checked at every step; a graded penalty
    (501, then +(deg-500) per extra overflow) mirrors cuda-torso's libeval.cu
    and gives CMA-ES a feasibility gradient,
  * the pass BAILS once penalty > 1000 (deeply infeasible) -- the same early
    stop that keeps the CPU evaluator fast on bad perms,
  * a perm is feasible iff no step exceeds the cap; its front is then
    width(t) = suffix-max(deg[t:]) over all t.

Public API:
  build_adj_words(n, adj_bits)      -> (adj_words uint32[n,W], W)
  GpuEvaluator(adj_words, n, W, cap)
      .eval(perms_int32[P,n])       -> (deg int32[P,n], status int8[P])
  cpu_eval_batch(perms, adj_words, n, W, cap) -> (deg, status)   # numpy ref
  status codes: 0 feasible, 1 mildly infeasible, 2 bailed (deeply infeasible)

The GPU path requires `numba` with CUDA; if unavailable, callers should fall
back to cpu_eval_batch (correct but not fast).
"""
from __future__ import annotations
import numpy as np

MAX_W = 96          # max 32-bit words per row -> supports n up to 96*32 = 3072
                    # (large-graph n=2426 -> W=76). Bump if you add bigger graphs.


# --------------------------------------------------------------------------- #
# Bitset packing
# --------------------------------------------------------------------------- #
def build_adj_words(n: int, adj_bits) -> tuple[np.ndarray, int]:
    """Pack the per-vertex python-int adjacency bitsets into a uint32[n, W]
    little-endian word matrix (word w holds bits [32w, 32w+32))."""
    W = (n + 31) // 32
    if W > MAX_W:
        raise ValueError(f"n={n} needs W={W} > MAX_W={MAX_W}; bump MAX_W.")
    adj = np.zeros((n, W), dtype=np.uint32)
    for u in range(n):
        b = int(adj_bits[u])
        w = 0
        while b:
            adj[u, w] = b & 0xFFFFFFFF
            b >>= 32
            w += 1
    return adj, W


# --------------------------------------------------------------------------- #
# CPU reference (numpy) -- the ground truth the GPU kernel must match.
# Mirrors eval_fitness's deg/penalty/bail logic step for step.
# --------------------------------------------------------------------------- #
def cpu_eval_batch(perms: np.ndarray, adj_words: np.ndarray, n: int, W: int,
                   cap: int = 500):
    """Reference batch evaluator using python-int bitsets (exact, slow).
    Returns (deg int32[P,n], status int8[P])."""
    P = perms.shape[0]
    # reconstruct python-int adjacency from the word matrix (so this ref is
    # self-contained and validates the packing too)
    adj_int = [0] * n
    for u in range(n):
        b = 0
        for w in range(W):
            b |= int(adj_words[u, w]) << (32 * w)
        adj_int[u] = b
    deg = np.zeros((P, n), dtype=np.int32)
    status = np.zeros(P, dtype=np.int8)
    full = (1 << n) - 1
    for p in range(P):
        perm = perms[p]
        rem = full
        temp = list(adj_int)
        penalty = 0
        bailed = False
        for i in range(n):
            u = int(perm[i])
            rem &= ~(1 << u)                 # rem = positions strictly after i
            succ = temp[u] & rem
            d = succ.bit_count()
            deg[p, i] = d
            if d > cap:
                penalty = 501 if penalty == 0 else penalty + (d - cap)
                if penalty > 1000:
                    status[p] = 2
                    bailed = True
                    break
            if succ:
                s = succ
                while s:
                    vbit = s & -s
                    s ^= vbit
                    v = vbit.bit_length() - 1
                    temp[v] |= succ ^ vbit
        if not bailed:
            status[p] = 1 if penalty > 0 else 0
    return deg, status


# --------------------------------------------------------------------------- #
# GPU kernel (numba CUDA) -- one ordering per thread.
# --------------------------------------------------------------------------- #
_KERNEL = None


def _build_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from numba import cuda, uint32, int32

    @cuda.jit(cache=True)
    def eval_kernel(perms, adj, work, deg, status, n, W, cap):
        p = cuda.grid(1)
        if p >= perms.shape[0]:
            return
        rem = cuda.local.array(MAX_W, uint32)
        succ = cuda.local.array(MAX_W, uint32)

        # init working adjacency = copy of adj; rem = all-ones over n bits
        for w in range(W):
            full = uint32(0xFFFFFFFF)
            rem[w] = full
        # mask off the high bits beyond n in the last word
        extra = W * 32 - n
        if extra > 0:
            rem[W - 1] = uint32(0xFFFFFFFF) >> extra
        for r in range(n):
            for w in range(W):
                work[p, r, w] = adj[r, w]

        penalty = int32(0)
        st = int32(0)
        for i in range(n):
            u = perms[p, i]
            # rem = positions strictly after i: clear bit u
            wu = u >> 5
            bu = u & 31
            rem[wu] &= ~(uint32(1) << bu)
            # succ = work[u] & rem ; d = popcount
            d = int32(0)
            for w in range(W):
                sc = work[p, u, w] & rem[w]
                succ[w] = sc
                d += cuda.popc(sc)
            deg[p, i] = d
            if d > cap:
                if penalty == 0:
                    penalty = int32(501)
                else:
                    penalty += (d - cap)
                if penalty > 1000:
                    st = int32(2)
                    break
            # fill-in: for each v in succ, work[v] |= (succ \ {v})
            for w in range(W):
                sc = succ[w]
                while sc != 0:
                    idx = cuda.ffs(sc) - 1          # 0-based bit in this word
                    vbit = uint32(1) << idx
                    v = (w << 5) + idx
                    for ww in range(W):
                        work[p, v, ww] |= succ[ww]
                    work[p, v, w] &= ~vbit          # a vertex is not its own nbr
                    sc ^= vbit
        if st != 2:
            st = int32(1) if penalty > 0 else int32(0)
        status[p] = st

    _KERNEL = eval_kernel
    return _KERNEL


class GpuEvaluator:
    """Holds the device-side adjacency + a reusable per-batch work buffer.

    Usage:
        ev = GpuEvaluator(adj_words, n, W, cap=500, capacity=4096)
        deg, status = ev.eval(perms_int32)     # perms.shape = (P<=capacity, n)
    """

    def __init__(self, adj_words: np.ndarray, n: int, W: int, cap: int = 500,
                 capacity: int = 4096):
        from numba import cuda
        self.cuda = cuda
        self.n, self.W, self.cap = n, W, cap
        self.kernel = _build_kernel()
        self.d_adj = cuda.to_device(np.ascontiguousarray(adj_words, dtype=np.uint32))
        self.capacity = int(capacity)
        # work buffer: capacity x n x W uint32 (the dominant allocation)
        self.d_work = cuda.device_array((self.capacity, n, W), dtype=np.uint32)
        self.d_deg = cuda.device_array((self.capacity, n), dtype=np.int32)
        self.d_status = cuda.device_array((self.capacity,), dtype=np.int8)

    @staticmethod
    def work_bytes(capacity, n, W):
        return capacity * n * W * 4

    def eval(self, perms: np.ndarray):
        """Evaluate up to `capacity` orderings. Returns (deg[P,n], status[P])."""
        perms = np.ascontiguousarray(perms, dtype=np.int32)
        P = perms.shape[0]
        if P > self.capacity:
            raise ValueError(f"P={P} > capacity={self.capacity}; raise capacity "
                             "or batch your population.")
        d_perms = self.cuda.to_device(perms)
        threads = 64
        blocks = (P + threads - 1) // threads
        self.kernel[blocks, threads](d_perms, self.d_adj, self.d_work,
                                     self.d_deg, self.d_status,
                                     self.n, self.W, self.cap)
        self.cuda.synchronize()
        deg = self.d_deg[:P].copy_to_host()
        status = self.d_status[:P].copy_to_host()
        return deg, status


# --------------------------------------------------------------------------- #
# Host-side scoring from deg[] (shared by GPU and CPU paths)
# --------------------------------------------------------------------------- #
def staircase_widths(deg: np.ndarray) -> np.ndarray:
    """width(t) = suffix-max(deg[t:]) for every t, vectorised over the batch.
    deg: int32[P,n] -> widths int32[P,n]."""
    return np.maximum.accumulate(deg[:, ::-1], axis=1)[:, ::-1]


if __name__ == "__main__":
    # tiny self-test of the numpy reference vs core.evaluate
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core import load_graph, build_adj_bitsets, graph_path, repo_root, evaluate, MAX_TW
    here = repo_root()
    n, adj = load_graph(graph_path(here, "small-graph"))
    ab = build_adj_bitsets(n, adj)
    aw, W = build_adj_words(n, ab)
    rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(n) for _ in range(8)], dtype=np.int32)
    deg, status = cpu_eval_batch(perms, aw, n, W, MAX_TW)
    widths = staircase_widths(deg)
    ok = True
    for p in range(perms.shape[0]):
        for t in (0, n // 3, n // 2, n - 1):
            w_ref, _ = evaluate(perms[p].tolist(), t, ab, n)
            w_gpu = int(widths[p, t]) if status[p] == 0 else MAX_TW + 1
            if status[p] == 0 and w_ref != w_gpu:
                ok = False
                print(f"MISMATCH p={p} t={t}: ref {w_ref} vs batch {w_gpu}")
    print("self-test:", "OK" if ok else "FAILED")

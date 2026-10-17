#!/usr/bin/env python3
"""
cuda_torso.py -- Spacekangaroos' cuda-torso, re-run as published, made resumable.

The search is leaderboard-references/cuda-torso/run.py line for line: the same
features (5 degree-profile values + 32 Laplacian eigenvectors, squares and
pairwise products, 740 weights), the same population of 1,024, the same
per-threshold elites, Gaussian mutation (sigma 0.3 on about half the genes),
the CoSyNE-style column copies (20 %), the same kernel and penalty, and the
same greedy 20-point selection for its own submission.

Changes, all outside the search:
  1. libeval_fixed.cu: `size_t adj_offset` instead of `int` -- the original
     overflows for N = 2,426 at idx >= 365 and crashes on large-graph at its
     own default batch. The kernel's +1 self-loop degree bug is KEPT: this is
     the method as published.
  2. Any instance: N is read from the .gr file (the original hardcodes the
     three official sizes).
  3. Resumable: elites, elite fitnesses, the next population, the generation
     counter and the numpy / torch RNG states are checkpointed every 10
     minutes (atomic writes) and on SIGTERM; a run continues from its
     checkpoint, and its budget (wall-clock GPU time) is persisted.
  4. Budgets: --until-budget stops after a checkpoint once the run's
     accumulated time reaches that many seconds (the supervisor's rounds).
  5. Bookkeeping for the comparison, at every checkpoint: its own submission
     (greedy selection over its elites, as published) is re-scored exactly,
     and every elite whose fitness changed since the last checkpoint is
     decoded and its ordering's exact staircase banked into the shared
     archive (te.Archive), whose exact best-20 is the column we report --
     the same post-processing every other method gets. Both scores are logged.
  6. Seeds: numpy and torch are seeded explicitly (the original is unseeded).

Backends for the kernel: "nvcc" builds libeval_fixed.so like run.sh, for the
GPU's own architecture (-arch=sm_XY from nvidia-smi; the toolkits of CUDA 13
and later cannot target Volta, and a library built without code for the GPU
never runs); "nvrtc" compiles the same __global__ source at run time with
CuPy, for servers without the CUDA toolkit. Same launch configuration
(<<<128, B/128>>>).

Kernel self-check, at the start of every session (setup, not counted in the
budget): the kernel's fitnesses for three feasible min-degree orderings, in
batch slots 0, B/2 and B-1, must equal an exact CPU emulation of the same
kernel code (emulate_kernel, which reproduces its +1 self-loop). The published
code never checks for CUDA errors, so a kernel that does not run would
otherwise go unnoticed; here the session stops with exit code 4 instead.
`--self-check-only` runs just the check (gpu/setup.sh uses it).

    python3 campaign/gpu/cuda_torso.py --instance data/large-graph.gr \
        --run-dir campaign/runs/large-graph/cuda_torso --until-budget 3600
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import c_size_t, c_bool, c_uint16, c_int, POINTER
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, CAMPAIGN)

import numpy as np  # noqa: E402
from numpy.linalg import eigh  # noqa: E402
from scipy.sparse.csgraph import laplacian  # noqa: E402

import torso_engine as te  # noqa: E402
from common import atomic_write_bytes, atomic_write_json, now_utc  # noqa: E402

CKPT_EVERY_S = 600
METHOD = "cuda_torso"
FORMAT_VERSION = 1


# --------------------------------------------------------------- kernel loading

def gpu_arch():
    """sm_XY of GPU 0 from nvidia-smi, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        return "sm_" + out[0].replace(".", "") if out else None
    except Exception:
        return None


def emulate_kernel(adj, orderings):
    """Exact CPU emulation of libeval_fixed.cu's _evaluate for the given
    orderings (no early stop, so meant for orderings below the cap): the
    kernel's fitnesses, i.e. the suffix maxima of its step degrees. Its fill
    loop starts at j = i and so marks every filled vertex adjacent to itself;
    a later row scan counts that self-loop, which this reproduces."""
    out = np.zeros((len(orderings), adj.shape[0]), dtype=np.int64)
    for b, order in enumerate(orderings):
        A = adj.copy()
        deg = np.empty(len(order), dtype=np.int64)
        for s, v in enumerate(order):
            nz = np.flatnonzero(A[v])
            deg[s] = len(nz)
            A[np.ix_(nz, nz)] = True
            A[nz, v] = False
            A[v, nz] = False
        pen = 0
        for d in deg:
            if d > 500:
                pen = 501 if pen == 0 else pen + int(d) - 500
        if pen:
            deg = deg + pen
        out[b] = np.maximum.accumulate(deg[::-1])[::-1]
    return out


def load_kernel(backend):
    """Return evaluate(adjs, perms, degrees, fitnesses, B, N) on torch tensors."""
    src = os.path.join(HERE, "libeval_fixed.cu")
    if backend == "nvcc":
        so = os.path.join(HERE, "libeval_fixed.so")
        if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
            arch = gpu_arch()
            subprocess.run(["nvcc"] + ([f"-arch={arch}"] if arch else []) +
                           ["-Xcompiler", "-fPIC", "-shared", "-o", so, src], check=True)
        lib = ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        ev = lib.evaluate
        ev.argtypes = [POINTER(c_bool), POINTER(c_uint16), POINTER(c_uint16), POINTER(c_int), c_size_t, c_size_t]

        def evaluate(adjs, perms, degrees, fitnesses, B, N):
            ev(ctypes.cast(adjs.data_ptr(), POINTER(c_bool)),
               ctypes.cast(perms.data_ptr(), POINTER(c_uint16)),
               ctypes.cast(degrees.data_ptr(), POINTER(c_uint16)),
               ctypes.cast(fitnesses.data_ptr(), POINTER(c_int)),
               B, N)
        return evaluate
    if backend == "nvrtc":
        import cupy
        code = open(src).read()
        kern = code[code.index("__global__"):code.index('extern "C"')]
        # NVRTC compiles without system headers: spell the integer types out
        # (same widths as on the x86-64 host: size_t is 64-bit unsigned)
        kern = kern.replace("size_t", "unsigned long long")
        code = ("typedef unsigned short uint16_t;\n"
                "extern \"C\" " + kern)
        fn = cupy.RawKernel(code, "_evaluate")

        def evaluate(adjs, perms, degrees, fitnesses, B, N):
            import torch
            fn((128,), (B // 128,), (cupy.uint64(adjs.data_ptr()), cupy.uint64(perms.data_ptr()),
                                     cupy.uint64(degrees.data_ptr()), cupy.uint64(fitnesses.data_ptr()),
                                     cupy.uint64(B), cupy.uint64(N)))
            torch.cuda.synchronize()
        return evaluate
    raise SystemExit(f"unknown backend {backend}")


# ------------------------------------------------- the published helper functions

def create_submission(elites, nodes, ts, graph):
    """Create a submission based on the elites. (run.py, unchanged)"""
    logits = elites @ nodes
    perms = logits.argsort(axis=1)
    return {
        "challenge": "spoc-3-torso-decompositions",
        "problem": graph,
        "decisionVector": [perm.tolist() + [t.item()] for perm, t in zip(perms[ts].cpu(), ts)],
    }


def calculate_hvi(ts, degrees, N):
    """Pick a subset of ts and calculate the hyper volume indicator (HVI). (run.py, unchanged)"""
    degrees = degrees.cpu().numpy()
    ts = ts.cpu().numpy()
    min_degree = np.full_like(degrees, N)
    min_t = np.full_like(ts, N)
    hvi = 0
    selected = []
    for _ in range(20):
        area = (min_t - ts) * (min_degree - degrees)
        argmax = np.argmax(area)
        selected.append(argmax)
        t = ts[argmax]
        degree = degrees[argmax]
        contribution = (min_degree[argmax] - degree) * (min_t[argmax] - t)

        i = argmax
        while i >= 0 and i < N and min_degree[i] > degree:
            min_degree[i] = degree
            i += 1

        i = argmax
        while i >= 0 and i < N and min_t[i] > t:
            min_t[i] = t
            i -= 1

        hvi -= contribution
    return np.array(selected), hvi


def feature_count(eigenvectors):
    raw_features = (eigenvectors + 5)
    features = raw_features + raw_features + (math.factorial(raw_features) // (math.factorial(2) * math.factorial(raw_features - 2)))
    return features


def init_adj(N, path):
    """Initialise an adjacency matrix based on the edges. (run.py; path instead of data/{graph}.gr)"""
    adj = np.zeros((N, N), dtype=np.bool_)
    with open(path, "r") as f:
        for line in f.readlines():
            src, dst = line.split(" ")
            src = int(src.strip())
            dst = int(dst.strip())
            adj[src, dst] = True
            adj[dst, src] = True
    return adj


def init_node_features(adj, eigenvectors):
    """Initialise node features based on the graph topology. (run.py, unchanged)"""
    N = adj.shape[0]
    degree_profile = 5
    raw_features = degree_profile + eigenvectors
    features = feature_count(eigenvectors)
    nodes = np.zeros((N, features), dtype=np.float32)

    for i in range(N):
        nodes[i, 0] = adj[i].sum()
    for i in range(N):
        nodes[i, 1] = nodes[adj[i], 0].min()
        nodes[i, 2] = nodes[adj[i], 0].max()
        nodes[i, 3] = nodes[adj[i], 0].mean()
        nodes[i, 4] = nodes[adj[i], 0].std()

    lap = laplacian(adj.astype(np.int8), normed=True)
    eig_vals, eig_vecs = eigh(lap)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = eig_vecs[:, 1: eigenvectors + 1]
    nodes[:, degree_profile: degree_profile + eigenvectors] = pe

    from itertools import combinations
    for i in range(raw_features):
        nodes[:, raw_features + i] = nodes[:, i] ** 2
    for ii, (i, j) in enumerate(combinations(list(range(raw_features)), 2)):
        nodes[:, raw_features + raw_features + ii] = nodes[:, i] * nodes[:, j]

    means = np.empty((features, ), dtype=np.float32)
    stds = np.empty((features, ), dtype=np.float32)
    for i in range(features):
        means[i] = nodes[:, i].mean()
        stds[i] = nodes[:, i].std()
    nodes = (nodes - means) / stds

    return nodes.T


# ------------------------------------------------------------------- the runner

class Run:
    def __init__(self, a):
        import torch
        self.torch = torch
        self.a = a
        self.run_dir = a.run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.instance = os.path.splitext(os.path.basename(a.instance))[0]
        self.eng = te.Engine(a.instance)
        self.N = self.eng.n
        self.stop = False
        self.budget_base = 0.0
        self.t0 = time.time()
        self.ckpt = os.path.join(self.run_dir, "ckpt.pkl")
        self.prev = os.path.join(self.run_dir, "ckpt.prev.pkl")
        self.logf = open(os.path.join(self.run_dir, "log.txt"), "a")
        import fcntl
        self._lockf = open(os.path.join(self.run_dir, "lock"), "a+")
        try:
            fcntl.flock(self._lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"{self.run_dir} is already running", file=sys.stderr)
            sys.exit(3)
        self._lockf.seek(0)
        self._lockf.truncate()
        self._lockf.write(str(os.getpid()))
        self._lockf.flush()
        signal.signal(signal.SIGTERM, self._sig)
        signal.signal(signal.SIGINT, self._sig)

    def _sig(self, *args):
        self.stop = True

    def log(self, msg):
        self.logf.write(f"{now_utc()} {msg}\n")
        self.logf.flush()

    def budget(self):
        return self.budget_base + (time.time() - self.t0)

    def main(self):
        torch = self.torch
        a = self.a
        torch.set_grad_enabled(False)
        device = a.device
        if device == "cuda":
            evaluate = load_kernel(a.backend)
        else:
            # tests without a GPU: the exact CPU emulation stands in for the kernel
            def evaluate(adjs_, perms_, degrees_, fitnesses_, B_, N_):
                self._cpu_kernel(adj, perms_, fitnesses_)
        B = a.batch_size
        N = self.N
        E = feature_count(a.eigenvectors)
        run_id = int(time.time())

        adj = init_adj(N, a.instance).astype(np.bool_)
        adjs = np.repeat(adj[None, :, :], B, 0)
        nodes = init_node_features(adj, a.eigenvectors)
        nodes = torch.from_numpy(nodes).to(device)

        state = self._load()
        if state is None:
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            population = np.random.normal(0.0, a.init_stdev, (B, E)).astype(np.float32)
            population = torch.from_numpy(population).to(device)
            elites = torch.empty((N, E), dtype=torch.float32).to(device)
            elite_fitnesses = torch.full((N, ), 999999, dtype=torch.int).to(device)
            generation0 = 0
            self.arch = te.Archive(N)
            self.dirty = torch.zeros(N, dtype=torch.bool, device=device)
            self.log(f"start cuda_torso on {self.instance} N={N} B={B} backend={a.backend} seed={a.seed}")
        else:
            np.random.set_state(state["np_rng"])
            torch.set_rng_state(state["torch_rng"])
            if device == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"])
            population = torch.from_numpy(state["population"]).to(device)
            elites = torch.from_numpy(state["elites"]).to(device)
            elite_fitnesses = torch.from_numpy(state["elite_fitnesses"]).to(device)
            generation0 = state["generation"]
            self.budget_base = state["budget"]
            self.arch = te.Archive.restore(state["archive"])
            self.dirty = torch.zeros(N, dtype=torch.bool, device=device)
            self.log(f"resumed at generation {generation0}, budget {self.budget_base:.0f}s")
        self.t0 = time.time()

        bkup_adjs = torch.from_numpy(adjs).to(device)
        adjs = torch.from_numpy(adjs).to(device)
        perms = torch.empty((B, N), dtype=torch.uint16).to(device)
        degrees = torch.empty((B, N), dtype=torch.uint32).to(device)
        fitnesses = torch.full((B, N), 999999, dtype=torch.int).to(device)
        if not a.no_self_check:
            if not self._self_check(evaluate, adj, adjs, bkup_adjs, perms, degrees, fitnesses, B, N):
                print(f"cuda_torso on {self.instance}: kernel self-check FAILED (see {self.run_dir}/log.txt)",
                      file=sys.stderr, flush=True)
                sys.exit(4)
            print(f"cuda_torso on {self.instance}: kernel self-check passed", flush=True)
            self.t0 = time.time()               # the check is setup, not search time
        if a.self_check_only:
            return
        elite_range = torch.ones((N, ), dtype=torch.float32, device=device) / B
        ts = torch.arange(N).to(device)
        logits = torch.empty((B, N), dtype=torch.float32).to(device)

        last_ckpt = time.time()
        last_hb = 0.0
        generation = generation0
        reason = "stopped"
        while True:
            if a.until_budget is not None and self.budget() >= a.until_budget:
                reason = "slice-done"
                break
            if self.stop:
                break
            start = time.perf_counter()
            logits[:] = population @ nodes
            perms[:] = logits.argsort(axis=1).to(torch.uint16)
            adjs[:, :, :] = bkup_adjs[:, :, :]
            evaluate(adjs, perms, degrees, fitnesses, B, N)

            # Save the best from the population
            best = fitnesses.min(axis=0)
            better = best.values <= elite_fitnesses
            elites[better] = population[best.indices][better]
            elite_fitnesses[better] = best.values[better]
            self.dirty |= better

            # Create next population
            idx = torch.multinomial(elite_range, B, replacement=N < B)
            population[:] = elites[idx]
            mask = torch.rand((B, E), device=device) > a.mutation_proba
            population += torch.normal(0.0, a.mutation_stdev, (B, E), device=device) * mask

            permutations = int(population.shape[0] * population.shape[1] * a.cosyne_proba)
            target_rows = torch.randint(0, population.shape[0], (permutations, ))
            origin_rows = torch.randint(0, population.shape[0], (permutations, ))
            cols = torch.randint(0, population.shape[1], (permutations, ))
            population[target_rows, cols] = population[origin_rows, cols]

            generation += 1
            if generation % a.log_every == 0:
                runtime = time.perf_counter() - start
                _, hvi = calculate_hvi(ts, elite_fitnesses, N)
                with open(os.path.join(self.run_dir, "kernel_log.txt"), "a") as f:
                    f.write("|".join(str(x) for x in [run_id, round(time.time(), 2), generation,
                                                       round(runtime, 4), int(best.values.sum()),
                                                       int(elite_fitnesses.sum()), int(hvi)]) + "\n")
            if time.time() - last_ckpt >= CKPT_EVERY_S:
                self._checkpoint(population, elites, elite_fitnesses, nodes, ts, generation, "periodic")
                last_ckpt = time.time()
            if time.time() - last_hb >= 60:
                with open(os.path.join(self.run_dir, "heartbeat"), "w") as f:
                    f.write(f"{time.time():.0f} {os.getpid()} {self.budget():.0f} generation {generation}\n")
                last_hb = time.time()
        self._checkpoint(population, elites, elite_fitnesses, nodes, ts, generation, reason)
        self.log(f"session end ({reason}) at generation {generation}, budget {self.budget():.0f}s")
        print(f"cuda_torso on {self.instance}: {reason}, budget {self.budget():.0f}s, "
              f"score {self.arch.score():,}", flush=True)

    # CPU emulation of the kernel, for tests on machines without a GPU (slow)
    def _cpu_kernel(self, adj, perms, fitnesses):
        torch = self.torch
        P = perms.cpu().numpy().astype(np.int64)
        fitnesses[:] = torch.from_numpy(emulate_kernel(adj, list(P)).astype(np.int32))

    def _self_check(self, evaluate, adj, adjs, bkup_adjs, perms, degrees, fitnesses, B, N):
        """The kernel against emulate_kernel on feasible min-degree orderings
        in batch slots 0, B/2 and B-1 (the last one exercises the 64-bit
        offsets). True when every fitness matches."""
        torch = self.torch
        rng = te.Rng(20261017)              # its own stream: the search's are untouched
        tests = []
        for slack in range(8):
            p, d, bad = self.eng.greedy(rng, slack % 4)
            if bad < 0 and int(d.max()) <= 499:     # the kernel reads up to +1: stay below its cap
                tests.append(p.astype(np.int64))
            if len(tests) == 3:
                break
        if not tests:
            self.log("kernel self-check skipped: no min-degree ordering below the cap")
            return True
        rows = sorted({0, B // 2, B - 1})
        P = np.empty((B, N), dtype=np.int64)
        P[:] = tests[0]
        want = []
        for j, r in enumerate(rows):
            P[r] = tests[j % len(tests)]
            want.append(tests[j % len(tests)])
        perms[:] = torch.from_numpy(P).to(perms.device).to(torch.uint16)
        adjs[:, :, :] = bkup_adjs[:, :, :]
        fitnesses.fill_(999999)
        evaluate(adjs, perms, degrees, fitnesses, B, N)
        got = fitnesses[rows].cpu().numpy().astype(np.int64)
        exp = emulate_kernel(adj, want)
        for i, r in enumerate(rows):
            if not np.array_equal(got[i], exp[i]):
                k = int(np.flatnonzero(got[i] != exp[i])[0])
                self.log(f"kernel self-check FAILED: batch slot {r}, threshold {k}: kernel {int(got[i][k])}, "
                         f"emulation {int(exp[i][k])} ({int((got[i] != exp[i]).sum())} of {N} thresholds differ)")
                return False
        self.log(f"kernel self-check passed: slots {rows}, {N} thresholds each (endpoint as the kernel reads it: {int(exp[0][0])})")
        return True

    # ------------------------------------------------------------- bookkeeping
    def _load(self):
        for path in (self.ckpt, self.prev):
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        d = pickle.load(f)
                    if d.get("method") == METHOD and d.get("version") == FORMAT_VERSION:
                        return d
                except Exception as e:
                    self.log(f"checkpoint {path} unreadable: {e!r}")
        return None

    def _bank_elites(self, elites, nodes):
        """Decode the elites that changed and bank their exact staircases."""
        torch = self.torch
        idx = torch.nonzero(self.dirty).flatten()
        if len(idx) == 0:
            return 0
        logits = elites[idx] @ nodes
        P = logits.argsort(axis=1).cpu().numpy().astype(np.int32)
        for p in P:
            self.arch.eval_add(self.eng, p)
        self.dirty[:] = False
        return len(P)

    def _checkpoint(self, population, elites, elite_fitnesses, nodes, ts, generation, reason):
        torch = self.torch
        banked = self._bank_elites(elites, nodes)
        # the method's own submission (greedy selection over its internal fitness), re-scored exactly
        selected_ts, hvi = calculate_hvi(ts, elite_fitnesses, self.N)
        sub = create_submission(elites, nodes, torch.from_numpy(selected_ts).to(nodes.device), self.instance)
        pts = []
        for dv in sub["decisionVector"]:
            p = np.asarray(dv[:-1], dtype=np.int32)
            pts.append((self.eng.fitness(p, dv[-1]), dv[-1]))
        own_score = -int(te.hv_points(pts, self.N))
        d = {
            "version": FORMAT_VERSION, "method": METHOD, "instance": self.instance,
            "generation": generation, "budget": self.budget(),
            "population": population.cpu().numpy(), "elites": elites.cpu().numpy(),
            "elite_fitnesses": elite_fitnesses.cpu().numpy(),
            "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if nodes.is_cuda else None,
            "archive": self.arch.export(), "saved_utc": now_utc(), "reason": reason,
        }
        if os.path.exists(self.ckpt):
            os.replace(self.ckpt, self.prev)
        atomic_write_bytes(self.ckpt, pickle.dumps(d, protocol=4))
        score = self.arch.score()
        best = {"challenge": "spoc-3-torso-decompositions", "problem": self.instance, "method": METHOD,
                "score": score, "own_selection_score": own_score, "budget_s": round(self.budget(), 1),
                "generation": generation, "saved_utc": now_utc(),
                "decisionVector": self.arch.decision_vectors()}
        atomic_write_json(os.path.join(self.run_dir, "best.json"), best)
        atomic_write_json(os.path.join(self.run_dir, "own_submission.json"), sub)
        new = not os.path.exists(os.path.join(self.run_dir, "anytime.csv"))
        with open(os.path.join(self.run_dir, "anytime.csv"), "a") as f:
            if new:
                f.write("budget_s,wall_utc,score,evaluations,points,kind\n")
            f.write(f"{self.budget():.1f},{now_utc()},{score},{generation},{own_score},ckpt\n")
        ws, tss = self.arch.selected()
        atomic_write_json(os.path.join(self.run_dir, "status.json"), {
            "method": METHOD, "instance": self.instance, "n": self.N,
            "state": "running" if reason == "periodic" else reason, "pid": os.getpid(),
            "budget_s": round(self.budget(), 1), "until_budget_s": self.a.until_budget,
            "score": score, "own_selection_score": own_score, "generation": generation,
            "evaluations": generation * self.a.batch_size, "banked_elites": banked,
            "front": [[int(w), int(t)] for w, t in zip(ws, tss)], "updated_utc": now_utc(),
            "extra": {"generation": generation, "own_selection_score": own_score}})
        with open(os.path.join(self.run_dir, "heartbeat"), "w") as f:
            f.write(f"{time.time():.0f} {os.getpid()} {self.budget():.0f} {score}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--until-budget", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default=os.environ.get("CUDA_TORSO_BACKEND", "nvcc"), choices=["nvcc", "nvrtc"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--self-check-only", action="store_true",
                    help="check the kernel against the CPU emulation and exit (no search, no checkpoint)")
    ap.add_argument("--no-self-check", action="store_true", help="skip the kernel self-check (tests)")
    # the published defaults (run.py)
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--init_stdev", type=float, default=0.3)
    ap.add_argument("--mutation_stdev", type=float, default=0.3)
    ap.add_argument("--mutation_proba", type=float, default=0.5)
    ap.add_argument("--cosyne_proba", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=1)
    a = ap.parse_args()
    Run(a).main()


if __name__ == "__main__":
    main()

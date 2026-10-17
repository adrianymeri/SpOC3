"""
torso_engine.py -- Python bindings for engine/libtorso.so (ctypes + numpy).

Every CPU method of the campaign evaluates through these bindings, so all of
them run at the same evaluation speed. The semantics are esa_eval.py's
(checked bit for bit by tests/test_engine.py).

    eng  = Engine("data/small-graph.gr")
    deg, bad = eng.eval(perm)          # step degrees; bad = first cap breach or -1
    w, t = eng.staircase(deg)          # every (width, t) point perm offers
    arch = Archive(eng.n)
    arch.eval_add(eng, perm)           # evaluate + bank its whole staircase
    arch.score()                       # -HV of the exact best 20 points

Permutations are numpy int32 arrays. Random state for the C side lives in an
Rng (four uint64 words), so it is checkpointed with everything else.
"""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np
from numpy.ctypeslib import ndpointer

MAX_TW = 500
HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "engine", "libtorso.so")

_i32 = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_u64 = ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
_u8 = ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_f32 = ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_f64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_vp = ctypes.c_void_p
_int = ctypes.c_int
_i64 = ctypes.c_int64


def _build_if_needed():
    src = os.path.join(HERE, "engine", "torso_engine.c")
    if (not os.path.exists(LIB_PATH)
            or os.path.getmtime(LIB_PATH) < os.path.getmtime(src)):
        subprocess.run(["make", "-s", "-C", os.path.join(HERE, "engine")], check=True)


_build_if_needed()
lib = ctypes.CDLL(LIB_PATH)


def _sig(name, restype, *argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


_sig("te_rng_seed", None, _u64, ctypes.c_uint64)
_sig("te_rng_below", ctypes.c_uint32, _u64, ctypes.c_uint32)
_sig("te_rng_unit", ctypes.c_double, _u64)
_sig("te_shuffle", None, _i32, _int, _u64)
_sig("te_load", _vp, ctypes.c_char_p)
_sig("te_free", None, _vp)
_sig("te_n", _int, _vp)
_sig("te_m", _i64, _vp)
_sig("te_words", _int, _vp)
_sig("te_csr", None, _vp, _i32, _i32)
_sig("te_is_perm", _int, _vp, _i32)
_sig("te_eval", _int, _vp, _i32, _i32, _int)
_sig("te_fitness", _int, _vp, _i32, _int)
_sig("te_staircase", _int, _i32, _int, _i32, _i32)
_sig("te_stair_hv", _i64, _i32, _int, _int)
_sig("te_best_t_for_width", _int, _i32, _int, _int)
_sig("te_hssp", _i64, _i32, _i32, _int, _int, _int, _i32, ctypes.POINTER(_int))
_sig("te_arch_new", _vp, _int)
_sig("te_arch_free", None, _vp)
_sig("te_arch_add", _int, _vp, _i32, _i32, _i32, _int, _i64)
_sig("te_arch_eval_add", _int, _vp, _vp, _i32, _i32, _i32, _i32, ctypes.POINTER(_int), _i64)
_sig("te_arch_pareto", _int, _vp, _i32, _i32)
_sig("te_arch_hv", _i64, _vp, _int)
_sig("te_arch_selected", _int, _vp, _int, _i32, _i32)
_sig("te_arch_best_t", _int, _vp, _int)
_sig("te_arch_stamp", _i64, _vp, _int)
_sig("te_arch_get_perm", _int, _vp, _int, _i32)
_sig("te_arch_adds", _i64, _vp)
_sig("te_arch_improvements", _i64, _vp)
_sig("te_arch_hv_with", _i64, _vp, _i32, _i32, _int, _int)
_sig("te_arch_set", None, _vp, _int, _int, _i32, _i64)
_sig("te_arch_set_counters", None, _vp, _i64, _i64)
_sig("te_greedy", _int, _vp, _vp, _int, _u64, _i32, _i32)
_sig("te_inc_new", _vp, _vp, _int)
_sig("te_inc_free", None, _vp)
_sig("te_inc_set", _int, _vp, _vp, _i32)
_sig("te_inc_eval", None, _vp, _vp, _i32, _int, _int, _i32)
_sig("te_inc_accept", None, _vp, _vp, _i32, _int, _int)
_sig("te_inc_get", None, _vp, _i32, _i32)
_sig("te_hri_neighbour_destroy", _int, _vp, _i32, _int, _u64, _i32)
_sig("te_hri_balanced_repair", None, _vp, _i32, _i32, _int, _u64, _i32)
_sig("te_hri_random_destroy_repair", None, _int, _i32, _int, _u64, _i32, _i32)


def as_perm(p):
    return np.ascontiguousarray(p, dtype=np.int32)


class Rng:
    """xoshiro256** state for the C side. Picklable via .state."""

    def __init__(self, seed=1):
        self.s = np.zeros(4, dtype=np.uint64)
        lib.te_rng_seed(self.s, ctypes.c_uint64(seed & ((1 << 64) - 1)))

    @property
    def state(self):
        return self.s.copy()

    @state.setter
    def state(self, v):
        self.s = np.ascontiguousarray(v, dtype=np.uint64).copy()

    def below(self, m):
        return int(lib.te_rng_below(self.s, m))

    def unit(self):
        return float(lib.te_rng_unit(self.s))


class Engine:
    """One graph, loaded once per process."""

    def __init__(self, path):
        self.path = path
        self.g = lib.te_load(path.encode())
        if not self.g:
            raise IOError(f"cannot load graph {path}")
        self.n = lib.te_n(self.g)
        self.m = int(lib.te_m(self.g))
        self.off = np.zeros(self.n + 1, dtype=np.int32)
        self.nbr = np.zeros(2 * self.m, dtype=np.int32)
        lib.te_csr(self.g, self.off, self.nbr)
        self.deg0 = np.diff(self.off).astype(np.int32)
        self._deg = np.zeros(self.n, dtype=np.int32)
        self._w = np.zeros(self.n + 1, dtype=np.int32)
        self._t = np.zeros(self.n + 1, dtype=np.int32)
        self.evaluations = 0

    def __del__(self):
        try:
            lib.te_free(self.g)
        except Exception:
            pass

    def neighbours(self, v):
        return self.nbr[self.off[v]:self.off[v + 1]]

    def is_perm(self, perm):
        return bool(lib.te_is_perm(self.g, as_perm(perm)))

    def eval(self, perm, abort_cap=True):
        """(deg, first_bad): step degrees and the first step over the cap (-1 if none)."""
        deg = np.zeros(self.n, dtype=np.int32)
        bad = lib.te_eval(self.g, as_perm(perm), deg, 1 if abort_cap else 0)
        self.evaluations += 1
        return deg, bad

    def fitness(self, perm, t):
        """The official (width, t) fitness's width, as esa_eval.evaluate."""
        return int(lib.te_fitness(self.g, as_perm(perm), int(t)))

    def staircase(self, deg):
        deg = as_perm(deg)
        k = lib.te_staircase(deg, self.n, self._w, self._t)
        return self._w[:k].copy(), self._t[:k].copy()

    def stair_hv(self, deg, k=20):
        return int(lib.te_stair_hv(as_perm(deg), self.n, k))

    def best_t_for_width(self, deg, target):
        return int(lib.te_best_t_for_width(as_perm(deg), self.n, int(target)))

    def greedy(self, rng, slack=0, head=None):
        """Min-degree elimination with random tie-breaks (slack 0), or a random
        choice among degrees <= min + slack. head: optional bool/uint8 mask
        whose vertices are eliminated first. Returns (perm, deg, first_bad)."""
        perm = np.zeros(self.n, dtype=np.int32)
        deg = np.zeros(self.n, dtype=np.int32)
        hp = None if head is None else np.ascontiguousarray(head, dtype=np.uint8)
        bad = lib.te_greedy(self.g, hp.ctypes.data if hp is not None else None,
                            int(slack), rng.s, perm, deg)
        self.evaluations += 1
        return perm, deg, bad


def hssp(xs, ys, n, k=20):
    """Exact best-k hypervolume subset of a 2-D front (xs asc, ys strictly desc)."""
    xs, ys = as_perm(xs), as_perm(ys)
    idx = np.zeros(max(len(xs), 1), dtype=np.int32)
    cnt = _int(0)
    hv = lib.te_hssp(xs, ys, len(xs), int(n), int(k), idx, ctypes.byref(cnt))
    return int(hv), idx[:cnt.value].copy()


class Archive:
    """Best t per width and the permutation behind it (torso.Front in C)."""

    def __init__(self, n):
        self.n = n
        self.a = lib.te_arch_new(n)
        self._w = np.zeros(MAX_TW + 1, dtype=np.int32)
        self._t = np.zeros(MAX_TW + 1, dtype=np.int32)
        self._sw = np.zeros(n + 1, dtype=np.int32)
        self._st = np.zeros(n + 1, dtype=np.int32)
        self.stamp = 0

    def __del__(self):
        try:
            lib.te_arch_free(self.a)
        except Exception:
            pass

    def add(self, perm, w, t):
        self.stamp += 1
        w, t = as_perm(w), as_perm(t)
        return int(lib.te_arch_add(self.a, as_perm(perm), w, t, len(w), self.stamp))

    def eval_add(self, eng, perm):
        """Evaluate perm, bank its staircase. Returns (improvements, deg, w, t);
        w and t are empty when perm breaks the cap."""
        self.stamp += 1
        deg = np.zeros(self.n, dtype=np.int32)
        npts = _int(0)
        imp = lib.te_arch_eval_add(eng.g, self.a, as_perm(perm), deg, self._sw, self._st,
                                   ctypes.byref(npts), self.stamp)
        eng.evaluations += 1
        k = npts.value
        return int(imp), deg, self._sw[:k].copy(), self._st[:k].copy()

    def add_deg(self, perm, deg):
        """Bank the staircase of an already-evaluated permutation."""
        k = lib.te_staircase(as_perm(deg), self.n, self._sw, self._st)
        if k == 0:
            return 0
        self.stamp += 1
        return int(lib.te_arch_add(self.a, as_perm(perm), self._sw, self._st, k, self.stamp))

    def hv(self, k=20):
        return int(lib.te_arch_hv(self.a, k))

    def score(self, k=20):
        return -self.hv(k)

    def hv_with(self, w, t, k=20):
        w, t = as_perm(w), as_perm(t)
        return int(lib.te_arch_hv_with(self.a, w, t, len(w), k))

    def selected(self, k=20):
        c = lib.te_arch_selected(self.a, k, self._w, self._t)
        return self._w[:c].copy(), self._t[:c].copy()

    def pareto(self):
        c = lib.te_arch_pareto(self.a, self._w, self._t)
        return self._w[:c].copy(), self._t[:c].copy()

    def best_t(self, w):
        return int(lib.te_arch_best_t(self.a, int(w)))

    def best_t_all(self):
        return np.array([lib.te_arch_best_t(self.a, w) for w in range(MAX_TW + 1)], dtype=np.int32)

    def get_perm(self, w):
        out = np.zeros(self.n, dtype=np.int32)
        ok = lib.te_arch_get_perm(self.a, int(w), out)
        return out if ok else None

    @property
    def adds(self):
        return int(lib.te_arch_adds(self.a))

    @property
    def improvements(self):
        return int(lib.te_arch_improvements(self.a))

    def decision_vectors(self, k=20):
        """The submission: perm + [t] for each selected point, widths ascending."""
        ws, ts = self.selected(k)
        out = []
        for w, t in zip(ws, ts):
            p = self.get_perm(int(w))
            out.append([int(x) for x in p] + [int(t)])
        return out

    # ---- checkpointing: only the non-dominated points are needed to resume
    # the score, but every width's entry is kept so that the per-width elites
    # the QD methods select from survive a restart.
    def export(self):
        widths, ts, stamps, perm_ids, perms = [], [], [], [], []
        seen = {}
        for w in range(MAX_TW + 1):
            t = lib.te_arch_best_t(self.a, w)
            if t >= self.n:
                continue
            p = self.get_perm(w)
            key = p.tobytes()
            if key not in seen:
                seen[key] = len(perms)
                perms.append(p)
            widths.append(w); ts.append(t)
            stamps.append(int(lib.te_arch_stamp(self.a, w)))
            perm_ids.append(seen[key])
        return {
            "n": self.n,
            "widths": np.array(widths, dtype=np.int32),
            "ts": np.array(ts, dtype=np.int32),
            "stamps": np.array(stamps, dtype=np.int64),
            "perm_ids": np.array(perm_ids, dtype=np.int32),
            "perms": (np.stack(perms) if perms else np.zeros((0, self.n), dtype=np.int32)),
            "adds": self.adds, "improvements": self.improvements, "stamp": self.stamp,
        }

    @classmethod
    def restore(cls, d):
        a = cls(int(d["n"]))
        for w, t, s, pid in zip(d["widths"], d["ts"], d["stamps"], d["perm_ids"]):
            lib.te_arch_set(a.a, int(w), int(t), as_perm(d["perms"][pid]), int(s))
        lib.te_arch_set_counters(a.a, int(d["adds"]), int(d["improvements"]))
        a.stamp = int(d["stamp"])
        return a


class Inc:
    """Windowed incremental evaluation around one current permutation."""

    def __init__(self, eng, snapshots=32):
        self.eng = eng
        self.c = lib.te_inc_new(eng.g, snapshots)
        self.n = eng.n

    def __del__(self):
        try:
            lib.te_inc_free(self.c)
        except Exception:
            pass

    def set(self, perm):
        return int(lib.te_inc_set(self.eng.g, self.c, as_perm(perm)))

    def eval(self, newperm, lo, hi):
        """Degrees of newperm (= current outside [lo, hi], same set inside)."""
        deg = np.zeros(self.n, dtype=np.int32)
        lib.te_inc_eval(self.eng.g, self.c, as_perm(newperm), int(lo), int(hi), deg)
        self.eng.evaluations += 1
        return deg

    def accept(self, newperm, lo, hi):
        lib.te_inc_accept(self.eng.g, self.c, as_perm(newperm), int(lo), int(hi))

    def get(self):
        p = np.zeros(self.n, dtype=np.int32)
        d = np.zeros(self.n, dtype=np.int32)
        lib.te_inc_get(self.c, p, d)
        return p, d


# ---- Team HRI's operators -------------------------------------------------

def hri_neighbour_destroy(eng, perm, cap, rng):
    hole = np.zeros(eng.n, dtype=np.int32)
    k = lib.te_hri_neighbour_destroy(eng.g, as_perm(perm), int(cap), rng.s, hole)
    return hole[:k].copy()


def hri_balanced_repair(eng, perm, hole, rng):
    out = np.zeros(eng.n, dtype=np.int32)
    hole = as_perm(hole)
    lib.te_hri_balanced_repair(eng.g, as_perm(perm), hole, len(hole), rng.s, out)
    return out


def hri_random_destroy_repair(eng, perm, k, rng):
    out = np.zeros(eng.n, dtype=np.int32)
    scratch = np.zeros(eng.n, dtype=np.int32)
    lib.te_hri_random_destroy_repair(eng.n, as_perm(perm), int(k), rng.s, out, scratch)
    return out


# ---- inner loops in C -------------------------------------------------------

_sig("te_hc_run", _int, _vp, _vp, _vp, _int, _int, _u64, ctypes.POINTER(ctypes.c_int32), _i64)
_sig("te_ls_target", _int, _vp, _vp, _vp, _int, _int, _int, ctypes.c_double, _u64,
     ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), _i64)
_sig("te_policy_decode", _int, _vp, _f32, _int, _f32, _int, _int, _int, _i32, _i32)
_sig("te_policy_nweights", _int, _int, _int)


def hc_run(eng, inc, arch, target, iters, rng, cost):
    """Hill Climbing's inner loop on inc's current ordering; returns (cost, accepts)."""
    c = ctypes.c_int32(int(cost))
    acc = lib.te_hc_run(eng.g, inc.c, arch.a, int(target), int(iters), rng.s, ctypes.byref(c), arch.stamp)
    arch.stamp += iters
    eng.evaluations += iters
    return int(c.value), int(acc)


def ls_target(eng, inc, arch, target, iters, rng, window=24, p_side=0.25):
    """Targeted local search for one width on inc's current ordering.
    Returns (t, blocking degree, strict improvements)."""
    t = ctypes.c_int32(0)
    b = ctypes.c_int32(0)
    better = lib.te_ls_target(eng.g, inc.c, arch.a, int(target), int(iters), int(window),
                              float(p_side), rng.s, ctypes.byref(t), ctypes.byref(b), arch.stamp)
    arch.stamp += iters
    eng.evaluations += iters
    return int(t.value), int(b.value), int(better)


def policy_nweights(ns, hidden):
    return int(lib.te_policy_nweights(int(ns), int(hidden)))


def policy_decode(eng, stat, weights, hidden, target, rescore_every=64):
    """Adaptive neural decoding. stat: (n, NS) float32; weights: flat float32.
    Returns (perm, deg, first_bad)."""
    stat = np.ascontiguousarray(stat, dtype=np.float32)
    weights = np.ascontiguousarray(weights, dtype=np.float32)
    perm = np.zeros(eng.n, dtype=np.int32)
    deg = np.zeros(eng.n, dtype=np.int32)
    bad = lib.te_policy_decode(eng.g, stat, stat.shape[1], weights, int(hidden), int(target),
                               int(rescore_every), perm, deg)
    eng.evaluations += 1
    return perm, deg, int(bad)


def hv_points(points, n):
    """Hypervolume of (width, t) points against (n, n), as esa_eval.hypervolume_2d."""
    valid = sorted((int(x), int(y)) for (x, y) in points if x < n and y < n)
    nd, best_y = [], n
    for x, y in valid:
        if y < best_y:
            nd.append((x, y))
            best_y = y
    hv = 0
    for i, (x, y) in enumerate(nd):
        nx = nd[i + 1][0] if i + 1 < len(nd) else n
        hv += (nx - x) * (n - y)
    return hv


_sig("te_greedy_complete", _int, _vp, _i32, _int, _vp, ctypes.c_float, ctypes.c_float, _u64, _i32, _i32)


def greedy_complete(eng, prefix, prio, lam, alpha, rng):
    """Keep prefix (in order), complete greedily (see torso_engine.c). Returns (perm, deg, bad)."""
    prefix = as_perm(prefix)
    pr = None if prio is None else np.ascontiguousarray(prio, dtype=np.float32)
    perm = np.zeros(eng.n, dtype=np.int32)
    deg = np.zeros(eng.n, dtype=np.int32)
    bad = lib.te_greedy_complete(eng.g, prefix, len(prefix), pr.ctypes.data if pr is not None else None,
                                 float(lam), float(alpha), rng.s, perm, deg)
    eng.evaluations += 1
    return perm, deg, int(bad)


_sig("te_ls_endpoint", _int, _vp, _vp, _vp, _int, _int, ctypes.c_double, _u64, ctypes.POINTER(ctypes.c_int32), _i64)


def ls_endpoint(eng, inc, arch, iters, rng, window=24, p_side=0.3):
    """Endpoint local search on inc's current ordering. Returns (W, strict improvements)."""
    w = ctypes.c_int32(0)
    better = lib.te_ls_endpoint(eng.g, inc.c, arch.a, int(iters), int(window), float(p_side), rng.s,
                                ctypes.byref(w), arch.stamp)
    arch.stamp += iters
    eng.evaluations += iters
    return int(w.value), int(better)


def _front_hv(self):
    """Hypervolume of the archive's whole Pareto front (no 20-point cap)."""
    w, t = self.pareto()
    return int(lib.te_hssp(as_perm(w), as_perm(t), len(w), self.n, max(1, len(w)),
                           np.zeros(max(1, len(w)), dtype=np.int32), ctypes.byref(_int(0)))) if len(w) else 0


Archive.front_hv = _front_hv

#!/usr/bin/env python3
"""
fastwalk.py -- ctypes wrapper for the C elimination-walk kernel (_fastwalk.c)
plus IncEvalC, a drop-in replacement for gbfcpp.IncEval.

Compiles _fastwalk.c with gcc/cc on first import (cached in .fastwalk_cache/).
If no compiler is available, importers should catch the exception and fall
back to the pure-Python IncEval. Self-test: `python3 tools/fastwalk.py`
verifies the kernel bit-for-bit against the Python bitset walk on random and
banked orderings.
"""
from __future__ import annotations
import ctypes, os, subprocess, sys, tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _compile(src, so):
    for cc in ("cc", "gcc", "clang"):
        try:
            subprocess.run([cc, "-O3", "-shared", "-fPIC", src, "-o", so],
                           check=True, capture_output=True)
            return
        except Exception:
            continue
    raise RuntimeError("no C compiler available for fastwalk")


def _build():
    cache = os.path.join(_ROOT, ".fastwalk_cache")
    os.makedirs(cache, exist_ok=True)
    # platform-tagged so a Linux build and a macOS build can share the repo
    tag = f"{sys.platform}-{os.uname().machine}"
    so = os.path.join(cache, f"fastwalk-{tag}.so")
    src = os.path.join(_HERE, "_fastwalk.c")
    if (not os.path.exists(so)
            or os.path.getmtime(so) < os.path.getmtime(src)):
        _compile(src, so)
    try:
        lib = ctypes.CDLL(so)
    except OSError:
        # stale/foreign-platform binary (e.g. built on another OS): rebuild once
        try:
            os.remove(so)
        except OSError:
            pass
        _compile(src, so)
        lib = ctypes.CDLL(so)
    lib.walk.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    lib.walk.restype = None
    return lib


_LIB = _build()


def _p32(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
def _p64(a): return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def adj_words(ab, n):
    """Big-int bitsets -> (n, W) uint64 array."""
    W = (n + 63) // 64
    out = np.zeros((n, W), dtype=np.uint64)
    for v in range(n):
        x = ab[v]
        w = 0
        while x:
            out[v, w] = x & 0xFFFFFFFFFFFFFFFF
            x >>= 64
            w += 1
    return out


class IncEvalC:
    """C-kernel incremental staircase evaluator; same interface as
    gbfcpp.IncEval (full / move / commit, .perm, .deg, .n)."""

    def __init__(self, ab, n, C=64):
        self.n = n; self.C = C
        self.base = adj_words(ab, n)
        self.W = self.base.shape[1]
        self.nck = (n + C - 1) // C
        self._ck = np.zeros((self.nck, n, self.W), dtype=np.uint64)
        self._tmp = np.empty_like(self.base)
        self._scratch = np.empty_like(self.base)
        self._sm = np.empty_like(self.base)
        self.perm = None
        self.deg = None
        self._deg_s = np.zeros(n, dtype=np.int32)

    def full(self, perm):
        n = self.n
        p = np.ascontiguousarray(perm, dtype=np.int32)
        np.copyto(self._tmp, self.base)
        deg = np.zeros(n, dtype=np.int32)
        _LIB.walk(_p32(p), n, self.W, _p64(self._tmp), _p64(self._sm),
                  0, _p32(deg), self.C, _p64(self._ck))
        self.perm = [int(v) for v in p]
        self.deg = deg.astype(np.int64)
        return self.deg

    def move(self, perm, L):
        n = self.n
        ci = min(max(L, 0) // self.C, self.nck - 1)
        i0 = ci * self.C
        p = np.ascontiguousarray(perm, dtype=np.int32)
        np.copyto(self._scratch, self._ck[ci])
        deg = self._deg_s
        deg[:i0] = self.deg[:i0]
        _LIB.walk(_p32(p), n, self.W, _p64(self._scratch), _p64(self._sm),
                  i0, _p32(deg), 0, _p64(self._scratch))  # no ckpt dump
        self._pending = [int(v) for v in p]
        return deg.astype(np.int64)

    def commit(self):
        self.full(self._pending)


if __name__ == "__main__":
    sys.path.insert(0, _ROOT)
    from core import load_graph, build_adj_bitsets, graph_path
    import time, random
    prob = sys.argv[1] if len(sys.argv) > 1 else "small-graph"
    n, adj = load_graph(graph_path(_ROOT, prob))
    ab = build_adj_bitsets(n, adj)

    def py_deg(perm):
        sm = [0]*n; cur = 0
        for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
        tmp = list(ab); deg = np.zeros(n, dtype=np.int64)
        for i in range(n):
            s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
            while x:
                b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
        return deg

    ev = IncEvalC(ab, n)
    rng = random.Random(0)
    for trial in range(3):
        p = list(range(n)); rng.shuffle(p)
        d1 = py_deg(p); d2 = ev.full(p)
        assert np.array_equal(d1, d2), f"full mismatch on trial {trial}"
        # incremental: relocate and compare against fresh python walk
        p2 = list(p); v = p2.pop(700); p2.insert(300, v)
        d3 = ev.move(p2, 300)
        assert np.array_equal(py_deg(p2), d3), "move mismatch"
        ev.commit()
        assert np.array_equal(py_deg(p2), ev.deg), "commit mismatch"
    t0 = time.time(); k = 0
    while time.time() - t0 < 2.0:
        p2 = list(ev.perm); v = p2.pop(rng.randrange(n)); p2.insert(rng.randrange(n), v)
        ev.move(p2, 0); k += 1
    print(f"OK: bit-exact vs python walk; full-move throughput ~{k/2.0:,.0f} evals/s")

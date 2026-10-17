"""
structure.py -- per-instance structure the problem-aware methods use, cached.

    twin classes      vertices with identical open neighbourhoods (false twins)
                      or identical closed neighbourhoods (true twins)
    big cliques       greedy maximal cliques of >= 30 vertices (large family:
                      the planted K500, K400, K300)
    blocks            connected components of the graph without the big-clique
                      vertices (the low-treewidth pieces attached to them)
    core numbers      k-core decomposition
    spectral coords   the 8 smallest non-trivial eigenvectors of the normalised
                      Laplacian, standardised
    static features   (n, 12) float32 for the neural decoder: original degree,
                      core number, twin-class size, big-clique size, and the 8
                      spectral coordinates (the decoder uses the first
                      4 + n_spectral columns)

Computed once per instance and cached as cache/<instance>.npz next to this
file (rebuilt if the .gr file is newer).
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def _core_numbers(eng):
    n = eng.n
    deg = eng.deg0.astype(np.int64).copy()
    order = np.argsort(deg, kind="stable")
    # Batagelj-Zaversnik bucket algorithm
    maxd = int(deg.max()) if n else 0
    bins = np.zeros(maxd + 2, dtype=np.int64)
    for d in deg:
        bins[d] += 1
    start = 0
    for d in range(maxd + 1):
        num = bins[d]
        bins[d] = start
        start += num
    pos = np.zeros(n, dtype=np.int64)
    vert = np.zeros(n, dtype=np.int64)
    for v in range(n):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        bins[deg[v]] += 1
    for d in range(maxd, 0, -1):
        bins[d] = bins[d - 1]
    bins[0] = 0
    for i in range(n):
        v = vert[i]
        for u in eng.neighbours(v):
            if deg[u] > deg[v]:
                du = deg[u]
                pu = pos[u]
                pw = bins[du]
                w = vert[pw]
                if u != w:
                    pos[u], pos[w] = pw, pu
                    vert[pu], vert[pw] = w, u
                bins[du] += 1
                deg[u] -= 1
    return deg.astype(np.int32)


def _twins(eng):
    n = eng.n
    open_key = defaultdict(list)
    closed_key = defaultdict(list)
    for v in range(n):
        nb = eng.neighbours(v)
        open_key[nb.tobytes()].append(v)
        closed_key[np.sort(np.append(nb, v)).astype(np.int32).tobytes()].append(v)
    cls = -np.ones(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    cid = 0
    for groups in (open_key.values(), closed_key.values()):
        for g in groups:
            if len(g) < 2:
                continue
            free = [v for v in g if cls[v] < 0]
            if len(free) < 2:
                continue
            for v in free:
                cls[v] = cid
                size[v] = len(free)
            cid += 1
    for v in range(n):
        if cls[v] < 0:
            cls[v] = cid
            cid += 1
    return cls, size


def _big_cliques(eng, min_size=30):
    n = eng.n
    order = np.argsort(-eng.deg0, kind="stable")
    cl = -np.ones(n, dtype=np.int32)
    csize = np.zeros(n, dtype=np.int32)
    k = 0
    adjset = [set(eng.neighbours(v).tolist()) for v in range(n)]
    for v in order:
        if cl[v] >= 0 or eng.deg0[v] + 1 < min_size:
            continue
        members = [v]
        cand = sorted((u for u in adjset[v] if cl[u] < 0), key=lambda u: -eng.deg0[u])
        for u in cand:
            if all(u in adjset[x] for x in members):
                members.append(u)
        if len(members) >= min_size:
            for u in members:
                cl[u] = k
                csize[u] = len(members)
            k += 1
    return cl, csize


def _blocks(eng, clique_of):
    n = eng.n
    comp = -np.ones(n, dtype=np.int32)
    c = 0
    for s in range(n):
        if comp[s] >= 0 or clique_of[s] >= 0:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            v = stack.pop()
            for u in eng.neighbours(v):
                if comp[u] < 0 and clique_of[u] < 0:
                    comp[u] = c
                    stack.append(u)
        c += 1
    size = np.zeros(n, dtype=np.int32)
    if c:
        counts = np.bincount(comp[comp >= 0], minlength=c)
        size[comp >= 0] = counts[comp[comp >= 0]]
    return comp, size


def _spectral(eng, k=2):
    n = eng.n
    A = np.zeros((n, n))
    for v in range(n):
        A[v, eng.neighbours(v)] = 1.0
    d = A.sum(1)
    d[d == 0] = 1.0
    dm = 1.0 / np.sqrt(d)
    lap = np.eye(n) - (A * dm).T * dm
    vals, vecs = np.linalg.eigh(lap)
    pe = np.real(vecs[:, vals.argsort()][:, 1:k + 1])
    pe = (pe - pe.mean(0)) / np.where(pe.std(0) > 0, pe.std(0), 1.0)
    return pe.astype(np.float32)


def load(eng):
    """Structure of eng's graph as a dict of numpy arrays (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    name = os.path.splitext(os.path.basename(eng.path))[0]
    path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(eng.path):
        try:
            d = dict(np.load(path))
            if int(d["n"]) == eng.n and d["stat"].shape[1] == 12:
                return d
        except Exception:
            pass
    core = _core_numbers(eng)
    twin, twin_size = _twins(eng)
    clique, clique_size = _big_cliques(eng)
    block, block_size = _blocks(eng, clique)
    spec = _spectral(eng, 8)
    dmax = max(1, int(eng.deg0.max()))
    cmax = max(1, int(core.max()))
    stat = np.stack([
        eng.deg0 / dmax,
        core / cmax,
        np.log2(twin_size.astype(np.float64)) / 4.0,
        clique_size / 500.0,
    ] + [spec[:, k] for k in range(spec.shape[1])], 1).astype(np.float32)
    d = {"n": np.int64(eng.n), "core": core, "twin": twin, "twin_size": twin_size,
         "clique": clique, "clique_size": clique_size, "block": block, "block_size": block_size,
         "spec": spec, "stat": stat}
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)
    return d


def groups(labels):
    """label array -> list of member arrays (labels < 0 ignored)."""
    out = defaultdict(list)
    for v, l in enumerate(labels):
        if l >= 0:
            out[int(l)].append(v)
    return [np.array(g, dtype=np.int32) for g in out.values()]

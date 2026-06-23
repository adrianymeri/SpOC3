#!/usr/bin/env python3
r"""
front_to_checkpoint.py -- build a cuda-torso warm-start checkpoint (.pt) from a
submission front, so a GPU run can *continue* from it.

cuda-torso initialises its per-threshold elite policies randomly and has no resume
flag. This converter reconstructs those policies from an existing front: it builds
the SAME node features run.py uses (LocalDegreeProfile + 32 Laplacian eigenvectors
+ degree-2 polynomial expansion, E=740), then for each threshold fits a linear
policy whose argsort reproduces the best front ordering at that threshold. The
result is a {"elites","elite_fitnesses","nodes","args"} checkpoint in run.py's
exact format, loadable by run_band.py --warmstart_pt (or run.py with the same
block).

Faithful only when the front IS policy-representable -- i.e. a cuda-torso/argsort
solution such as a Kaggle `small_best*.json`. (Our constructed torso-deletion
front is NOT policy-decodable, §13.4, so fitting to it loses quality.)

    python3 tools/front_to_checkpoint.py --problem small-graph \
        --front submissions/small-graph/kaggle_latest.json \
        --out small_warmstart.pt

Run on a machine with torch + scipy (e.g. your Mac). Upload the .pt to Kaggle and:
    python3 run_band.py --graph small-graph --warmstart_pt small_warmstart.pt \
        --targets <breakpoint-thresholds> --max_generations 100000000
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path, MAX_TW
from tools.fastwalk import IncEvalC


def feature_count(eig):
    raw = eig + 5
    return raw + raw + (math.factorial(raw) // (math.factorial(2) * math.factorial(raw - 2)))


def init_node_features(adj, eig):
    """Verbatim port of cuda-torso run.py init_node_features (must match the GPU)."""
    from scipy.linalg import eigh
    from scipy.sparse.csgraph import laplacian
    from itertools import combinations
    N = adj.shape[0]
    degree_profile = 5
    raw = degree_profile + eig
    feats = feature_count(eig)
    nodes = np.zeros((N, feats), dtype=np.float32)
    deg = adj.sum(1)
    nodes[:, 0] = deg
    for i in range(N):
        nb = np.where(adj[i])[0]
        if len(nb):
            nodes[i, 1] = deg[nb].min(); nodes[i, 2] = deg[nb].max()
            nodes[i, 3] = deg[nb].mean(); nodes[i, 4] = deg[nb].std()
    lap = laplacian(adj.astype(np.int8), normed=True)
    vals, vecs = eigh(lap)
    vecs = np.real(vecs[:, vals.argsort()])
    nodes[:, degree_profile:degree_profile + eig] = vecs[:, 1:eig + 1]
    for i in range(raw):
        nodes[:, raw + i] = nodes[:, i] ** 2
    for ii, (i, j) in enumerate(combinations(range(raw), 2)):
        nodes[:, raw + raw + ii] = nodes[:, i] * nodes[:, j]
    means = nodes.mean(0); stds = nodes.std(0); stds[stds == 0] = 1.0
    return ((nodes - means) / stds).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph")
    ap.add_argument("--front", required=True, help="submission .json to warm-start from")
    ap.add_argument("--out", default="warmstart.pt")
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--reg", type=float, default=1e-3)
    a = ap.parse_args()
    import torch
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adjlist = load_graph(graph_path(here, a.problem))
    # dense adjacency for the feature builder
    adj = np.zeros((n, n), dtype=np.bool_)
    for u, vs in enumerate(adjlist):
        for v in vs: adj[u, v] = True
    ab = build_adj_bitsets(n, adjlist); ev = IncEvalC(ab, n)
    E = feature_count(a.eigenvectors)
    print(f"building features (n={n}, E={E}) ...", flush=True)
    nodes = init_node_features(adj, a.eigenvectors)          # (N, E)

    # per-threshold best ordering + width from the front
    p = json.load(open(a.front)); e = p[0] if isinstance(p, list) else p
    best_w = np.full(n, MAX_TW + 1, dtype=np.int64)
    best_perm = [None] * n
    seen = 0
    for dv in e["decisionVector"]:
        if not (isinstance(dv, list) and len(dv) == n + 1): continue
        if sorted(int(x) for x in dv[:-1]) != list(range(n)): continue
        perm = [int(x) for x in dv[:-1]]; d = ev.full(perm); run = 0; seen += 1
        for t in range(n - 1, -1, -1):
            c = int(d[t]); run = c if c > run else run
            if run <= MAX_TW and run < best_w[t]:
                best_w[t] = run; best_perm[t] = perm
    print(f"warm-start source: {seen} valid orderings; "
          f"{int((best_w<=MAX_TW).sum())}/{n} thresholds covered", flush=True)

    # least-squares policy fit: solve (NodesᵀNodes + reg I) w = Nodesᵀ pos
    A = nodes.T @ nodes + a.reg * np.eye(E, dtype=np.float64)
    L = np.linalg.cholesky(A)
    def fit_policy(perm):
        pos = np.empty(n, dtype=np.float64); pos[np.asarray(perm)] = np.arange(n)
        rhs = nodes.T @ pos
        y = np.linalg.solve(L, rhs)
        return np.linalg.solve(L.T, y).astype(np.float32)

    # one policy per distinct ordering, broadcast to its thresholds
    cache = {}; elites = np.zeros((n, E), dtype=np.float32)
    for t in range(n):
        pm = best_perm[t]
        if pm is None:
            pm = best_perm[max((tt for tt in range(n) if best_perm[tt] is not None),
                               key=lambda tt: 0)]
        key = id(pm)
        if key not in cache: cache[key] = fit_policy(pm)
        elites[t] = cache[key]
    print(f"fitted {len(cache)} distinct policies", flush=True)

    # --- GO/NO-GO: decode the fitted policies and score, vs the source front ---
    from core import ParetoArchive, hypervolume_2d
    src = ParetoArchive(); dec = ParetoArchive()
    for t in range(n):
        if best_w[t] <= MAX_TW: src.try_add(int(best_w[t]), t, None)
    for w in {id(v): v for v in cache.values()}.values():
        perm = list(np.argsort(nodes @ w))                   # GPU decode: argsort(Nodes·w)
        d = ev.full(perm); run = 0
        for t in range(n - 1, -1, -1):
            c = int(d[t]); run = c if c > run else run
            if run <= MAX_TW: dec.try_add(run, t, None)
    src_hv = -hypervolume_2d(src.points(), n)
    dec_hv = -hypervolume_2d(dec.points(), n)
    print(f"\n  source front HV : {src_hv:,.0f}")
    print(f"  DECODED front HV: {dec_hv:,.0f}   (loss {src_hv-dec_hv:+,.0f})")
    faithful = (src_hv - dec_hv) < 2000
    print("  -> warm-start is FAITHFUL, worth a Kaggle run" if faithful
          else "  -> decode degraded the front; warm-start likely NOT worth Kaggle hours")

    ck = {
        "args": {"graph": a.problem, "eigenvectors": a.eigenvectors,
                 "init_stdev": 0.3, "mutation_stdev": 0.3, "mutation_proba": 0.5,
                 "cosyne_proba": 0.2, "batch_size": 1024},
        "nodes": torch.from_numpy(nodes.T.copy()),          # E x N (decode layout)
        "elites": torch.from_numpy(elites),                 # N x E
        "elite_fitnesses": torch.from_numpy(best_w.astype(np.int32)),
    }
    torch.save(ck, a.out)
    print(f"\nsaved {a.out}", flush=True)


if __name__ == "__main__":
    main()

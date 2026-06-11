#!/usr/bin/env python3
"""
qne_search.py -- Quality-diversity NeuroEvolution (QNE).

Learned from the leaderboard winner (cuda-torso): the decode is the SAME as ours
(argsort of a linear policy over node features), but the winning structure is
different in three ways we were missing:

  1. PER-THRESHOLD ELITES (the key): keep a SEPARATE best policy for every
     threshold t, not one compromise policy + a pooled front. The realized front
     is the envelope of N threshold-specialists -- which is why the winner sits
     ~20 width-units below our single-policy front at *every* t.
  2. FULL POLYNOMIAL FEATURES: degree-profile(5) + Laplacian eigenvectors(K),
     then squares and ALL pairwise interactions, normalised (not a random subset).
  3. NEUROEVOLUTION: batch mutation + COSYNE gene-level recombination, large
     population, many generations.

This reproduces that algorithm on OUR bitset GPU evaluator (more memory-efficient
than the winner's dense N*N adjacency, so larger batches fit). It adds one thing
of our own: an optional GBDT-learned feature column (`--gbdt`), so we can ablate
whether boosted-tree augmentation improves the state-of-the-art method.

    python3 tools/qne_search.py --problem large-graph --batch 1024 --budget 3600
    python3 tools/qne_search.py --problem large-graph --gbdt --budget 3600   # +GBDT
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, math, time
from itertools import combinations
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root, MAX_TW,
                  submission_path, write_submission, LEADERBOARD_TARGETS, min_degree_perm)
from algorithms.continuous.gpu_eval import build_adj_words, cpu_eval_batch, staircase_widths
from algorithms.continuous.cmaes_torso import get_features


def poly_features(F):
    """[raw | squares | all pairwise interactions], standardised -> (N, E)."""
    N, r = F.shape
    parts = [F, F * F]
    inter = np.empty((N, r * (r - 1) // 2), dtype=np.float64)
    for k, (i, j) in enumerate(combinations(range(r), 2)):
        inter[:, k] = F[:, i] * F[:, j]
    Phi = np.hstack(parts + [inter])
    mu = Phi.mean(0); sd = Phi.std(0); sd[sd == 0] = 1.0
    return ((Phi - mu) / sd).astype(np.float32)


def base_features(here, problem, n, adj, k_eig):
    """degree-profile(5) + K normalized-Laplacian eigenvectors -- the winner's raw
    features, via our cached numpy-only builder (no scipy needed)."""
    F, _ = get_features(here, problem, n, adj, k_eig)
    return np.asarray(F, dtype=np.float64)


def widths_of(perms, eval_fn, n):
    """width(t) per policy. Feasible rows -> true staircase width. Infeasible rows
    -> a uniformly-large but GRADED value (mirrors cuda-torso's penalty), so the
    least-infeasible policy is still preferred and neuroevolution can climb toward
    feasibility from an all-infeasible start."""
    deg, status = eval_fn(perms)
    W = staircase_widths(deg).astype(np.int32)
    over = np.clip(deg - MAX_TW, 0, None)               # per-step cap overflow
    pen = over.sum(axis=1).astype(np.int32)             # total violation (graded)
    has = pen > 0
    infeas = (status != 0) | has
    if infeas.any():
        # 501 base + graded violation, uniform across t so it's "all bad, ranked"
        W[infeas] = (MAX_TW + 1 + 501 * has[infeas] + pen[infeas])[:, None]
    return W


def hvi_select(elite_w, n, k=20):
    """Greedy top-k HVI selection over the per-threshold best widths (winner's rule)."""
    w = elite_w.astype(np.int64); ts = np.arange(n)
    min_d = np.full(n, n, dtype=np.int64); min_t = np.full(n, n, dtype=np.int64)
    sel = []
    for _ in range(k):
        area = (min_t - ts) * (min_d - w)
        a = int(np.argmax(area))
        sel.append(a); t = ts[a]; d = w[a]
        i = a
        while 0 <= i < n and min_d[i] > d: min_d[i] = d; i += 1
        i = a
        while 0 <= i < n and min_t[i] > t: min_t[i] = t; i -= 1
    return sel


def make_eval(aw, n, W, cap, capacity):
    try:
        from numba import cuda
        if not cuda.is_available(): raise RuntimeError("no cuda")
        from algorithms.continuous.gpu_eval import GpuEvaluator
        ev = GpuEvaluator(aw, n, W, cap=cap, capacity=capacity)
        return (lambda p: ev.eval(p)), "GPU"
    except Exception as e:  # noqa: BLE001
        print(f"[GPU unavailable: {e}; numpy reference]", flush=True)
        return (lambda p: cpu_eval_batch(p, aw, n, W, cap)), "CPU-ref"


def run(problem, budget_s, seed, batch, here, k_eig=32, use_gbdt=False,
        mut_std=0.3, mut_p=0.5, cosyne_p=0.2, init_std=0.3, algo=None):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    aw, Wd = build_adj_words(n, ab)
    target = LEADERBOARD_TARGETS.get(problem)
    algo = algo or ("qne_gbdt" if use_gbdt else "qne")
    rng = np.random.default_rng(seed)
    print(f"\n=== QNE -- {problem} (batch={batch}, gbdt={use_gbdt}) ===")
    t0 = time.time()
    F = base_features(here, problem, n, adj, k_eig)
    Phi = poly_features(F)                                  # (N, E)
    if use_gbdt:
        Phi = np.hstack([Phi, gbdt_column(here, problem, n, ab, F, seed)[:, None]]).astype(np.float32)
    E = Phi.shape[1]; nodesT = Phi.T.copy()                # (E, N)
    eval_fn, lab = make_eval(aw, n, Wd, MAX_TW, batch)
    print(f"features E={E}  evaluator={lab}  built {time.time()-t0:.1f}s", flush=True)

    # WARM START. Fit policies whose argsort reproduces (a) a min-degree ordering
    # and (b) every banked elite ordering, and put them in the initial population.
    # So gen 0 already sits at the banked best (-5.43M on large), and the
    # per-threshold neuroevolution REFINES from there instead of crawling up from
    # random over ~100k gens (the winner's compute, which we don't have).
    A = (nodesT @ nodesT.T + 1.0 * np.eye(E, dtype=np.float64))   # (E,E)
    Achol = np.linalg.cholesky(A)
    def fit_policy(perm):
        pos = np.empty(n); pos[np.asarray(perm)] = np.arange(n)
        rhs = nodesT @ pos
        z = np.linalg.solve(Achol, rhs); return np.linalg.solve(Achol.T, z).astype(np.float32)
    seeds = [fit_policy(min_degree_perm(n, ab))]
    for p in _banked_orderings(here, problem, n)[:batch - 1]:
        seeds.append(fit_policy(p))
    S = np.array(seeds, dtype=np.float32)
    pop = (rng.normal(0, init_std, (batch, E))).astype(np.float32)
    k = min(len(S), batch); pop[:k] = S[:k]                      # seed feasible policies
    pop[k:] += S[0]                                              # rest = min-degree + noise
    elite_pol = np.tile(S[0], (n, 1)).astype(np.float32)
    elite_w = np.full(n, n + 2000, dtype=np.int32)
    gen = 0; t0 = time.time()
    while time.time() - t0 < budget_s:
        perms = np.argsort(pop @ nodesT, axis=1).astype(np.int32)
        W = widths_of(perms, eval_fn, n)                   # (batch, n)
        bidx = W.argmin(0); bw = W[bidx, np.arange(n)]     # best width per threshold
        better = bw < elite_w
        elite_pol[better] = pop[bidx[better]]
        elite_w[better] = bw[better]
        # next population: sample per-threshold elites, mutate, COSYNE
        idx = rng.integers(0, n, size=batch)
        pop = elite_pol[idx].copy()
        mask = rng.random((batch, E)) > mut_p
        pop += (rng.normal(0, mut_std, (batch, E)) * mask).astype(np.float32)
        npx = int(batch * E * cosyne_p)
        tr = rng.integers(0, batch, npx); orr = rng.integers(0, batch, npx); cc = rng.integers(0, E, npx)
        pop[tr, cc] = pop[orr, cc]
        gen += 1
        if time.time() - t0 >= 5 * (gen // max(1, gen)) or True:
            if gen % 5 == 0 or time.time() - t0 >= budget_s:
                sel = hvi_select(elite_w, n)
                hv = sum((n - elite_w[s]) for s in sel)    # quick proxy
                feas = int((elite_w < MAX_TW + 1).sum())
                msg = f"  gen {gen:5d} | feasible-t {feas:4d}/{n} | proxyHV {hv:>12,} | t {time.time()-t0:5.0f}s"
                print(msg, flush=True)

    # build submission from per-threshold elites
    sel = hvi_select(elite_w, n)
    logits = elite_pol[sel] @ nodesT
    perms = np.argsort(logits, axis=1).astype(np.int32)
    # official re-score of the selected (perm, t) pairs via our archive
    from core import ParetoArchive, evaluate as core_eval
    arch = ParetoArchive()
    for r, t in enumerate(sel):
        p = perms[r].tolist()
        w, _ = core_eval(p, int(t), ab, n)
        if w <= MAX_TW: arch.try_add(int(w), int(t), p)
    # also harvest full staircases of the selected policies
    for r in range(len(sel)):
        dd, ss = eval_fn(perms[r:r+1]);
        if ss[0] == 0:
            wmax = staircase_widths(dd)[0]
            run_ = 0
            for t in range(n-1, -1, -1):
                if wmax[t] > run_: run_ = int(wmax[t])
                if run_ <= MAX_TW: arch.try_add(run_, t, perms[r].tolist())
    final = -arch.hypervolume(n)
    print(f"\nFinished {time.time()-t0:.0f}s, {gen} gens")
    print(f"Official score: {final:,.0f}")
    if target is not None:
        g = final - target
        print(f"Gap to target ({target:,}): {g:>+,.0f}  ({'BEAT THE LEADER' if g < 0 else f'{abs(g):,.0f} short'})")
    top = arch.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = submission_path(here, problem, algo); write_submission(dvs, problem, out)
    print(f"Wrote {out} ({len(dvs)} vectors)")
    return final


def _banked_orderings(here, problem, n):
    """All banked elite orderings for warm-starting the per-threshold elites."""
    out = []; seen = set()
    sub = os.path.join(here, "submissions", problem)
    files = ([os.path.join(sub, "portfolio.json")] +
             sorted(glob.glob(os.path.join(sub, "*.json"))) +
             sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))))
    for fp in files:
        if not os.path.exists(fp): continue
        try: dvs = json.load(open(fp))[0]["decisionVector"]
        except Exception: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1 and \
                    sorted(int(x) for x in dv[:-1]) == list(range(n)):
                k = tuple(int(x) for x in dv[:-1])
                if k not in seen: seen.add(k); out.append(list(k))
    return out


def gbdt_column(here, problem, n, ab, F, seed):
    """OUR contribution: a GBDT-learned position score as an extra feature."""
    from algorithms.continuous.gbdt_torso import make_gbdt, training_set
    elites = []
    for stem in ("gaps", "portfolio", "gpucma", "gbdt"):
        for fp in glob.glob(os.path.join(here, "submissions", problem, f"{stem}.json")):
            try: dvs = json.load(open(fp))[0]["decisionVector"]
            except Exception: continue
            for dv in dvs:
                if isinstance(dv, list) and len(dv) == n + 1 and sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    elites.append([int(x) for x in dv[:-1]])
    if len(elites) < 4:
        import random
        elites += [min_degree_perm(n, ab, rng=random.Random(seed + s)) for s in range(8)]
    X, y, w = training_set([(1.0, p) for p in elites[:80]], F, n)
    _, gb = make_gbdt("auto", seed)
    try:
        gb.fit(X, y); col = np.asarray(gb.predict(F), float)
    except Exception:
        col = np.zeros(n)
    return ((col - col.mean()) / (col.std() + 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--gbdt", action="store_true", help="add the GBDT-learned feature column (our ablation)")
    ap.add_argument("--algo", default=None)
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.batch, repo_root(),
        k_eig=args.k_eig, use_gbdt=args.gbdt, algo=args.algo)


if __name__ == "__main__":
    main()

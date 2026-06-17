#!/usr/bin/env python3
"""
gbfc.py -- Gradient-Boosted Front Construction (the novel method).

Idea: the multi-objective front is a collection of THRESHOLD SPECIALISTS (a
different best ordering per torso size). The leaderboard winner discovers them by
brute force (100k generations of neuroevolution). Boosting builds a strong
ensemble from weak specialists, each fixing what the others get wrong -- so we
build the front the same way, on purpose:

  pool <- banked orderings
  repeat R rounds:
    1. compute the pooled front width(t) and its RESIDUAL = gap to the treewidth
       lower bound at each torso size (the recoverable room; core ~ 0).
    2. pick the worst-served threshold band (largest residual).
    3. fit a GBDT-decoded ordering SPECIALISED to that band (a short CMA search
       whose fitness is the hypervolume restricted to the band), warm-started
       from the current best-in-band ordering -- the boosting "weak learner".
    4. add the specialist to the pool. The residual shifts. Repeat.

GBDT is the weak learner; the residual is the boosting target; the pool is the
additive ensemble. This obtains the winner's per-threshold specialisation by a
principled, sample-efficient algorithm instead of brute-force compute.

    python3 tools/gbfc.py --problem large-graph --rounds 12 --round-budget 60
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, math, time, random
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root, ParetoArchive,
                  hypervolume_2d, MAX_TW, min_degree_perm, submission_path, write_submission,
                  load_decision_vectors, LEADERBOARD_TARGETS, treewidth_lower_bound_mmd)
from algorithms.continuous.cmaes_torso import SepCMAES, get_features
from algorithms.continuous.gbdt_torso import make_gbdt, training_set
from algorithms.continuous.gpu_eval import build_adj_words, cpu_eval_batch, staircase_widths

INFEAS = 1e9


def standardize(M):
    mu = M.mean(0); sd = M.std(0); sd[sd == 0] = 1.0
    return ((M - mu) / sd).astype(np.float64)


def make_eval(aw, n, W, cap, capacity):
    try:
        from numba import cuda
        if not cuda.is_available(): raise RuntimeError("no cuda")
        from algorithms.continuous.gpu_eval import GpuEvaluator
        ev = GpuEvaluator(aw, n, W, cap=cap, capacity=capacity)
        return (lambda p: ev.eval(p)), "GPU"
    except Exception as e:  # noqa: BLE001
        print(f"[GPU unavailable: {e}; numpy ref]", flush=True)
        return (lambda p: cpu_eval_batch(p, aw, n, W, cap)), "CPU-ref"


def deg_seq(perm, ab, n):
    sm = [0]*n; cur = 0
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    return deg


def banked(here, problem, n, ab=None):
    out = []; seen = set(); sub = os.path.join(here, "submissions", problem)
    for fp in [os.path.join(sub, "portfolio.json")] + sorted(glob.glob(os.path.join(sub, "*.json"))) \
              + sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))):
        if not os.path.exists(fp): continue
        dvs = load_decision_vectors(fp)
        if not dvs: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n+1 and sorted(int(x) for x in dv[:-1]) == list(range(n)):
                k = tuple(int(x) for x in dv[:-1])
                if k not in seen: seen.add(k); out.append(list(k))
    # fallback for fresh checkouts (e.g. Colab): the distilled warm-start file
    ws = os.path.join(here, "data", f"warmstart_{problem}.json")
    if os.path.exists(ws):
        try:
            for p in json.load(open(ws)).get("perms", []):
                k = tuple(int(x) for x in p)
                if len(p) == n and sorted(k) == list(range(n)) and k not in seen:
                    seen.add(k); out.append(list(p))
        except Exception: pass
    # last resort: min-degree orderings, so GBFC never starts empty
    if not out and ab is not None:
        for s in range(8):
            out.append(min_degree_perm(n, ab, rng=random.Random(s)))
    return out


def lb_grid(pool_perm, ab, adj, n, npts=6):
    """treewidth LB of the induced torso at a grid of thresholds (recoverable-room ref)."""
    ts = sorted(set(int(round(i*(n-1)/(npts-1))) for i in range(npts)))
    lb = {}
    for t in ts:
        V = pool_perm[t:]; Vset = set(V); idx = {v: j for j, v in enumerate(V)}; m = len(V)
        sub = [0]*m
        for v in V:
            for u in adj[v]:
                if u in Vset: sub[idx[v]] |= 1 << idx[u]
        lb[t] = treewidth_lower_bound_mmd(m, sub)
    return ts, lb


def fit_gbdt_col(F, pool, n, seed, t_lo, t_hi, ab):
    """GBDT column trained on the pool orderings that are BEST in [t_lo,t_hi]."""
    scored = []
    for p in pool:
        d = deg_seq(p, ab, n); w = np.maximum.accumulate(d[::-1])[::-1]
        scored.append((int(w[t_lo:t_hi+1].max()), p))     # band width (lower = better)
    scored.sort(key=lambda z: z[0])
    top = [p for _, p in scored[:max(4, len(scored)//3)]]
    X, y, w = training_set([(1.0, p) for p in top], F, n)
    _, gb = make_gbdt("auto", seed)
    try:
        gb.fit(X, y); col = np.asarray(gb.predict(F), float)
    except Exception: col = np.zeros(n)
    return standardize(col[:, None])[:, 0]


def search_band(F, gcol, ab, aw, n, W, band_ts, eval_fn, arch, pop, budget, seed, best_in_band):
    """Short CMA over a GBDT-augmented decode, fitness = HV over the band only."""
    Phi = np.hstack([standardize(F), standardize(F*F), gcol[:, None]]).astype(np.float64)
    d = Phi.shape[1]
    def ridge(perm):
        pos = np.empty(n); pos[np.asarray(perm)] = np.arange(n)
        Pb = np.hstack([Phi, np.ones((n, 1))]); A = Pb.T@Pb + np.eye(d+1)
        return np.linalg.solve(A, Pb.T@pos)[:-1]
    opt = SepCMAES(d, 0.4, seed); opt.lam = pop; opt.mu = pop//2
    wts = np.log(opt.mu+0.5)-np.log(np.arange(1, opt.mu+1)); opt.w = wts/wts.sum()
    opt.mueff = 1/np.sum(opt.w**2); opt.mean = ridge(best_in_band)
    t0 = time.time()
    while time.time()-t0 < budget:
        X = opt.ask(); perms = np.argsort(Phi@X.T, axis=0).T.astype(np.int32)
        deg, status = eval_fn(perms); widths = staircase_widths(deg)
        fit = np.full(pop, INFEAS)
        for p in np.where(status == 0)[0]:
            front = [(int(widths[p, t]), int(t)) for t in band_ts]
            fit[p] = -hypervolume_2d(front, n)
            for w_, t_ in [(int(widths[p, t]), int(t)) for t in range(n)]:
                arch.try_add(w_, t_, perms[p].tolist())
        opt.tell(fit)


def run(problem, rounds, round_budget, seed, here, pop=64, k_eig=32, algo="gbfc"):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    aw, Wd = build_adj_words(n, ab); target = LEADERBOARD_TARGETS.get(problem)
    print(f"\n=== GBFC -- {problem} (rounds={rounds}, {round_budget}s/round) ===")
    F, _ = get_features(here, problem, n, adj, k_eig); F = np.asarray(F)
    eval_fn, lab = make_eval(aw, n, Wd, MAX_TW, pop)
    pool = banked(here, problem, n, ab)[:60]      # cap: keep the front-defining set
    arch = ParetoArchive()
    for p in pool:
        d = deg_seq(p, ab, n); w = np.maximum.accumulate(d[::-1])[::-1]
        for t in range(n):
            if w[t] <= MAX_TW: arch.try_add(int(w[t]), t, p)
    best = -arch.hypervolume(n)
    print(f"evaluator={lab}; pooled {len(pool)} banked -> {best:,.0f}", flush=True)

    # recoverable-room reference: treewidth LB at a grid (computed once)
    base = min(pool, key=lambda p: int(np.maximum.accumulate(deg_seq(p, ab, n)[::-1])[::-1][0]))
    ts, lb = lb_grid(base, ab, adj, n); print("LB grid ready", flush=True)

    out = submission_path(here, problem, algo)
    def save_sub():
        top = arch.top_k_by_hv_contribution(20, n)
        write_submission([list(p)+[int(t)] for (_, t, p) in top], problem, out)
    save_sub()                                       # checkpoint immediately

    nb = 8; edges = np.linspace(0, n, nb+1).astype(int); exhausted = set()
    for r in range(rounds):
        # current pooled front widths
        W = np.full(n, n, dtype=np.int64)
        for p in pool:
            W = np.minimum(W, np.maximum.accumulate(deg_seq(p, ab, n)[::-1])[::-1])
        # residual per band: mean (width - interpolated LB), recoverable room
        lbi = np.interp(np.arange(n), ts, [lb[t] for t in ts])
        resid = np.clip(W - lbi, 0, None)
        band_gap = [resid[edges[b]:edges[b+1]].mean() for b in range(nb)]
        cand = [bi for bi in range(nb) if bi not in exhausted]
        if not cand: cand = list(range(nb)); exhausted.clear()    # all tried: retry
        b = max(cand, key=lambda bi: band_gap[bi]); lo, hi = edges[b], min(edges[b+1], n-1)
        prev = best
        band_ts = sorted(set(int(round(lo + i*(hi-lo)/19)) for i in range(20)))
        # best current ordering in this band -> warm start
        bestp = min(pool, key=lambda p: int(np.maximum.accumulate(deg_seq(p, ab, n)[::-1])[::-1][lo:hi+1].max()))
        gcol = fit_gbdt_col(F, pool, n, seed+r, lo, hi, ab)            # GBDT weak learner
        search_band(F, gcol, ab, aw, n, Wd, band_ts, eval_fn, arch, pop, round_budget, seed+r, bestp)
        # refresh pool from the (possibly improved) archive top set
        top = arch.top_k_by_hv_contribution(40, n); pool = [list(p) for _, _, p in top] + pool[:40]
        cur = -arch.hypervolume(n)
        if cur >= prev - 1: exhausted.add(b)          # band gave nothing: move on
        best = min(best, cur)
        save_sub()                                   # checkpoint after EVERY round (timeout-safe)
        msg = f"  round {r+1:2d} | band [{lo},{hi}] gap~{band_gap[b]:.0f} | score {best:,.0f}"
        if target is not None: msg += f" | gap {best-target:+,.0f}" + (" BEAT!" if best < target else "")
        print(msg, flush=True)

    final = -arch.hypervolume(n)
    print(f"\nFinished. Official score: {final:,.0f}")
    if target is not None:
        g = final-target; print(f"Gap to target ({target:,}): {g:+,.0f} ({'BEAT' if g<0 else f'{abs(g):,.0f} short'})")
    save_sub()
    print(f"Wrote {out}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--round-budget", type=float, default=60.0)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algo", default="gbfc")
    args = ap.parse_args()
    run(args.problem, args.rounds, args.round_budget, args.seed, repo_root(),
        pop=args.pop, algo=args.algo)


if __name__ == "__main__":
    main()

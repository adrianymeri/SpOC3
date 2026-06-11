#!/usr/bin/env python3
"""
qnegbfc.py -- QNE + GBFC HYBRID (the leaderboard shot, GBDT-central).

Why a hybrid. GBFC plateaus because it warm-starts from the banked front and only
boosts locally -- it stays in our basin. The leaderboard winner instead explored
WIDELY (per-threshold neuroevolution from random, ~100k generations). This hybrid
combines both:

  * ENGINE: the winner's per-threshold neuroevolution (QNE) over the full
    polynomial spectral features -- wide exploration, a separate elite policy per
    threshold. Seeded PARTLY from the banked best (so it never regresses) and
    PARTLY random (so it explores like the winner).
  * BOOST: every `--inject-every` generations, GBFC's residual-targeted GBDT
    weak learner -- find the worst threshold band, fit a GBDT specialist on the
    elite orderings best in that band, decode candidates, and merge any
    improvement into the per-threshold elites.

The controlled comparison (`--no-gbdt` disables the injection) isolates GBDT's
contribution to the SOTA method -- the novelty claim. Checkpoint-safe: the
submission is rewritten periodically, so long Kaggle/own-GPU runs never lose work.

    python3 tools/qnegbfc.py --problem small-graph --batch 1024 --budget 36000
    python3 tools/qnegbfc.py --problem small-graph --no-gbdt --budget 36000   # ablation
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, time, random
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root, MAX_TW,
                  submission_path, write_submission, LEADERBOARD_TARGETS, min_degree_perm,
                  ParetoArchive, treewidth_lower_bound_mmd)
from algorithms.continuous.gpu_eval import build_adj_words, cpu_eval_batch, staircase_widths
from algorithms.continuous.cmaes_torso import get_features
from tools.qne_search import poly_features, hvi_select, make_eval, widths_of
from tools.gbfc import deg_seq, banked as gbfc_banked, fit_gbdt_col


def run(problem, budget_s, seed, batch, here, k_eig=32, use_gbdt=True,
        inject_every=40, ckpt_every=120, mut_std=0.3, mut_p=0.5, cosyne_p=0.2,
        rand_frac=0.5, algo=None):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    aw, Wd = build_adj_words(n, ab); target = LEADERBOARD_TARGETS.get(problem)
    algo = algo or ("qnegbfc" if use_gbdt else "qne_wide")
    rng = np.random.default_rng(seed)
    print(f"\n=== QNE+GBFC hybrid -- {problem} (batch={batch}, gbdt={use_gbdt}) ===")
    t0 = time.time()
    F = np.asarray(get_features(here, problem, n, adj, k_eig)[0])
    Phi = poly_features(F); E = Phi.shape[1]; nodesT = Phi.T.copy()
    eval_fn, lab = make_eval(aw, n, Wd, MAX_TW, batch)
    print(f"features E={E}  evaluator={lab}  built {time.time()-t0:.1f}s", flush=True)

    # ridge fits to seed feasible/banked policies
    A = nodesT @ nodesT.T + 1.0 * np.eye(E); Achol = np.linalg.cholesky(A)
    def fit(perm):
        pos = np.empty(n); pos[np.asarray(perm)] = np.arange(n)
        z = np.linalg.solve(Achol, nodesT @ pos); return np.linalg.solve(Achol.T, z).astype(np.float32)
    pool = gbfc_banked(here, problem, n, ab)[:batch]
    seeds = [fit(min_degree_perm(n, ab))] + [fit(p) for p in pool]
    S = np.array(seeds[:batch], dtype=np.float32)

    # population: part banked-seed, part random (wide exploration)
    pop = rng.normal(0, mut_std, (batch, E)).astype(np.float32)
    k = min(len(S), int(batch * (1 - rand_frac)))
    pop[:k] = S[:k]                                    # banked anchors (no regression)
    elite_pol = np.tile(S[0], (n, 1)).astype(np.float32)
    elite_w = np.full(n, n + 2000, dtype=np.int32)

    # treewidth-LB grid for residual targeting (computed once)
    base = pool[0] if pool else min_degree_perm(n, ab)
    ts = sorted(set(int(round(i*(n-1)/5)) for i in range(6))); lb = {}
    for t in ts:
        V = base[t:]; Vs = set(V); idx = {v: j for j, v in enumerate(V)}; m = len(V)
        sub = [0]*m
        for v in V:
            for u in adj[v]:
                if u in Vs: sub[idx[v]] |= 1 << idx[u]
        lb[t] = treewidth_lower_bound_mmd(m, sub)
    lbi = np.interp(np.arange(n), ts, [lb[t] for t in ts])
    nb = 8; edges = np.linspace(0, n, nb+1).astype(int)

    out = submission_path(here, problem, algo)
    # PERSISTENT best-of-union archive, SEEDED with the banked front so the
    # reported/saved score never regresses below the banked best -- the hybrid
    # only has to find genuine improvements on top.
    arch = ParetoArchive()
    for p in pool[:80]:
        w = np.maximum.accumulate(deg_seq(p, ab, n)[::-1])[::-1]
        for t in range(n-1, -1, -1):
            if int(w[t]) <= MAX_TW: arch.try_add(int(w[t]), t, list(p))
    def archive_from_elites():
        sel = hvi_select(elite_w, n)
        perms = np.argsort(elite_pol[sel] @ nodesT, axis=1).astype(np.int32)
        for r in range(len(sel)):
            d, s = eval_fn(perms[r:r+1])
            if s[0] == 0:
                w = staircase_widths(d)[0]; run_ = 0
                for t in range(n-1, -1, -1):
                    if w[t] > run_: run_ = int(w[t])
                    if run_ <= MAX_TW: arch.try_add(run_, t, perms[r].tolist())
        return arch
    def save():
        archive_from_elites()
        top = arch.top_k_by_hv_contribution(20, n)
        write_submission([list(p)+[int(t)] for (_, t, p) in top], problem, out)
        return -arch.hypervolume(n)

    gen = 0; best = 1e18; lastck = t0
    while time.time() - t0 < budget_s:
        pop_perms = np.argsort(pop @ nodesT, axis=1).astype(np.int32)
        W = widths_of(pop_perms, eval_fn, n)
        bi = W.argmin(0); bw = W[bi, np.arange(n)]
        better = bw < elite_w
        elite_pol[better] = pop[bi[better]]; elite_w[better] = bw[better]
        # neuroevolution: sample elites + mutate + COSYNE, keep a random fraction
        idx = rng.integers(0, n, size=batch); pop = elite_pol[idx].copy()
        nr = int(batch * rand_frac)
        pop[:nr] = rng.normal(0, mut_std, (nr, E)).astype(np.float32) + S[0]   # explorers
        mask = rng.random((batch, E)) > mut_p
        pop += (rng.normal(0, mut_std, (batch, E)) * mask).astype(np.float32)
        npx = int(batch * E * cosyne_p)
        tr = rng.integers(0, batch, npx); orr = rng.integers(0, batch, npx); cc = rng.integers(0, E, npx)
        pop[tr, cc] = pop[orr, cc]
        gen += 1
        # ---- GBFC injection (the GBDT boost) ----
        if use_gbdt and gen % inject_every == 0:
            W_el = elite_w.astype(np.float64); resid = np.clip(W_el - lbi, 0, None)
            b = int(np.argmax([resid[edges[i]:edges[i+1]].mean() for i in range(nb)]))
            lo, hi = edges[b], min(edges[b+1], n-1)
            elite_perms = [np.argsort(elite_pol[t] @ nodesT).tolist() for t in
                           sorted(set(int(round(lo + i*(hi-lo)/9)) for i in range(10)))]
            gcol = fit_gbdt_col(F, elite_perms, n, seed+gen, lo, hi, ab)
            Phi_g = np.hstack([Phi, gcol[:, None]]).astype(np.float32)
            # Build a DIVERSE specialist pool. The old code only ridge-refit existing
            # elites (local -> imitation-bounded -> flat). We now also GENERATE new
            # orderings the current pool does not contain, so the GBDT half explores.
            spec = []
            # (a) ridge fits over the band's best (local refinement, as before)
            for ep in elite_perms[:6]:
                pos = np.empty(n); pos[np.asarray(ep)] = np.arange(n)
                Pb = np.hstack([Phi_g, np.ones((n, 1))]); Aa = Pb.T@Pb + np.eye(E+2)
                w = np.linalg.solve(Aa, Pb.T@pos)
                spec.append(np.argsort(Phi_g @ w[:-1]).astype(np.int32))
            # (b) the PURE GBDT-score orderings (off the current policy manifold)
            spec.append(np.argsort(gcol).astype(np.int32))
            spec.append(np.argsort(-gcol).astype(np.int32))
            # (c) WIDE GBDT-guided explorers: random policies over [Phi | gcol] with the
            #     learned column up-weighted -> genuinely new orderings, not pool echoes
            for _ in range(24):
                x = rng.normal(0, 1, E + 1).astype(np.float32)
                x[-1] *= float(rng.uniform(2.0, 6.0))      # emphasise the learned column
                spec.append(np.argsort(Phi_g @ x).astype(np.int32))
            # (d) large-perturbation variants of the band's best elite policies
            for t in sorted(set(int(round(lo + i*(hi-lo)/5)) for i in range(6))):
                base = np.zeros(E + 1, dtype=np.float32); base[:E] = elite_pol[t]
                for _ in range(3):
                    xb = base + rng.normal(0, 0.8, E + 1).astype(np.float32)
                    spec.append(np.argsort(Phi_g @ xb).astype(np.int32))
            sp = np.array(spec, dtype=np.int32)
            dW = widths_of(sp, eval_fn, n)
            sbi = dW.argmin(0); sbw = dW[sbi, np.arange(n)]
            imp = sbw < elite_w
            if imp.any():
                # the specialist orderings are explicit; store via nearest policy fit
                for t in np.where(imp)[0]:
                    elite_w[t] = sbw[t]
                    elite_pol[t] = fit(sp[sbi[t]].tolist())
        if time.time() - lastck >= ckpt_every or time.time() - t0 >= budget_s:
            sc = save(); best = min(best, sc); lastck = time.time()
            feas = int((elite_w < MAX_TW + 1).sum())
            msg = f"  gen {gen:6d} | feasible-t {feas:4d}/{n} | official {sc:,.0f} | t {time.time()-t0:6.0f}s"
            if target is not None: msg += f" | gap {sc-target:+,.0f}" + (" BEAT!" if sc < target else "")
            print(msg, flush=True)

    sc = save()
    print(f"\nFinished {time.time()-t0:.0f}s, {gen} gens. Official score: {sc:,.0f}")
    if target is not None:
        g = sc-target; print(f"Gap to target ({target:,}): {g:+,.0f} ({'BEAT' if g<0 else f'{abs(g):,.0f} short'})")
    print(f"Wrote {out}")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=36000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--no-gbdt", action="store_true", help="disable GBFC injection (ablation)")
    ap.add_argument("--inject-every", type=int, default=40)
    ap.add_argument("--rand-frac", type=float, default=0.5)
    ap.add_argument("--algo", default=None)
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.batch, repo_root(), k_eig=args.k_eig,
        use_gbdt=not args.no_gbdt, inject_every=args.inject_every, rand_frac=args.rand_frac,
        algo=args.algo)


if __name__ == "__main__":
    main()

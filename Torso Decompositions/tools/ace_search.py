#!/usr/bin/env python3
"""
ace_search.py -- ACE: Adaptive-Constructor Evolution (the off-manifold method).

Every prior method (linear, poly, GAPS) decodes by ranking all vertices ONCE from
static features: argsort(score(node)). That is a fixed manifold. ACE breaks it by
making the decision PER STEP on the LIVE residual graph, and EVOLVING the picking
rule so it can surpass the orderings it was bootstrapped from.

At each elimination step, among a shortlist of low-degree live vertices, each
candidate is described by DYNAMIC features (current residual degree, immediate
fill-in cost, #eliminated neighbours) plus STATIC spectral features plus the
GAPS **GBDT** score (the learned teacher, as one input). A linear policy w scores
them; the best is eliminated; the graph updates; repeat. CMA-ES evolves w to
maximise the top-20 hypervolume -- "GBDT bootstraps, evolution surpasses."

  * off the static-decode manifold (decisions depend on the live state),
  * uses everything we have (the §6b constructor, GAPS's GBDT, spectral features),
  * w is warm-started to imitate min-degree, then evolved past it.

CPU works (small/medium); large wants the GPU-batched constructor (future).

    python3 tools/ace_search.py --problem medium-graph --budget 120
    python3 tools/ace_search.py --path extra_instances/data/inst_12_n500_d35.gr --budget 30
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, math, random, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root, ParetoArchive,
                  hypervolume_2d, MAX_TW, min_degree_perm, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import SepCMAES
from algorithms.continuous.gbdt_torso import (build_rich_features, make_gbdt, training_set,
                                              _construct)


def standardize(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def add_staircase(perm, deg, n, arch):
    if deg.max() > MAX_TW:
        # graded: still add feasible suffix where width<=cap
        pass
    run = 0
    for t in range(n - 1, -1, -1):
        if deg[t] > run: run = deg[t]
        if run <= MAX_TW:
            arch.try_add(int(run), t, list(perm))


def gather_elites(here, problem, path, n, ab, k_eig, F, seed):
    """Elite orderings to train the bootstrap GBDT: submission orderings if a named
    instance, else min-degree orderings (works for synthetic paths too)."""
    elites = []
    if problem:
        for stem in ("gaps", "portfolio", "gpucma", "gbdt"):
            for fp in glob.glob(os.path.join(here, "submissions", problem, f"{stem}.json")):
                try: dvs = json.load(open(fp))[0]["decisionVector"]
                except Exception: continue
                for dv in dvs:
                    if isinstance(dv, list) and len(dv) == n + 1 and \
                            sorted(int(x) for x in dv[:-1]) == list(range(n)):
                        elites.append([int(x) for x in dv[:-1]])
    if len(elites) < 4:
        for s in range(8):
            elites.append(min_degree_perm(n, ab, rng=random.Random(seed + s)))
    return elites[:80]


def run(problem, path, budget_s, seed, here, k_eig=32, pop=24, m_orders=3, algo="ace"):
    if path:
        n, adj = load_graph(path); name = os.path.splitext(os.path.basename(path))[0]
    else:
        n, adj = load_graph(graph_path(here, problem)); name = problem
    ab = build_adj_bitsets(n, adj)
    t_grid = sorted({int(round(i * (n - 1) / 39)) for i in range(40)})
    target = LEADERBOARD_TARGETS.get(problem) if problem else None
    print(f"\n=== ACE (adaptive-constructor evolution) -- {name} ===")
    print(f"n = {n}, edges = {sum(map(len, adj))//2}")
    F = build_rich_features(here, "ace_" + name, n, adj, ab, k_eig)

    # bootstrap GBDT teacher (static-feature -> position), as one policy input
    elites = gather_elites(here, problem, path, n, ab, k_eig, F, seed)
    X, y, w = training_set([(1.0, p) for p in elites], F, n)
    _, gb = make_gbdt("auto", seed)
    try:
        gb.fit(X, y); gbdt_static = standardize(gb.predict(F))
    except Exception:
        gbdt_static = np.zeros(n)
    F_aug = np.hstack([F, gbdt_static[:, None]])           # spectral + GBDT teacher column
    fdim = 3 + F_aug.shape[1]                              # [cur_deg, elimc, fill | F_aug]

    arch = ParetoArchive()
    # seed archive with EVERY elite's full staircase -> warm-start at the current
    # best front (fair test: can evolution push BELOW it?), not a weak 8-elite start.
    for p in elites:
        d = _front_deg(p, ab, n); add_staircase(p, d, n, arch)
    best = -arch.hypervolume(n)
    print(f"bootstrap archive {len(arch)} -> score {best:,.0f}  (policy dim {fdim})", flush=True)

    opt = SepCMAES(fdim, sigma0=0.4, seed=seed); opt.lam = pop; opt.mu = pop // 2
    wts = np.log(opt.mu + 0.5) - np.log(np.arange(1, opt.mu + 1)); opt.w = wts / wts.sum()
    opt.mueff = 1 / np.sum(opt.w ** 2)
    # WARM START the policy to imitate min-degree: high score for LOW residual degree
    w0 = np.zeros(fdim); w0[0] = -1.0; w0[2] = -0.3        # -cur_deg, mild -fill
    opt.mean = w0.copy()

    rng = np.random.default_rng(seed)
    t0 = time.time(); gen = 0
    while time.time() - t0 < budget_s:
        P = opt.ask(); fit = np.empty(pop)
        for i in range(pop):
            wv = P[i]
            sub = ParetoArchive()
            for m in range(m_orders):
                temp = 0.0 if m == 0 else float(rng.uniform(0.2, 0.7))
                perm, deg = _construct(lambda feat: feat @ wv, F_aug, ab, n, rng,
                                       shortlist=48, temp=temp)
                add_staircase(perm, deg, n, sub)
                add_staircase(perm, deg, n, arch)
            fit[i] = -sub.hypervolume(n) if len(sub) else 1e9
        opt.tell(fit)
        gen += 1
        cur = -arch.hypervolume(n); best = min(best, cur)
        msg = f"  gen {gen:3d} | score {best:,.0f} | t {time.time()-t0:5.0f}s"
        if target is not None:
            g = best - target; msg += f" | gap {g:+,.0f}" + (" BEAT!" if g < 0 else "")
        print(msg, flush=True)

    final = -arch.hypervolume(n)
    print(f"\nFinished in {time.time()-t0:.0f}s, {gen} gens, archive {len(arch)}")
    print(f"Official score: {final:,.0f}")
    if target is not None:
        g = final - target
        print(f"Gap to target ({target:,}): {g:>+,.0f}  ({'BEAT' if g < 0 else f'{abs(g):,.0f} short'})")
    if problem:
        top = arch.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = submission_path(here, problem, algo); write_submission(dvs, problem, out)
        print(f"Wrote {out} ({len(dvs)} vectors)")
    return final


def _front_deg(perm, ab, n):
    sm = [0]*n; cur = 0
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    return deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None, choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--path", default=None, help="arbitrary .gr instance (synthetic)")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--m-orders", type=int, default=3)
    ap.add_argument("--algo", default="ace")
    args = ap.parse_args()
    if not args.problem and not args.path:
        args.problem = "medium-graph"
    run(args.problem, args.path, args.budget, args.seed, repo_root(),
        k_eig=args.k_eig, pop=args.pop, m_orders=args.m_orders, algo=args.algo)


if __name__ == "__main__":
    main()

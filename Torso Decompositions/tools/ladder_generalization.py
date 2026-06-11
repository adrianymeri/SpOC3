#!/usr/bin/env python3
"""
ladder_generalization.py -- does the GAPS PRINCIPLE generalise across graphs?

THESIS §10.4 claims a principle: escaping the decode-expressiveness plateau needs
a decode that is *learned and adaptive to the instance*, not merely more
expressive. On the competition large-graph the ladder was

    linear  ~=  polynomial  ~=  random-nonlinear  <<  learned-GBDT.

This tool tests whether that ORDERING holds across many structurally-varied
graphs (the 20 synthetic instances: 5 sizes x 4 densities; extendable to external
corpora). For each instance it runs a short, identical CMA-ES search under four
decodes:

    linear   : argsort(F . x)                      (41-d spectral)
    poly     : argsort(Phi . x), Phi=[F, F^2]      (fixed nonlinear basis)
    random   : Phi + 128 random pairwise products  (undirected nonlinearity)
    gbdt     : argsort(Phi . x + b.g_GBDT(F))      (LEARNED, DAgger-retrained)

and reports the achieved hypervolume for each, the GBDT margin (gbdt - best of
the other three), and -- aggregated by density -- on how many instances the
learned-adaptive decode wins. A consistent win, especially growing with density,
turns the principle from an anecdote on one graph into a property of the problem
class.

CPU by default (numpy reference); uses the GPU batch evaluator if numba+CUDA.

    python3 tools/ladder_generalization.py --budget 12 --pop 96
    python3 tools/ladder_generalization.py --instances 4 --budget 8   # quick
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, math, random, re, time
import numpy as np
from core import (load_graph, build_adj_bitsets, repo_root, ParetoArchive,
                  hypervolume_2d, MAX_TW, min_degree_perm)
from algorithms.continuous.cmaes_torso import SepCMAES
from algorithms.continuous.gbdt_torso import build_rich_features, make_gbdt, training_set
from algorithms.continuous.gpu_eval import build_adj_words, cpu_eval_batch, staircase_widths
from tools.gaps_search import poly_expand, standardize, ridge_policy, row_penalties, make_evaluator

INFEAS = 1e9


def search(F, base, ab, aw, n, W, t_grid, gbdt_on, pop, budget, seed, eval_fn, gbdt_every=6):
    gbdt_col = np.zeros((n, 1))
    def feats():
        return np.hstack([base, standardize(gbdt_col)]) if gbdt_on else base
    arch = ParetoArchive(); elite = []
    def harvest(perms, deg, status):
        wid = staircase_widths(deg)
        fit = np.full(perms.shape[0], INFEAS); pen = row_penalties(deg, status, MAX_TW)
        for p in np.where(status == 0)[0]:
            front = [(int(wid[p, t]), int(t)) for t in t_grid]
            for w, t in front: arch.try_add(w, t, perms[p].tolist())
            hv = hypervolume_2d(front, n); fit[p] = -hv; elite.append((hv, perms[p].copy()))
        m = np.where(status == 1)[0]; fit[m] = INFEAS + pen[m]; fit[status == 2] = INFEAS + 2000
        return fit
    def retrain():
        if not elite: return
        elite.sort(key=lambda z: -z[0]); top = elite[:60]
        X, y, w = training_set([(h, list(p)) for h, p in top], F, n)
        _, mdl = make_gbdt("auto", seed)
        try:
            mdl.fit(X, y, sample_weight=w); gbdt_col[:, 0] = np.asarray(mdl.predict(F), float)
        except Exception: pass
        del elite[60:]
    # warm start: min-degree
    sp = np.array([min_degree_perm(n, ab, rng=random.Random(seed + s)) for s in range(3)], dtype=np.int32)
    d0, s0 = eval_fn(sp); harvest(sp, d0, s0)
    if gbdt_on: retrain()
    dim = feats().shape[1]
    opt = SepCMAES(dim, 0.5, seed); opt.lam = pop; opt.mu = pop // 2
    wts = np.log(opt.mu + 0.5) - np.log(np.arange(1, opt.mu + 1)); opt.w = wts / wts.sum()
    opt.mueff = 1 / np.sum(opt.w ** 2); opt.mean = ridge_policy(feats(), sp[0])
    t0 = time.time(); gen = 0
    while time.time() - t0 < budget:
        Phi = feats(); X = opt.ask()
        perms = np.argsort(Phi @ X.T, axis=0).T.astype(np.int32)
        deg, status = eval_fn(perms); fit = harvest(perms, deg, status); opt.tell(fit)
        gen += 1
        if gbdt_on and gen % gbdt_every == 0: retrain()
    return -arch.hypervolume(n)


def run_instance(path, k_eig, pop, budget, seed):
    name = os.path.splitext(os.path.basename(path))[0]
    n, adj = load_graph(path); ab = build_adj_bitsets(n, adj); aw, W = build_adj_words(n, ab)
    t_grid = sorted({int(round(i * (n - 1) / 19)) for i in range(20)})
    F = build_rich_features(repo_root(), "syn_" + name, n, adj, ab, k_eig)
    eval_fn, _ = make_evaluator(aw, n, W, MAX_TW, pop)
    Flin = standardize(F); Ppoly = poly_expand(F, 0, seed); Prand = poly_expand(F, 128, seed)
    rungs = {
        "linear": (Flin, False), "poly": (Ppoly, False),
        "random": (Prand, False), "gbdt": (Ppoly, True),
    }
    out = {}
    for nm, (base, on) in rungs.items():
        out[nm] = search(F, base, ab, aw, n, W, t_grid, on, pop, budget, seed, eval_fn)
    return name, n, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-eig", type=int, default=16)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--budget", type=float, default=12.0, help="seconds per rung")
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    here = repo_root()
    files = sorted(glob.glob(os.path.join(here, "extra_instances", "data", "*.gr")))[:args.instances]
    print(f"expressiveness ladder across {len(files)} instances "
          f"(pop={args.pop}, budget={args.budget}s/rung)\n")
    print(f"{'instance':<22}{'dens':>5}{'linear':>12}{'poly':>12}{'random':>12}{'gbdt':>12}{'GBDT margin':>13}{'win':>5}")
    rows = []
    for fp in files:
        name, n, o = run_instance(fp, args.k_eig, args.pop, args.budget, args.seed)
        d = re.search(r"_d(\d+)", name); dens = int(d.group(1)) if d else 0
        others = max(o["linear"], o["poly"], o["random"])
        margin = o["gbdt"] - others           # more negative gbdt = better => margin<0 means gbdt wins
        win = "Y" if o["gbdt"] < others else "."
        rows.append((dens, margin, win, o))
        print(f"{name:<22}{dens:>5}{o['linear']:>12,.0f}{o['poly']:>12,.0f}"
              f"{o['random']:>12,.0f}{o['gbdt']:>12,.0f}{margin:>+13,.0f}{win:>5}", flush=True)
    wins = sum(1 for _, _, w, _ in rows if w == "Y")
    print(f"\nlearned-GBDT decode wins on {wins}/{len(rows)} instances")
    print("by density (mean GBDT margin; negative = GBDT better):")
    for dens in sorted(set(d for d, _, _, _ in rows)):
        ms = [m for d, m, _, _ in rows if d == dens]
        wd = sum(1 for d, _, w, _ in rows if d == dens and w == "Y")
        print(f"  d{dens:<3} mean margin {sum(ms)/len(ms):>+12,.0f}   (GBDT wins {wd}/{len(ms)})")
    print("\nThe principle (§10.4) holds where GBDT wins; a consistent win, especially")
    print("strengthening with density, generalises it from one graph to the class.")


if __name__ == "__main__":
    main()

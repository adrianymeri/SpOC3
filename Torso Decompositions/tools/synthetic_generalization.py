#!/usr/bin/env python3
"""
synthetic_generalization.py -- does the GBDT-adaptive contribution generalise,
and does it track graph density?

For each of the 20 synthetic instances (4 densities x 5 sizes) it:
  1. builds a baseline portfolio from K min-degree orderings (random tie-breaks),
  2. trains the LightGBM adaptive constructor on those orderings,
  3. builds M adaptive orderings and pools them in,
  4. reports the GBDT contribution = HV(baseline + GBDT) - HV(baseline),
     using full-staircase threshold extraction throughout.

The hypothesis (THESIS.md s6b): the contribution is ~0 on sparse graphs and
grows with density. Output is grouped by density so the trend is visible.

Usage:
    python3 tools/synthetic_generalization.py            # all 20
    python3 tools/synthetic_generalization.py --max-instances 4 --budget-orders 30
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, glob, os, random, re
import numpy as np
from core import (load_graph, build_adj_bitsets, min_degree_perm, ParetoArchive,
                  hypervolume_2d, repo_root, MAX_TW)
from algorithms.continuous.gbdt_torso import (build_rich_features, _construct,
                                              _replay_rows, make_gbdt)
from algorithms.continuous.cmaes_torso import eval_fitness


def add_staircase(perm, deg_seq, n, arch):
    if deg_seq.max() > MAX_TW:
        return
    run = 0
    for t in range(n - 1, -1, -1):
        if deg_seq[t] > run:
            run = deg_seq[t]
        arch.try_add(int(run), t, list(perm))


def run_instance(path, k_eig, k_base, m_orders, backend, seed):
    name = os.path.splitext(os.path.basename(path))[0]
    n, adj = load_graph(path); ab = build_adj_bitsets(n, adj)
    here = repo_root()
    t_grid = sorted({int(round(i * (n - 1) / 19)) for i in range(20)})
    rng = np.random.default_rng(seed)
    F = build_rich_features(here, "syn_" + name, n, adj, ab, k_eig)

    # 1. baseline: K min-degree orderings (random tie-breaks)
    base = ParetoArchive(); elites = []
    for s in range(k_base):
        perm = min_degree_perm(n, ab, rng=random.Random(seed + s))
        hv = -eval_fitness(perm, ab, n, t_grid, base)
        if hv > 0:
            elites.append((hv, perm))
    if not elites:
        return name, n, None, None, 0.0
    elites.sort(key=lambda z: -z[0])
    base_score = -base.hypervolume(n)

    # 2-3. train GBDT, build adaptive orderings, pool into the SAME archive
    Xs, ys = [], []
    for _, perm in elites[:k_base]:
        Xe, ye, _ge = _replay_rows(perm, F, ab, n, 64, 4, rng); Xs.append(Xe); ys.append(ye)
    _, model = make_gbdt(backend, seed)
    model.fit(np.vstack(Xs), np.concatenate(ys))
    sf = lambda feat: model.predict(feat)
    for i in range(m_orders):
        temp = 0.0 if i == 0 else float(rng.uniform(0.2, 0.8))
        perm, ds = _construct(sf, F, ab, n, rng, 64, temp)
        add_staircase(perm, ds, n, base)
    gbdt_score = -base.hypervolume(n)
    return name, n, base_score, gbdt_score, base_score - gbdt_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-eig", type=int, default=16)
    ap.add_argument("--k-base", type=int, default=8, help="# min-degree baseline orderings")
    ap.add_argument("--budget-orders", type=int, default=40, help="# adaptive orderings to build")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-instances", type=int, default=20)
    args = ap.parse_args()
    here = repo_root()
    files = sorted(glob.glob(os.path.join(here, "extra_instances", "data", "*.gr")))[:args.max_instances]
    print(f"{'instance':<22}{'n':>6}{'density':>8}{'baseline':>14}{'+GBDT':>14}{'contribution':>14}")
    rows = []
    for fp in files:
        name, n, b, g, c = run_instance(fp, args.k_eig, args.k_base,
                                        args.budget_orders, args.backend, args.seed)
        d = re.search(r"_d(\d+)", name)
        dens = int(d.group(1)) if d else 0
        if b is None:
            print(f"{name:<22}{n:>6}{dens:>8}{'(no feasible)':>14}")
            continue
        rows.append((dens, c))
        print(f"{name:<22}{n:>6}{dens:>8}{b:>14,.0f}{g:>14,.0f}{c:>+14,.0f}")
    # group by density
    print("\nGBDT contribution by density (mean HV added):")
    for dens in sorted(set(d for d, _ in rows)):
        cs = [c for d, c in rows if d == dens]
        print(f"  d{dens:<3}  mean +{sum(cs)/len(cs):>12,.0f}  ({len(cs)} instances)")


if __name__ == "__main__":
    main()

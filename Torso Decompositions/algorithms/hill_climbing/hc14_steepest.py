#!/usr/bin/env python3
"""
hc14_steepest.py -- Stochastic best-improvement HC (steepest-ascent variant).

The 13 prior variants in this project are all *stochastic first-improvement*
hill climbers: at each step they sample one neighbour, decide, and (if
accepted) move.  The classical HC taxonomy splits along this axis:

    - First-improvement (hc1 ... hc13): one neighbour per step, accept if
      better.  Cheap per step; many small accepts; high iteration count.
    - Best-improvement (this file): sample K neighbours per step, take
      the lex-best one, accept iff strictly better than the working
      solution.  Fewer but larger accepts; lower iteration count.

A literal "all neighbours" steepest ascent is O(n^2) per step (every
pairwise swap) and intractable at n >= 1 357; we follow the standard
metaheuristics practice of *stochastic* best-improvement: at each step
draw K candidate moves uniformly from the same 15-operator set as hc5,
evaluate them all, and adopt the best.  K is a tunable parameter
(default K = 16, the typical sweet spot for permutation problems
reported by Whitley et al. 1996).

References
----------
- Selman, B., & Kautz, H. (1993).  Domain-independent extensions to
  GSAT: Solving large structured satisfiability problems.  IJCAI.
- Hoos, H. H., & Stützle, T. (2004).  Stochastic Local Search:
  Foundations and Applications, Ch. 1.5 (first- vs best-improvement).
- Whitley, D., Mathias, K., & Pyeatt, L. (1996).  Hyperplane ranking
  in simple genetic algorithms.  ICGA, p. 235--242.

Difference from hc4
-------------------
Identical operators, identical warm start, identical Pareto archive,
identical lex (max_w, soft) acceptance rule -- the *only* change is that
at each step we sample K candidates instead of 1 and pick the lex-best
before deciding whether to move.  This isolates the first- vs best-
improvement axis cleanly.
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

import argparse
import os
import random
import time
from typing import Callable, List, Tuple

import numpy as np

from core import (
    repo_root,
    ensure_seeded,
    LEADERBOARD_TARGETS,
    ParetoArchive,
    build_adj_bitsets,
    evaluate_full,
    graph_path,
    load_graph,
    build_warm_start,
    min_degree_perm,
    submission_path,
    write_submission,
)

# Reuse the exact operator set from hc4 (including the new op_3opt).
from algorithms.hill_climbing.hc5_operators import OPERATORS


def run(
    problem: str,
    budget_s: float,
    seed: int,
    progress_every: int,
    here: str,
    num_t_seeds: int = 20,
    sample_k: int = 16,
):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc14_steepest -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"warm-start t-grid = {num_t_seeds}, "
          f"operators = {len(OPERATORS)}, sample_k = {sample_k}")
    print()

    # --- Warm start (same as hc4) ---
    md_t0 = time.time()
    md_rng = random.Random(seed)
    md, ws_label = build_warm_start(n, adj_bits, rng=md_rng)
    print(f"  {ws_label} warm start built in "
          f"{time.time() - md_t0:.1f}s")

    if num_t_seeds == 1:
        t_grid = [0]
    else:
        t_grid = sorted({
            int(round(i * (n - 1) / (num_t_seeds - 1)))
            for i in range(num_t_seeds)
        })
    archive = ParetoArchive()
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(md, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), md[:])
    init_hv = archive.hypervolume(n)
    print(f"  seeded archive with {len(archive)} non-dominated points "
          f"(initial score = {-init_hv:,.0f})")

    # --- Working solution ---
    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
        cur_perm, cur_t, adj_bits, n)

    op_accepts = {name: 0 for name, _ in OPERATORS}
    op_attempts = {name: 0 for name, _ in OPERATORS}
    op_wins = {name: 0 for name, _ in OPERATORS}     # how often this op produced the best of K

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    rejected_no_improve = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        # --- Sample K candidates (with adaptive operator weights) ---
        weights = [
            (op_accepts[name] + 1) / (op_attempts[name] + 1)
            for name, _ in OPERATORS
        ]
        samples = random.choices(OPERATORS, weights=weights, k=sample_k)

        best = None  # (w, soft, perm, t, bn_idx, bn_mask, op_name)
        for op_name, op_fn in samples:
            cand_perm, cand_t = op_fn(cur_perm, n, cur_t, cur_bn_idx, cur_bn_mask)
            op_attempts[op_name] += 1
            _, cand_w, cand_bn_idx, cand_bn_mask, cand_soft = evaluate_full(
                cand_perm, cand_t, adj_bits, n)

            # Archive additions still happen for every candidate so the
            # final submission front benefits from the wider sampling
            # (this is part of why best-improvement helps -- 16 archive
            # tries per outer step vs 1 in hc4).
            added = archive.try_add(int(cand_w), int(cand_t), cand_perm)
            if added:
                archive_adds += 1
                op_accepts[op_name] += 1

            if best is None or (cand_w, cand_soft) < (best[0], best[1]):
                best = (cand_w, cand_soft, cand_perm, cand_t,
                        cand_bn_idx, cand_bn_mask, op_name)

        # --- Best-improvement acceptance: only move if strictly better ---
        b_w, b_soft, b_perm, b_t, b_bn_idx, b_bn_mask, b_op = best
        local_better = (
            b_w < cur_w
            or (b_w == cur_w and b_soft < cur_soft)
        )
        if local_better:
            cur_perm = b_perm
            cur_t = b_t
            cur_w = b_w
            cur_soft = b_soft
            cur_bn_idx = b_bn_idx
            cur_bn_mask = b_bn_mask
            op_wins[b_op] += 1
        else:
            rejected_no_improve += 1

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            hv = archive.hypervolume(n)
            print(f"  iter {iters:>7d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | wins {sum(op_wins.values()):5d} | "
                  f"focus(w={cur_w}, t={cur_t}) | "
                  f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

    # --- Report ---
    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"K={sample_k} -> {iters * sample_k:,} candidate evaluations")
    print(f"Archive adds = {archive_adds}, best-of-K accepts = "
          f"{sum(op_wins.values())}, rejected (no improvement) = "
          f"{rejected_no_improve}")
    print(f"Final archive size: {len(archive)}")

    final_hv = archive.hypervolume(n)
    final_score = -final_hv
    print(f"Hypervolume:    {final_hv:,.0f}  /  max possible (n*n) = {n*n:,}")
    print(f"Official score: {final_score:,.0f}")
    if target is not None:
        gap = final_score - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print(f"\nOperator stats (op_wins = best-of-K count, op_accepts = "
          f"archive adds, op_attempts = times sampled):")
    sorted_ops = sorted(op_wins.items(), key=lambda kv: -kv[1])
    for name, w in sorted_ops:
        acc = op_accepts[name]
        att = op_attempts[name]
        print(f"  {name:18s}  wins {w:5d}  archive_adds {acc:5d}  "
              f"attempts {att:6d}")

    print(f"\nPareto front ({len(archive)} points):")
    entries = archive.entries()
    for w, t, _ in entries[:25]:
        print(f"  width={w:3d}  t={t:5d}  (size={n - t})")
    if len(entries) > 25:
        print(f"  ... ({len(entries) - 25} more)")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc14")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    ap.add_argument("--sample-k", type=int, default=16,
                    help="number of candidate moves sampled per HC step")
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds, sample_k=args.sample_k)


if __name__ == "__main__":
    main()

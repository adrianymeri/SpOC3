#!/usr/bin/env python3
"""
hc2_simple.py — the simplest possible Hill Climbing baseline.

Algorithm
---------
- Single objective: minimise max_degree (= torso width).
- Threshold t held fixed at 0  (the entire permutation is the torso).
- One operator: pairwise swap.
- Acceptance: strict improvement (max_degree must strictly decrease).
- Random initial permutation.  No warm start, no archive, no restarts.
- Submission contains a single decision vector.

This is the honest reference point for everything else in the series.
We expect:
  - Many iterations needed because random swaps on n≈1000+ rarely hit
    the bottleneck.
  - Final width will be much higher than min-degree heuristic baseline
    (which hc3 will introduce as a comparison).
  - HV will be poor because the front contains exactly one point.
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

import numpy as np

from core import (
    repo_root,
    LEADERBOARD_TARGETS,
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
    submission_path,
    write_submission,
)


def run(
    problem: str,
    budget_s: float,
    seed: int,
    progress_every: int,
    here: str,
):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc2_simple — {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}")
    print(f"operator = swap, t fixed at 0, accept strict improvement")
    print()

    # --- Initial random permutation ---
    perm = [int(x) for x in np.random.permutation(n)]
    cur_w, _ = evaluate(perm, 0, adj_bits, n)
    best_perm = perm[:]
    best_w = cur_w
    print(f"  initial random perm: width = {cur_w}")

    deadline = time.time() + budget_s
    iters = 0
    accepts = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1
        # Single operator: random pairwise swap
        i, j = random.sample(range(n), 2)
        perm[i], perm[j] = perm[j], perm[i]
        cand_w, _ = evaluate(perm, 0, adj_bits, n)

        if cand_w < cur_w:
            # Accept
            cur_w = cand_w
            accepts += 1
            if cur_w < best_w:
                best_w = cur_w
                best_perm = perm[:]
        else:
            # Reject — undo the swap
            perm[i], perm[j] = perm[j], perm[i]

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            # Single-point HV: one point at (best_w, 0).
            hv = hypervolume_2d([(best_w, 0)], n)
            print(f"  iter {iters:>8d} | accepts {accepts:5d} | "
                  f"best width = {best_w:3d} | "
                  f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

    # --- Final report ---
    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"accepts = {accepts}, accept rate = {accepts/max(iters,1):.4%}")
    print(f"Best width found: {best_w}")

    # Single-point Pareto front: (best_w, 0)
    final_hv = hypervolume_2d([(best_w, 0)], n)
    final_score = -final_hv
    print(f"Hypervolume:    {final_hv:,.0f}  /  max possible (n*n) = {n*n:,}")
    print(f"Official score: {final_score:,.0f}")
    if target is not None:
        gap = final_score - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    # --- Submission ---
    decision_vector = best_perm + [0]
    out_path = submission_path(here, problem, "hc2")
    write_submission([decision_vector], problem, out_path)
    print(f"\nWrote submission: {out_path}  (1 vector)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=2000)
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here)


if __name__ == "__main__":
    main()

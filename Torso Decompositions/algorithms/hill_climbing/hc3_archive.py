#!/usr/bin/env python3
"""
hc3_archive.py — Hill Climbing with a Pareto archive.

Difference from hc2_simple
--------------------------
- t is now part of the decision and may be mutated.
- A `ParetoArchive` holds every non-dominated (max_degree, t) point we
  have ever generated, with its decision vector.  The archive is the
  submission (top-20 by HV contribution).
- Acceptance is two-tiered:
    1. Archive-extending: any candidate whose fitness is non-dominated
       by the archive is added.  Dominated archive members are pruned.
    2. Local exploitation: the candidate also becomes the new working
       solution if it strictly improves the current (max_degree,
       soft_cost) lex order.
- Still a SINGLE operator (random swap of two perm positions, with a
  10% chance instead to randomly resample t).  Operator diversity is
  the next step (hc3 onwards).

Goal of this file: isolate the effect of "archive + free t" against
hc2_simple (swap-only, fixed t=0, single point).  Everything else is
identical to hc1.
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
    ParetoArchive,
    build_adj_bitsets,
    evaluate_descent,
    evaluate_full,
    graph_path,
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

    print(f"\n=== hc3_archive — {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}")
    print(f"operator = swap (90%) | t-resample (10%)")
    print(f"acceptance = Pareto archive + lex (cap_violations, max_step_d, soft)")
    print()

    # --- Initial state: random permutation, random t ---
    perm = [int(x) for x in np.random.permutation(n)]
    t_cur = random.randint(0, n - 1)
    _, cur_w, _, _, _ = evaluate_full(perm, t_cur, adj_bits, n)
    cur_max_d, cur_caps, cur_soft = evaluate_descent(perm, t_cur, adj_bits, n)

    archive = ParetoArchive()
    archive.try_add(int(cur_w), int(t_cur), perm[:])
    print(f"  initial: width = {cur_w}, t = {t_cur}, "
          f"caps = {cur_caps}, max_step_d = {cur_max_d}, "
          f"archive size = {len(archive)}")

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        # Operator: 90% pairwise swap, 10% random new t.
        if random.random() < 0.9:
            i, j = random.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]
            new_t = t_cur
            mut_was_swap = True
        else:
            new_t = random.randint(0, n - 1)
            mut_was_swap = False

        _, cand_w, _, _, _ = evaluate_full(perm, new_t, adj_bits, n)
        cand_max_d, cand_caps, cand_soft = evaluate_descent(
            perm, new_t, adj_bits, n)

        added = archive.try_add(int(cand_w), int(new_t), perm[:])
        if added:
            archive_adds += 1

        # Lex acceptance on the descent metric — works even when capped.
        local_better = (
            (cand_caps, cand_max_d, cand_soft)
            < (cur_caps, cur_max_d, cur_soft)
        )

        if added or local_better:
            # Accept: keep the mutated state.
            cur_w = cand_w
            cur_max_d = cand_max_d
            cur_caps = cand_caps
            cur_soft = cand_soft
            t_cur = new_t
        else:
            # Reject: undo.
            if mut_was_swap:
                perm[i], perm[j] = perm[j], perm[i]
            # If t-resample was rejected, we just don't update t_cur.

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            hv = archive.hypervolume(n)
            entries = archive.entries()
            min_w = entries[0][0] if entries else None
            max_t = entries[-1][1] if entries else None
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cur_w}, t={t_cur}, caps={cur_caps}) | "
                  f"min_w={min_w} max_t={max_t} | "
                  f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

    # --- Final report ---
    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}")
    print(f"Final archive size: {len(archive)}")

    final_hv = archive.hypervolume(n)
    final_score = -final_hv
    print(f"Hypervolume:    {final_hv:,.0f}  /  max possible (n*n) = {n*n:,}")
    print(f"Official score: {final_score:,.0f}")
    if target is not None:
        gap = final_score - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    # Pareto front summary
    print(f"\nPareto front ({len(archive)} points):")
    for w, t, _ in archive.entries():
        print(f"  width={w:3d}  t={t:5d}  (size={n - t})")

    # --- Submission: top 20 by HV contribution ---
    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc3")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


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

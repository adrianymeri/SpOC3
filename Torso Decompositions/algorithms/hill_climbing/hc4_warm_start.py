#!/usr/bin/env python3
"""
hc4_warm_start.py — Pareto archive HC + min-degree warm start.

Difference from hc3_archive
---------------------------
Identical algorithm and operator set (single swap, 10% t-resample).  The
ONLY change is the initial archive: instead of a single random
(perm, t) pair, we seed the archive with the min-degree elimination
order replicated across a 20-point t-grid.

Expected effect: best width drops from ~50 (random init) to whatever
min-degree gives on this graph (≈20 for small-graph).  This isolates
the impact of "warm start" alone.
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
    ensure_seeded,
    evaluate_full,
    graph_path,
    load_graph,
    build_warm_start,
    min_degree_perm,
    submission_path,
    write_submission,
)


def run(
    problem: str,
    budget_s: float,
    seed: int,
    progress_every: int,
    here: str,
    num_t_seeds: int = 20,
):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc4_warm_start — {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"warm-start t-grid = {num_t_seeds}")
    print(f"operator = swap (90%) | t-resample (10%)")
    print(f"acceptance = Pareto archive + lex (max_degree, soft_cost)")
    print()

    # --- Build min-degree warm start, seed the archive ---
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

    # --- Working solution: lowest-width entry in the archive ---
    cur_w, t_cur, _seed_perm = ensure_seeded(archive, md, adj_bits, n)
    perm = list(_seed_perm)
    _, _, _, _, cur_soft = evaluate_full(perm, t_cur, adj_bits, n)
    print(f"  starting focus: width = {cur_w}, t = {t_cur}")
    print()

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        if random.random() < 0.9:
            i, j = random.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]
            new_t = t_cur
            mut_was_swap = True
        else:
            new_t = random.randint(0, n - 1)
            mut_was_swap = False

        _, cand_w, _, _, cand_soft = evaluate_full(perm, new_t, adj_bits, n)

        added = archive.try_add(int(cand_w), int(new_t), perm[:])
        if added:
            archive_adds += 1

        local_better = (
            cand_w < cur_w
            or (cand_w == cur_w and cand_soft < cur_soft)
        )

        if added or local_better:
            cur_w = cand_w
            cur_soft = cand_soft
            t_cur = new_t
        else:
            if mut_was_swap:
                perm[i], perm[j] = perm[j], perm[i]

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            hv = archive.hypervolume(n)
            entries = archive.entries()
            min_w = entries[0][0] if entries else None
            max_t = entries[-1][1] if entries else None
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cur_w}, t={t_cur}) | "
                  f"min_w={min_w} max_t={max_t} | "
                  f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

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

    # Pareto front summary (truncated for large fronts)
    entries = archive.entries()
    print(f"\nPareto front ({len(entries)} points):")
    for w, t, _ in entries[:25]:
        print(f"  width={w:3d}  t={t:5d}  (size={n - t})")
    if len(entries) > 25:
        print(f"  ... ({len(entries) - 25} more)")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc4")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds)


if __name__ == "__main__":
    main()

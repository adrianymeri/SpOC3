#!/usr/bin/env python3
"""
hc6_gap_fill.py — Pareto archive HC + per-t local-search gap filling.

Difference from hc5_operators
-----------------------------
hc4 ran free-t archive HC on the min-degree warm start.  Its Pareto
front has a giant horizontal gap: on small-graph everything between
t=0 and t≈1294 is empty, because random mutations from min-degree
can't drop below width=20 and no operator deliberately searches for
intermediate-t solutions with reduced width.

hc5 fixes that with a TWO-PHASE algorithm:

  Phase 1: Warm-start sweep.  Build min-degree perm; seed the archive
           by evaluating it at every t in a fine grid (default 40
           values evenly spaced over [0, n-1]).

  Phase 2: Round-robin per-t local search.  Repeatedly sweep through
           the t-grid.  For each t:
             a. Pick the archive entry whose t is nearest as the
                starting permutation (so we begin where the front is
                already strong).
             b. Run K iterations of permutation-only HC at fixed t,
                using the full operator set minus the t-mutators.
             c. Every accepted candidate is added to the archive.

The total time budget is split: ~10% for warm-start, the rest equally
divided across t-grid points across as many sweeps as fit.

Hypothesis: actively searching at intermediate t values populates the
front gap, lifting HV closer to -1,829,919.
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


# ---------------------------------------------------------------------------
# Permutation-only operators (no t mutation in fixed-t local search).
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]
    return out


def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def op_insert(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return out
    v = out.pop(i)
    out.insert(j, v)
    return out


def op_reverse(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = sorted(random.sample(range(n), 2))
    out[i:j + 1] = reversed(out[i:j + 1])
    return out


def op_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    block_size = random.randint(2, max(3, n // 20))
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out


def op_bottleneck_to_head(perm, n, t, bn_idx, bn_mask):
    if bn_idx < 0 or t <= 0:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    if random.random() < 0.7 or bn_mask == 0:
        target_v = out[bn_idx]
        del out[bn_idx]
    else:
        bits = []
        m = bn_mask
        while m:
            b = m & -m
            bits.append(b.bit_length() - 1)
            m ^= b
        v = random.choice(bits)
        idx = out.index(v)
        target_v = v
        del out[idx]
    new_pos = random.randint(0, max(0, t - 1))
    out.insert(new_pos, target_v)
    return out


def op_swap_head_tail(perm, n, t, bn_idx, bn_mask):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randrange(0, t)
    j = random.randrange(t, n)
    out[i], out[j] = out[j], out[i]
    return out


def op_2opt_torso(perm, n, t, bn_idx, bn_mask):
    if t >= n - 2:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(t, n - 2)
    j = random.randint(i + 1, n - 1)
    out[i:j + 1] = reversed(out[i:j + 1])
    return out


PERM_OPS: List[Tuple[str, Callable]] = [
    ("swap",              op_swap),
    ("adjacent_swap",     op_adjacent_swap),
    ("insert",            op_insert),
    ("reverse",           op_reverse),
    ("block_move",        op_block_move),
    ("bottleneck->head",  op_bottleneck_to_head),
    ("swap_head_tail",    op_swap_head_tail),
    ("2opt_torso",        op_2opt_torso),
]


# ---------------------------------------------------------------------------
# Per-t local search
# ---------------------------------------------------------------------------

def local_search_at_t(
    seed_perm: List[int],
    t: int,
    iters_budget: int,
    n: int,
    adj_bits: List[int],
    archive: ParetoArchive,
    op_accepts: dict,
    op_attempts: dict,
) -> int:
    """Run HC at fixed t for at most `iters_budget` iterations.  Adds
    every accepted candidate to the archive.  Returns the number of
    archive additions.

    Acceptance: lex (max_degree, soft_cost) — we move the working perm
    when it strictly improves on the current.  Archive is independently
    updated for any non-dominated candidate (Pareto-extension).
    """
    perm = seed_perm[:]
    _, cur_w, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
        perm, t, adj_bits, n)
    adds = 0
    for _ in range(iters_budget):
        weights = [
            (op_accepts[name] + 1) / (op_attempts[name] + 1)
            for name, _ in PERM_OPS
        ]
        op_name, op_fn = random.choices(PERM_OPS, weights=weights, k=1)[0]
        cand = op_fn(perm, n, t, cur_bn_idx, cur_bn_mask)
        op_attempts[op_name] += 1

        _, cand_w, cand_bn_idx, cand_bn_mask, cand_soft = evaluate_full(
            cand, t, adj_bits, n)

        if archive.try_add(int(cand_w), int(t), cand):
            adds += 1
            op_accepts[op_name] += 1

        if (cand_w < cur_w) or (cand_w == cur_w and cand_soft < cur_soft):
            perm = cand
            cur_w, cur_soft = cand_w, cand_soft
            cur_bn_idx, cur_bn_mask = cand_bn_idx, cand_bn_mask
    return adds


def nearest_seed(archive: ParetoArchive, target_t: int) -> List[int]:
    """Return the perm of the archive entry whose t is closest to target_t.
    Ties broken by lower width (better starting point)."""
    entries = archive.entries()
    best = min(entries, key=lambda e: (abs(e[1] - target_t), e[0]))
    return list(best[2])


def run(
    problem: str,
    budget_s: float,
    seed: int,
    progress_every: int,
    here: str,
    num_t_seeds: int = 40,
    sweep_iter_budget: int = 200,
):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc6_gap_fill — {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"t-grid = {num_t_seeds}, ls iters/visit = {sweep_iter_budget}")
    print()

    # --- Warm start (with randomised tiebreaks so seed actually varies it) ---
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
    # Guard against an all-over-width warm start (dense instances): keep
    # the archive non-empty so nearest_seed() never indexes an empty list.
    ensure_seeded(archive, md, adj_bits, n)
    init_hv = archive.hypervolume(n)
    print(f"  seeded archive with {len(archive)} non-dominated points "
          f"(initial score = {-init_hv:,.0f})")
    print()

    # --- Phase 2: round-robin per-t local search ---
    op_accepts = {name: 0 for name, _ in PERM_OPS}
    op_attempts = {name: 0 for name, _ in PERM_OPS}

    deadline = time.time() + budget_s
    sweep = 0
    total_adds = 0
    last_print = 0

    while time.time() < deadline:
        sweep += 1
        # Visit t-grid in random order each sweep.
        order = t_grid[:]
        random.shuffle(order)
        for tt in order:
            if time.time() >= deadline:
                break
            seed_perm = nearest_seed(archive, tt)
            adds = local_search_at_t(
                seed_perm, tt, sweep_iter_budget,
                n, adj_bits, archive,
                op_accepts, op_attempts,
            )
            total_adds += adds

            if total_adds - last_print >= progress_every:
                hv = archive.hypervolume(n)
                entries = archive.entries()
                min_w = entries[0][0] if entries else None
                max_t = entries[-1][1] if entries else None
                elapsed = budget_s - (deadline - time.time())
                print(f"  sweep {sweep:2d} | adds {total_adds:5d} | "
                      f"archive {len(archive):3d} | "
                      f"min_w={min_w} max_t={max_t} | "
                      f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
                last_print = total_adds

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, sweeps = {sweep}, "
          f"archive adds = {total_adds}")
    print(f"Final archive size: {len(archive)}")

    final_hv = archive.hypervolume(n)
    final_score = -final_hv
    print(f"Hypervolume:    {final_hv:,.0f}  /  max possible (n*n) = {n*n:,}")
    print(f"Official score: {final_score:,.0f}")
    if target is not None:
        gap = final_score - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print(f"\nOperator stats (accept rate, attempts):")
    sorted_ops = sorted(op_accepts.items(), key=lambda kv: -kv[1])
    for name, acc in sorted_ops:
        att = op_attempts[name]
        rate = acc / att if att else 0
        print(f"  {name:18s}  {acc:5d} / {att:6d}   ({rate:6.2%})")

    print(f"\nPareto front ({len(archive)} points):")
    entries = archive.entries()
    for w, t, _ in entries[:25]:
        print(f"  width={w:3d}  t={t:5d}  (size={n - t})")
    if len(entries) > 25:
        print(f"  ... ({len(entries) - 25} more)")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc6")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--num-t-seeds", type=int, default=40)
    ap.add_argument("--sweep-iter-budget", type=int, default=200)
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds,
        sweep_iter_budget=args.sweep_iter_budget)


if __name__ == "__main__":
    main()

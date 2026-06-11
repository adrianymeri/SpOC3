#!/usr/bin/env python3
"""
hc7_kbottleneck.py — Pareto archive HC + K-vertex bottleneck-relocate operator.

Difference from hc5_operators
-----------------------------
hc4 uses a single-vertex bottleneck-to-head operator that relocates one
bottleneck vertex per move.  But on small-graph (and many real
instances) min-degree leaves *multiple* vertices simultaneously at
max_width — so moving one immediately exposes the next as the new
bottleneck and the global max_width does not drop.

hc6 adds a NEW operator: `k_bottleneck_to_head`.  In a single move:
  1. Find every torso position whose vertex achieves the current
     max_width ("ties at the top").
  2. Pick K of them at random (K = 2..len(ties)).
  3. Relocate all K vertices into random positions of the head [0, t).

This is still pure hill climbing (deterministic accept/reject), but a
single move now changes K positions.  Hypothesis: this is enough to
escape the width=20 plateau on small-graph.

The rest of the algorithm (warm start, archive, adaptive operator
selection) is identical to hc4.
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
    MAX_TW,
    ParetoArchive,
    build_adj_bitsets,
    evaluate_full,
    evaluate_with_bottlenecks,
    graph_path,
    load_graph,
    build_warm_start,
    min_degree_perm,
    submission_path,
    write_submission,
)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bottlenecks):
    out = perm[:]
    i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]
    return out, t


def op_adjacent_swap(perm, n, t, bottlenecks):
    out = perm[:]
    i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, t


def op_insert(perm, n, t, bottlenecks):
    out = perm[:]
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return out, t
    v = out.pop(i)
    out.insert(j, v)
    return out, t


def op_reverse(perm, n, t, bottlenecks):
    out = perm[:]
    i, j = sorted(random.sample(range(n), 2))
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t


def op_block_move(perm, n, t, bottlenecks):
    out = perm[:]
    block_size = random.randint(2, max(3, n // 20))
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out, t


def op_long_block_move(perm, n, t, bottlenecks):
    out = perm[:]
    block_size = random.randint(max(3, n // 50), max(4, n // 5))
    block_size = min(block_size, n - 1)
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out, t


def op_bottleneck_to_head(perm, n, t, bottlenecks):
    if not bottlenecks or t <= 0:
        return op_swap(perm, n, t, bottlenecks)
    out = perm[:]
    bn_idx, bn_mask = random.choice(bottlenecks)
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
    return out, t


def op_k_bottleneck_to_head(perm, n, t, bottlenecks):
    """NEW in hc6: relocate K bottleneck vertices simultaneously.

    K is chosen uniformly from {2, ..., len(bottlenecks)} when there are
    multiple ties at the top.  If only one bottleneck exists this falls
    back to the single-vertex variant.  The K chosen vertices are pulled
    out of their current torso positions and inserted at random head
    positions.  In one move we can drop max_width if and only if all K
    are removed and no other torso vertex was at the same max width.
    """
    if not bottlenecks or t <= 0:
        return op_swap(perm, n, t, bottlenecks)
    if len(bottlenecks) < 2:
        return op_bottleneck_to_head(perm, n, t, bottlenecks)
    out = perm[:]
    k_max = len(bottlenecks)
    k = random.randint(2, k_max)
    chosen = random.sample(bottlenecks, k)
    # Sort by position descending so deletions don't invalidate later indices.
    chosen.sort(key=lambda b: -b[0])
    targets: List[int] = []
    for bn_idx, _ in chosen:
        targets.append(out[bn_idx])
        del out[bn_idx]
    # Insert each target at a random head position.  We use the *original*
    # threshold t as the head boundary (after deletions the indices have
    # shifted, but we still want the targets in the head region).
    for v in targets:
        upper = max(0, t - 1)
        new_pos = random.randint(0, upper)
        out.insert(new_pos, v)
    return out, t


def op_swap_head_tail(perm, n, t, bottlenecks):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bottlenecks)
    out = perm[:]
    i = random.randrange(0, t)
    j = random.randrange(t, n)
    out[i], out[j] = out[j], out[i]
    return out, t


def op_2opt_torso(perm, n, t, bottlenecks):
    if t >= n - 2:
        return op_reverse(perm, n, t, bottlenecks)
    out = perm[:]
    i = random.randint(t, n - 2)
    j = random.randint(i + 1, n - 1)
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t


def op_t_shift(perm, n, t, bottlenecks):
    span = max(1, n // 20)
    new_t = max(0, min(n - 1, t + random.randint(-span, span)))
    return perm[:], new_t


def op_t_random(perm, n, t, bottlenecks):
    return perm[:], random.randint(0, n - 1)


OPERATORS: List[Tuple[str, Callable]] = [
    ("swap",                  op_swap),
    ("adjacent_swap",         op_adjacent_swap),
    ("insert",                op_insert),
    ("reverse",               op_reverse),
    ("block_move",            op_block_move),
    ("long_block_move",       op_long_block_move),
    ("bottleneck->head",      op_bottleneck_to_head),
    ("k_bottleneck->head",    op_k_bottleneck_to_head),   # NEW
    ("swap_head_tail",        op_swap_head_tail),
    ("2opt_torso",            op_2opt_torso),
    ("t_shift",               op_t_shift),
    ("t_random",              op_t_random),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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

    print(f"\n=== hc7_kbottleneck — {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"warm-start t-grid = {num_t_seeds}, "
          f"operators = {len(OPERATORS)} (k_bottleneck->head NEW)")
    print()

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

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    cur_w, cur_bn_list, cur_soft = evaluate_with_bottlenecks(
        cur_perm, cur_t, adj_bits, n)

    op_accepts = {name: 0 for name, _ in OPERATORS}
    op_attempts = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        weights = [
            (op_accepts[name] + 1) / (op_attempts[name] + 1)
            for name, _ in OPERATORS
        ]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]

        cand_perm, cand_t = op_fn(cur_perm, n, cur_t, cur_bn_list)
        op_attempts[op_name] += 1

        cand_w, cand_bn_list, cand_soft = evaluate_with_bottlenecks(
            cand_perm, cand_t, adj_bits, n)

        added = archive.try_add(int(cand_w), int(cand_t), cand_perm)
        if added:
            archive_adds += 1
            op_accepts[op_name] += 1

        local_better = (
            cand_w < cur_w
            or (cand_w == cur_w and cand_soft < cur_soft)
        )

        if added or local_better:
            cur_perm = cand_perm
            cur_t = cand_t
            cur_w = cand_w
            cur_soft = cand_soft
            cur_bn_list = cand_bn_list

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            hv = archive.hypervolume(n)
            entries = archive.entries()
            min_w = entries[0][0] if entries else None
            max_t = entries[-1][1] if entries else None
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cur_w}, t={cur_t}, |bn|={len(cur_bn_list)}) | "
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

    print(f"\nOperator stats (accept rate, attempts):")
    sorted_ops = sorted(op_accepts.items(), key=lambda kv: -kv[1])
    for name, acc in sorted_ops:
        att = op_attempts[name]
        rate = acc / att if att else 0
        print(f"  {name:22s}  {acc:5d} / {att:6d}   ({rate:6.2%})")

    print(f"\nPareto front ({len(archive)} points):")
    entries = archive.entries()
    for w, t, _ in entries[:25]:
        print(f"  width={w:3d}  t={t:5d}  (size={n - t})")
    if len(entries) > 25:
        print(f"  ... ({len(entries) - 25} more)")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc7")
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

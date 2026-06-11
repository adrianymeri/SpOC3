#!/usr/bin/env python3
"""
hc12_incremental.py -- HC with an incremental bitset evaluator.

This is the same algorithmic skeleton as hc4 (Pareto archive +
min-degree warm start + adaptive operator selection) with ONE
change: every operator declares the leftmost position it modified,
and the evaluator reuses the cached chordal-completion state for
positions [0, leftmost) instead of rebuilding it from scratch.

For t-only moves (operator changes the threshold but not the perm),
the fitness is recomputed in O(n) from the cached per-step degrees
-- no chordal-completion work at all.

The point of this file is to isolate the speed-up the incremental
evaluator buys us:

    hc4  vs.  hc12      same algorithm, two evaluators.

We expect hc12 to do markedly more iterations per second on every
problem, and the gap widens on dense graphs where each chordal walk
is expensive.

Operators (subset of hc4's set, each reports leftmost_changed):

    swap                    swap two random positions          (leftmost = min(i,j))
    adjacent_swap           swap perm[i] and perm[i+1]         (leftmost = i)
    insert                  pop perm[i], insert at j           (leftmost = min(i,j))
    reverse                 reverse a random segment           (leftmost = i)
    3opt                    double-reverse perm[i:j], perm[j:k] (leftmost = i)
    block_move              cut block, paste elsewhere         (leftmost = min(start, dst))
    t_shift                 t += small step                    (perm unchanged)
    t_random                resample t uniformly               (perm unchanged)

We deliberately drop hc4's `bottleneck->head` operator from this
file: it needs the bottleneck-vertex position which costs an extra
full walk to compute, and would re-introduce the very cost we are
trying to eliminate.  Recovering that operator in an incremental
form is a small extension (track bn_idx during the partial walk);
left as future work.

Acceptance: archive-extending for the Pareto archive, AND the
working solution moves only when max_degree strictly decreases.
Decoupling the two is what lets the incremental cache stay valid
across many archive additions without repeated full re-walks.
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
    IncrementalEvaluator,
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


# ---------------------------------------------------------------------------
# Operators -- each returns (new_perm, new_t, leftmost_changed).
# leftmost_changed = None means "perm is unchanged, only t may have moved".
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    new_perm = list(perm)
    i, j = sorted(random.sample(range(n), 2))
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm, t, i


def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    new_perm = list(perm)
    i = random.randrange(n - 1)
    new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
    return new_perm, t, i


def op_insert(perm, n, t, bn_idx, bn_mask):
    new_perm = list(perm)
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return new_perm, t, None     # no change
    v = new_perm.pop(i)
    new_perm.insert(j, v)
    return new_perm, t, min(i, j)


def op_reverse(perm, n, t, bn_idx, bn_mask):
    new_perm = list(perm)
    i, j = sorted(random.sample(range(n), 2))
    new_perm[i:j + 1] = reversed(new_perm[i:j + 1])
    return new_perm, t, i


def op_3opt(perm, n, t, bn_idx, bn_mask):
    """3-opt double-reverse: pick i < j < k and simultaneously reverse
    perm[i:j] AND perm[j:k].  Everything before position i is unchanged,
    so leftmost_changed = i and the incremental cache for steps [0, i)
    stays valid -- the same incremental contract as op_reverse."""
    if n < 6:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    new_perm = list(perm)
    i = random.randint(0, n - 5)
    j = random.randint(i + 2, n - 3)
    k = random.randint(j + 2, n - 1)
    new_perm[i:j] = new_perm[i:j][::-1]
    new_perm[j:k] = new_perm[j:k][::-1]
    return new_perm, t, i


def op_block_move(perm, n, t, bn_idx, bn_mask):
    new_perm = list(perm)
    block_size = random.randint(2, max(3, n // 20))
    start = random.randint(0, n - block_size)
    block = new_perm[start:start + block_size]
    del new_perm[start:start + block_size]
    insert_pos = random.randint(0, len(new_perm))
    new_perm[insert_pos:insert_pos] = block
    return new_perm, t, min(start, insert_pos)


def op_t_shift(perm, n, t):
    span = max(1, n // 20)
    new_t = max(0, min(n - 1, t + random.randint(-span, span)))
    return list(perm), new_t, None       # perm untouched


def op_t_random(perm, n, t):
    return list(perm), random.randint(0, n - 1), None     # perm untouched


# Simpler signatures than hc4 -- no bn_idx/bn_mask threading.
OPERATORS: List[Tuple[str, Callable]] = [
    ("swap",          lambda p, n, t: op_swap(p, n, t, -1, 0)),
    ("adjacent_swap", lambda p, n, t: op_adjacent_swap(p, n, t, -1, 0)),
    ("insert",        lambda p, n, t: op_insert(p, n, t, -1, 0)),
    ("reverse",       lambda p, n, t: op_reverse(p, n, t, -1, 0)),
    ("3opt",          lambda p, n, t: op_3opt(p, n, t, -1, 0)),
    ("block_move",    lambda p, n, t: op_block_move(p, n, t, -1, 0)),
    ("t_shift",       op_t_shift),
    ("t_random",      op_t_random),
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

    print(f"\n=== hc12_incremental -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"warm-start t-grid = {num_t_seeds}, "
          f"operators = {len(OPERATORS)}")
    print(f"evaluator = IncrementalEvaluator "
          f"(stride = max(1, n // 64) = {max(1, n // 64)})")
    print()

    # --- Warm start ---
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

    # Initialise the incremental evaluator from the current working soln.
    ev = IncrementalEvaluator(adj_bits, n)
    ev.full_eval(cur_perm, cur_t)

    op_accepts = {name: 0 for name, _ in OPERATORS}
    op_attempts = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        # Adaptive selection: weight ~ recent accept rate with +1 priors.
        weights = [
            (op_accepts[name] + 1) / (op_attempts[name] + 1)
            for name, _ in OPERATORS
        ]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]

        cand_perm, cand_t, leftmost = op_fn(cur_perm, n, cur_t)
        op_attempts[op_name] += 1

        # ---- INCREMENTAL EVALUATION ----
        # Re-walks only steps [leftmost, n) when leftmost is given, or
        # does an O(n) max-scan over cached step degrees when perm is
        # unchanged (leftmost = None).
        cand_w, _ = ev.eval_after_change(
            cand_perm, cand_t, leftmost_changed=leftmost)

        # archive uses (w, t, perm), so feed it the candidate directly
        added = archive.try_add(int(cand_w), int(cand_t), cand_perm)
        if added:
            archive_adds += 1
            op_accepts[op_name] += 1

        # Working solution moves only when max_degree STRICTLY drops.
        # This decouples the working solution from the archive: the
        # archive may add many points without forcing a cache rebuild.
        # The cache stays valid until cur_perm actually changes.
        if cand_w < cur_w:
            cur_perm = cand_perm
            cur_t = cand_t
            cur_w = cand_w
            # commit the new working solution to the incremental cache
            ev.accept(cur_perm, cur_t)
        elif leftmost is None and cand_w == cur_w and cand_t < cur_t:
            # t-only improvement with no perm change is "free" -- cache
            # already valid, just update cur_t.
            cur_t = cand_t

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            hv = archive.hypervolume(n)
            entries = archive.entries()
            min_w = entries[0][0] if entries else None
            max_t = entries[-1][1] if entries else None
            print(f"  iter {iters:>8,d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cur_w}, t={cur_t}) | "
                  f"min_w={min_w} max_t={max_t} | "
                  f"score = {-hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}")
    print(f"Throughput: {iters / max(elapsed_total, 0.001):,.0f} iters/sec")
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
    out_path = submission_path(here, problem, "hc12")
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

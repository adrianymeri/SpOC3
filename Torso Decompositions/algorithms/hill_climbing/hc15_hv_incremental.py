#!/usr/bin/env python3
"""
hc15_hv_incremental.py -- HV-improvement acceptance on top of the
incremental bitset evaluator.

Motivation
----------
hc9 (HV-improvement acceptance, SMS-EMOA / IBEA style) is the best
variant on the sparse official instances; hc12 (incremental bitset
evaluator) is the best on the denser / larger synthetic cells, where
moves-per-second binds.  hc15 asks the obvious crossbreed question:

    Can we keep hc9's HV-aligned acceptance rule AND pay hc12's
    cheaper per-move evaluation cost at the same time?

The two designs pull in opposite directions on one axis:

  * hc12 deliberately DECOUPLES the working solution from the archive
    -- the working solution only moves when `max_width` strictly drops
    -- precisely so the incremental cache stays valid across many
    archive additions and rarely has to rebuild.

  * hc9 MOVES the working solution whenever the archive HV would
    improve, even to a lex-worse focus.  Under an incremental cache
    that means more frequent commits (cache rebuilds from
    `leftmost_changed` onward).

hc15 takes hc9's acceptance rule and hc12's evaluator and measures the
net effect.  The candidate width comes from the incremental evaluator
(cheap); the acceptance test is the marginal-HV test (hc9).  When a
candidate is accepted the working solution moves and the incremental
cache is committed via `ev.accept(...)`.

This file is ADDITIVE.  It introduces no new operators and does not
touch core.py, so it cannot perturb the locked canonical scoreboard
of hc1..hc14.  It writes its submission under the label "hc15".

Operators: the same 6-operator incremental subset as hc12 (swap,
adjacent_swap, insert, reverse, 3opt, block_move) plus the two t-moves
(t_shift, t_random), each reporting `leftmost_changed` so the cache
can be reused for the untouched prefix.

Acceptance: HV-improvement (SMS-EMOA / IBEA).  References:
Beume, Naujoks & Emmerich (2007), EJOR 181(3):1653-1669;
Zitzler & Künzli (2004), PPSN VIII, LNCS 3242:832-842.
"""

from __future__ import annotations

# --- sys.path bootstrap ---
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
    build_warm_start,
    evaluate_full,
    graph_path,
    hypervolume_2d,
    load_graph,
    submission_path,
    write_submission,
)


# ---------------------------------------------------------------------------
# Operators -- each returns (new_perm, new_t, leftmost_changed).
# leftmost_changed = None means "perm is unchanged, only t may have moved".
# These are the exact incremental-contract operators used by hc12.
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
    """3-opt double-reverse: pick i < j < k and reverse perm[i:j] AND
    perm[j:k].  Everything before position i is unchanged, so
    leftmost_changed = i -- same incremental contract as op_reverse."""
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


# Same simple signatures as hc12 -- no bn_idx/bn_mask threading.
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
# Marginal HV helper (identical to hc9)
# ---------------------------------------------------------------------------

def hv_with_candidate(archive_points, cand, n):
    """HV of the union of `archive_points` and `cand`.  Linear time."""
    pts = list(archive_points) + [cand]
    return hypervolume_2d(pts, n)


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
    warm_start: str = "auto",
):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc15_hv_incremental -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    print(f"reference point = ({n}, {n})")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"warm-start t-grid = {num_t_seeds}, operators = {len(OPERATORS)}")
    print(f"evaluator = IncrementalEvaluator "
          f"(stride = max(1, n // 64) = {max(1, n // 64)})")
    print("acceptance = HV-improvement (SMS-EMOA / IBEA)")
    print()

    # --- Warm start ---
    md_t0 = time.time()
    md, ws_label = build_warm_start(n, adj_bits, rng=random.Random(seed),
                                    method=warm_start)
    print(f"  {ws_label} warm start built in {time.time() - md_t0:.1f}s")

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

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    cur_hv = archive.hypervolume(n)
    print(f"  seeded archive with {len(archive)} non-dominated points "
          f"(initial score = {-cur_hv:,.0f})")

    # Initialise the incremental evaluator from the working solution.
    ev = IncrementalEvaluator(adj_bits, n)
    ev.full_eval(cur_perm, cur_t)

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1

        weights = [
            (op_acc[name] + 1) / (op_att[name] + 1)
            for name, _ in OPERATORS
        ]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]

        cand_perm, cand_t, leftmost = op_fn(cur_perm, n, cur_t)
        op_att[op_name] += 1

        # ---- INCREMENTAL EVALUATION (hc12) ----
        cand_w, _ = ev.eval_after_change(
            cand_perm, cand_t, leftmost_changed=leftmost)

        # ---- HV-IMPROVEMENT ACCEPTANCE (hc9) ----
        # Accept iff adding (cand_w, cand_t) strictly increases archive HV.
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            if archive.try_add(int(cand_w), int(cand_t), cand_perm):
                archive_adds += 1
            op_acc[op_name] += 1
            # Move the working solution and COMMIT the incremental cache so
            # subsequent eval_after_change calls reuse the new prefix.
            cur_perm = cand_perm
            cur_t = cand_t
            cur_w = cand_w
            ev.accept(cur_perm, cur_t)
            cur_hv = archive.hypervolume(n)   # authoritative recompute

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            entries = archive.entries()
            min_w = entries[0][0] if entries else None
            max_t = entries[-1][1] if entries else None
            print(f"  iter {iters:>8,d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cur_w}, t={cur_t}) | "
                  f"min_w={min_w} max_t={max_t} | "
                  f"score = {-cur_hv:>14,.0f} | t = {elapsed:5.1f}s")
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
    sorted_ops = sorted(op_acc.items(), key=lambda kv: -kv[1])
    for name, acc in sorted_ops:
        att = op_att[name]
        rate = acc / att if att else 0
        print(f"  {name:18s}  {acc:5d} / {att:6d}   ({rate:6.2%})")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc15")
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
    ap.add_argument("--warm-start", default="auto",
                    choices=["auto", "min-fill", "min-degree"])
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds, warm_start=args.warm_start)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
hc13_tabu.py -- Tabu Search.

Reference
---------
Glover, F. (1989).  "Tabu Search -- Part I."  ORSA Journal on
Computing 1(3): 190-206.
Glover, F. (1990).  "Tabu Search -- Part II."  ORSA Journal on
Computing 2(1): 4-32.

Difference from hc4
-------------------
Adds short-term memory to plain HC.  After each accepted move, the
move's "signature" (an attribute of the move, e.g. the pair of
vertices swapped) is appended to a tabu list of length L = sqrt(n).
Candidates whose move signature is currently in the tabu list are
REJECTED before evaluation, even if they would otherwise improve.

This breaks the cycling that plain HC tends to fall into on a
plateau: once the search swaps (v1, v2) and stalls, the reverse
swap (v2, v1) is forbidden for the next L iterations, forcing the
search to explore elsewhere.

Aspiration criterion: a tabu move is OVERRIDDEN (accepted anyway)
if its candidate would extend the Pareto archive -- i.e. if the
candidate is non-dominated and not already in the archive.  This
follows the standard Glover prescription: never miss a new best
just because the move is tabu.

Move signatures used here (one per operator):
    swap, adjacent_swap, swap_head_tail   -> frozenset({v1, v2})
    insert, block_move, long_block_move    -> frozenset({moved vertex, target position vertex})
    reverse, 2opt_torso                    -> frozenset({left endpoint vertex, right endpoint vertex})
    bottleneck->head                       -> ('bn', moved vertex)
    t_shift, t_random                      -> ('t', new_t)

The signature is intentionally COARSE (we use frozensets so the
direction of the swap is not distinguished).  This is the standard
attributive tabu representation; finer signatures would over-fit and
let the search re-discover the same plateau.
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
from collections import deque
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


# ---------------------------------------------------------------------------
# Operators -- return (new_perm, new_t, signature)
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]
    return out, t, frozenset({perm[i], perm[j]})

def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, t, frozenset({perm[i], perm[i + 1]})

def op_insert(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return out, t, frozenset({perm[i]})
    moved = perm[i]; target = perm[j]
    out.insert(j, out.pop(i))
    return out, t, frozenset({moved, target})

def op_reverse(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = sorted(random.sample(range(n), 2))
    sig = frozenset({perm[i], perm[j]})
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t, sig

def op_3opt(perm, n, t, bn_idx, bn_mask):
    """3-opt double-reverse: pick i < j < k and simultaneously reverse
    perm[i:j] AND perm[j:k].  A genuinely 3-opt neighbour.  Move
    signature for the tabu list is the frozenset of the three
    boundary vertices {perm[i], perm[j], perm[k]}."""
    if n < 6:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(0, n - 5)
    j = random.randint(i + 2, n - 3)
    k = random.randint(j + 2, n - 1)
    sig = frozenset({perm[i], perm[j], perm[k]})
    out[i:j] = out[i:j][::-1]
    out[j:k] = out[j:k][::-1]
    return out, t, sig

def op_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    block_size = random.randint(2, max(3, n // 20))
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    sig = frozenset(block[:2]) if block else frozenset()
    return out, t, sig

def op_long_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    block_size = random.randint(max(3, n // 50), max(4, n // 5))
    block_size = min(block_size, n - 1)
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    sig = frozenset(block[:2]) if block else frozenset()
    return out, t, sig

def op_bottleneck_to_head(perm, n, t, bn_idx, bn_mask):
    if bn_idx < 0 or t <= 0:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    if random.random() < 0.7 or bn_mask == 0:
        target_v = out[bn_idx]; del out[bn_idx]
    else:
        bits = []; m = bn_mask
        while m:
            b = m & -m; bits.append(b.bit_length() - 1); m ^= b
        v = random.choice(bits); idx = out.index(v); target_v = v; del out[idx]
    new_pos = random.randint(0, max(0, t - 1))
    out.insert(new_pos, target_v)
    return out, t, ('bn', target_v)

def op_swap_head_tail(perm, n, t, bn_idx, bn_mask):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]; i = random.randrange(0, t); j = random.randrange(t, n)
    sig = frozenset({perm[i], perm[j]})
    out[i], out[j] = out[j], out[i]
    return out, t, sig

def op_2opt_torso(perm, n, t, bn_idx, bn_mask):
    if t >= n - 2:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(t, n - 2); j = random.randint(i + 1, n - 1)
    sig = frozenset({perm[i], perm[j]})
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t, sig

def op_t_shift(perm, n, t, bn_idx, bn_mask):
    span = max(1, n // 20)
    new_t = max(0, min(n - 1, t + random.randint(-span, span)))
    return perm[:], new_t, ('t', new_t)

def op_t_random(perm, n, t, bn_idx, bn_mask):
    new_t = random.randint(0, n - 1)
    return perm[:], new_t, ('t', new_t)


OPERATORS: List[Tuple[str, Callable]] = [
    ("swap",              op_swap),
    ("adjacent_swap",     op_adjacent_swap),
    ("insert",            op_insert),
    ("reverse",           op_reverse),
    ("3opt",              op_3opt),
    ("block_move",        op_block_move),
    ("long_block_move",   op_long_block_move),
    ("bottleneck->head",  op_bottleneck_to_head),
    ("swap_head_tail",    op_swap_head_tail),
    ("2opt_torso",        op_2opt_torso),
    ("t_shift",           op_t_shift),
    ("t_random",          op_t_random),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, progress_every, here,
        num_t_seeds=20, tabu_length=None,
        max_tabu_skip_retries=8):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    if tabu_length is None:
        tabu_length = max(8, int(n ** 0.5))

    print(f"\n=== hc13_tabu -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"t-grid = {num_t_seeds}, operators = {len(OPERATORS)}")
    print(f"tabu list length L = {tabu_length}, "
          f"max tabu-skip retries = {max_tabu_skip_retries}")
    print()

    md_t0 = time.time()
    md, ws_label = build_warm_start(n, adj_bits, rng=random.Random(seed))
    print(f"  {ws_label} warm start built in {time.time() - md_t0:.1f}s")

    t_grid = sorted({int(round(i * (n - 1) / (num_t_seeds - 1)))
                     for i in range(num_t_seeds)}) if num_t_seeds > 1 else [0]
    archive = ParetoArchive()
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(md, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), md[:])
    print(f"  seeded archive with {len(archive)} non-dominated points "
          f"(initial score = {-archive.hypervolume(n):,.0f})")

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
        cur_perm, cur_t, adj_bits, n)

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}
    tabu = deque(maxlen=tabu_length)
    tabu_set = set()                       # for O(1) membership test
    tabu_rejected = 0
    aspirated = 0

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1
        weights = [(op_acc[n_] + 1) / (op_att[n_] + 1) for n_, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]

        # Attempt to sample a non-tabu move (up to a bounded number of
        # retries before we just take whatever we got -- aspiration may
        # still let it through).
        cand_perm = cand_t = sig = None
        for _ in range(max_tabu_skip_retries):
            cand_perm, cand_t, sig = op_fn(cur_perm, n, cur_t,
                                           cur_bn_idx, cur_bn_mask)
            if sig not in tabu_set:
                break
        op_att[op_name] += 1

        _, cand_w, cand_bn_idx, cand_bn_mask, cand_soft = evaluate_full(
            cand_perm, cand_t, adj_bits, n)

        is_tabu = sig in tabu_set
        # Aspiration criterion: a tabu move is allowed if it adds to the
        # archive (extends or refines the Pareto front).
        archive_added = archive.try_add(int(cand_w), int(cand_t), cand_perm)
        if archive_added:
            archive_adds += 1
            op_acc[op_name] += 1
            if is_tabu:
                aspirated += 1

        if is_tabu and not archive_added:
            tabu_rejected += 1
            continue

        # Standard lex-improvement acceptance for the working solution.
        local_better = (cand_w < cur_w
                        or (cand_w == cur_w and cand_soft < cur_soft))
        if archive_added or local_better:
            cur_perm = cand_perm
            cur_t = cand_t
            cur_w = cand_w
            cur_soft = cand_soft
            cur_bn_idx = cand_bn_idx
            cur_bn_mask = cand_bn_mask
            # Register the move signature in the tabu list.
            if len(tabu) == tabu_length:
                tabu_set.discard(tabu[0])     # evict oldest
            tabu.append(sig)
            tabu_set.add(sig)

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | tabu-rej {tabu_rejected:5d} | "
                  f"asp {aspirated:4d} | "
                  f"focus(w={cur_w}, t={cur_t}) | "
                  f"score = {-archive.hypervolume(n):>14,.0f} | "
                  f"t = {elapsed:5.1f}s")
            last_print = iters

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}, "
          f"tabu-rejected = {tabu_rejected}, aspirated = {aspirated}")
    print(f"Final archive size: {len(archive)}")
    final_hv = archive.hypervolume(n)
    print(f"Official score: {-final_hv:,.0f}")
    if target is not None:
        gap = -final_hv - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print(f"\nOperator stats (accept rate):")
    for name, _ in OPERATORS:
        att = op_att[name]
        rate = op_acc[name] / att if att else 0
        print(f"  {name:18s}  {op_acc[name]:5d} / {att:6d}   ({rate:6.2%})")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc13")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    ap.add_argument("--tabu-length", type=int, default=None,
                    help="tabu list length L (default: max(8, sqrt(n)))")
    args = ap.parse_args()
    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds, tabu_length=args.tabu_length)


if __name__ == "__main__":
    main()

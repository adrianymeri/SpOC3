#!/usr/bin/env python3
"""
hc9_hv_accept.py -- HV-improvement acceptance (SMS-EMOA / IBEA style).

References
----------
Beume, N., Naujoks, B. and Emmerich, M. (2007).  "SMS-EMOA:
Multiobjective selection based on dominated hypervolume."
European Journal of Operational Research 181(3): 1653-1669.

Zitzler, E. and Künzli, S. (2004).  "Indicator-based selection in
multiobjective search."  PPSN VIII, LNCS 3242: 832-842.

Difference from hc4
-------------------
hc4's acceptance for the working solution is lex `(max_w, soft)`.
That works inside a single neighbourhood but rejects candidates
which would *extend* the Pareto front in an under-explored t region
because they don't dominate the current focus.

hc9 replaces that with the SMS-EMOA / IBEA criterion: **accept the
candidate iff its addition strictly increases the archive's
hypervolume**.  This aligns the acceptance criterion with the
scoring indicator and directly attacks the "missing middle of the
front" problem on small-graph.

Concretely, after the operator builds a candidate (cand_w, cand_t):

    delta_hv = archive.hv_with(cand) - archive.hv()
    if delta_hv > 0:
        accept   # archive try_add will succeed; working soln moves

Note that the Pareto archive itself already rejects dominated
candidates -- so `delta_hv > 0` is equivalent to "candidate is
non-dominated AND not a duplicate".  The novelty is that the
WORKING SOLUTION moves whenever delta_hv > 0, even when the
candidate is lex-worse than the current focus.
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
    build_warm_start,
    evaluate_full,
    graph_path,
    hypervolume_2d,
    load_graph,
    min_degree_perm,
    op_bottleneck_relocate,
    op_min_fill_reinsert,
    op_or_opt,
    submission_path,
    write_submission,
)


# Adjacency bitsets for the current run, set in run(); read by the
# min-fill-reinsert operator wrapper (operators are module-level and
# cannot otherwise see adj_bits).
_ADJ_BITS: List[int] | None = None


# ---------------------------------------------------------------------------
# Operators (identical to hc4)
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]
    return out, t

def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, t

def op_insert(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.randrange(n), random.randrange(n)
    if i == j:
        return out, t
    out.insert(j, out.pop(i))
    return out, t

def op_reverse(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = sorted(random.sample(range(n), 2))
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t

def op_3opt(perm, n, t, bn_idx, bn_mask):
    """3-opt double-reverse: pick i < j < k and simultaneously reverse
    perm[i:j] AND perm[j:k].  A genuinely 3-opt neighbour not reachable
    from any single reverse or block_move from the current perm."""
    if n < 6:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(0, n - 5)
    j = random.randint(i + 2, n - 3)
    k = random.randint(j + 2, n - 1)
    out[i:j] = out[i:j][::-1]
    out[j:k] = out[j:k][::-1]
    return out, t



def op_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    block_size = random.randint(2, max(3, n // 20))
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out, t

def op_long_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    block_size = random.randint(max(3, n // 50), max(4, n // 5))
    block_size = min(block_size, n - 1)
    start = random.randint(0, n - block_size)
    block = out[start:start + block_size]
    del out[start:start + block_size]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out, t

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
    return out, t

def op_swap_head_tail(perm, n, t, bn_idx, bn_mask):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randrange(0, t); j = random.randrange(t, n)
    out[i], out[j] = out[j], out[i]
    return out, t

def op_2opt_torso(perm, n, t, bn_idx, bn_mask):
    if t >= n - 2:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(t, n - 2); j = random.randint(i + 1, n - 1)
    out[i:j + 1] = reversed(out[i:j + 1])
    return out, t

def op_t_shift(perm, n, t, bn_idx, bn_mask):
    span = max(1, n // 20)
    return perm[:], max(0, min(n - 1, t + random.randint(-span, span)))

def op_t_random(perm, n, t, bn_idx, bn_mask):
    return perm[:], random.randint(0, n - 1)


# --- New operators (delegate to the shared core implementations) ---
def op_or_opt_w(perm, n, t, bn_idx, bn_mask):
    return op_or_opt(perm, n, t)

def op_bottleneck_relocate_w(perm, n, t, bn_idx, bn_mask):
    return op_bottleneck_relocate(perm, n, t, bn_idx)

def op_min_fill_reinsert_w(perm, n, t, bn_idx, bn_mask):
    return op_min_fill_reinsert(perm, n, t, bn_idx, _ADJ_BITS)


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
    ("or_opt",            op_or_opt_w),
    ("bottleneck_relocate", op_bottleneck_relocate_w),
    ("min_fill_reinsert", op_min_fill_reinsert_w),
    ("t_shift",           op_t_shift),
    ("t_random",          op_t_random),
]


def _call_op(op_fn, perm, n, t, bn_idx, bn_mask):
    out = op_fn(perm, n, t, bn_idx, bn_mask)
    return out if isinstance(out, tuple) and len(out) == 2 else (out, t)


# ---------------------------------------------------------------------------
# Marginal HV helper
# ---------------------------------------------------------------------------

def hv_with_candidate(archive_points, cand, n):
    """HV of the union of `archive_points` and `cand`.  Linear time."""
    pts = list(archive_points) + [cand]
    return hypervolume_2d(pts, n)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, progress_every, here, num_t_seeds=20,
        warm_start="auto"):
    global _ADJ_BITS
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)
    _ADJ_BITS = adj_bits

    print(f"\n=== hc9_hv_accept -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"t-grid = {num_t_seeds}, operators = {len(OPERATORS)}")
    print("acceptance = HV-improvement (SMS-EMOA / IBEA)")
    print()

    md_t0 = time.time()
    md, ws_label = build_warm_start(n, adj_bits, rng=random.Random(seed),
                                    method=warm_start)
    print(f"  {ws_label} warm start built in {time.time() - md_t0:.1f}s")

    t_grid = sorted({int(round(i * (n - 1) / (num_t_seeds - 1)))
                     for i in range(num_t_seeds)}) if num_t_seeds > 1 else [0]
    archive = ParetoArchive()
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(md, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), md[:])

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(
        cur_perm, cur_t, adj_bits, n)
    cur_hv = archive.hypervolume(n)
    print(f"  initial archive {len(archive)}, score = {-cur_hv:,.0f}")

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1
        weights = [(op_acc[n_] + 1) / (op_att[n_] + 1) for n_, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]

        cand_perm, cand_t = _call_op(op_fn, cur_perm, n, cur_t,
                                     cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1

        _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
            cand_perm, cand_t, adj_bits, n)

        # --- HV-IMPROVEMENT ACCEPTANCE ------------------------------
        # Compute the HV the archive would have if (cand_w, cand_t) were
        # added.  If that strictly exceeds the current archive HV, accept.
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            # archive try_add will succeed for non-dominated candidates;
            # update both the archive and the working solution.
            if archive.try_add(int(cand_w), int(cand_t), cand_perm):
                archive_adds += 1
            op_acc[op_name] += 1
            cur_perm = cand_perm
            cur_t = cand_t
            cur_bn_idx = cand_bn_idx
            cur_bn_mask = cand_bn_mask
            cur_hv = archive.hypervolume(n)   # recompute from authoritative archive

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | "
                  f"focus(w={cand_w}, t={cur_t}) | "
                  f"score = {-cur_hv:>14,.0f} | t = {elapsed:5.1f}s")
            last_print = iters

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}")
    print(f"Final archive size: {len(archive)}")
    final_hv = archive.hypervolume(n)
    print(f"Official score: {-final_hv:,.0f}")
    if target is not None:
        gap = -final_hv - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print(f"\nOperator stats:")
    for name, _ in OPERATORS:
        att = op_att[name]
        rate = op_acc[name] / att if att else 0
        print(f"  {name:18s}  {op_acc[name]:5d} / {att:6d}   ({rate:6.2%})")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc9")
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
    ap.add_argument("--warm-start", default="auto",
                    choices=["auto", "min-fill", "min-degree"])
    args = ap.parse_args()
    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds, warm_start=args.warm_start)


if __name__ == "__main__":
    main()

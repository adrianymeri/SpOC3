#!/usr/bin/env python3
"""
hc11_ils.py -- Iterated Local Search (ILS).

Reference
---------
Lourenço, H. R., Martin, O. C. and Stützle, T. (2003).
"Iterated Local Search."  In: Handbook of Metaheuristics, F. Glover
and G. A. Kochenberger (eds.), Springer.

Difference from hc4 / hc7
-------------------------
hc4 climbs a single basin until time runs out.  hc7 escapes via
random restarts -- but a random restart throws away every gain
(including the warm start).

ILS sits between the two.  When the search stagnates for K consecutive
iterations without an archive addition, it applies a STRONG
PERTURBATION (a few large operators applied without acceptance) to
the best-so-far working solution, and then resumes local search from
the perturbed state.  The warm-start gains carry across restarts.

Concretely, the outer loop is:

    S, S_best = warm_start
    while time:
        S = local_search(S, max_idle=K)
        if quality(S) < quality(S_best):
            S_best = S
        S = perturb(S_best, strength)

where `perturb` applies `strength` operators (default: 5) of type
`long_block_move` / `reverse` / `shuffle-segment` WITHOUT acceptance,
producing a state that is far from `S_best` but still inherits its
structure (vertex set partitioning).

ILS-specific hyperparameters (paper-style defaults):
    K         = stagnation window before perturbation        (default 500)
    strength  = number of perturbation moves applied         (default 5)
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


# ---------------------------------------------------------------------------
# Operators (identical to hc4)
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]; return out, t

def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]; return out, t

def op_insert(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.randrange(n), random.randrange(n)
    if i == j: return out, t
    out.insert(j, out.pop(i)); return out, t

def op_reverse(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i, j = sorted(random.sample(range(n), 2))
    out[i:j + 1] = reversed(out[i:j + 1]); return out, t

def op_3opt(perm, n, t, bn_idx, bn_mask):
    """3-opt double-reverse: pick i < j < k and simultaneously reverse
    perm[i:j] AND perm[j:k].  A genuinely 3-opt neighbour."""
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
    out = perm[:]; bs = random.randint(2, max(3, n // 20))
    s = random.randint(0, n - bs); block = out[s:s + bs]; del out[s:s + bs]
    ip = random.randint(0, len(out)); out[ip:ip] = block; return out, t

def op_long_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; bs = random.randint(max(3, n // 50), max(4, n // 5))
    bs = min(bs, n - 1); s = random.randint(0, n - bs)
    block = out[s:s + bs]; del out[s:s + bs]
    ip = random.randint(0, len(out)); out[ip:ip] = block; return out, t

def op_bottleneck_to_head(perm, n, t, bn_idx, bn_mask):
    if bn_idx < 0 or t <= 0:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    if random.random() < 0.7 or bn_mask == 0:
        tv = out[bn_idx]; del out[bn_idx]
    else:
        bits = []; m = bn_mask
        while m:
            b = m & -m; bits.append(b.bit_length() - 1); m ^= b
        v = random.choice(bits); idx = out.index(v); tv = v; del out[idx]
    np_ = random.randint(0, max(0, t - 1)); out.insert(np_, tv); return out, t

def op_swap_head_tail(perm, n, t, bn_idx, bn_mask):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]; i = random.randrange(0, t); j = random.randrange(t, n)
    out[i], out[j] = out[j], out[i]; return out, t

def op_2opt_torso(perm, n, t, bn_idx, bn_mask):
    if t >= n - 2: return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]; i = random.randint(t, n - 2); j = random.randint(i + 1, n - 1)
    out[i:j + 1] = reversed(out[i:j + 1]); return out, t

def op_t_shift(perm, n, t, bn_idx, bn_mask):
    span = max(1, n // 20)
    return perm[:], max(0, min(n - 1, t + random.randint(-span, span)))

def op_t_random(perm, n, t, bn_idx, bn_mask):
    return perm[:], random.randint(0, n - 1)


OPERATORS: List[Tuple[str, Callable]] = [
    ("swap", op_swap), ("adjacent_swap", op_adjacent_swap),
    ("insert", op_insert), ("reverse", op_reverse),
    ("3opt",              op_3opt),
    ("block_move", op_block_move), ("long_block_move", op_long_block_move),
    ("bottleneck->head", op_bottleneck_to_head),
    ("swap_head_tail", op_swap_head_tail), ("2opt_torso", op_2opt_torso),
    ("t_shift", op_t_shift), ("t_random", op_t_random),
]


# Perturbation operators (chosen to be DISRUPTIVE -- the ILS literature
# recommends moves "stronger than the local-search neighbourhood").
PERTURBATION_OPS = [op_long_block_move, op_reverse]


def _call_op(op_fn, perm, n, t, bn_idx, bn_mask):
    out = op_fn(perm, n, t, bn_idx, bn_mask)
    return out if isinstance(out, tuple) and len(out) == 2 else (out, t)


def perturb(perm, n, t, strength):
    """Apply `strength` random PERTURBATION_OPS without acceptance.
    Returns the perturbed (perm, t).  Bottleneck info is not maintained
    because the candidate is large-step from the current state."""
    out = perm[:]
    out_t = t
    for _ in range(strength):
        op = random.choice(PERTURBATION_OPS)
        out, out_t = _call_op(op, out, n, out_t, -1, 0)
    return out, out_t


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, progress_every, here,
        num_t_seeds=20, K_stagnation=500, perturb_strength=5):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc11_ils -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"t-grid = {num_t_seeds}, operators = {len(OPERATORS)}")
    print(f"ILS: stagnation K = {K_stagnation}, "
          f"perturb strength = {perturb_strength}")
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

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
        cur_perm, cur_t, adj_bits, n)
    best_w, best_t, best_perm = cur_w, cur_t, cur_perm[:]
    best_soft = cur_soft

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}
    perturbations = 0

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    idle = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1
        weights = [(op_acc[n_] + 1) / (op_att[n_] + 1) for n_, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
        cand_perm, cand_t = _call_op(op_fn, cur_perm, n, cur_t,
                                     cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1

        _, cand_w, cand_bn_idx, cand_bn_mask, cand_soft = evaluate_full(
            cand_perm, cand_t, adj_bits, n)

        added = archive.try_add(int(cand_w), int(cand_t), cand_perm)
        if added:
            archive_adds += 1
            op_acc[op_name] += 1
            idle = 0
        local_better = (cand_w < cur_w
                        or (cand_w == cur_w and cand_soft < cur_soft))
        if added or local_better:
            cur_perm = cand_perm; cur_t = cand_t; cur_w = cand_w
            cur_soft = cand_soft
            cur_bn_idx = cand_bn_idx; cur_bn_mask = cand_bn_mask
            # Track best-so-far for ILS perturbation base
            if (cur_w, cur_soft) < (best_w, best_soft):
                best_w = cur_w; best_t = cur_t
                best_perm = cur_perm[:]; best_soft = cur_soft
        else:
            idle += 1

        # --- ILS PERTURBATION on stagnation ----------------------
        if idle >= K_stagnation:
            perturbations += 1
            idle = 0
            cur_perm, cur_t = perturb(best_perm, n, best_t, perturb_strength)
            _, cur_w, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
                cur_perm, cur_t, adj_bits, n)

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | perturbs {perturbations:3d} | "
                  f"focus(w={cur_w}, t={cur_t}) | best(w={best_w}, t={best_t}) | "
                  f"score = {-archive.hypervolume(n):>14,.0f} | "
                  f"t = {elapsed:5.1f}s")
            last_print = iters

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}, perturbations = {perturbations}")
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
    out_path = submission_path(here, problem, "hc11")
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
    ap.add_argument("--K", type=int, default=500,
                    help="stagnation window before perturbation")
    ap.add_argument("--strength", type=int, default=5,
                    help="number of perturbation moves applied")
    args = ap.parse_args()
    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds,
        K_stagnation=args.K, perturb_strength=args.strength)


if __name__ == "__main__":
    main()

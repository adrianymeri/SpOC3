#!/usr/bin/env python3
"""
vns.py -- Variable Neighborhood Search for the Torso-Decomposition
bi-objective.

References
----------
Mladenovic, N. and Hansen, P. (1997).  "Variable neighborhood search."
Computers & Operations Research 24(11): 1097-1100.

Hansen, P., Mladenovic, N. and Moreno Perez, J.A. (2010).  "Variable
neighbourhood search: methods and applications."  Annals of Operations
Research 175(1): 367-407.

Method
------
VNS alternates *shaking* (a random kick into a progressively larger
neighbourhood) with *local search*, restarting the neighbourhood ladder
whenever the global Pareto front improves.  Concretely, with current
incumbent x and ladder index k in [1 .. k_max]:

  1. Shake: apply k random operator moves from the shared 15-operator pool to
     x, producing x' (this is the "k-th neighbourhood").  A larger k means a
     more violent perturbation, helping escape local optima.
  2. Local search: HV-improvement descent from x' (accept a move iff it
     strictly increases the archive HV -- the same criterion as hc9/GRASP),
     for a short slice, giving x''.
  3. Move-or-not: if the archive HV improved during this round, accept x'' as
     the new incumbent and reset k = 1; otherwise keep the old incumbent and
     increase k (k = k + 1, wrapping back to 1 past k_max).

Every candidate is offered to one global archive, so the score is the HV of
the union over the whole run -- comparable to every other chapter.

Hyperparameters exposed for tuning: --k-max (ladder depth), --shake-strength
(operator moves per ladder rung), --ls-steps (descent steps per local-search
phase), plus the shared --num-t-seeds and --warm-start.
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

# --- sys.path bootstrap -----------------------------------------------------
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from core import (
    LEADERBOARD_TARGETS,
    ParetoArchive,
    build_warm_start,
    ensure_seeded,
    evaluate_full,
    repo_root,
)
from algorithms.meta_common import (
    OPERATORS,
    build_t_grid,
    call_op,
    finalize,
    hv_with_candidate,
    load_problem,
    seed_archive,
)


def shake(perm, t, n, adj_bits, k, strength, deadline=None):
    """Apply k*strength random operator moves -- the k-th neighbourhood.
    Honors `deadline` (wall-clock) so a violent shake at high k never
    overshoots the global budget on the larger instances."""
    cur_perm, cur_t = list(perm), t
    _, _, bn_idx, bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    for _ in range(max(1, k * strength)):
        if deadline is not None and time.time() >= deadline:
            break
        _, op_fn = random.choice(OPERATORS)
        cur_perm, cur_t = call_op(op_fn, cur_perm, n, cur_t, bn_idx, bn_mask)
        _, _, bn_idx, bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    return cur_perm, cur_t


def local_descent(archive, perm, t, adj_bits, n, steps, op_acc, op_att,
                  deadline=None):
    """Bounded HV-improvement descent from (perm, t).  Returns archive adds.
    Stops after `steps` moves OR when `deadline` (wall-clock) is reached,
    whichever comes first, so a single descent never overshoots the global
    budget on the larger instances."""
    cur_perm = list(perm)
    cur_t = t
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    cur_hv = archive.hypervolume(n)
    adds = 0
    for _ in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        weights = [(op_acc[nm] + 1) / (op_att[nm] + 1) for nm, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
        cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                    cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1
        _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
            cand_perm, cand_t, adj_bits, n)
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            if archive.try_add(int(cand_w), int(cand_t), cand_perm):
                adds += 1
            op_acc[op_name] += 1
            cur_perm = cand_perm
            cur_t = cand_t
            cur_bn_idx = cand_bn_idx
            cur_bn_mask = cand_bn_mask
            cur_hv = archive.hypervolume(n)
    return adds, cur_perm, cur_t


# Structured neighbourhood ordering for VND, cheapest/most-local first.
# Names not present in the active OPERATORS pool are silently skipped.
VND_ORDER = [
    "t_shift", "t_random", "adjacent_swap", "swap", "insert", "or_opt",
    "2opt_torso", "reverse", "block_move", "bottleneck->head",
    "bottleneck_relocate", "min_fill_reinsert", "swap_head_tail",
    "3opt", "long_block_move",
]


def vnd_descent(archive, perm, t, adj_bits, n, deadline, op_acc, op_att,
                k_per=8):
    """True Variable Neighbourhood Descent (Hansen & Mladenovic 2001).

    Unlike the stochastic single-sample HV-descent, VND walks a *fixed
    ordered* list of neighbourhoods N_1..N_L.  For the current neighbourhood
    it draws ``k_per`` candidates (best-improvement within the neighbourhood),
    and if the best of them strictly grows the archive HV it applies the move
    and resets to N_1; otherwise it advances to N_{l+1}.  The descent stops
    when a full pass over all neighbourhoods yields no improvement (a true VND
    local optimum) or the deadline is hit.  Returns (adds, perm, t)."""
    op_map = dict(OPERATORS)
    order = [(nm, op_map[nm]) for nm in VND_ORDER if nm in op_map]
    cur_perm = list(perm)
    cur_t = t
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    cur_hv = archive.hypervolume(n)
    adds = 0
    l = 0
    while l < len(order) and time.time() < deadline:
        op_name, op_fn = order[l]
        best = None   # (new_hv, perm, t, w, bn_idx, bn_mask)
        for _ in range(k_per):
            if time.time() >= deadline:
                break
            cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                        cur_bn_idx, cur_bn_mask)
            op_att[op_name] += 1
            _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
                cand_perm, cand_t, adj_bits, n)
            new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
            if best is None or new_hv > best[0]:
                best = (new_hv, cand_perm, cand_t, cand_w,
                        cand_bn_idx, cand_bn_mask)
        if best is not None and best[0] > cur_hv:
            if archive.try_add(int(best[3]), int(best[2]), best[1]):
                adds += 1
            op_acc[op_name] += 1
            cur_perm, cur_t = best[1], best[2]
            cur_bn_idx, cur_bn_mask = best[4], best[5]
            cur_hv = archive.hypervolume(n)
            l = 0                       # improvement -> back to N_1
        else:
            l += 1                      # no luck -> next neighbourhood
    return adds, cur_perm, cur_t


def run(problem, budget_s, seed, progress_every, here, num_t_seeds=20,
        warm_start="auto", k_max=5, shake_strength=2, ls_steps=300,
        local_search="hv-descent", vnd_k=8, algo="vns"):
    random.seed(seed)
    np.random.seed(seed)

    n, adj, adj_bits = load_problem(here, problem)

    print(f"\n=== vns (Variable Neighborhood Search) -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    if local_search not in ("hv-descent", "vnd"):
        raise ValueError(f"unknown local_search {local_search!r} "
                         "(hv-descent|vnd)")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, t-grid = {num_t_seeds}, "
          f"k_max = {k_max}, shake = {shake_strength}, ls_steps = {ls_steps}")
    print(f"local search = {local_search}"
          + (f"  (VND, k/neighbourhood = {vnd_k})" if local_search == "vnd"
             else "  (stochastic HV-descent)"))
    print()

    md_t0 = time.time()
    md, ws_label = build_warm_start(n, adj_bits, rng=random.Random(seed),
                                    method=warm_start)
    print(f"  {ws_label} warm start built in {time.time() - md_t0:.1f}s")

    t_grid = build_t_grid(n, num_t_seeds)
    archive = ParetoArchive()
    seed_archive(archive, md, adj_bits, n, t_grid)
    inc_w, inc_t, inc_perm = ensure_seeded(archive, md, adj_bits, n)
    inc_perm = list(inc_perm)
    print(f"  initial archive {len(archive)}, score = {-archive.hypervolume(n):,.0f}")

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    start = time.time()
    deadline = start + budget_s
    k = 1
    rounds = 0
    total_adds = 0
    last_print = 0

    while time.time() < deadline:
        rounds += 1
        hv_before = archive.hypervolume(n)
        s_perm, s_t = shake(inc_perm, inc_t, n, adj_bits, k, shake_strength,
                            deadline=deadline)
        if local_search == "vnd":
            adds, x_perm, x_t = vnd_descent(archive, s_perm, s_t, adj_bits, n,
                                            deadline, op_acc, op_att,
                                            k_per=vnd_k)
        else:
            adds, x_perm, x_t = local_descent(archive, s_perm, s_t, adj_bits, n,
                                              ls_steps, op_acc, op_att,
                                              deadline=deadline)
        total_adds += adds
        hv_after = archive.hypervolume(n)
        if hv_after > hv_before:
            inc_perm, inc_t = x_perm, x_t   # better front -> move, reset ladder
            k = 1
        else:
            k = k + 1 if k < k_max else 1   # intensify shake, wrap ladder

        if rounds - last_print >= progress_every:
            print(f"  round {rounds:>6d} | k {k:2d} | archive {len(archive):3d} "
                  f"| adds {total_adds:5d} | score = "
                  f"{-hv_after:>14,.0f} | t = {time.time() - start:5.1f}s")
            last_print = rounds

    out_path, nvec, score = finalize(archive, n, here, problem, algo)
    final_hv = -score   # HV of the submitted top-20 front (== verify score)
    print()
    print(f"Finished in {time.time() - start:.1f}s, rounds = {rounds:,}, "
          f"archive adds = {total_adds}")
    print(f"Final archive size: {len(archive)}")
    print(f"Official score: {-final_hv:,.0f}")
    if target is not None:
        gap = -final_hv - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print("\nOperator stats:")
    for name, _ in OPERATORS:
        att = op_att[name]
        rate = op_acc[name] / att if att else 0
        print(f"  {name:18s}  {op_acc[name]:5d} / {att:6d}   ({rate:6.2%})")

    print(f"\nWrote submission: {out_path}  ({nvec} vectors)")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    ap.add_argument("--warm-start", default="auto",
                    choices=["auto", "min-fill", "min-degree"])
    ap.add_argument("--k-max", type=int, default=5,
                    help="neighbourhood ladder depth")
    ap.add_argument("--shake-strength", type=int, default=2,
                    help="operator moves per ladder rung")
    ap.add_argument("--ls-steps", type=int, default=300,
                    help="HV-descent steps per local-search phase")
    ap.add_argument("--local-search", default="hv-descent",
                    choices=["hv-descent", "vnd"],
                    help="local search: stochastic HV-descent (default) or "
                         "true Variable Neighbourhood Descent")
    ap.add_argument("--vnd-k", type=int, default=8,
                    help="candidates per neighbourhood in VND")
    ap.add_argument("--algo", default="vns",
                    help="submission filename stem (override for tuning)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.progress_every, repo_root(),
        num_t_seeds=args.num_t_seeds, warm_start=args.warm_start,
        k_max=args.k_max, shake_strength=args.shake_strength,
        ls_steps=args.ls_steps, local_search=args.local_search,
        vnd_k=args.vnd_k, algo=args.algo)


if __name__ == "__main__":
    main()

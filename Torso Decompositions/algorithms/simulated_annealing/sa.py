#!/usr/bin/env python3
"""
sa.py -- Simulated Annealing for the Torso-Decomposition bi-objective.

References
----------
Kirkpatrick, S., Gelatt, C.D. and Vecchi, M.P. (1983).  "Optimization by
simulated annealing."  Science 220(4598): 671-680.

Cerny, V. (1985).  "Thermodynamical approach to the travelling salesman
problem: an efficient simulation algorithm."  Journal of Optimization
Theory and Applications 45(1): 41-51.

Method
------
A single walker explores the same 15-operator neighbourhood used by the
Hill-Climbing chapter (algorithms/hill_climbing/), but instead of only
accepting improving moves it uses the **Metropolis criterion**: a worsening
move of energy increase dE is accepted with probability exp(-dE / T), where
the temperature T is lowered over the run by a cooling schedule.

The scalar energy is the negative single-point hypervolume,
    E(w, t) = -(n - w) * (n - t)
(the area of the rectangle the decision (w, t) dominates against the
reference point (n, n)).  Minimising E pushes the walker toward large
dominated area, which is exactly what the -HV objective rewards.

Independently of acceptance, *every* candidate is offered to the Pareto
archive (archive.try_add), so the archive records the best front the walk
ever touches even while the walker itself wanders.  The reported score is
the archive HV, identical in definition to every other chapter.

Cooling schedules (CLI --schedule)
----------------------------------
  geometric : T_{k+1} = alpha * T_k                 (classic, default)
  linear    : T decays linearly from T0 to ~0 over the wall-clock budget
  adaptive  : geometric, but T is reheated when the acceptance rate over a
              window falls below a floor -- keeps the walk from freezing
              early on the larger instances.

Hyperparameters exposed for tuning: --t0, --alpha, --schedule, --steps-per-T,
plus the shared --num-t-seeds and --warm-start.
"""

from __future__ import annotations

import argparse
import math
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
    energy,
    finalize,
    load_problem,
    seed_archive,
)


def domination_amount(cand_w, cand_t, archive_points, n):
    """AMOSA 'amount of domination': the average, over all archive points
    that dominate the candidate, of the normalised dominated-area
    ``((cand_w - p_w)/n) * ((cand_t - p_t)/n)``.  Returns 0.0 when the
    candidate is non-dominated by the archive.  Bandyopadhyay, Saha,
    Maulik & Deb (2008), 'A Simulated Annealing-Based Multiobjective
    Optimization Algorithm: AMOSA', IEEE TEC 12(3): 269-283; the
    dominance-energy lineage is Suppapitnarm et al. (2000), SMOSA."""
    doms = []
    for pw, pt in archive_points:
        if pw <= cand_w and pt <= cand_t and (pw < cand_w or pt < cand_t):
            doms.append(((cand_w - pw) / n) * ((cand_t - pt) / n))
    return (sum(doms) / len(doms)) if doms else 0.0


def run(problem, budget_s, seed, progress_every, here, num_t_seeds=20,
        warm_start="auto", t0=None, alpha=0.95, schedule="geometric",
        steps_per_t=200, accept="scalar", algo="sa"):
    random.seed(seed)
    np.random.seed(seed)

    if accept not in ("scalar", "amosa"):
        raise ValueError(f"unknown accept rule {accept!r} (scalar|amosa)")

    n, adj, adj_bits = load_problem(here, problem)

    print(f"\n=== sa (Simulated Annealing) -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, t-grid = {num_t_seeds}, "
          f"operators = {len(OPERATORS)}")
    print(f"acceptance = {accept}"
          + ("  (single-scalar energy)" if accept == "scalar"
             else "  (AMOSA dominance-amount, bi-objective)"))
    print(f"schedule = {schedule}, alpha = {alpha}, steps/T = {steps_per_t}")
    print()

    md_t0 = time.time()
    md, ws_label = build_warm_start(n, adj_bits, rng=random.Random(seed),
                                    method=warm_start)
    print(f"  {ws_label} warm start built in {time.time() - md_t0:.1f}s")

    t_grid = build_t_grid(n, num_t_seeds)
    archive = ParetoArchive()
    seed_archive(archive, md, adj_bits, n, t_grid)

    cur_w, cur_t, cur_perm = ensure_seeded(archive, md, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    cur_E = energy(cur_w, cur_t, n)
    print(f"  initial archive {len(archive)}, score = {-archive.hypervolume(n):,.0f}")

    # --- automatic T0 calibration -------------------------------------------
    # If t0 is not given, set it so a "typical" worsening move (estimated from
    # a short random sample) is accepted with ~0.4 probability initially.
    if t0 is None:
        deltas = []
        for _ in range(60):
            op_name, op_fn = random.choice(OPERATORS)
            cp, ct = call_op(op_fn, cur_perm, n, cur_t, cur_bn_idx, cur_bn_mask)
            _, cw, _, _, _ = evaluate_full(cp, ct, adj_bits, n)
            d = energy(cw, ct, n) - cur_E
            if d > 0:
                deltas.append(d)
        avg_d = (sum(deltas) / len(deltas)) if deltas else float((n * n) / 10)
        t0 = max(1.0, -avg_d / math.log(0.4))
    T = float(t0)
    print(f"  T0 = {T:,.1f}")

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    accepts = 0
    archive_adds = 0
    last_print = 0
    win_att = win_acc = 0          # acceptance window for adaptive reheating
    best_E = cur_E

    while time.time() < deadline:
        for _ in range(steps_per_t):
            if time.time() >= deadline:
                break
            iters += 1
            # adaptive operator weights, identical scheme to the HC chapter
            weights = [(op_acc[nm] + 1) / (op_att[nm] + 1) for nm, _ in OPERATORS]
            op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
            cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                        cur_bn_idx, cur_bn_mask)
            op_att[op_name] += 1
            win_att += 1

            _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
                cand_perm, cand_t, adj_bits, n)

            # AMOSA needs the domination amount measured against the archive
            # BEFORE the candidate is inserted, so capture it first.
            if accept == "amosa":
                dom_amt = domination_amount(cand_w, cand_t, archive.points(), n)

            # always record into the archive (it rejects dominated/dupes)
            if archive.try_add(int(cand_w), int(cand_t), cand_perm):
                archive_adds += 1

            cand_E = energy(cand_w, cand_t, n)
            if accept == "amosa":
                # Bi-objective Metropolis on the normalised domination amount.
                # A normalised temperature T/t0 (1 -> ~0 over the run) keeps the
                # acceptance scale independent of the energy magnitude.
                t_norm = max(T / max(t0, 1e-9), 1e-3)
                move = dom_amt <= 0.0 or random.random() < math.exp(-dom_amt / t_norm)
            else:
                dE = cand_E - cur_E
                move = dE <= 0 or random.random() < math.exp(-dE / max(T, 1e-9))
            if move:
                cur_perm = cand_perm
                cur_t = cand_t
                cur_w = cand_w
                cur_E = cand_E
                cur_bn_idx = cand_bn_idx
                cur_bn_mask = cand_bn_mask
                op_acc[op_name] += 1
                accepts += 1
                win_acc += 1
                if cand_E < best_E:
                    best_E = cand_E

            if iters - last_print >= progress_every:
                elapsed = budget_s - (deadline - time.time())
                print(f"  iter {iters:>8d} | T {T:>11,.1f} | archive "
                      f"{len(archive):3d} | adds {archive_adds:5d} | "
                      f"acc {accepts/max(iters,1):5.1%} | "
                      f"score = {-archive.hypervolume(n):>14,.0f} | "
                      f"t = {elapsed:5.1f}s")
                last_print = iters

        # --- cool down -------------------------------------------------------
        if schedule == "geometric":
            T *= alpha
        elif schedule == "linear":
            frac = (deadline - time.time()) / budget_s   # 1 -> 0
            T = max(1e-6, t0 * max(0.0, frac))
        elif schedule == "adaptive":
            T *= alpha
            if win_att >= 5 * steps_per_t:
                rate = win_acc / win_att
                if rate < 0.05:           # nearly frozen -> reheat
                    T = max(T, 0.5 * t0)
                win_att = win_acc = 0
        else:
            raise ValueError(f"unknown schedule {schedule!r}")

    elapsed_total = budget_s - (deadline - time.time())
    out_path, nvec, score = finalize(archive, n, here, problem, algo)
    final_hv = -score   # HV of the submitted top-20 front (== verify score)
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"accepts = {accepts:,}, archive adds = {archive_adds}")
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
    ap.add_argument("--progress-every", type=int, default=4000)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    ap.add_argument("--warm-start", default="auto",
                    choices=["auto", "min-fill", "min-degree"])
    ap.add_argument("--t0", type=float, default=None,
                    help="initial temperature (auto-calibrated if omitted)")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="geometric cooling factor")
    ap.add_argument("--schedule", default="geometric",
                    choices=["geometric", "linear", "adaptive"])
    ap.add_argument("--steps-per-t", type=int, default=200,
                    help="Metropolis steps between cooling updates")
    ap.add_argument("--accept", default="scalar",
                    choices=["scalar", "amosa"],
                    help="acceptance rule: scalar energy (default) or "
                         "AMOSA bi-objective dominance-amount")
    ap.add_argument("--algo", default="sa",
                    help="submission filename stem (override for tuning)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.progress_every, repo_root(),
        num_t_seeds=args.num_t_seeds, warm_start=args.warm_start,
        t0=args.t0, alpha=args.alpha, schedule=args.schedule,
        steps_per_t=args.steps_per_t, accept=args.accept, algo=args.algo)


if __name__ == "__main__":
    main()

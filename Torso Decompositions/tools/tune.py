#!/usr/bin/env python3
"""
tune.py -- coarse grid-search hyperparameter tuning for the metaheuristic
chapter (Simulated Annealing, GRASP, VNS).

Why grid search (and not Optuna / random / Bayesian)?
-----------------------------------------------------
For a paper the tuning procedure must be *exactly reproducible* and easy to
report in a table.  A small, hand-chosen grid at a fixed seed satisfies both:
the reader can reconstruct the entire sweep from the printed grid, and there
are no surprises from a stochastic search controller or an external optimiser
dependency.  Each cell is run at the canonical seed (42) and the canonical
per-instance budget, and the cell that minimises the score (-HV, lower is
better) is reported as the tuned configuration.

Usage
-----
    python3 tools/tune.py sa    --problem small-graph [--budget 25] [--seed 42]
    python3 tools/tune.py grasp --problem medium-graph
    python3 tools/tune.py vns   --problem large-graph

Add --full to sweep the dense grid (slower); the default is a coarse grid
sized for an interactive sandbox run.  Results are appended to
extra_instances/tuning_<tech>.csv and the best cell is printed.

Each cell writes to a throwaway submission stem ("<tech>_tune") so the
canonical submissions/ files are never touched.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from core import LEADERBOARD_TARGETS, repo_root  # noqa: E402
import algorithms.simulated_annealing.sa as sa   # noqa: E402
import algorithms.grasp.grasp as grasp           # noqa: E402
import algorithms.vns.vns as vns                 # noqa: E402
import algorithms.aco.aco as aco                 # noqa: E402
import algorithms.nsga2.nsga2 as nsga2           # noqa: E402

CANONICAL_BUDGET = {"small-graph": 25, "medium-graph": 12, "large-graph": 25}

# ---------------------------------------------------------------------------
# Grids.  COARSE is sized for a sandbox sweep; FULL for the workstation.
# ---------------------------------------------------------------------------
GRIDS = {
    "sa": {
        "coarse": {"schedule": ["geometric", "adaptive"],
                   "alpha": [0.90, 0.97],
                   "steps_per_t": [200]},
        "full":   {"schedule": ["geometric", "linear", "adaptive"],
                   "alpha": [0.85, 0.90, 0.95, 0.99],
                   "steps_per_t": [100, 200, 400]},
    },
    "grasp": {
        "coarse": {"alpha": [0.15, 0.35], "restarts": [6, 12]},
        "full":   {"alpha": [0.0, 0.1, 0.2, 0.3, 0.5],
                   "restarts": [4, 8, 16, 32]},
    },
    "vns": {
        "coarse": {"k_max": [3, 6], "shake_strength": [1, 3], "ls_steps": [300]},
        "full":   {"k_max": [2, 4, 6, 10],
                   "shake_strength": [1, 2, 4],
                   "ls_steps": [150, 300, 600]},
    },
    # --- enhanced variants (additive; lift each family's design ceiling) ----
    "sa_amosa": {   # SA with AMOSA bi-objective (dominance-amount) acceptance
        "coarse": {"schedule": ["geometric", "adaptive"],
                   "alpha": [0.90, 0.97],
                   "steps_per_t": [200]},
        "full":   {"schedule": ["geometric", "linear", "adaptive"],
                   "alpha": [0.85, 0.90, 0.95, 0.99],
                   "steps_per_t": [100, 200, 400]},
    },
    "grasp_pr": {   # GRASP + path-relinking + reactive-alpha (alpha is adaptive)
        "coarse": {"restarts": [8, 16], "elite_size": [5]},
        "full":   {"restarts": [4, 8, 16, 32],
                   "elite_size": [3, 5, 8]},
    },
    "vns_vnd": {    # VNS with true Variable Neighbourhood Descent local search
        "coarse": {"k_max": [3, 6], "shake_strength": [1, 3], "vnd_k": [8]},
        "full":   {"k_max": [2, 4, 6, 10],
                   "shake_strength": [1, 2, 4],
                   "vnd_k": [4, 8, 16]},
    },
    # --- ACO bridge chapter -------------------------------------------------
    "aco": {        # Ant System (baseline)
        "coarse": {"ants": [10], "beta": [2.0, 4.0], "rho": [0.1, 0.3]},
        "full":   {"ants": [5, 10, 20],
                   "alpha": [1.0],
                   "beta": [1.0, 2.0, 4.0],
                   "rho": [0.05, 0.1, 0.3]},
    },
    "aco_mmas": {   # MAX-MIN Ant System (enhanced)
        "coarse": {"ants": [10], "beta": [2.0, 4.0], "rho": [0.1, 0.3]},
        "full":   {"ants": [5, 10, 20],
                   "alpha": [1.0],
                   "beta": [1.0, 2.0, 4.0],
                   "rho": [0.05, 0.1, 0.3]},
    },
    # --- hybrid ACO (construction + shared HV descent per ant) --------------
    # Descent is the budget bottleneck, so the colony is smaller and we tune
    # the per-ant descent depth (ls_steps) instead of a wide ants sweep.
    "aco_ls": {     # Ant System + local search
        "coarse": {"ants": [5], "beta": [2.0, 4.0], "ls_steps": [100, 200]},
        "full":   {"ants": [5, 10],
                   "beta": [1.0, 2.0, 4.0],
                   "rho": [0.1, 0.3],
                   "ls_steps": [100, 200, 400]},
    },
    "aco_mmas_ls": {  # MAX-MIN Ant System + local search
        "coarse": {"ants": [5], "beta": [2.0, 4.0], "ls_steps": [100, 200]},
        "full":   {"ants": [5, 10],
                   "beta": [1.0, 2.0, 4.0],
                   "rho": [0.1, 0.3],
                   "ls_steps": [100, 200, 400]},
    },
    # --- population-based MOEA chapter (NSGA-II / SMS-EMOA) ------------------
    "nsga2": {      # NSGA-II: non-dominated rank + crowding-distance selection
        "coarse": {"pop": [30, 60], "pc": [0.9], "pm": [0.3, 0.6]},
        "full":   {"pop": [20, 40, 80],
                   "pc": [0.7, 0.9],
                   "pm": [0.2, 0.4, 0.6]},
    },
    "sms": {        # SMS-EMOA: steady-state HV-contribution selection
        "coarse": {"pop": [30, 60], "pc": [0.9], "pm": [0.3, 0.6]},
        "full":   {"pop": [20, 40, 80],
                   "pc": [0.7, 0.9],
                   "pm": [0.2, 0.4, 0.6]},
    },
    # --- memetic variants (the high-ceiling '_ls' families) -----------------
    # Per-offspring HV-descent is the budget bottleneck, so the population is
    # smaller and we tune the descent depth (ls_steps) instead of a wide sweep.
    "nsga2_ls": {   # memetic NSGA-II (rank+crowding + HV-descent)
        "coarse": {"pop": [20, 40], "pm": [0.4], "ls_steps": [50, 100]},
        "full":   {"pop": [20, 40],
                   "pm": [0.3, 0.6],
                   "ls_steps": [50, 100, 200]},
    },
    "sms_ls": {     # memetic SMS-EMOA (HV selection + HV-descent)
        "coarse": {"pop": [20, 40], "pm": [0.4], "ls_steps": [50, 100]},
        "full":   {"pop": [20, 40],
                   "pm": [0.3, 0.6],
                   "ls_steps": [50, 100, 200]},
    },
}

# throwaway submission stems written during tuning (never canonical)
_TUNE_STEMS = ("sa_tune", "grasp_tune", "vns_tune",
               "sa_amosa_tune", "grasp_pr_tune", "vns_vnd_tune",
               "aco_tune", "aco_mmas_tune",
               "aco_ls_tune", "aco_mmas_ls_tune",
               "nsga2_tune", "sms_tune",
               "nsga2_ls_tune", "sms_ls_tune")


def _cells(grid):
    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def run_cell(tech, problem, budget, seed, here, params):
    """Run one configuration and return its score (-HV; lower is better)."""
    if tech == "sa":
        return sa.run(problem, budget, seed, 10 ** 9, here,
                      schedule=params["schedule"], alpha=params["alpha"],
                      steps_per_t=params["steps_per_t"], algo="sa_tune")
    if tech == "sa_amosa":
        return sa.run(problem, budget, seed, 10 ** 9, here,
                      schedule=params["schedule"], alpha=params["alpha"],
                      steps_per_t=params["steps_per_t"],
                      accept="amosa", algo="sa_amosa_tune")
    if tech == "grasp":
        return grasp.run(problem, budget, seed, here,
                         alpha=params["alpha"], restarts=params["restarts"],
                         algo="grasp_tune")
    if tech == "grasp_pr":
        return grasp.run(problem, budget, seed, here,
                         restarts=params["restarts"],
                         elite_size=params["elite_size"],
                         path_relinking=True, reactive_alpha=True,
                         algo="grasp_pr_tune")
    if tech == "vns":
        return vns.run(problem, budget, seed, 10 ** 9, here,
                       k_max=params["k_max"],
                       shake_strength=params["shake_strength"],
                       ls_steps=params["ls_steps"], algo="vns_tune")
    if tech == "vns_vnd":
        return vns.run(problem, budget, seed, 10 ** 9, here,
                       k_max=params["k_max"],
                       shake_strength=params["shake_strength"],
                       local_search="vnd", vnd_k=params["vnd_k"],
                       algo="vns_vnd_tune")
    if tech == "aco":
        return aco.run(problem, budget, seed, here,
                       ants=params["ants"], beta=params["beta"],
                       rho=params["rho"], alpha=params.get("alpha", 1.0),
                       variant="as", algo="aco_tune")
    if tech == "aco_mmas":
        return aco.run(problem, budget, seed, here,
                       ants=params["ants"], beta=params["beta"],
                       rho=params["rho"], alpha=params.get("alpha", 1.0),
                       variant="mmas", algo="aco_mmas_tune")
    if tech == "aco_ls":
        return aco.run(problem, budget, seed, here,
                       ants=params["ants"], beta=params["beta"],
                       rho=params.get("rho", 0.1), alpha=params.get("alpha", 1.0),
                       variant="as", local_search=True,
                       ls_steps=params["ls_steps"], algo="aco_ls_tune")
    if tech == "aco_mmas_ls":
        return aco.run(problem, budget, seed, here,
                       ants=params["ants"], beta=params["beta"],
                       rho=params.get("rho", 0.1), alpha=params.get("alpha", 1.0),
                       variant="mmas", local_search=True,
                       ls_steps=params["ls_steps"], algo="aco_mmas_ls_tune")
    if tech == "nsga2":
        return nsga2.run(problem, budget, seed, here,
                         pop=params["pop"], pc=params["pc"], pm=params["pm"],
                         variant="nsga2", algo="nsga2_tune")
    if tech == "sms":
        return nsga2.run(problem, budget, seed, here,
                         pop=params["pop"], pc=params["pc"], pm=params["pm"],
                         variant="sms", algo="sms_tune")
    if tech == "nsga2_ls":
        return nsga2.run(problem, budget, seed, here,
                         pop=params["pop"], pc=params.get("pc", 0.9),
                         pm=params["pm"], variant="nsga2",
                         local_search=True, ls_steps=params["ls_steps"],
                         algo="nsga2_ls_tune")
    if tech == "sms_ls":
        return nsga2.run(problem, budget, seed, here,
                         pop=params["pop"], pc=params.get("pc", 0.9),
                         pm=params["pm"], variant="sms",
                         local_search=True, ls_steps=params["ls_steps"],
                         algo="sms_ls_tune")
    raise ValueError(tech)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tech", choices=["sa", "grasp", "vns",
                                     "sa_amosa", "grasp_pr", "vns_vnd",
                                     "aco", "aco_mmas",
                                     "aco_ls", "aco_mmas_ls",
                                     "nsga2", "sms",
                                     "nsga2_ls", "sms_ls"])
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-cell budget (default = canonical)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full", action="store_true",
                    help="use the dense grid (workstation)")
    args = ap.parse_args()

    here = repo_root()
    budget = args.budget or CANONICAL_BUDGET[args.problem]
    grid = GRIDS[args.tech]["full" if args.full else "coarse"]
    cells = list(_cells(grid))

    print(f"\n### tuning {args.tech} on {args.problem} "
          f"({'full' if args.full else 'coarse'} grid, {len(cells)} cells, "
          f"{budget:g}s each, seed {args.seed}) ###\n")

    out_csv = os.path.join(here, "extra_instances", f"tuning_{args.tech}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    best = None
    t0 = time.time()
    for i, params in enumerate(cells):
        score = run_cell(args.tech, args.problem, budget, args.seed, here, params)
        row = {"tech": args.tech, "problem": args.problem, "seed": args.seed,
               "budget": budget, **params, "score": round(score, 3)}
        rows.append(row)
        if best is None or score < best[0]:
            best = (score, params)
        print(f"  [{i + 1}/{len(cells)}] {params} -> score = {score:,.0f}")

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)

    # clean up throwaway submissions
    for stem in _TUNE_STEMS:
        p = os.path.join(here, "submissions", args.problem, f"{stem}.json")
        if os.path.exists(p):
            os.remove(p)

    print(f"\nBest {args.tech} on {args.problem} "
          f"({time.time() - t0:.1f}s total):")
    print(f"  params = {best[1]}")
    print(f"  score  = {best[0]:,.0f}")
    print(f"Appended {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

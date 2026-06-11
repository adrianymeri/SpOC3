#!/usr/bin/env python3
"""
run_metaheuristics.py -- one-command driver for the metaheuristics
finishing sweep (Stage A-C of the green-light plan).

For every family -- baseline AND enhanced variant --
    sa, sa_amosa, grasp, grasp_pr, vns, vns_vnd
it runs the *full* hyperparameter grid (tools/tune.py --full) on every
requested instance at the reporting seed (42), then prints a consolidated
"best config per (tech, instance)" table parsed from the tuning CSVs.

This is the script you run on your workstation.  It only sweeps hyper-
parameters at seed 42; it never touches canonical submissions (tune.py
writes throwaway *_tune stems and deletes them).  When it finishes, send me
the files it lists under "CSVs written" and I lock the winning configs;
the multi-seed robustness + Friedman/Nemenyi head-to-head is the next
script (tools/meta_multiseed.py), run after the configs are locked.

Usage
-----
    python3 tools/run_metaheuristics.py                 # all 6 techs, 3 instances, full grids
    python3 tools/run_metaheuristics.py --techs sa,sa_amosa
    python3 tools/run_metaheuristics.py --problems small-graph,medium-graph
    python3 tools/run_metaheuristics.py --quick         # coarse grids (smoke test only)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_TECHS = ["sa", "sa_amosa", "grasp", "grasp_pr", "vns", "vns_vnd",
             "aco", "aco_mmas", "aco_ls", "aco_mmas_ls", "nsga2", "sms",
             "nsga2_ls", "sms_ls"]
ALL_PROBLEMS = ["small-graph", "medium-graph", "large-graph"]
# hyperparameter columns per tech (everything in the CSV that isn't bookkeeping)
_META_COLS = {"tech", "problem", "seed", "budget", "score"}


def best_rows(csv_path, seed):
    """Return {problem: (score, params_dict)} for the best (lowest -HV) cell
    per problem at the given seed, reading only the freshest run is not
    possible from a CSV that appends, so we take the global best per problem
    -- identical configs collapse, and a better re-run simply wins."""
    best = {}
    if not os.path.exists(csv_path):
        return best
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if int(float(row["seed"])) != seed:
                continue
            p = row["problem"]
            score = float(row["score"])
            params = {k: v for k, v in row.items() if k not in _META_COLS}
            if p not in best or score < best[p][0]:
                best[p] = (score, params)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--techs", default=",".join(ALL_TECHS))
    ap.add_argument("--problems", default=",".join(ALL_PROBLEMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids (smoke test only, NOT for reporting)")
    args = ap.parse_args()

    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    grid_flag = [] if args.quick else ["--full"]

    print("=" * 72)
    print(f"Metaheuristics sweep: techs={techs}")
    print(f"                      instances={problems}")
    print(f"                      grid={'COARSE (quick)' if args.quick else 'FULL'}"
          f", seed={args.seed}")
    print("=" * 72)

    t_start = time.time()
    csvs = []
    for tech in techs:
        csv_path = os.path.join(_HERE, "extra_instances", f"tuning_{tech}.csv")
        csvs.append(csv_path)
        for problem in problems:
            print(f"\n>>> {tech} on {problem} ...")
            cmd = [sys.executable, os.path.join(_HERE, "tools", "tune.py"),
                   tech, "--problem", problem, "--seed", str(args.seed),
                   *grid_flag]
            rc = subprocess.call(cmd, cwd=_HERE)
            if rc != 0:
                print(f"  ! tune.py exited {rc} for {tech}/{problem}")

    # --- consolidated best-config table -----------------------------------
    print("\n" + "=" * 72)
    print("BEST CONFIG PER (tech, instance)  [score = -HV, more negative better]")
    print("=" * 72)
    for tech in techs:
        csv_path = os.path.join(_HERE, "extra_instances", f"tuning_{tech}.csv")
        best = best_rows(csv_path, args.seed)
        print(f"\n{tech}")
        for problem in problems:
            if problem in best:
                score, params = best[problem]
                ps = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  {problem:<13} {score:>14,.0f}   [{ps}]")
            else:
                print(f"  {problem:<13} (no rows)")

    print("\n" + "=" * 72)
    print(f"Done in {time.time() - t_start:.1f}s.")
    print("CSVs written (send these back):")
    for c in csvs:
        if os.path.exists(c):
            print(f"  {c}")
    print("\nNext: I lock the winning configs from these CSVs, then you run")
    print("  python3 tools/meta_multiseed.py --seeds 1,2,3,4,5,6,7,8,9,10,11")
    print("for the seed-robustness + Friedman/Nemenyi head-to-head.")


if __name__ == "__main__":
    main()

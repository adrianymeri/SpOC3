#!/usr/bin/env python3
"""
multiseed.py -- run a small multi-seed sweep without overwriting the
canonical seed=42 submissions.

For each (algo, problem, seed) it:
  1. Backs up the canonical seed=42 submission JSON to a temp file
  2. Runs the algorithm at the given seed
  3. Re-scores the freshly-written submission JSON
  4. Records (algo, problem, seed, score, elapsed_s) into a CSV
  5. Restores the canonical seed=42 submission JSON

Output: extra_instances/multiseed.csv
"""

from __future__ import annotations

import sys
import os
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import argparse
import contextlib
import csv
import importlib
import io
import json
import shutil
import time

from core import (
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
)

# Algorithm signatures (mirror bench_extra._hcN)
def _hcN(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20)

ALGOS = {
    "hc5":  ("hc5_operators",    _hcN),
    "hc9":  ("hc9_hv_accept",    _hcN),
    "hc11": ("hc11_ils",         _hcN),
    "hc14": ("hc14_steepest",    _hcN),
    "hc15": ("hc15_hv_incremental", _hcN),
}

BUDGETS = {"small-graph": 25.0, "medium-graph": 12.0, "large-graph": 25.0}


def score_of(sub_path: str, gr_path: str) -> float:
    with open(sub_path) as f:
        payload = json.load(f)
    entry = payload[0] if isinstance(payload, list) else payload
    n, adj = load_graph(gr_path)
    ab = build_adj_bitsets(n, adj)
    fits = [evaluate(dv[:-1], dv[-1], ab, n) for dv in entry["decisionVector"]]
    return -hypervolume_2d(fits, n)


def run_one(algo: str, problem: str, seed: int, here: str) -> tuple[float, float]:
    mod_name, invoker = ALGOS[algo]
    mod = importlib.import_module(f"algorithms.hill_climbing.{mod_name}")
    sub_path = os.path.join(here, "submissions", problem, f"{algo}.json")
    backup_path = sub_path + ".canonical.bak"

    # Back up the seed=42 canonical submission.
    if os.path.exists(sub_path):
        shutil.copy2(sub_path, backup_path)

    t0 = time.time()
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            invoker(mod, problem, BUDGETS[problem], seed, here)
    except Exception as exc:
        print(f"  ! {algo} {problem} seed={seed}: exception {exc!r}")
        if os.path.exists(backup_path):
            shutil.move(backup_path, sub_path)
        return float("nan"), time.time() - t0

    elapsed = time.time() - t0
    gr = graph_path(here, problem)
    score = score_of(sub_path, gr) if os.path.exists(sub_path) else float("nan")

    # Restore the canonical seed=42 submission.
    if os.path.exists(backup_path):
        shutil.move(backup_path, sub_path)

    return score, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="hc5,hc9,hc11,hc14",
                    help="comma-separated algo short names (hc5 = operators, "
                         "hc9 = HV-accept, hc11 = ILS, hc14 = steepest)")
    ap.add_argument("--problems", default="small-graph,medium-graph,large-graph")
    ap.add_argument("--seeds", default="1,2",
                    help="comma-separated additional seeds (seed 42 is the canonical baseline)")
    ap.add_argument("--out", default="extra_instances/multiseed.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = (not args.append) or (not os.path.exists(out_path))
    f = open(out_path, mode, newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["algo", "problem", "seed", "score", "elapsed_s"])

    total = len(algos) * len(problems) * len(seeds)
    done = 0
    t_start = time.time()
    print(f"Multi-seed sweep: {total} runs ({algos} × {problems} × seeds {seeds})")

    for problem in problems:
        for algo in algos:
            for seed in seeds:
                done += 1
                score, elapsed = run_one(algo, problem, seed, _HERE)
                w.writerow([algo, problem, seed, f"{score:.0f}", f"{elapsed:.2f}"])
                f.flush()
                print(f"  [{done:>3}/{total}] {algo:<5} {problem:<14} seed={seed}  "
                      f"score = {score:>14,.0f}  ({elapsed:5.1f}s)")

    f.close()
    total_elapsed = time.time() - t_start
    print(f"\nWrote {out_path}, total {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

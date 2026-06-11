#!/usr/bin/env python3
"""
regen_canonical.py -- regenerate every saved official-instance
submission at seed=42 with the canonical wall-clock budgets, so the
repository's submissions/*/hc*.json files are byte-reproducible from
the code.

Output: extra_instances/canonical_seed42.csv with columns
  (algo, problem, score, elapsed_s, budget_s, prev_score, delta)

  prev_score is read from the existing submission JSON BEFORE the rerun
  (so the diff captures any drift); after the rerun, the submission
  JSON is the new canonical artifact.

Usage:
    python3 tools/regen_canonical.py                # all 17 × 3 = 51
    python3 tools/regen_canonical.py --algos hc9    # subset
    python3 tools/regen_canonical.py --algos sa,grasp,vns   # metaheuristics only
    python3 tools/regen_canonical.py --problems small-graph
    python3 tools/regen_canonical.py --dry-run      # report only, no write

This is meant to be run on the canonical hardware so that the saved
JSON files in submissions/ are the byte-exact output of the code at
seed=42 under the standard budgets.
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

CANONICAL_SEED = 42

BUDGETS = {
    "small-graph":  25.0,
    "medium-graph": 12.0,
    "large-graph":  25.0,
}

# Mirror tools/bench_extra.ALGO_REGISTRY (kept duplicated to avoid an
# accidental import-order coupling: this script must work even if
# bench_extra is moved or renamed).
def _hc_simple_or_archive(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, 10 ** 18, here)

def _hcN(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20)

def _hc_initial(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, here,
                   interval_min=1.0, interval_max=3.0, show_final=False)


# --- metaheuristic chapter: tuned best params per problem (grid search at
#     seed 42, canonical budgets; see extra_instances/tuning_{sa,grasp,vns}.csv).
SA_BEST = {
    "small-graph":  dict(schedule="linear",    alpha=0.95, steps_per_t=400),
    "medium-graph": dict(schedule="geometric", alpha=0.85, steps_per_t=100),
    "large-graph":  dict(schedule="geometric", alpha=0.95, steps_per_t=200),
}
GRASP_BEST = {
    "small-graph":  dict(alpha=0.0, restarts=4),
    "medium-graph": dict(alpha=0.0, restarts=8),
    "large-graph":  dict(alpha=0.3, restarts=4),
}
VNS_BEST = {
    "small-graph":  dict(k_max=2, shake_strength=2, ls_steps=600),
    "medium-graph": dict(k_max=2, shake_strength=2, ls_steps=150),
    "large-graph":  dict(k_max=2, shake_strength=1, ls_steps=150),
}

def _sa(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, 10 ** 18, here, **SA_BEST[problem])

def _grasp(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, here, **GRASP_BEST[problem])

def _vns(mod, problem, budget, seed, here):
    return mod.run(problem, budget, seed, 10 ** 18, here, **VNS_BEST[problem])


# Registry maps short name -> (fully-qualified module, invoker).
ALGOS = {
    "hc1":   ("algorithms.hill_climbing.hc1_initial",       _hc_initial),
    "hc2":   ("algorithms.hill_climbing.hc2_simple",        _hc_simple_or_archive),
    "hc3":   ("algorithms.hill_climbing.hc3_archive",       _hc_simple_or_archive),
    "hc4":   ("algorithms.hill_climbing.hc4_warm_start",    _hcN),
    "hc5":   ("algorithms.hill_climbing.hc5_operators",     _hcN),
    "hc6":   ("algorithms.hill_climbing.hc6_gap_fill",      _hcN),
    "hc7":   ("algorithms.hill_climbing.hc7_kbottleneck",   _hcN),
    "hc8":   ("algorithms.hill_climbing.hc8_lahc",          _hcN),
    "hc9":   ("algorithms.hill_climbing.hc9_hv_accept",     _hcN),
    "hc10":  ("algorithms.hill_climbing.hc10_torso_warm",   _hcN),
    "hc11":  ("algorithms.hill_climbing.hc11_ils",          _hcN),
    "hc12":  ("algorithms.hill_climbing.hc12_incremental",  _hcN),
    "hc13":  ("algorithms.hill_climbing.hc13_tabu",         _hcN),
    "hc14":  ("algorithms.hill_climbing.hc14_steepest",     _hcN),
    "hc15":  ("algorithms.hill_climbing.hc15_hv_incremental", _hcN),
    "sa":    ("algorithms.simulated_annealing.sa",          _sa),
    "grasp": ("algorithms.grasp.grasp",                     _grasp),
    "vns":   ("algorithms.vns.vns",                         _vns),
}

PROBLEMS = ["small-graph", "medium-graph", "large-graph"]


def score_submission(sub_path: str, gr_path: str):
    """Return the official -HV score of a saved submission, or None
    if the file is missing or unreadable."""
    if not os.path.exists(sub_path):
        return None
    try:
        with open(sub_path) as f:
            payload = json.load(f)
        entry = payload[0] if isinstance(payload, list) else payload
        dvs = entry["decisionVector"]
        n, adj = load_graph(gr_path)
        ab = build_adj_bitsets(n, adj)
        fits = [evaluate(dv[:-1], dv[-1], ab, n) for dv in dvs]
        return -hypervolume_2d(fits, n)
    except Exception:
        return None


def run_one(algo: str, problem: str, here: str) -> tuple:
    """Execute one (algo, problem) at seed=42 with the canonical budget.

    Returns (score_after, elapsed_s, score_before, error).
    score_before is the score of the existing submission JSON, if any.
    """
    mod_path, invoker = ALGOS[algo]
    mod = importlib.import_module(mod_path)
    sub_path = os.path.join(here, "submissions", problem, f"{algo}.json")
    gr = graph_path(here, problem)

    prev_score = score_submission(sub_path, gr)

    budget = BUDGETS[problem]
    t0 = time.time()
    err = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            invoker(mod, problem, budget, CANONICAL_SEED, here)
    except Exception as exc:
        err = repr(exc)
    elapsed = time.time() - t0

    new_score = score_submission(sub_path, gr)
    return new_score, elapsed, prev_score, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default=",".join(ALGOS.keys()),
                    help="comma-separated algo short names (default: all 17 "
                         "— 15 HC + sa + grasp + vns)")
    ap.add_argument("--problems", default=",".join(PROBLEMS),
                    help="comma-separated problems")
    ap.add_argument("--out", default="extra_instances/canonical_seed42.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="report current scores; do not regenerate")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]

    if args.dry_run:
        print("DRY RUN: existing submission scores (no regen)")
        print(f"  {'algo':<6} {'problem':<14} {'score':>16}")
        for problem in problems:
            for algo in algos:
                sub_path = os.path.join(_HERE, "submissions", problem,
                                        f"{algo}.json")
                gr = graph_path(_HERE, problem)
                score = score_submission(sub_path, gr)
                s = f"{score:>16,.0f}" if score is not None else "missing".rjust(16)
                print(f"  {algo:<6} {problem:<14} {s}")
        return

    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = len(algos) * len(problems)
    done = 0
    t_total = time.time()

    print(f"Canonical regen: {total} runs at seed={CANONICAL_SEED}")
    print(f"  algos    = {algos}")
    print(f"  problems = {problems}")
    print(f"  budgets  = {BUDGETS}")
    print()
    print(f"  {'algo':<6} {'problem':<14} {'budget':>6} {'elapsed':>8} "
          f"{'prev score':>16} {'new score':>16} {'delta':>10}")

    rows = []
    for problem in problems:
        for algo in algos:
            done += 1
            new, elapsed, prev, err = run_one(algo, problem, _HERE)
            if err is not None:
                print(f"  ! {algo} {problem}: {err}")
                rows.append([algo, problem,
                             "" if new is None else f"{new:.0f}",
                             f"{elapsed:.2f}",
                             f"{BUDGETS[problem]:.0f}",
                             "" if prev is None else f"{prev:.0f}",
                             "",
                             err])
                continue
            delta = None
            if prev is not None and new is not None:
                delta = new - prev
            ps = f"{prev:>16,.0f}" if prev is not None else "n/a".rjust(16)
            ns = f"{new:>16,.0f}"  if new  is not None else "FAIL".rjust(16)
            ds = f"{delta:>+10,.0f}" if delta is not None else "n/a".rjust(10)
            print(f"  {algo:<6} {problem:<14} {BUDGETS[problem]:>6.0f} "
                  f"{elapsed:>7.1f}s {ps} {ns} {ds}")
            rows.append([algo, problem,
                         "" if new is None else f"{new:.0f}",
                         f"{elapsed:.2f}",
                         f"{BUDGETS[problem]:.0f}",
                         "" if prev is None else f"{prev:.0f}",
                         "" if delta is None else f"{delta:.0f}",
                         ""])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "problem", "score", "elapsed_s", "budget_s",
                    "prev_score", "delta", "error"])
        for row in rows:
            w.writerow(row)

    print()
    print(f"Wrote {out_path}")
    print(f"Total wall time: {(time.time() - t_total) / 60:.1f} min")

    # Summary of drift
    drifted = [r for r in rows if r[6] and r[6] != "0"]
    if drifted:
        print(f"\n{len(drifted)} cells drifted from previous saved scores:")
        for r in drifted:
            print(f"  {r[0]:<6} {r[1]:<14}  Δ = {int(r[6]):+,}")
    else:
        print("\nNo drift: all cells reproduced their previous scores exactly.")


if __name__ == "__main__":
    main()

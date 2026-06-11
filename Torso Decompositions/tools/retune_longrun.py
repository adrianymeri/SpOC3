#!/usr/bin/env python3
"""
retune_longrun.py -- budget-scaled, multi-seed hyperparameter re-tune.

Why this exists
---------------
Every metaheuristic family was grid-searched by `tools/tune.py`, but at the
*canonical* (tight) budget and a *single* seed (42).  Optimal hyperparameters
are budget-dependent -- population size, GRASP alpha/restarts, and SA cooling
in particular systematically shift as the iteration count grows.  So the
recorded "best cell" is provably best *at 25 s, seed 42*, not best in general.

This driver converts that conditional result into an unconditional one.  For
every family it re-runs the SAME full grid that `tune.py` defines, but at a
budget you choose (e.g. 300 s) across N seeds, takes the best cell by
mean-score-over-seeds, and **diffs it against the recorded canonical winner**
read straight out of `extra_instances/tuning_<tech>.csv`.  The output is a
single CONFIRMED / MOVED table per (family, instance):

  - CONFIRMED : the long-run, multi-seed optimum is the same cell already in
                use -> the parameter green light is now unconditional.
  - MOVED     : a different cell wins at the larger budget -> free score; the
                printed row tells you exactly which parameter to change.

SAFE / ADDITIVE: cells write only to throwaway `<tech>_tune` submission stems
(same as tune.py) and are deleted after each family.  No canonical submission
is touched.  A full log is appended to `extra_instances/retune_longrun.csv`.

Parallelism is at the (family, instance) granularity -- collision-free because
each such pair owns a distinct throwaway stem + directory -- so set --workers
to your core count.

Usage
-----
    # default: all 14 families x 3 instances, full grid, 600 s/cell, seeds 1-3
    python3 tools/retune_longrun.py --budget 600 --seeds 1,2,3 --workers 8

    # just the headline families at a big budget, more seeds
    python3 tools/retune_longrun.py --techs grasp,sms,sa,vns --seeds 1,2,3,4,5 --budget 900

    # print the plan + recorded canonical winners, run nothing
    python3 tools/retune_longrun.py --plan-only
"""

from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import concurrent.futures as cf
import csv
import os
import time
from collections import defaultdict

from core import repo_root  # noqa: E402
import tools.tune as tune    # noqa: E402

HERE = repo_root()
ALL_TECHS = list(tune.GRIDS.keys())
ALL_PROBLEMS = ["small-graph", "medium-graph", "large-graph"]


# ---------------------------------------------------------------------------
# Read the recorded canonical winner straight out of the tune.py CSV log.
# ---------------------------------------------------------------------------

def recorded_best(tech: str, problem: str):
    """Return (params_dict, score) of the best recorded cell for this
    (tech, problem) from extra_instances/tuning_<tech>.csv, restricted to the
    grid keys we will re-sweep.  None if no log row exists."""
    path = os.path.join(HERE, "extra_instances", f"tuning_{tech}.csv")
    if not os.path.exists(path):
        return None
    grid_keys = list(tune.GRIDS[tech]["full"].keys())
    best = None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("problem") != problem:
                continue
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            if best is None or score < best[1]:
                params = {k: row[k] for k in grid_keys if k in row}
                best = (params, score)
    return best


def _params_as_str(params: dict, keys) -> tuple:
    """Stringify a param dict on the given keys for type-agnostic comparison
    (CSV values are strings, grid values are typed)."""
    out = []
    for k in keys:
        v = params.get(k)
        # normalise "0.9" vs 0.9 vs "0.90"
        try:
            out.append(("num", float(v)))
        except (TypeError, ValueError):
            out.append(("str", str(v)))
    return tuple(out)


# ---------------------------------------------------------------------------
# One (tech, problem) worker: full grid x seeds, return per-cell mean score.
# ---------------------------------------------------------------------------

def run_pair(tech: str, problem: str, budget: float, seeds):
    grid = tune.GRIDS[tech]["full"]
    cells = list(tune._cells(grid))
    per_cell = defaultdict(list)
    rows = []
    for seed in seeds:
        for params in cells:
            score = tune.run_cell(tech, problem, budget, seed, HERE, params)
            key = tuple(sorted(params.items()))
            per_cell[key].append(score)
            rows.append({"tech": tech, "problem": problem, "seed": seed,
                         "budget": budget, **params, "score": round(score, 3)})
    # clean throwaway stems for this tech in this problem dir
    for stem in tune._TUNE_STEMS:
        p = os.path.join(HERE, "submissions", problem, f"{stem}.json")
        if os.path.exists(p):
            os.remove(p)
    # best cell by mean score over seeds
    best_key = min(per_cell, key=lambda k: sum(per_cell[k]) / len(per_cell[k]))
    best_params = dict(best_key)
    best_mean = sum(per_cell[best_key]) / len(per_cell[best_key])
    return tech, problem, best_params, best_mean, rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--techs", default=",".join(ALL_TECHS),
                    help="comma-separated families (default: all 14)")
    ap.add_argument("--problems", default=",".join(ALL_PROBLEMS),
                    help="comma-separated instances (default: all 3)")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="per-cell wall budget in seconds (default 600)")
    ap.add_argument("--seeds", default="1,2,3",
                    help="comma-separated seeds (default 1,2,3)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel (tech,problem) workers (default cores-1)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print plan + recorded canonical winners, run nothing")
    args = ap.parse_args()

    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bad = [t for t in techs if t not in tune.GRIDS]
    if bad:
        ap.error(f"unknown techs: {bad}")

    pairs = [(t, p) for t in techs for p in problems]
    n_cells = sum(len(list(tune._cells(tune.GRIDS[t]["full"]))) for t, _ in pairs)
    total_runs = n_cells * len(seeds)
    est_min = (total_runs * args.budget) / max(1, args.workers) / 60

    print("=" * 78)
    print("LONG-RUN MULTI-SEED RE-TUNE  (additive: only *_tune throwaway stems)")
    print("=" * 78)
    print(f"families : {len(techs)}   instances: {len(problems)}   "
          f"pairs: {len(pairs)}")
    print(f"seeds    : {seeds}")
    print(f"budget   : {args.budget:g}s/cell   workers: {args.workers}")
    print(f"total cell-runs: {total_runs}   rough wall estimate: "
          f"{est_min:.0f} min ({est_min/60:.1f} h)")
    print("\nRecorded canonical winners (the values currently in use):")
    print(f"  {'family':<13}{'instance':<14}{'recorded best cell':<44}score")
    for t, p in pairs:
        rb = recorded_best(t, p)
        cell = (", ".join(f"{k}={v}" for k, v in rb[0].items()) if rb
                else "(no CSV row)")
        sc = f"{rb[1]:,.0f}" if rb else "-"
        print(f"  {t:<13}{p:<14}{cell:<44}{sc}")
    print("=" * 78)

    if args.plan_only:
        print("\n--plan-only: nothing executed.")
        return

    print(f"\nRunning {len(pairs)} (family,instance) workers ...")
    t0 = time.time()
    results = []
    all_rows = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_pair, t, p, args.budget, seeds): (t, p)
                for t, p in pairs}
        for fut in cf.as_completed(futs):
            t, p = futs[fut]
            try:
                tech, problem, best_params, best_mean, rows = fut.result()
                results.append((tech, problem, best_params, best_mean))
                all_rows.extend(rows)
                print(f"  done {tech}/{problem}: best mean {best_mean:,.0f}")
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {t}/{p}: {e}")
    print(f"\nsweep done in {(time.time() - t0)/60:.1f} min")

    # append the full log
    if all_rows:
        out_csv = os.path.join(HERE, "extra_instances", "retune_longrun.csv")
        keys = sorted({k for r in all_rows for k in r})
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                w.writeheader()
            w.writerows(all_rows)

    # ---- the verdict table ----
    print("\n" + "=" * 78)
    print("VERDICT  (CONFIRMED = canonical cell still wins; MOVED = free score)")
    print("=" * 78)
    print(f"  {'family':<12}{'instance':<13}{'verdict':<10}detail")
    moved = 0
    for tech, problem, lp, lmean in sorted(results):
        rb = recorded_best(tech, problem)
        keys = list(tune.GRIDS[tech]["full"].keys())
        if rb is None:
            print(f"  {tech:<12}{problem:<13}{'NEW':<10}"
                  f"longrun best {lp} (no recorded baseline)")
            continue
        same = _params_as_str(lp, keys) == _params_as_str(rb[0], keys)
        if same:
            print(f"  {tech:<12}{problem:<13}{'CONFIRMED':<10}"
                  f"{', '.join(f'{k}={lp[k]}' for k in keys)}")
        else:
            moved += 1
            rec = ", ".join(f"{k}={rb[0].get(k)}" for k in keys)
            new = ", ".join(f"{k}={lp[k]}" for k in keys)
            print(f"  {tech:<12}{problem:<13}{'MOVED':<10}"
                  f"recorded[{rec}] -> longrun[{new}]  "
                  f"(mean {lmean:,.0f} vs {rb[1]:,.0f})")
    print("=" * 78)
    if moved == 0:
        print("ALL CONFIRMED -> parameters are at their best values at this "
              "budget across all seeds. Unconditional green light.")
    else:
        print(f"{moved} cell(s) MOVED -> re-tuning at scale found free score. "
              "Update the canonical configs for the MOVED rows, then re-run "
              "the portfolio.")
    print("=" * 78)


if __name__ == "__main__":
    main()

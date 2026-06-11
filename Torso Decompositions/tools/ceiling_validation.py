#!/usr/bin/env python3
"""
ceiling_validation.py -- final multi-seed validation of the project ceiling.

Run this on YOUR machine (not the sandbox).  It is the one experiment the
ceiling analysis (docs/FUTURE.md s 1e) flagged as still able to move the
score: the multi-seed union on large-graph, plus an optional multi-seed
sms_ls sweep on small-graph.  Everything else has been measured and is at
its limit.

What it does
------------
  1. Runs N independent GRASP seeds on large-graph at the canonical 25 s
     budget, each written to its OWN additive stem  grasp_s<seed>.json.
  2. (optional) Runs M independent sms_ls seeds on small-graph, each to
     sms_ls_s<seed>.json -- the variant whose best-of-many-seeds run once
     posted -1,818,628, above the current portfolio point.
  3. Re-pools the instance-specific portfolio across all three instances
     (tools/portfolio.py), folding the new seeds into the union.
  4. Re-verifies all three portfolio.json end-to-end (tools/verify_submission.py).

SAFE / ADDITIVE: writes only new  grasp_s<seed>  / sms_ls_s<seed>  stems and
the  portfolio  stem.  Canonical HC and GRASP submissions are read-only
inputs; nothing pre-existing is overwritten.

The conclusion this tests
-------------------------
If, after a few dozen seeds, the portfolio union does not improve beyond the
current  small -1,816,667 / medium -1,617,086 / large -5,025,243, the ceiling
claim is confirmed empirically as well as analytically: "I tried everything,
and it has reached its limit."

Usage
-----
    # default: 24 large GRASP seeds + 24 small sms_ls seeds, then re-pool+verify
    python3 tools/ceiling_validation.py

    # just the large multi-seed union, 48 seeds, 8 parallel workers
    python3 tools/ceiling_validation.py --large-seeds 48 --small-seeds 0 --workers 8

    # dry run: print the plan and the current baseline, run nothing
    python3 tools/ceiling_validation.py --plan-only
"""

from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
GRASP = os.path.join(HERE, "algorithms", "grasp", "grasp.py")
NSGA2 = os.path.join(HERE, "algorithms", "nsga2", "nsga2.py")
PORTFOLIO = os.path.join(HERE, "tools", "portfolio.py")
VERIFY = os.path.join(HERE, "tools", "verify_submission.py")

# Baseline portfolio scores at the time this script was written (seed 42,
# canonical budgets).  Printed for comparison; not used in any decision.
BASELINE = {
    "small-graph": -1_816_667,
    "medium-graph": -1_617_086,
    "large-graph": -5_025_243,
}


def _run(cmd, label):
    """Run one subprocess, stream nothing, return (label, rc, tail)."""
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-3:])
    print(f"  [{label}] rc={p.returncode}  {time.time() - t0:5.1f}s  {tail.splitlines()[-1] if tail else ''}")
    return label, p.returncode


def seed_jobs(large_seeds, small_seeds, budget):
    jobs = []
    for s in range(1, large_seeds + 1):
        jobs.append((
            [PY, "-u", GRASP, "--problem", "large-graph",
             "--budget", str(budget), "--seed", str(s),
             "--algo", f"grasp_s{s}"],
            f"large grasp seed {s}",
        ))
    for s in range(1, small_seeds + 1):
        jobs.append((
            [PY, "-u", NSGA2, "--problem", "small-graph",
             "--variant", "sms", "--local-search", "--pop", "20",
             "--budget", str(budget), "--seed", str(s),
             "--algo", f"sms_ls_s{s}"],
            f"small sms_ls seed {s}",
        ))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large-seeds", type=int, default=24,
                    help="# independent GRASP seeds on large-graph (default 24)")
    ap.add_argument("--small-seeds", type=int, default=24,
                    help="# independent sms_ls seeds on small-graph (default 24)")
    ap.add_argument("--budget", type=float, default=25.0,
                    help="per-seed wall budget in seconds (canonical 25)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel processes (default: cores - 1)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the plan + current baseline, run nothing")
    args = ap.parse_args()

    jobs = seed_jobs(args.large_seeds, args.small_seeds, args.budget)
    est_s = (len(jobs) * args.budget) / max(1, args.workers)

    print("=" * 74)
    print("CEILING VALIDATION  (additive: only grasp_s*/sms_ls_s*/portfolio stems)")
    print("=" * 74)
    print(f"large GRASP seeds : {args.large_seeds}")
    print(f"small sms_ls seeds: {args.small_seeds}")
    print(f"per-seed budget   : {args.budget:.0f}s   workers: {args.workers}")
    print(f"total seed jobs   : {len(jobs)}   rough wall estimate: {est_s/60:.0f} min")
    print("\ncurrent baseline portfolio (beat any of these to break the ceiling):")
    for prob, sc in BASELINE.items():
        print(f"  {prob:<13} {sc:>14,}")
    print("=" * 74)

    if args.plan_only:
        print("\n--plan-only: nothing executed.")
        return

    # 1-2. run the seed sweep in parallel.
    print(f"\n[1/3] running {len(jobs)} seed jobs ...")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, cmd, label) for cmd, label in jobs]
        fail = sum(1 for f in cf.as_completed(futs) if f.result()[1] != 0)
    print(f"  seed sweep done in {(time.time() - t0)/60:.1f} min  ({fail} non-zero rc)")

    # 3. re-pool the portfolio across all instances.
    print("\n[2/3] re-pooling portfolio across all instances ...")
    subprocess.run([PY, "-u", PORTFOLIO,
                    "--problems", "small-graph,medium-graph,large-graph"],
                   cwd=HERE)

    # 4. re-verify all three portfolios end-to-end.
    print("\n[3/3] verifying portfolio submissions ...")
    for prob in ("small-graph", "medium-graph", "large-graph"):
        path = os.path.join(HERE, "submissions", prob, "portfolio.json")
        r = subprocess.run([PY, VERIFY, path, "--quiet"],
                           cwd=HERE, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if any(k in line for k in ("Official", "Capped", "Leaderboard")):
                print(f"  {prob:<13} | {line.strip()}")

    print("\n" + "=" * 74)
    print("Compare the 'Official score' lines above against the baseline table.")
    print("No improvement after this sweep == ceiling confirmed empirically.")
    print("=" * 74)


if __name__ == "__main__":
    main()

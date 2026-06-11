#!/usr/bin/env python3
"""
cmaes_sweep.py -- multi-seed sweep of the continuous-encoding optimiser.

This is the production driver for the paradigm that actually closes the
leaderboard gap (docs/FUTURE.md s 1g): `algorithms/continuous/cmaes_torso.py`.
It runs that optimiser across N independent seeds x the chosen instances, in
parallel on your cores, pools the resulting fronts into the portfolio, and
re-verifies -- exactly the additive/checkpointed pattern of
`tools/ceiling_validation.py`, but for the winning method instead of GRASP.

What it does
------------
  1. For each (instance, seed): run cmaes_torso at the given budget /
     encoding / eigenvector count, written to its OWN additive stem
     `cmaes_s<seed>.json`.  Each file is its own checkpoint -- interrupt and
     resume freely; finished seeds are skipped with --resume.
  2. Re-pool the instance portfolios (now unioning the cmaes fronts on top of
     every prior method) via tools/portfolio.py.
  3. Re-verify all three portfolios end-to-end.

SAFE / ADDITIVE: writes only `cmaes_s<seed>` and `portfolio` stems.  Canonical
HC / GRASP / cmaes submissions are read-only inputs; nothing is overwritten.
`portfolio.py` already refuses to pool throwaway `*_tune` stems.

Usage
-----
    # 8 seeds on every instance, 600 s each, 32 eigenvectors
    python3 tools/cmaes_sweep.py --seeds 8 --budget 600 --eigenvectors 32 --workers 8

    # large only, fcmaes engine, longer
    python3 tools/cmaes_sweep.py --problems large-graph --engine fcmaes \
        --seeds 8 --budget 1800 --workers 8

    # show the plan, run nothing
    python3 tools/cmaes_sweep.py --plan-only
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
CMAES = os.path.join(HERE, "algorithms", "continuous", "cmaes_torso.py")
PORTFOLIO = os.path.join(HERE, "tools", "portfolio.py")
VERIFY = os.path.join(HERE, "tools", "verify_submission.py")
ALL_PROBLEMS = ["small-graph", "medium-graph", "large-graph"]

# Current portfolio baselines to beat (clean, post-contamination-fix).
BASELINE = {
    "small-graph": -1_819_283,
    "medium-graph": -1_617_086,
    "large-graph": -5_033_531,
}


def _run_one(problem, seed, budget, encoding, eigenvectors, engine, tag=""):
    stem = f"cmaes{tag}_s{seed}"
    out = os.path.join(HERE, "submissions", problem, f"{stem}.json")
    t0 = time.time()
    p = subprocess.run(
        [PY, "-u", CMAES, "--problem", problem, "--seed", str(seed),
         "--budget", str(budget), "--encoding", encoding,
         "--eigenvectors", str(eigenvectors), "--engine", engine,
         "--algo", stem],
        cwd=HERE, capture_output=True, text=True)
    tail = ""
    for line in p.stdout.splitlines():
        if "Official score" in line:
            tail = line.strip()
    print(f"  [{problem} seed {seed}] rc={p.returncode} {time.time()-t0:5.0f}s  {tail}")
    return problem, seed, p.returncode


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default=",".join(ALL_PROBLEMS))
    ap.add_argument("--seeds", type=int, default=8,
                    help="number of seeds (1..N) per instance")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--encoding", default="spectral", choices=["spectral", "direct"])
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--engine", default="builtin", choices=["builtin", "fcmaes"])
    ap.add_argument("--tag", default="",
                    help="stem suffix so engines stay additive, e.g. --tag f "
                         "writes cmaesf_s<seed> (default: cmaes_s<seed>)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true",
                    help="skip (instance,seed) whose cmaes_s<seed>.json already exists")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    seeds = list(range(1, args.seeds + 1))

    jobs = []
    for problem in problems:
        for seed in seeds:
            if args.resume and os.path.exists(
                    os.path.join(HERE, "submissions", problem, "seeds",
                                 f"cmaes{args.tag}_s{seed}.json")):
                continue
            jobs.append((problem, seed))
    est_min = len(jobs) * args.budget / max(1, args.workers) / 60

    print("=" * 76)
    print("CMA-ES CONTINUOUS-ENCODING SWEEP  (additive: cmaes_s*/portfolio stems)")
    print("=" * 76)
    print(f"problems     : {problems}")
    print(f"seeds        : {seeds}")
    print(f"budget       : {args.budget:g}s   encoding: {args.encoding}   "
          f"eigvecs: {args.eigenvectors}   engine: {args.engine}")
    print(f"jobs         : {len(jobs)}   workers: {args.workers}   "
          f"rough wall: {est_min:.0f} min")
    print("\nportfolio baselines to beat:")
    for p in problems:
        print(f"  {p:<13} {BASELINE.get(p, 0):>14,}")
    print("=" * 76)
    if args.plan_only:
        print("\n--plan-only: nothing executed.")
        return

    # pre-warm the spectral feature cache: one eigendecomposition per instance,
    # built sequentially here, so the parallel workers all hit the cache instead
    # of each rebuilding the (slow) dense eigendecomposition simultaneously.
    if args.encoding == "spectral":
        print("\n[0/3] pre-warming spectral feature cache ...")
        from algorithms.continuous import cmaes_torso as ct
        from core import load_graph, graph_path
        for problem in sorted({p for p, _ in jobs}):
            n, adj = load_graph(graph_path(HERE, problem))
            tw = time.time()
            _, cached = ct.get_features(HERE, problem, n, adj, args.eigenvectors)
            print(f"  {problem:<13} {'cached' if cached else f'built {time.time()-tw:.0f}s'}")

    print(f"\n[1/3] running {len(jobs)} cmaes jobs ...")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, prob, s, args.budget, args.encoding,
                          args.eigenvectors, args.engine, args.tag)
                for prob, s in jobs]
        fail = sum(1 for f in cf.as_completed(futs) if f.result()[2] != 0)
    print(f"  sweep done in {(time.time()-t0)/60:.1f} min  ({fail} non-zero rc)")

    print("\n[2/3] re-pooling portfolio ...")
    subprocess.run([PY, "-u", PORTFOLIO, "--problems", ",".join(problems)], cwd=HERE)

    print("\n[3/3] verifying portfolios ...")
    for problem in problems:
        path = os.path.join(HERE, "submissions", problem, "portfolio.json")
        r = subprocess.run([PY, VERIFY, path, "--quiet"],
                           cwd=HERE, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if any(k in line for k in ("Official", "Capped", "Leaderboard")):
                print(f"  {problem:<13} | {line.strip()}")

    print("\n" + "=" * 76)
    print("Compare 'Official score' against the baselines above. The cmaes fronts")
    print("should pull every instance toward the leaderboard top.")
    print("=" * 76)


if __name__ == "__main__":
    main()

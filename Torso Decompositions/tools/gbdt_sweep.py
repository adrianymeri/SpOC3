#!/usr/bin/env python3
"""
gbdt_sweep.py -- multi-seed sweep of the GBDT-boosted continuous encoder.

Runs algorithms/continuous/gbdt_torso.py across N seeds x the chosen instances
(different seeds => different GBDT models => front diversity), each written to an
additive `gbdt_s<seed>` stem (routed into submissions/<instance>/seeds/), then
re-pools the portfolio and re-verifies. Same additive/safe pattern as
tools/cmaes_sweep.py. The gbdt fronts pool on top of every cmaes front.

Usage
-----
    # install the strongest backend first (optional; falls back to sklearn HistGBR):
    #   pip3 install lightgbm
    python3 tools/gbdt_sweep.py --seeds 6 --budget 900 --backend lightgbm --workers 6
    python3 tools/gbdt_sweep.py --problems large-graph --seeds 4 --budget 1800
    python3 tools/gbdt_sweep.py --plan-only
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, concurrent.futures as cf, os, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
GBDT = os.path.join(HERE, "algorithms", "continuous", "gbdt_torso.py")
PORTFOLIO = os.path.join(HERE, "tools", "portfolio.py")
VERIFY = os.path.join(HERE, "tools", "verify_submission.py")
ALL = ["small-graph", "medium-graph", "large-graph"]
BASELINE = {"small-graph": -1_828_237, "medium-graph": -1_708_822, "large-graph": -5_383_985}


def _run(problem, seed, budget, eig, backend, rounds):
    stem = f"gbdt_s{seed}"
    t0 = time.time()
    p = subprocess.run([PY, "-u", GBDT, "--problem", problem, "--seed", str(seed),
                        "--budget", str(budget), "--eigenvectors", str(eig),
                        "--backend", backend, "--rounds", str(rounds), "--algo", stem],
                       cwd=HERE, capture_output=True, text=True)
    tail = ""
    for ln in p.stdout.splitlines():
        if "Official score" in ln:
            tail = ln.strip()
    print(f"  [{problem} seed {seed}] rc={p.returncode} {time.time()-t0:5.0f}s  {tail}", flush=True)
    if p.returncode != 0:
        print("\n".join(p.stderr.splitlines()[-4:]))
    return problem, seed, p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(ALL))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "lightgbm", "xgboost", "hist", "ridge"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    jobs = []
    for prob in problems:
        for s in range(1, args.seeds + 1):
            if args.resume and os.path.exists(os.path.join(HERE, "submissions", prob, "seeds", f"gbdt_s{s}.json")):
                continue
            jobs.append((prob, s))
    print("=" * 74)
    print("GBDT-BOOSTED CONTINUOUS SWEEP  (additive: gbdt_s*/portfolio stems)")
    print("=" * 74)
    print(f"problems {problems}  seeds 1..{args.seeds}  budget {args.budget:g}s  "
          f"eig {args.eigenvectors}  backend {args.backend}  rounds {args.rounds}")
    print(f"jobs {len(jobs)}  workers {args.workers}  "
          f"rough wall {len(jobs)*args.budget/max(1,args.workers)/60:.0f} min")
    for p in problems:
        print(f"  baseline {p:<13} {BASELINE.get(p,0):>14,}")
    print("=" * 74)
    if args.plan_only:
        print("\n--plan-only: nothing executed."); return

    print(f"\n[1/3] running {len(jobs)} gbdt jobs ...")
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, p, s, args.budget, args.eigenvectors, args.backend, args.rounds)
                for p, s in jobs]
        fail = sum(1 for f in cf.as_completed(futs) if f.result()[2] != 0)
    print(f"  done in {(time.time()-t0)/60:.1f} min ({fail} non-zero rc)")

    print("\n[2/3] re-pooling portfolio ...")
    subprocess.run([PY, "-u", PORTFOLIO, "--problems", ",".join(problems)], cwd=HERE)
    print("\n[3/3] verifying ...")
    for prob in problems:
        r = subprocess.run([PY, VERIFY, os.path.join(HERE, "submissions", prob, "portfolio.json"), "--quiet"],
                           cwd=HERE, capture_output=True, text=True)
        for ln in r.stdout.splitlines():
            if any(k in ln for k in ("Official", "Capped", "Leaderboard")):
                print(f"  {prob:<13} | {ln.strip()}")


if __name__ == "__main__":
    main()

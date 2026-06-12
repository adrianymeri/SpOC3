#!/usr/bin/env python3
"""
gbfcpp_swarm.py -- parallel GBFC++ driver (one worker per core).

GBFC++ is single-threaded; a modern CPU has 8+ cores idle. The swarm runs N
workers in WAVES: each wave spawns N gbfcpp processes with distinct seeds and
distinct submission stems (gbfcpp, gbfcpp2, ..., gbfcppN), waits for them, then
folds every stem into portfolio.json (tools/portfolio.py) so the NEXT wave's
workers all start from the union of everything any worker found. Diversity
while running, sharing between waves.

    python3 tools/gbfcpp_swarm.py --problem small-graph --workers 6 \
        --waves 40 --rounds 6 --round-budget 60

Ctrl-C is safe at any time (workers checkpoint every round); rerunning
resumes. Total wall time ~= waves * rounds * round_budget.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable or "python3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--waves", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=6, help="rounds per worker per wave")
    ap.add_argument("--round-budget", type=float, default=60.0)
    ap.add_argument("--base-seed", type=int, default=1000)
    args = ap.parse_args()

    # every worker runs single-threaded libraries: with N workers, library
    # thread pools at -1 spawn N x cores threads and crush the host (observed:
    # load average 2,500+ from LightGBM n_jobs=-1 alone)
    wenv = dict(os.environ,
                GBDT_NJOBS="1", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    for wave in range(args.waves):
        t0 = time.time()
        procs = []
        for i in range(args.workers):
            stem = "gbfcpp" if i == 0 else f"gbfcpp{i+1}"
            seed = args.base_seed + wave * args.workers + i
            cmd = [PY, os.path.join(HERE, "gbfcpp.py"),
                   "--problem", args.problem,
                   "--rounds", str(args.rounds),
                   "--round-budget", str(args.round_budget),
                   "--seed", str(seed), "--algo", stem]
            if i == 0:
                cmd += ["--t0", "0.4"]   # cool exploiter (greedy polishing)
            if i > 0:
                # diversify: hotter annealing on odd workers, and a rotating
                # breakpoint-width partition so EVERY breakpoint gets attention
                # instead of all workers piling onto the softmax favourites
                cmd += ["--t0", f"{1.0 + 0.4 * i:.1f}"]
                k = args.workers - 1
                widths = [w for w in range(32) if w % k == (i - 1 + wave) % k]
                cmd += ["--only-widths", ",".join(map(str, widths))]
            procs.append(subprocess.Popen(
                cmd, cwd=ROOT, env=wenv,
                stdout=subprocess.DEVNULL if i else None,   # show worker 0 only
                stderr=subprocess.STDOUT if i else None))
        try:
            for p in procs:
                p.wait()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            print("\ninterrupted -- checkpoints are saved; rerun to resume")
            return
        # fold every stem into portfolio.json so the next wave shares progress
        merge = subprocess.run(
            [PY, os.path.join(HERE, "portfolio.py"), "--problems", args.problem],
            cwd=ROOT, capture_output=True, text=True)
        best = [ln for ln in merge.stdout.splitlines() if "UNION" in ln]
        print(f"\n== wave {wave+1}/{args.waves} done in {time.time()-t0:,.0f}s "
              f"| {best[0].strip() if best else 'merge: see portfolio.py'} ==\n",
              flush=True)


if __name__ == "__main__":
    main()

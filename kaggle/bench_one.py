#!/usr/bin/env python3
"""
bench_one.py -- run ONE solver across every instance, on one core.

Built to be launched several times at once (see run_mac.sh). Each process
handles a single solver and writes its own CSV, so four of them running side
by side never touch the same file and never wait on each other.

    python3 bench_one.py --solver hill_climbing --seconds 1200 --seeds 3

Why one core: hill climbing is pure Python and can only ever use one, so
capping the others matches them to it. Every CPU method then gets exactly the
same compute, which is what makes the columns comparable. run_mac.sh sets the
thread limits; this script checks they took and refuses to run otherwise.

Resumes automatically -- anything already in the CSV is skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "leaderboard-references"))

from esa_eval import build_adj_bitsets, evaluate, hypervolume_2d, MAX_TW
from torso import Graph
import hill_climbing

INSTANCES = (["small-graph", "medium-graph", "large-graph"]
             + [f"synth-{i}" for i in range(1, 8)])
FIELDS = ["instance", "n", "solver", "seed", "score", "seconds", "valid"]

THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]


def score_vectors(graph, vectors):
    """Re-score any solver's output through esa_eval. Returns (ok, score)."""
    n = graph.n
    bits = build_adj_bitsets(n, graph.adj)
    points = []
    for vec in vectors:
        perm, t = [int(x) for x in vec[:-1]], int(vec[-1])
        if len(vec) != n + 1 or sorted(perm) != list(range(n)):
            return False, 0
        if not 0 <= t < n:
            return False, 0
        width, t = evaluate(perm, t, bits, n)
        if width > MAX_TW:
            return False, 0
        points.append((width, t))
    if len(points) > 20 or len(points) != len(set(points)):
        return False, 0
    return True, -int(hypervolume_2d(points, n))


def solve(solver, graph, seconds, seed):
    if solver == "min_degree":
        return hill_climbing.min_degree_only(graph, seed)
    if solver == "hill_climbing":
        return hill_climbing.solve(graph, seconds, seed=seed,
                                   start="min_degree")[0]
    if solver == "hri":
        import hri_lns
        return hri_lns.solve(graph, seconds, seed=seed, start="min_degree")[0]
    if solver == "fast_cma_es":
        import cmaes
        return cmaes.solve(graph, seconds, seed=seed)[0]
    raise ValueError(solver)


def main():
    ap = argparse.ArgumentParser(description="Run one solver on every instance.")
    ap.add_argument("--solver", required=True,
                    choices=["hill_climbing", "hri", "fast_cma_es", "min_degree"])
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="data/v2",
                    help="instance folder, relative to the project root. "
                         "data/v2 holds the twin-preserving synthetics; "
                         "data holds the first (non-discriminating) set.")
    ap.add_argument("--allow-threads", action="store_true",
                    help="skip the one-core check (results not comparable)")
    a = ap.parse_args()

    unset = [v for v in THREAD_VARS if os.environ.get(v) != "1"]
    if unset and not a.allow_threads:
        raise SystemExit(
            "refusing to run: these are not pinned to 1 -> "
            + ", ".join(unset)
            + "\nLaunch via run_mac.sh, or pass --allow-threads if you know "
              "the numbers will not be compared against other solvers.")

    out = a.out or os.path.join(HERE, f"benchmark-{a.solver}.csv")

    done, kept = set(), []
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                key = (r["instance"], int(r["seed"]))
                if key not in done:
                    done.add(key)
                    kept.append(r)

    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in kept:
        w.writerow({k: r.get(k, "") for k in FIELDS})
    f.flush()

    seeds = [1] if a.solver == "min_degree" else list(range(1, a.seeds + 1))
    todo = sum(1 for i in INSTANCES for s in seeds if (i, s) not in done)
    print(f"[{a.solver}] {todo} runs to do "
          f"({len(done)} already in {os.path.basename(out)}), "
          f"{a.seconds:.0f}s each, 1 core, instances from {a.data_dir}",
          flush=True)

    for name in INSTANCES:
        path = os.path.join(ROOT, a.data_dir, f"{name}.gr")
        if not os.path.exists(path):
            print(f"[{a.solver}] MISSING {path}", flush=True)
            continue
        graph = Graph.load(path)
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            front = solve(a.solver, graph, a.seconds, seed)
            ok, score = score_vectors(graph, front.decision_vectors())
            elapsed = time.time() - t0
            w.writerow({"instance": name, "n": graph.n, "solver": a.solver,
                        "seed": seed, "score": score,
                        "seconds": round(elapsed, 1), "valid": ok})
            f.flush()
            print(f"[{a.solver}] {name} seed {seed}: {score:>14,} "
                  f"({elapsed/60:.1f} min){'' if ok else '  INVALID'}",
                  flush=True)

    f.close()
    print(f"[{a.solver}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()

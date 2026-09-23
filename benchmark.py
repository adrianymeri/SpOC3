#!/usr/bin/env python3
"""
benchmark.py -- run every solver on every instance and tabulate the result.

    python3 benchmark.py                                  # all instances, 60s, 3 seeds
    python3 benchmark.py --seconds 120 --seeds 5
    python3 benchmark.py --instances data/toy.gr data/synth-1.gr --seconds 20
    python3 benchmark.py --solvers min_degree hill_climbing

Writes benchmark.csv (one row per solver/instance/seed) and prints a summary.

The columns
-----------
    fast_cma_es       fast-cma-es -- uses the real fcmaes package when it is
                      installed, otherwise a ported separable CMA-ES
    hri               Team HRI -- destroy-and-repair LNS, median placement
    spacekangaroos    the winning entry -- neuro-evolution over a scoring rule
    hill_climbing     ours -- plain hill climbing

Every solver gets the same wall-clock budget per instance, scores through the
same esa_eval.py, and every result is re-validated before it is recorded.

Starting points
---------------
hill_climbing and hri search orderings, so they need one to start from; both
get the SAME min-degree construction, built fresh from the graph inside each
run. spacekangaroos and fast_cma_es search weight vectors and decode with
argsort(w . features), so there is no ordering to hand them -- they start
from random weights.

--start random gives the two permutation solvers a literal random ordering.
Be warned that this is void on 6 of the 10 instances: a random permutation
busts the MAX_TW=500 cap at every prefix, yields no legal point, and leaves
the search nothing to climb. See README.md 10.4.

A no-search baseline is available as `--solvers min_degree ...` if you want
to see how much of any score is the greedy construction rather than search.

What these numbers are: CPU reimplementations at laptop scale, run at equal
budget. Spacekangaroos originally ran thousands of candidates per generation
on a GPU and this manages tens; HRI published no source at all, so that
column is written from their paper. Treat the table as **methods compared at
equal CPU budget**, not as this laptop against the competition leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "leaderboard-references"))

from esa_eval import build_adj_bitsets, evaluate, hypervolume_2d, MAX_TW
from torso import Graph
import hill_climbing

try:
    import neuroevo_cpu
except ImportError:
    neuroevo_cpu = None
try:
    import hri_lns
except ImportError:
    hri_lns = None
try:
    import cmaes
except ImportError:
    cmaes = None


# solver key -> the leaderboard team it reimplements
TEAMS = {
    "min_degree":      "min-degree (baseline)",
    "hill_climbing":   "Hill Climbing (ours)",
    "spacekangaroos":  "Spacekangaroos",
    "hri":             "Team HRI",
    "fast_cma_es":     "fast-cma-es",
}


def check(graph, front):
    """Re-validate a front the way validate.py would. Returns (ok, score)."""
    n = graph.n
    bits = build_adj_bitsets(n, graph.adj)
    points = []
    for vector in front.decision_vectors():
        perm, t = [int(x) for x in vector[:-1]], int(vector[-1])
        if len(vector) != n + 1 or sorted(perm) != list(range(n)):
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


def solve(name, graph, seconds, seed, start="min_degree"):
    """`start` applies to the two permutation-space solvers.

    Spacekangaroos and fast-cma-es search weight vectors, not orderings, so
    their starting point is a random weight vector either way -- there is no
    ordering to hand them.
    """
    if name == "min_degree":
        return hill_climbing.min_degree_only(graph, seed)
    if name == "hill_climbing":
        front, _ = hill_climbing.solve(graph, seconds, seed=seed, start=start)
        return front
    if name == "spacekangaroos":
        if neuroevo_cpu is None:
            return None
        front, _, _ = neuroevo_cpu.solve(graph, seconds, seed=seed)
        return front
    if name == "hri":
        if hri_lns is None:
            return None
        front, _, _ = hri_lns.solve(graph, seconds, seed=seed, start=start)
        return front
    if name == "fast_cma_es":
        if cmaes is None:
            return None
        front, _, _ = cmaes.solve(graph, seconds, seed=seed)
        return front
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description="Compare solvers across instances.")
    ap.add_argument("--instances", nargs="*", default=None)
    # the four columns of the comparison sheet. min_degree is available
    # via --solvers if you ever want the no-search baseline.
    ap.add_argument("--solvers", nargs="*",
                    default=["fast_cma_es", "hri", "spacekangaroos",
                             "hill_climbing"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="benchmark.csv")
    ap.add_argument("--start", default="min_degree",
                    choices=["random", "min_degree"],
                    help="starting ordering for BOTH permutation-space "
                         "solvers (hill_climbing and hri), so neither gets "
                         "an advantage the other does not. 'random' is from "
                         "scratch; 'min_degree' gives both the same greedy "
                         "construction to start from.")
    args = ap.parse_args()

    instances = args.instances
    if not instances:
        # the comparison sheet's row order
        order = (["data/small-graph.gr", "data/medium-graph.gr",
                  "data/large-graph.gr"]
                 + [f"data/synth-{i}.gr" for i in range(1, 8)])
        instances = [p for p in order if os.path.exists(p)]

    print(f"{len(instances)} instances x {len(args.solvers)} solvers "
          f"x {args.seeds} seeds, {args.seconds:.0f}s each")
    print(f"permutation-space solvers start from: {args.start}")
    total = len(instances) * len(args.solvers) * args.seeds * args.seconds
    print(f"estimated wall time: {total / 60:.0f} min\n")

    results = {}
    fields = ["instance", "n", "solver", "seed", "score", "seconds"]

    # Written after EVERY run, not at the end: an 8-hour job that dies at
    # hour 7 should still leave you everything it had finished.
    csv_file = open(args.out, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()
    csv_file.flush()

    for path in instances:
        graph = Graph.load(path)
        name = os.path.basename(path).replace(".gr", "")
        print(f"{name}  (n={graph.n}, edges={graph.edge_count})")

        for solver in args.solvers:
            scores = []
            for seed in range(1, args.seeds + 1):
                # min_degree does no searching, so one seed is the whole story
                if solver == "min_degree" and seed > 1:
                    continue
                t0 = time.time()
                front = solve(solver, graph, args.seconds, seed,
                              start=args.start)
                elapsed = time.time() - t0
                if front is None:
                    print(f"   {solver:<14} unavailable (numpy missing?)")
                    break
                ok, score = check(graph, front)
                if not ok:
                    print(f"   {solver:<14} seed {seed}: INVALID RESULT")
                    continue
                scores.append(score)
                writer.writerow({"instance": name, "n": graph.n,
                                 "solver": solver, "seed": seed,
                                 "score": score, "seconds": round(elapsed, 1)})
                csv_file.flush()
            if scores:
                best = min(scores)
                mean = statistics.mean(scores)
                results[(name, solver)] = (best, mean)
                spread = (f"  (mean {mean:,.0f} over {len(scores)})"
                          if len(scores) > 1 else "")
                print(f"   {solver:<14} best {best:>14,}{spread}")
        print()

    csv_file.close()

    # summary table, best score per solver
    print("=" * 78)
    print(f"BEST SCORE PER SOLVER  ({args.seconds:.0f}s budget, "
          f"{args.seeds} seeds, more negative is better)")
    print("=" * 78)
    labels = [TEAMS.get(s, s) for s in args.solvers]
    width = max(18, max(len(l) for l in labels) + 2)
    print(f"| {'instance':<16} |" + "".join(f" {l:>{width-2}} |" for l in labels))
    print("|" + "-" * 18 + "|" + "".join("-" * width + "|" for _ in args.solvers))
    for path in instances:
        name = os.path.basename(path).replace(".gr", "")
        line = f"| {name:<16} |"
        best_here = min((results[(name, s)][0] for s in args.solvers
                         if (name, s) in results), default=None)
        for s in args.solvers:
            if (name, s) in results:
                score = results[(name, s)][0]
                mark = "*" if score == best_here else " "
                line += f" {score:>{width-3},}{mark}|"
            else:
                line += f" {'-':>{width-2}} |"
        print(line)
    print("\n* = best on that instance")
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 78)
    print("PASTE INTO THE SHEET  (tab-separated, sheet row order)")
    print("=" * 78)
    for path in instances:
        name = os.path.basename(path).replace(".gr", "")
        cells = [str(results[(name, s)][0]) if (name, s) in results else ""
                 for s in args.solvers]
        print(name + "\t" + "\t".join(cells))

    print("\nColumn names are the leaderboard teams whose method each solver")
    print("reimplements. All run on CPU at equal budget -- see")
    print("leaderboard-references/README.md for what that does and does not mean.")

    if "min_degree" in args.solvers and "hill_climbing" in args.solvers:
        gains = [results[(os.path.basename(p).replace('.gr', ''), 'min_degree')][0]
                 - results[(os.path.basename(p).replace('.gr', ''), 'hill_climbing')][0]
                 for p in instances
                 if (os.path.basename(p).replace('.gr', ''), 'min_degree') in results
                 and (os.path.basename(p).replace('.gr', ''), 'hill_climbing') in results]
        if gains:
            print(f"\nhill climbing over its own starting heuristic: "
                  f"min {min(gains):,} / median {statistics.median(gains):,.0f} "
                  f"/ max {max(gains):,} HV")
            print("If those numbers are small, the greedy construction is doing "
                  "the work, not the search.")


if __name__ == "__main__":
    main()

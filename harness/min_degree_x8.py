#!/usr/bin/env python3
"""
min_degree_x8.py -- the no-search baseline column of the results sheet.

`benchmark.py --solvers min_degree` runs ONE min-degree construction
(hill_climbing.min_degree_only, seed 1). The results sheet reports the
*best of eight* constructions with random tie-breaking, which is what
hill_climbing.py actually builds before it starts searching -- eight
constructions donate their staircases to one shared front, exactly as
Starts.min_degree is used inside the climber.

This script produces that column and nothing else, so every figure in the
sheet has a committed source file.

    cd harness && python3 min_degree_x8.py    ->  benchmark-min_degree-x8.csv

Seeds 1..8, fixed. The construction breaks degree ties with rng.choice, so
the seed set is part of the result: a different set of eight seeds moves
small-graph, medium-graph, synth-4 and synth-5 by a few thousand HV, and
leaves the other six unchanged (they saturate).
"""
import csv, os, random, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torso import Graph, Solution, Front
import hill_climbing as hc

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INSTANCES = ["small-graph", "medium-graph", "large-graph",
             "synth-1", "synth-2", "synth-3", "synth-4",
             "synth-5", "synth-6", "synth-7"]
SEEDS = range(1, 9)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "benchmark-min_degree-x8.csv")

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["instance", "n", "solver", "seed", "score", "seconds", "valid"])
    for name in INSTANCES:
        graph = Graph.load(os.path.join(ROOT, "data", f"{name}.gr"))
        t0 = time.time()
        front = Front(graph.n)
        for seed in SEEDS:
            front.add_solution(
                Solution(graph, hc.Starts.min_degree(graph, random.Random(seed))))
        score, secs = int(front.score()), time.time() - t0
        w.writerow([name, graph.n, "min_degree_x8", "1-8", score,
                    round(secs, 1), True])
        fh.flush()
        print(f"{name:<14} n={graph.n:<5} {score:>14,}  ({secs:.1f}s)", flush=True)
print(f"\nwrote {OUT}")

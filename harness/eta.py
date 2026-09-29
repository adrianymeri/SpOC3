#!/usr/bin/env python3
"""
eta.py -- how much longer the ablation has to run.

    python3 eta.py

Counts what each process has finished, costs the remaining runs using the
per-instance wall-clock already measured in benchmark-hill_climbing.csv (the
budget is 1,200 s but the per-target-width construction sits outside the timed
loop, and on synth-6 that pushes a run past 2,800 s), and reports the finish
time of the slowest process -- they run in parallel, so that is the finish.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OPS = ["swap_neighbours", "swap_any", "move_vertex", "reverse_segment", "all"]
INSTANCES = (["small-graph", "medium-graph", "large-graph"]
             + [f"synth-{i}" for i in range(1, 8)])
SEEDS = [1, 2, 3]
DEFAULT_COST = 1200.0


def cost_model(folder):
    """Measured seconds per run, per instance, from the four-operator run."""
    path = os.path.join(folder, "benchmark-hill_climbing.csv")
    if not os.path.exists(path):
        return {i: DEFAULT_COST for i in INSTANCES}
    seen = defaultdict(list)
    for r in csv.DictReader(open(path)):
        seen[r["instance"]].append(float(r["seconds"]))
    return {i: (statistics.mean(seen[i]) if seen.get(i) else DEFAULT_COST)
            for i in INSTANCES}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else HERE
    cost = cost_model(folder)
    now = dt.datetime.now()

    print(f"{'process':<18}{'done':>7}{'left':>7}{'remaining':>13}{'finishes':>12}")
    print("-" * 57)

    worst = 0.0
    for op in OPS:
        path = os.path.join(folder, f"ablation-{op}.csv")
        done = set()
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                done.add((r["instance"], int(r["seed"])))
        todo = [(i, s) for i in INSTANCES for s in SEEDS if (i, s) not in done]
        left = sum(cost[i] for i, _ in todo)
        worst = max(worst, left)
        eta = now + dt.timedelta(seconds=left)
        h, m = divmod(int(left // 60), 60)
        print(f"{op:<18}{len(done):>4}/30{len(todo):>7}{h:>9}h {m:02d}m"
              f"{eta.strftime('%a %H:%M'):>12}")

    h, m = divmod(int(worst // 60), 60)
    eta = now + dt.timedelta(seconds=worst)
    print("-" * 57)
    print(f"all five finish in about {h}h {m:02d}m -- "
          f"{eta.strftime('%A %d %b, %H:%M')}")
    print("\n(estimate assumes the same per-instance wall-clock as the "
          "four-operator run;\nsynth-6 alone averages 2,874 s against its "
          "1,200 s budget and dominates the tail)")


if __name__ == "__main__":
    main()

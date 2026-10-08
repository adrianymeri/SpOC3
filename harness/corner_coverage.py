#!/usr/bin/env python3
"""
corner_coverage.py -- why some operators accept nothing, measured without search.

Standalone and read-only with respect to everything else: it imports
hill_climbing, torso and operators_more, runs no climb, and writes its own
CSV. Safe to run while ablation_more.py is working.

THE ARGUMENT
------------
Eliminating a SET of vertices leaves a torso that does not depend on the
order within that set, so deg[i] is a function only of the vertex at i and
the SET of vertices below i. A move rearranging a window [lo, hi] therefore
cannot change deg[i] for any i outside that window.

Which positions matter? Solution.staircase() sweeps t backwards keeping a
running maximum, and records a point each time the maximum steps up. So
deg[i] affects the staircase exactly when

        deg[i] > max(deg[i+1:])

and those positions -- the staircase corners -- are the only ones whose
degree can set best_t_for_width at ANY target width.

Put the two together: a move whose window contains no corner leaves the
whole staircase unchanged, and therefore cannot improve the cost at any
target. It can only be neutral or worse. Measuring how often each
operator's window covers a corner is therefore an UPPER BOUND on how often
that operator can possibly help -- computable with no search, no budget and
no tuning.

That turns the ablation's zero-accept rows from an observation into an
explanation. reverse_segment disturbs 2-10 positions out of n; it covers a
corner only rarely, which is the mechanism behind swap_neighbours being
measured at 1 accepted move in roughly 1.5M attempts.

    python3 corner_coverage.py --samples 2000
    python3 corner_coverage.py --samples 500 --start random

Output: corner_coverage.csv, one row per (instance, operator).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from bench_one import INSTANCES, ROOT                               # noqa: E402
from torso import Graph, Solution                                   # noqa: E402
import hill_climbing                                                # noqa: E402
import operators_more                                               # noqa: E402

FIELDS = ["instance", "n", "operator", "family", "samples", "moved",
          "covers_corner", "coverage", "median_span", "mean_span",
          "width", "corners", "first_corner", "first_corner_rel"]


def corners(deg):
    """Positions where the running maximum from the right steps up."""
    out = []
    run = -1
    for i in range(len(deg) - 1, -1, -1):
        if deg[i] > run:
            run = deg[i]
            out.append(i)
    return out


def changed_window(a, b):
    n = len(a)
    lo = 0
    while lo < n and a[lo] == b[lo]:
        lo += 1
    if lo == n:
        return None, None
    hi = n - 1
    while hi > lo and a[hi] == b[hi]:
        hi -= 1
    return lo, hi


def main():
    ap = argparse.ArgumentParser(
        description="Per-operator upper bound on usefulness, no search.")
    ap.add_argument("--samples", type=int, default=2000,
                    help="moves sampled per operator per instance")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="min_degree",
                    choices=["min_degree", "random"],
                    help="construction to measure against; whether an "
                         "operator can do anything may depend on it")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    operators_more.install()
    names = list(hill_climbing.Operators.NAMES)
    out = a.out or os.path.join(HERE, "corner_coverage.csv")

    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()

    print(f"{len(names)} operators, {a.samples} samples each, "
          f"start '{a.start}'\n")

    for name in INSTANCES:
        path = os.path.join(ROOT, a.data_dir, f"{name}.gr")
        if not os.path.exists(path):
            print(f"  MISSING {path}", flush=True)
            continue
        graph = Graph.load(path)
        operators_more.bind(graph)
        rng = random.Random(a.seed)
        perm = (hill_climbing.Starts.random_order(graph, rng)
                if a.start == "random"
                else hill_climbing.Starts.min_degree(graph, rng))
        deg = Solution(graph, perm).degrees()
        spots = set(corners(deg))
        first = min(spots) if spots else -1

        print(f"  {name}: n={graph.n}, width {max(deg)}, "
              f"{len(spots)} staircase corners, earliest at {first} "
              f"(rel {first / graph.n:.2f})")
        print(f"  {'operator':<20}{'family':<11}{'can-help':>10}"
              f"{'median span':>13}")

        for op in names:
            fn = hill_climbing.Operators.BY_NAME[op]
            covered = moved = 0
            spans = []
            for _ in range(a.samples):
                cand = fn(perm, rng)
                lo, hi = changed_window(perm, cand)
                if lo is None:
                    continue          # the move was a no-op
                moved += 1
                spans.append(hi - lo + 1)
                # does the window contain any corner?
                if any(lo <= c <= hi for c in spots):
                    covered += 1
            cov = covered / moved if moved else 0.0
            med = statistics.median(spans) if spans else 0
            mean = statistics.mean(spans) if spans else 0
            fam = operators_more.FAMILY.get(op, "original")
            print(f"  {op:<20}{fam:<11}{cov:>9.1%}{med:>13,.0f}")
            w.writerow({"instance": name, "n": graph.n, "operator": op,
                        "family": fam, "samples": a.samples, "moved": moved,
                        "covers_corner": covered, "coverage": round(cov, 6),
                        "median_span": med, "mean_span": round(mean, 1),
                        "width": max(deg), "corners": len(spots),
                        "first_corner": first,
                        "first_corner_rel": round(first / graph.n, 4)})
            f.flush()
        print()

    f.close()
    print(f"DONE -> {out}")
    print("\n'can-help' is an upper bound: a move whose window contains no")
    print("staircase corner leaves the staircase unchanged and so cannot")
    print("lower the cost at any target width. 'moved' excludes no-ops.")


if __name__ == "__main__":
    main()

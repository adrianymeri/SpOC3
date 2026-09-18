#!/usr/bin/env python3
"""
validate.py -- check a submission is legal and score it.

Scoring goes through esa_eval.py, the evaluator ported from the main
project, so the number this prints is the number the competition would use.

Every vector is also re-checked with a plain set-based walk written here.
That second opinion shares no code with esa_eval, so if the two ever
disagree you hear about it instead of quietly trusting one of them.

    python3 validate.py --instance data/toy.gr --submission out/toy.json
"""

from __future__ import annotations

import argparse
import json
import sys

from esa_eval import MAX_TW, build_adj_bitsets, evaluate, hypervolume_2d
from torso import Graph

MAX_VECTORS = 20


def slow_check(graph, perm, t):
    """Independent set-based recomputation of (width, t)."""
    position = {v: i for i, v in enumerate(perm)}
    work = [set(a) for a in graph.adj]
    width = 0
    capped = False
    for i, u in enumerate(perm):
        survivors = {v for v in work[u] if position[v] > i}
        degree = len(survivors)
        if degree > MAX_TW:
            capped = True                      # the cap applies at every step
        if i >= t and degree > width:
            width = degree
        for v in survivors:
            work[v] |= survivors - {v}
    return (MAX_TW + 1 if capped else width), t


def main():
    ap = argparse.ArgumentParser(description="Validate a torso submission.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip the set-based second opinion (it is slow on "
                         "large instances)")
    args = ap.parse_args()

    graph = Graph.load(args.instance)
    n = graph.n
    bits = build_adj_bitsets(n, graph.adj)

    payload = json.load(open(args.submission))
    if isinstance(payload, list):
        payload = payload[0]
    vectors = payload["decisionVector"]

    print(f"=== validating {args.submission} ===")
    print(f"instance: {graph.describe()}")
    print(f"vectors:  {len(vectors)}")
    print()

    problems = []
    points = []

    if len(vectors) > MAX_VECTORS:
        problems.append(f"{len(vectors)} vectors, limit is {MAX_VECTORS}")

    for index, vector in enumerate(vectors):
        label = f"vector {index}"

        if len(vector) != n + 1:
            problems.append(f"{label}: length {len(vector)}, expected {n + 1}")
            continue

        perm = [int(x) for x in vector[:-1]]
        t = int(vector[-1])

        if sorted(perm) != list(range(n)):
            problems.append(f"{label}: not a permutation of 0..{n - 1}")
            continue
        if not 0 <= t < n:
            problems.append(f"{label}: threshold {t} outside [0, {n})")
            continue

        width, t = evaluate(perm, t, bits, n)

        if not args.skip_slow:
            if slow_check(graph, perm, t) != (width, t):
                problems.append(f"{label}: the two evaluators disagree -- "
                                f"this is a bug, not a bad answer")

        if width > MAX_TW:
            problems.append(f"{label}: width {width} exceeds the {MAX_TW} "
                            f"cap -- solution is void")
            continue

        points.append((width, t))
        if not args.quiet:
            print(f"  {label:>10}: width {width:4d}, t {t:6d}, "
                  f"torso {n - t:6d}   OK")

    duplicates = len(points) - len(set(points))
    if duplicates:
        problems.append(f"{duplicates} duplicate point(s)")

    dominated = sum(1 for a in points
                    if any(b != a and b[0] <= a[0] and b[1] <= a[1]
                           for b in points))
    if dominated:
        problems.append(f"{dominated} dominated point(s) "
                        f"(wasting submission slots)")

    print()
    if problems:
        print("FAILED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("all checks passed")

    if points:
        score = -int(hypervolume_2d(points, n))
        print()
        print(f"score: {int(score):,}")
        print("(negative area against the corner (n, n); "
              "more negative is better)")
        if "score" in payload:
            claimed = int(payload["score"])
            if claimed == score:
                print(f"claimed by the submission: {claimed:,}  -> AGREES")
            else:
                print(f"claimed by the submission: {claimed:,}  -> DISAGREES")
                problems.append("claimed score does not match recomputation")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
validate.py -- check a submission is legal, and score it independently.

This deliberately re-implements the checks from scratch rather than
importing anything from hill_climbing.py. If the search has a bug that
makes it believe a bad answer is good, a validator that shared its code
would believe it too.

What it checks, per decision vector:

    1. the right length            (n vertices + 1 threshold)
    2. a genuine permutation       (every vertex 0..n-1 exactly once)
    3. a threshold in range        (0 <= t < n)
    4. the width cap               (no elimination step wider than 500)

and across the whole submission:

    5. at most 20 vectors          (the competition limit)
    6. no duplicate or dominated points
    7. the total score, recomputed from the graph

Exit code 0 if everything passes, 1 if anything fails.

    python3 validate.py --instance data/toy.gr --submission out/toy.json

Pure Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys

from torso import Graph, MAX_WIDTH


MAX_VECTORS = 20


def evaluate_independently(graph, perm, t):
    """Recompute (width, t) from first principles, with sets.

    Written the slow, obvious way on purpose: this is the reference the
    search is being checked against, so it should be the version that is
    easiest to confirm by eye.
    """
    n = graph.n
    position = {v: i for i, v in enumerate(perm)}
    work = [set(a) for a in graph.adj]

    width = 0
    capped = False
    for i, u in enumerate(perm):
        survivors = {v for v in work[u] if position[v] > i}
        degree = len(survivors)
        if degree > MAX_WIDTH:
            capped = True                  # cap applies at EVERY step
        if i >= t and degree > width:
            width = degree
        for v in survivors:
            work[v] |= survivors - {v}

    return (MAX_WIDTH + 1 if capped else width), t


def area(points, n):
    """Area dominated by these points relative to (n, n)."""
    valid = sorted((w, t) for (w, t) in points if w < n and t < n)
    frontier = []
    best_t = n
    for w, t in valid:
        if t < best_t:
            frontier.append((w, t))
            best_t = t
    total = 0
    for i, (w, t) in enumerate(frontier):
        next_w = frontier[i + 1][0] if i + 1 < len(frontier) else n
        total += (next_w - w) * (n - t)
    return total


def main():
    ap = argparse.ArgumentParser(description="Validate a torso submission.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    graph = Graph.load(args.instance)
    n = graph.n

    with open(args.submission) as f:
        payload = json.load(f)
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

        perm, t = [int(x) for x in vector[:-1]], int(vector[-1])

        if sorted(perm) != list(range(n)):
            problems.append(f"{label}: not a permutation of 0..{n - 1}")
            continue

        if not (0 <= t < n):
            problems.append(f"{label}: threshold {t} outside [0, {n})")
            continue

        width, t = evaluate_independently(graph, perm, t)

        if width > MAX_WIDTH:
            problems.append(f"{label}: width {width} exceeds the "
                            f"{MAX_WIDTH} cap -- solution is void")
            continue

        points.append((width, t))
        if not args.quiet:
            print(f"  {label:>10}: width {width:4d}, t {t:6d}, "
                  f"torso {n - t:6d}   OK")

    duplicates = len(points) - len(set(points))
    if duplicates:
        problems.append(f"{duplicates} duplicate point(s)")

    dominated = 0
    for a in points:
        for b in points:
            if b != a and b[0] <= a[0] and b[1] <= a[1]:
                dominated += 1
                break
    if dominated:
        problems.append(f"{dominated} dominated point(s) "
                        f"(wasting submission slots)")

    print()
    if problems:
        print("FAILED:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("all checks passed")

    if points:
        score = -area(points, n)
        print()
        print(f"independently recomputed score: {score:,}")
        print("(negative area against the corner (n, n); "
              "more negative is better)")
        if "score" in payload:
            claimed = int(payload["score"])
            agree = "AGREES" if claimed == score else "DISAGREES"
            print(f"score claimed by the submission:  {claimed:,}  -> {agree}")
            if claimed != score:
                problems.append("claimed score does not match recomputation")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

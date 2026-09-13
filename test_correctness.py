#!/usr/bin/env python3
"""
test_correctness.py -- prove the code is right before trusting any score.

Everything in this project rests on two calculations being correct:

    the evaluator   (perm, t)  ->  width
    the hypervolume  points    ->  area

If either is wrong by one unit, every number this project prints is wrong.
So neither is taken on trust. Each is pinned against an independent
ground truth computed a completely different way.

    1. hand-worked toy example       the README's Part 2 table, typed in
                                     by hand and checked line by line
    2. fast vs slow evaluator        bitsets vs plain sets, random orders
    3. hypervolume vs brute force    the strip formula vs literally
                                     counting covered grid squares
    4. staircase consistency         every point it claims is re-checked
                                     by evaluating that t directly
    5. validator rejects bad input   five kinds of malformed submission

Run it:

    python3 test_correctness.py

Exit code 0 means everything passed.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

from torso import Graph, Solution, hypervolume, MAX_WIDTH


passed = 0
failed = 0


def check(name, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")
        print(f"          got  {got}")
        print(f"          want {want}")


# ===========================================================================
# 1. The hand-worked toy example from README Part 2
# ===========================================================================

def test_hand_worked_example():
    print("\n1. hand-worked toy example (README Part 2)")

    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (6, 8), (7, 8), (7, 9), (8, 9), (9, 10), (10, 11), (0, 11)]
    adj = [set() for _ in range(12)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    graph = Graph(12, adj)

    check("toy has 16 edges", graph.edge_count, 16)

    # the order traced by hand in the README
    hand = Solution(graph, [5, 4, 11, 10, 0, 1, 2, 3, 9, 6, 7, 8])
    check("hand-traced deg[]", hand.degrees(),
          [2, 2, 2, 2, 3, 3, 2, 2, 3, 2, 1, 0])
    check("hand-traced staircase", hand.staircase(),
          [(0, 11), (1, 10), (2, 9), (3, 0)])
    check("hand-traced score", -hypervolume(hand.staircase(), 12), -114)

    # the better order the climber finds, also quoted in the README
    climbed = Solution(graph, [7, 1, 8, 2, 9, 5, 0, 3, 10, 6, 11, 4])
    check("climber deg[]", climbed.degrees(),
          [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0])
    check("climber staircase", climbed.staircase(),
          [(0, 11), (1, 10), (2, 2), (3, 0)])
    check("climber score", -hypervolume(climbed.staircase(), 12), -121)


# ===========================================================================
# 2. Fast evaluator vs slow evaluator
# ===========================================================================

def test_fast_matches_slow():
    print("\n2. bitset evaluator vs plain-set evaluator")

    rng = random.Random(1)
    for path, trials in [("data/toy.gr", 40), ("data/small-graph.gr", 6)]:
        if not os.path.exists(path):
            print(f"  SKIP  {path} not present")
            continue
        graph = Graph.load(path)
        agree = 0
        for _ in range(trials):
            perm = list(range(graph.n))
            rng.shuffle(perm)
            if Solution(graph, perm).check():
                agree += 1
        check(f"{os.path.basename(path)}: {trials} random orders agree",
              agree, trials)


# ===========================================================================
# 3. Hypervolume vs brute force
# ===========================================================================

def brute_force_area(points, n):
    """Count covered grid squares one at a time. Unarguably correct,
    and far too slow for anything but a tiny n -- which is the point."""
    covered = 0
    for x in range(n):
        for y in range(n):
            for (w, t) in points:
                if w <= x and t <= y:
                    covered += 1
                    break
    return covered


def test_hypervolume_vs_brute_force():
    print("\n3. hypervolume formula vs brute-force grid count")

    rng = random.Random(2)
    n = 14
    agree = 0
    trials = 120
    for _ in range(trials):
        pts = [(rng.randrange(n), rng.randrange(n))
               for _ in range(rng.randint(1, 6))]
        if hypervolume(pts, n) == brute_force_area(pts, n):
            agree += 1
    check(f"{trials} random fronts agree", agree, trials)

    # edge cases
    check("empty front", hypervolume([], 10), 0)
    check("point at the origin covers everything", hypervolume([(0, 0)], 10), 100)
    check("point on the reference covers nothing",
          hypervolume([(10, 10)], 10), 0)
    check("duplicates counted once",
          hypervolume([(2, 3), (2, 3)], 10), hypervolume([(2, 3)], 10))
    check("dominated point adds nothing",
          hypervolume([(2, 3), (5, 7)], 10), hypervolume([(2, 3)], 10))


# ===========================================================================
# 4. The staircase really is what it claims
# ===========================================================================

def test_staircase_is_consistent():
    print("\n4. every staircase point re-checked by direct evaluation")

    rng = random.Random(3)
    graph = Graph.load("data/toy.gr")
    problems = 0
    for _ in range(40):
        perm = list(range(graph.n))
        rng.shuffle(perm)
        solution = Solution(graph, perm)
        for width, t in solution.staircase():
            # evaluating that t directly must give exactly this width
            if solution.width_at(t) != width:
                problems += 1
            # and t must be the SMALLEST one giving that width
            if t > 0 and solution.width_at(t - 1) <= width:
                problems += 1
    check("no inconsistent staircase points", problems, 0)


# ===========================================================================
# 5. The validator rejects malformed submissions
# ===========================================================================

def run_validator(graph_path, payload):
    """Call validate.py as a subprocess; return its exit code."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        name = f.name
    try:
        result = subprocess.run(
            [sys.executable, "validate.py", "--instance", graph_path,
             "--submission", name, "--quiet"],
            capture_output=True, text=True)
        return result.returncode
    finally:
        os.unlink(name)


def test_validator_catches_bad_input():
    print("\n5. validator rejects malformed submissions")

    path = "data/toy.gr"
    graph = Graph.load(path)
    n = graph.n
    good_perm = list(range(n))

    check("accepts a legal submission",
          run_validator(path, {"decisionVector": [good_perm + [3]]}), 0)

    check("rejects a repeated vertex",
          run_validator(path, {"decisionVector": [[0] * n + [3]]}), 1)

    check("rejects the wrong length",
          run_validator(path, {"decisionVector": [good_perm[:-1] + [3]]}), 1)

    check("rejects an out-of-range threshold",
          run_validator(path, {"decisionVector": [good_perm + [n + 5]]}), 1)

    check("rejects more than 20 vectors",
          run_validator(path, {"decisionVector":
                               [good_perm + [i] for i in range(21)]}), 1)

    check("rejects a wrong claimed score",
          run_validator(path, {"score": -999999,
                               "decisionVector": [good_perm + [3]]}), 1)


# ===========================================================================

def main():
    print("=" * 62)
    print("correctness checks")
    print("=" * 62)

    test_hand_worked_example()
    test_fast_matches_slow()
    test_hypervolume_vs_brute_force()
    test_staircase_is_consistent()
    test_validator_catches_bad_input()

    print()
    print("=" * 62)
    print(f"{passed} passed, {failed} failed")
    print("=" * 62)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

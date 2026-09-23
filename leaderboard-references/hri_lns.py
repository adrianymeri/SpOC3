#!/usr/bin/env python3
"""
hri_lns.py -- Team HRI's method: multi-objective Large Neighbourhood Search.

HRI placed second on the SpOC-3 leaderboard. No source was published, so this
is written from their paper (`spoc_2023_team_hri.pdf`) and from S. Limmer's
description of the operator set, cited with permission in the main project.

The method is destroy-and-repair rather than mutate-and-test:

  operator set B (the primary one, run until it stops making progress)
      neighbour destroy   pick a seed vertex, remove it and its neighbours
                          from the ordering -- a large, structured hole
      balanced repair     reinsert each removed vertex at the MEDIAN position
                          of its already-placed neighbours

  operator set A (the fallback once B stalls)
      random destroy      remove a few vertices at random
      random repair       put them back at random positions

The median-placement rule is the interesting part and it is not arbitrary: it
is the balanced-insertion rule from Biedl et al., *Discrete Applied
Mathematics* 148 (2005), Section 5. Placing a vertex at the median of its
neighbours keeps roughly as many of them before it as after, which is what
keeps the elimination width down.

Scoring goes through ../esa_eval.py, the same evaluator every other solver
here uses.

    python3 hri_lns.py --instance ../data/small-graph.gr --seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torso import Graph, Solution, Front


def neighbour_destroy(perm, adj, rng, cap):
    """Remove a seed vertex and its neighbours. Capped on dense graphs."""
    v = perm[rng.randrange(len(perm))]
    hole = {v} | set(adj[v])
    if len(hole) > cap:
        hole = {v} | set(rng.sample(sorted(adj[v]), cap - 1))
    return hole


def balanced_repair(perm, destroyed, adj, rng):
    """Reinsert each removed vertex at the median of its placed neighbours.

    Biedl et al. 2005, Section 5. For an even number of placed neighbours any
    slot between the two middle ones is equally good; for an odd number we
    put it on whichever side of the median neighbour leaves that neighbour
    better balanced.
    """
    base = [u for u in perm if u not in destroyed]
    pos = {u: i for i, u in enumerate(base)}

    # densest first: they have the most constraints, so place them early
    for u in sorted(destroyed, key=lambda u: -len(adj[u])):
        placed = sorted(pos[x] for x in adj[u] if x in pos)
        k = len(placed)
        if k == 0:
            j = rng.randint(0, len(base))
        elif k % 2 == 0:
            j = placed[k // 2]
        else:
            mid = placed[k // 2]
            w = base[mid]
            wn = [pos[x] for x in adj[w] if x in pos]
            before = sum(1 for x in wn if x < mid)
            after = len(wn) - before
            j = mid if before + 1 - after <= after + 1 - before else mid + 1
        base.insert(min(j, len(base)), u)
        pos = {x: i for i, x in enumerate(base)}
    return base


def random_destroy_repair(perm, rng, k):
    """Operator set A: pull out k random vertices, put them back anywhere."""
    p = list(perm)
    idx = sorted(rng.sample(range(len(p)), k), reverse=True)
    for u in [p.pop(i) for i in idx]:
        p.insert(rng.randint(0, len(p)), u)
    return p


def solve(graph, seconds, seed=1, destroy_cap=220, small_k=6, stall=3000,
          verbose=False, start="random"):
    """Run the LNS and return (front, iterations, accepts).

    start="random" follows the paper. start="min_degree" gives it the same
    constructive start hill_climbing uses, which is what benchmark.py does:
    at equal budget the starting point is worth more than either search, so
    comparing the searches means equalising it first.

    Note that on the larger instances a random start is not merely slow, it
    is void -- a random permutation busts the 500 cap everywhere, so the LNS
    accepts nothing and scores 0. See README.md 10.4.
    """
    rng = random.Random(seed)
    n = graph.n
    adj = [sorted(a) for a in graph.adj]
    front = Front(n)

    if start == "min_degree":
        import hill_climbing
        perm = hill_climbing.Starts.min_degree(graph, rng)
    else:
        perm = list(range(n))
        rng.shuffle(perm)
    current = Solution(graph, perm)
    front.add_solution(current)
    best = front.score()

    mode = "B"
    since = 0
    iterations = accepts = 0
    deadline = time.time() + seconds

    while time.time() < deadline:
        iterations += 1
        if mode == "B":
            hole = neighbour_destroy(current.perm, adj, rng, destroy_cap)
            candidate_perm = balanced_repair(current.perm, hole, adj, rng)
        else:
            candidate_perm = random_destroy_repair(current.perm, rng, small_k)

        candidate = Solution(graph, candidate_perm)
        front.add_solution(candidate)
        since += 1

        # accept on improvement of the whole front, which is what HRI's
        # multi-objective formulation optimises
        score = front.score()
        if score < best:
            best = score
            current = candidate
            accepts += 1
            since = 0
            if verbose:
                print(f"  it {iterations}: {score:,}")

        if since >= stall:
            mode = "A" if mode == "B" else "B"
            since = 0
            if verbose:
                print(f"  [stall -> operator set {mode}]")

    return front, iterations, accepts


def main():
    ap = argparse.ArgumentParser(description="Team HRI's LNS.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--destroy-cap", type=int, default=220)
    ap.add_argument("--small-k", type=int, default=6)
    ap.add_argument("--stall", type=int, default=3000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    graph = Graph.load(a.instance)
    print(f"=== HRI LNS on {os.path.basename(a.instance)} ===")
    print(graph.describe())
    t0 = time.time()
    front, iterations, accepts = solve(graph, a.seconds, a.seed,
                                       a.destroy_cap, a.small_k, a.stall,
                                       verbose=True)
    print(f"\n{iterations:,} iterations, {accepts:,} accepted, "
          f"{time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"instance": os.path.basename(a.instance),
                       "solver": "hri_lns", "n": graph.n,
                       "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

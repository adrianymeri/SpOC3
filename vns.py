#!/usr/bin/env python3
"""
vns.py -- Variable Neighbourhood Search: when stuck, kick harder.

Mladenovic and Hansen (1997), "Variable neighborhood search", Computers &
Operations Research 24(11): 1097-1100.

Hill climbing gets stuck because its moves are small: once no single swap
helps, it is finished. VNS keeps the same small moves but adds a ladder. It
holds an index k and, each round:

    1. shake     -- apply k * strength random moves to the incumbent. A
                    bigger k is a more violent kick.
    2. descend   -- run a short local search from wherever that landed
    3. move or not -- if the front improved, take the new solution and reset
                    k to 1. If it did not, keep the old one and try k + 1.

So the perturbation grows exactly while progress stalls, and collapses back
the moment something works. That is the whole idea: one incumbent, one
ladder, no population.

The descent accepts a move only if it improves the **front's hypervolume**,
not the width at a fixed target. That is a stronger criterion than hill
climbing's and it is what the original uses.

Every candidate evaluated donates its whole staircase to the shared front,
and the score comes from esa_eval.py like everything else here.

    python3 vns.py --instance data/small-graph.gr --seconds 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from torso import Graph, Solution, Front
import meta
import hill_climbing


def shake(graph, perm, t, rng, moves):
    """The k-th neighbourhood: `moves` random moves from the shared pool."""
    for _ in range(max(1, moves)):
        perm, t = meta.move(graph, perm, t, rng)
    return perm, t


def descend(graph, front, perm, t, rng, steps, deadline):
    """Short descent on the (perm, t) energy. Returns (perm, t, energy)."""
    n = graph.n
    solution = Solution(graph, perm)
    front.add_solution(solution)
    best = meta.energy(solution, t, n)
    for _ in range(steps):
        if time.time() >= deadline:
            break
        cand_perm, cand_t = meta.move(graph, perm, t, rng, solution)
        candidate = Solution(graph, cand_perm)
        front.add_solution(candidate)
        e = meta.energy(candidate, cand_t, n)
        if e < best:                      # lower energy = more area dominated
            best, perm, t, solution = e, cand_perm, cand_t, candidate
    return perm, t, best


def solve(graph, seconds, seed=1, k_max=5, strength=1, ls_steps=400,
          start="min_degree", verbose=False):
    """Run VNS and return (front, rounds, improvements)."""
    rng = random.Random(seed)
    front = Front(graph.n)

    perm, t, solution = meta.start_state(graph, rng, start)
    front.add_solution(solution)
    best = meta.energy(solution, t, graph.n)

    k = 1
    rounds = improvements = 0
    deadline = time.time() + seconds

    while time.time() < deadline:
        rounds += 1
        kp, kt = shake(graph, perm, t, rng, k * strength)
        lp, lt, e = descend(graph, front, kp, kt, rng, ls_steps, deadline)

        if e < best:
            # the kick paid off: adopt it and go back to gentle moves
            best, perm, t = e, lp, lt
            improvements += 1
            k = 1
        else:
            # no progress: kick harder next time, wrapping at the top
            k = k + 1 if k < k_max else 1

        if verbose and rounds % 20 == 0:
            print(f"  round {rounds:>6,}  k={k}  {front.score():,}")

    return front, rounds, improvements


# --- front-aware variant -------------------------------------------------
#
# Hill Climbing's skeleton -- eight target widths, a min-degree construction
# per width, the same four generic operators, minimise t at the target width
# -- with the shake/descend/ladder acceptance rule in place of plain greedy
# acceptance. That one substitution is the whole difference.

def _vns_at_width(graph, front, target_width, seconds, rng, k_max, strength,
                  ls_steps):
    n = graph.n
    perm = hill_climbing.Starts.min_degree(graph, rng)
    best_sol = Solution(graph, perm)
    front.add_solution(best_sol)
    best = meta.cost_at(best_sol, target_width, n)

    k = 1
    rounds = improvements = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        rounds += 1
        # shake: k * strength random moves off the incumbent
        kicked = list(perm)
        for _ in range(max(1, k * strength)):
            kicked = rng.choice(hill_climbing.Operators.ALL)(kicked, rng)
        sol = Solution(graph, kicked)
        front.add_solution(sol)
        cost = meta.cost_at(sol, target_width, n)
        # descend: accept only improvements
        for _ in range(ls_steps):
            if time.time() >= deadline:
                break
            cand_perm = rng.choice(hill_climbing.Operators.ALL)(kicked, rng)
            cand = Solution(graph, cand_perm)
            front.add_solution(cand)
            c = meta.cost_at(cand, target_width, n)
            if c < cost:
                kicked, cost = cand_perm, c
        if cost < best:
            perm, best = kicked, cost      # the kick paid off
            improvements += 1
            k = 1
        else:
            k = k + 1 if k < k_max else 1  # kick harder next time
    return rounds, improvements


def solve_front(graph, seconds, seed=1, widths=8, k_max=5, strength=1,
                ls_steps=400, verbose=False):
    """Front-aware VNS: Hill Climbing's skeleton, shake-ladder acceptance.
    Returns (front, rounds, improvements)."""
    rng = random.Random(seed)
    front = Front(graph.n)
    targets = hill_climbing.target_widths(graph, front, widths, "min_degree")
    per_width = seconds / len(targets)
    rounds = improvements = 0
    for w in targets:
        r, i = _vns_at_width(graph, front, w, per_width, rng, k_max, strength,
                             ls_steps)
        rounds += r
        improvements += i
        if verbose:
            print(f"  width {w:>4}: {r:,} rounds, {i} improved, "
                  f"{front.score():,}", flush=True)
    return front, rounds, improvements


def main():
    ap = argparse.ArgumentParser(description="Variable Neighbourhood Search.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=5,
                    help="how far the ladder goes before wrapping")
    ap.add_argument("--strength", type=int, default=1,
                    help="operator moves per ladder rung")
    ap.add_argument("--ls-steps", type=int, default=400,
                    help="descent steps after each shake")
    ap.add_argument("--start", default="min_degree",
                    choices=["min_degree", "random"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    graph = Graph.load(a.instance)
    print(f"=== VNS on {os.path.basename(a.instance)} ===")
    print(graph.describe())
    t0 = time.time()
    front, rounds, improvements = solve(graph, a.seconds, a.seed, a.k_max,
                                        a.strength, a.ls_steps, a.start,
                                        verbose=True)
    print(f"\n{rounds:,} rounds, {improvements:,} improved the front, "
          f"{time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"instance": os.path.basename(a.instance),
                       "solver": "vns", "n": graph.n, "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
grasp.py -- Greedy Randomised Adaptive Search Procedure: build, improve, repeat.

Feo and Resende (1995), "Greedy randomized adaptive search procedures",
Journal of Global Optimization 6(2): 109-133.

Hill climbing builds one starting solution and then spends the whole budget
improving it. GRASP does the opposite: it builds many, improves each a
little, and keeps whatever the best of them found. Each restart has two
phases.

**Construction.** Build an elimination order the way min-degree does -- take
the lowest-degree vertex, eliminate it, repeat -- except that instead of
always taking the minimum we take a random vertex from the Restricted
Candidate List, every vertex whose degree falls in

    [d_min,  d_min + alpha * (d_max - d_min)]

alpha = 0 is exactly min-degree, deterministic. alpha = 1 is a uniformly
random order. In between you get orders that are min-degree-ish but all
different, which is the point: hill climbing's eight restarts differ only by
tie-breaking, while these differ by construction.

**Local search.** A short descent from the constructed order, accepting a
move only if it grows the front's hypervolume.

Every restart feeds the same front, so the score is the union over all of
them -- and, as everywhere else here, it is computed by esa_eval.py.

    python3 grasp.py --instance data/small-graph.gr --seconds 60
    python3 grasp.py --instance data/small-graph.gr --seconds 60 --alpha 0.0
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


def construct(graph, rng, alpha):
    """Min-degree elimination, but choosing from a Restricted Candidate List.

    Same incremental degree table as hill_climbing.Starts.min_degree -- the
    only change is which vertex gets picked at each step.
    """
    n = graph.n
    alive = set(range(n))
    work = [set(a) for a in graph.adj]
    degree = {v: len(work[v]) for v in range(n)}
    order = []

    while alive:
        d_min = min(degree[v] for v in alive)
        d_max = max(degree[v] for v in alive)
        cutoff = d_min + alpha * (d_max - d_min)
        rcl = [v for v in alive if degree[v] <= cutoff]
        v = rng.choice(rcl)

        order.append(v)
        alive.discard(v)
        survivors = work[v] & alive
        for u in survivors:
            work[u] |= survivors - {u}
        for u in survivors:
            degree[u] = len(work[u] & alive)

    return order


def descend(graph, front, perm, rng, deadline):
    """Descent on the (perm, t) energy from a constructed order."""
    n = graph.n
    solution = Solution(graph, perm)
    front.add_solution(solution)
    stairs = solution.staircase()
    t = max(stairs, key=lambda wt: (n - wt[0]) * (n - wt[1]))[1] if stairs else 0
    best = meta.energy(solution, t, n)
    steps = 0
    while time.time() < deadline:
        steps += 1
        cand_perm, cand_t = meta.move(graph, perm, t, rng, solution)
        candidate = Solution(graph, cand_perm)
        front.add_solution(candidate)
        e = meta.energy(candidate, cand_t, n)
        if e < best:
            best, perm, t, solution = e, cand_perm, cand_t, candidate
    return steps


# Rotated over the restarts. 0.0 is pure min-degree, so at least one restart
# reproduces the greedy construction exactly -- without it GRASP can finish
# *below* the baseline it is built on, which it did at a fixed alpha of 0.3.
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]


def solve(graph, seconds, seed=1, alpha=None, restarts=8, verbose=False):
    """Run GRASP and return (front, restarts_done, descent_steps).

    alpha=None rotates through ALPHAS; a number fixes it.
    """
    rng = random.Random(seed)
    front = Front(graph.n)

    start = time.time()
    deadline = start + seconds
    per_restart = seconds / max(1, restarts)

    done = steps = 0
    while time.time() < deadline and done < restarts:
        a_r = ALPHAS[done % len(ALPHAS)] if alpha is None else alpha
        done += 1
        perm = construct(graph, rng, a_r)
        stop = min(deadline, start + done * per_restart)
        steps += descend(graph, front, perm, rng, stop)
        if verbose:
            print(f"  restart {done}/{restarts}  alpha={a_r}  "
                  f"{front.score():,}")

    return front, done, steps


# --- front-aware variant -------------------------------------------------
#
# Hill Climbing's skeleton with ONE substitution: the min-degree construction
# is replaced by a greedy-randomised one drawn from a restricted candidate
# list. Same eight target widths, same four operators, same greedy
# acceptance, same objective. So GRASP against Hill Climbing isolates the
# construction, exactly as this file's annealing and VNS counterparts isolate
# the acceptance rule.

def _grasp_at_width(graph, front, target_width, seconds, rng, alpha):
    n = graph.n
    perm = construct(graph, rng, alpha)
    sol = Solution(graph, perm)
    front.add_solution(sol)
    cost = meta.cost_at(sol, target_width, n)
    steps = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        steps += 1
        cand_perm = rng.choice(hill_climbing.Operators.ALL)(perm, rng)
        cand = Solution(graph, cand_perm)
        front.add_solution(cand)
        c = meta.cost_at(cand, target_width, n)
        if c < cost:
            perm, cost = cand_perm, c
    return steps


def solve_front(graph, seconds, seed=1, widths=8, alpha=None, verbose=False):
    """Front-aware GRASP: Hill Climbing's skeleton, RCL construction.
    alpha=None rotates through ALPHAS across the widths; 0.0 is pure
    min-degree, so at least one width reproduces the greedy construction.
    Returns (front, constructions, descent_steps)."""
    rng = random.Random(seed)
    front = Front(graph.n)
    targets = hill_climbing.target_widths(graph, front, widths, "min_degree")
    per_width = seconds / len(targets)
    steps = 0
    for i, w in enumerate(targets):
        a = ALPHAS[i % len(ALPHAS)] if alpha is None else alpha
        s = _grasp_at_width(graph, front, w, per_width, rng, a)
        steps += s
        if verbose:
            print(f"  width {w:>4}: alpha={a}, {s:,} steps, "
                  f"{front.score():,}", flush=True)
    return front, len(targets), steps


def main():
    ap = argparse.ArgumentParser(description="GRASP.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="fix the RCL greediness (0 = pure min-degree, "
                         "1 = uniformly random). Default rotates over "
                         f"{ALPHAS}.")
    ap.add_argument("--restarts", type=int, default=8,
                    help="constructions, each getting an equal time slice")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    graph = Graph.load(a.instance)
    print(f"=== GRASP on {os.path.basename(a.instance)} "
          f"(alpha={a.alpha if a.alpha is not None else ALPHAS}) ===")
    print(graph.describe())
    t0 = time.time()
    front, done, steps = solve(graph, a.seconds, a.seed, a.alpha, a.restarts,
                               verbose=True)
    print(f"\n{done} restarts, {steps:,} descent steps, {time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"instance": os.path.basename(a.instance),
                       "solver": "grasp", "n": graph.n, "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
simulated_annealing.py -- hill climbing that is allowed to go downhill.

Kirkpatrick, Gelatt and Vecchi (1983), "Optimization by simulated
annealing", Science 220(4598): 671-680.

The only difference from hill_climbing.py is the acceptance rule. Hill
climbing keeps a candidate if it is better and throws it away otherwise, so
it can never leave a local optimum. Simulated annealing keeps a worse
candidate too, with probability

    exp(-dE / T)

where dE is how much worse it is and T is a temperature that falls over the
run. Early on T is high and the walk roams; late on T is near zero and the
rule collapses back into plain hill climbing.

Energy
------
A solution's energy is the negative area it dominates against the reference
point (n, n):

    E = -(n - width) * (n - t)

taken over the best point of its staircase. Lower energy means a larger
dominated rectangle, which is what the score rewards. Width-cap violations
have no staircase at all and are given the worst possible energy.

T0 is calibrated rather than guessed: we sample random moves, measure how much
they typically worsen the energy, and set T0 so such a move is accepted with
probability about 0.4 at the start.

Three schedules, as in the original: geometric (T *= alpha), linear (T decays
to zero over the budget), and adaptive -- geometric plus a reheat whenever the
acceptance rate over a window falls below 5%. Adaptive is the default here
because a frozen walk stops feeding the front, and the front is where the
score comes from.

As in every other solver here, each candidate evaluated -- accepted or not --
donates its whole staircase to a shared front, and the score is the front's
hypervolume through esa_eval.py.

    python3 simulated_annealing.py --instance data/small-graph.gr --seconds 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from torso import Graph, Solution, Front
import meta
import hill_climbing


def calibrate_t0(graph, perm, t, rng, samples=80):
    """Pick T0 so a typical worsening move is accepted with p ~= 0.4."""
    n = graph.n
    base = meta.energy(Solution(graph, perm), t, n)
    worse = []
    for _ in range(samples):
        p, tt = meta.move(graph, perm, t, rng, Solution(graph, perm))
        d = meta.energy(Solution(graph, p), tt, n) - base
        if d > 0:
            worse.append(d)
    typical = sum(worse) / len(worse) if worse else (n * n) / 10.0
    return max(1.0, -typical / math.log(0.4))


def solve(graph, seconds, seed=1, t0=None, alpha=0.95, steps_per_t=200,
          start="min_degree", schedule="adaptive", verbose=False):
    """Run the anneal and return (front, iterations, accepts)."""
    rng = random.Random(seed)
    n = graph.n
    front = Front(n)

    perm, t, solution = meta.start_state(graph, rng, start)
    front.add_solution(solution)
    current_E = meta.energy(solution, t, n)

    T = calibrate_t0(graph, perm, t, rng) if t0 is None else float(t0)
    if verbose:
        print(f"  T0 = {T:,.0f}")

    iterations = accepts = 0
    win_att = win_acc = 0           # acceptance window, for reheating
    t_start = time.time()
    deadline = t_start + seconds
    T0 = T

    while time.time() < deadline:
        for _ in range(steps_per_t):
            if time.time() >= deadline:
                break
            iterations += 1
            cand_perm, cand_t = meta.move(graph, perm, t, rng, solution)
            candidate = Solution(graph, cand_perm)
            front.add_solution(candidate)

            win_att += 1
            cand_E = meta.energy(candidate, cand_t, n)
            dE = cand_E - current_E
            # downhill always; uphill sometimes, less and less as T falls
            if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                perm, t, current_E, solution = (cand_perm, cand_t, cand_E,
                                                candidate)
                accepts += 1
                win_acc += 1

        # --- cool down -------------------------------------------------
        if schedule == "geometric":
            T *= alpha
        elif schedule == "linear":
            left = (deadline - time.time()) / seconds        # 1 -> 0
            T = max(1e-6, T0 * max(0.0, left))
        else:                                                # adaptive
            # Geometric, but reheat when the walk has nearly frozen. On the
            # bigger instances plain geometric cooling stops accepting
            # anything long before the budget runs out, and a frozen walk
            # stops feeding the front -- which is where the score actually
            # comes from.
            T *= alpha
            if win_att >= 5 * steps_per_t:
                if win_acc / win_att < 0.05:
                    T = max(T, 0.5 * T0)
                win_att = win_acc = 0
        if verbose and iterations % (steps_per_t * 20) < steps_per_t:
            print(f"  it {iterations:>7,}  T {T:>11,.0f}  "
                  f"acc {accepts/max(iterations,1):5.1%}  {front.score():,}")

    return front, iterations, accepts


# --- front-aware variant -------------------------------------------------
#
# Identical skeleton to hill_climbing.solve: eight target widths, a fresh
# min-degree construction per width, the same four generic operators, and the
# same objective (minimise t at the target width). The ONLY difference from
# Hill Climbing is this file's acceptance rule. That is what makes the
# comparison a controlled one.

def _anneal_at_width(graph, front, target_width, seconds, rng, alpha,
                     steps_per_t):
    n = graph.n
    perm = hill_climbing.Starts.min_degree(graph, rng)
    current = Solution(graph, perm)
    front.add_solution(current)
    cur_cost = meta.cost_at(current, target_width, n)

    # Calibrate T0 so a typical worsening move is accepted at p ~= 0.4.
    worse = []
    for _ in range(60):
        cand = Solution(graph, rng.choice(hill_climbing.Operators.ALL)(perm, rng))
        d = meta.cost_at(cand, target_width, n) - cur_cost
        if d > 0:
            worse.append(d)
    typical = sum(worse) / len(worse) if worse else max(1.0, n / 100.0)
    T = T0 = max(1.0, -typical / math.log(0.4))

    iters = accepts = 0
    win_att = win_acc = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        for _ in range(steps_per_t):
            if time.time() >= deadline:
                break
            iters += 1
            cand_perm = rng.choice(hill_climbing.Operators.ALL)(perm, rng)
            cand = Solution(graph, cand_perm)
            front.add_solution(cand)
            cand_cost = meta.cost_at(cand, target_width, n)
            d = cand_cost - cur_cost
            win_att += 1
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
                perm, cur_cost = cand_perm, cand_cost
                accepts += 1
                win_acc += 1
        T *= alpha
        if win_att >= 5 * steps_per_t:
            if win_acc / win_att < 0.05:
                T = max(T, 0.5 * T0)          # reheat; a frozen walk stops
            win_att = win_acc = 0             # feeding the front
    return iters, accepts


def solve_front(graph, seconds, seed=1, widths=8, alpha=0.95, steps_per_t=200,
                verbose=False):
    """Front-aware simulated annealing: Hill Climbing's skeleton, Metropolis
    acceptance. Returns (front, iterations, accepts)."""
    rng = random.Random(seed)
    front = Front(graph.n)
    targets = hill_climbing.target_widths(graph, front, widths, "min_degree")
    per_width = seconds / len(targets)
    iters = accepts = 0
    for w in targets:
        i, a = _anneal_at_width(graph, front, w, per_width, rng, alpha,
                                steps_per_t)
        iters += i
        accepts += a
        if verbose:
            print(f"  width {w:>4}: {i:,} iters, {a:,} accepted, "
                  f"{front.score():,}", flush=True)
    return front, iters, accepts


def main():
    ap = argparse.ArgumentParser(description="Simulated annealing.")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t0", type=float, default=None,
                    help="initial temperature (default: calibrated)")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="cooling factor applied once per temperature level")
    ap.add_argument("--steps-per-t", type=int, default=200)
    ap.add_argument("--schedule", default="adaptive",
                    choices=["geometric", "linear", "adaptive"],
                    help="adaptive = geometric plus reheating when the "
                         "acceptance rate collapses")
    ap.add_argument("--start", default="min_degree",
                    choices=["min_degree", "random"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    graph = Graph.load(a.instance)
    print(f"=== simulated annealing on {os.path.basename(a.instance)} ===")
    print(graph.describe())
    t0 = time.time()
    front, iterations, accepts = solve(graph, a.seconds, a.seed, a.t0, a.alpha,
                                       a.steps_per_t, a.start, a.schedule,
                                       verbose=True)
    print(f"\n{iterations:,} iterations, {accepts:,} accepted "
          f"({accepts/max(iterations,1):.1%}), {time.time() - t0:.1f}s")
    print(f"SCORE: {front.score():,}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"instance": os.path.basename(a.instance),
                       "solver": "simulated_annealing", "n": graph.n,
                       "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

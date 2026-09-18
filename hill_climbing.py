#!/usr/bin/env python3
"""
Plain hill climbing for torso decomposition.

Tweak a copy, keep it if it's better, repeat. That's the whole algorithm.

Two objectives (width and t) are handled by fixing a target width and
climbing to minimise t at that width, so each climb is single-objective.
Run one climb per target width and the results form a front. Every
permutation evaluated donates its whole staircase to that front, including
the ones the climb rejects.

    python3 hill_climbing.py --instance data/toy.gr --seconds 5
    python3 hill_climbing.py --instance data/large-graph.gr --seconds 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from torso import Graph, Solution, Front, MAX_WIDTH


class Operators:
    """The moves. Each returns a new list; none touches its argument."""

    @staticmethod
    def swap_neighbours(perm, rng):
        new = list(perm)
        i = rng.randrange(len(new) - 1)
        new[i], new[i + 1] = new[i + 1], new[i]
        return new

    @staticmethod
    def swap_any(perm, rng):
        new = list(perm)
        i, j = rng.randrange(len(new)), rng.randrange(len(new))
        new[i], new[j] = new[j], new[i]
        return new

    @staticmethod
    def move_vertex(perm, rng):
        new = list(perm)
        v = new.pop(rng.randrange(len(new)))
        new.insert(rng.randrange(len(new) + 1), v)
        return new

    @staticmethod
    def reverse_segment(perm, rng):
        new = list(perm)
        i = rng.randrange(len(new) - 1)
        j = min(len(new), i + rng.randint(2, 10))
        new[i:j] = reversed(new[i:j])
        return new

    ALL = [swap_neighbours, swap_any, move_vertex, reverse_segment]


class Starts:
    """Initial orderings."""

    @staticmethod
    def random_order(graph, rng):
        perm = list(range(graph.n))
        rng.shuffle(perm)
        return perm

    @staticmethod
    def min_degree(graph, rng):
        """Repeatedly eliminate a lowest-degree vertex, ties broken randomly.

        Random starts are useless on the bigger graphs: they break the 500
        cap immediately, so every candidate is void and there is nothing to
        compare. This gives a legal ordering to start from.

        Keeping a degree table instead of recounting each step is the
        difference between O(n^2) and O(n^3) -- 5 seconds on large-graph
        rather than not finishing.
        """
        n = graph.n
        alive = set(range(n))
        work = [set(a) for a in graph.adj]
        degree = {v: len(work[v]) for v in range(n)}
        order = []

        while alive:
            low = min(degree[v] for v in alive)
            v = rng.choice([x for x in alive if degree[x] == low])
            order.append(v)
            alive.discard(v)

            survivors = work[v] & alive
            for u in survivors:
                work[u] |= survivors - {u}
            for u in survivors:
                degree[u] = len(work[u] & alive)

        return order


class HillClimber:
    def __init__(self, graph, front, rng, verbose=True):
        self.graph = graph
        self.front = front
        self.rng = rng
        self.verbose = verbose
        self.evaluations = 0
        self.accepts = 0

    def cost(self, solution, target_width):
        """Smallest t within target_width; n if the width is unreachable."""
        self.evaluations += 1
        t = solution.best_t_for_width(target_width)
        return self.graph.n if t is None else t

    def climb(self, target_width, seconds, start="min_degree"):
        deadline = time.time() + seconds

        perm = (Starts.random_order(self.graph, self.rng) if start == "random"
                else Starts.min_degree(self.graph, self.rng))
        current = Solution(self.graph, perm)
        current_cost = self.cost(current, target_width)
        self.front.add_solution(current)

        if self.verbose:
            shown = "unreachable" if current_cost >= self.graph.n else f"t={current_cost}"
            print(f"  width {target_width:>4}: {shown}", end="", flush=True)

        while time.time() < deadline:
            operator = self.rng.choice(Operators.ALL)
            candidate = Solution(self.graph, operator(current.perm, self.rng))
            candidate_cost = self.cost(candidate, target_width)
            self.front.add_solution(candidate)

            if candidate_cost < current_cost:
                current, current_cost = candidate, candidate_cost
                self.accepts += 1

        if self.verbose:
            shown = "unreachable" if current_cost >= self.graph.n else f"t={current_cost}"
            print(f"  ->  {shown}")
        return current_cost


def target_widths(graph, front, count):
    """Spread targets over the widths this graph actually reaches.

    A min-degree probe tells us the range; aiming outside it wastes budget
    on widths that are either impossible or trivial.
    """
    probe = Solution(graph, Starts.min_degree(graph, random.Random(0)))
    front.add_solution(probe)
    stairs = probe.staircase()

    if not stairs:
        return sorted({max(1, (i + 1) * MAX_WIDTH // count) for i in range(count)})

    widths = [w for w, _ in stairs]
    low, high = min(widths), max(widths)
    if low == high:
        return [low]
    step = (high - low) / max(1, count - 1)
    return sorted({int(round(low + i * step)) for i in range(count)})


def run(instance, seconds, widths, seed, start, out_path, self_check):
    rng = random.Random(seed)
    graph = Graph.load(instance)

    print(f"=== hill climbing on {os.path.basename(instance)} ===")
    print(graph.describe())

    if self_check:
        probe = Solution(graph, Starts.random_order(graph, random.Random(1)))
        if not probe.check():
            raise SystemExit("evaluators disagree -- refusing to continue")
        print("evaluator self-check: bitsets and sets AGREE")

    front = Front(graph.n)
    targets = target_widths(graph, front, widths)
    per_width = seconds / len(targets)
    print(f"{len(targets)} target widths, {per_width:.1f}s each, "
          f"seed {seed}, start '{start}'")
    print()

    climber = HillClimber(graph, front, rng)
    started = time.time()
    for width in targets:
        climber.climb(width, per_width, start=start)
    elapsed = time.time() - started

    points = front.best_k(20)
    print()
    print(f"{climber.evaluations:,} evaluations, {climber.accepts:,} accepted, "
          f"{elapsed:.1f}s")
    print(f"front has {len(front.pareto())} points, submitting {len(points)}")
    print()
    print("  width |      t | torso")
    for width, t in points:
        print(f"  {width:5d} | {t:6d} | {graph.n - t:6d}")
    print()
    print(f"SCORE: {front.score():,}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"instance": os.path.basename(instance),
                       "solver": "hill_climbing",
                       "n": graph.n,
                       "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Hill climbing for torso decomposition.")
    ap.add_argument("--instance", default="data/toy.gr")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--widths", type=int, default=8,
                    help="how many target widths to split the budget across")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="min_degree", choices=["min_degree", "random"])
    ap.add_argument("--out", default="")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the two evaluators agree first (small instances)")
    a = ap.parse_args()
    run(a.instance, a.seconds, a.widths, a.seed, a.start, a.out, a.self_check)


if __name__ == "__main__":
    main()

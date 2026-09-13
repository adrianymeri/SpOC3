#!/usr/bin/env python3
"""
hill_climbing.py -- plain hill climbing on the torso decomposition problem.

THE ALGORITHM
-------------
Hill climbing is the simplest local search there is:

    1. start from some solution S
    2. make a small random change to a copy of it  ->  R
    3. if R is better than S, keep R; otherwise throw R away
    4. repeat until out of time

That is all. There is no population, no temperature, no memory, no
learning. Steps 2 and 3 are the whole algorithm and they are 6 lines of
code in `HillClimber.climb`.

WHAT "BETTER" MEANS HERE
------------------------
The problem has two objectives (width, t) and we want a whole front of
trade-offs, not one point. We get that in the simplest possible way:

    pick a target width W, then hill-climb to MINIMISE t at that width.

Each climb is therefore an ordinary single-objective hill climb -- one
number going down -- which is exactly what makes it easy to explain. Run
one climb per target width, collect the results, and you have a front.

A useful accident of the problem (see torso.py) is that one permutation
already gives a point at EVERY width. So every climb, whatever width it
was aiming at, donates its whole staircase to the shared front. Aiming at
a width just decides which number the climber is pushing down.

    python3 hill_climbing.py --instance data/toy.gr --seconds 5
    python3 hill_climbing.py --instance data/small-graph.gr --seconds 60
    python3 hill_climbing.py --instance data/large-graph.gr --seconds 600

Pure Python standard library only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import time

from torso import Graph, Solution, Front, MAX_WIDTH


# ===========================================================================
# Operators -- the "small random change" of step 2
# ===========================================================================
#
# Each operator takes a permutation and returns a NEW one. None of them
# modifies the input: hill climbing always tweaks a *copy*, so that if the
# change turns out to be bad we still have the original.

class Operators:
    """The four moves the climber can make. All are deliberately simple."""

    @staticmethod
    def swap_neighbours(perm, rng):
        """Swap two positions that sit next to each other."""
        new = list(perm)
        i = rng.randrange(len(new) - 1)
        new[i], new[i + 1] = new[i + 1], new[i]
        return new

    @staticmethod
    def swap_any(perm, rng):
        """Swap two positions chosen anywhere in the order."""
        new = list(perm)
        i = rng.randrange(len(new))
        j = rng.randrange(len(new))
        new[i], new[j] = new[j], new[i]
        return new

    @staticmethod
    def move_vertex(perm, rng):
        """Take one vertex out and reinsert it somewhere else."""
        new = list(perm)
        i = rng.randrange(len(new))
        v = new.pop(i)
        new.insert(rng.randrange(len(new) + 1), v)
        return new

    @staticmethod
    def reverse_segment(perm, rng):
        """Reverse a short run of consecutive positions."""
        new = list(perm)
        i = rng.randrange(len(new) - 1)
        j = min(len(new), i + rng.randint(2, 10))
        new[i:j] = reversed(new[i:j])
        return new

    ALL = [
        ("swap_neighbours", swap_neighbours),
        ("swap_any", swap_any),
        ("move_vertex", move_vertex),
        ("reverse_segment", reverse_segment),
    ]


# ===========================================================================
# Starting points
# ===========================================================================

class Starts:
    """Two ways to produce the initial solution of step 1."""

    @staticmethod
    def random_order(graph, rng):
        perm = list(range(graph.n))
        rng.shuffle(perm)
        return perm

    @staticmethod
    def min_degree(graph, rng):
        """Repeatedly eliminate a lowest-degree vertex.

        The classic textbook heuristic for orderings of this kind, and a
        far better starting point than a random shuffle: on the bigger
        instances a random order breaks the width cap outright, so the
        climber would have nothing to work with.

        Two details keep this fast enough for large-graph (n = 2426).
        Written the naive way -- rescanning every surviving vertex each
        step to find the minimum -- it is O(n^3) and does not finish.

        1. A heap holds (degree, vertex) so the minimum is cheap to find.
           We never delete from the heap; instead we push updated entries
           and skip stale ones when they surface ("lazy deletion").
        2. Eliminating v only changes the degree of v's surviving
           neighbours, so those are the only ones we recompute.

        Ties are broken randomly, so different seeds give different starts.
        """
        n = graph.n
        work = list(graph.bits)            # adjacency + fill-in, as bitsets
        alive = (1 << n) - 1               # bit v set while v survives
        degree = [work[v].bit_count() for v in range(n)]

        # random tiebreak keeps equal-degree choices seed-dependent
        heap = [(degree[v], rng.random(), v) for v in range(n)]
        heapq.heapify(heap)

        order = []
        while heap:
            stored_degree, _, v = heapq.heappop(heap)
            bit = 1 << v
            if not (alive & bit):
                continue                   # already eliminated
            if stored_degree != degree[v]:
                continue                   # stale entry, a newer one exists

            order.append(v)
            alive &= ~bit
            survivors = work[v] & alive

            # fill-in: join the survivors into a clique
            rest = survivors
            while rest:
                lowest = rest & -rest
                rest ^= lowest
                u = lowest.bit_length() - 1
                work[u] |= survivors & ~lowest

            # only the survivors' degrees can have changed
            rest = survivors
            while rest:
                lowest = rest & -rest
                rest ^= lowest
                u = lowest.bit_length() - 1
                new_degree = (work[u] & alive).bit_count()
                if new_degree != degree[u]:
                    degree[u] = new_degree
                    heapq.heappush(heap, (new_degree, rng.random(), u))

        return order


# ===========================================================================
# The hill climber
# ===========================================================================

class HillClimber:
    """Plain hill climbing, minimising t at one target width."""

    def __init__(self, graph, front, rng, verbose=True):
        self.graph = graph
        self.front = front
        self.rng = rng
        self.verbose = verbose
        self.evaluations = 0
        self.accepts = 0

    def cost(self, solution, target_width):
        """The single number we are pushing down.

        Smallest t that stays within `target_width`. If the permutation
        cannot reach that width at all, the cost is n (worse than any real
        answer), which keeps the comparison in step 3 a plain `<`.
        """
        self.evaluations += 1
        t = solution.best_t_for_width(target_width)
        return self.graph.n if t is None else t

    def climb(self, target_width, seconds, start="min_degree"):
        """Run hill climbing until the time runs out. Returns the best t."""
        deadline = time.time() + seconds

        # ---- step 1: an initial solution ----
        if start == "random":
            perm = Starts.random_order(self.graph, self.rng)
        else:
            perm = Starts.min_degree(self.graph, self.rng)

        current = Solution(self.graph, perm)
        current_cost = self.cost(current, target_width)
        self.front.add_solution(current)

        if self.verbose:
            shown = "impossible" if current_cost >= self.graph.n \
                else f"t = {current_cost}"
            print(f"  width {target_width:>4}: start {shown}", end="", flush=True)

        while time.time() < deadline:
            # ---- step 2: a small random change to a COPY ----
            name, operator = self.rng.choice(Operators.ALL)
            candidate = Solution(self.graph, operator(current.perm, self.rng))
            candidate_cost = self.cost(candidate, target_width)

            # every permutation we look at donates its whole staircase,
            # even if it loses the comparison below
            self.front.add_solution(candidate)

            # ---- step 3: keep it only if it is better ----
            if candidate_cost < current_cost:
                current = candidate
                current_cost = candidate_cost
                self.accepts += 1
            # ---- step 4: loop ----

        if self.verbose:
            final = "impossible" if current_cost >= self.graph.n \
                else f"t = {current_cost}"
            print(f"  ->  {final}")
        return current_cost


# ===========================================================================
# Driving it: one climb per target width
# ===========================================================================

def choose_target_widths(graph, front, count):
    """Pick the widths to aim at.

    A first min-degree ordering tells us the range of widths this graph
    actually reaches. We then spread `count` targets evenly over that
    range, because aiming at widths the graph cannot reach -- or ones it
    reaches trivially -- would waste the whole budget.
    """
    rng = random.Random(0)
    probe = Solution(graph, Starts.min_degree(graph, rng))
    front.add_solution(probe)

    stairs = probe.staircase()
    if not stairs:
        # the probe broke the cap; fall back to a plain spread
        return sorted({max(1, (i + 1) * MAX_WIDTH // count)
                       for i in range(count)})

    widths = [w for (w, _) in stairs]
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
        ok = probe.check()
        print(f"evaluator self-check (bitsets vs sets): "
              f"{'AGREE' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("evaluator mismatch -- refusing to continue")

    front = Front(graph.n)
    targets = choose_target_widths(graph, front, widths)
    per_width = seconds / len(targets)
    print(f"targets: {len(targets)} widths, {per_width:.1f}s each, "
          f"seed {seed}, start '{start}'")
    print()

    climber = HillClimber(graph, front, rng)
    started = time.time()
    for width in targets:
        climber.climb(width, per_width, start=start)
    elapsed = time.time() - started

    points = front.best_k(20)
    print()
    print(f"evaluations {climber.evaluations:,}   "
          f"improving moves {climber.accepts:,}   "
          f"elapsed {elapsed:.1f}s")
    print(f"front: {len(front.pareto())} points, submitting "
          f"{len(points)}")
    print()
    print("  width |  t     | torso size")
    for width, t in points:
        print(f"  {width:5d} | {t:6d} | {graph.n - t:6d}")
    print()
    print(f"SCORE (negative area, more negative is better): "
          f"{front.score():,}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"instance": os.path.basename(instance),
                       "n": graph.n,
                       "score": front.score(),
                       "decisionVector": front.decision_vectors()}, f)
        print(f"wrote {out_path}")
        print(f"check it with:  python3 validate.py --instance {instance} "
              f"--submission {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plain hill climbing for torso decomposition.")
    ap.add_argument("--instance", default="data/toy.gr",
                    help="path to a .gr edge list")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="total time budget, split across the widths")
    ap.add_argument("--widths", type=int, default=8,
                    help="how many target widths to climb at")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="min_degree",
                    choices=["min_degree", "random"])
    ap.add_argument("--out", default="",
                    help="write the submission here, e.g. out/toy.json")
    ap.add_argument("--self-check", action="store_true",
                    help="verify the fast evaluator against the slow one "
                         "before starting (use on toy/small only)")
    args = ap.parse_args()

    run(args.instance, args.seconds, args.widths, args.seed,
        args.start, args.out, args.self_check)


if __name__ == "__main__":
    main()

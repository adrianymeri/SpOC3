#!/usr/bin/env python3
"""
torso.py -- the problem: graphs, solutions, and the front we submit.

All scoring is delegated to esa_eval.py, which is the evaluator ported from
the main project. Nothing here reimplements it except degrees_slow(), which
exists only as a readable cross-check.

See README Part 1 for what the problem actually is.
"""

from __future__ import annotations

from esa_eval import (MAX_TW, build_adj_bitsets, step_degrees,
                      hypervolume_2d, ParetoArchive)

MAX_WIDTH = MAX_TW


class Graph:
    """An undirected graph from a .gr edge list.

    Held twice: adj as sets (readable) and bits as integers (fast). Both say
    the same thing; degrees_slow() vs degrees() checks that they agree.
    """

    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.bits = build_adj_bitsets(n, adj)

    @classmethod
    def load(cls, path):
        edges = []
        max_node = 0
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0] == "p":
                    continue
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node = max(max_node, u, v)
        n = max_node + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        return cls(n, adj)

    def save(self, path):
        with open(path, "w") as f:
            for u in range(self.n):
                for v in sorted(self.adj[u]):
                    if u < v:
                        f.write(f"{u} {v}\n")

    @property
    def edge_count(self):
        return sum(len(a) for a in self.adj) // 2

    def describe(self):
        d = [len(a) for a in self.adj]
        return (f"n = {self.n}, edges = {self.edge_count}, "
                f"degree min/avg/max = {min(d)}/{sum(d) / self.n:.1f}/{max(d)}")


class Solution:
    """One elimination order.

    The threshold t is not stored: a single permutation answers the question
    for every t at once, which is what staircase() returns.
    """

    def __init__(self, graph, perm):
        self.graph = graph
        self.perm = list(perm)
        self._deg = None

    def degrees(self):
        """deg[i] = surviving neighbours of perm[i] when it was eliminated."""
        if self._deg is None:
            self._deg = step_degrees(self.perm, self.graph.bits, self.graph.n)
        return self._deg

    def degrees_slow(self):
        """Same thing with plain sets. Only used to check degrees()."""
        position = {v: i for i, v in enumerate(self.perm)}
        work = [set(a) for a in self.graph.adj]
        deg = [0] * self.graph.n
        for i, u in enumerate(self.perm):
            survivors = {v for v in work[u] if position[v] > i}
            deg[i] = len(survivors)
            for v in survivors:
                work[v] |= survivors - {v}
        return deg

    def check(self):
        return self.degrees() == self.degrees_slow()

    def width_at(self, t):
        deg = self.degrees()
        if max(deg) > MAX_WIDTH:
            return MAX_WIDTH + 1
        return max(deg[t:]) if t < len(deg) else 0

    def staircase(self):
        """Every (width, t) this permutation offers, cheapest t per width.

        Sweep t backwards keeping a running max. Each time the max steps up
        we have the smallest t reaching that width. Empty if the cap is broken.
        """
        deg = self.degrees()
        if max(deg) > MAX_WIDTH:
            return []
        best = {}
        run = 0
        for t in range(self.graph.n - 1, -1, -1):
            if deg[t] > run:
                run = deg[t]
            best[run] = t          # t only decreases, so this keeps the min
        return sorted(best.items())

    def best_t_for_width(self, target):
        """Smallest t staying within target width, or None."""
        best = None
        for width, t in self.staircase():
            if width <= target and (best is None or t < best):
                best = t
        return best

    def show(self, t, limit=8):
        torso = self.perm[t:]
        print(f"t = {t}, torso size = {len(torso)}, width = {self.width_at(t)}")
        for label, seq in (("eliminated", self.perm[:t]), ("torso", torso)):
            text = str(seq[:limit])
            if len(seq) > limit:
                text = text[:-1] + ", ...]"
            print(f"  {label:10} ({len(seq)}): {text}")


def hypervolume(points, n):
    """Area dominated by points, against the corner (n, n)."""
    return hypervolume_2d(points, n)


class Front:
    """The points we will submit, and their score.

    Keyed by width, holding the smallest t seen there and the permutation
    that achieved it.
    """

    def __init__(self, n):
        self.n = n
        self.points = {}

    def add(self, width, t, perm):
        if width > MAX_WIDTH or width >= self.n or t >= self.n:
            return False
        seen = self.points.get(width)
        if seen is not None and seen[0] <= t:
            return False
        self.points[width] = (t, list(perm))
        return True

    def add_solution(self, solution):
        improved = False
        for width, t in solution.staircase():
            if self.add(width, t, solution.perm):
                improved = True
        return improved

    def pareto(self):
        out = []
        best_t = self.n
        for width in sorted(self.points):
            t = self.points[width][0]
            if t < best_t:
                out.append((width, t))
                best_t = t
        return out

    def best_k(self, k=20):
        """The optimal k points, by the exact HSSP dynamic program."""
        archive = ParetoArchive()
        for width, (t, perm) in self.points.items():
            archive.try_add(width, t, perm)
        chosen = archive.top_k_by_hv_contribution(k, self.n)
        return sorted((w, t) for (w, t, _) in chosen)

    def score(self, k=20):
        # hypervolume_2d returns a float; the score is always a whole number
        return -int(hypervolume_2d(self.best_k(k), self.n))

    def decision_vectors(self, k=20):
        return [list(self.points[w][1]) + [int(t)] for w, t in self.best_k(k)]

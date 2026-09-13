#!/usr/bin/env python3
"""
torso.py -- the problem itself. Nothing clever lives in this file.

THE PROBLEM (SpOC-3 "Torso Decompositions")
-------------------------------------------
You are given an undirected graph G on n vertices. You choose:

    perm   an elimination ORDER: a permutation of 0..n-1
    t      a THRESHOLD: how many vertices at the front of perm are
           "eliminated" before we start measuring

Eliminating a vertex u means: remove u, and join all of u's not-yet-
eliminated neighbours into a clique (this is called "fill-in"). Eliminating
vertices in order therefore keeps ADDING edges among the survivors.

At each step i we record deg[i] = how many not-yet-eliminated neighbours the
vertex perm[i] had at the moment it was eliminated. The two objectives are:

    width = max(deg[i] for i >= t)     "how wide did the torso get"
    t                                  "how many vertices we had to remove"

BOTH ARE MINIMISED. Smaller t means we removed fewer vertices, which means
the surviving torso is LARGER (its size is n - t).

One hard rule from the organisers: if deg[i] > 500 at ANY step -- including
the eliminated head, before t -- the whole solution is void. We return
width = 501 to mark that.

THE KEY FACT THIS FILE EXPLOITS
-------------------------------
For one fixed perm you do NOT need a separate evaluation per t. Compute
deg[] once, then for every t:

    width(t) = max(deg[t], deg[t+1], ..., deg[n-1])

which is just a running maximum taken from the back. So a single pass over
one permutation hands you a whole STAIRCASE of (width, t) trade-off points.
That is what `Solution.staircase()` returns, and it is why the search in
hill_climbing.py can be so simple.

SCORING
-------
The competition takes at most 20 of your points and measures the area they
dominate, relative to the corner (n, n). More area is better. The official
score is the NEGATIVE of that area, so more negative is better.

Pure Python standard library only -- no numpy, no external packages.
"""

from __future__ import annotations


# The organisers' hard cap. Any elimination step wider than this voids
# the solution.
MAX_WIDTH = 500


# ===========================================================================
# Graph
# ===========================================================================

class Graph:
    """An undirected graph read from a `.gr` edge list.

    The `.gr` format is about as simple as it gets: one edge per line,
    two integers separated by a space. Lines starting with `p` are ignored
    (some files carry a header). Vertices are numbered from 0.

    We keep the graph in TWO equivalent forms:

        adj   list of sets -- adj[v] is the set of v's neighbours.
              This is the obvious representation and it is what you should
              read to understand the problem.

        bits  list of ints -- bits[v] has bit u set exactly when u is a
              neighbour of v. A Python int is an arbitrary-precision
              integer, so it doubles as a bitset of any size, and `&`,
              `|`, `.bit_count()` are then whole-set operations done in C.

    Both describe the same graph. `adj` is for reading; `bits` is for speed.
    Measured on the competition instances, one evaluation costs:

        small-graph  (n=1357)   sets 0.04 s    bitsets 0.01 s
        medium-graph (n=1399)   sets 3.9  s    bitsets 0.15 s
        large-graph  (n=2426)   sets 35.4 s    bitsets 0.63 s

    Hill climbing needs thousands of evaluations, so the set version is
    unusable on the big instances -- hence both. `Solution.check()` proves
    they agree.
    """

    def __init__(self, n, adj):
        self.n = n
        self.adj = adj
        self.bits = [0] * n
        for v in range(n):
            mask = 0
            for u in adj[v]:
                mask |= 1 << u
            self.bits[v] = mask

    @classmethod
    def load(cls, path):
        """Read a `.gr` edge list from disk."""
        edges = []
        max_node = 0
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts or parts[0] == "p":
                    continue
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                if u > max_node:
                    max_node = u
                if v > max_node:
                    max_node = v
        n = max_node + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:                     # ignore self-loops
                adj[u].add(v)
                adj[v].add(u)
        return cls(n, adj)

    def save(self, path):
        """Write this graph back out as a `.gr` edge list."""
        with open(path, "w") as f:
            for u in range(self.n):
                for v in sorted(self.adj[u]):
                    if u < v:              # write each edge once
                        f.write(f"{u} {v}\n")

    @property
    def edge_count(self):
        return sum(len(a) for a in self.adj) // 2

    def describe(self):
        degrees = [len(a) for a in self.adj]
        return (f"n = {self.n}, edges = {self.edge_count}, "
                f"degree min/avg/max = {min(degrees)}/"
                f"{sum(degrees) / self.n:.1f}/{max(degrees)}")


# ===========================================================================
# Solution
# ===========================================================================

class Solution:
    """One elimination order, plus everything it tells us.

    A Solution is just a permutation. The threshold t is NOT stored,
    because one permutation gives a good answer for EVERY t at once
    (see `staircase`).
    """

    def __init__(self, graph, perm):
        self.graph = graph
        self.perm = list(perm)
        self._deg = None                   # computed lazily, then cached

    # -- evaluation -------------------------------------------------------

    def degrees(self):
        """deg[i] = neighbours of perm[i] still alive when it is eliminated.

        This is the fast bitset version. `degrees_slow` below is the same
        thing written with sets; the two are checked against each other by
        `check()`.
        """
        if self._deg is not None:
            return self._deg

        n = self.graph.n
        perm = self.perm

        # later[i] = bitset of all vertices at positions AFTER i.
        # Built from the back so each one is the previous plus one vertex.
        later = [0] * n
        seen = 0
        for i in range(n - 1, -1, -1):
            later[i] = seen
            seen |= 1 << perm[i]

        work = list(self.graph.bits)       # adjacency we will add fill-in to
        deg = [0] * n

        for i in range(n):
            u = perm[i]
            # u's neighbours that have not been eliminated yet
            survivors = work[u] & later[i]
            deg[i] = survivors.bit_count()

            # fill-in: make those survivors a clique.
            # Walk the set bits of `survivors` one at a time.
            rest = survivors
            while rest:
                lowest = rest & -rest      # isolate lowest set bit
                rest ^= lowest
                v = lowest.bit_length() - 1
                work[v] |= survivors & ~lowest   # everyone except v itself

        self._deg = deg
        return deg

    def degrees_slow(self):
        """The same computation with plain sets -- the readable version.

        Kept so you can see what `degrees()` actually does, and so
        `check()` can prove the fast one is right. Too slow for the big
        instances (35 s per call on large-graph), fine for toy and small.
        """
        n = self.graph.n
        position = {v: i for i, v in enumerate(self.perm)}
        work = [set(a) for a in self.graph.adj]
        deg = [0] * n

        for i, u in enumerate(self.perm):
            survivors = {v for v in work[u] if position[v] > i}
            deg[i] = len(survivors)
            for v in survivors:            # fill-in: join them all together
                work[v] |= survivors - {v}

        return deg

    def check(self):
        """Return True if the fast and slow evaluators agree.

        Run this on a small instance to convince yourself (or your
        supervisor) that the bitset trick is not doing anything sneaky.
        """
        return self.degrees() == self.degrees_slow()

    # -- objectives -------------------------------------------------------

    def width_at(self, t):
        """The width objective for this permutation at threshold t.

        width = max(deg[i] for i >= t), or 501 if any step anywhere
        breaks the organisers' cap.
        """
        deg = self.degrees()
        if max(deg) > MAX_WIDTH:
            return MAX_WIDTH + 1
        if t >= len(deg):
            return 0
        return max(deg[t:])

    def staircase(self):
        """Every (width, t) trade-off point this permutation offers.

        Walk t from the back to the front keeping a running maximum. Each
        time the running maximum steps up we have found the smallest t that
        achieves that width -- which is exactly the point we want, because
        smaller t is better.

        Returns a list of (width, t) sorted by increasing width. Returns []
        if the permutation breaks the cap and is therefore void.
        """
        deg = self.degrees()
        if max(deg) > MAX_WIDTH:
            return []

        n = self.graph.n
        best_t_for_width = {}
        running_max = 0
        for t in range(n - 1, -1, -1):
            if deg[t] > running_max:
                running_max = deg[t]
            # going backwards t only gets smaller, so overwrite freely:
            # the last value we store for a width is the smallest t.
            best_t_for_width[running_max] = t

        return sorted(best_t_for_width.items())

    def best_t_for_width(self, target_width):
        """Smallest t whose width is <= target_width (None if impossible).

        This is the single number the hill climber minimises.
        """
        best = None
        for width, t in self.staircase():
            if width <= target_width:
                if best is None or t < best:
                    best = t
        return best

    def show(self, t, max_items=8):
        """Print the solution the way a human wants to see it."""
        width = self.width_at(t)
        torso = self.perm[t:]
        print(f"Solution: t = {t}, torso size = {len(torso)}, "
              f"width = {width}")

        head = self.perm[:t]
        head_str = str(head[:max_items])
        if len(head) > max_items:
            head_str = head_str[:-1] + ", ...]"
        print(f"  eliminated ({len(head)}): {head_str}")

        torso_str = str(torso[:max_items])
        if len(torso) > max_items:
            torso_str = torso_str[:-1] + ", ...]"
        print(f"  torso      ({len(torso)}): {torso_str}")


# ===========================================================================
# Front + scoring
# ===========================================================================

def hypervolume(points, n):
    """Area dominated by `points` relative to the corner (n, n).

    A point (w, t) covers the rectangle from (w, t) to (n, n). We want the
    area of the union of those rectangles. Sort by width, sweep left to
    right, and only keep points that improve on the best t so far.
    """
    valid = sorted((w, t) for (w, t) in points if w < n and t < n)
    if not valid:
        return 0

    frontier = []
    best_t = n
    for w, t in valid:
        if t < best_t:
            frontier.append((w, t))
            best_t = t

    area = 0
    for i, (w, t) in enumerate(frontier):
        next_w = frontier[i + 1][0] if i + 1 < len(frontier) else n
        area += (next_w - w) * (n - t)
    return area


class Front:
    """The set of (width, t) points we will submit, and its score.

    Keeps only non-dominated points: (w1,t1) dominates (w2,t2) when it is
    no worse on both objectives and strictly better on at least one.
    """

    def __init__(self, n):
        self.n = n
        self.points = {}                   # width -> (t, perm)

    def add(self, width, t, perm):
        """Record a point. Returns True if it improved the front."""
        if width > MAX_WIDTH or width >= self.n or t >= self.n:
            return False
        existing = self.points.get(width)
        if existing is not None and existing[0] <= t:
            return False                   # we already have this or better
        self.points[width] = (t, list(perm))
        return True

    def add_solution(self, solution):
        """Add every point of a solution's staircase."""
        improved = False
        for width, t in solution.staircase():
            if self.add(width, t, solution.perm):
                improved = True
        return improved

    def pareto(self):
        """Non-dominated points, sorted by width."""
        result = []
        best_t = self.n
        for width in sorted(self.points):
            t = self.points[width][0]
            if t < best_t:
                result.append((width, t))
                best_t = t
        return result

    def best_k(self, k=20):
        """The k points that between them cover the most area.

        The competition scores at most 20 points. We pick them greedily:
        repeatedly add whichever remaining point increases the area most.
        Greedy is a sensible, easily explained choice here; it is not
        guaranteed optimal, and `validate.py` reports what it achieves.
        """
        candidates = self.pareto()
        if len(candidates) <= k:
            return candidates

        chosen = []
        remaining = list(candidates)
        while len(chosen) < k and remaining:
            best_point = None
            best_area = -1
            for point in remaining:
                area = hypervolume(chosen + [point], self.n)
                if area > best_area:
                    best_area = area
                    best_point = point
            chosen.append(best_point)
            remaining.remove(best_point)
        return sorted(chosen)

    def score(self, k=20):
        """Official score of our best k points: negative area."""
        return -hypervolume(self.best_k(k), self.n)

    def decision_vectors(self, k=20):
        """The submission: each chosen point as [perm..., t]."""
        vectors = []
        for width, t in self.best_k(k):
            perm = self.points[width][1]
            vectors.append(list(perm) + [int(t)])
        return vectors

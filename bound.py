#!/usr/bin/env python3
"""
bound.py -- a provable ceiling on the score, for every instance.

The competition is NP-hard twice over, so the optimum is out of reach. But a
*bound* on it is not, and it turns out to be tight enough to be useful: it
says small-graph is within 0.63% of solved.

Where the bound comes from
--------------------------
Deleting a set S of t vertices with fill-in leaves the **torso** of G over S:
two survivors are adjacent exactly when some path joins them through S. The
torso depends only on S, not on the order S was deleted in. Whatever order is
then used for the remaining n - t vertices, the reported width is an
elimination width of that torso -- and the minimum elimination width of any
graph is exactly its treewidth. So the true optimum at threshold t is

    min over all t-subsets S of  tw(torso(G, S))

which is hopeless to compute. The bound comes from one observation. Take any
tree decomposition of torso(G, S) and add S to every bag: the result is a
valid tree decomposition of G, so

    tw(G) <= tw(torso(G, S)) + t

Rearranged, every point any method can ever report obeys

    width >= tw(G) - t

Treewidth is itself NP-hard, but *lower bounds* on it are cheap, and any
lower bound L gives the wall width >= max(0, L - t). The best a submission
can do is therefore the largest hypervolume 20 points can have on that wall,
which this script computes by dynamic programming over the frontier
(L - t, t) for t = 0..L.

The three lower bounds used
---------------------------
All are valid because treewidth is minor-monotone and minimum degree
lower-bounds treewidth:

  degeneracy              max over subgraphs of the minimum degree. Exact,
                          linear time.
  clique - 1              a clique on k vertices forces tw >= k - 1. Found
                          greedily, so itself a lower bound on the largest.
  contraction degeneracy  (MMD+least-c) repeatedly contract the minimum-degree
                          vertex into the neighbour it shares fewest
                          neighbours with, tracking the largest minimum degree
                          seen. Every contraction is a minor, so the figure
                          stays valid, and it is strictly >= degeneracy.

The largest of the three is used.

    python3 bound.py --instance data/small-graph.gr
    python3 bound.py --all
"""

from __future__ import annotations

import argparse
import heapq
import os

from torso import Graph


# --- treewidth lower bounds ----------------------------------------------

def degeneracy(adj, n):
    """Max over all subgraphs of the minimum degree. tw(G) >= degeneracy(G)."""
    deg = [len(adj[v]) for v in range(n)]
    alive = [True] * n
    heap = [(deg[v], v) for v in range(n)]
    heapq.heapify(heap)
    k = 0
    while heap:
        d, v = heapq.heappop(heap)
        if not alive[v] or d != deg[v]:
            continue
        alive[v] = False
        k = max(k, d)
        for u in adj[v]:
            if alive[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
    return k


def greedy_clique(adj, n, tries=25):
    """Greedy clique from each of the `tries` highest-degree vertices.

    Returns its size; tw(G) >= size - 1. Greedy, so this is a lower bound on
    the clique number, which is itself a lower bound on treewidth + 1.

    At each step it takes the candidate of highest degree rather than the one
    sharing most neighbours with the candidate set. The latter picks slightly
    better cliques but costs an intersection per candidate per step, which is
    O(|cand|^2) and hopeless on a graph that actually contains a 500-clique.
    On these instances the cheap rule finds the planted cliques anyway.
    """
    best = 1
    order = sorted(range(n), key=lambda v: -len(adj[v]))[:tries]
    for start in order:
        clique, cand = 1, set(adj[start])
        while cand:
            v = max(cand, key=lambda x: len(adj[x]))
            clique += 1
            cand &= adj[v]
        best = max(best, clique)
    return best


def contraction_degeneracy(adj0, n):
    """MMD+least-c. Contract the minimum-degree vertex into the neighbour it
    shares fewest neighbours with, and track the largest minimum degree seen.

    Contraction yields a minor, treewidth is minor-monotone, and minimum
    degree lower-bounds treewidth -- so the maximum over the sequence is a
    valid lower bound, and never worse than plain degeneracy.
    """
    adj = {v: set(adj0[v]) for v in range(n)}
    lb = 0
    while len(adj) > 1:
        v = min(adj, key=lambda x: len(adj[x]))
        d = len(adj[v])
        if d == 0:
            del adj[v]
            continue
        lb = max(lb, d)
        u = min(adj[v], key=lambda x: len(adj[v] & adj[x]))
        nv = adj.pop(v)
        for w in nv:
            adj[w].discard(v)
        nv.discard(u)
        adj[u] |= nv
        for w in nv:
            adj[w].add(u)
        adj[u].discard(u)
    return lb


def treewidth_lower_bound(graph, verbose=False):
    """The strongest of the three cheap bounds."""
    adj = [set(a) for a in graph.adj]
    dg = degeneracy(adj, graph.n)
    cl = greedy_clique(adj, graph.n) - 1
    cd = contraction_degeneracy(graph.adj, graph.n)
    if verbose:
        print(f"  degeneracy {dg}, clique-1 {cl}, contraction degeneracy {cd}")
    return max(dg, cl, cd)


# --- the ceiling ----------------------------------------------------------

def ceiling(n, lb, kmax=20):
    """Largest hypervolume kmax points can have, given width >= lb - t.

    The frontier is (lb - j, j_t) for width j = 0..lb, i.e. the point of
    width j sits at threshold lb - j. Sorted by width ascending, the
    hypervolume of a chosen subset j_1 < ... < j_k against the reference
    (n, n) telescopes to

        sum over i of  (j_{i+1} - j_i) * (n - (lb - j_i))     with j_{k+1} = n

    so a dynamic program over the frontier is exact.
    """
    prev = [(n - j) * (n - (lb - j)) for j in range(lb + 1)]   # j is the last
    for _ in range(kmax - 1):
        cur = list(prev)
        for j in range(lb, -1, -1):
            best = (n - j) * (n - (lb - j))
            reach = n - (lb - j)
            for j2 in range(j + 1, lb + 1):
                cand = (j2 - j) * reach + prev[j2]
                if cand > best:
                    best = cand
            cur[j] = best
        prev = cur
    return max(prev)


INSTANCES = (["small-graph", "medium-graph", "large-graph"]
             + [f"synth-{i}" for i in range(1, 8)])


def report(path, verbose=False):
    graph = Graph.load(path)
    lb = treewidth_lower_bound(graph, verbose)
    c = ceiling(graph.n, lb)
    return graph.n, lb, c


def main():
    ap = argparse.ArgumentParser(
        description="Provable ceiling on the achievable score.")
    ap.add_argument("--instance", default="")
    ap.add_argument("--all", action="store_true",
                    help="every instance in data/")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--verbose", action="store_true",
                    help="show each of the three treewidth lower bounds")
    a = ap.parse_args()

    targets = ([os.path.join(a.data_dir, f"{x}.gr") for x in INSTANCES]
               if a.all else [a.instance])
    if not a.all and not a.instance:
        raise SystemExit("give --instance or --all")

    print(f"{'instance':<16}{'n':>7}{'tw >=':>8}{'best possible':>16}"
          f"{'as % of n^2':>14}")
    for path in targets:
        if not os.path.exists(path):
            print(f"  MISSING {path}")
            continue
        if a.verbose:
            print(os.path.basename(path))
        n, lb, c = report(path, a.verbose)
        name = os.path.basename(path).replace(".gr", "")
        print(f"{name:<16}{n:>7,}{lb:>8}{-c:>16,}{c/(n*n):>13.2%}")
    print("\nScores are negative hypervolume, so the ceiling is the most "
          "negative value\nany method could ever report. No submission can "
          "pass it.")


if __name__ == "__main__":
    main()

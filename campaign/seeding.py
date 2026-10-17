"""
seeding.py -- structure-aware starting orderings for the four designed methods
(EA-1, EA-2, NE-1, NE-2). The baselines keep the start they were defined with
(min-degree with random tie-breaks), as does EA-3 NSGA-II.

    min-degree        greedy elimination, random ties, slack 0..3
    clique quota      for graphs with big cliques: for a target width w, the
                      head takes |C| - 1 - w vertices of every big clique C
                      (the clique-packing bound), cheapest attachment first
                      (vertices with no outside neighbour cost no fill); the
                      head is eliminated min-degree first, then the torso

Every ordering is returned with its exact degrees; the caller banks them.
"""

from __future__ import annotations

import numpy as np

import structure


def min_degree_orders(eng, crng, count):
    out = []
    for i in range(count):
        perm, deg, bad = eng.greedy(crng, i % 4)
        out.append((perm, deg))
    return out


def quota_head(eng, S, w, g):
    """Head mask for target width w from the clique-packing quotas."""
    cl = S["clique"]
    head = np.zeros(eng.n, dtype=np.uint8)
    for c in sorted(set(cl[cl >= 0].tolist())):
        mem = np.where(cl == c)[0]
        k = len(mem) - 1 - w
        if k <= 0:
            continue
        outdeg = np.array([int((cl[eng.neighbours(v)] != c).sum()) for v in mem])
        order = np.lexsort((g.random(len(mem)), outdeg))
        head[mem[order[:k]]] = 1
    return head


def quota_orders(eng, S, crng, g, widths):
    out = []
    if not (S["clique"] >= 0).any():
        return out
    for w in widths:
        head = quota_head(eng, S, int(w), g)
        if head.sum() == 0:
            continue
        perm, deg, bad = eng.greedy(crng, 0, head=head)
        out.append((perm, deg))
    return out


def portfolio(eng, crng, g, n_min_degree=16, n_quota=24):
    """The starting set: min-degree orderings plus clique-quota constructions
    spread over the widths below the largest clique."""
    S = structure.load(eng)
    out = min_degree_orders(eng, crng, n_min_degree)
    cs = S["clique_size"]
    if (S["clique"] >= 0).any():
        top = int(cs.max()) - 1
        widths = np.unique(np.linspace(0, top - 1, n_quota).astype(int))
        out += quota_orders(eng, S, crng, g, widths)
    return out

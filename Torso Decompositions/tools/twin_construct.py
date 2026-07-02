#!/usr/bin/env python3
"""twin_construct.py -- structure-driven CONSTRUCTION of orderings from the
twin-quotient graph (the Spacekangaroos direction, confirmed by S. Limmer's
personal communication: the large-graph top exploited planted graph structure
to *construct* solutions).

True-twin classes are interchangeable units (automorphism => identical widths).
We therefore eliminate on the QUOTIENT graph (medium: 882 super-vertices of
1399; large: 1709 of 2426, incl. planted classes of 375@deg499, 102@399,
62@299) with weighted greedy rules, then expand each super-vertex to its twins
as one contiguous block. One IncEvalC pass per ordering yields the full
(width, t) staircase; orderings are pooled with the existing corpus under the
exact capped-20 objective and saved only on strict improvement.

    python3 tools/twin_construct.py --problem large-graph --tries 200 --seed 1
"""
from __future__ import annotations
import argparse, collections, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path
from tools.fastwalk import IncEvalC
from tools.archive_evolve import (order_front, capped_hv, load_seed_orders,
                                  top_k_owners, LEADERBOARD_TARGETS)


def twin_quotient(n, adj):
    """Classes (incl. singletons), quotient adjacency, weights."""
    h = collections.defaultdict(list)
    for v in range(n):
        h[frozenset(adj[v] | {v})].append(v)
    classes = list(h.values())
    cid = {}
    for i, c in enumerate(classes):
        for v in c:
            cid[v] = i
    q = [set() for _ in classes]
    for v in range(n):
        for u in adj[v]:
            if cid[u] != cid[v]:
                q[cid[v]].add(cid[u])
    w = [len(c) for c in classes]
    return classes, q, w


def quotient_order(classes, q, w, rng, rule, jitter):
    """Greedy weighted elimination on the quotient graph -> class order.
    rule 'deg'  : eliminate min weighted-degree first (peel periphery,
                  dense planted classes sink into the torso tail)
    rule 'degw' : same but counting own class residue (clique inside)
    Fill is applied: neighbours of the eliminated class become a clique."""
    m = len(classes)
    g = [set(x) for x in q]
    alive = set(range(m))
    deg = [sum(w[u] for u in g[c]) for c in range(m)]   # weighted alive-degree
    own = [(w[c] - 1) if rule == "degw" else 0 for c in range(m)]
    out = []
    while alive:
        best, bestscore = None, None
        for c in alive:
            s = deg[c] + own[c] + rng.random() * jitter
            if bestscore is None or s < bestscore:
                best, bestscore = c, s
        alive.discard(best)
        nb = [u for u in g[best] if u in alive]
        for u in nb:
            deg[u] -= w[best]
        for i in range(len(nb)):
            gi = g[nb[i]]
            for j in range(i + 1, len(nb)):
                if nb[j] not in gi:
                    gi.add(nb[j]); g[nb[j]].add(nb[i])
                    deg[nb[i]] += w[nb[j]]; deg[nb[j]] += w[nb[i]]
        out.append(best)
    return out


def expand(order, classes, rng):
    perm = []
    for c in order:
        mem = list(classes[c])
        rng.shuffle(mem)          # identity irrelevant (automorphism)
        perm.extend(mem)
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--tries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=40)
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem))
    ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(a.problem)
    rng = random.Random(a.seed)

    classes, q, w = twin_quotient(n, adj)
    nt = sum(1 for c in classes if len(c) > 1)
    print(f"=== twin-construct {a.problem} | quotient {len(classes)} super-vertices "
          f"({nt} twin classes, biggest {max(w)}) ===", flush=True)

    members = load_seed_orders(here, a.problem, n, ev, a.pool)
    cur, top = capped_hv(members, n)
    print(f"    corpus capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''}", flush=True)

    accepts = 0
    t0 = time.time()
    for it in range(a.tries):
        rule = "deg" if it % 2 == 0 else "degw"
        jitter = rng.choice([0.0, 0.5, 2.0, 8.0])
        co = quotient_order(classes, q, w, rng, rule, jitter)
        perm = expand(co, classes, rng)
        pts = order_front(perm, ev, n)
        trial = members + [(perm, pts)]
        h, top = capped_hv(trial, n)
        if h < cur - 0.5:
            cur = h; accepts += 1
            keep = set(tuple(p) for (_, _, p) in top)
            members = [(p, x) for (p, x) in trial if tuple(p) in keep]
            if (perm, pts) not in members:
                members.append((perm, pts))
            print(f"  try {it}: *** capped-20 {cur:,.0f}"
                  f"{f'  gap {cur-target:+,.0f}' if target else ''} "
                  f"(construct accept #{accepts}) ***", flush=True)
            dvs = [list(p) + [int(t)] for (_, t, p) in top]
            out = os.path.join(here, "submissions", a.problem, "twin_construct.json")
            json.dump({"challenge": "spoc-3-torso-decompositions",
                       "problem": a.problem, "decisionVector": dvs}, open(out, "w"))
        if it % 10 == 0 and it:
            print(f"  [try {it}/{a.tries}  capped-20 {cur:,.0f}  {accepts} accepts  "
                  f"{time.time()-t0:.0f}s]", flush=True)
    print(f"\nfinal capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | {accepts} construct accepts",
          flush=True)


if __name__ == "__main__":
    main()

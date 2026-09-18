#!/usr/bin/env python3
"""
generate.py -- make instances.

    python3 generate.py --family small  --seed 1 --out data/small-extra-1.gr
    python3 generate.py --family medium --seed 4 --out data/medium-extra-1.gr
    python3 generate.py --family large  --seed 6 --out data/large-extra-1.gr
    python3 generate.py --kind toy --out data/toy.gr

Why three separate families
---------------------------
The obvious way to make a new instance that resembles an old one is to keep
the degree sequence and rewire the edges (a double edge swap preserves every
vertex's degree exactly). I tried that first and it does not work here:

    instance        template width    after rewiring
    small-graph               20        274
    medium-graph             276        621
    large-graph              499        499

small-graph and medium-graph are *structured* graphs that happen to have a
particular degree sequence, and rewiring throws the structure away. The
result has the same degrees and is an order of magnitude harder -- a
different problem wearing the same costume. Only large-graph survives,
because its difficulty sits in three big cliques and those are frozen.

So each family is built the way its template is built, and the generator is
checked against the template's measured width rather than its degree list.

    family   construction                        n      edges    width
    small    grid + pendants                  1357     ~2380       ~21
    medium   dense blocks + cross edges       1399    ~14000      ~260
    large    template cliques, relabelled,    2426    253895       499
             periphery rewired
"""

from __future__ import annotations

import argparse
import json
import os
import random

from torso import Graph


# --------------------------------------------------------------------------
# small: a sparse, triangle-free, low-width grid with pendant vertices
# --------------------------------------------------------------------------

def make_small(seed, rows=10, cols=115, pendants=207, drop=100):
    """A rows x cols grid with pendants hanging off it.

    A grid is triangle-free and has small separators, which is what gives
    small-graph its low width. Dropping a handful of grid edges brings the
    edge count down towards the template without disturbing the width.
    """
    rng = random.Random(seed)
    core = rows * cols
    n = core + pendants
    adj = [set() for _ in range(n)]

    def at(r, c):
        return r * cols + c

    grid_edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                grid_edges.append((at(r, c), at(r, c + 1)))
            if r + 1 < rows:
                grid_edges.append((at(r, c), at(r + 1, c)))
    rng.shuffle(grid_edges)
    for u, v in grid_edges[drop:]:
        adj[u].add(v)
        adj[v].add(u)

    for p in range(pendants):
        v = core + p
        u = rng.randrange(core)
        adj[v].add(u)
        adj[u].add(v)

    return Graph(n, adj), {"family": "small", "rows": rows, "cols": cols,
                           "pendants": pendants, "grid_edges_dropped": drop}


# --------------------------------------------------------------------------
# medium: dense blocks joined by a controlled number of cross edges
# --------------------------------------------------------------------------

def make_medium(seed, n=1399, nblocks=5, p_in=0.0705, cross=360):
    """Blocks of roughly n/nblocks vertices, dense inside, sparse between.

    The block size sets the width (you cannot eliminate a block without
    paying for its separator) and the cross edges tune how much the blocks
    interfere with each other.
    """
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    size = n // nblocks
    blocks = [list(range(i * size, (i + 1) * size if i < nblocks - 1 else n))
              for i in range(nblocks)]

    for block in blocks:
        for i, u in enumerate(block):
            for v in block[i + 1:]:
                if rng.random() < p_in:
                    adj[u].add(v)
                    adj[v].add(u)

    for _ in range(cross):
        a, b = rng.sample(range(nblocks), 2)
        u, v = rng.choice(blocks[a]), rng.choice(blocks[b])
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)

    return Graph(n, adj), {"family": "medium", "blocks": nblocks,
                           "block_sizes": [len(b) for b in blocks],
                           "p_inside": p_in, "cross_edges": cross}


# --------------------------------------------------------------------------
# large: keep the template's cliques, relabel, rewire everything else
# --------------------------------------------------------------------------

def find_cliques(adj, n, min_size=50, limit=6):
    degree = [len(adj[v]) for v in range(n)]
    used, found = set(), []
    for _ in range(limit):
        clique = set()
        for v in sorted((v for v in range(n) if v not in used),
                        key=lambda v: -degree[v]):
            if all(u in adj[v] for u in clique):
                clique.add(v)
        if len(clique) < min_size:
            break
        found.append(clique)
        used |= clique
    return found


def make_large(seed, template_path, swaps_per_edge=20):
    """large-graph is three cliques plus a sparse periphery.

    The cliques are the instance -- a K500 forces width >= 499 whatever you
    do -- so they are kept. Vertices are relabelled so the cliques land on
    different ids, and the ~4,500 peripheral edges are rewired by double
    edge swaps, which preserves every degree exactly.
    """
    rng = random.Random(seed)
    template = Graph.load(template_path)
    n = template.n

    new_id = list(range(n))
    rng.shuffle(new_id)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in template.adj[u]:
            adj[new_id[u]].add(new_id[v])

    cliques = find_cliques(adj, n)
    frozen = {(u, v) for clique in cliques for u in clique for v in clique
              if u < v and v in adj[u]}
    edges = [(u, v) for u in range(n) for v in adj[u]
             if u < v and (u, v) not in frozen]

    target, done, tries = swaps_per_edge * max(1, len(edges)), 0, 0
    while done < target and tries < target * 12:
        tries += 1
        i, j = rng.randrange(len(edges)), rng.randrange(len(edges))
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) != 4 or d in adj[a] or b in adj[c]:
            continue
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if e1 in frozen or e2 in frozen:
            continue
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        adj[a].add(d); adj[d].add(a)
        adj[c].add(b); adj[b].add(c)
        edges[i], edges[j] = e1, e2
        done += 1

    return Graph(n, adj), {"family": "large",
                           "template": os.path.basename(template_path),
                           "cliques_kept": sorted((len(c) for c in cliques),
                                                  reverse=True),
                           "vertices_relabelled": True,
                           "peripheral_edges_rewired": done}


# --------------------------------------------------------------------------
# standalone instances
# --------------------------------------------------------------------------

def make_toy():
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (6, 8), (7, 8), (7, 9), (8, 9), (9, 10), (10, 11), (0, 11)]
    adj = [set() for _ in range(12)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return Graph(12, adj), {"kind": "toy", "note": "two clusters + bridge"}


def make_random(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return Graph(n, adj), {"kind": "random", "n": n, "p": p}


# --------------------------------------------------------------------------

def min_degree_width(graph, seed=0):
    """Difficulty proxy: the width min-degree elimination achieves."""
    rng = random.Random(seed)
    n = graph.n
    alive = set(range(n))
    work = [set(a) for a in graph.adj]
    degree = {v: len(work[v]) for v in range(n)}
    width = 0
    while alive:
        low = min(degree[v] for v in alive)
        v = rng.choice([x for x in alive if degree[x] == low])
        alive.discard(v)
        survivors = work[v] & alive
        width = max(width, len(survivors))
        for u in survivors:
            work[u] |= survivors - {u}
        for u in survivors:
            degree[u] = len(work[u] & alive)
    return width


def main():
    ap = argparse.ArgumentParser(description="Generate torso instances.")
    ap.add_argument("--family", default="", choices=["", "small", "medium", "large"])
    ap.add_argument("--kind", default="toy", choices=["toy", "random"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--template", default="data/large-graph.gr",
                    help="template for --family large")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.family == "small":
        graph, meta = make_small(args.seed)
    elif args.family == "medium":
        graph, meta = make_medium(args.seed)
    elif args.family == "large":
        graph, meta = make_large(args.seed, args.template)
    elif args.kind == "toy":
        graph, meta = make_toy()
    else:
        graph, meta = make_random(args.n, args.p, args.seed)

    meta["seed"] = args.seed
    meta["n"] = graph.n
    meta["edges"] = graph.edge_count
    if graph.n <= 3000:
        meta["min_degree_width"] = min_degree_width(graph)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    graph.save(args.out)
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {args.out}")
    print(f"  {graph.describe()}")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

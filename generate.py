#!/usr/bin/env python3
"""
generate.py -- build the synthetic instances.

    python3 generate.py --all          # reproduce every file in data/
    python3 generate.py --kind toy --out data/toy.gr

small-graph, medium-graph and large-graph are ESA's and are not generated
here. Everything else in data/ comes out of this file, and `--all`
reproduces it byte for byte.

    instance        made by                              from
    toy             make_toy                             --
    synth-1..3      make_small(seed 1..3)                --
    synth-4, 5      rewire_preserving(seed 1,2, 12)      medium-graph
    synth-6, 7      rewire_preserving(seed 1,2, 0.05)    large-graph

What has to be preserved, and why
---------------------------------
An instance is only useful if it can tell two methods apart. The measure is
**headroom**: how much better the best solver does than plain min-degree. An
instance with no headroom ranks nothing -- every method lands on the greedy
answer and the table reads as a tie.

Three approaches were tried. Two failed, and they failed in ways worth
recording, because both looked correct by every statistic that is easy to
check.

**1. Degree-preserving edge swaps.** The obvious way to make a new instance
resembling an old one: keep every vertex's degree, rewire the edges. It
destroys the instance.

    instance        template width    after rewiring
    small-graph               20        274
    medium-graph             276        621
    large-graph              499        499

small-graph and medium-graph are *structured* graphs that happen to have a
given degree sequence; rewiring keeps the costume and throws away the
problem. Only large-graph survives, because its difficulty sits in three
planted cliques and those stay frozen.

**2. Per-family constructive generators.** Build each family the way its
template is built -- grid plus pendants for small, dense blocks plus cross
edges for medium, frozen cliques plus a rewired periphery for large. These
matched vertex count, edge count and degree distribution, and still could
not separate two solvers:

    instance        headroom       vs the official instance
    small-graph       +7,701
      synth-1..3      +1,434 to +2,971        0.19-0.39x
    medium-graph     +83,927
      synth-4, 5     +32,376 to +105,715      0.39-1.26x
    large-graph     +506,790
      synth-6, 7      +5,344 to +6,688        0.01x

synth-6 and synth-7 had one per cent of large-graph's discriminating power.
Both weight-space solvers returned *no feasible solution at all* on synth-4
and synth-5.

**3. Twin-preserving rewiring — this is what ships.** The missing ingredient
was **twin classes**: sets of vertices with identical closed neighbourhoods.
Once you eliminate one member of a class, its partners' neighbourhoods are
already cliques, so a method that finds the classes and eliminates them
together pays once instead of many times. min-degree has no way to see that.
It is precisely the structure that separates a good ordering from a greedy
one -- and randomly rewiring a periphery destroys it.

    instance        twin classes   the constructive version had
    medium-graph        507                    0
    large-graph         117                   38

`rewire_preserving` relabels every vertex, collapses the twin classes into a
quotient graph, double-edge-swaps the quotient with the planted cliques
frozen, and expands back. Degrees, twins and cliques survive by
construction; the wiring between classes is new. Result:

    synth-4, 5   twin classes 0 -> 507. fcmaes and Spacekangaroos went from
                 returning nothing to beating min-degree.
    synth-6, 7   twin classes 38 -> 117, headroom +6,688 -> +373,526 and
                 +5,344 -> +136,449 (56x and 26x).

Two details that cost real debugging. A quotient edge stands for |a| x |b|
real edges, so swapping between classes of *different sizes* silently
changes the edge count -- it cost medium 9% of its edges until the size
check went in. And large-graph tolerates almost no perturbation: at 12
swaps per edge every feature-sort method went infeasible, so synth-6 and
synth-7 use 0.05. Its periphery is only ~4,500 of 253,895 edges and is
highly structured.

Why synth-1..3 still use the constructive generator
---------------------------------------------------
small-graph has no closed twins, so twin-preserving rewiring is just
degree-preserving randomisation there -- it turned a width-21 grid into a
width-277 random graph.

It does have *open* twins (same neighbour set, different closed
neighbourhood): 207 pendants on 35 hosts, six apiece. A variant that
reproduced that exactly -- 35 classes covering 207 vertices, matching
small-graph's 35 and 206 -- raised headroom at a 45-second probe from +136
to +425 HV, and stopped there. At the same probe small-graph itself yields
only +1,492 (0.08%), and scaling the synth-1 figure to full budget lands
near +2,100, under the 2,218 HV seed-noise floor.

So the regime is tight, not the generator: on sparse low-treewidth instances
min-degree is already within ~0.1% of anything any method finds. synth-1..3
are reported as ties, which is the honest reading and a result in its own
right -- this problem only begins to discriminate as density and planted
structure increase.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from torso import Graph


# --------------------------------------------------------------------------
# synth-1..3: a sparse, triangle-free, low-width grid with pendants
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
# synth-4..7: rewire an official instance, holding its twin classes fixed
# --------------------------------------------------------------------------

def twin_classes(graph):
    """Group vertices by closed neighbourhood. Returns a list of lists."""
    buckets = defaultdict(list)
    for v in range(graph.n):
        buckets[frozenset(graph.adj[v] | {v})].append(v)
    return list(buckets.values())


def find_cliques(adj, n, min_size=50, limit=6):
    """Greedy pass for the planted cliques. They set the width floor."""
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


def rewire_preserving(template, seed, swaps_per_edge=12):
    """Relabel, then swap edges between twin classes, keeping cliques frozen."""
    rng = random.Random(seed)
    n = template.n

    # 1. relabel
    new_id = list(range(n))
    rng.shuffle(new_id)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in template.adj[u]:
            adj[new_id[u]].add(new_id[v])
    relabelled = Graph(n, adj)

    # 2. twin classes, and the class each vertex belongs to
    classes = twin_classes(relabelled)
    owner = {}
    for i, members in enumerate(classes):
        for v in members:
            owner[v] = i

    # 3. cliques stay exactly as they are -- they are the instance
    cliques = find_cliques(adj, n)
    in_clique = {v for c in cliques for v in c}

    # 4. quotient edges, excluding anything touching a clique
    qedges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v and u not in in_clique and v not in in_clique:
                a, b = owner[u], owner[v]
                if a != b:
                    qedges.add((min(a, b), max(a, b)))
    qedges = list(qedges)
    qadj = defaultdict(set)
    for a, b in qedges:
        qadj[a].add(b)
        qadj[b].add(a)

    # 5. double edge swaps on the quotient: degree-preserving per class
    target = swaps_per_edge * max(1, len(qedges))
    done = tries = 0
    while done < target and tries < target * 12:
        tries += 1
        i, j = rng.randrange(len(qedges)), rng.randrange(len(qedges))
        if i == j:
            continue
        a, b = qedges[i]
        c, d = qedges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) != 4:
            continue
        if d in qadj[a] or b in qadj[c]:
            continue
        # A quotient edge stands for |a| x |b| real edges, so swapping between
        # classes of different sizes silently changes the edge count -- it cost
        # medium 9% of its edges before this check. Requiring |a|=|c| and
        # |b|=|d| makes the two sides equal and keeps every degree exact.
        if (len(classes[a]) != len(classes[c])
                or len(classes[b]) != len(classes[d])):
            continue
        qadj[a].discard(b); qadj[b].discard(a)
        qadj[c].discard(d); qadj[d].discard(c)
        qadj[a].add(d); qadj[d].add(a)
        qadj[c].add(b); qadj[b].add(c)
        qedges[i] = (min(a, d), max(a, d))
        qedges[j] = (min(c, b), max(c, b))
        done += 1

    # 6. expand the quotient back to vertices
    out = [set() for _ in range(n)]
    for u in range(n):                      # keep clique and intra-class edges
        for v in adj[u]:
            if u in in_clique or v in in_clique or owner[u] == owner[v]:
                out[u].add(v)
                out[v].add(u)
    for a, b in qedges:                     # rewired inter-class edges
        for u in classes[a]:
            for v in classes[b]:
                out[u].add(v)
                out[v].add(u)

    multi = [c for c in classes if len(c) > 1]
    meta = {"source": "rewire_preserving", "seed": seed,
            # classes with more than one member -- a singleton is not a twin,
            # and counting them made the meta report 882 where the structural
            # comparison in the README reports 507.
            "twin_classes": len(multi),
            "vertices_in_twins": sum(len(c) for c in multi),
            "cliques": sorted((len(c) for c in cliques), reverse=True),
            "quotient_swaps": done}
    return Graph(n, out), meta


def overlap(a, b):
    """Fraction of a's edges that b also has -- how much really changed."""
    ea = {(u, v) for u in range(a.n) for v in a.adj[u] if u < v}
    eb = {(u, v) for u in range(b.n) for v in b.adj[u] if u < v}
    return len(ea & eb) / max(1, len(ea))

# --------------------------------------------------------------------------
# toy, and a difficulty probe
# --------------------------------------------------------------------------

def make_toy():
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6),
             (6, 7), (6, 8), (7, 8), (7, 9), (8, 9), (9, 10), (10, 11), (0, 11)]
    adj = [set() for _ in range(12)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return Graph(12, adj), {"kind": "toy", "note": "two clusters + bridge"}


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


# --------------------------------------------------------------------------
# reproducing data/
# --------------------------------------------------------------------------

# (output name, template or None, seed, swaps_per_edge)
PLAN = [
    ("synth-1", None, 1, None),
    ("synth-2", None, 2, None),
    ("synth-3", None, 3, None),
    ("synth-4", "data/medium-graph.gr", 1, 12),
    ("synth-5", "data/medium-graph.gr", 2, 12),
    ("synth-6", "data/large-graph.gr", 1, 0.05),
    ("synth-7", "data/large-graph.gr", 2, 0.05),
]


def build(name, template_path, seed, swaps, out_dir="data"):
    """One instance, plus the .meta.json recording how it was made."""
    if template_path is None:
        graph, meta = make_small(seed)
        meta["source"] = "make_small"
    else:
        template = Graph.load(template_path)
        graph, meta = rewire_preserving(template, seed, swaps)
        meta["template"] = os.path.basename(template_path)
        meta["swaps_per_edge"] = swaps
        meta["edge_overlap_with_template"] = round(overlap(template, graph), 4)
    meta["seed"] = seed
    meta["n"] = graph.n
    meta["edges"] = graph.edge_count
    path = os.path.join(out_dir, f"{name}.gr")
    graph.save(path)
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return graph, meta


def main():
    ap = argparse.ArgumentParser(description="Generate torso instances.")
    ap.add_argument("--all", action="store_true",
                    help="rebuild every generated file in data/")
    ap.add_argument("--kind", default="", choices=["", "toy"])
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="data")
    a = ap.parse_args()

    if a.all:
        print(f"{'instance':<10}{'n':>6}{'edges':>8}{'twins':>8}{'overlap':>9}")
        print("-" * 41)
        graph, meta = make_toy()
        graph.save(os.path.join(a.out_dir, "toy.gr"))
        with open(os.path.join(a.out_dir, "toy.gr.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{'toy':<10}{graph.n:>6}{graph.edge_count:>8}{'-':>8}{'-':>9}")
        for name, template, seed, swaps in PLAN:
            g, m = build(name, template, seed, swaps, a.out_dir)
            ov = m.get("edge_overlap_with_template")
            ov = f"{ov:.1%}" if ov is not None else "-"
            twins = m.get("twin_classes", "-")
            print(f"{name:<10}{g.n:>6}{g.edge_count:>8}{twins:>8}{ov:>9}")
        return

    if a.kind == "toy":
        if not a.out:
            raise SystemExit("--kind toy needs --out")
        graph, meta = make_toy()
        graph.save(a.out)
        with open(a.out + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"wrote {a.out}: n={graph.n}, edges={graph.edge_count}")
        return

    raise SystemExit("nothing to do -- try --all")


if __name__ == "__main__":
    main()

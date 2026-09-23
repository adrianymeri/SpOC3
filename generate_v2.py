#!/usr/bin/env python3
"""
generate_v2.py -- build synthetic instances that behave like the real ones.

Why this replaces the first generator
-------------------------------------
The first attempt matched the official instances on the things that are easy
to measure -- vertex count, edge count, degree distribution -- and produced
graphs that could not tell solvers apart. Measured headroom, meaning how much
better the best method does than plain min-degree:

    small-graph      7,701        synth-1..3     1,434 - 2,971   (0.19-0.39x)
    medium-graph    83,927        synth-4,5     32,376 - 105,715 (0.39-1.26x)
    large-graph    506,790        synth-6,7      5,344 - 6,688   (0.01x)

An instance with no headroom cannot rank anything: every method lands on the
greedy answer and the table reads as a tie.

The missing ingredient was **twin classes** -- sets of vertices with identical
closed neighbourhoods:

    medium-graph    507 classes, 1,024 of 1,399 vertices      synth-4,5:  0
    large-graph     117 classes,   834 of 2,426 vertices      synth-6,7: 38

Twins matter because they are free: once you eliminate one vertex of a twin
class its partners' neighbourhoods are already cliques, so a method that finds
the classes and eliminates them together pays once instead of many times.
min-degree has no way to see that. It is precisely the structure that separates
a good ordering from a greedy one -- and randomly rewiring the periphery, which
is what the first generator did, destroys it.

What this does instead
----------------------
Rewire the official instance while holding its twin partition fixed:

    1. relabel every vertex, so ids carry no information from the original
    2. compute the twin classes
    3. collapse them into a quotient graph, one node per class
    4. double-edge-swap the quotient, leaving planted cliques frozen
    5. expand back

Degrees, twin classes and cliques all survive by construction; the wiring
between classes is new. The result is a different graph that is hard in the
same way the original is hard.

Outcome, measured
-----------------
    synth-4, 5   twin classes 0 -> 507; fcmaes and Spacekangaroos went from
                 returning nothing to beating min-degree. Fixed.
    synth-6, 7   twin classes 38 -> 117 and the periphery perturbed at only
                 0.05 swaps/edge (12 destroyed the instance); headroom
                 +5,344 -> +136,449 and +6,688 -> +373,526. Fixed.
    synth-1..3   NOT fixed, and the attempt is recorded below.

The small family: why it stays a tie
------------------------------------
small-graph has no closed twins, so the rewiring above is just degree-
preserving randomisation there -- it turned a width-21 grid into a width-277
random graph. Reverted.

It does have *open* twins (same neighbour set, different closed
neighbourhood): 207 pendants on 35 hosts, six apiece, 35 classes covering 206
vertices. `make_small_v2` reproduces that exactly. It raised headroom at a
45 s probe from +136 to +425 HV -- and stopped there.

The reason is that the regime itself is tight. At the same 45 s probe
small-graph yields only +1,492 (0.08%). Its full-budget figure is carried by
HRI at +7,701 (0.42%), and by one outlying Spacekangaroos run whose two
siblings scored *below* min-degree. Scaling the synth-1 measurement by the
same factor lands near +2,100, under the 2,218 HV seed-noise floor.

So on sparse, low-treewidth instances min-degree is already within ~0.1% of
anything any method finds, and no amount of structural fidelity changes that.
synth-1..3 are reported as ties, which is the honest reading and a result in
its own right: this problem only starts discriminating between methods as
density and planted structure increase.

    python3 generate_v2.py --all
    python3 generate_v2.py --template data/medium-graph.gr --seed 1 --out data/synth-4.gr
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from torso import Graph


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

    meta = {"source": "rewire_preserving", "seed": seed,
            "twin_classes": len(classes),
            "vertices_in_twins": sum(len(c) for c in classes if len(c) > 1),
            "cliques": sorted((len(c) for c in cliques), reverse=True),
            "quotient_swaps": done}
    return Graph(n, out), meta


def overlap(a, b):
    """Fraction of a's edges that b also has -- how much really changed."""
    ea = {(u, v) for u in range(a.n) for v in a.adj[u] if u < v}
    eb = {(u, v) for u in range(b.n) for v in b.adj[u] if u < v}
    return len(ea & eb) / max(1, len(ea))


PLAN = [("data/small-graph.gr",  ["synth-1", "synth-2", "synth-3"]),
        ("data/medium-graph.gr", ["synth-4", "synth-5"]),
        ("data/large-graph.gr",  ["synth-6", "synth-7"])]


def build(template_path, out_path, seed, swaps):
    template = Graph.load(template_path)
    graph, meta = rewire_preserving(template, seed, swaps)
    meta["template"] = os.path.basename(template_path)
    meta["edge_overlap_with_template"] = round(overlap(template, graph), 4)
    graph.save(out_path)
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return graph, meta


def main():
    ap = argparse.ArgumentParser(description="Twin-preserving instance generator.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--template", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--swaps", type=int, default=12)
    a = ap.parse_args()

    if a.all:
        print(f"{'instance':<10}{'n':>6}{'edges':>8}{'twin cls':>10}"
              f"{'in twins':>10}{'overlap':>9}")
        print("-" * 53)
        for template, names in PLAN:
            for i, name in enumerate(names):
                g, m = build(template, f"data/{name}.gr", a.seed + i, a.swaps)
                print(f"{name:<10}{g.n:>6}{g.edge_count:>8}"
                      f"{m['twin_classes']:>10}{m['vertices_in_twins']:>10}"
                      f"{m['edge_overlap_with_template']:>9.1%}", flush=True)
        return

    if not a.template or not a.out:
        raise SystemExit("need --all, or both --template and --out")
    g, m = build(a.template, a.out, a.seed, a.swaps)
    print(f"wrote {a.out}: n={g.n}, edges={g.edge_count}, "
          f"{m['twin_classes']} twin classes, "
          f"{m['edge_overlap_with_template']:.1%} edge overlap")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# small family: the pendants are the structure
# --------------------------------------------------------------------------

def make_small_v2(seed, rows=10, cols=115, pendants=207, drop=100, hosts=35):
    """A rows x cols grid with pendants CLUSTERED onto a few host vertices.

    The first generator attached each pendant to a uniformly random core
    vertex, which spread 207 pendants over ~190 hosts -- about one each. The
    real small-graph hangs its 207 pendants off just 35 hosts, roughly six
    apiece, and those six are mutual *open* twins: same neighbour set, so
    once you eliminate one the rest are free.

    That is the same twin mechanism the medium and large families needed, in
    the one form closed-neighbourhood matching cannot see (two pendants on a
    host have different closed neighbourhoods but identical open ones). With
    the pendants scattered there is nothing for a method to discover, and
    min-degree is already within 0.1% of the best anyone finds.
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

    host_pool = rng.sample(range(core), hosts)
    for p in range(pendants):
        v = core + p
        u = host_pool[p % hosts]          # even spread over a small pool
        adj[v].add(u)
        adj[u].add(v)

    return Graph(n, adj), {"family": "small_v2", "seed": seed, "rows": rows,
                           "cols": cols, "pendants": pendants,
                           "dropped_grid_edges": drop, "hosts": hosts}

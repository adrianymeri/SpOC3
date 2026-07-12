#!/usr/bin/env python3
"""
family_gen.py -- P3(a): generator for the SpOC-3 planted family.

Reproduces the disclosed construction (Bannach p.c.: disjoint union of
low-treewidth graphs glued by a dense graph), fitted to the measured
statistics of large-graph (THESIS §15.1): glue = disjoint cliques with
true-twin interiors; components = partial k-trees (tw <= k); externals carry
a small number of component attachments.

    python3 tools/family_gen.py --scale 0.25 --seed 1 --out extra_instances/fam_s25_1.gr

Writes <out> (edge list) and <out>.meta.json (ground-truth structure).
"""
from __future__ import annotations
import argparse, json, os, random, sys


def partial_ktree(nv, k, keep, rng):
    """random k-tree on nv vertices, then keep each non-skeleton edge w.p. keep."""
    edges = set()
    for i in range(min(k + 1, nv)):
        for j in range(i + 1, min(k + 1, nv)):
            edges.add((i, j))
    cliques = [list(range(min(k + 1, nv)))]
    for v in range(k + 1, nv):
        base = rng.choice(cliques)
        drop = rng.randrange(len(base))
        newc = [u for i, u in enumerate(base) if i != drop] + [v]
        for u in newc[:-1]:
            if rng.random() < keep or len(edges) < nv:
                edges.add((min(u, v), max(u, v)))
        cliques.append(newc)
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tw", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    s = a.scale
    # glue: three cliques with twin interiors (large-graph ratios)
    clique_specs = [(round(500 * s), round(375 * s)),
                    (round(400 * s), round(102 * s)),
                    (round(300 * s), round(62 * s))]
    # components: sizes drawn like large-graph's (1..161 scaled)
    comp_sizes = []
    remaining = round(1226 * s)
    profile = [161, 145, 126, 124, 120, 104, 65, 54, 50, 46, 35, 35, 35, 35,
               35, 21, 9, 6, 5, 5, 4, 3, 1, 1, 1]
    for c in profile:
        cs = max(1, round(c * s))
        if remaining - cs < 0:
            break
        comp_sizes.append(cs); remaining -= cs
    edges = set()
    vid = 0
    twins_meta, ext_meta, comp_meta = [], [], []
    ext_all = []
    for Ksz, tw_n in clique_specs:
        ids = list(range(vid, vid + Ksz)); vid += Ksz
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.add((ids[i], ids[j]))
        twins_meta.append(ids[:tw_n])
        ext_meta.append(ids[tw_n:])
        ext_all += ids[tw_n:]
    for cs in comp_sizes:
        ids = list(range(vid, vid + cs)); vid += cs
        for (u, v) in partial_ktree(cs, min(a.tw, cs - 1), 0.75, rng):
            edges.add((ids[u], ids[v]))
        comp_meta.append(ids)
    # attachments: each external gets Poisson-ish comp attachments (mean ~6,
    # capped like large's <=39); some cross-clique external-external edges
    comp_all = [v for c in comp_meta for v in c]
    for x in ext_all:
        for _ in range(min(39, max(1, int(rng.expovariate(1 / 6.0))))):
            if comp_all:
                edges.add(tuple(sorted((x, rng.choice(comp_all)))))
        if rng.random() < 0.35:
            y = rng.choice(ext_all)
            if y != x:
                edges.add(tuple(sorted((x, y))))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        for u, v in sorted(edges):
            fh.write(f"{u} {v}\n")
    json.dump({"n": vid, "cliques": [len(t) + len(e) for t, e in
                                     zip(twins_meta, ext_meta)],
               "twins": twins_meta, "externals": ext_meta,
               "components": comp_meta, "tw": a.tw, "scale": s,
               "seed": a.seed},
              open(a.out + ".meta.json", "w"))
    print(f"wrote {a.out}: n={vid}, m={len(edges)}, "
          f"cliques {[len(t)+len(e) for t,e in zip(twins_meta, ext_meta)]}, "
          f"{len(comp_meta)} components")


if __name__ == "__main__":
    main()

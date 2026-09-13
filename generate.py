#!/usr/bin/env python3
"""
generate.py -- make instances to run the hill climber on.

Three kinds, from easiest to hardest to reason about:

    toy      A tiny hand-sized graph. Small enough to draw on paper and
             to check the evaluator by hand. Use this to explain the
             problem to somebody.

    random   Erdos-Renyi: n vertices, each possible edge present with
             probability p. No structure at all -- a neutral baseline.

    planted  The shape the real competition instances have: several
             separate low-width "components" glued together by one dense
             "glue" that touches all of them. The glue is what forces the
             width up, so you know in advance roughly where the difficulty
             lives. A `.meta.json` file records the planted structure.

    python3 generate.py --kind toy    --out data/toy.gr
    python3 generate.py --kind random  --n 200 --p 0.05 --out data/rand200.gr
    python3 generate.py --kind planted --n 400 --out data/planted400.gr

Pure Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random

from torso import Graph


def make_toy(rng):
    """A 12-vertex graph: two small clusters joined by a short bridge.

    Deliberately hand-checkable. Eliminating the leaf-ish vertices first
    costs almost nothing; the width is driven by the two clusters.
    """
    edges = [
        # cluster A
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
        # bridge
        (4, 5), (5, 6),
        # cluster B
        (6, 7), (6, 8), (7, 8), (7, 9), (8, 9), (9, 10),
        # a couple of pendants
        (10, 11), (0, 11),
    ]
    n = 12
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return Graph(n, adj), {"kind": "toy", "note": "two clusters + bridge"}


def make_random(n, p, rng):
    """Erdos-Renyi G(n, p)."""
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return Graph(n, adj), {"kind": "random", "n": n, "p": p}


def make_planted(n, components, glue_size, density, rng):
    """Low-width components glued by one dense core.

    This mirrors how the competition instances are built: eliminating a
    component is cheap, but the glue is dense and forces the width up. If
    you want an instance whose difficulty you can predict, use this.
    """
    glue_size = min(glue_size, n - components * 3)
    glue = list(range(glue_size))
    rest = list(range(glue_size, n))
    rng.shuffle(rest)

    adj = [set() for _ in range(n)]

    # 1. the glue: a dense random core
    for i, u in enumerate(glue):
        for v in glue[i + 1:]:
            if rng.random() < density:
                adj[u].add(v)
                adj[v].add(u)

    # 2. split the remaining vertices into components
    blocks = [[] for _ in range(components)]
    for i, v in enumerate(rest):
        blocks[i % components].append(v)

    # 3. each component: a tree plus a few extra edges -> low width
    sizes = []
    for block in blocks:
        sizes.append(len(block))
        for i in range(1, len(block)):
            parent = block[rng.randrange(i)]
            adj[block[i]].add(parent)
            adj[parent].add(block[i])
        for _ in range(len(block) // 4):
            a, b = rng.sample(block, 2) if len(block) >= 2 else (None, None)
            if a is not None and a != b:
                adj[a].add(b)
                adj[b].add(a)

    # 4. attach every component to the glue -- this is what couples them
    for block in blocks:
        for v in rng.sample(block, max(1, len(block) // 10)):
            g = rng.choice(glue)
            adj[v].add(g)
            adj[g].add(v)

    meta = {"kind": "planted", "n": n, "glue_size": glue_size,
            "components": components, "component_sizes": sizes,
            "glue_density": density,
            "note": "width is driven by the glue; components are cheap"}
    return Graph(n, adj), meta


def main():
    ap = argparse.ArgumentParser(description="Generate torso instances.")
    ap.add_argument("--kind", default="toy",
                    choices=["toy", "random", "planted"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.05,
                    help="edge probability (random only)")
    ap.add_argument("--components", type=int, default=4,
                    help="number of low-width blocks (planted only)")
    ap.add_argument("--glue-size", type=int, default=30,
                    help="size of the dense core (planted only)")
    ap.add_argument("--density", type=float, default=0.6,
                    help="density inside the glue (planted only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/toy.gr")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if args.kind == "toy":
        graph, meta = make_toy(rng)
    elif args.kind == "random":
        graph, meta = make_random(args.n, args.p, rng)
    else:
        graph, meta = make_planted(args.n, args.components,
                                   args.glue_size, args.density, rng)

    meta["seed"] = args.seed
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    graph.save(args.out)
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {args.out}")
    print(f"  {graph.describe()}")
    print(f"  structure: {meta}")
    print(f"run it:  python3 hill_climbing.py --instance {args.out} "
          f"--seconds 10")


if __name__ == "__main__":
    main()

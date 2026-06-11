#!/usr/bin/env python3
"""
generate_instances.py -- create 20 synthetic graph instances for
extended benchmarking of the HC series.

Each instance is an Erdős-Rényi random graph G(n, p) with
p = avg_deg / (n - 1), sampling each potential edge once.  Files are
written under `extra_instances/data/inst_NN.gr` in the same plain
edge-list format as the official small/medium/large-graph.gr.

The 20 instances span 5 sizes × 4 average degrees:
    n        in {200, 350, 500, 750, 1000}
    avg_deg  in {3, 8, 18, 35}

This grid was chosen so that:
    - The sparsest cells (n=200, avg_deg=3) finish a 3 s HC run
      in well under that budget.
    - The densest cells (n=1000, avg_deg=35 -> ~17.5k edges) still
      finish without triggering the 500-width limit on min-degree
      warm starts.
    - The grid brackets the official small / medium graphs in
      density (small: 3.36, medium: 19.73).

Reproducible via seed = 42.
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [200, 350, 500, 750, 1000]
AVG_DEGS = [3, 8, 18, 35]


def erdos_renyi(n: int, avg_deg: float, rng: random.Random) -> List[Tuple[int, int]]:
    """Generate G(n, p) with p = avg_deg / (n - 1).

    Returns a deduplicated edge list (u, v) with u < v.  We sample
    edge-by-edge to avoid building the dense O(n^2) edge set in memory
    for the larger cells.
    """
    if n < 2:
        return []
    p = avg_deg / (n - 1)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def write_gr(path: str, edges: List[Tuple[int, int]]) -> None:
    """Write a .gr edge-list file.  Same format as small/medium/large-graph.gr."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def instance_name(idx: int, n: int, d: int) -> str:
    return f"inst_{idx:02d}_n{n}_d{d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--here", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(args.here, "extra_instances", "data")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    instances = []
    print(f"Generating 20 instances (seed = {args.seed})\n")
    print(f"  {'#':>2}  {'name':<22}  {'n':>5}  {'avg':>5}  {'edges':>6}  "
          f"{'max':>5}")
    idx = 0
    for n in SIZES:
        for d in AVG_DEGS:
            idx += 1
            edges = erdos_renyi(n, d, rng)
            # observed avg degree from sampled edge set
            real_avg = 2 * len(edges) / n if n > 0 else 0
            # max degree
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            max_d = max(deg) if deg else 0
            name = instance_name(idx, n, d)
            path = os.path.join(out_dir, f"{name}.gr")
            write_gr(path, edges)
            instances.append({
                "idx": idx, "name": name, "n": n, "target_avg": d,
                "edges": len(edges), "real_avg": real_avg, "max_deg": max_d,
            })
            print(f"  {idx:>2}  {name:<22}  {n:>5}  {real_avg:>5.1f}  "
                  f"{len(edges):>6}  {max_d:>5}")

    summary_path = os.path.join(args.here, "extra_instances", "instances.csv")
    with open(summary_path, "w") as f:
        f.write("idx,name,n,target_avg,edges,real_avg,max_deg\n")
        for inst in instances:
            f.write(f"{inst['idx']},{inst['name']},{inst['n']},"
                    f"{inst['target_avg']},{inst['edges']},"
                    f"{inst['real_avg']:.2f},{inst['max_deg']}\n")
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()

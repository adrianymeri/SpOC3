#!/usr/bin/env python3
"""
nested_dissection.py -- Randomised nested dissection for SpOC-3 HV optimisation.

CMA-ES and GBFC++ search locally around existing orderings.  Nested dissection
exploits GLOBAL graph structure: find a balanced vertex separator S, recursively
order each part, place S last.  This explores a fundamentally different region
of permutation space and is known to produce near-optimal elimination orderings
for sparse graphs.

Algorithm:
  1. Compute the Fiedler vector (2nd-smallest Laplacian eigenvector) of the
     current subgraph.
  2. The middle sep_frac of vertices (by Fiedler value) form the separator S.
  3. Recurse on each side, then append S.
  4. At small subgraphs, use random ordering (explored by restarts).

Randomisation: add Gaussian noise to the Fiedler vector before sorting,
producing structurally diverse orderings across restarts.

    python3 tools/nested_dissection.py --problem small-graph --restarts 1000
    python3 tools/nested_dissection.py --problem small-graph --restarts 5000 --noise 0.3
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, MAX_TW, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from tools.gbfc import banked
from tools.gbfcpp import staircase, breakpoints, IncEval


# ── graph helpers ─────────────────────────────────────────────────────────────

def subgraph_laplacian(verts: list[int], adj: list[list[int]]) -> sp.csr_matrix:
    """Laplacian of the subgraph induced on verts."""
    m = len(verts)
    v2i = {v: i for i, v in enumerate(verts)}
    rows, cols = [], []
    for v in verts:
        for u in adj[v]:
            if u in v2i:
                rows.append(v2i[v]); cols.append(v2i[u])
    if not rows:
        return sp.csr_matrix((m, m), dtype=float)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m), dtype=float)
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return D - A


def fiedler_sort(verts: list[int], adj: list[list[int]],
                 noise: float, rng: np.random.Generator) -> np.ndarray:
    """Return vertices sorted by (Fiedler vector + noise)."""
    m = len(verts)
    if m <= 2:
        return np.array(verts)
    L = subgraph_laplacian(verts, adj)
    try:
        k = min(3, m - 1)
        _, vecs = spla.eigsh(L, k=k, which='SM', tol=1e-3, maxiter=300)
        f = vecs[:, 1]
    except Exception:
        f = rng.uniform(0, 1, m)
    if noise > 0:
        f = f + rng.normal(0, noise * (f.max() - f.min() + 1e-9), m)
    return np.array(verts)[np.argsort(f)]


# ── nested dissection ─────────────────────────────────────────────────────────

def nd_order(verts: list[int], adj: list[list[int]],
             rng: np.random.Generator,
             noise: float = 0.15,
             sep_frac: float = 0.20,
             base: int = 30) -> list[int]:
    """
    Recursive nested dissection ordering.
    Returns a permutation of verts.
    """
    n = len(verts)
    if n <= base:
        arr = list(verts)
        rng.shuffle(arr)
        return arr

    sorted_v = fiedler_sort(verts, adj, noise, rng)
    s = max(2, int(n * sep_frac))
    lo = (n - s) // 2
    hi = lo + s

    part_A = sorted_v[:lo].tolist()
    sep    = sorted_v[lo:hi].tolist()
    part_B = sorted_v[hi:].tolist()

    if not part_A:
        return nd_order(part_B, adj, rng, noise, sep_frac, base) + sep
    if not part_B:
        return nd_order(part_A, adj, rng, noise, sep_frac, base) + sep

    order_A = nd_order(part_A, adj, rng, noise, sep_frac, base)
    order_B = nd_order(part_B, adj, rng, noise, sep_frac, base)
    return order_A + order_B + sep


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--restarts", type=int, default=1000,
                    help="number of randomised ND restarts")
    ap.add_argument("--noise", type=float, default=0.15,
                    help="Gaussian noise on Fiedler values (more = more diverse)")
    ap.add_argument("--sep-frac", type=float, default=0.20,
                    help="separator fraction of each subgraph (0.10–0.35)")
    ap.add_argument("--base", type=int, default=30,
                    help="random ordering below this subgraph size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(args.problem)
    verts = list(range(n))

    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
    except Exception:
        ev = IncEval(ab, n)

    # seed archive from existing banked pool
    pool = banked(here, args.problem, n, ab)[:60]
    arch = ParetoArchive()
    for p in pool:
        s = staircase(ev.full(p))
        for wt, t in breakpoints(s, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(p))
    base_hv = -arch.hypervolume(n)

    print(f"nested_dissection -- {args.problem}  n={n}")
    print(f"baseline HV : {base_hv:,.0f}" +
          (f"  gap: {base_hv - target:+,.0f}" if target else ""))
    print(f"restarts={args.restarts}  noise={args.noise}  "
          f"sep_frac={args.sep_frac}  base={args.base}")

    best_hv = base_hv
    n_improved = 0
    t0 = time.time()

    for r in range(args.restarts):
        # vary sep_frac slightly each restart for structural diversity
        sf = args.sep_frac + rng.uniform(-0.06, 0.06)
        sf = float(np.clip(sf, 0.08, 0.38))

        order = nd_order(verts, adj, rng,
                         noise=args.noise,
                         sep_frac=sf,
                         base=args.base)
        assert len(order) == n, f"got {len(order)}, expected {n}"

        hv_before = arch.hypervolume(n)
        s2 = staircase(ev.full(order))
        for wt, t in breakpoints(s2, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(order))
        hv_after = arch.hypervolume(n)

        if hv_after > hv_before:
            new_hv = -hv_after
            if new_hv < best_hv:
                n_improved += 1
                best_hv = new_hv
                gap_str = f"  gap {best_hv - target:+,.0f}" if target else ""
                print(f"  r={r:5d}: HV {best_hv:,.0f}  "
                      f"gain {base_hv - best_hv:+,.0f}{gap_str}",
                      flush=True)
                if target and best_hv <= target:
                    print("*** BEAT leaderboard! ***", flush=True)

        if (r + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{r+1:5d}/{args.restarts}] HV {best_hv:,.0f}  "
                  f"improved={n_improved}  {elapsed:.0f}s "
                  f"({elapsed/(r+1):.1f}s/restart)", flush=True)

    elapsed = time.time() - t0
    print(f"\nfinal HV  : {best_hv:,.0f}")
    print(f"gain      : {base_hv - best_hv:+,.0f}")
    if target:
        print(f"gap       : {best_hv - target:+,.0f}")
    print(f"time      : {elapsed:.0f}s  restarts={args.restarts}  improved={n_improved}")

    if best_hv < base_hv:
        out = submission_path(here, args.problem, "nd")
        top = arch.top_k_by_hv_contribution(20, n)
        write_submission([list(p) + [int(t)] for (_, t, p) in top],
                         args.problem, out)
        print(f"wrote {out}")
    else:
        print("no improvement over baseline pool")


if __name__ == "__main__":
    main()

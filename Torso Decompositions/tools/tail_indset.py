#!/usr/bin/env python3
"""
tail_indset.py  --  Gain HV by placing a large independent set at the end.

Key insight from diagnose_gap.py:
  t*(0) = 1356  (we only achieve width 0 at the very last node)
  Width 0 at threshold t means perm[t..n-1] is an independent set in G.
  The graph is SPARSE (avg deg ~3.4).  Max independent set likely ≥ 300 nodes.
  Putting 181+ independent nodes last → t*(0) ≤ 1176 → +180 HV.

Strategy:
  1. Take our best ordering.
  2. Find a large independent set I in G.
  3. Build a new ordering: non-I nodes first (in their original relative order),
     then I nodes (ordered by min-degree within I to minimise fill-in).
  4. Recompute staircase → check HV gain.
  5. Save improved ordering.

Usage:
    cd "Torso Decompositions"
    python3 tools/tail_indset.py --problem small-graph
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, random
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  hypervolume_2d, load_decision_vectors, submission_path,
                  write_submission, LEADERBOARD_TARGETS)


# ── staircase helper (same as diagnose_gap) ───────────────────────────────────

def staircase(perm, n, ab):
    tmp = list(ab)
    deg = np.zeros(n, dtype=np.int32)
    cur = 0; sm = [0] * n
    for i in range(n - 1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return np.maximum.accumulate(deg[::-1])[::-1]


def pareto_front(W, n):
    front = []; prev_t = n
    for w in range(n):
        idx = np.where(W <= w)[0]
        if not len(idx): continue
        t = int(idx[0])
        if t < prev_t: front.append((w, t)); prev_t = t
        if t == 0: break
    return front


# ── greedy maximum-weight independent set ─────────────────────────────────────

def greedy_indset(n, adj, rng=None, shuffle=True):
    """
    Greedy independent set: always pick the node with fewest remaining neighbours
    (min-degree-first in the residual graph) -- much larger than random greedy.
    """
    if rng is None: rng = random.Random(0)
    deg = [len(adj[v]) for v in range(n)]
    available = set(range(n))
    indset = []
    while available:
        # pick min-degree node in available set
        v = min(available, key=lambda x: deg[x])
        indset.append(v)
        # remove v and all its neighbours
        to_remove = [v] + [u for u in adj[v] if u in available]
        for u in to_remove:
            if u in available:
                available.discard(u)
                for w in adj[u]:
                    deg[w] -= 1
    return indset


def random_indset(n, adj, rng, iters=200):
    """Run greedy with random tie-breaking multiple times, return best."""
    best = []
    for _ in range(iters):
        order = list(range(n)); rng.shuffle(order)
        deg = [len(adj[v]) for v in range(n)]
        available = set(range(n))
        indset = []
        for v in order:
            if v not in available: continue
            indset.append(v)
            available.discard(v)
            for u in adj[v]:
                available.discard(u)
        if len(indset) > len(best): best = indset
    return best


# ── ordering construction ─────────────────────────────────────────────────────

def build_tail_ordering(base_perm, indset_nodes, n, adj, ab):
    """
    New ordering = [base_perm nodes NOT in indset] + [indset nodes ordered by
    min-degree within the indset subgraph].

    The prefix (non-indset nodes) keeps the same relative order as base_perm,
    preserving the existing breakpoints at higher widths.
    """
    indset = set(indset_nodes)
    prefix = [v for v in base_perm if v not in indset]

    # order indset nodes by min-degree within indset subgraph (good for fill-in)
    sub_deg = {v: sum(1 for u in adj[v] if u in indset) for v in indset}
    suffix = sorted(indset_nodes, key=lambda v: sub_deg[v])

    return prefix + suffix


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--indset-iters", type=int, default=500,
                    help="Random restarts for independent set search.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target_hv = -LEADERBOARD_TARGETS[args.problem]
    rng = random.Random(args.seed)

    print(f"\n{'='*60}")
    print(f"  TAIL INDEPENDENT-SET OPTIMISER  --  {args.problem}")
    print(f"  Target HV = {target_hv:,}  (gap budget = 180)")
    print(f"{'='*60}\n")

    # --- load best ordering ---------------------------------------------------
    sub = os.path.join(here, "submissions", args.problem)
    fps = ([os.path.join(sub, "portfolio.json")]
           + sorted(glob.glob(os.path.join(sub, "*.json")))
           + sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))))
    best_perm = None; best_hv = 0
    for fp in fps:
        if not os.path.exists(fp): continue
        dvs = load_decision_vectors(fp)
        if not dvs: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = [int(x) for x in dv[:-1]]
                if sorted(p) != list(range(n)): continue
                W = staircase(p, n, ab)
                fr = pareto_front(W, n)
                hv = hypervolume_2d(fr, n)
                if hv > best_hv: best_hv = hv; best_perm = p

    print(f"Best base ordering HV = {best_hv:,}  (gap = {target_hv - best_hv:.0f})\n")

    # --- find a large independent set ----------------------------------------
    print(f"Searching for large independent set ({args.indset_iters} random restarts)...")

    # greedy min-degree (deterministic, usually best single run)
    indset_det = greedy_indset(n, adj)
    print(f"  Deterministic min-degree greedy: |I| = {len(indset_det)}")

    # randomised
    indset_rand = random_indset(n, adj, rng, iters=args.indset_iters)
    print(f"  Randomised greedy (best of {args.indset_iters}): |I| = {len(indset_rand)}")

    best_indset = indset_det if len(indset_det) >= len(indset_rand) else indset_rand
    print(f"  → using |I| = {len(best_indset)}  "
          f"(max possible t*(0) improvement = {len(best_indset) - 1} HV)\n")

    # verify it's actually independent
    for v in best_indset:
        for u in adj[v]:
            assert u not in set(best_indset), f"Not independent! ({v},{u})"

    # --- build and evaluate new ordering -------------------------------------
    results = []
    for indset_size in sorted({len(best_indset), min(len(best_indset), 250),
                                min(len(best_indset), 200), 181, 100, 50}):
        if indset_size > len(best_indset): continue
        # take the first indset_size nodes from best_indset (already sorted by sub-degree)
        indset_sub = best_indset[:indset_size]
        new_perm = build_tail_ordering(best_perm, indset_sub, n, adj, ab)
        W_new = staircase(new_perm, n, ab)
        fr_new = pareto_front(W_new, n)
        hv_new = hypervolume_2d(fr_new, n)
        t0_new = int(np.where(W_new <= 0)[0][0]) if (W_new == 0).any() else n
        delta = hv_new - best_hv
        print(f"  indset_size={indset_size:4d}  t*(0)={t0_new}  HV={hv_new:,.0f}  Δ={delta:+.0f}")
        results.append((hv_new, new_perm, fr_new))

    # --- save best result ----------------------------------------------------
    best_result = max(results, key=lambda x: x[0])
    best_new_hv, best_new_perm, best_new_front = best_result
    gain = best_new_hv - best_hv

    print(f"\nBest new HV = {best_new_hv:,.0f}  (Δ = {gain:+.0f}, "
          f"gap to target = {target_hv - best_new_hv:.0f})")

    if gain > 0:
        out = submission_path(here, args.problem, "tail_indset")
        # write as a single-solution submission using the Pareto front points
        pairs = [(t, best_new_hv) for (w, t) in best_new_front]
        write_submission([[*best_new_perm, t] for (w, t) in best_new_front],
                         args.problem, out)
        print(f"Saved → {out}")
    else:
        print("No improvement — tail independent-set trick didn't help on this ordering.")
        print("Try: --indset-iters 2000 or run after more GBFC++ rounds.")


if __name__ == "__main__":
    main()

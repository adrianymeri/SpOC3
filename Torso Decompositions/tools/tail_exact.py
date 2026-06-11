#!/usr/bin/env python3
"""
tail_exact.py -- EXACT re-elimination of the low-width tail breakpoints.

For w <= 2 the question "does the filled torso admit an elimination with
every fill degree <= w?" is decidable exactly in polynomial time by complete
reduction rules (Arnborg-Proskurowski):
  w=1: repeatedly remove vertices of degree <= 1 (no fill). Succeeds iff the
       torso is a forest.
  w=2: repeatedly remove vertices of degree <= 2, adding the fill edge for
       degree-2 removals. Succeeds iff treewidth <= 2 (series-parallel).
The removal order IS a width-w elimination order, so every success rewrites
the suffix of an existing banked ordering and provably shifts the (w, t)
breakpoint one step left -- something stochastic search can only approximate.

For each target w and each of the top banked prefixes, we walk t' leftwards
from the current breakpoint, recompute the exact fill state at t' (prefix
p[0:t'] eliminated), run the reduction, and bank every success.

    python3 tools/tail_exact.py --problem small-graph --widths 1,2
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, MAX_TW, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from tools.gbfc import banked
from tools.gbfcpp import staircase, breakpoints, IncEval


def fill_suffix_adj(perm, t, ab, n):
    """Eliminate perm[0:t]; return the filled adjacency (big-int bitsets,
    restricted to the suffix set) plus the suffix list."""
    sm = [0]*n; cur = 0
    for i in range(n-1, -1, -1):
        sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab)
    for i in range(t):
        s = tmp[perm[i]] & sm[i]; x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    suf = perm[t:]
    mask = 0
    for v in suf:
        mask |= 1 << v
    H = {v: (tmp[v] & mask) & ~(1 << v) for v in suf}
    return H, suf


def reduce_order(H, w):
    """Complete reduction for w<=2. Returns elimination order or None."""
    H = {v: int(b) for v, b in H.items()}
    deg = {v: H[v].bit_count() for v in H}
    stack = [v for v in H if deg[v] <= w]
    out = []
    alive = set(H)
    while stack:
        v = stack.pop()
        if v not in alive or deg[v] > w:
            continue
        nb = H[v]
        nbl = []
        x = nb
        while x:
            b = x & -x; x ^= b; nbl.append(b.bit_length()-1)
        alive.discard(v); out.append(v)
        # remove v from neighbours
        for u in nbl:
            H[u] &= ~(1 << v); deg[u] -= 1
        # fill: clique among nbl (only matters for w=2, |nbl|=2)
        if w >= 2 and len(nbl) == 2:
            a, b2 = nbl
            if not (H[a] >> b2) & 1:
                H[a] |= 1 << b2; H[b2] |= 1 << a
                deg[a] += 1; deg[b2] += 1
        for u in nbl:
            if u in alive and deg[u] <= w:
                stack.append(u)
        H[v] = 0
    return out if not alive else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--widths", default="1,2")
    ap.add_argument("--prefixes", type=int, default=12, help="banked prefixes to try")
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(args.problem)
    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
    except Exception:
        ev = IncEval(ab, n)

    pool = banked(here, args.problem, n, ab)[:60]
    W = np.full(n, n, dtype=np.int64)
    stairs = []
    for p in pool:
        s = staircase(ev.full(p)); stairs.append(s)
        W = np.minimum(W, s)
    bps = dict(breakpoints(W, n))
    arch = ParetoArchive()
    for p, s in zip(pool, stairs):
        for wt, t in breakpoints(s, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(p))
    base = -arch.hypervolume(n)
    print(f"tail-exact -- {args.problem}: pooled {base:,.0f}")

    for w in [int(x) for x in args.widths.split(",") if x]:
        if w > 2:
            print(f"w={w}: only w<=2 has complete reduction rules; skipping")
            continue
        t_cur = bps.get(w, None)
        if t_cur is None:
            print(f"w={w}: no current breakpoint"); continue
        print(f"w={w}: current breakpoint t={t_cur}")
        best_t = t_cur
        # rank prefixes: lowest staircase value just left of the breakpoint
        order = sorted(range(len(pool)),
                       key=lambda i: int(stairs[i][max(0, t_cur-1)]))
        for pi in order[:args.prefixes]:
            p = pool[pi]
            t2 = best_t - 1
            improved_with_this_prefix = False
            while t2 >= 0:
                H, suf = fill_suffix_adj(p, t2, ab, n)
                ro = reduce_order(H, w)
                if ro is None:
                    break
                newp = p[:t2] + ro
                s2 = staircase(ev.full(newp))
                assert int(s2[t2]) <= w, "reduction order failed verification"
                for wt, t in breakpoints(s2, n):
                    if wt <= MAX_TW:
                        arch.try_add(wt, t, list(newp))
                p = newp                      # continue pushing with the new tail
                best_t = min(best_t, t2)
                improved_with_this_prefix = True
                t2 -= 1
            if improved_with_this_prefix:
                print(f"  prefix #{pi}: pushed to t={best_t}  (+{t_cur-best_t} cells)")
        print(f"w={w}: final exact breakpoint t={best_t} "
              f"({'optimal for these prefixes' if best_t < t_cur else 'no improvement'})")

    final = -arch.hypervolume(n)
    out = submission_path(here, args.problem, "texact")
    top = arch.top_k_by_hv_contribution(20, n)
    write_submission([list(p)+[int(t)] for (_, t, p) in top], args.problem, out)
    print(f"\npooled+exact: {final:,.0f}  (was {base:,.0f}, gain {base-final:+,.0f})")
    if target is not None:
        print(f"gap to leader: {final-target:+,.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

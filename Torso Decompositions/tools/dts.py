#!/usr/bin/env python3
"""
dts.py -- Degenerate-Tail Synthesis: build front breakpoints combinatorially.

Why. A breakpoint (w, t) requires the suffix S = perm[t:] to be eliminable
with every fill degree <= w. Fill only ADDS edges, so a necessary condition is
that G[S] itself is w-degenerate. Conversely t_w >= n - d_w(G), where d_w(G)
is the maximum size of a w-degenerate induced subgraph (w=0: independent set;
w=1: induced forest; ...). GBFC++ moves 1-2 vertices at a time and cannot jump
to a structurally different S; DTS constructs S directly:

  1. for each target w: local search for a LARGE w-degenerate set S
     (greedy peel + plateau swaps);
  2. order S by repeated min-degree removal (a w-degeneracy order, which for
     w<=1 provably incurs zero internal fill);
  3. order the prefix V\\S to minimise fill INTO S (greedy: eliminate vertices
     with fewest remaining S-neighbour pairs first, min-degree tie-break);
  4. evaluate exactly, archive any cell that beats the pooled staircase, and
     report d_w(found) vs the current n - t_w -- the prospective room.

The constructed orderings land in submissions/<problem>/dts.json (additive);
GBFC++ then repairs/pushes them further (its pool picks them up automatically).

    python3 tools/dts.py --problem small-graph --widths 0,1,2,3,4,5 --budget 60
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, random, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, MAX_TW, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from tools.gbfc import banked
from tools.gbfcpp import staircase, breakpoints, IncEval


def degenerate_set(n, adj, w, budget, rng, S0=None):
    """Local search for a large w-degenerate induced subgraph.
    Greedy peel construction + random restarts + 1-swap plateau moves.
    Returns a set of vertices."""
    deg_full = [len(adj[v]) for v in range(n)]

    def peel():
        """Construct S greedily: while G[R] is not w-degenerate, delete a
        highest-degree vertex of its non-degenerate core. O(n^2)-ish."""
        R = set(range(n))
        while True:
            # peel the w-degenerate fringe; what remains is the core
            tmp = set(R)
            deg = {v: sum(1 for u in adj[v] if u in tmp) for v in tmp}
            stack = [v for v in tmp if deg[v] <= w]
            while stack:
                v = stack.pop()
                if v not in tmp:
                    continue
                tmp.discard(v)
                for u in adj[v]:
                    if u in tmp:
                        deg[u] -= 1
                        if deg[u] == w:
                            stack.append(u)
            if not tmp:                  # R is w-degenerate
                return R
            # delete a max-core-degree vertex (random tie-break)
            mx = max(deg[v] for v in tmp)
            cands = [v for v in tmp if deg[v] == mx]
            R.discard(cands[rng.randrange(len(cands))])

    best = peel()
    if S0:
        best = max(best, set(S0), key=len)
    t0 = time.time()
    S = set(best)
    while time.time() - t0 < budget:
        # plateau move: remove 1 random member, try to add 2 outsiders
        S2 = set(S)
        if S2 and rng.random() < 0.7:
            S2.discard(random.choice(tuple(S2)))
        outs = [v for v in range(n) if v not in S2]
        rng.shuffle(outs)
        added = 0
        for v in outs[:400]:
            S3 = S2 | {v}
            # quick check: v's degree into S2 and w-degeneracy via peeling test
            if sum(1 for u in adj[v] if u in S2) <= w and _is_wdeg(S3, adj, w):
                S2 = S3; added += 1
                if added >= 2:
                    break
        if len(S2) > len(S):
            S = S2
            if len(S) > len(best):
                best = set(S)
        elif len(S2) == len(S) and rng.random() < 0.5:
            S = S2                      # drift on plateaus
    return best


def _is_wdeg(S, adj, w):
    """Is G[S] w-degenerate? (repeated low-degree peeling)"""
    tmp = set(S)
    deg = {v: sum(1 for u in adj[v] if u in tmp) for v in tmp}
    stack = [v for v in tmp if deg[v] <= w]
    while stack:
        v = stack.pop()
        if v not in tmp: continue
        tmp.discard(v)
        for u in adj[v]:
            if u in tmp:
                deg[u] -= 1
                if deg[u] == w:
                    stack.append(u)
    return not tmp


def order_suffix(S, adj, w):
    """w-degeneracy elimination order of S (each vertex <= w later members)."""
    tmp = set(S); out = []
    deg = {v: sum(1 for u in adj[v] if u in tmp) for v in tmp}
    import heapq
    h = [(deg[v], v) for v in tmp]; heapq.heapify(h)
    while h:
        d, v = heapq.heappop(h)
        if v not in tmp or deg[v] != d:
            continue
        tmp.discard(v); out.append(v)
        for u in adj[v]:
            if u in tmp:
                deg[u] -= 1
                heapq.heappush(h, (deg[u], u))
    return out


def order_prefix(P, S, adj):
    """Order V\\S to minimise fill into S: greedily eliminate the vertex whose
    remaining neighbourhood contains the fewest S-pairs (then min-degree)."""
    import heapq
    P = set(P); out = []
    alive = set(P) | set(S)
    while P:
        bestv, bestkey = None, None
        for v in P:
            nb = [u for u in adj[v] if u in alive]
            s_nb = sum(1 for u in nb if u in S)
            key = (s_nb * (s_nb - 1) // 2, len(nb))
            if bestkey is None or key < bestkey:
                bestkey, bestv = key, v
        P.discard(bestv); alive.discard(bestv); out.append(bestv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--widths", default="0,1,2,3,4,5")
    ap.add_argument("--budget", type=float, default=60.0, help="LS seconds per width")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    here = repo_root()
    rng = random.Random(args.seed)
    n, adj = load_graph(graph_path(here, args.problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(args.problem)
    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
    except Exception:
        ev = IncEval(ab, n)

    pool = banked(here, args.problem, n, ab)[:60]
    W = np.full(n, n, dtype=np.int64)
    for p in pool:
        W = np.minimum(W, staircase(ev.full(p)))
    bps = dict(breakpoints(W, n))
    arch = ParetoArchive()
    for p in pool:
        ws = staircase(ev.full(p))
        for wt, t in breakpoints(ws, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(p))
    base = -arch.hypervolume(n)
    print(f"DTS -- {args.problem}: pooled front {base:,.0f}")
    print(f"{'w':>3} {'cur t_w':>8} {'cur |S|':>8} {'found |S|':>9} {'room':>6} {'realised t':>10}")

    for w in [int(x) for x in args.widths.split(",") if x != ""]:
        t_cur = bps.get(w)
        cur_sz = (n - t_cur) if t_cur is not None else 0
        # warm-start the degenerate-set LS from the current suffix
        S0 = None
        if t_cur is not None:
            bestp = min(pool, key=lambda p: int(staircase(ev.full(p))[min(t_cur, n-1)]))
            S0 = bestp[t_cur:]
        S = degenerate_set(n, adj, w, args.budget, rng, S0=S0)
        room = len(S) - cur_sz
        # synthesize the ordering and evaluate exactly
        suf = order_suffix(S, adj, w)
        pre = order_prefix(set(range(n)) - set(S), S, adj)
        perm = pre + suf
        ws = staircase(ev.full(perm))
        # realized: smallest t with width <= w in the synthesized ordering
        ok = np.where(ws <= w)[0]
        t_real = int(ok[0]) if len(ok) else -1
        for wt, t in breakpoints(ws, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(perm))
        print(f"{w:>3} {t_cur if t_cur is not None else -1:>8} {cur_sz:>8} "
              f"{len(S):>9} {room:>+6} {t_real:>10}", flush=True)

    final = -arch.hypervolume(n)
    out = submission_path(here, args.problem, "dts")
    top = arch.top_k_by_hv_contribution(20, n)
    write_submission([list(p)+[int(t)] for (_, t, p) in top], args.problem, out)
    print(f"\npooled+DTS: {final:,.0f}  (was {base:,.0f}, gain {base-final:+,.0f})")
    if target is not None:
        print(f"gap to leader: {final-target:+,.0f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

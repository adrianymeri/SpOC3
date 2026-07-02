#!/usr/bin/env python3
"""quota_repair.py -- constructor v2: per-width class-QUOTA repair.

The planted twin classes obey a closed-form inclusion law: at torso width w, a
class c (a clique of |c| twins with e_c external neighbours) can contribute at
most ~ w+1-e_c members to the torso.  The evolved front saturates this bound at
the top widths (325/375 @ w449, 275/375 @ w399 -- exact) but sits 2-5 members
short at mid widths and is non-monotone in the mid range: named free vertices.

This tool takes each capped-20 owner ordering and, for every big class below
quota, moves 1..slack head-resident members across the breakpoint (inserted at
the torso boundary or a little deeper), re-evaluates the exact staircase, and
accepts on strict capped-20 improvement (saved incrementally).

    python3 tools/quota_repair.py --problem large-graph --seed 1
"""
from __future__ import annotations
import argparse, collections, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path
from tools.fastwalk import IncEvalC
from tools.archive_evolve import (order_front, capped_hv, load_seed_orders,
                                  LEADERBOARD_TARGETS)


def big_classes(n, adj, min_size=8):
    h = collections.defaultdict(list)
    for v in range(n):
        h[frozenset(adj[v] | {v})].append(v)
    out = []
    for c in h.values():
        if len(c) >= min_size:
            e = len(adj[c[0]]) - (len(c) - 1)      # external neighbourhood size
            out.append((sorted(c), e))
    return sorted(out, key=lambda x: -len(x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--max-move", type=int, default=8,
                    help="max members moved across the breakpoint per attempt")
    ap.add_argument("--offsets", default="0,4,16,64",
                    help="insertion depths past the breakpoint to try")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem))
    ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(a.problem)
    rng = random.Random(a.seed)
    offsets = [int(x) for x in a.offsets.split(",")]

    cls = big_classes(n, adj)
    print(f"=== quota-repair {a.problem} | {len(cls)} big classes "
          f"{[(len(c), e) for c, e in cls]} ===", flush=True)

    members = load_seed_orders(here, a.problem, n, ev, a.pool)
    cur, top = capped_hv(members, n)
    print(f"    corpus capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''}", flush=True)

    def save(top):
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", a.problem, "quota_repair.json")
        json.dump({"challenge": "spoc-3-torso-decompositions",
                   "problem": a.problem, "decisionVector": dvs}, open(out, "w"))

    accepts = tries = 0
    t0 = time.time()
    owners = [(list(p), int(t)) for (w, t, p) in top]
    for oi, (perm, t) in enumerate(owners):
        torso = set(perm[t:])
        # owner's width at its own breakpoint (min width at torso >= this size)
        own_w = min(w for (w, tt, _) in
                    [(pt[0], pt[1], None) for pt in order_front(perm, ev, n)]
                    if tt <= t)
        for c, e in cls:
            used = sum(1 for v in c if v in torso)
            quota = max(0, min(len(c), own_w + 1 - e))
            slack = quota - used
            if slack <= 0:
                continue
            head_members = [v for v in perm[:t] if v in set(c)]
            if not head_members:
                continue
            for k in range(1, min(slack, a.max_move, len(head_members)) + 1):
                mv = head_members[:k]
                base = [v for v in perm if v not in set(mv)]
                for off in offsets:
                    j = min(t - k + off, len(base))
                    cand = base[:j] + mv + base[j:]
                    pts = order_front(cand, ev, n)
                    tries += 1
                    trial = members + [(cand, pts)]
                    h, top2 = capped_hv(trial, n)
                    if h < cur - 0.5:
                        cur = h; accepts += 1
                        keep = set(tuple(p) for (_, _, p) in top2)
                        members = [(p, x) for (p, x) in trial if tuple(p) in keep]
                        if (cand, pts) not in members:
                            members.append((cand, pts))
                        print(f"  owner {oi} (t={t}, w={own_w}) class|{len(c)}| "
                              f"+{k}@{off}: *** capped-20 {cur:,.0f}"
                              f"{f'  gap {cur-target:+,.0f}' if target else ''} "
                              f"(accept #{accepts}) ***", flush=True)
                        save(top2)
        print(f"  [owner {oi+1}/{len(owners)} done  {tries} tries  {accepts} accepts  "
              f"{time.time()-t0:.0f}s]", flush=True)
    print(f"\nfinal capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | {accepts} quota-repair accepts",
          flush=True)


if __name__ == "__main__":
    main()

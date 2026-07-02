#!/usr/bin/env python3
"""hri_lns.py -- multi-objective Large Neighborhood Search with the SpOC-3
winners' own operators (S. Limmer, personal communication, July 2026, cited
with permission), run under OUR exact capped-20 objective.

Operator set B (primary, "until no further progress"):
  neighbor destroy  -- remove a seed vertex and its original-graph neighbours
                       from the permutation (large destroy size);
  balanced repair   -- reinsert each vertex at the MEDIAN position of its
                       already-inserted neighbours (cf. Sec. 5 of the DAM paper
                       Limmer references).
Operator set A (fallback after stall):
  random destroy    -- remove a few random vertices (small destroy size);
  random repair     -- reinsert at random positions.

Acceptance: strict improvement of the exact capped-20 hypervolume over the
pooled archive (saved incrementally on every win).

    python3 tools/hri_lns.py --problem large-graph --iters 500000 --seed 1
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path
from tools.fastwalk import IncEvalC
from tools.archive_evolve import (order_front, capped_hv, load_seed_orders,
                                  LEADERBOARD_TARGETS)


def neighbor_destroy(perm, adj, rng, cap):
    seed = rng.randrange(len(perm))
    v = perm[seed]
    d = {v} | set(adj[v])
    if len(d) > cap:
        d = {v} | set(rng.sample(list(adj[v]), cap - 1))
    return d


def balanced_repair(perm, destroyed, adj, rng):
    """MEDIAN PLACEMENT (Biedl et al., DAM 148, 2005, Sec. 5), faithful:
    k even -> insert between w_{k/2} and w_{k/2+1};
    k odd  -> immediately before/after the median neighbour, choosing the side
              that minimises the median neighbour's imbalance (Lemma 16)."""
    base = [u for u in perm if u not in destroyed]
    pos = {u: i for i, u in enumerate(base)}
    order = sorted(destroyed, key=lambda u: -len(adj[u]))   # insertion ordering
    for u in order:
        nb = sorted(pos[x] for x in adj[u] if x in pos)
        k = len(nb)
        if k == 0:
            j = rng.randint(0, len(base))
        elif k % 2 == 0:
            j = nb[k // 2]                     # any slot in (w_{k/2}, w_{k/2+1}]
        else:
            mid = nb[k // 2]
            w = base[mid]
            wn = [pos[x] for x in adj[w] if x in pos]
            pred = sum(1 for x in wn if x < mid)
            succ = len(wn) - pred
            # inserting u BEFORE w adds a predecessor to w; AFTER adds a successor
            j = mid if pred + 1 - succ <= succ + 1 - pred else mid + 1
        base.insert(min(j, len(base)), u)
        pos = {x: i for i, x in enumerate(base)}
    return base


def random_destroy_repair(perm, rng, k):
    p = list(perm)
    idx = sorted(rng.sample(range(len(p)), k), reverse=True)
    moved = [p.pop(i) for i in idx]
    for u in moved:
        p.insert(rng.randint(0, len(p)), u)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--iters", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--destroy-cap", type=int, default=220,
                    help="max neighbor-destroy size (large avg degree is 209)")
    ap.add_argument("--small-k", type=int, default=6,
                    help="random destroy size for operator set A")
    ap.add_argument("--stall", type=int, default=3000,
                    help="iterations without accept before switching B -> A (and back)")
    a = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj_sets = load_graph(graph_path(here, a.problem))
    adj = [sorted(s) for s in adj_sets]
    ab = build_adj_bitsets(n, adj_sets)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(a.problem)
    rng = random.Random(a.seed)

    members = load_seed_orders(here, a.problem, n, ev, a.pool)
    cur, top = capped_hv(members, n)
    print(f"=== HRI-LNS {a.problem} | capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | "
          f"B: neighbor-destroy(<= {a.destroy_cap}) + balanced-repair; "
          f"A: random k={a.small_k} ===", flush=True)

    accepts = 0
    since = 0
    mode = "B"
    t0 = time.time()
    for it in range(a.iters):
        src = list(members[rng.randrange(len(members))][0])
        if mode == "B":
            d = neighbor_destroy(src, adj, rng, a.destroy_cap)
            cand = balanced_repair(src, d, adj, rng)
        else:
            cand = random_destroy_repair(src, rng, a.small_k)
        pts = order_front(cand, ev, n)
        trial = members + [(cand, pts)]
        h, top2 = capped_hv(trial, n)
        since += 1
        if h < cur - 0.5:
            cur = h; accepts += 1; since = 0
            keep = set(tuple(p) for (_, _, p) in top2)
            members = [(p, x) for (p, x) in trial if tuple(p) in keep]
            if (cand, pts) not in members:
                members.append((cand, pts))
            print(f"  it {it} [{mode}]: *** capped-20 {cur:,.0f}"
                  f"{f'  gap {cur-target:+,.0f}' if target else ''} "
                  f"(accept #{accepts}) ***", flush=True)
            dvs = [list(p) + [int(t)] for (_, t, p) in top2]
            out = os.path.join(here, "submissions", a.problem, "hri_lns.json")
            json.dump({"challenge": "spoc-3-torso-decompositions",
                       "problem": a.problem, "decisionVector": dvs}, open(out, "w"))
        if since >= a.stall:
            mode = "A" if mode == "B" else "B"
            since = 0
            print(f"  [stall -> switching to operator set {mode}]", flush=True)
        if it % 500 == 0 and it:
            print(f"  [it {it}/{a.iters} [{mode}] capped-20 {cur:,.0f}  "
                  f"{accepts} accepts  {time.time()-t0:.0f}s]", flush=True)
    print(f"\nfinal capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | {accepts} LNS accepts",
          flush=True)


if __name__ == "__main__":
    main()

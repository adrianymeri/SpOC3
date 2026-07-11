#!/usr/bin/env python3
"""
clique_prefix.py -- Lever A: deterministic clique-prefix construction for large-graph.

The glue of large-graph is exactly three vertex-disjoint cliques K500 (375-twin
class @499), K400 (102 @399), K300 (62 @299).  The packing bound (docs/
PLANTED_STRUCTURE_CERTIFICATES.md) says a torso of width <= w requires a head
with >= (499-w) vertices of K500, (399-w) of K400, (299-w) of K300 -- and the
envelope meets this bound exactly at w in {299,332,365,399,449,499}.

This tool CONSTRUCTS the bound solution for every target w in [w-min, w-max):
  head = quota twins of each class (zero-fill inside a clique), then clique
  externals ranked by outside-degree (v0 heuristic; --rank gbdt hooks in a
  LightGBM ranker later), tail = a template achiever ordering restricted to
  the remaining vertices.  Every breakpoint of every constructed ordering is
  banked; the best 60 by HV contribution are written to
  submissions/large-graph/clique_prefix.json (cap_submit pools it from there).

    python3 tools/clique_prefix.py --wmin 125 --wmax 299 --templates cap20.json
"""
from __future__ import annotations
import argparse, collections, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def find_cliques(n, adj):
    """The three planted cliques = closed nbhds of the big true-twin classes."""
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    out = []
    for vs in h.values():
        if len(vs) >= 50:                      # 375 / 102 / 62
            K = set(vs) | set(adj[vs[0]])
            out.append((sorted(vs), sorted(K)))
    out.sort(key=lambda ck: -len(ck[1]))       # K500, K400, K300
    assert [len(k) for _, k in out] == [500, 400, 300], \
        f"unexpected clique sizes {[len(k) for _, k in out]}"
    ks = [set(k) for _, k in out]
    assert not (ks[0] & ks[1]) and not (ks[0] & ks[2]) and not (ks[1] & ks[2])
    return out                                  # [(twins, clique), ...] big->small


def head_for_width(w, cliques, adj, rank="outdeg"):
    """Quota head for target width w: twins first, then ranked externals."""
    degs = [499, 399, 299]
    head = []
    for (twins, K), d in zip(cliques, degs):
        q = max(0, d - w)
        Kset = set(K)
        take = list(twins[:q])
        if q > len(twins):
            ext = [v for v in K if v not in set(twins)]
            if rank == "outdeg":                # v0: fewest edges out of the clique
                ext.sort(key=lambda v: (len(adj[v] - Kset), len(adj[v])))
            take = list(twins) + ext[:q - len(twins)]
        head.append(take)
    # order: all twins (largest clique first), then externals by outside-degree
    tw_part = [v for part, (twins, K) in zip(head, cliques)
               for v in part if v in set(twins)]
    Kall = {}
    for (twins, K) in cliques:
        for v in K:
            Kall[v] = set(K)
    ex_part = [v for part in head for v in part if v not in set(tw_part)]
    ex_part.sort(key=lambda v: len(adj[v] - Kall[v]))
    return tw_part + ex_part


def load_templates(names, n):
    perms = []
    for name in names:
        fp = os.path.join(HERE, "submissions", PROBLEM, name)
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1:
                    p = [int(x) for x in dv[:-1]]
                    if sorted(p) == list(range(n)):
                        perms.append((int(dv[-1]), p))
        except Exception as ex:
            print(f"  template {name}: skipped ({ex})")
    # prefer low-t (big-torso) templates; dedupe by t
    seen, out = set(), []
    for t, p in sorted(perms):
        if t not in seen:
            seen.add(t); out.append((t, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmin", type=int, default=125)
    ap.add_argument("--wmax", type=int, default=299)
    ap.add_argument("--templates", default="cap20.json",
                    help="comma-separated submission files to use as tail templates")
    ap.add_argument("--max-templates", type=int, default=8)
    ap.add_argument("--rank", default="outdeg", choices=["outdeg", "random"])
    ap.add_argument("--algo", default="clique_prefix")
    a = ap.parse_args()

    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ab = build_adj_bitsets(n, adj_l)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS[PROBLEM]
    cliques = find_cliques(n, adj)
    print(f"cliques: {[len(k) for _, k in cliques]} | twins: {[len(t) for t, _ in cliques]}")

    templates = load_templates(a.templates.split(","), n)[:a.max_templates]
    print(f"{len(templates)} tail templates (t = {[t for t, _ in templates]})")

    arc = ParetoArchive()
    t0 = time.time()
    results = []
    for w in range(a.wmin, a.wmax):
        H = head_for_width(w, cliques, adj, a.rank)
        Hset = set(H); tpos = len(H)
        best = None
        for tt, tp in templates:
            perm = H + [v for v in tp if v not in Hset]
            df = ev.full(perm)
            if max(int(x) for x in df) > MAX_TW:
                continue                        # over-width somewhere: invalid
            r = 0
            for t in range(n - 1, -1, -1):
                c = int(df[t]); r = c if c > r else r
                if r <= MAX_TW:
                    arc.try_add(r, t, perm)
            wt = max(int(df[i]) for i in range(tpos, n))
            if best is None or wt < best:
                best = wt
        results.append((w, tpos, best))
        if w % 20 == 0 or w == a.wmax - 1:
            print(f"  w={w:3d} quota t={tpos:3d} achieved width={best} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    hv = -arc.hypervolume(n)
    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(x, t) for x, t, _ in top], n)
    print(f"\nconstructed-archive alone: envelope {hv:,.0f} | cap20 {cap:,.0f} "
          f"(target {target:,.0f})")
    print("\n  w | quota-t | achieved-width (bound attained iff == w)")
    hit = sum(1 for w, tp, b in results if b is not None and b <= w)
    for w, tp, b in results:
        if b is not None and (b <= w or w % 10 == 0):
            print(f" {w:4d} | {tp:4d} | {b}")
    print(f"\nbound attained at {hit}/{len(results)} widths")

    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")
    top60 = arc.top_k_by_hv_contribution(60, n)
    write_submission([list(p) + [int(t)] for (_, t, p) in top60], PROBLEM, out)
    print(f"wrote {out}  ({len(top60)} points) -- now run cap_submit to re-pool")


if __name__ == "__main__":
    main()

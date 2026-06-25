#!/usr/bin/env python3
r"""
clique_break.py -- obstruction-guided search for a larger width-w torso.

Manual analysis (this session) showed that growing our width-w torso by its best
boundary vertex always welds a **(w+2)-clique** into the torso, which forces
treewidth >= w+1 (a k-clique has treewidth k-1). A clique cannot be undone by
re-ordering, so the *only* way to grow band w is a size-(T_w+1) torso whose fill
forms **no** (w+2)-clique anywhere. This tool targets exactly that: each round it
locates the obstruction clique and tries to *break it* with a net-+1 restructure
(drop clique vertices, add non-clique low-boundary vertices), verifying every
candidate with exact branch-and-bound treewidth. Any acceptance is a proven +1 HV.

This is informed search, not random: it spends its exact-verification budget only
on candidates that have a chance of dodging the known obstruction.

    python3 tools/clique_break.py --problem small-graph --bands 8,9,10 \
        --budget 14400 --tw-timeout 3.0 --kick 3

Best on the lower high-bands (8-10) where exact treewidth is tractable.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(400000)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
import tools.torso_deletion as td
from tools.fastwalk import IncEvalC


def adjdict(Slist, ab, n):
    tor, _ = td.torso_adj(Slist, ab, n); a = {s: set() for s in Slist}
    for s in Slist:
        x = tor[s]
        while x:
            b = x & -x; x &= x - 1; u = b.bit_length() - 1
            if u in a: a[s].add(u)
    return a


def tw_le(a0, k, dl):
    def rec(a):
        if time.time() > dl: raise TimeoutError
        ch = True
        while ch:
            ch = False
            for v in list(a):
                ns = a[v]
                if len(ns) <= k and all(ns - {b} <= a[b] for b in ns):
                    for b in ns: a[b] |= ns; a[b].discard(b); a[b].discard(v)
                    del a[v]; ch = True; break
        if not a: return True
        if min(len(a[v]) for v in a) > k: return False
        v = min(a, key=lambda u: len(a[u])); ns = a[v]
        if len(ns) > k: return False
        a2 = {x: set(y) for x, y in a.items()}
        for b in ns: a2[b] |= ns; a2[b].discard(b); a2[b].discard(v)
        del a2[v]; return rec(a2)
    return rec({x: set(y) for x, y in a0.items()})


def max_clique(a):
    """greedy maximum clique over adjacency dict a (set values)."""
    deg = {s: len(a[s]) for s in a}; best = []
    for seed in sorted(a, key=lambda s: -deg[s])[:30]:
        cand = set(a[seed]); clq = [seed]
        while cand:
            x = max(cand, key=lambda u: len(a[u] & cand))
            if all(x in a[c] for c in clq): clq.append(x); cand &= a[x]
            else: cand.discard(x)
        if len(clq) > len(best): best = clq
    return best


def run(problem, here, bands, budget_s, tw_timeout, kick, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; d = ev.full(perm); r = 0
                    for t in range(n - 1, -1, -1):
                        c = int(d[t]); r = c if c > r else r
                        if r <= MAX_TW: arc.try_add(r, t, perm)
        except Exception:
            continue
    by_w = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, list(p))

    def front_hv():
        a = ParetoArchive()
        for w in by_w: a.try_add(w, by_w[w][0], None)
        return -hypervolume_2d(a.points(), n)
    print(f"=== clique-break {problem} | front {front_hv():,.0f}"
          f"{f'  gap {front_hv()-target:+,.0f}' if target else ''} ===", flush=True)

    rng = np.random.default_rng(seed); t0 = time.time(); wins = 0; full = set(range(n))
    for W in bands:
        if W not in by_w: continue
        t_star, perm = by_w[W]; S = set(perm[t_star:]); best = len(S)
        # the obstruction clique on the canonical single-vertex grow
        Sm = 0
        for s in S: Sm |= 1 << s
        v0 = min((u for u in range(n) if not (Sm >> u) & 1),
                 key=lambda u: (ab[u] & Sm).bit_count())
        C0 = max_clique(adjdict(list(S) + [v0], ab, n))
        print(f"  w={W}: T_w={len(S)} | obstruction clique size {len(C0)} "
              f"(must avoid {W+2}-cliques)", flush=True)
        tested = to = 0; tw0 = time.time()
        while time.time() - t0 < budget_s and time.time() - tw0 < budget_s / max(1, len(bands)):
            # recompute the live obstruction clique on the current best+grow
            Sm = 0
            for s in S: Sm |= 1 << s
            vg = min((u for u in range(n) if not (Sm >> u) & 1),
                     key=lambda u: (ab[u] & Sm).bit_count())
            C = set(max_clique(adjdict(list(S) + [vg], ab, n)))
            # net-+1 clique-break: drop `kick` clique vertices, add `kick+1`
            # low-boundary NON-clique vertices
            dropc = [c for c in C if c in S]
            if not dropc: dropc = list(S)
            drop = set(int(x) for x in rng.choice(dropc, size=min(kick, len(dropc)),
                                                  replace=False))
            base = S - drop; bm = 0
            for s in base: bm |= 1 << s
            outs = sorted([u for u in range(n) if u not in base and u not in C],
                          key=lambda u: (ab[u] & bm).bit_count())
            pool = outs[:max(kick * 4, 16)]; rng.shuffle(pool)
            add = pool[:kick + 1]
            cand = base | set(int(x) for x in add)
            if len(cand) <= best:
                tested += 1; continue
            # quick reject: does it still contain a (W+2)-clique?
            ad = adjdict(list(cand), ab, n)
            if len(max_clique(ad)) >= W + 2:
                tested += 1; continue
            try:
                if tw_le(ad, W, time.time() + tw_timeout):
                    best = len(cand); S = set(cand)
                    permnew = [u for u in full if u not in S] + list(S)
                    by_w[W] = (n - len(S), permnew); wins += 1
                    c = front_hv()
                    print(f"  w={W}: *** WIN clique-broken torso {best} *** front {c:,.0f}"
                          f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)
                tested += 1
            except TimeoutError:
                to += 1
        print(f"  w={W}: done  best {best} (was {n-t_star})  "
              f"[{tested} clique-guided exact tests, {to} timeouts, {time.time()-tw0:.0f}s]",
              flush=True)

    fin = front_hv()
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | {wins} bands improved")
    if wins:
        a = ParetoArchive()
        for w in by_w:
            if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
        top = a.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "clique_break.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--bands", default="8,9,10")
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--tw-timeout", type=float, default=3.0)
    ap.add_argument("--kick", type=int, default=3, help="clique vertices to drop per net-+1 move")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        bands, a.budget, a.tw_timeout, a.kick, a.seed)


if __name__ == "__main__":
    main()

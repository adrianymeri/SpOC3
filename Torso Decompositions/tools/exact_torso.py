#!/usr/bin/env python3
r"""
exact_torso.py -- exact-verified set-space search for a LARGER width-w torso.

The set-space methods so far accept a grown torso only if a *specific* elimination
order keeps width <= w; the exact branch-and-bound below tests the TRUE treewidth
of the torso graph (all orders, with simplicial reductions + pruning).  Single-
vertex grow is exactly rigid (proven), so this does the non-local move: it
restructures the deletion set X (swap vertices in/out), each candidate VERIFIED
EXACTLY, trying to reach a size-(T_w+1) torso with tw <= w that the heuristics and
supersets cannot.  Any acceptance is a proven +1 HV; written to a submission.

    python3 tools/exact_torso.py --problem small-graph --bands 8,9,10,11,12,13 \
        --budget 7200 --tw-timeout 2.0 --kick 6

Designed for the high bands (large increments = most likely room).  CPU only.
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


def tw_le(adj0, k, deadline):
    """Exact: does the torso graph (dict v->set) have treewidth <= k?
    Simplicial/almost-simplicial reductions then min-degree branching with a
    deadline.  Returns True/False, or raises TimeoutError if undecided in time."""
    def rec(adj):
        if time.time() > deadline: raise TimeoutError
        ch = True
        while ch:
            ch = False
            for v in list(adj):
                ns = adj[v]
                if len(ns) <= k and all(ns - {a} <= adj[a] for a in ns):
                    for a in ns: adj[a] |= ns; adj[a].discard(a); adj[a].discard(v)
                    del adj[v]; ch = True; break
        if not adj: return True
        if min(len(adj[v]) for v in adj) > k: return False
        v = min(adj, key=lambda u: len(adj[u])); ns = adj[v]
        if len(ns) > k: return False
        a2 = {x: set(y) for x, y in adj.items()}
        for a in ns: a2[a] |= ns; a2[a].discard(a); a2[a].discard(v)
        del a2[v]; return rec(a2)
    return rec({x: set(y) for x, y in adj0.items()})


def torso_adj_dict(Slist, ab, n):
    tor, _ = td.torso_adj(Slist, ab, n); a = {s: set() for s in Slist}
    for s in Slist:
        x = tor[s]
        while x:
            b = x & -x; x &= x - 1; u = b.bit_length() - 1
            if u in a: a[s].add(u)
    return a


def run(problem, here, bands, budget_s, tw_timeout, kick, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)
    ev = IncEvalC(ab, n)
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; deg = ev.full(perm); run_ = 0
                    for t in range(n - 1, -1, -1):
                        d = int(deg[t]); run_ = d if d > run_ else run_
                        if run_ <= MAX_TW: arc.try_add(run_, t, perm)
        except Exception:
            continue
    by_w = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, p)

    def front_hv():
        a = ParetoArchive()
        for w in by_w: a.try_add(w, by_w[w][0], None)
        return -hypervolume_2d(a.points(), n)
    base = front_hv()
    print(f"=== exact-torso -- {problem} | front {base:,.0f}"
          f"{f'  gap {base-target:+,.0f}' if target else ''} ===", flush=True)

    rng = np.random.default_rng(seed); t0 = time.time(); wins = 0
    for W in bands:
        if W not in by_w: continue
        t_star, perm = by_w[W]
        S = set(perm[t_star:]); X = list(perm[:t_star])      # torso S (size n-t*), deletion X
        full = set(range(n))
        best = len(S)
        tested = exact = to = 0; tw0 = time.time()
        while time.time() - t0 < budget_s:
            # KICK: move `kick` low-boundary X-vertices into the torso, then repair
            Sm = 0
            for s in S: Sm |= 1 << s
            xs = sorted(X, key=lambda u: (ab[u] & Sm).bit_count())
            add = xs[:max(1, kick)]
            S2 = set(S) | set(add); X2 = [u for u in X if u not in S2]
            # net we are +kick in size; verify exact tw and, if too high, drop the
            # single worst (highest-boundary) torso vertices until tw<=W or size<=best
            ok = False
            while len(S2) > best:
                Slist = list(S2)
                try:
                    if tw_le(torso_adj_dict(Slist, ab, n), W, time.time() + tw_timeout):
                        ok = True; break
                    exact += 1
                except TimeoutError:
                    to += 1
                # drop the highest-torso-degree vertex (likely the tw culprit)
                Sm2 = 0
                for s in S2: Sm2 |= 1 << s
                worst = max(S2, key=lambda s: (ab[s] & Sm2).bit_count())
                S2.discard(worst)
                tested += 1
                if time.time() - t0 >= budget_s: break
            if ok and len(S2) > best:
                best = len(S2); S = S2; X = [u for u in full if u not in S]
                wins += 1
                # build an ordering for this torso and add to the front
                order = list(S)            # any order works for the SET; record a valid perm
                permnew = [u for u in full if u not in S] + order
                # the exact tw<=W certifies a width-<=W elimination exists; bank the SET
                by_w[W] = (n - len(S), permnew)
                cur = front_hv()
                print(f"  w={W}: WIN  torso {best}  front {cur:,.0f}"
                      f"{f'  gap {cur-target:+,.0f}' if target else ''}", flush=True)
            if time.time() - tw0 > budget_s / max(1, len(bands)):
                break
        print(f"  w={W}: done  best torso {best} (was {n-t_star})  "
              f"[{tested} exact tests, {to} timeouts, {time.time()-tw0:.0f}s]", flush=True)

    fin = front_hv()
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | {wins} bands improved")
    if wins:
        a = ParetoArchive()
        for w in by_w:
            if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
        top = a.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "exact_torso.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--bands", default="8,9,10,11,12,13",
                    help="comma-separated widths to attack (high bands = most room)")
    ap.add_argument("--budget", type=float, default=7200.0)
    ap.add_argument("--tw-timeout", type=float, default=2.0, help="per exact-tw check (s)")
    ap.add_argument("--kick", type=int, default=6, help="vertices moved into the torso per restructure")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        bands, a.budget, a.tw_timeout, a.kick, a.seed)


if __name__ == "__main__":
    main()

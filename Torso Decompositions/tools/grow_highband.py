#!/usr/bin/env python3
r"""
grow_highband.py -- search the HIGH bands with a SCALABLE constructive width
check (exact B&B times out past ~700 vertices, so exact_torso never actually
searched w>=11).

For a candidate torso S, a min-degree elimination ordering gives a constructive
UPPER bound on its width, computed fast on 1000+ vertex torsos and aborted the
instant the running width exceeds the band w.  If a torso LARGER than our current
best admits a width-<=w min-degree ordering, that ordering is a *direct proof* of
a bigger torso -> +1 HV.  We try to shrink the deletion set X (move vertices into
the torso) at each high band, re-ordering with min-degree each time.

    python3 tools/grow_highband.py --problem small-graph --bands 11,12,13,14 \
        --budget 14400 --kick 6

CPU only; complements exact_torso (which owns the low bands).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
import tools.torso_deletion as td
from tools.fastwalk import IncEvalC


def mindeg_width(Slist, ab, n, cap):
    """Min-degree elimination width of torso(Slist); abort once width > cap.
    Returns (width, order) where order is the elimination order if width<=cap,
    else (width, None).  Constructive: a returned order proves width<=cap."""
    tor, _ = td.torso_adj(Slist, ab, n)
    adjm = {s: tor[s] for s in Slist}
    alive = 0
    for s in Slist: alive |= 1 << s
    deg = {s: (adjm[s] & alive).bit_count() for s in Slist}
    width = 0; order = []
    rem = len(Slist)
    while rem:
        # min-degree vertex among alive
        best = -1; bestd = 1 << 30
        for v in Slist:
            if (alive >> v) & 1:
                d = deg[v]
                if d < bestd:
                    bestd = d; best = v
                    if d == 0: break
        if bestd > width:
            width = bestd
            if width > cap:
                return width, None
        order.append(best)
        nb = adjm[best] & alive
        alive &= ~(1 << best)
        # fill: make nb a clique, refresh affected degrees
        y = nb
        while y:
            b = y & -y; y &= y - 1; u = b.bit_length() - 1
            adjm[u] |= nb; adjm[u] &= ~(1 << u); adjm[u] &= ~(1 << best)
        y = nb
        while y:
            b = y & -y; y &= y - 1; u = b.bit_length() - 1
            deg[u] = (adjm[u] & alive).bit_count()
        rem -= 1
    return width, order


def run(problem, here, bands, budget_s, kick, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem); ev = IncEvalC(ab, n)
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; d = ev.full(perm); run_ = 0
                    for t in range(n - 1, -1, -1):
                        c = int(d[t]); run_ = c if c > run_ else run_
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
    print(f"=== grow-highband -- {problem} | front {front_hv():,.0f}"
          f"{f'  gap {front_hv()-target:+,.0f}' if target else ''} ===", flush=True)

    rng = np.random.default_rng(seed); t0 = time.time(); wins = 0; full = set(range(n))
    for W in bands:
        if W not in by_w: continue
        t_star, perm = by_w[W]
        S = set(perm[t_star:]); best = len(S)
        # sanity: confirm our own best torso orders at width<=W under min-degree
        w0, _ = mindeg_width(list(S), ab, n, W + 2)
        tested = 0; tw0 = time.time()
        while time.time() - t0 < budget_s:
            X = [u for u in range(n) if u not in S]
            Sm = 0
            for s in S: Sm |= 1 << s
            # try to move a RANDOM low-boundary subset of X into the torso
            X.sort(key=lambda u: (ab[u] & Sm).bit_count())
            pool = X[:max(kick * 4, 20)]; rng.shuffle(pool)
            add = pool[:int(rng.integers(1, kick + 1))]
            S2 = set(S) | set(int(v) for v in add)
            wdt, order = mindeg_width(list(S2), ab, n, W)
            tested += 1
            if order is not None and len(S2) > best:
                best = len(S2); S = S2
                permnew = [u for u in full if u not in S] + list(order)
                by_w[W] = (n - len(S), permnew); wins += 1
                c = front_hv()
                print(f"  w={W}: WIN  torso {best}  front {c:,.0f}"
                      f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)
            if time.time() - tw0 > budget_s / max(1, len(bands)):
                break
        print(f"  w={W}: done  best {best} (was {n-t_star}, mindeg w0={w0})  "
              f"[{tested} tests, {time.time()-tw0:.0f}s]", flush=True)

    fin = front_hv()
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | {wins} bands improved")
    if wins:
        a = ParetoArchive()
        for w in by_w:
            if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
        top = a.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "grow_highband.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--bands", default="11,12,13,14")
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--kick", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        bands, a.budget, a.kick, a.seed)


if __name__ == "__main__":
    main()

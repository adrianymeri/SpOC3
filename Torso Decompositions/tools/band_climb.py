#!/usr/bin/env python3
r"""
band_climb.py -- targeted ordering-space search for the HIGH bands.

The high bands cannot be attacked in set space: exact treewidth times out past
~700 vertices and min-fill/min-degree are too loose to recognise a width-w torso
(min-degree calls our true width-13 torso "16").  The ONLY oracle that is both
tight and scalable is the C-kernel, which returns the exact width of a *given*
ordering in microseconds at any size -- so the high bands must be searched in
ORDERING space.

Key identity: for an elimination order with per-step degrees deg[i],
    torso_size(w) = n - 1 - max{ i : deg[i] > w }.
So maximising the width-w torso == driving the LAST over-w position ("obstacle")
as far left as possible.  We warm-start from the front's best ordering and run
simulated annealing with relocation / segment / swap moves, scored by the
C-kernel, energy = (obstacle, soft = Sigma deg^2 over the suffix) so the soft
term smooths the plateaus the integer obstacle alone cannot.  Every obstacle
decrease is a real +1 (or more) torso-size -> +HV, banked immediately.

    python3 tools/band_climb.py --problem small-graph --bands 11,12,13,14 \
        --budget 14400 --t0 2.0 --restarts 8

CPU only.  Complements exact_torso (low bands, set space, proven rigid there).
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
import tools.torso_deletion as td
from tools.fastwalk import IncEvalC


def load_front(here, problem, n, ev):
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
    return by_w


def energy(deg, W, n):
    """obstacle = last position with deg>W ; soft = Sigma deg^2 over positions
    strictly after the obstacle (smooth secondary).  Lower is better."""
    obst = -1
    for i in range(n - 1, -1, -1):
        if deg[i] > W:
            obst = i; break
    soft = 0
    for i in range(obst + 1, n):
        d = int(deg[i]); soft += d * d
    return obst, soft


def run(problem, here, bands, budget_s, T0, restarts, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem); ev = IncEvalC(ab, n)
    by_w = load_front(here, problem, n, ev)

    def front_hv():
        a = ParetoArchive()
        for w in by_w: a.try_add(w, by_w[w][0], None)
        return -hypervolume_2d(a.points(), n)
    print(f"=== band-climb -- {problem} | front {front_hv():,.0f}"
          f"{f'  gap {front_hv()-target:+,.0f}' if target else ''} ===", flush=True)

    rng = np.random.default_rng(seed); t0 = time.time(); wins = 0
    for W in bands:
        if W not in by_w: continue
        base_perm = by_w[W][1]; base_obst = by_w[W][0] - 1   # n - t = size; obstacle = t-1
        tw0 = time.time(); moves = 0; accepts = 0
        best_obst = base_obst
        for rs in range(restarts):
            if time.time() - t0 >= budget_s: break
            perm = list(base_perm)
            deg = ev.full(perm)
            obst, soft = energy(deg, W, n)
            T = T0
            seg = max(2, n // 200)
            while time.time() - t0 < budget_s and time.time() - tw0 < budget_s / max(1, len(bands)):
                moves += 1
                # propose a move biased to the core just left of / around the obstacle
                mv = rng.random()
                cand = list(perm)
                lo = max(0, obst - 3 * seg)
                if mv < 0.5:                                   # relocate one vertex
                    i = int(rng.integers(lo, n)); j = int(rng.integers(lo, n))
                    v = cand.pop(i); cand.insert(j, v)
                elif mv < 0.8:                                 # swap two
                    i = int(rng.integers(lo, n)); j = int(rng.integers(lo, n))
                    cand[i], cand[j] = cand[j], cand[i]
                else:                                          # reverse a short segment
                    i = int(rng.integers(lo, max(lo + 1, n - seg)))
                    cand[i:i + seg] = cand[i:i + seg][::-1]
                dd = ev.full(cand)
                o2, s2 = energy(dd, W, n)
                # energy: lexicographic (obstacle, soft) flattened with annealed accept
                dE = (o2 - obst) + 1e-6 * (s2 - soft)
                if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
                    perm = cand; obst = o2; soft = s2; accepts += 1
                    if o2 < best_obst:
                        best_obst = o2
                        by_w[W] = (o2 + 1, list(perm)); wins += 1
                        c = front_hv()
                        print(f"  w={W}: WIN  torso {n-(o2+1)} (obstacle {base_obst}->{o2})  "
                              f"front {c:,.0f}{f'  gap {c-target:+,.0f}' if target else ''}",
                              flush=True)
                        base_perm = list(perm)
                T *= 0.9995
        print(f"  w={W}: done  obstacle {base_obst}->{best_obst}  torso {n-(best_obst+1)} "
              f"(was {n-(base_obst+1)})  [{moves} moves, {accepts} accepts, "
              f"{time.time()-tw0:.0f}s]", flush=True)

    fin = front_hv()
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | {wins} improvements")
    if wins:
        a = ParetoArchive()
        for w in by_w: a.try_add(w, by_w[w][0], by_w[w][1])
        top = a.top_k_by_hv_contribution(20, n)
        dvs = [list(p) + [int(t)] for (_, t, p) in top]
        out = os.path.join(here, "submissions", problem, "band_climb.json")
        json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                   "decisionVector": dvs}, open(out, "w"))
        print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--bands", default="11,12,13,14")
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--t0", dest="T0", type=float, default=2.0, help="initial SA temperature")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        bands, a.budget, a.T0, a.restarts, a.seed)


if __name__ == "__main__":
    main()

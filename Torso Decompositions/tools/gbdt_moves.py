#!/usr/bin/env python3
r"""
gbdt_moves.py -- GBDT as a learned SEARCH OPERATOR (move-value model).

Every other GBDT in this repo either scores a finished ordering or generates one.
This uses boosting in its most native role: predict the *value of a move*.  A move
is "relocate vertex v from position i to position j" in an elimination ordering;
its value is the change it makes to the exact **capped-20 HSSP** objective (the
competition score).  The loop:

  1. take the best orderings (owners of the visible-width points);
  2. generate many candidate relocation moves; featurise each (vertex features +
     source/target band context);
  3. a GBDT predicts each move's ΔHV; we spend the expensive exact evaluation only
     on the top-predicted moves -- so we search ~10-50x more of the move space per
     unit compute than blind local search;
  4. accept any move that improves the capped-20; log (move-features -> real ΔHV)
     and periodically retrain the GBDT on that growing dataset (it gets better at
     spotting improving moves the longer it runs).

This is the reviewer's "GBDT predicts the move, not the score" -- the most GBDT-
native attack, aimed squarely at the visible widths.  ADDITIVE: writes
submissions/<problem>/gbdt_moves.json only on a verified capped-20 improvement.

    python3 tools/gbdt_moves.py --problem medium-graph --rounds 100000 \
        --cand 400 --eval 30 --seed 0
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC
from algorithms.continuous.cmaes_torso import get_features
from algorithms.continuous.np_gbdt import NpGBDT


def order_front(perm, ev, n):
    d = ev.full(perm); r = 0; a = ParetoArchive()
    for t in range(n - 1, -1, -1):
        c = int(d[t]); r = c if c > r else r
        if r <= MAX_TW: a.try_add(r, t, perm)
    # also return the running-max width profile (band context per position)
    prof = np.empty(n, dtype=np.int32); r = 0
    for t in range(n - 1, -1, -1):
        c = int(d[t]); r = c if c > r else r; prof[t] = r
    return [(w, t) for (w, t, _) in a.entries()], prof


def capped(points_wt, n, k=20):
    a = ParetoArchive()
    for (w, t) in points_wt: a.try_add(w, t, None)
    top = a.top_k_by_hv_contribution(k, n)
    return -hypervolume_2d([(w, t) for (w, t, _) in top], n)


def load_seed(here, problem, n, ev, cap):
    arc = ParetoArchive(); seen = {}
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    p = tuple(int(x) for x in dv[:-1])
                    if p not in seen:
                        pts, prof = order_front(list(p), ev, n)
                        seen[p] = (pts, prof)
                        for (w, t) in pts: arc.try_add(w, t, list(p))
        except Exception:
            continue
    # owners of the HSSP-optimal 20 points = the orderings worth perturbing
    owners = [tuple(p) for (_, _, p) in arc.top_k_by_hv_contribution(20, n)]
    base_pts = [(w, t) for (w, t, _) in arc.entries()]
    srcs = []
    for o in owners:
        if o in seen: srcs.append((list(o), seen[o][1]))
    return base_pts, srcs, seen


def move_features(F, v, i, j, n, prof):
    return np.concatenate([F[v], [i / n, j / n, (j - i) / n,
                                  prof[i] / MAX_TW, prof[min(j, n - 1)] / MAX_TW]])


def run(problem, here, rounds, cand, neval, seed):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    F = np.asarray(get_features(here, problem, n, adj, 32)[0])
    base_pts, srcs, seen = load_seed(here, problem, n, ev, 40)
    best = capped(base_pts, n)
    print(f"=== gbdt-moves {problem} | {len(srcs)} source orders | "
          f"capped-20 {best:,.0f}{f'  gap {best-target:+,.0f}' if target else ''} ===",
          flush=True)
    rng = random.Random(seed)
    DX, DY = [], []; model = None; accepts = 0; t0 = time.time()

    for rd in range(rounds):
        sp, prof = srcs[rng.randrange(len(srcs))]
        # generate candidate relocation moves
        cands = []
        for _ in range(cand):
            i = rng.randrange(n); j = rng.randrange(n)
            if i == j: continue
            cands.append((sp[i], i, j))
        feats = np.array([move_features(F, v, i, j, n, prof) for (v, i, j) in cands])
        if model is not None and len(cands):
            order = np.argsort(-model.predict(feats))     # highest predicted ΔHV first
        else:
            order = np.arange(len(cands)); rng.shuffle(order)
        tried = 0
        for oi in order[:neval]:
            v, i, j = cands[int(oi)]
            nperm = sp[:]; x = nperm.pop(i); nperm.insert(j if j < i else j - 1, x)
            pts, nprof = order_front(nperm, ev, n)
            c = capped(base_pts + pts, n)
            dhv = best - c                                 # >0 == improved capped-20
            DX.append(feats[int(oi)]); DY.append(dhv); tried += 1
            if c < best - 0.5:
                best = c; accepts += 1
                base_pts += pts
                srcs.append((nperm, nprof))
                a = ParetoArchive()
                for (w, t) in base_pts: a.try_add(w, t, None)
                top = a.top_k_by_hv_contribution(20, n)
                # persist: need perms for the top points -> rebuild owner archive
                oa = ParetoArchive()
                for (perm_, _pf) in srcs:
                    pp, _ = order_front(perm_, ev, n)
                    for (w, t) in pp: oa.try_add(w, t, perm_)
                dvs = [list(p) + [int(t)] for (w, t, p) in oa.top_k_by_hv_contribution(20, n)]
                out = os.path.join(here, "submissions", problem, "gbdt_moves.json")
                json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                           "decisionVector": dvs}, open(out, "w"))
                print(f"  rd {rd}: *** capped-20 {best:,.0f}"
                      f"{f'  gap {best-target:+,.0f}' if target else ''} "
                      f"(accept #{accepts}) ***", flush=True)
        # retrain the move-value GBDT periodically
        if len(DY) >= 400 and rd % 25 == 0:
            X = np.array(DX); y = np.array(DY)
            model = NpGBDT(n_estimators=120, learning_rate=0.1, max_depth=4,
                           subsample=0.8, seed=seed)
            model.fit(X, y)
        if rd % 10 == 0:
            pos = int((np.array(DY) > 0).sum()) if DY else 0
            gh = "guided" if model is not None else "random"
            print(f"  [rd {rd} {gh}  capped {best:,.0f}  {accepts} accepts  "
                  f"{pos}/{len(DY)} improving moves seen  {time.time()-t0:.0f}s]", flush=True)

    print(f"\nfinal capped-20 {best:,.0f}"
          f"{f'  gap {best-target:+,.0f}' if target else ''} | {accepts} improvements", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--rounds", type=int, default=100000)
    ap.add_argument("--cand", type=int, default=400)
    ap.add_argument("--eval", dest="neval", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.rounds, a.cand, a.neval, a.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
minfill_refine.py -- per-torso re-elimination to tighten width(t) toward the bound.

The front-gap diagnostic (tools/front_gap_analysis.py) showed the dense core is
already at the treewidth lower bound, but the mid/small-torso front sits a
consistent ~20 width-units above it -- a fill-in-quality gap that static-policy
CMA-ES cannot close. This is the classical fix: for a grid of thresholds t, take
the head order [0,t) from a strong base ordering, build the residual fill-in
graph on the torso, and RE-ELIMINATE that torso with a greedy fill-in-aware rule
(min-degree, fast; or min-fill, tighter). Each re-elimination is specialised to
its torso size and can beat the global ordering's width(t) there. All results are
pooled additively (stem `minfill`) and harvested as full staircases.

CPU-only; no GPU. This both (a) attempts to improve the verified result and
(b) tests the diagnosis: if width drops toward the LB, the gap was fill-in
quality; if it barely moves, the gap was lower-bound slack (we are near-optimal).

    python3 tools/minfill_refine.py --problem large-graph --rule mindeg --grid 24
    python3 tools/minfill_refine.py --problem large-graph --rule minfill --grid 12
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, hypervolume_2d, MAX_TW, submission_path, write_submission)


def base_ordering(here, prob, n):
    """Best pooled ordering by full-torso width, to supply head orders."""
    best = None; bestw = 10**9
    for stem in ("gaps", "portfolio", "gpucma"):
        for fp in glob.glob(os.path.join(here, "submissions", prob, f"{stem}.json")):
            try: dvs = json.load(open(fp))[0]["decisionVector"]
            except Exception: continue
            for dv in dvs:
                if isinstance(dv, list) and len(dv) == n+1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    p = [int(x) for x in dv[:-1]]
                    w = width_at(p, build_adj_bitsets_cached(here, prob, n), n, 0)
                    if w < bestw: bestw = w; best = p
    return best


_ABC = {}
def build_adj_bitsets_cached(here, prob, n):
    if prob not in _ABC:
        _, adj = load_graph(graph_path(here, prob)); _ABC[prob] = build_adj_bitsets(n, adj)
    return _ABC[prob]


def deg_seq(perm, ab, n):
    sm = [0]*n; cur = 0
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    return deg


def width_at(perm, ab, n, t):
    return int(np.maximum.accumulate(deg_seq(perm, ab, n)[::-1])[::-1][t])


def residual_on_torso(perm, ab, n, t):
    """Eliminate head [0,t) (with fill); return residual adjacency (python-int
    bitsets re-indexed over the torso vertices V = perm[t:]) and V."""
    tmp = list(ab)
    sm = [0]*n; cur = 0
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    for i in range(t):
        s = tmp[perm[i]] & sm[i]; x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    V = perm[t:]; idx = {v: j for j, v in enumerate(V)}; m = len(V)
    Vbit = 0
    for v in V: Vbit |= 1 << v
    sub = [0]*m
    for v in V:
        nb = tmp[v] & Vbit & ~(1 << v); x = nb
        while x:
            b = x & -x; x ^= b; u = b.bit_length()-1; sub[idx[v]] |= 1 << idx[u]
    return sub, V


def greedy_eliminate(sub, m, rule):
    """Greedy elimination of a residual graph (bitset list). Returns
    (order, width) where width = max degree over the elimination."""
    g = list(sub); alive = (1 << m) - 1; order = []; width = 0
    for _ in range(m):
        # candidates = alive vertices; pick by rule
        best_v = -1; best_key = None
        x = alive
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1
            nb = g[v] & alive
            d = nb.bit_count()
            if rule == "mindeg":
                key = d
            else:  # minfill: new edges needed among neighbours
                fill = 0; y = nb
                nbl = []
                while y:
                    bb = y & -y; y ^= bb; nbl.append(bb.bit_length()-1)
                for a in range(len(nbl)):
                    ga = g[nbl[a]]
                    for c in range(a+1, len(nbl)):
                        if not (ga >> nbl[c]) & 1: fill += 1
                key = (fill, d)
            if best_key is None or key < best_key:
                best_key = key; best_v = v; best_nb = nb
        # eliminate best_v: connect its neighbours, record degree
        d = best_nb.bit_count(); width = max(width, d)
        order.append(best_v); alive &= ~(1 << best_v)
        y = best_nb
        while y:
            bb = y & -y; y ^= bb; u = bb.bit_length()-1
            g[u] |= best_nb & ~(1 << u)
    return order, width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--rule", default="mindeg", choices=["mindeg", "minfill"])
    ap.add_argument("--grid", type=int, default=24, help="# thresholds to re-eliminate at")
    ap.add_argument("--algo", default="minfill")
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem)); ab = build_adj_bitsets(n, adj)
    _ABC[args.problem] = ab
    base = base_ordering(here, args.problem, n)
    if base is None:
        print("no base ordering found"); return
    t_grid = sorted(set(int(round(i*(n-1)/(args.grid))) for i in range(1, args.grid)))
    print(f"=== minfill_refine {args.problem} (rule={args.rule}, {len(t_grid)} thresholds) ===")
    arch = ParetoArchive()
    # seed with the base front
    d = deg_seq(base, ab, n); sm = np.maximum.accumulate(d[::-1])[::-1]
    for t in range(n): arch.try_add(int(sm[t]), t, base)
    base_hv = -arch.hypervolume(n)
    print(f"base front HV: {base_hv:,.0f}")
    t0 = time.time(); improved = 0
    for t in t_grid:
        sub, V = residual_on_torso(base, ab, n, t)
        order, w = greedy_eliminate(sub, len(V), args.rule)
        torso_perm = [V[j] for j in order]
        full = base[:t] + torso_perm
        dd = deg_seq(full, ab, n); ss = np.maximum.accumulate(dd[::-1])[::-1]
        if ss.max() <= MAX_TW or True:
            added = False
            for tt in range(n):
                if arch.try_add(int(ss[tt]), tt, full): added = True
            if int(ss[t]) < int(sm[t]): improved += 1
            print(f"  t={t:5d} torso={len(V):5d}  re-elim width={w:4d}  "
                  f"(base width(t)={int(sm[t])})  {'↓' if int(ss[t])<int(sm[t]) else ''}",
                  flush=True)
    final_hv = -arch.hypervolume(n)
    print(f"\nthresholds improved: {improved}/{len(t_grid)}  in {time.time()-t0:.0f}s")
    print(f"front HV: {base_hv:,.0f} -> {final_hv:,.0f}  ({final_hv-base_hv:+,.0f})")
    top = arch.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = submission_path(here, args.problem, args.algo)
    write_submission(dvs, args.problem, out)
    print(f"Wrote {out} ({len(dvs)} vectors)")


if __name__ == "__main__":
    main()

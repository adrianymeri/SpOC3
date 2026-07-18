#!/usr/bin/env python3
"""
width_demote.py -- constructive zone absorption for large-graph (THESIS §15 follow-up).

Measurement (2026-07-12): the achiever at each selected width b carries only
75-370 torso steps wider than the previous selected width a; in four zones
(82<-99, 65<-82, 154<-170, 99<-112, 18<-46) evicting every violator would
STILL leave a net head saving vs the current achiever at a (up to +346 at
82<-99). So: demote big-torso orderings to lower widths by greedily evicting
the widest offending step into the head, with exact re-evaluation each move,
banking every intermediate staircase point.

    python3 tools/width_demote.py                 # all zones, both rules
    python3 tools/width_demote.py --zones 82,99   # just the big one
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"
SEL = [18, 46, 65, 82, 99, 112, 133, 154, 170, 195, 218, 233, 252, 273, 299]


def pool(n, ev, prob, wcap):
    """per-width best (t, perm) for w<=wcap, staircase-correct."""
    best = {}
    files = sorted(f for f in glob.glob(os.path.join(HERE, "submissions", prob, "*.json"))
                   if not f.endswith("_platform.json"))
    for fp in files:
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e.get("decisionVector", []):
                if not (isinstance(dv, list) and len(dv) == n + 1):
                    continue
                p = [int(x) for x in dv[:-1]]
                if sorted(p) != list(range(n)):
                    continue
                df = ev.full(p); r = 0
                if int(max(df)) > MAX_TW:   # dirty head => void at ESA
                    continue
                for t in range(n - 1, -1, -1):
                    c = int(df[t]); r = c if c > r else r
                    if r <= wcap and (r not in best or t < best[r][0]):
                        best[r] = (t, p)
        except Exception:
            pass
    # cumulative best over widths
    bw, cur = {}, (10 ** 9, None)
    for w in range(0, wcap + 1):
        if w in best and best[w][0] < cur[0]:
            cur = best[w]
        bw[w] = cur
    return bw


def demote(a, b, t0, perm0, n, ev, arc, rule, budget_t, max_evals=100000):
    """evict offenders from the (b)-achiever until torso width <= a or hopeless."""
    perm = list(perm0); t = t0
    evals = 0
    while evals < max_evals:
        df = ev.full(perm); evals += 1
        if max(int(x) for x in df) > MAX_TW:
            return None, evals
        r = 0
        for tt in range(n - 1, -1, -1):
            c = int(df[tt]); r = c if c > r else r
            if r <= MAX_TW:
                arc.try_add(r, tt, perm)
        viol = [p for p in range(t, n) if int(df[p]) > a]
        if not viol:
            return (a, t), evals
        if t + len(viol) >= budget_t + 40:      # hopeless-by-margin cutoff
            return None, evals
        if rule == "widest":
            p = max(viol, key=lambda p: int(df[p]))
        else:                                    # "first": earliest offender
            p = viol[0]
        v = perm.pop(p)
        perm.insert(t, v)
        t += 1
    return None, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=PROBLEM, choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--wcap", type=int, default=0,
                    help="max width to pool (default: large 299, medium/small n-1)")
    ap.add_argument("--zones", default="", help="comma list of zone LOWER widths to attack")
    ap.add_argument("--rules", default="widest,first")
    ap.add_argument("--algo", default="width_demote")
    a = ap.parse_args()
    prob = a.problem
    n, adj_l = load_graph(graph_path(HERE, prob))
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    target = LEADERBOARD_TARGETS[prob]
    wcap = a.wcap or (299 if prob == "large-graph" else n - 1)

    print(f"pooling envelope (w<={wcap}) ...", flush=True)
    bw = pool(n, ev, prob, wcap)
    arc = ParetoArchive()
    for w in range(0, wcap + 1):
        t, p = bw[w]
        if p is not None:
            arc.try_add(w, t, p)

    # live zones: the HSSP-optimal 20 widths of the current pool
    sel = sorted(w for w, _, _ in arc.top_k_by_hv_contribution(20, n))
    print(f"live selected widths: {sel}", flush=True)
    zones = [int(x) for x in a.zones.split(",") if x] or None
    t0 = time.time()
    for i, lo in enumerate(sel):
        if zones is not None and lo not in zones:
            continue
        hi = sel[i + 1] if i + 1 < len(sel) else wcap + 1
        ta = bw[lo][0]
        # demote FROM the zone's envelope floor (best achiever inside the zone)
        fw = min(range(lo, min(hi, wcap + 1)), key=lambda x: bw[x][0])
        tb, pb = bw[fw]
        if pb is None or tb >= ta:
            continue
        best_res = None
        for rule in a.rules.split(","):
            res, evals = demote(lo, fw, tb, list(pb), n, ev, arc, rule, ta)
            tag = f"-> t={res[1]}" if res else "failed"
            print(f"  zone {lo:3d}<-{fw:3d} rule={rule:6s} start t={tb} target beat t<{ta} "
                  f"{tag} ({evals} evals) [{time.time()-t0:.0f}s]", flush=True)
            if res and (best_res is None or res[1] < best_res[1]):
                best_res = res
        if best_res and best_res[1] < ta:
            print(f"  *** ZONE {lo} IMPROVED: t {ta} -> {best_res[1]} "
                  f"(zone width {hi-lo}, ~{(ta-best_res[1])*(hi-lo):,d} HV) ***", flush=True)

    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(w, t) for w, t, _ in top], n)
    print(f"\ndemote archive cap20: {cap:,.0f} (target {target:,.0f}, gap {cap-target:+,.0f})")
    out = os.path.join(HERE, "submissions", prob, f"{a.algo}.json")
    top60 = arc.top_k_by_hv_contribution(60, n)
    write_submission([list(p) + [int(t)] for (_, t, p) in top60], prob, out)
    print(f"wrote {out} -- run cap_submit to pool")


if __name__ == "__main__":
    main()

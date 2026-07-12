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


def pool(n, ev):
    """per-width best (t, perm) for w<=299, staircase-correct."""
    best = {}
    files = sorted(f for f in glob.glob(os.path.join(HERE, "submissions", PROBLEM, "*.json"))
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
                    if r <= 299 and (r not in best or t < best[r][0]):
                        best[r] = (t, p)
        except Exception:
            pass
    # cumulative best over widths
    bw, cur = {}, (10 ** 9, None)
    for w in range(0, 300):
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
    ap.add_argument("--zones", default="", help="comma list of zone UPPER widths, e.g. 99,82")
    ap.add_argument("--rules", default="widest,first")
    ap.add_argument("--algo", default="width_demote")
    a = ap.parse_args()
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    target = LEADERBOARD_TARGETS[PROBLEM]

    print("pooling envelope (w<=299) ...", flush=True)
    bw = pool(n, ev)
    arc = ParetoArchive()
    for w in range(0, 300):
        t, p = bw[w]
        if p is not None:
            arc.try_add(w, t, p)

    zones = [int(x) for x in a.zones.split(",") if x] or None
    pairs = [(SEL[i], SEL[i + 1]) for i in range(len(SEL) - 1)
             if zones is None or SEL[i + 1] in zones]
    t0 = time.time()
    for lo, hi in pairs:
        ta = bw[lo][0]; tb, pb = bw[hi]
        if pb is None:
            continue
        best_res = None
        for rule in a.rules.split(","):
            res, evals = demote(lo, hi, tb, pb, n, ev, arc, rule, ta)
            tag = f"-> t={res[1]}" if res else "failed"
            print(f"  zone {lo:3d}<-{hi:3d} rule={rule:6s} start t={tb} target beat t<{ta} "
                  f"{tag} ({evals} evals) [{time.time()-t0:.0f}s]", flush=True)
            if res and (best_res is None or res[1] < best_res[1]):
                best_res = res
        if best_res and best_res[1] < ta:
            print(f"  *** ZONE {lo} IMPROVED: t {ta} -> {best_res[1]} "
                  f"(zone width {hi-lo}, ~{(ta-best_res[1])*(hi-lo):,d} HV) ***", flush=True)

    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(w, t) for w, t, _ in top], n)
    print(f"\ndemote archive cap20: {cap:,.0f} (target {target:,.0f}, gap {cap-target:+,.0f})")
    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")
    top60 = arc.top_k_by_hv_contribution(60, n)
    write_submission([list(p) + [int(t)] for (_, t, p) in top60], PROBLEM, out)
    print(f"wrote {out} -- run cap_submit to pool")


if __name__ == "__main__":
    main()

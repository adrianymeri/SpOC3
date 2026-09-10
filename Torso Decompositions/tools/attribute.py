#!/usr/bin/env python3
"""attribute.py -- WHICH ARM produced the points we actually submit?

The campaign runs many arms into one shared pool, and cap_submit reports only
the pooled score.  That makes the thesis's central claim -- that GBDT drives
the result -- unmeasured: nobody has ever checked which arm supplied the 20
orderings that are actually scored.

This tool answers it.  It rebuilds the envelope while remembering the SOURCE
FILE of every point, runs the same exact HSSP top-20 selection cap_submit
uses, and then reports, for each of the 20 submitted points, which arm it came
from -- plus the share of submitted hypervolume attributable to GBDT-driven
arms versus classical ones.

Attribution rule: a point is credited to the file that first achieved its
(width, threshold) pair.  Where several arms independently reach an identical
point, credit goes to the earliest file by modification time, so a late-
starting arm cannot claim a point it merely reproduced.  Peer copies synced
between machines (mac_*, srv_*) are stripped to their original stem so a
sync does not relabel an arm's own work.

    python3 tools/attribute.py --problem small-graph
    python3 tools/attribute.py --problem medium-graph --by-file
"""
from __future__ import annotations
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC

# arms whose search is driven by a learned gradient-boosted model
GBDT_MARKERS = ("gbdt", "gbfc", "gaps", "rank", "landscape", "mapelite",
                "moves", "sweep", "ace", "dts")
# Pool AGGREGATES: not arms.  cap20.json is written by cap_submit,
# full_envelope.json by consolidate_pool.py, portfolio.json by portfolio.py /
# refine_thresholds.py -- each is a deduplicated re-packaging of points other
# arms already found.  Crediting them would invent a phantom contributor, so
# they are processed LAST (any arm that also holds the point claims it first)
# and whatever only they contain is reported as unattributable rather than
# silently scored as "classical".
DERIVED = ("cap20", "cap20_platform", "consolidated", "full_envelope",
           "portfolio", "front", "checkpoint", "diversify")


def stem_of(path):
    s = os.path.basename(path)[:-5]           # drop .json
    for pref in ("mac_", "srv_"):             # peer copies -> original arm
        if s.startswith(pref):
            s = s[len(pref):]
    return s


def is_derived(stem):
    return any(stem.startswith(d) or stem == d for d in DERIVED)


def is_gbdt(stem):
    s = stem.lower()
    return any(m in s for m in GBDT_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--by-file", action="store_true", help="per-file breakdown too")
    a = ap.parse_args()

    n, adj = load_graph(graph_path(HERE, a.problem))
    ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(a.problem)

    allf = sorted(glob.glob(os.path.join(HERE, "submissions", a.problem, "*.json")),
                  key=lambda f: os.path.getmtime(f))      # earliest first
    # real arms first, aggregates last -> an aggregate can only claim a point
    # that no actual arm contains.
    files = ([f for f in allf if not is_derived(stem_of(f))] +
             [f for f in allf if is_derived(stem_of(f))])
    best = {}          # width -> (t, perm, stem)
    used = 0
    seen = set()       # a perm is credited to the EARLIEST file that contains it
    for fp in files:
        stem = stem_of(fp)
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            dvs = e["decisionVector"]
        except Exception:
            continue
        used += 1
        for dv in dvs:
            if not (isinstance(dv, list) and len(dv) == n + 1):
                continue
            tp = tuple(int(x) for x in dv[:-1])
            if tp in seen:
                continue
            seen.add(tp)
            perm = list(tp)
            if sorted(perm) != list(range(n)):
                continue
            df = np.asarray(ev.full(perm), dtype=np.int64)
            r = np.maximum.accumulate(df[::-1])       # r[i] = width at t = n-1-i
            if int(r[-1]) > MAX_TW:
                continue
            idx = np.flatnonzero(np.append(np.diff(r) != 0, True))  # staircase only
            for w_, t_ in zip(r[idx].tolist(), ((n - 1) - idx).tolist()):
                if w_ > MAX_TW:
                    continue
                if w_ not in best or t_ < best[w_][0]:
                    best[w_] = (t_, perm, stem)

    arc = ParetoArchive()
    for w, (t, p, _) in best.items():
        arc.try_add(w, t, p)
    top = arc.top_k_by_hv_contribution(a.k, n)
    cap = hypervolume_2d([(w, t) for (w, t, _) in top], n)
    chosen = sorted((w, t) for (w, t, _) in top)

    print(f"=== attribution {a.problem} | {used} arm files | "
          f"best-{a.k} = {-cap:,.0f}"
          f"{f'  gap {-cap-target:+,.0f}' if target else ''} ===")
    print(f"\n  {'w':>5} | {'torso':>6} | {'strip HV':>10} | source arm")
    agg = {"GBDT": [0, 0.0], "classical": [0, 0.0], "aggregate": [0, 0.0]}
    per_file = {}
    endpoint_owner = None
    for i, (w, t) in enumerate(chosen):
        nxt = chosen[i + 1][0] if i + 1 < len(chosen) else n
        strip = (nxt - w) * (n - t)
        stem = best[w][2]
        tag = ("aggregate" if is_derived(stem)
               else "GBDT" if is_gbdt(stem) else "classical")
        agg[tag][0] += 1; agg[tag][1] += strip
        d = per_file.setdefault(stem, [0, 0.0, tag])
        d[0] += 1; d[1] += strip
        mark = "  <== ENDPOINT" if i == len(chosen) - 1 else ""
        if i == len(chosen) - 1:
            endpoint_owner = (stem, tag)
        print(f"  {w:5d} | {n-t:6d} | {strip:10,.0f} | {stem}  [{tag}]{mark}")

    tot = sum(v[1] for v in agg.values()) or 1.0
    print()
    for tag in ("GBDT", "classical", "aggregate"):
        cnt, hv = agg[tag]
        print(f"  {tag:<10} : {cnt:2d}/{len(chosen)} points | "
              f"{hv:,.0f} HV ({100*hv/tot:.1f}%)")
    print(f"\n  ENDPOINT (t=0) owner: {endpoint_owner[0]}  [{endpoint_owner[1]}]")
    print("  CAVEAT: the endpoint's strip runs to the reference point, so it")
    print("  carries most of the hypervolume BY CONSTRUCTION. Quote the point")
    print("  counts and the endpoint owner; the HV %% alone overstates whoever")
    print("  happens to own that one point.")

    if a.by_file:
        print(f"\n  per-arm breakdown:")
        for stem, (cnt, hv, tag) in sorted(per_file.items(), key=lambda kv: -kv[1][1]):
            print(f"    {stem:<34} {cnt:2d} pts  {hv:12,.0f} HV  [{tag}]")


if __name__ == "__main__":
    main()

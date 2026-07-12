#!/usr/bin/env python3
"""
sanitize_pool.py -- strip ESA-invalid orderings from a submissions pool.

2026-07-12 incident: verify_submission caught 6/20 vectors in the large
cap20.json scoring 501 -- their HEADS contain steps wider than MAX_TW, which
voids the whole decision vector under the official contract, while suffix-only
banking (cap_submit/arms, now all fixed) still credited their torso points.
Likely source: raw cuda-torso batch dumps copied into the pool.

This tool re-evaluates EVERY decision vector in submissions/<problem>/*.json
and rewrites each file keeping only fully-valid orderings (permutation valid,
every elimination step <= MAX_TW). Run it after every GPU-dump ingest and
once now on both machines:

    python3 tools/sanitize_pool.py --problem large-graph
    python3 tools/sanitize_pool.py --all
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, MAX_TW,
                  LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC


def sanitize(problem):
    n, adj_l = load_graph(graph_path(HERE, problem))
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    files = sorted(f for f in glob.glob(os.path.join(HERE, "submissions", problem, "*.json"))
                   if not f.endswith("_platform.json"))
    tot_kept = tot_drop = 0
    t0 = time.time()
    cache = {}
    for fp in files:
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            dvs = e.get("decisionVector", [])
        except Exception:
            print(f"  {os.path.basename(fp)}: unreadable, skipped")
            continue
        keep = []
        drop = 0
        for dv in dvs:
            ok = False
            if isinstance(dv, list) and len(dv) == n + 1:
                perm = tuple(int(x) for x in dv[:-1])
                t = int(dv[-1])
                if 0 <= t <= n and sorted(perm) == list(range(n)):
                    if perm in cache:
                        ok = cache[perm]
                    else:
                        df = ev.full(list(perm))
                        ok = int(max(df)) <= MAX_TW
                        cache[perm] = ok
            if ok:
                keep.append([int(x) for x in dv])
            else:
                drop += 1
        tot_kept += len(keep); tot_drop += drop
        if drop:
            write_submission(keep, problem, fp)
            print(f"  {os.path.basename(fp)}: kept {len(keep)}, DROPPED {drop}", flush=True)
    print(f"{problem}: kept {tot_kept}, dropped {tot_drop} invalid vectors "
          f"across {len(files)} files [{time.time()-t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None, choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    probs = list(LEADERBOARD_TARGETS) if a.all or not a.problem else [a.problem]
    for p in probs:
        sanitize(p)


if __name__ == "__main__":
    main()

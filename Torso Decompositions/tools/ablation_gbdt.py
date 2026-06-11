#!/usr/bin/env python3
"""
ablation_gbdt.py -- the definitive, viva-proof measure of GBDT's contribution.

For each instance it builds the full-staircase top-20 portfolio TWICE -- once
using every ordering, once EXCLUDING the gbdt* orderings -- holding everything
else identical. The difference is GBDT's *independent* contribution to the
submitted score. This is the controlled experiment an examiner will ask for.

Usage:
    python3 tools/ablation_gbdt.py                       # all three
    python3 tools/ablation_gbdt.py --problems large-graph
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, glob, json, os
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, hypervolume_2d, MAX_TW)

ALL = ["small-graph", "medium-graph", "large-graph"]


def pool(here, problem, exclude_gbdt):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    files = (glob.glob(os.path.join(here, "submissions", problem, "*.json"))
             + glob.glob(os.path.join(here, "submissions", problem, "seeds", "*.json")))
    perms = set()
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem == "portfolio":
            continue
        if exclude_gbdt and stem.startswith("gbdt"):
            continue
        try:
            dvs = json.load(open(fp))[0]["decisionVector"]
        except Exception:
            continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1 and \
                    sorted(int(x) for x in dv[:-1]) == list(range(n)):
                perms.add(tuple(int(x) for x in dv[:-1]))
    arch = ParetoArchive()
    for p in perms:
        sm = [0] * n; cur = 0
        for i in range(n - 1, -1, -1):
            sm[i] = cur; cur |= 1 << p[i]
        tmp = list(ab); deg = [0] * n; cap = False
        for i in range(n):
            s = tmp[p[i]] & sm[i]; d = s.bit_count(); deg[i] = d
            if d > MAX_TW:
                cap = True; break
            x = s
            while x:
                b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
        if cap:
            continue
        run = 0
        for t in range(n - 1, -1, -1):
            if deg[t] > run:
                run = deg[t]
            arch.try_add(run, t, list(p))
    top = arch.top_k_by_hv_contribution(20, n)
    return -hypervolume_2d([(w, t) for w, t, _ in top], n), len(perms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(ALL))
    args = ap.parse_args()
    here = repo_root()
    print(f"{'instance':<13}{'WITH gbdt':>16}{'WITHOUT gbdt':>16}{'GBDT share':>14}")
    for prob in [p.strip() for p in args.problems.split(",") if p.strip()]:
        full, nf = pool(here, prob, False)
        nog, ng = pool(here, prob, True)
        print(f"{prob:<13}{full:>16,.0f}{nog:>16,.0f}{full-nog:>+14,.0f}")


if __name__ == "__main__":
    main()

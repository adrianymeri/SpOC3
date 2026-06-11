#!/usr/bin/env python3
"""
refine_thresholds.py -- free HV by extracting each ordering's FULL threshold
staircase.

The portfolio pools only the *submitted* (perm, t) points. But for any feasible
ordering, one elimination pass gives the width at EVERY threshold (suffix-max of
the degree sequence) -- so each ordering induces a whole (width, t) staircase,
not one point. Pooling the full staircases of all known orderings and taking the
top-20 can only match or beat the portfolio (more candidate points for the same
HSSP selection), at zero search cost.

Writes the improved front to submissions/<instance>/portfolio.json (the score
never decreases; verified 0-capped). Additive: reads every existing submission,
overwrites only the portfolio stem.

Usage:
    python3 tools/refine_thresholds.py                 # all three
    python3 tools/refine_thresholds.py --problems medium-graph
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, glob, json, os
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, hypervolume_2d, submission_path, write_submission, MAX_TW)

ALL = ["small-graph", "medium-graph", "large-graph"]


def staircase_into(perm, ab, n, arch):
    """Add the full (width, t) staircase of one ordering to the archive."""
    sm = [0] * n; cur = 0
    for i in range(n - 1, -1, -1):
        sm[i] = cur; cur |= 1 << perm[i]
    tmp = list(ab); deg = [0] * n
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; d = s.bit_count(); deg[i] = d
        if d > MAX_TW:
            return False
        x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    run = 0
    for t in range(n - 1, -1, -1):
        if deg[t] > run:
            run = deg[t]
        arch.try_add(run, t, list(perm))
    return True


def run(here, problems, write):
    for prob in problems:
        n, adj = load_graph(graph_path(here, prob)); ab = build_adj_bitsets(n, adj)
        files = (glob.glob(os.path.join(here, "submissions", prob, "*.json"))
                 + glob.glob(os.path.join(here, "submissions", prob, "seeds", "*.json")))
        perms = set()
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            if stem == "portfolio":
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
            staircase_into(list(p), ab, n, arch)
        top = arch.top_k_by_hv_contribution(20, n)
        score = -hypervolume_2d([(w, t) for w, t, _ in top], n)
        print(f"{prob:<13} full-staircase top-20 = {score:>16,.0f}  (from {len(perms)} orderings)")
        if write:
            dvs = [list(p) + [int(t)] for (_, t, p) in top]
            out = submission_path(here, prob, "portfolio")
            write_submission(dvs, prob, out)
            print(f"              wrote {out}  ({len(dvs)} vectors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(ALL))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(repo_root(), [p.strip() for p in args.problems.split(",") if p.strip()],
        write=not args.dry_run)


if __name__ == "__main__":
    main()

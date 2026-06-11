#!/usr/bin/env python3
"""
variance_report.py -- run-to-run variance of the per-seed submissions.

Reports, per instance and per method family (cmaes / cmaesf / gbdt), the
distribution of single-seed submitted scores: mean +/- std, min, max, count.
This supplies the variance an examiner expects alongside the best-of-union
portfolio headline.

Usage:
    python3 tools/variance_report.py
    python3 tools/variance_report.py --problems small-graph,medium-graph
"""
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse, glob, json, os, re, statistics
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  evaluate, hypervolume_2d, MAX_TW)

ALL = ["small-graph", "medium-graph", "large-graph"]
FAMILIES = [("cmaes", r"cmaes_s\d+$"), ("cmaesf", r"cmaesf_s\d+$"), ("gbdt", r"gbdt_s\d+$")]


def score_file(fp, ab, n):
    try:
        dvs = json.load(open(fp))[0]["decisionVector"]
    except Exception:
        return None
    feas = []
    for dv in dvs:
        if not isinstance(dv, list) or len(dv) != n + 1:
            continue
        perm = [int(x) for x in dv[:-1]]
        if sorted(perm) != list(range(n)):
            continue
        w, t = evaluate(perm, int(dv[-1]), ab, n)
        if w <= MAX_TW:
            feas.append((int(w), int(t)))
    return -hypervolume_2d(feas, n) if feas else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(ALL))
    args = ap.parse_args()
    here = repo_root()
    print(f"{'instance':<13}{'family':<8}{'n':>4}{'mean':>15}{'std':>12}{'min':>15}{'max':>15}")
    for prob in [p.strip() for p in args.problems.split(",") if p.strip()]:
        n, adj = load_graph(graph_path(here, prob)); ab = build_adj_bitsets(n, adj)
        seeddir = os.path.join(here, "submissions", prob, "seeds")
        files = glob.glob(os.path.join(seeddir, "*.json"))
        for fam, pat in FAMILIES:
            scores = []
            for fp in files:
                stem = os.path.splitext(os.path.basename(fp))[0]
                if re.fullmatch(pat, stem):
                    s = score_file(fp, ab, n)
                    if s is not None:
                        scores.append(s)
            if not scores:
                continue
            mean = statistics.mean(scores)
            std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            print(f"{prob:<13}{fam:<8}{len(scores):>4}{mean:>15,.0f}{std:>12,.0f}"
                  f"{min(scores):>15,.0f}{max(scores):>15,.0f}")


if __name__ == "__main__":
    main()

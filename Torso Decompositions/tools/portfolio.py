#!/usr/bin/env python3
"""
portfolio.py -- instance-specific best-of portfolio submission.

The study established that the best *method* differs by instance (population
MOEAs on the sparse small-graph, GRASP on the dense large-graph, the
perturbative families on medium).  This tool turns that observation into a
submission with zero new search:

  1. For each instance it re-evaluates EVERY existing submission JSON
     (submissions/<problem>/*.json) end-to-end through the canonical
     `core.evaluate` path -- the same scorer `verify_submission.py` uses.
  2. It reports the best single method per instance.
  3. It POOLS the Pareto points of every method into one archive and writes
     the top-20-by-HV-contribution subset as `submissions/<problem>/portfolio.json`.

The union can only match or beat the best single method: every method's
non-dominated points compete for the 20 submission slots, and the
HV-subset-selection DP keeps the 20 that jointly cover the most area.  This is
a strict free win -- no new runs, no tuning.

ADDITIVE / SAFE: writes only to the `portfolio` stem.  Canonical HC and every
other submission JSON are read-only inputs here; nothing is overwritten.

Usage
-----
    python3 tools/portfolio.py                      # all 3 instances, write portfolio.json
    python3 tools/portfolio.py --problems small-graph
    python3 tools/portfolio.py --dry-run            # report only, write nothing
    python3 tools/portfolio.py --exclude portfolio  # skip stems (default: portfolio)
"""

from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import glob
import json
import os
from typing import List, Tuple

from core import (
    LEADERBOARD_TARGETS,
    MAX_TW,
    ParetoArchive,
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
    submission_path,
    write_submission,
)

ALL_PROBLEMS = ["small-graph", "medium-graph", "large-graph"]


def _load_vectors(path: str):
    with open(path) as f:
        payload = json.load(f)
    entry = payload[0] if isinstance(payload, list) else payload
    return entry["decisionVector"]


def _score_file(path: str, adj_bits, n: int):
    """Return (score, n_feasible, fits) for one submission file, or None if
    the file is malformed.  score = -HV over its feasible vectors."""
    try:
        dvs = _load_vectors(path)
    except Exception:
        return None
    fits: List[Tuple[int, int, List[int]]] = []
    for dv in dvs:
        if not isinstance(dv, list) or len(dv) != n + 1:
            continue
        perm = [int(x) for x in dv[:-1]]
        t = int(dv[-1])
        if sorted(perm) != list(range(n)):
            continue
        w, t_ret = evaluate(perm, t, adj_bits, n)
        fits.append((int(w), int(t_ret), perm))
    feasible = [(w, t) for (w, t, _) in fits if w <= MAX_TW]
    score = -hypervolume_2d(feasible, n) if feasible else 0.0
    return score, len(feasible), fits


def run(here: str, problems: List[str], exclude: List[str], dry_run: bool):
    print("=" * 74)
    print("INSTANCE-SPECIFIC PORTFOLIO  [score = -HV, more negative is better]")
    print("=" * 74)
    grand = []
    for problem in problems:
        n, adj = load_graph(graph_path(here, problem))
        adj_bits = build_adj_bitsets(n, adj)
        target = LEADERBOARD_TARGETS.get(problem)

        files = sorted(
            glob.glob(os.path.join(here, "submissions", problem, "*.json"))
            + glob.glob(os.path.join(here, "submissions", problem, "seeds", "*.json")))
        per_file = []
        union = ParetoArchive()
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            # never pool throwaway tuning stems (e.g. grasp_tune, sa_tune) --
            # they are transient grid cells, not reproducible named methods.
            if stem in exclude or stem.endswith("_tune"):
                continue
            res = _score_file(fp, adj_bits, n)
            if res is None:
                continue
            score, nfeas, fits = res
            per_file.append((score, stem, nfeas))
            for w, t, perm in fits:
                if w <= MAX_TW:
                    union.try_add(w, t, perm)

        if not per_file:
            print(f"\n{problem}: (no readable submissions)")
            continue

        per_file.sort()  # most-negative (best) first
        best_score, best_stem, _ = per_file[0]

        union_top = union.top_k_by_hv_contribution(20, n)
        union_score = -hypervolume_2d([(w, t) for (w, t, _) in union_top], n)

        print(f"\n{problem}  (n = {n}, {len(per_file)} methods scored)")
        print(f"  best single method : {best_stem:<12} {best_score:>16,.0f}")
        print(f"  UNION (top-20)     : {'portfolio':<12} {union_score:>16,.0f}"
              f"   ({union_score - best_score:+,.0f} vs best single)")
        # show the methods contributing to the union front
        contributing = sorted({stem for (s, stem, _) in per_file[:6]})
        print(f"  top contributors   : {', '.join(s for _, s, _ in per_file[:5])}")
        if target is not None:
            print(f"  leaderboard target : {target:>16,}   gap {union_score - target:>+,.0f}")

        grand.append((problem, best_stem, best_score, union_score))

        if not dry_run:
            top = union.top_k_by_hv_contribution(20, n)
            dvs = [list(p) + [int(t)] for (_, t, p) in top]
            out = submission_path(here, problem, "portfolio")
            write_submission(dvs, problem, out)
            print(f"  wrote              : {out}  ({len(dvs)} vectors)")

    print("\n" + "=" * 74)
    print("SUMMARY")
    for problem, best_stem, best_score, union_score in grand:
        print(f"  {problem:<13} best={best_stem:<10} {best_score:>15,.0f}"
              f"   union={union_score:>15,.0f}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=",".join(ALL_PROBLEMS))
    ap.add_argument("--exclude", default="portfolio",
                    help="comma-separated submission stems to skip (default: portfolio)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; do not write portfolio.json")
    args = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]
    run(here, problems, exclude, args.dry_run)


if __name__ == "__main__":
    main()

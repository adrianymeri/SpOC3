#!/usr/bin/env python3
"""
compare_variants.py -- head-to-head harness for the three strongest HC
variants (hc9 HV-accept, hc12 incremental, hc15 = the crossbreed).

Runs each variant on each official problem over one or more seeds at a
fixed wall-time budget, re-scores the written submission from scratch
(authoritative HV via tools.bench_extra.submission_score), and prints a
markdown table: per-(problem, seed) winner, mean ± stdev per variant,
and the head-to-head delta of hc15 against hc9 and hc12.

This answers the question hc15 was built to test: does pairing hc9's
HV-improvement acceptance with hc12's incremental evaluator actually
beat either parent, and on which density regime?

Reproducibility note: scores are wall-clock-budgeted and therefore
hardware-dependent.  Run this on the reference workstation that
produced canonical_seed42.csv before quoting any number in the paper;
sandbox / laptop runs are indicative only.

Usage:
    python3 tools/compare_variants.py                       # default seeds/budgets
    python3 tools/compare_variants.py --seeds 42,1,2,3,4
    python3 tools/compare_variants.py --variants hc9,hc12,hc15 \
        --problems small-graph,medium-graph,large-graph --out cmp.md
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from tools.bench_extra import run_one, ALGO_REGISTRY

# Match tools/multiseed.py budgets so numbers are comparable.
DEFAULT_BUDGETS = {"small-graph": 25.0, "medium-graph": 12.0, "large-graph": 25.0}


def fmt(x):
    return f"{x:,.0f}" if x == x else "nan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="hc9,hc12,hc15")
    ap.add_argument("--problems", default="small-graph,medium-graph,large-graph")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-problem budget (default: 25/12/25)")
    ap.add_argument("--out", default="extra_instances/variant_comparison.md")
    ap.add_argument("--csv", default="extra_instances/variant_comparison.csv")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for v in variants:
        if v not in ALGO_REGISTRY:
            print(f"ERROR: unknown variant {v!r}; known: {sorted(ALGO_REGISTRY)}")
            sys.exit(2)

    # scores[(problem, variant)] = list over seeds
    scores: dict = {}
    rows = []   # raw csv rows
    for problem in problems:
        budget = args.budget if args.budget is not None \
            else DEFAULT_BUDGETS.get(problem, 25.0)
        for seed in seeds:
            for v in variants:
                sc, el = run_one(v, problem, budget, seed, _HERE)
                scores.setdefault((problem, v), []).append(sc)
                rows.append((problem, v, seed, budget, sc, el))
                print(f"  {problem:13s} {v:5s} seed={seed:<3d} "
                      f"budget={budget:5.0f}s  score={fmt(sc):>14s}  "
                      f"({el:5.1f}s)")

    # --- markdown report ---
    out = []
    out.append("# Head-to-head: HC HV-acceptance vs incremental vs crossbreed")
    out.append("")
    out.append(f"- Variants: {', '.join(variants)}")
    out.append(f"- Seeds: {seeds}")
    out.append(f"- Budgets (s): " +
               ", ".join(f"{p}={args.budget if args.budget is not None else DEFAULT_BUDGETS.get(p,25.0):.0f}"
                         for p in problems))
    out.append("- Score = official −HV (higher = closer to 0 = better). "
               "Re-scored from the written submission, not the live print.")
    out.append("- **Wall-clock-budgeted: hardware-dependent. Quote only "
               "reference-workstation runs.**")
    out.append("")

    out.append("## Mean ± stdev per variant")
    out.append("")
    hdr = "| problem | " + " | ".join(variants) + " | best |"
    sep = "|---|" + "|".join("---:" for _ in variants) + "|---|"
    out.append(hdr)
    out.append(sep)
    for problem in problems:
        cells = []
        means = {}
        for v in variants:
            vals = [x for x in scores[(problem, v)] if x == x]
            if vals:
                m = statistics.fmean(vals)
                means[v] = m
                sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{m:,.0f} ± {sd:,.0f}")
            else:
                cells.append("nan")
        best_v = max(means, key=means.get) if means else "—"
        out.append(f"| {problem} | " + " | ".join(cells) + f" | **{best_v}** |")
    out.append("")

    # --- hc15 head-to-head deltas (if hc15 present) ---
    if "hc15" in variants:
        out.append("## hc15 deltas vs parents (positive = hc15 better)")
        out.append("")
        out.append("| problem | hc15 − hc9 | hc15 − hc12 |")
        out.append("|---|---:|---:|")
        for problem in problems:
            m15 = statistics.fmean([x for x in scores[(problem, "hc15")] if x == x]) \
                if any(x == x for x in scores[(problem, "hc15")]) else float("nan")
            def delta(other):
                if other not in variants:
                    return "—"
                ov = [x for x in scores[(problem, other)] if x == x]
                if not ov or m15 != m15:
                    return "nan"
                d = m15 - statistics.fmean(ov)
                return f"{d:+,.0f}"
            out.append(f"| {problem} | {delta('hc9')} | {delta('hc12')} |")
        out.append("")
        out.append("> Reading: hc15 carries hc9's HV-improvement acceptance on "
                   "hc12's incremental evaluator. A positive delta on the sparse "
                   "instances would say the HV-aligned acceptance survives the "
                   "switch to the cheaper evaluator; a positive delta on the dense "
                   "instance would say the extra moves-per-second outweigh hc9's "
                   "more frequent cache rebuilds.")
        out.append("")

    report = "\n".join(out)
    print()
    print(report)

    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    csv_path = os.path.join(_HERE, args.csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "variant", "seed", "budget_s", "score", "elapsed_s"])
        w.writerows(rows)
    print(f"\nWrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

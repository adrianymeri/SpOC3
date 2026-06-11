#!/usr/bin/env python3
"""
analyze_multiseed.py -- read extra_instances/multiseed.csv and emit a
publication-ready summary.

Expects the CSV produced by tools/multiseed.py with columns
  algo, problem, seed, score, elapsed_s

Outputs (to stdout, also written to extra_instances/multiseed_analysis.md):

  1. Per (variant, instance) summary: n, mean, median, min, max, std.
  2. Wilcoxon signed-rank tests, hc9 vs each of {hc5, hc11, hc14}, per
     instance.  Effect size = median pairwise difference; we also
     report the n of paired samples actually used.
  3. A one-paragraph prose synthesis suitable for pasting into the
     paper's Discussion section in place of the current paragraph 4.

Wilcoxon is computed via scipy.stats.wilcoxon if available; if scipy
is missing, we fall back to a simple permutation test (10,000 reps)
so the script has no hard dependency outside the stdlib.

Usage:
    python3 tools/analyze_multiseed.py
    python3 tools/analyze_multiseed.py --csv extra_instances/multiseed.csv
    python3 tools/analyze_multiseed.py --variants hc5,hc9,hc11,hc14
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def _permutation_pvalue(diffs, n_reps=10000, rng_seed=0):
    """Two-sided permutation test on paired differences.

    Null: differences are symmetric around 0 (equivalent to sign-flip).
    We use the mean of differences as the test statistic.
    """
    import random
    rng = random.Random(rng_seed)
    diffs = [d for d in diffs if d != 0.0]
    if not diffs:
        return 1.0
    obs = sum(diffs) / len(diffs)
    n = len(diffs)
    hits = 0
    for _ in range(n_reps):
        s = 0.0
        for d in diffs:
            s += d if rng.random() < 0.5 else -d
        if abs(s / n) >= abs(obs):
            hits += 1
    # Conservative add-one estimator.
    return (hits + 1) / (n_reps + 1)


def paired_pvalue(a, b):
    """Two-sided p-value for the null that a and b are equal in
    location, computed on paired samples (a, b) of the same length."""
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    if HAVE_SCIPY:
        try:
            res = _scipy_wilcoxon(a, b, zero_method="wilcox",
                                  alternative="two-sided")
            return float(res.pvalue)
        except Exception:
            pass
    return _permutation_pvalue(diffs)


def summarise(values):
    """Return (n, mean, median, min, max, std)."""
    n = len(values)
    if n == 0:
        return (0, math.nan, math.nan, math.nan, math.nan, math.nan)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    lo = min(values)
    hi = max(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    return (n, mean, median, lo, hi, std)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def load_multiseed(csv_path):
    """Return data[(algo, problem)] = {seed: score}."""
    data = defaultdict(dict)
    if not os.path.exists(csv_path):
        return data
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            algo = row["algo"]
            problem = row["problem"]
            seed = int(row["seed"])
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            data[(algo, problem)][seed] = score
    return data


def load_canonical(csv_path="submissions"):
    """Pick up the canonical seed=42 score from the saved submission JSON
    so that multiseed.csv can be merged with the seed=42 baseline."""
    import json
    import importlib
    from core import (build_adj_bitsets, evaluate, graph_path,
                      hypervolume_2d, load_graph)
    out = {}
    for problem in ["small-graph", "medium-graph", "large-graph"]:
        gr = graph_path(_HERE, problem)
        n, adj = load_graph(gr)
        ab = build_adj_bitsets(n, adj)
        sub_dir = os.path.join(_HERE, "submissions", problem)
        if not os.path.isdir(sub_dir):
            continue
        for fname in os.listdir(sub_dir):
            if not fname.endswith(".json"):
                continue
            algo = fname[:-5]
            with open(os.path.join(sub_dir, fname)) as f:
                payload = json.load(f)
            entry = payload[0] if isinstance(payload, list) else payload
            fits = [evaluate(dv[:-1], dv[-1], ab, n) for dv in entry["decisionVector"]]
            out[(algo, problem)] = -hypervolume_2d(fits, n)
    return out


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def render_report(data, variants, problems, baseline="hc9"):
    out = []
    out.append("# Multi-seed analysis: official instances")
    out.append("")
    out.append(f"- Variants compared: {', '.join(variants)}")
    out.append(f"- Statistical baseline: {baseline}")
    out.append(f"- Significance test: "
               f"{'Wilcoxon signed-rank (scipy)' if HAVE_SCIPY else 'paired permutation test, 10 000 reps'}")
    out.append("")

    # Per-(variant, problem) summary
    out.append("## 1. Per-cell summary statistics")
    out.append("")
    out.append("| variant | problem | n | mean | median | min | max | σ |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for problem in problems:
        for v in variants:
            scores = list(data.get((v, problem), {}).values())
            n, mean, med, lo, hi, std = summarise(scores)
            if n == 0:
                continue
            out.append(f"| {v} | {problem} | {n} | "
                       f"{mean:,.0f} | {med:,.0f} | {lo:,.0f} | {hi:,.0f} | {std:,.0f} |")
    out.append("")

    # Wilcoxon hc9 vs each other variant, per problem
    out.append(f"## 2. Pairwise test: {baseline} vs each other variant, per problem")
    out.append("")
    out.append(f"| problem | comparison | n_pairs | median Δ ({baseline}-other) | p-value |")
    out.append("|---|---|---:|---:|---:|")
    for problem in problems:
        base = data.get((baseline, problem), {})
        if not base:
            continue
        for v in variants:
            if v == baseline:
                continue
            other = data.get((v, problem), {})
            common = sorted(set(base.keys()) & set(other.keys()))
            if len(common) < 2:
                continue
            base_vals = [base[s] for s in common]
            other_vals = [other[s] for s in common]
            diffs = [b - o for b, o in zip(base_vals, other_vals)]
            med_diff = statistics.median(diffs)
            p = paired_pvalue(base_vals, other_vals)
            out.append(f"| {problem} | {baseline} vs {v} | {len(common)} | "
                       f"{med_diff:+,.0f} | {p:.4f} |")
    out.append("")

    # Prose synthesis
    out.append("## 3. Suggested prose for the paper Discussion")
    out.append("")
    bits = []
    for problem in problems:
        scored = []
        for v in variants:
            scores = list(data.get((v, problem), {}).values())
            if not scores:
                continue
            mean = statistics.fmean(scores)
            std  = statistics.stdev(scores) if len(scores) > 1 else 0.0
            scored.append((v, mean, std, len(scores)))
        if not scored:
            continue
        winner = min(scored, key=lambda r: r[1])  # lower (more negative) is better
        bits.append(f"On {problem} the best mean over {winner[3]} seeds is "
                    f"{winner[0]} ({winner[1]:,.0f} ± {winner[2]:,.0f} σ).")
    out.append("> " + " ".join(bits))
    out.append("")
    out.append("Replace the discussion-paragraph 4 σ tuple with the actual values from §1.")
    out.append("")

    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="extra_instances/multiseed.csv")
    ap.add_argument("--variants", default="hc5,hc9,hc11,hc14")
    ap.add_argument("--problems",
                    default="small-graph,medium-graph,large-graph")
    ap.add_argument("--baseline", default="hc9")
    ap.add_argument("--out", default="extra_instances/multiseed_analysis.md")
    ap.add_argument("--include-canonical", action="store_true", default=True,
                    help="merge in seed=42 score from saved submissions")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    problems = [p.strip() for p in args.problems.split(",")]

    csv_path = os.path.join(_HERE, args.csv)
    data = load_multiseed(csv_path)

    if args.include_canonical:
        canon = load_canonical()
        for (algo, problem), score in canon.items():
            if algo in variants and problem in problems:
                # seed=42 is the canonical baseline
                data[(algo, problem)].setdefault(42, score)

    report = render_report(data, variants, problems, baseline=args.baseline)
    print(report)
    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

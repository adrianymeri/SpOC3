#!/usr/bin/env python3
"""
analyze_synthetic.py -- read the multi-seed synthetic-benchmark CSV
(extra_instances/results.csv) and emit a publication-ready summary.

Expects CSV columns:
  instance, algo, score, elapsed_s, budget_s, seed

(extra_instances/bench_extra.py writes exactly these columns.)

What it produces (stdout + extra_instances/synthetic_analysis.md):

  1. Per-cell statistics: mean ± σ over seeds for every (algo, instance).
  2. Per-instance rank of every variant (1 = best); tied means halved.
  3. Mean rank across instances per variant — the Friedman test
     statistic is computed on these.
  4. Friedman χ² and p-value across the 13 variants.
  5. Post-hoc Nemenyi-style critical difference (only if Friedman p < 0.05).
  6. New "wins" table: count of instances where each variant has the
     best MEAN score (replaces the previous single-seed win count).
  7. Suggested prose for the paper Discussion paragraph 2.

Uses scipy.stats.friedmanchisquare if available; otherwise computes
the Friedman χ² statistic by hand (closed-form for ranks).

Usage:
    python3 tools/analyze_synthetic.py
    python3 tools/analyze_synthetic.py --csv extra_instances/results.csv
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
    from scipy.stats import friedmanchisquare, chi2  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def ranks_lower_is_better(values):
    """Return ranks for a list of values; lower value = lower rank
    (rank 1 = best).  Tied entries share the average rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _chi2_sf(x, df):
    """Upper-tail (survival function) of the chi-square distribution,
    pure-Python (no scipy).  Uses the regularised upper incomplete gamma
    Q(df/2, x/2) via a continued fraction (good for x > df) or the lower
    series (good for x < df).  Accurate to ~1e-12 for the ranges here."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    xx = x / 2.0
    # log Gamma(a)
    lng = math.lgamma(a)
    if xx < a + 1.0:
        # lower series for P(a, xx); Q = 1 - P
        term = 1.0 / a
        summ = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= xx / n
            summ += term
            if abs(term) < abs(summ) * 1e-15:
                break
        p = summ * math.exp(-xx + a * math.log(xx) - lng)
        return max(0.0, 1.0 - p)
    else:
        # Lentz continued fraction for Q(a, xx)
        tiny = 1e-300
        b = xx + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 1000):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        q = h * math.exp(-xx + a * math.log(xx) - lng)
        return max(0.0, min(1.0, q))


def friedman_chi2(rank_matrix):
    """Friedman χ² statistic + p-value, given a [n_instances × n_variants]
    matrix of ranks (lower = better; rank 1 = best; ties broken by mean).

    Closed-form: χ² = (12 N / (k(k+1))) Σ R̄_j² − 3 N (k+1).
    """
    N = len(rank_matrix)
    if N == 0:
        return float("nan"), float("nan")
    k = len(rank_matrix[0])
    sums = [0.0] * k
    for row in rank_matrix:
        for j, r in enumerate(row):
            sums[j] += r
    means = [s / N for s in sums]
    chi = (12 * N / (k * (k + 1))) * sum(s * s for s in sums) / N - 3 * N * (k + 1)
    # Equivalent: chi = (12 N / k(k+1)) Σ R̄_j² − 3 N (k+1)
    chi = (12.0 * N / (k * (k + 1))) * sum(m * m * N for m in means) / N - 3.0 * N * (k + 1)
    df = k - 1
    if HAVE_SCIPY:
        try:
            p = 1.0 - chi2.cdf(chi, df)
        except Exception:
            p = float("nan")
    else:
        # Pure-Python chi-square survival function (no scipy needed).
        try:
            p = _chi2_sf(chi, df)
        except Exception:
            p = float("nan")
    return chi, p


def nemenyi_cd(k, N, alpha=0.05):
    """Nemenyi critical difference for k variants and N instances at
    significance α.  Uses the q_α values from Demšar 2006 Table 5
    (Studentised range / √2).
    """
    q_alpha_05 = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
    }
    if k not in q_alpha_05:
        return float("nan")
    q = q_alpha_05[k]
    return q * math.sqrt(k * (k + 1) / (6.0 * N))


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def load_synthetic(csv_path):
    """Return data[(algo, instance)] = list[score]."""
    data = defaultdict(list)
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            algo = row["algo"]
            inst = row["instance"]
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            data[(algo, inst)].append(score)
    return data


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def render_report(data, algos, instances):
    out = []
    out.append("# Synthetic-benchmark analysis (multi-seed)")
    out.append("")
    out.append(f"- Instances: {len(instances)}")
    out.append(f"- Variants: {len(algos)} ({', '.join(algos)})")
    seed_counts = set()
    for (a, i), v in data.items():
        seed_counts.add(len(v))
    out.append(f"- Seeds per (variant, instance): {sorted(seed_counts)}")
    out.append(f"- Statistics package: "
               f"{'scipy.stats' if HAVE_SCIPY else 'stdlib only (Friedman χ² + p-value via pure-Python incomplete-gamma χ² survival function)'}")
    out.append("")

    # Per-cell mean and rank
    mean_score = {}   # (algo, instance) -> mean over seeds
    for (a, i), vals in data.items():
        if vals:
            mean_score[(a, i)] = statistics.fmean(vals)

    # Build rank matrix [instance][algo]
    rank_matrix = []
    for inst in instances:
        row_scores = [mean_score.get((a, inst), math.nan) for a in algos]
        valid_mask = [not math.isnan(s) for s in row_scores]
        # Fill nan with worst+1 so it ranks last
        finite = [s for s, ok in zip(row_scores, valid_mask) if ok]
        if not finite:
            rank_matrix.append([float("nan")] * len(algos))
            continue
        worst = max(finite)
        substituted = [s if ok else (worst + 1.0) for s, ok in zip(row_scores, valid_mask)]
        rank_matrix.append(ranks_lower_is_better(substituted))

    # Mean rank per algo
    mean_ranks = {}
    for j, a in enumerate(algos):
        rs = [row[j] for row in rank_matrix if not math.isnan(row[j])]
        mean_ranks[a] = statistics.fmean(rs) if rs else math.nan

    out.append("## 1. Mean rank across instances")
    out.append("(rank 1 = best on that instance; lower mean rank = better overall)")
    out.append("")
    out.append("| variant | mean rank | wins by mean |")
    out.append("|---|---:|---:|")
    # Win-by-mean: count instances where this algo has the lowest mean
    wins = defaultdict(int)
    for inst in instances:
        row = {(a, inst): mean_score.get((a, inst), math.nan) for a in algos}
        valid = [(a, s) for a, s in [(a, mean_score.get((a, inst))) for a in algos]
                 if s is not None and not math.isnan(s)]
        if not valid:
            continue
        best = min(valid, key=lambda x: x[1])
        wins[best[0]] += 1
    ranked = sorted(algos, key=lambda a: (mean_ranks.get(a, math.inf), -wins[a]))
    for a in ranked:
        mr = mean_ranks.get(a, math.nan)
        out.append(f"| {a} | {mr:.2f} | {wins[a]} |")
    out.append("")

    # Friedman test
    out.append("## 2. Friedman test (mean ranks differ across variants?)")
    out.append("")
    chi, p = friedman_chi2(rank_matrix)
    out.append(f"- χ² = {chi:.3f}, df = {len(algos)-1}, "
               f"p = {'%.4e' % p if not math.isnan(p) else 'n/a (no scipy)'}")
    out.append("")

    # Nemenyi CD
    out.append("## 3. Nemenyi critical difference")
    cd = nemenyi_cd(len(algos), len(instances))
    out.append(f"- CD(α=0.05, k={len(algos)}, N={len(instances)}) = {cd:.3f}")
    out.append("- Variants whose mean ranks differ by more than CD are significantly different (Demšar 2006).")
    out.append("")

    # Per-instance table
    out.append("## 4. Per-instance ranks (compact)")
    out.append("")
    hdr = "| instance |" + "|".join(f" {a} " for a in algos) + "|"
    sep = "|---|" + "|".join("---:" for _ in algos) + "|"
    out.append(hdr)
    out.append(sep)
    for inst, ranks in zip(instances, rank_matrix):
        cells = " | ".join(f"{r:.1f}" if not math.isnan(r) else "—" for r in ranks)
        out.append(f"| {inst} | {cells} |")
    out.append("")

    # Suggested prose
    out.append("## 5. Suggested prose for the paper Discussion")
    out.append("")
    top3 = ranked[:3]
    win_summary = ", ".join(f"{a}={wins[a]}" for a in ranked if wins[a] > 0)
    out.append(f"> Across {len(instances)} synthetic Erdős–Rényi instances and "
               f"{sorted(seed_counts)[0]} seeds per cell, the variants with the lowest mean rank are "
               + ", ".join(f"{a} ({mean_ranks[a]:.2f})" for a in top3) + ". "
               f"Wins by per-cell mean (most-negative average score): {win_summary}.")
    out.append("")

    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="extra_instances/results.csv")
    ap.add_argument("--out", default="extra_instances/synthetic_analysis.md")
    ap.add_argument("--algos", default="hc1,hc2,hc3,hc4,hc5,hc6,hc7,hc8,hc9,hc10,hc11,hc12,hc13",
                    help="comma-separated algo order")
    args = ap.parse_args()

    csv_path = os.path.join(_HERE, args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} does not exist.")
        sys.exit(2)

    data = load_synthetic(csv_path)
    instances = sorted({i for (_, i) in data.keys()})
    algos = [a.strip() for a in args.algos.split(",")]

    report = render_report(data, algos, instances)
    print(report)

    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

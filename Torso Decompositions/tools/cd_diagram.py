#!/usr/bin/env python3
"""
cd_diagram.py -- Demšar critical-difference diagram for the HC variants.

Turns the existing Friedman / Nemenyi numbers (already computed by
tools/analyze_synthetic.py) into the standard horizontal CD figure
that metaheuristics reviewers expect: variants placed on a mean-rank
axis, with groups whose mean ranks differ by less than the Nemenyi
critical difference (CD) joined by a thick bar ("not significantly
different at α = 0.05").

Reference: Demšar, J. (2006). "Statistical Comparisons of Classifiers
over Multiple Data Sets." JMLR 7:1-30, §3.2.2 and Fig. 1.

This tool is read-only over the benchmark CSV and additive: it imports
the pure-Python rank / Friedman / Nemenyi helpers from
analyze_synthetic.py (no scipy required) so the statistics are
identical to the text report.  The variant set is data-driven, so once
hc15 appears in results.csv (after a workstation regen) it is included
automatically.

Usage:
    python3 tools/cd_diagram.py
    python3 tools/cd_diagram.py --csv extra_instances/results.csv \
        --out extra_instances/cd_diagram.png
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

# --- path bootstrap so "tools.analyze_synthetic" imports cleanly ---
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")           # headless render
import matplotlib.pyplot as plt

from tools.analyze_synthetic import (
    load_synthetic,
    ranks_lower_is_better,
    friedman_chi2,
    nemenyi_cd,
)


def _algo_sort_key(a: str):
    """Order hc1, hc2, ... hc15 numerically; anything else lexicographically."""
    if a.startswith("hc") and a[2:].isdigit():
        return (0, int(a[2:]))
    return (1, a)


def compute_mean_ranks(data, algos, instances):
    """Return (mean_ranks dict, N) using the same recipe as the text report:
    per-instance mean over seeds -> per-instance ranks (1 = best, ties
    averaged) -> mean rank per algo.  nan cells are substituted with
    worst+1 so a variant that crashed on an instance ranks last there."""
    mean_score = {}
    for (a, i), vals in data.items():
        if vals:
            mean_score[(a, i)] = statistics.fmean(vals)

    rank_matrix = []
    for inst in instances:
        row = [mean_score.get((a, inst), math.nan) for a in algos]
        finite = [s for s in row if not math.isnan(s)]
        if not finite:
            continue
        worst = max(finite)
        subbed = [s if not math.isnan(s) else worst + 1.0 for s in row]
        rank_matrix.append(ranks_lower_is_better(subbed))

    mean_ranks = {}
    for j, a in enumerate(algos):
        rs = [r[j] for r in rank_matrix if not math.isnan(r[j])]
        mean_ranks[a] = statistics.fmean(rs) if rs else math.nan
    return mean_ranks, rank_matrix


def cliques(order, mean_ranks, cd):
    """Maximal groups of consecutive (by rank) variants whose rank span
    is <= CD.  Returns list of (lo_rank, hi_rank) bars to draw, dropping
    any clique fully contained in another."""
    bars = []
    n = len(order)
    for i in range(n):
        j = i
        while j + 1 < n and (mean_ranks[order[j + 1]] - mean_ranks[order[i]]) <= cd:
            j += 1
        if j > i:
            bars.append((mean_ranks[order[i]], mean_ranks[order[j]]))
    # Drop bars contained in another bar
    pruned = []
    for lo, hi in bars:
        if any((lo2 <= lo and hi <= hi2 and (lo2, hi2) != (lo, hi))
               for lo2, hi2 in bars):
            continue
        pruned.append((lo, hi))
    # Deduplicate
    out = []
    for b in pruned:
        if b not in out:
            out.append(b)
    return out


def draw(mean_ranks, k, N, cd, chi, p, out_path, title):
    # Sort best (lowest rank) first
    order = sorted([a for a in mean_ranks if not math.isnan(mean_ranks[a])],
                   key=lambda a: mean_ranks[a])
    if not order:
        print("No finite mean ranks; nothing to draw.")
        return
    lo = 1
    hi = max(int(math.ceil(max(mean_ranks[a] for a in order))), 2)

    # Split: better half branches RIGHT (near rank 1), worse half LEFT.
    # Within each side, list in rank order top-down so leader lines never
    # cross (standard Demšar 2006 Fig. 1 layout).
    n = len(order)
    half = (n + 1) // 2
    right = order[:half]                    # best ranks
    left = order[half:]                     # worst ranks
    rows = max(len(right), len(left))

    axis_y = 0.0                            # top horizontal axis
    row_gap = 1.0
    first_row = -1.2                        # first label row sits below axis
    bars_top = 0.55                         # CD cliques drawn just under axis

    fig_h = 1.6 + row_gap * rows * 0.55
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.invert_xaxis()                       # rank 1 (best) on the right
    ax.set_yticks([])
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(lo, hi + 1))
    ax.set_xlabel("mean rank (1 = best)")

    def place(methods, side):
        for i, a in enumerate(methods):
            r = mean_ranks[a]
            y = first_row - i * row_gap
            if side == "right":
                label_x, ha, dx = lo - 0.5, "left", -0.05
            else:
                label_x, ha, dx = hi + 0.5, "right", 0.05
            ax.plot([r, r], [axis_y, y], color="black", lw=0.8)      # drop line
            ax.plot([r, label_x], [y, y], color="black", lw=0.8)     # leader
            ax.text(label_x + dx, y, f"{a}  ({r:.2f})",
                    va="center", ha=ha, fontsize=9)

    place(right, "right")
    place(left, "left")

    # Top axis line
    ax.plot([lo - 0.5, hi + 0.5], [axis_y, axis_y], color="black", lw=1.0)

    # CD bars for non-significant cliques, stacked just below the axis
    bars = cliques(order, mean_ranks, cd)
    for bi, (l, h) in enumerate(bars):
        y = bars_top - 0.16 * bi
        ax.plot([l, h], [y, y], color="firebrick", lw=4, solid_capstyle="round")

    # CD ruler, below the lowest label
    ruler_y = first_row - (rows - 1) * row_gap - 1.0
    ax.plot([hi - cd, hi], [ruler_y, ruler_y], color="black", lw=2)
    for xx in (hi - cd, hi):
        ax.plot([xx, xx], [ruler_y - 0.12, ruler_y + 0.12], color="black", lw=2)
    ax.text(hi - cd / 2, ruler_y + 0.25, f"CD = {cd:.2f}",
            ha="center", fontsize=9)

    ax.set_ylim(ruler_y - 0.6, bars_top + 0.4)

    p_str = ("%.2e" % p) if (p == p) else "n/a"
    ax.set_title(f"{title}\nFriedman χ² = {chi:.1f} (df = {k-1}, p = {p_str}),  "
                 f"k = {k} variants, N = {N} instances,  Nemenyi α = 0.05",
                 fontsize=10, pad=24)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {pdf_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="extra_instances/results.csv")
    ap.add_argument("--out", default="extra_instances/cd_diagram.png")
    ap.add_argument("--algos", default=None,
                    help="comma-separated algo order; default = all present, "
                         "ordered hc1..hcN")
    ap.add_argument("--title", default="HC variants on the synthetic benchmark")
    args = ap.parse_args()

    csv_path = os.path.join(_HERE, args.csv)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} does not exist.")
        sys.exit(2)

    data = load_synthetic(csv_path)
    instances = sorted({i for (_, i) in data.keys()})
    if args.algos:
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    else:
        algos = sorted({a for (a, _) in data.keys()}, key=_algo_sort_key)

    mean_ranks, rank_matrix = compute_mean_ranks(data, algos, instances)
    N = len(rank_matrix)
    k = len(algos)
    chi, p = friedman_chi2(rank_matrix)
    cd = nemenyi_cd(k, N)

    print(f"k = {k} variants, N = {N} instances")
    print(f"Friedman χ² = {chi:.3f}, p = {p:.4e}" if p == p
          else f"Friedman χ² = {chi:.3f}, p = n/a")
    print(f"Nemenyi CD(α=0.05) = {cd:.3f}")
    print("Mean ranks (best first):")
    for a in sorted(algos, key=lambda a: mean_ranks.get(a, math.inf)):
        print(f"  {a:5s}  {mean_ranks[a]:.3f}")

    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    draw(mean_ranks, k, N, cd, chi, p, out_path, args.title)


if __name__ == "__main__":
    main()

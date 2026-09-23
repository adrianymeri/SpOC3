#!/usr/bin/env python3
"""
summarise.py -- rebuild the comparison table from benchmark.csv.

benchmark.py writes its CSV row by row, so a long run that is interrupted
still leaves everything it had finished. This turns whatever is in that file
into the table and the paste block, without re-running anything.

    python3 summarise.py                    # reads benchmark.csv
    python3 summarise.py --csv other.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict

TEAMS = {
    "fast_cma_es":     "fast-cma-es",
    "hri":             "Team HRI",
    "spacekangaroos":  "Spacekangaroos",
    "hill_climbing":   "Hill Climbing (ours)",
    "min_degree":      "min-degree (baseline)",
}

# the comparison sheet's row order
ROWS = (["small-graph", "medium-graph", "large-graph"]
        + [f"synth-{i}" for i in range(1, 8)])

COLUMNS = ["fast_cma_es", "hri", "spacekangaroos", "hill_climbing"]


def main():
    ap = argparse.ArgumentParser(description="Rebuild the table from a CSV.")
    ap.add_argument("--csv", default="benchmark.csv")
    a = ap.parse_args()

    scores = defaultdict(list)
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            scores[(row["instance"], row["solver"])].append(int(row["score"]))

    if not scores:
        raise SystemExit(f"{a.csv} has no rows yet")

    rows = [r for r in ROWS if any((r, s) in scores for s in COLUMNS)]
    missing = sorted({i for i, _ in scores} - set(ROWS))
    rows += missing                       # anything not in the sheet order

    done = sum(len(v) for v in scores.values())
    print(f"{done} runs recorded, {len(rows)} instances\n")

    labels = [TEAMS.get(s, s) for s in COLUMNS]
    width = max(18, max(len(l) for l in labels) + 2)
    print(f"| {'instance':<16} |" + "".join(f" {l:>{width-2}} |" for l in labels))
    print("|" + "-" * 18 + "|" + "".join("-" * width + "|" for _ in COLUMNS))
    for name in rows:
        best_here = min((min(scores[(name, s)]) for s in COLUMNS
                         if (name, s) in scores), default=None)
        line = f"| {name:<16} |"
        for s in COLUMNS:
            if (name, s) in scores:
                best = min(scores[(name, s)])
                line += f" {best:>{width-3},}{'*' if best == best_here else ' '}|"
            else:
                line += f" {'-':>{width-2}} |"
        print(line)
    print("\n* = best on that instance")

    print("\n" + "=" * 78)
    print("PASTE INTO THE SHEET  (tab-separated, sheet row order)")
    print("=" * 78)
    for name in rows:
        cells = [str(min(scores[(name, s)])) if (name, s) in scores else ""
                 for s in COLUMNS]
        print(name + "\t" + "\t".join(cells))

    # seed spread: a wide spread means one run is not enough to trust
    spreads = [(max(v) - min(v), name, s) for (name, s), v in scores.items()
               if len(v) > 1]
    if spreads:
        worst = max(spreads)
        print(f"\nseed spread: median {statistics.median(x[0] for x in spreads):,.0f}"
              f" HV, worst {worst[0]:,} on {worst[1]} / {worst[2]}")


if __name__ == "__main__":
    main()

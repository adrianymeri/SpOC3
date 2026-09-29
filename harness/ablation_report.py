#!/usr/bin/env python3
"""
ablation_report.py -- read the ablation CSVs and say which operator worked.

    python3 ablation_report.py ~/Desktop/SpOC3/_hc_readme/harness

Prints four tables:

  1. SCORE            best of three seeds, per instance per operator
  2. CONTRIBUTION     the same, minus the min-degree construction each run
                      starts from. This is the honest measure: min-degree
                      alone already scores 87-99% of best known, so a raw
                      score column mostly reports the construction, not the
                      search. What the search added is the difference.
  3. ACCEPT RATE      accepted moves / evaluations. Far more sensitive than
                      the score -- an operator can be busy and still not move
                      the front.
  4. ATTRIBUTION      from the `all` run only: when the four compete, which
                      one produced the accepted moves. Shares are reported
                      against each operator's own try count, so an operator
                      is not rewarded merely for being drawn often.

A per-instance seed spread is printed with the scores. A column that leads by
less than that spread has not won anything.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

OPS = ["swap_neighbours", "swap_any", "move_vertex", "reverse_segment"]
COLS = OPS + ["all"]
ROWS = (["small-graph", "medium-graph", "large-graph"]
        + [f"synth-{i}" for i in range(1, 8)])


def fmt(v, width=12):
    return f"{v:>{width},}" if v is not None else " " * (width - 1) + "-"


def table(title, rows, header, note=""):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'instance':<14}" + "".join(f"{h:>17}" for h in header))
    for name, cells in rows:
        print(f"{name:<14}" + "".join(f"{c:>17}" for c in cells))
    if note:
        print(note)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "ablation-*.csv")))
    if not files:
        raise SystemExit(f"no ablation-*.csv found in {folder}")

    scores = defaultdict(list)          # (instance, op) -> [score, ...]
    evals = defaultdict(int)
    accs = defaultdict(int)
    attr_acc = defaultdict(int)         # operator -> accepts in the `all` run
    attr_try = defaultdict(int)
    seen = set()

    for path in files:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["instance"], r["operator"], r["seed"])
                if key in seen:
                    continue
                seen.add(key)
                op = r["operator"]
                if str(r.get("valid", "True")).lower() != "true":
                    continue
                scores[(r["instance"], op)].append(int(r["score"]))
                evals[(r["instance"], op)] += int(r.get("evaluations", 0) or 0)
                accs[(r["instance"], op)] += int(r.get("accepts", 0) or 0)
                if op == "all":
                    for nm in OPS:
                        attr_acc[nm] += int(r.get(f"acc_{nm}", 0) or 0)
                        attr_try[nm] += int(r.get(f"try_{nm}", 0) or 0)

    # min-degree is the construction every climb starts from: the floor.
    base = {}
    md = os.path.join(folder, "benchmark-min_degree.csv")
    if os.path.exists(md):
        for r in csv.DictReader(open(md)):
            v = int(r["score"])
            base[r["instance"]] = min(base.get(r["instance"], v), v)

    print(f"read {len(files)} file(s), {len(seen)} runs")

    # --- 1. scores ------------------------------------------------------
    srows, spreads = [], []
    for name in ROWS:
        cells = []
        for c in COLS:
            v = scores.get((name, c))
            cells.append(f"{min(v):,}" if v else "-")
            if v and len(v) > 1:
                spreads.append(max(v) - min(v))
        if any(x != "-" for x in cells):
            sp = [max(v) - min(v) for c in COLS
                  if (v := scores.get((name, c))) and len(v) > 1]
            cells.append(f"±{max(sp):,}" if sp else "")
            srows.append((name, cells))
    table("1. SCORE  (best of 3 seeds; higher magnitude = better)",
          srows, COLS + ["seed spread"])

    noise = statistics.median(spreads) if spreads else 0
    print(f"\nmedian seed spread across all cells: {noise:,.0f} HV "
          f"-- differences smaller than this are ties")

    # --- 2. contribution over min-degree --------------------------------
    if base:
        crows = []
        for name in ROWS:
            if name not in base:
                continue
            cells = []
            for c in COLS:
                v = scores.get((name, c))
                cells.append(f"{base[name] - min(v):+,}" if v else "-")
            if any(x != "-" for x in cells):
                crows.append((name, cells))
        table("2. CONTRIBUTION OVER MIN-DEGREE  (what the search actually added)",
              crows, COLS)
        print("\nmin-degree alone is the whole score minus these numbers.")
    else:
        print("\n(benchmark-min_degree.csv not found -- skipping contribution)")

    # --- 3. accept rate -------------------------------------------------
    arows = []
    for name in ROWS:
        cells = []
        for c in COLS:
            e, k = evals.get((name, c), 0), accs.get((name, c), 0)
            cells.append(f"{k/e:.4%}" if e else "-")
        if any(x != "-" for x in cells):
            arows.append((name, cells))
    table("3. ACCEPT RATE  (accepted moves / evaluations, all 3 seeds pooled)",
          arows, COLS)

    # --- 4. attribution inside the four-operator run --------------------
    if attr_try:
        print("\n4. ATTRIBUTION  (the `all` run: who earns the accepts when "
              "all four compete)")
        print("-" * 74)
        print(f"{'operator':<20}{'tried':>14}{'accepted':>12}"
              f"{'own rate':>12}{'share of accepts':>18}")
        total = sum(attr_acc.values())
        for nm in sorted(OPS, key=lambda x: -attr_acc[x]):
            t, k = attr_try[nm], attr_acc[nm]
            print(f"{nm:<20}{t:>14,}{k:>12,}"
                  f"{(k/t if t else 0):>11.4%}"
                  f"{(k/total if total else 0):>17.1%}")
        print(f"{'TOTAL':<20}{sum(attr_try.values()):>14,}{total:>12,}")
        print("\nShares near 25% mean the four are interchangeable: the pool "
              "is drawn uniformly,\nso an operator that is no better than the "
              "others earns its draw share and nothing more.")

    # --- verdict --------------------------------------------------------
    if base:
        tot = {}
        for c in COLS:
            gains = [base[n] - min(scores[(n, c)])
                     for n in ROWS if n in base and (n, c) in scores]
            if gains:
                tot[c] = sum(gains)
        if tot:
            print("\nTOTAL CONTRIBUTION SUMMED OVER ALL INSTANCES")
            print("-" * 46)
            for c, v in sorted(tot.items(), key=lambda kv: -kv[1]):
                print(f"  {c:<20}{v:>+14,}")

    missing = [(i, c) for i in ROWS for c in COLS if (i, c) not in scores]
    if missing:
        print(f"\nMISSING {len(missing)} cells:")
        for c in sorted({x for _, x in missing}):
            inst = [i for i, y in missing if y == c]
            print(f"  {c:<20} {len(inst)}: {', '.join(inst[:5])}"
                  f"{' ...' if len(inst) > 5 else ''}")
    else:
        print("\nall 50 cells filled")


if __name__ == "__main__":
    main()

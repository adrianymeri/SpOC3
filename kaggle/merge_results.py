#!/usr/bin/env python3
"""
merge_results.py -- combine the per-solver CSVs from parallel Kaggle sessions.

The four notebooks run at the same time on separate machines, so none of them
can see the others' output. Download each benchmark-<solver>.csv, drop them in
one folder, and run this.

    python3 merge_results.py ~/Downloads/kaggle-results

Prints the five columns for the sheet and writes merged.csv next to the inputs.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

COLS = ["fast_cma_es", "hri", "spacekangaroos", "hill_climbing"]
LABEL = {"fast_cma_es": "fast-cma-es", "hri": "Team HRI",
         "spacekangaroos": "Spacekangaroos", "hill_climbing": "Hill Climbing",
         "min_degree": "min-degree"}
ROWS = (["small-graph", "medium-graph", "large-graph"]
        + [f"synth-{i}" for i in range(1, 8)])


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(folder, "**", "benchmark*.csv"),
                             recursive=True))
    if not files:
        raise SystemExit(f"no benchmark*.csv found under {folder}")

    scores = defaultdict(list)
    seen = set()
    rows = []
    crashed = []
    BUDGET = 1200.0
    for path in files:
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["instance"], r["solver"], r["seed"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
                # An invalid row that finished in seconds is a crash, not a
                # result -- leave the cell empty rather than writing a 0 that
                # reads as "the method found nothing".
                ok = str(r.get("valid", "True")).lower() == "true"
                secs = float(r.get("seconds", 0))
                if not ok and secs < 0.5 * BUDGET:
                    crashed.append((r["instance"], r["solver"], secs))
                    continue
                scores[(r["instance"], r["solver"])].append(int(r["score"]))
    print(f"merged {len(files)} file(s), {len(rows)} runs")
    if crashed:
        by = defaultdict(list)
        for inst, solver, secs in crashed:
            by[solver].append(inst)
        for solver, insts in by.items():
            u = sorted(set(insts))
            print(f"  {len(insts)} CRASHED runs excluded ({LABEL.get(solver, solver)}"
                  f": {', '.join(u)}) -- cells left empty, not 0")
    print()

    out = os.path.join(folder, "merged.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    spreads = [max(v) - min(v) for v in scores.values() if len(v) > 1]
    noise = statistics.median(spreads) if spreads else 0

    have_md = any(s == "min_degree" for _, s in scores)
    header = ["instance"] + [LABEL[c] for c in COLS] + ["Gap from Best"]
    if have_md:
        header.append("min-degree baseline")
    print(",".join(header))

    for name in ROWS:
        vals = {c: min(scores[(name, c)]) for c in COLS if (name, c) in scores}
        if not vals:
            continue
        live = [v for v in vals.values() if v != 0]
        best = min(live) if live else 0
        hc = vals.get("hill_climbing")
        gap = (hc - best) if hc is not None else ""
        line = [name] + [str(vals.get(c, "")) for c in COLS] + [str(gap)]
        if have_md:
            md = scores.get((name, "min_degree"))
            line.append(str(min(md)) if md else "")
        print(",".join(line))

    print(f"\nseed spread (median): {noise:,.0f} HV "
          f"-- gaps smaller than this are ties, not wins")

    missing = [(i, c) for i in ROWS for c in COLS if (i, c) not in scores]
    if missing:
        print(f"\nMISSING {len(missing)} cells:")
        for solver in sorted({c for _, c in missing}):
            inst = [i for i, c in missing if c == solver]
            print(f"  {LABEL[solver]:<16} {len(inst)} instances: "
                  f"{', '.join(inst[:5])}{' ...' if len(inst) > 5 else ''}")
    else:
        print("\nall 40 cells filled")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_report.py -- turn the benchmark CSVs into the results spreadsheet.

    python3 make_report.py --template "Computational results.xlsx" --out final.xlsx

Re-run it whenever more runs land; it always reads whatever is in the CSVs.

How a cell is decided
---------------------
Best of the seeds, where a run counts if it was valid, or if it was invalid
but spent the budget -- that second case is a real "found nothing in 20
minutes" and is written as an em dash. A run that came back invalid in
seconds is a crash and is dropped, so the cell reads empty rather than
claiming a method failed.

Gap from Best is written as a live formula against the Best Known column, so
it updates itself when the AlbShkenca column is filled in.

Ties
----
Differences below the seed-noise floor (median spread across seeds) are not
wins. The script computes that floor from the data and flags any row whose
top two solvers sit inside it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict

import openpyxl
from openpyxl.styles import Font

SOLVERS = ["fast_cma_es", "hri", "spacekangaroos", "hill_climbing"]
LABEL = {"fast_cma_es": "fcmaes", "hri": "Team HRI",
         "spacekangaroos": "Spacekangaroos", "hill_climbing": "Hill Climbing",
         "min_degree": "min-degree"}
ROWS = (["small-graph", "medium-graph", "large-graph"]
        + [f"synth-{i}" for i in range(1, 8)])
# published ESA SpOC-3 leaderboard tops; the synthetics are new, so they have none
LEADERBOARD = {"small-graph": -1829919, "medium-graph": -1745122,
               "large-graph": -5493062}
BUDGET = 1200.0
NONE_TXT = "—"          # no feasible solution inside the budget

# Best of 8 min-degree constructions with random tie-breaking -- exactly what
# hill_climbing builds before it searches a single step. Seconds of compute.
MD8 = {"small-graph": -1813181, "medium-graph": -1615495, "large-graph": -4793652,
       "synth-1": -1818533, "synth-2": -1819873, "synth-3": -1819873,
       "synth-4": -1553005, "synth-5": -1560655, "synth-6": -4793652,
       "synth-7": -4793652}


def load(folder):
    """Returns scores[(instance, solver)] = [score, ...] and a crash list."""
    scores, seen, crashed = defaultdict(list), set(), []
    for path in sorted(glob.glob(os.path.join(folder, "benchmark-*.csv"))):
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["instance"], r["solver"], r["seed"])
                if key in seen:
                    continue
                seen.add(key)
                ok = str(r.get("valid", "True")).lower() == "true"
                secs = float(r.get("seconds", 0))
                if not ok and secs < 0.5 * BUDGET:
                    crashed.append(key)
                    continue
                scores[(r["instance"], r["solver"])].append(int(r["score"]))
    return scores, crashed


def cell_value(scores, inst, solver):
    """Best of the seeds: a number, the em dash, or None if nothing ran."""
    v = scores.get((inst, solver))
    if not v:
        return None
    live = [x for x in v if x != 0]
    return min(live) if live else NONE_TXT


def main():
    ap = argparse.ArgumentParser(description="Build the results spreadsheet.")
    ap.add_argument("--csv-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scores, crashed = load(a.csv_dir)
    spreads = [max(v) - min(v) for v in scores.values() if len(v) > 1]
    noise = statistics.median(spreads) if spreads else 0

    wb = openpyxl.load_workbook(a.template)
    for ws in wb.worksheets:
        # openpyxl strips the cached value of a spilling array formula and
        # LibreOffice cannot evaluate it, so 300 cells would become #NAME?.
        a2 = ws["A2"].value
        if a2 is not None and "SEQUENCE" in str(getattr(a2, "text", a2)):
            for r in range(2, ws.max_row + 1):
                ws.cell(row=r, column=1).value = r - 1

        ws["J1"] = "min-degree x8 (no search)"
        ws["K1"] = "Significance"
        last = 1
        for row in range(2, ws.max_row + 1):
            inst = ws.cell(row=row, column=2).value
            if inst not in ROWS:
                continue
            last = row

            vals = {s: cell_value(scores, inst, s) for s in SOLVERS}
            for col, s in zip("FGHI", SOLVERS):
                c = ws[f"{col}{row}"]
                c.value = "" if vals[s] is None else vals[s]
                c.number_format = "#,##0"

            # Hill climbing runs EIGHT independent min-degree constructions
            # internally before any search. Showing a single construction as
            # the baseline understates what it gets for free -- on synth-3 by
            # 2,679 HV, which is most of its apparent margin. Match the
            # baseline to what the solver actually starts from.
            md = MD8.get(inst, cell_value(scores, inst, "min_degree"))
            ws[f"J{row}"] = "" if md is None else md
            ws[f"J{row}"].number_format = "#,##0"

            numeric = {s: v for s, v in vals.items() if isinstance(v, int)}
            if numeric:
                order = sorted(numeric.items(), key=lambda kv: kv[1])
                if ws[f"D{row}"].value is None:      # no leaderboard for synthetics
                    ws[f"D{row}"] = order[0][1]
                    ws[f"D{row}"].number_format = "#,##0"
                    ws[f"E{row}"] = LABEL[order[0][0]] + " (this study)"
                gap = (order[1][1] - order[0][1]) if len(order) > 1 else None
                if gap is not None and gap < noise:
                    ws[f"K{row}"] = f"tie (top two within {gap:,} HV)"
                    ws[f"K{row}"].font = Font(name="Arial", size=10, italic=True)

            # percentage behind Best Known: uses our own best once column N is
            # filled, else the Hill Climbing column
            ws[f"M{row}"] = (f'=IFERROR(IF(N{row}<>"",(D{row}-N{row})/D{row},'
                             f'IF(ISNUMBER(I{row}),(D{row}-I{row})/D{row},"")),"")')
            ws[f"M{row}"].number_format = "0.00%"

        notes = [
            f"— = no feasible solution within the {BUDGET/60:.0f}-minute "
            "budget (every ordering exceeded the 500-width cap); not a missing run.",
            "Protocol: 1200 s per instance per seed, 3 seeds, best reported. "
            "fcmaes / Team HRI / Hill Climbing / min-degree on one CPU core each; "
            "Spacekangaroos = cuda-torso on an NVIDIA T4 GPU at batch 1024 "
            "(512 for n=2426). All results re-scored and validated through the "
            "same esa_eval.py.",
            f"Seed-noise floor = {noise:,.0f} HV (median spread across seeds). "
            "Differences below it are ties, not wins - see the Significance column.",
            "Spacekangaroos' three runs are independent samples, not seeded "
            "replicates: cuda-torso draws from torch's global RNG, which the "
            "harness does not control. Its spread is correspondingly wider.",
            "Best Known for small/medium/large = published ESA SpOC-3 leaderboard "
            "top. For synth-1..7 (new instances, no leaderboard) = best reached "
            "in this study, with the method named in the Team column.",
            "synth-1..3 are sparse and low-treewidth; min-degree is already "
            "within ~0.1% of every method there, so those rows do not separate "
            "approaches. Reported as ties by design.",
            "min-degree x8 = best of eight greedy constructions with random "
            "tie-breaking, which is what Hill Climbing builds before searching. "
            "On synth-3 it reaches -1,819,873 in ~2 s against Hill Climbing's "
            "-1,819,944 in 20 minutes: the search contributes 71 HV (0.004%).",
        ]
        for i, text in enumerate(notes):
            c = ws.cell(row=last + 2 + i, column=2)
            c.value = text
            c.font = Font(name="Arial", size=9, italic=True)

    wb.save(a.out)

    print(f"seed-noise floor: {noise:,.0f} HV")
    if crashed:
        print(f"{len(crashed)} crashed runs dropped")
    missing = [(i, s) for i in ROWS for s in SOLVERS
               if cell_value(scores, i, s) is None]
    if missing:
        print(f"STILL MISSING {len(missing)} cells:")
        for inst, s in missing:
            print(f"   {inst} / {LABEL[s]}")
    else:
        print("all 40 solver cells filled")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
report.py -- regenerate every table in the results sheet, from the CSVs.

One command reproduces the whole sheet. If a number appears in the write-up
and not in this output, it does not belong in the write-up.

    python3 report.py                  # all four blocks
    python3 report.py --block 2        # just the controlled experiment

Block 1  Comparison          this study against the three leaderboard entries
Block 2  Controlled experiment   one factor changed at a time
Block 3  Operator attribution    who earns the accepts when all four compete
Block 4  Operator ablation       each operator run alone, per instance

Everything is derived from benchmark-*.csv and ablation-*.csv in this folder,
except the Upper Bound column, which calls bound.py. Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

INSTANCES = (["small-graph", "medium-graph", "large-graph"]
             + [f"synth-{i}" for i in range(1, 8)])
OPERATORS = ["move_vertex", "swap_any", "reverse_segment", "swap_neighbours"]

# The three published ESA SpOC-3 leaderboard tops. The synthetic instances
# have no leaderboard, so their "best known" is the best in the row.
PUBLISHED = {"small-graph": -1829919, "medium-graph": -1745122,
             "large-graph": -5493062}

LEADERBOARD = [("fast_cma_es", "fast-cma-es"), ("hri", "Team HRI"),
               ("spacekangaroos", "Spacekangaroos")]
STUDY = [("hill_climbing", "Hill Climbing"), ("sa_front", "Simulated Annealing"),
         ("vns_front", "VNS"), ("grasp_front", "GRASP")]


def load(name):
    """best-of-seeds per instance, and the per-instance seed spreads."""
    path = os.path.join(HERE, f"benchmark-{name}.csv")
    if not os.path.exists(path):
        return {}, []
    per = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if str(r.get("valid", "True")).lower() != "true":
            continue
        per[r["instance"]].append(int(r["score"]))
    best = {i: min(v) for i, v in per.items()}
    spread = [max(v) - min(v) for v in per.values() if len(v) > 1]
    return best, spread


def noise_floor(names):
    """Median seed spread over the given solvers -- differences below it are ties."""
    allsp = []
    for n in names:
        allsp += load(n)[1]
    return statistics.median(allsp) if allsp else 0


def block1(M, floor):
    print("\nBLOCK 1 -- Comparison: this study against the ESA SpOC-3 leaderboard")
    print("=" * 132)
    try:
        from bound import report as bound_report
        bounds = {i: bound_report(os.path.join(ROOT, "data", f"{i}.gr"))[2]
                  for i in INSTANCES}
    except Exception as e:                       # bound.py is slow; stay usable
        print(f"  (bound.py unavailable: {e} -- Upper Bound column omitted)")
        bounds = {}

    head = (f"{'instance':<14}{'n':>7}{'upper bound':>15}{'best known':>15}  "
            f"{'held by':<16}" + "".join(f"{lbl:>15}" for _, lbl in LEADERBOARD + STUDY)
            + f"{'best study':>15}  {'method':<20}{'gap study':>11}{'gap HC':>9}"
            f"{'margin':>11}")
    print(head)
    for i in INSTANCES:
        row = {k: M[k].get(i) for k, _ in LEADERBOARD + STUDY}
        live = {k: v for k, v in row.items() if v is not None}
        bk = PUBLISHED.get(i, min(live.values()))
        held = "" if i in PUBLISHED else next(
            lbl for k, lbl in LEADERBOARD + STUDY if live.get(k) == bk)
        ours = {k: row[k] for k, _ in STUDY if row[k] is not None}
        bo = min(ours.values())
        bm = next(lbl for k, lbl in STUDY if ours.get(k) == bo)
        vals = sorted(live.values())
        margin = vals[1] - vals[0] if len(vals) > 1 else 0
        n = N[i]
        ub = f"{-bounds[i]:,}" if i in bounds else "-"
        cells = "".join(f"{row[k]:>15,}" if row[k] is not None else f"{'-':>15}"
                        for k, _ in LEADERBOARD + STUDY)
        print(f"{i:<14}{n:>7,}{ub:>15}{bk:>15,}  {held:<16}{cells}"
              f"{bo:>15,}  {bm:<20}{abs(bo-bk)/abs(bk):>11.2%}"
              f"{abs(row['hill_climbing']-bk)/abs(bk):>9.2%}{margin:>11,}")
    print(f"\n  seed-noise floor {floor:,.0f} HV -- a margin below this is a tie, not a win")


def block2(M, floor):
    print("\nBLOCK 2 -- Controlled experiment: one factor changed at a time")
    print("=" * 110)
    print("  All four share Hill Climbing's skeleton: eight target widths, a construction")
    print("  per width, the same four generic operators, the same objective.\n")
    hc = M["hill_climbing"]
    spec = [("sa_front", "Simulated Annealing", "min-degree", "Metropolis"),
            ("vns_front", "VNS", "min-degree", "shake ladder"),
            ("grasp_front", "GRASP", "randomised (RCL)", "keep if better")]
    print(f"{'method':<22}{'construction':<20}{'acceptance':<18}"
          f"{'net vs HC':>12}{'best inst':>12}{'W':>4}{'T':>4}{'L':>4}")
    print(f"{'Hill Climbing':<22}{'min-degree':<20}{'keep if better':<18}"
          f"{'-- baseline --':>12}")
    for key, lbl, con, acc in spec:
        d = [hc[i] - M[key][i] for i in INSTANCES]
        w = sum(1 for x in d if x > floor)
        l = sum(1 for x in d if x < -floor)
        print(f"{lbl:<22}{con:<20}{acc:<18}{sum(d):>+12,}{max(d):>+12,}"
              f"{w:>4}{10-w-l:>4}{l:>4}")


def block3():
    path = os.path.join(HERE, "ablation-all.csv")
    if not os.path.exists(path):
        return
    print("\nBLOCK 3 -- Operator attribution: who earns the accepts when all four compete")
    print("=" * 96)
    tried = defaultdict(int)
    acc = defaultdict(int)
    lead = defaultdict(int)
    per = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(path)):
        for o in OPERATORS:
            tried[o] += int(r[f"try_{o}"])
            acc[o] += int(r[f"acc_{o}"])
            per[r["instance"]][o] += int(r[f"acc_{o}"])
    live = 0
    for i, d in per.items():
        if sum(d.values()):
            live += 1
            lead[max(OPERATORS, key=lambda o: d[o])] += 1
    total = sum(acc.values())
    print(f"{'operator':<20}{'tried':>13}{'accepted':>11}{'accept rate':>14}"
          f"{'share':>9}{'instances led':>15}")
    for o in sorted(OPERATORS, key=lambda x: -acc[x]):
        print(f"{o:<20}{tried[o]:>13,}{acc[o]:>11,}{acc[o]/tried[o]:>14.4%}"
              f"{acc[o]/total:>9.1%}{lead[o]:>15}")
    print(f"{'TOTAL':<20}{sum(tried.values()):>13,}{total:>11,}"
          f"{total/sum(tried.values()):>14.4%}{1:>9.1%}{live:>15}")
    print(f"\n  instances led is out of the {live} instances that accepted anything at all")


def block4():
    data = {}
    for o in OPERATORS:
        path = os.path.join(HERE, f"ablation-{o}.csv")
        if not os.path.exists(path):
            return
        ev, ac, sc = defaultdict(int), defaultdict(int), {}
        for r in csv.DictReader(open(path)):
            i = r["instance"]
            ev[i] += int(r["evaluations"])
            ac[i] += int(r["accepts"])
            sc[i] = min(sc.get(i, int(r["score"])), int(r["score"]))
        data[o] = (ev, ac, sc)
    print("\nBLOCK 4 -- Operator ablation: each operator run alone")
    print("=" * 120)
    print(f"{'instance':<14}" + "".join(f"{o:>18}" for o in OPERATORS)
          + f"{'best score':>18}{'its accepts':>13}{'spread':>10}")
    for i in INSTANCES:
        rates = []
        for o in OPERATORS:
            ev, ac, _ = data[o]
            r = ac[i] / ev[i] if ev[i] else 0
            rates.append(f"{r:.4%}" if r else "none")
        best = min(OPERATORS, key=lambda o: data[o][2][i])
        scores = [data[o][2][i] for o in OPERATORS]
        print(f"{i:<14}" + "".join(f"{x:>18}" for x in rates)
              + f"{best:>18}{data[best][1][i]:>13,}{max(scores)-min(scores):>10,}")
    print("\n  'none' means not one move was ever accepted. Where the best-scoring")
    print("  operator accepted nothing, the score is not measuring the search.")


N = {}


def main():
    ap = argparse.ArgumentParser(description="Regenerate the results sheet.")
    ap.add_argument("--block", type=int, choices=[1, 2, 3, 4], default=0)
    a = ap.parse_args()

    for r in csv.DictReader(open(os.path.join(HERE, "benchmark-hill_climbing.csv"))):
        N[r["instance"]] = int(r["n"])

    names = [k for k, _ in LEADERBOARD + STUDY]
    M = {k: load(k)[0] for k in names}
    missing = [k for k in names if not M[k]]
    if missing:
        raise SystemExit(f"missing benchmark CSVs for: {', '.join(missing)}")
    floor = noise_floor([k for k, _ in STUDY])

    if a.block in (0, 1):
        block1(M, floor)
    if a.block in (0, 2):
        block2(M, floor)
    if a.block in (0, 3):
        block3()
    if a.block in (0, 4):
        block4()


if __name__ == "__main__":
    main()

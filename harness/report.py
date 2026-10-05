#!/usr/bin/env python3
"""
report.py -- regenerate every table in the results sheet, from the CSVs.

One command reproduces the whole sheet. If a number appears in the write-up
and not in this output, it does not belong in the write-up.

    python3 report.py                  # blocks 1-4
    python3 report.py --block 2        # just the controlled experiment

Block 1  Comparison          this study against the three leaderboard entries
Block 2  Controlled experiment   one factor changed at a time, + replication
Block 3  Operator attribution    who earns the accepts when all four compete
Block 4  Operator ablation       each operator run alone, per instance
Block 5  Construction decomposition  opt-in (slow); --block 5

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

# The three published ESA SpOC-3 leaderboard tops, and who holds them. These
# are external facts rather than measurements -- they are the only numbers in
# this file not derived from the CSVs, and they are declared here so that the
# sheet contains nothing this script cannot emit.
#
# The synthetic instances have no leaderboard, so their "best known" is the
# best in the row and their holder is read off the row.
PUBLISHED = {"small-graph": -1829919, "medium-graph": -1745122,
             "large-graph": -5493062}
HELD_BY = {"small-graph": "Spacekangaroos", "medium-graph": "Team HRI",
           "large-graph": "Spacekangaroos"}

LEADERBOARD = [("fast_cma_es", "fast-cma-es"), ("hri", "Team HRI"),
               ("spacekangaroos", "Spacekangaroos")]
STUDY = [("hill_climbing", "Hill Climbing"), ("sa_front", "Simulated Annealing"),
         ("vns_front", "VNS"), ("grasp_front", "GRASP")]

# Hill Climbing was run twice, a full 10 instances x 3 seeds each time. Because
# the climb loop is bounded by wall-clock rather than by an iteration count, the
# two runs are two independent samples of the same method, not a repeat -- see
# REPLICATE below, which measures what that costs.
#
# Every number in the results sheet is computed against v2, so v2 is pinned
# here as the baseline. v1 is kept on disk deliberately: it is the replication
# evidence, and it is the strongest single argument in the chapter.
#
# This override exists so that the choice is made in exactly one place. Before
# it, report.py read v1 while the sheet showed v2, and the two disagreed on the
# SIGN of two of the four arms.
BASELINE = {"hill_climbing": "benchmark-hill_climbing-v2.csv"}
REPLICATE = ("hill_climbing", "benchmark-hill_climbing.csv", "v1")


def load(name, filename=None):
    """best-of-seeds per instance, and the per-instance seed spreads."""
    path = os.path.join(HERE, filename or BASELINE.get(name, f"benchmark-{name}.csv"))
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
    """Median seed spread over the given solvers -- the OPTIMISTIC floor.

    Two seeds inside one run share a machine, a thermal state and a session,
    so their agreement measures less than it appears to. Prefer
    resample_error() below wherever both are available.
    """
    allsp = []
    for n in names:
        allsp += load(n)[1]
    return statistics.median(allsp) if allsp else 0


def resample_error(hc):
    """How far the baseline differs from an independent run of ITSELF.

    The honest floor -- nothing can be called a win by less than the amount
    the reference method varies between two runs of the same code at the same
    budget.

    TWO figures, and they are not interchangeable:

      aggregate  the sum over all ten instances. Use it against Block 2's
                 "net HV" column, which is also a sum over ten instances.
      per_inst   the worst single-instance difference. Use it against Block
                 1's per-instance "margin" column.

    Comparing a per-instance margin against the aggregate is a category
    error -- it inflates the floor by roughly the instance count and turns
    real separations into spurious ties. An earlier version of this function
    did exactly that.

    Returns (aggregate, per_inst, {instance: difference}) or None.
    """
    name, alt_csv, _ = REPLICATE
    alt, _ = load(name, alt_csv)
    if not alt:
        return None
    inst = [i for i in INSTANCES if i in hc and i in alt]
    if not inst:
        return None
    per = {i: abs(alt[i] - hc[i]) for i in inst}
    return (abs(sum(alt[i] - hc[i] for i in inst)), max(per.values()), per)


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

    margins = {}
    head = (f"{'instance':<14}{'n':>7}{'upper bound':>15}{'best known':>15}  "
            f"{'held by':<16}" + "".join(f"{lbl:>15}" for _, lbl in LEADERBOARD + STUDY)
            + f"{'best study':>15}  {'method':<20}{'gap study':>11}{'gap HC':>9}"
            f"{'margin':>11}")
    print(head)
    for i in INSTANCES:
        row = {k: M[k].get(i) for k, _ in LEADERBOARD + STUDY}
        live = {k: v for k, v in row.items() if v is not None}
        bk = PUBLISHED.get(i, min(live.values()))
        held = HELD_BY.get(i) or next(
            lbl for k, lbl in LEADERBOARD + STUDY if live.get(k) == bk)
        ours = {k: row[k] for k, _ in STUDY if row[k] is not None}
        bo = min(ours.values())
        bm = next(lbl for k, lbl in STUDY if ours.get(k) == bo)
        vals = sorted(live.values())
        margin = vals[1] - vals[0] if len(vals) > 1 else 0
        margins[i] = margin
        n = N[i]
        ub = f"{-bounds[i]:,}" if i in bounds else "-"
        cells = "".join(f"{row[k]:>15,}" if row[k] is not None else f"{'-':>15}"
                        for k, _ in LEADERBOARD + STUDY)
        print(f"{i:<14}{n:>7,}{ub:>15}{bk:>15,}  {held:<16}{cells}"
              f"{bo:>15,}  {bm:<20}{abs(bo-bk)/abs(bk):>11.2%}"
              f"{abs(row['hill_climbing']-bk)/abs(bk):>9.2%}{margin:>11,}")
    print(f"\n  seed-noise floor {floor:,.0f} HV (median within-run seed spread).")
    re_ = resample_error(M["hill_climbing"])
    if re_:
        _, per_inst, _ = re_
        safe = [i for i in INSTANCES if margins.get(i, 0) >= per_inst]
        tie = [i for i in INSTANCES if margins.get(i, 0) < per_inst]
        print(f"  Per-instance re-sample floor {per_inst:,} HV -- the largest amount the")
        print(f"  baseline differs from an independent run of itself on any one instance.")
        print(f"  Judge the margin column against this, NOT against Block 2's aggregate:\n")
        print(f"    margin clears {per_inst:,} HV:  {len(safe)} of {len(margins)}"
              f"  ({', '.join(safe)})")
        print(f"    margin below it (tie):    {len(tie)} of {len(margins)}"
              f"  ({', '.join(tie)})")
        print(f"\n  The 'method' column names a best performer on every row, but on {len(tie)} of them")
        print(f"  it is naming the luckiest sample rather than a better method. Those rows")
        print(f"  should be read as ties.")


# Each arm is Hill Climbing with exactly one of the three components replaced.
# Bold entry = the thing that differs from the baseline row.
ARMS = [
    # key            label                  construction        acceptance        move pool
    ("hc_bottleneck", "Move pool",          "min-degree",       "keep if better", "bottleneck-aware"),
    ("vns_front",     "VNS",                "min-degree",       "shake ladder",   "generic x4"),
    ("sa_front",      "Simulated Annealing", "min-degree",      "Metropolis",     "generic x4"),
    ("grasp_front",   "GRASP",              "randomised (RCL)", "keep if better", "generic x4"),
]


def _delta(hc, scores, floor):
    """net / best / W-T-L over the instances both sides actually have."""
    inst = [i for i in INSTANCES if i in hc and i in scores]
    d = [hc[i] - scores[i] for i in inst]
    w = sum(1 for x in d if x > floor)
    l = sum(1 for x in d if x < -floor)
    return sum(d), max(d), w, len(d) - w - l, l, len(d)


def block2(M, floor):
    print("\nBLOCK 2 -- Controlled experiment: one component changed at a time")
    print("=" * 122)
    print("  Every row is Hill Climbing with ONE component swapped. All share the same")
    print("  skeleton: eight target widths, a construction per width, minimise t at the")
    print("  target width, 1,200 s per seed, three seeds, one core. Sorted by net effect.\n")
    hc = M["hill_climbing"]

    # W/T/L are per-instance counts, so they are thresholded with the
    # per-instance re-sample floor -- NOT the within-run seed spread, which
    # is about fourteen times smaller and reports three of these four arms
    # as wins. The results sheet uses the same figure; keep them identical.
    re_ = resample_error(hc)
    wtl_floor = re_[1] if re_ else floor

    print(f"{'method':<22}{'construction':<20}{'acceptance':<17}{'move pool':<19}"
          f"{'net vs HC':>12}{'best inst':>12}{'W':>4}{'T':>4}{'L':>4}{'n':>4}")
    print(f"{'Hill Climbing':<22}{'min-degree':<20}{'keep if better':<17}"
          f"{'generic x4':<19}{'-- baseline --':>12}")
    rows = []
    for key, lbl, con, acc, pool in ARMS:
        if not M.get(key):
            continue
        rows.append((_delta(hc, M[key], wtl_floor), key, lbl, con, acc, pool))
    for (net, best, w, t, l, cnt), key, lbl, con, acc, pool in sorted(rows):
        print(f"{lbl:<22}{con:<20}{acc:<17}{pool:<19}{net:>+12,}{best:>+12,}"
              f"{w:>4}{t:>4}{l:>4}{cnt:>4}")
    print(f"\n  W/T/L thresholded at {wtl_floor:,} HV (per-instance re-sample floor), not at")
    print(f"  the {floor:,.0f} HV within-run seed spread. 'n' is instances compared -- below 10")
    print("  means that arm has an instance still re-running.")

    # --- replication: the same baseline, sampled twice -----------------------
    #
    # Hill Climbing was run twice at full budget. Net-vs-baseline is linear in
    # the baseline, so re-drawing it shifts EVERY arm by the same amount --
    # there is one number here, not one per arm. That number is the baseline's
    # own sampling error, and it is the honest noise floor for this column:
    # a method cannot be said to beat Hill Climbing by less than the amount
    # Hill Climbing differs from itself.
    name, alt_csv, alt_lbl = REPLICATE
    alt, _ = load(name, alt_csv)
    if not alt:
        return
    inst = [i for i in INSTANCES if i in hc and i in alt]
    shift = sum(alt[i] - hc[i] for i in inst)
    per = max(abs(alt[i] - hc[i]) for i in inst)
    print(f"\n  (Per-instance differences run 1 - {per:,} HV; that figure, not the")
    print(f"  aggregate below, is the one to use against Block 1's margin column.)")
    print(f"\n  Replication -- Hill Climbing against an independent run of itself")
    print(f"  ({alt_lbl}, {os.path.basename(alt_csv)}, same budget, same seeds):\n")
    print(f"    baseline re-sample shifts every arm by   {shift:>+12,} HV")
    print(f"    worst single instance moves by           {per:>12,} HV")
    print(f"    median within-run seed spread            {floor:>12,.0f} HV")
    print(f"\n  The first number is {abs(shift)/floor:.0f}x the third. The seed spread understates")
    print("  the real uncertainty badly, because two seeds inside one run share a")
    print("  machine and a session; two runs do not. Judge the column against the")
    print(f"  re-sample figure:\n")
    print(f"  {'method':<22}{'net vs HC':>13}{'vs floor':>11}   verdict")
    for key, lbl, *_ in ARMS:
        if not M.get(key):
            continue
        net = _delta(hc, M[key], floor)[0]
        mult = net / abs(shift) if shift else 0
        verdict = ("indistinguishable from zero" if abs(net) < abs(shift)
                   else "marginal" if abs(net) < 2 * abs(shift)
                   else "REAL")
        print(f"  {lbl:<22}{net:>+13,}{mult:>10.1f}x   {verdict}")
    print("\n  Every component of the SEARCH lands inside the baseline's own sampling")
    print("  error. Construction clears it by an order of magnitude. That is the")
    print("  chapter's claim, measured against the only yardstick that cannot be")
    print("  accused of being too generous to it.")


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


def block5(M):
    """Where Hill Climbing's gain comes from: construction or search?

    Slow (it builds eight min-degree orderings per instance, ~50 s each on the
    2,426-vertex graphs), so it is opt-in rather than part of the default run.

    This is the measurement that makes the GRASP result precise. GRASP differs
    from Hill Climbing by using a randomised construction -- but Hill Climbing
    already builds eight constructions of its own, one per target width. If
    those eight were genuinely different, "construction" could not be the
    lever. They are not: min-degree tie-breaking produces the same ordering
    over and over on most instances, so Hill Climbing effectively has one
    starting point and GRASP has eight. The lever is construction DIVERSITY,
    not construction count.
    """
    import random
    from torso import Graph, Solution, Front
    import hill_climbing

    print("\nBLOCK 5 -- Where Hill Climbing's gain comes from")
    print("=" * 86)
    print(f"{'instance':<14}{'1 construction':>17}{'8 constructions':>18}"
          f"{'full run':>15}{'from constr.':>14}{'from search':>14}")
    tc = ts = 0
    for i in INSTANCES:
        g = Graph.load(os.path.join(ROOT, "data", f"{i}.gr"))
        f1 = Front(g.n)
        f1.add_solution(Solution(g, hill_climbing.Starts.min_degree(g, random.Random(0))))
        f8 = Front(g.n)
        for s in range(8):
            f8.add_solution(Solution(g, hill_climbing.Starts.min_degree(g, random.Random(s))))
        one, eight, full = f1.score(), f8.score(), M["hill_climbing"][i]
        tc += one - eight
        ts += eight - full
        print(f"{i:<14}{one:>17,}{eight:>18,}{full:>15,}"
              f"{one-eight:>+14,}{eight-full:>+14,}", flush=True)
    print(f"{'TOTAL':<14}{'':>17}{'':>18}{'':>15}{tc:>+14,}{ts:>+14,}")
    print(f"\n  The seven extra constructions are worth nothing at all wherever the")
    print(f"  'from constr.' column reads 0 -- min-degree keeps rebuilding the same")
    print(f"  ordering there. That is why GRASP's randomised construction is a real")
    print(f"  change and Hill Climbing's eight are not.")


N = {}


def main():
    ap = argparse.ArgumentParser(description="Regenerate the results sheet.")
    ap.add_argument("--block", type=int, choices=[1, 2, 3, 4, 5], default=0)
    a = ap.parse_args()

    for r in csv.DictReader(open(os.path.join(HERE, "benchmark-hill_climbing.csv"))):
        N[r["instance"]] = int(r["n"])

    names = [k for k, _ in LEADERBOARD + STUDY]
    M = {k: load(k)[0] for k in names}
    missing = [k for k in names if not M[k]]
    if missing:
        raise SystemExit(f"missing benchmark CSVs for: {', '.join(missing)}")

    # Controlled-experiment arms that are not also Block 1 columns. Tolerated
    # if absent so the report still runs while an arm is re-running.
    for k, *_ in ARMS:
        if k not in M:
            M[k] = load(k)[0]
    floor = noise_floor([k for k, _ in STUDY])

    if a.block in (0, 1):
        block1(M, floor)
    if a.block in (0, 2):
        block2(M, floor)
    if a.block in (0, 3):
        block3()
    if a.block in (0, 4):
        block4()
    if a.block == 5:
        block5(M)


if __name__ == "__main__":
    main()

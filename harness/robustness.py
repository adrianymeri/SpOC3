#!/usr/bin/env python3
"""
robustness.py -- does each verdict survive a change of statistic?

The results sheet names a best method on every row from the best of three
seeds. This script asks what the three seeds themselves support, using each
method's OWN spread instead of one floor borrowed from Hill Climbing:

  A  per row: best-of-3 winner, median winner, and whether every seed of the
     winner beats every seed of every rival (complete separation -- with
     three seeds a side, the exact one-sided rank-test p is 1/20 = 0.05).
     A row is RESOLVED when the two winners agree and the seeds separate.
  B  is there a session effect? Hill Climbing ran twice; reshuffle its six
     seeds per instance between the two sessions and see how often the
     ten-instance total moves as far as it did.
  C  the operator arms at the seed level, against both control samples,
     and how many of their runs overran the wall-clock budget.
  D  the budget sweep: what 10x the budget buys each search arm, set
     against the construction's lead at every budget.
  E  where the no-search construction would sit on the ESA leaderboards.

Every number the v4 workbook adds is printed here. Standard library only.

    python3 harness/robustness.py
"""

from __future__ import annotations

import csv
import glob
import os
import random
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INSTANCES = ["small-graph", "synth-1", "synth-2", "synth-3", "medium-graph",
             "synth-4", "synth-5", "large-graph", "synth-6", "synth-7"]
DENSE = {"large-graph", "synth-6", "synth-7"}
METHODS = [("fcmaes", "benchmark-fast_cma_es.csv"),
           ("Team HRI", "benchmark-hri.csv"),
           ("Spacekangaroos", "benchmark-spacekangaroos.csv"),
           ("Hill Climbing", "benchmark-hill_climbing.csv"),
           ("min-degree x8", "benchmark-min_degree-x8.csv")]


def seeds(filename, valid_only=True):
    """{instance: [scores]} -- every seed, not just the best."""
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, filename))):
        if valid_only and str(r.get("valid", "True")).lower() != "true":
            continue
        out.setdefault(r["instance"], []).append(int(r["score"]))
    return out


def fmt(x):
    return f"{x:,}"


def block_a():
    print("A  Per-row verdicts from each method's own seeds")
    print("=" * 118)
    S = {m: seeds(f) for m, f in METHODS}
    print(f"{'instance':<14}{'best-of-3 winner':<18}{'median winner':<17}"
          f"{'margin':>9}{'seed gap':>10}{'verdict':>12}   floor rule (v3)")
    resolved = 0
    for i in INSTANCES:
        have = {m: S[m][i] for m in S if S[m].get(i)}
        best = sorted(have, key=lambda m: min(have[m]))
        med = sorted(have, key=lambda m: statistics.median(have[m]))
        w, r2 = best[0], best[1]
        margin = min(have[r2]) - min(have[w])
        # winner's WORST seed against the best seed of any rival
        gap = min(min(have[m]) for m in have if m != w) - max(have[w])
        ok = (w == med[0]) and gap > 0
        resolved += ok
        old = "separates" if margin >= 7881 else "tie"
        print(f"{i:<14}{w:<18}{med[0]:<17}{fmt(margin):>9}{fmt(gap):>10}"
              f"{'resolved' if ok else 'unresolved':>12}   {old}")
    print(f"\n  {resolved} of 10 rows resolved. 'seed gap' > 0 means every seed of the")
    print("  winner beats every seed of every rival; min-degree x8 is one value.")
    print("  Seed spreads (worst - best) that the single 7,881 HV floor ignores:")
    for m, _ in METHODS[:4]:
        sp = {i: max(v) - min(v) for i, v in S[m].items() if len(v) == 3}
        top = max(sp, key=sp.get)
        print(f"    {m:<16} median {fmt(int(statistics.median(sp.values()))):>7} HV,"
              f" largest {fmt(sp[top]):>7} HV ({top})")


def block_b(n_perm=100_000):
    print("\nB  Session effect: Hill Climbing run twice, seeds reshuffled")
    print("=" * 118)
    v1 = seeds("benchmark-hill_climbing.csv")
    v2 = seeds("benchmark-hill_climbing-v2.csv")
    obs = sum(min(v1[i]) - min(v2[i]) for i in INSTANCES)
    per = {i: abs(min(v1[i]) - min(v2[i])) for i in INSTANCES}
    worst = max(per.values())
    rng = random.Random(0)
    hits = hits_worst = 0
    for _ in range(n_perm):
        tot = 0
        top = 0
        for i in INSTANCES:
            pool = v1[i] + v2[i]
            rng.shuffle(pool)
            d = min(pool[:3]) - min(pool[3:])
            tot += d
            top = max(top, abs(d))
        hits += abs(tot) >= abs(obs)
        hits_worst += top >= worst
    print(f"  ten-instance total moved by {fmt(obs)} HV between sessions")
    print(f"  share of {fmt(n_perm)} reshuffles that move it as far: {hits / n_perm:.3f}")
    print(f"  worst single instance moved by {fmt(worst)} HV; share of reshuffles in which")
    print(f"  some instance moves at least that far: {hits_worst / n_perm:.3f}")
    print("  per instance, |session change| against each session's own seed spread:")
    for i in INSTANCES:
        print(f"    {i:<14}{fmt(per[i]):>7}   spreads {fmt(max(v1[i]) - min(v1[i])):>7}"
              f" / {fmt(max(v2[i]) - min(v2[i])):>6}")


def block_c():
    print("\nC  Operator arms at the seed level, against both control samples")
    print("=" * 118)
    v1 = seeds("benchmark-hill_climbing.csv")
    v2 = seeds("benchmark-hill_climbing-v2.csv")
    over = total = 0
    for path in sorted(glob.glob(os.path.join(HERE, "ablation_more-*.csv"))):
        rows = list(csv.DictReader(open(path)))
        total += len(rows)
        over += sum(float(r["seconds"]) > 1400 for r in rows)
        arm = {}
        for r in rows:
            arm.setdefault(r["instance"], []).append(int(r["score"]))
        win = sum(max(arm[i]) < min(v1[i] + v2[i]) for i in INSTANCES)
        loss = sum(min(arm[i]) > max(v1[i] + v2[i]) for i in INSTANCES)
        print(f"  {os.path.basename(path)[14:-4]:<20} wins {win}   losses {loss}"
              f"   (all 3 seeds against all 6 control seeds)")
    print(f"  runs over 1,400 s wall-clock: {over} of {total}; a stall costs a run")
    print("  search time, so it biases against the arm that suffers it.")


def block_d():
    print("\nD  Budget sweep: what 10x the budget buys")
    print("=" * 118)
    best = {}
    for r in csv.DictReader(open(os.path.join(HERE, "budget_sweep.csv"))):
        k = (r["solver"], int(r["budget"]), r["instance"])
        best[k] = min(best.get(k, 0), int(r["score"]))
    arms = ["grasp_front", "hill_climbing", "hc_bottleneck", "all"]
    tot = {(a, b): sum(best[(a, b, i)] for i in INSTANCES)
           for a in arms for b in (1200, 3600, 12000)}
    g = [tot[("grasp_front", b)] for b in (1200, 3600, 12000)]
    ref = round(statistics.mean(g))
    print(f"  construction (grasp_front) totals {', '.join(fmt(x) for x in g)};"
          f" range {fmt(max(g) - min(g))} HV")
    print("  -- it draws different constructions at each budget, and with one draw per")
    print("  budget that movement cannot be read as a trend.")
    print(f"  Mean {fmt(ref)} is used as the fixed reference.\n")
    print(f"  {'arm':<16}{'gain over 10x':>15}{'lead at 1,200 s':>18}"
          f"{'lead at 12,000 s':>19}{'lead vs mean':>15}{'gain / lead':>13}")
    for a in arms[1:]:
        gain = tot[(a, 1200)] - tot[(a, 12000)]
        l1 = tot[(a, 1200)] - g[0]
        l3 = tot[(a, 12000)] - g[2]
        lm = tot[(a, 12000)] - ref
        print(f"  {a:<16}{gain:>+15,}{fmt(l1):>18}{fmt(l3):>19}{fmt(lm):>15}"
              f"{gain / lm:>12.0%}")


def block_e():
    path = os.path.join(ROOT, "leaderboard-references", "esa_leaderboard.csv")
    if not os.path.exists(path):
        return
    print("\nE  Where the no-search construction would sit on ESA's leaderboards")
    print("=" * 118)
    md = seeds("benchmark-min_degree-x8.csv")
    board = {}
    for r in csv.DictReader(open(path)):
        if int(r["score"]) < 0:                 # scored entries only
            board.setdefault(r["problem"], []).append(int(r["score"]))
    for p in ("small-graph", "medium-graph", "large-graph"):
        s = min(md[p])
        better = sum(x < s for x in board[p])
        ties = sum(x == s for x in board[p])
        print(f"  {p:<14} min-degree x8 {fmt(s):>11}   place {better + 1}"
              f"{' (tied)' if ties else ''} of {len(board[p]) + 1}")


if __name__ == "__main__":
    block_a()
    block_b()
    block_c()
    block_d()
    block_e()

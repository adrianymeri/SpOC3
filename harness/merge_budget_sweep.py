#!/usr/bin/env python3
"""
merge_budget_sweep.py -- read the budget sweep and answer the question.

run_budget_sweep.sh writes one CSV per (solver, budget, instance) so the
jobs can run in parallel and resume independently. This collects them into
one tidy table and prints the comparison the sweep exists to make.

THE QUESTION
------------
The Limits note says: "A much longer run is the obvious way for search to
catch up, and nothing here rules it out." The controlled experiment put
construction at +341,263 HV against the baseline while the three search
levers reached +9,562 to +42,468, all at a single 1,200 s budget. So: does
the gap close when search is given more time?

WHAT TO LOOK AT
---------------
For each budget, the table reports every solver's median score and its
difference from hill_climbing at the SAME budget. Those differences are
what matters, not the raw scores, because raw scores improve with budget
for everyone.

  - if the search levers' advantage stays flat as budget grows, more time
    does not help search and the paper's conclusion holds at 10x budget
  - if construction's advantage shrinks as budget grows, search is
    catching up and the conclusion is budget-dependent
  - if nothing moves beyond the noise floor, the sweep is inconclusive and
    says so

Differences are judged against the BETWEEN-SESSION floors the sheet uses --
21,519 HV aggregate, 7,881 HV per instance -- not against the seed spread.
The Limits note is explicit that within-run seed spread understates real
uncertainty by about fourteen times, so using it here would manufacture
significance.

    python3 merge_budget_sweep.py
    python3 merge_budget_sweep.py --csv budget_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FLOOR_PER_INSTANCE = 7881        # between-session, from the sheet
FLOOR_AGGREGATE = 21519


def load(pattern):
    """Read every per-job CSV. Both runners' shapes share these columns."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)[:-4]
        # <solver>-b<budget>-<instance>
        try:
            solver, rest = base.split("-b", 1)
            budget, instance = rest.split("-", 1)
            budget = int(budget)
        except ValueError:
            print(f"  skipping unparseable name: {base}", file=sys.stderr)
            continue
        for r in csv.DictReader(open(path)):
            if not r.get("score"):
                continue
            rows.append({
                "solver": solver, "budget": budget,
                "instance": r["instance"], "n": int(r["n"]),
                "seed": int(r["seed"]), "score": int(r["score"]),
                "seconds": float(r.get("seconds") or 0),
                "valid": r.get("valid", ""),
                "evaluations": int(r["evaluations"]) if r.get("evaluations")
                else "",
                "accepts": int(r["accepts"]) if r.get("accepts") else "",
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Collect the budget sweep.")
    ap.add_argument("--dir", default=os.path.join(HERE, "budget"))
    ap.add_argument("--csv", default=os.path.join(HERE, "budget_sweep.csv"))
    ap.add_argument("--baseline", default="hill_climbing",
                    help="solver every other one is compared against")
    a = ap.parse_args()

    rows = load(os.path.join(a.dir, "*.csv"))
    if not rows:
        raise SystemExit(f"no CSVs found in {a.dir}")

    bad = [r for r in rows if r["valid"] not in ("True", "", None)]
    budgets = sorted({r["budget"] for r in rows})
    solvers = sorted({r["solver"] for r in rows})
    instances = sorted({r["instance"] for r in rows})

    print(f"{len(rows)} runs: {len(solvers)} solvers x {len(budgets)} "
          f"budgets x {len(instances)} instances")
    print(f"budgets: {budgets}")
    if bad:
        print(f"  !! {len(bad)} runs are INVALID -- investigate before using")
    missing = (len(solvers) * len(budgets) * len(instances) * 3) - len(rows)
    if missing > 0:
        print(f"  {missing} runs still missing (sweep incomplete)")

    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.csv}\n")

    # total score per (solver, budget, seed), summed over instances, then
    # median over seeds -- the sheet's aggregate view
    def agg(solver, budget):
        per_seed = {}
        for r in rows:
            if r["solver"] == solver and r["budget"] == budget:
                per_seed.setdefault(r["seed"], []).append(r["score"])
        full = [sum(v) for v in per_seed.values()
                if len(v) == len(instances)]
        return statistics.median(full) if full else None

    print("AGGREGATE over all instances, median of seeds")
    print("difference is vs "
          f"{a.baseline} at the SAME budget; more negative score is better")
    head = f"  {'solver':<16}"
    for b in budgets:
        head += f"{b:>14}s"
    print(head)
    print("  " + "-" * (16 + 15 * len(budgets)))
    base = {b: agg(a.baseline, b) for b in budgets}
    for s in solvers:
        line = f"  {s:<16}"
        for b in budgets:
            v = agg(s, b)
            line += f"{v:>15,.0f}" if v is not None else f"{'--':>15}"
        print(line)
    print()
    print(f"  {'vs ' + a.baseline:<16}")
    for s in solvers:
        if s == a.baseline:
            continue
        line = f"  {s:<16}"
        for b in budgets:
            v, bv = agg(s, b), base[b]
            if v is None or bv is None:
                line += f"{'--':>15}"
            else:
                d = bv - v                     # positive = s is better
                mark = "" if abs(d) > FLOOR_AGGREGATE else "~"
                line += f"{d:>+14,.0f}{mark}"
        print(line)
    print(f"\n  ~ marks a difference inside the {FLOOR_AGGREGATE:,} HV "
          f"between-session aggregate floor,\n    i.e. not distinguishable "
          f"from run-to-run drift.")

    # does the advantage grow with budget?
    if len(budgets) >= 2:
        print("\nTREND -- advantage at the largest budget minus at the "
              "smallest")
        lo, hi = budgets[0], budgets[-1]
        for s in solvers:
            if s == a.baseline:
                continue
            v_lo, v_hi = agg(s, lo), agg(s, hi)
            b_lo, b_hi = base[lo], base[hi]
            if None in (v_lo, v_hi, b_lo, b_hi):
                continue
            d_lo, d_hi = b_lo - v_lo, b_hi - v_hi
            change = d_hi - d_lo
            verdict = ("grew" if change > FLOOR_AGGREGATE else
                       "shrank" if change < -FLOOR_AGGREGATE else
                       "unchanged within the floor")
            print(f"  {s:<16}{d_lo:>+12,.0f} -> {d_hi:>+12,.0f}   "
                  f"({change:>+11,.0f}, {verdict})")
        print(f"\n  'unchanged within the floor' at {hi}s vs {lo}s means "
              f"{hi // lo}x the budget\n  did not change that lever's "
              f"standing -- which is the answer the Limits\n  note asks for.")

    # evaluations actually reached, as a sanity check on the budget
    ev = [r for r in rows if r["evaluations"] != ""]
    if ev:
        print("\nEVALUATIONS reached (the 'all' arm), median per instance")
        print(f"  {'instance':<14}" + "".join(f"{b:>12}s" for b in budgets))
        for inst in instances:
            line = f"  {inst:<14}"
            for b in budgets:
                vals = [r["evaluations"] for r in ev
                        if r["instance"] == inst and r["budget"] == b]
                line += (f"{statistics.median(vals):>13,.0f}" if vals
                         else f"{'--':>13}")
            print(line)
        print("  these should scale roughly with budget; if they do not, the "
              "budget\n  is being spent on construction rather than search.")


if __name__ == "__main__":
    main()

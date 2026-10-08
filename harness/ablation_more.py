#!/usr/bin/env python3
"""
ablation_more.py -- ablation.py, extended to the operators in operators_more.

ablation.py builds its --operator choices from Operators.NAMES at import
time, and never binds a graph, so it cannot see or run the added moves.
This is the same runner with two lines more: install the new operators
first, and bind the graph before each run so the structure-aware moves can
see it. ablation.py itself is untouched, and so are its CSVs.

Everything else is deliberately identical to ablation.py and bench_one.py:
the same ten instances, the same seeds, the same one-core pinning check,
the same re-scoring of the submitted vectors through esa_eval, the same
resume-on-restart, the same column layout. That is the point -- the new
operators have to be measured on the same compute, the same time and the
same resources as the four already in the sheet, or the columns cannot be
put side by side.

    python3 ablation_more.py --operator three_opt_aimed --seconds 1200 --seeds 3
    python3 ablation_more.py --operator all   --seconds 1200 --seeds 3
    python3 ablation_more.py --list

`--operator all` draws from every operator, original and new, with
per-operator accept counts recorded: the attribution question, which one
earns the improvements when they all compete. A single --operator name is
the ablation question: what that move achieves on its own.

Output goes to ablation_more-<operator>.csv, so nothing overwrites the
existing ablation-*.csv files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from bench_one import INSTANCES, THREAD_VARS, ROOT, score_vectors   # noqa: E402
from torso import Graph                                             # noqa: E402
import hill_climbing                                                # noqa: E402
import operators_more                                               # noqa: E402
import fast_degrees                                                  # noqa: E402

# Must happen before OPS is read, so the new names are selectable.
ADDED = operators_more.install()
OPS = list(hill_climbing.Operators.NAMES)
FIELDS = (["instance", "n", "operator", "family", "seed", "score", "seconds",
           "valid", "evaluations", "accepts"]
          + [f"acc_{nm}" for nm in OPS]
          + [f"try_{nm}" for nm in OPS])


def main():
    ap = argparse.ArgumentParser(
        description="Hill climbing with one operator enabled, extended pool.")
    ap.add_argument("--operator", default="",
                    help="one of --list, or 'all' for the competing pool")
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--list", action="store_true",
                    help="print the selectable operators and exit")
    ap.add_argument("--fast", action="store_true",
                    help="use the delta evaluator (exact, same deg[]); "
                         "reaches more candidates in the same budget")
    ap.add_argument("--speed", action="store_true",
                    help="measure the delta evaluator instead of running an "
                         "ablation: same operator, seed and wall-clock with "
                         "and without it, so the ratio of candidates reached "
                         "IS the speedup")
    ap.add_argument("--allow-threads", action="store_true",
                    help="skip the one-core check (results not comparable)")
    a = ap.parse_args()

    if a.list:
        print(f"{len(OPS)} operators ({len(OPS) - len(ADDED)} original, "
              f"{len(ADDED)} added):\n")
        for nm in OPS:
            fam = operators_more.FAMILY.get(nm, "original")
            print(f"  {nm:<20}{fam}")
        print("\n  all                 every operator above, competing")
        return

    if not a.operator:
        raise SystemExit("--operator is required (or --list)")
    if a.operator != "all" and a.operator not in OPS:
        raise SystemExit(f"unknown operator {a.operator!r}; see --list")

    unset = [v for v in THREAD_VARS if os.environ.get(v) != "1"]
    if unset and not a.allow_threads:
        raise SystemExit(
            "refusing to run: these are not pinned to 1 -> "
            + ", ".join(unset)
            + "\nLaunch via run_mac.sh, or pass --allow-threads if you know "
              "the numbers will not be compared against other solvers.")

    ops = None if a.operator == "all" else [a.operator]

    if a.speed:
        print(f"[speed] {a.operator}: {a.seconds:.0f}s per arm per instance, "
              f"seed 1, exact vs delta\n")
        print(f"  {'instance':<14}{'n':>7}{'exact':>12}{'delta':>12}"
              f"{'speedup':>10}")
        ratios = []
        for name in INSTANCES:
            path = os.path.join(ROOT, a.data_dir, f"{name}.gr")
            if not os.path.exists(path):
                continue
            graph = Graph.load(path)
            got = {}
            for fast in (False, True):
                operators_more.bind(graph)
                fast_degrees.reset()
                if fast:
                    fast_degrees.install()
                else:
                    fast_degrees.uninstall()
                _, cl = hill_climbing.solve(graph, a.seconds, seed=1,
                                            start="min_degree", ops=ops)
                got[fast] = cl.evaluations
            fast_degrees.uninstall()
            r = got[True] / got[False] if got[False] else float("nan")
            ratios.append(r)
            print(f"  {name:<14}{graph.n:>7}{got[False]:>12,}"
                  f"{got[True]:>12,}{r:>9.2f}x")
        if ratios:
            import math
            gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            print(f"\n  geometric mean over {len(ratios)} instances: "
                  f"{gm:.2f}x")
            print("  (both arms identical except the evaluator; deg[] is "
                  "bit-exact either way)")
        return

    if a.fast:
        fast_degrees.install()
        print("[fast] delta evaluator installed (exact)", flush=True)
    family = ("all" if a.operator == "all"
              else operators_more.FAMILY.get(a.operator, "original"))
    out = a.out or os.path.join(HERE, f"ablation_more-{a.operator}.csv")

    done, kept = set(), []
    if os.path.exists(out):
        with open(out) as f:
            for r in csv.DictReader(f):
                key = (r["instance"], int(r["seed"]))
                if key not in done:
                    done.add(key)
                    kept.append(r)

    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in kept:
        w.writerow({k: r.get(k, "") for k in FIELDS})
    f.flush()

    seeds = list(range(1, a.seeds + 1))
    todo = sum(1 for i in INSTANCES for s in seeds if (i, s) not in done)
    print(f"[{a.operator}] {todo} runs to do "
          f"({len(done)} already in {os.path.basename(out)}), "
          f"{a.seconds:.0f}s each, 1 core, {len(OPS)}-operator pool",
          flush=True)

    for name in INSTANCES:
        path = os.path.join(ROOT, a.data_dir, f"{name}.gr")
        if not os.path.exists(path):
            print(f"[{a.operator}] MISSING {path}", flush=True)
            continue
        graph = Graph.load(path)
        # The structure-aware moves need the graph; rebinding also clears the
        # per-perm degree memo between instances.
        operators_more.bind(graph)
        fast_degrees.reset()
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            front, climber = hill_climbing.solve(
                graph, a.seconds, seed=seed, start="min_degree", ops=ops)
            ok, score = score_vectors(graph, front.decision_vectors())
            elapsed = time.time() - t0

            row = {"instance": name, "n": graph.n, "operator": a.operator,
                   "family": family, "seed": seed, "score": score,
                   "seconds": round(elapsed, 1), "valid": ok,
                   "evaluations": climber.evaluations,
                   "accepts": climber.accepts}
            for nm in OPS:
                row[f"acc_{nm}"] = climber.accepts_by_op.get(nm, 0)
                row[f"try_{nm}"] = climber.tries_by_op.get(nm, 0)
            w.writerow(row)
            f.flush()

            rate = climber.accepts / max(1, climber.evaluations)
            print(f"[{a.operator}] {name} seed {seed}: {score:>14,} "
                  f"({elapsed/60:.1f} min, {climber.evaluations:,} evals, "
                  f"{climber.accepts:,} acc = {rate:.3%})"
                  f"{'' if ok else '  INVALID'}", flush=True)

    f.close()
    if a.fast:
        print(f"[{a.operator}] {fast_degrees.report()}", flush=True)
    print(f"[{a.operator}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()

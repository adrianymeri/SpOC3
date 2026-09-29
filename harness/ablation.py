#!/usr/bin/env python3
"""
ablation.py -- run hill climbing with ONE operator enabled, to find out which
of the four is actually doing the work.

hill_climbing.py draws uniformly from four moves: swap_neighbours, swap_any,
move_vertex, reverse_segment. The benchmark tells us what the four achieve
together; it says nothing about which of them earns the improvements. This
runs the same climber with the pool restricted to a single operator, so the
columns can be compared directly against each other and against the
four-operator result already in benchmark-hill_climbing.csv.

    python3 ablation.py --operator swap_any --seconds 1200 --seeds 3
    python3 ablation.py --operator all      --seconds 1200 --seeds 3

`--operator all` is the control: the normal four-operator climber, but with
per-operator accept counts recorded. That answers the complementary question
-- when all four compete, which one produces the accepted moves.

Everything else matches bench_one.py: same instances, same seeds, same
one-core pinning, same re-scoring through esa_eval.py, same resume-on-restart
behaviour. Output goes to ablation-<operator>.csv so the benchmark CSVs and
merge_results.py are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from bench_one import INSTANCES, THREAD_VARS, ROOT, score_vectors   # noqa: E402
from torso import Graph                                             # noqa: E402
import hill_climbing                                                # noqa: E402

OPS = hill_climbing.Operators.NAMES
FIELDS = (["instance", "n", "operator", "seed", "score", "seconds", "valid",
           "evaluations", "accepts"]
          + [f"acc_{nm}" for nm in OPS]
          + [f"try_{nm}" for nm in OPS])


def main():
    ap = argparse.ArgumentParser(
        description="Hill climbing with one operator enabled.")
    ap.add_argument("--operator", required=True, choices=OPS + ["all"])
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--allow-threads", action="store_true",
                    help="skip the one-core check (results not comparable)")
    a = ap.parse_args()

    unset = [v for v in THREAD_VARS if os.environ.get(v) != "1"]
    if unset and not a.allow_threads:
        raise SystemExit(
            "refusing to run: these are not pinned to 1 -> "
            + ", ".join(unset)
            + "\nExport the thread limits first, or pass --allow-threads if "
              "you know the numbers will not be compared against others.")

    ops = None if a.operator == "all" else [a.operator]
    out = a.out or os.path.join(HERE, f"ablation-{a.operator}.csv")

    # Resume: keep whatever is already there, skip those (instance, seed) pairs.
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
          f"{a.seconds:.0f}s each, 1 core, instances from {a.data_dir}",
          flush=True)

    for name in INSTANCES:
        path = os.path.join(ROOT, a.data_dir, f"{name}.gr")
        if not os.path.exists(path):
            print(f"[{a.operator}] MISSING {path}", flush=True)
            continue
        graph = Graph.load(path)
        for seed in seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            front, climber = hill_climbing.solve(
                graph, a.seconds, seed=seed, start="min_degree", ops=ops)
            ok, score = score_vectors(graph, front.decision_vectors())
            elapsed = time.time() - t0

            row = {"instance": name, "n": graph.n, "operator": a.operator,
                   "seed": seed, "score": score,
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
    print(f"[{a.operator}] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()

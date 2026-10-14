#!/usr/bin/env python3
"""
delta_check.py -- is fast_degrees.step_degrees bit-identical to esa_eval's?

The delta evaluator re-walks only the window a move changed (fast_degrees.py).
This checks it the way a climb uses it: from a min-degree ordering, draw a
candidate with a random one of the seventeen operators, compare the delta
evaluator's deg[] with a full esa_eval walk of the same candidate, and now and
then accept the candidate, so that consecutive calls are chained the way they
are inside hill_climbing.py. Every instance, both code paths (windowed replay
and full-walk fallback).

    python3 harness/delta_check.py [--candidates 60] [--instances ...]
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from esa_eval import step_degrees as reference                      # noqa: E402
from torso import Graph                                             # noqa: E402
import hill_climbing as hc                                          # noqa: E402
import operators_more                                               # noqa: E402
import fast_degrees                                                 # noqa: E402
from bench_one import INSTANCES                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", type=int, default=60)
    ap.add_argument("--accept", type=float, default=0.3)
    ap.add_argument("--instances", nargs="*", default=INSTANCES)
    a = ap.parse_args()

    operators_more.install()
    ops = list(zip(hc.Operators.NAMES, hc.Operators.ALL))
    print(f"[delta] {len(ops)} operators, {a.candidates} candidates per instance, "
          f"accept p = {a.accept}\n")
    print(f"  {'instance':<14}{'n':>6}{'identical':>12}{'windowed':>10}{'full':>6}"
          f"{'ops used':>10}{'s':>6}")
    total = same_all = 0
    for name in a.instances:
        t0 = time.time()
        graph = Graph.load(os.path.join(ROOT, "data", f"{name}.gr"))
        operators_more.bind(graph)
        rng = random.Random(7)
        n, bits = graph.n, graph.bits
        current = hc.Starts.min_degree(graph, random.Random(1))
        fast_degrees.reset()
        for k in fast_degrees.stats:
            fast_degrees.stats[k] = 0
        same, used = 0, set()
        for _ in range(a.candidates):
            nm, op = rng.choice(ops)
            used.add(nm)
            cand = op(current, rng)
            if fast_degrees.step_degrees(cand, bits, n) == reference(cand, bits, n):
                same += 1
            else:
                print(f"    MISMATCH on {name} after {nm}")
            if rng.random() < a.accept:
                current = cand
        total += a.candidates
        same_all += same
        st = fast_degrees.stats
        print(f"  {name:<14}{n:>6}{f'{same}/{a.candidates}':>12}{st['window']:>10}"
              f"{st['full']:>6}{len(used):>10}{time.time() - t0:>6.0f}", flush=True)
    print(f"\n  identical deg[] on {same_all} of {total} candidates")
    sys.exit(0 if same_all == total else 1)


if __name__ == "__main__":
    main()

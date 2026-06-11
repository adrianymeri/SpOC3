#!/usr/bin/env python3
"""
bench_extra.py -- run a subset of hc* variants on the 20 generated
extra instances and produce a CSV/Markdown summary.

The hc* variants have slightly different run() signatures; we adapt
them all to a single (problem, budget_s, seed, here) call by passing
sensible defaults for the extra keyword arguments.

Stdout from each hc run is suppressed so the benchmark output stays
readable.  The score is read back from the saved submission JSON
via verify_submission's load helper.
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import contextlib
import csv
import importlib
import io
import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

from core import (
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
)


# Map algo short name -> (module name, callable invoker)
def _hc_simple_or_archive(mod, problem, budget, seed, here):
    """For hc2_simple / hc3_archive: run(problem, budget, seed, progress_every, here)."""
    return mod.run(problem, budget, seed, 10 ** 18, here)

def _hcN(mod, problem, budget, seed, here):
    """For hc4-hc14 (except hc1 initial): adds num_t_seeds=20."""
    return mod.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20)

def _hc_initial(mod, problem, budget, seed, here):
    """For hc1_initial: takes interval_min / interval_max / show_final."""
    return mod.run(problem, budget, seed, here,
                   interval_min=1.0, interval_max=3.0, show_final=False)


ALGO_REGISTRY: Dict[str, Tuple[str, Callable]] = {
    "hc1":  ("hc1_initial",      _hc_initial),
    "hc2":  ("hc2_simple",       _hc_simple_or_archive),
    "hc3":  ("hc3_archive",      _hc_simple_or_archive),
    "hc4":  ("hc4_warm_start",   _hcN),
    "hc5":  ("hc5_operators",    _hcN),
    "hc6":  ("hc6_gap_fill",     _hcN),
    "hc7":  ("hc7_kbottleneck",  _hcN),
    "hc8":  ("hc8_lahc",         _hcN),
    "hc9":  ("hc9_hv_accept",    _hcN),
    "hc10": ("hc10_torso_warm",  _hcN),
    "hc11": ("hc11_ils",         _hcN),
    "hc12": ("hc12_incremental", _hcN),
    "hc13": ("hc13_tabu",        _hcN),
    "hc14": ("hc14_steepest",    _hcN),
    "hc15": ("hc15_hv_incremental", _hcN),
}


def submission_score(sub_path: str, gr_path: str) -> float:
    """Read a submission JSON and recompute its score from scratch."""
    with open(sub_path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        entry = payload[0]
    else:
        entry = payload
    dvs = entry["decisionVector"]
    n, adj = load_graph(gr_path)
    ab = build_adj_bitsets(n, adj)
    fits = []
    for dv in dvs:
        perm = dv[:-1]
        t = int(dv[-1])
        fits.append(evaluate(perm, t, ab, n))
    return -hypervolume_2d(fits, n)


def run_one(algo: str, problem: str, budget: float, seed: int,
            here: str, verbose: bool = False) -> Tuple[float, float]:
    """Run one (algo, problem) pair, return (score, elapsed_s)."""
    mod_name, invoker = ALGO_REGISTRY[algo]
    # Algorithms live under the algorithms/ subpackage after the restructure.
    try:
        mod = importlib.import_module(f"algorithms.hill_climbing.{mod_name}")
    except ModuleNotFoundError:
        # Fall back to old-style flat import (in case the layout reverts).
        mod = importlib.import_module(mod_name)
    sub_path = os.path.join(here, "submissions", problem, f"{algo}.json")
    if os.path.exists(sub_path):
        os.remove(sub_path)
    t0 = time.time()
    out = io.StringIO()
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(None if verbose else out), \
             contextlib.redirect_stderr(None if verbose else err):
            invoker(mod, problem, budget, seed, here)
    except Exception as exc:
        return float("nan"), time.time() - t0
    elapsed = time.time() - t0
    if not os.path.exists(sub_path):
        return float("nan"), elapsed
    gr = graph_path(here, problem)
    return submission_score(sub_path, gr), elapsed


def list_instances(here: str) -> List[str]:
    data_dir = os.path.join(here, "data")
    names = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith(".gr") and f.startswith("inst_"):
            names.append(f[:-3])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", type=str, default="hc4,hc9",
                    help="comma-separated list of algos to run "
                         "(default: hc4,hc9)")
    ap.add_argument("--instances", type=str, default=None,
                    help="comma-separated subset of instance names; default = all")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="wall-time budget per run, seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="extra_instances/results.csv")
    ap.add_argument("--append", action="store_true",
                    help="append to existing results.csv instead of overwriting")
    args = ap.parse_args()

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "extra_instances")
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    instances = list_instances(here)
    if args.instances:
        wanted = {x.strip() for x in args.instances.split(",")}
        instances = [n for n in instances if n in wanted]

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = (not args.append) or (not os.path.exists(out_path))
    f = open(out_path, mode, newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["instance", "algo", "score", "elapsed_s",
                    "budget_s", "seed"])

    print(f"Running {len(algos)} algos × {len(instances)} instances "
          f"= {len(algos) * len(instances)} runs at budget {args.budget}s "
          f"each, seed {args.seed}")
    print()
    total = len(algos) * len(instances)
    done = 0
    t_start = time.time()
    for inst in instances:
        for algo in algos:
            done += 1
            score, elapsed = run_one(algo, inst, args.budget,
                                     args.seed, here)
            w.writerow([inst, algo, f"{score:.0f}", f"{elapsed:.2f}",
                        args.budget, args.seed])
            f.flush()
            print(f"  [{done:>3}/{total}] {inst:<22} {algo:<5}  "
                  f"score = {score:>14,.0f}  ({elapsed:5.1f}s wall)")
    total_elapsed = time.time() - t_start
    f.close()
    print(f"\nWrote {out_path}, total {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

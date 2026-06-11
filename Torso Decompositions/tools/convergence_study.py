#!/usr/bin/env python3
"""
convergence_study.py -- per-algorithm anytime convergence to the limit.

The question this answers
-------------------------
"Given unlimited runtime, is each algorithm producing its best result?"  For
stochastic search that is not a yes/no you can assert -- it is a limit you must
*show* you have reached.  This driver does exactly that: for each (algorithm,
instance) it runs the method at a GEOMETRIC LADDER of wall budgets
(25, 50, 100, 200, ... s), several seeds per rung, and measures how the score
improves as compute grows.  When the marginal gain from the last doubling of
budget falls below a threshold, the curve has flattened and that method is
CONVERGED -- its best result, at the tuned hyperparameters, is in hand.  If the
last rung is still improving by more than the threshold, it is RISING and the
ladder should be extended (the driver tells you so).

This is the rigorous, publishable form of the green light: a method is "at its
limit" iff its anytime HV curve is flat AND (separately) its seed-union curve
is flat (the latter is what `tools/ceiling_validation.py` probes).  Run both.

How each family is run
----------------------
  - Metaheuristics (grasp, sa, vns, nsga2, sms): in-process via
    `tools.tune.run_cell` at the TUNED config read from
    `extra_instances/tuning_<tech>.csv` (the recorded grid winner for that
    instance).  Returns the submitted top-20 score directly.
  - HC variants (hc9, hc13, hc7, hc15): invoked through their CLI, then the
    written submission is re-scored end-to-end and the canonical JSON is
    restored from a backup -- so the canonical HC submissions are never
    altered.

SAFE: metaheuristic cells write throwaway `<tech>_tune` stems (deleted after
each pair); HC cells back up and restore their canonical stem.  A full log is
appended PER PAIR to `extra_instances/convergence_study.csv`, so the run is
safe to interrupt and resume -- finished pairs are never recomputed if you
pass --resume.

Usage
-----
    # default ladder, contenders + HC frontier, 3 seeds, parallel
    python3 tools/convergence_study.py --workers 8

    # push the ladder further out, more seeds, GRASP only on large
    python3 tools/convergence_study.py --algos grasp --problems large-graph \
        --budgets 25,50,100,200,400,800,1600,3200 --seeds 1,2,3,4,5

    # print the plan + tuned configs that will be used, run nothing
    python3 tools/convergence_study.py --plan-only
"""

from __future__ import annotations

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import List, Tuple

from core import (  # noqa: E402
    MAX_TW, build_adj_bitsets, evaluate, graph_path,
    hypervolume_2d, load_graph, repo_root,
)
import tools.tune as tune  # noqa: E402

HERE = repo_root()
PY = sys.executable

# Metaheuristic families: in-process via run_cell at the tuned config.
META = ["grasp", "sa", "vns", "nsga2", "sms"]
# HC frontier variants: CLI + backup/restore of the canonical stem.
HC = {
    "hc9":  ("algorithms/hill_climbing/hc9_hv_accept.py", "hc9"),
    "hc13": ("algorithms/hill_climbing/hc13_tabu.py", "hc13"),
    "hc7":  ("algorithms/hill_climbing/hc7_kbottleneck.py", "hc7"),
    "hc15": ("algorithms/hill_climbing/hc15_hv_incremental.py", "hc15"),
}
ALL_ALGOS = META + list(HC)
ALL_PROBLEMS = ["small-graph", "medium-graph", "large-graph"]
DEFAULT_LADDER = [25, 50, 100, 200, 400, 800]
_SCORE_RE = re.compile(r"Official score:\s*(-?[\d,]+)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coerce(val: str, template):
    """Coerce a CSV string to the type of a grid template value."""
    if isinstance(template, bool):
        return val in ("True", "true", "1")
    if isinstance(template, int):
        return int(float(val))
    if isinstance(template, float):
        return float(val)
    return val


def tuned_params(tech: str, problem: str):
    """Recorded grid winner for (tech, problem) from tuning_<tech>.csv,
    coerced to the grid's value types.  None if no log row."""
    path = os.path.join(HERE, "extra_instances", f"tuning_{tech}.csv")
    if not os.path.exists(path):
        return None
    grid = tune.GRIDS[tech]["full"]
    keys = list(grid.keys())
    best = None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("problem") != problem:
                continue
            try:
                score = float(row["score"])
            except (KeyError, ValueError):
                continue
            if best is None or score < best[1]:
                params = {k: _coerce(row[k], grid[k][0]) for k in keys if k in row}
                best = (params, score)
    return best[0] if best else None


def _score_submission(path: str, adj_bits, n: int) -> float:
    """Re-score a written submission end-to-end (submitted top-20 HV)."""
    with open(path) as f:
        payload = json.load(f)
    entry = payload[0] if isinstance(payload, list) else payload
    feas: List[Tuple[int, int]] = []
    for dv in entry["decisionVector"]:
        if not isinstance(dv, list) or len(dv) != n + 1:
            continue
        perm = [int(x) for x in dv[:-1]]
        t = int(dv[-1])
        if sorted(perm) != list(range(n)):
            continue
        w, t_ret = evaluate(perm, t, adj_bits, n)
        if w <= MAX_TW:
            feas.append((int(w), int(t_ret)))
    return -hypervolume_2d(feas, n) if feas else 0.0


# ---------------------------------------------------------------------------
# one (algo, problem) worker: full ladder x seeds
# ---------------------------------------------------------------------------

def run_pair(algo: str, problem: str, budgets, seeds, epsilon: float):
    rows = []
    n, adj = load_graph(graph_path(HERE, problem))
    adj_bits = build_adj_bitsets(n, adj)

    is_hc = algo in HC
    backup = None
    canon_path = None
    params = None
    if is_hc:
        script, stem = HC[algo]
        canon_path = os.path.join(HERE, "submissions", problem, f"{stem}.json")
        if os.path.exists(canon_path):
            with open(canon_path) as f:
                backup = f.read()
    else:
        params = tuned_params(algo, problem)
        if params is None:
            return algo, problem, [], "NO-TUNED-CONFIG"

    try:
        for budget in budgets:
            for seed in seeds:
                if is_hc:
                    script, stem = HC[algo]
                    subprocess.run(
                        [PY, os.path.join(HERE, script), "--problem", problem,
                         "--budget", str(budget), "--seed", str(seed)],
                        cwd=HERE, capture_output=True, text=True)
                    score = _score_submission(canon_path, adj_bits, n)
                else:
                    score = tune.run_cell(algo, problem, budget, seed, HERE, params)
                rows.append({"algo": algo, "problem": problem, "budget": budget,
                             "seed": seed, "score": round(score, 3)})
    finally:
        # restore canonical HC stem; clean throwaway meta stems
        if is_hc and backup is not None:
            with open(canon_path, "w") as f:
                f.write(backup)
        if not is_hc:
            for st in tune._TUNE_STEMS:
                p = os.path.join(HERE, "submissions", problem, f"{st}.json")
                if os.path.exists(p):
                    os.remove(p)

    verdict = _verdict(rows, budgets, epsilon)
    return algo, problem, rows, verdict


def _verdict(rows, budgets, epsilon: float) -> str:
    """CONVERGED if the last budget-doubling improved the median score by less
    than epsilon (relative); else RISING."""
    by_budget = {}
    for r in rows:
        by_budget.setdefault(r["budget"], []).append(r["score"])
    meds = [(b, statistics.median(by_budget[b])) for b in budgets if b in by_budget]
    if len(meds) < 2:
        return "INSUFFICIENT"
    (_, s_prev), (_, s_last) = meds[-2], meds[-1]
    # more negative = better; improvement = how much score dropped
    rel = (s_prev - s_last) / abs(s_prev) if s_prev else 0.0
    return f"CONVERGED ({rel:+.4%})" if rel < epsilon else f"RISING ({rel:+.4%})"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", default=",".join(ALL_ALGOS))
    ap.add_argument("--problems", default=",".join(ALL_PROBLEMS))
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_LADDER)),
                    help="geometric wall-budget ladder in seconds")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--epsilon", type=float, default=0.001,
                    help="relative gain below which the curve is 'flat' (0.001 = 0.1%%)")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true",
                    help="skip (algo,problem) pairs already complete in the log")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bad = [a for a in algos if a not in ALL_ALGOS]
    if bad:
        ap.error(f"unknown algos: {bad} (known: {ALL_ALGOS})")

    out_csv = os.path.join(HERE, "extra_instances", "convergence_study.csv")
    done = set()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["algo"], r["problem"]))

    pairs = [(a, p) for a in algos for p in problems if (a, p) not in done]
    runs = len(pairs) * len(budgets) * len(seeds)
    cpu_s = sum(b for _ in pairs for b in budgets) * len(seeds)
    print("=" * 80)
    print("CONVERGENCE STUDY  (anytime HV vs budget ladder; tuned configs)")
    print("=" * 80)
    print(f"algos    : {algos}")
    print(f"problems : {problems}")
    print(f"ladder   : {budgets} s")
    print(f"seeds    : {seeds}   epsilon: {args.epsilon:.2%}   workers: {args.workers}")
    print(f"pairs    : {len(pairs)} ({len(done)} already done, skipped)" if args.resume
          else f"pairs    : {len(pairs)}")
    print(f"cell-runs: {runs}   approx CPU-time: {cpu_s/3600:.1f} h   "
          f"wall @{args.workers}: {cpu_s/max(1,args.workers)/3600:.1f} h")
    print("\nTuned configs that will be used (metaheuristics):")
    for a in algos:
        if a in META:
            for p in problems:
                tp = tuned_params(a, p)
                print(f"  {a:<7}{p:<14}{tp}")
    print("=" * 80)
    if args.plan_only:
        print("\n--plan-only: nothing executed.")
        return

    fieldnames = ["algo", "problem", "budget", "seed", "score"]
    write_header = not os.path.exists(out_csv)
    results = []
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_pair, a, p, budgets, seeds, args.epsilon): (a, p)
                for a, p in pairs}
        for fut in cf.as_completed(futs):
            a, p = futs[fut]
            try:
                algo, problem, rows, verdict = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {a}/{p}: {e}")
                continue
            # checkpoint this pair immediately
            if rows:
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        w.writeheader(); write_header = False
                    w.writerows(rows)
            results.append((algo, problem, rows, verdict))
            print(f"  done {algo:<7}/{problem:<13} {verdict}")
    print(f"\nsweep done in {(time.time()-t0)/60:.1f} min")

    # ---- verdict table ----
    print("\n" + "=" * 80)
    print("VERDICT  (CONVERGED = anytime curve flat at last doubling; "
          "RISING = extend ladder)")
    print("=" * 80)
    print(f"  {'algo':<7}{'instance':<14}{'score@min':>14}{'score@max':>14}  verdict")
    rising = 0
    for algo, problem, rows, verdict in sorted(results):
        if not rows:
            print(f"  {algo:<7}{problem:<14}{'-':>14}{'-':>14}  {verdict}")
            continue
        by_b = {}
        for r in rows:
            by_b.setdefault(r["budget"], []).append(r["score"])
        bmin, bmax = min(by_b), max(by_b)
        smin = statistics.median(by_b[bmin])
        smax = statistics.median(by_b[bmax])
        if verdict.startswith("RISING"):
            rising += 1
        print(f"  {algo:<7}{problem:<14}{smin:>14,.0f}{smax:>14,.0f}  {verdict}")
    print("=" * 80)
    if rising == 0:
        print("ALL CONVERGED -> every method's anytime curve is flat at these "
              "budgets. Combined with a flat seed-union curve "
              "(tools/ceiling_validation.py at rising seed counts), this is the "
              "green light: best results, at tuned hyperparameters, reached.")
    else:
        print(f"{rising} method(s) STILL RISING -> extend --budgets further out "
              "and re-run with --resume; they have not reached their limit yet.")
    print("=" * 80)


if __name__ == "__main__":
    main()

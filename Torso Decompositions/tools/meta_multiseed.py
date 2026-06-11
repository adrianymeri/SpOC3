#!/usr/bin/env python3
"""
meta_multiseed.py -- multi-seed head-to-head harness for the four-family
comparison (Hill Climbing vs Simulated Annealing vs GRASP vs VNS), including
the *enhanced* metaheuristic variants:

  hc9        HC baseline (HV-accept, hc9)              -- reference local search
  sa         SA, scalar energy                         -- baseline SA
  sa_amosa   SA, AMOSA bi-objective acceptance         -- enhanced SA
  grasp      GRASP, fixed alpha                         -- baseline GRASP
  grasp_pr   GRASP + path-relinking + reactive-alpha    -- enhanced GRASP
  vns        VNS, stochastic HV-descent                 -- baseline VNS
  vns_vnd    VNS + true VND local search                -- enhanced VNS

Why this harness (and not tools/multiseed.py)?
----------------------------------------------
tools/multiseed.py only knows the HC entry points.  A *publishable* head-to-
head across the four families needs identical seed symmetry for every family
plus a non-parametric significance test.  This script runs every config at the
same set of seeds on the same instances, re-scores each freshly-written
submission with the canonical evaluator (so the recorded number equals the
verify score, not a runtime print), and then runs a Friedman omnibus test with
a Nemenyi post-hoc (critical-distance) ranking.

Canonical submissions are NEVER touched: every run writes to a throwaway stem
("<name>_ms") that is deleted afterwards.

Usage
-----
    python3 tools/meta_multiseed.py                       # default sweep
    python3 tools/meta_multiseed.py --configs sa,sa_amosa,grasp,grasp_pr \
            --problems small-graph,medium-graph --seeds 1,2,3,7,42
    python3 tools/meta_multiseed.py --append

Output: extra_instances/meta_multiseed.csv  (one row per config x problem x seed)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from core import (  # noqa: E402
    build_adj_bitsets,
    evaluate,
    graph_path,
    hypervolume_2d,
    load_graph,
    repo_root,
)
import algorithms.simulated_annealing.sa as sa            # noqa: E402
import algorithms.grasp.grasp as grasp                    # noqa: E402
import algorithms.vns.vns as vns                          # noqa: E402
import algorithms.aco.aco as aco                          # noqa: E402
import algorithms.nsga2.nsga2 as nsga2                    # noqa: E402
import importlib                                          # noqa: E402

CANONICAL_BUDGET = {"small-graph": 25.0, "medium-graph": 12.0,
                    "large-graph": 25.0}


# --- locked tuned configs ---------------------------------------------------
# Best cell per (config, problem) from the full-grid seed-42 sweep
# (tools/run_metaheuristics.py -> extra_instances/tuning_*.csv).  These are the
# hyperparameters carried into the multi-seed head-to-head so seeds 1..11 race
# the *tuned* configurations, not library defaults.  hc9 is the untuned HC
# reference local search.
TUNED = {
    "sa": {
        "small-graph":  dict(schedule="linear",    alpha=0.90, steps_per_t=400),
        "medium-graph": dict(schedule="geometric", alpha=0.85, steps_per_t=100),
        "large-graph":  dict(schedule="geometric", alpha=0.85, steps_per_t=100),
    },
    "sa_amosa": {
        "small-graph":  dict(schedule="linear",    alpha=0.85, steps_per_t=100),
        "medium-graph": dict(schedule="geometric", alpha=0.85, steps_per_t=100),
        "large-graph":  dict(schedule="geometric", alpha=0.85, steps_per_t=100),
    },
    "grasp": {
        "small-graph":  dict(alpha=0.0, restarts=4),
        "medium-graph": dict(alpha=0.0, restarts=8),
        "large-graph":  dict(alpha=0.3, restarts=4),
    },
    "grasp_pr": {
        "small-graph":  dict(restarts=16, elite_size=8),
        "medium-graph": dict(restarts=8,  elite_size=3),
        "large-graph":  dict(restarts=4,  elite_size=3),
    },
    "vns": {
        "small-graph":  dict(k_max=2, shake_strength=2, ls_steps=600),
        "medium-graph": dict(k_max=2, shake_strength=1, ls_steps=300),
        "large-graph":  dict(k_max=2, shake_strength=1, ls_steps=150),
    },
    "vns_vnd": {
        "small-graph":  dict(k_max=2, shake_strength=2, vnd_k=16),
        "medium-graph": dict(k_max=2, shake_strength=2, vnd_k=4),
        "large-graph":  dict(k_max=2, shake_strength=4, vnd_k=8),
    },
    # ACO configs LOCKED from the full-grid seed-42 sweep (tuning_aco.csv /
    # tuning_aco_mmas.csv).  large-graph is the DOCUMENTED NEGATIVE RESULT:
    # every pure-construction cell scored -0 (empty archive -- only ~16 ants
    # built in 25s on n=2426/253,895 edges), so the large cell carries the
    # least-bad-but-still-failing {ants=5, beta=1.0, rho=0.05} for the record.
    "aco": {
        "small-graph":  dict(ants=20, beta=4.0, rho=0.1),   # -1,798,483
        "medium-graph": dict(ants=10, beta=4.0, rho=0.05),  # -1,513,761
        "large-graph":  dict(ants=5,  beta=1.0, rho=0.05),  # -0 (fails)
    },
    "aco_mmas": {
        "small-graph":  dict(ants=10, beta=4.0, rho=0.3),   # -1,795,913
        "medium-graph": dict(ants=10, beta=4.0, rho=0.05),  # -1,513,761
        "large-graph":  dict(ants=5,  beta=1.0, rho=0.05),  # -0 (fails)
    },
    # Hybrid ACO (construction + shared HV descent) LOCKED from the hybrid
    # full-grid seed-42 sweep (tuning_aco_ls.csv / tuning_aco_mmas_ls.csv).
    # KEY RESULT: the shared descent clears the large-graph feasibility wall
    # -- every large cell now scores -4,670,050 (vs -0 for pure construction);
    # the score is flat across the large grid because descent, not the colony,
    # supplies the front there.
    "aco_ls": {
        "small-graph":  dict(ants=5,  beta=4.0, rho=0.3, ls_steps=200),  # -1,796,376
        "medium-graph": dict(ants=5,  beta=4.0, rho=0.1, ls_steps=200),  # -1,524,335
        "large-graph":  dict(ants=5,  beta=4.0, rho=0.1, ls_steps=200),  # -4,670,050
    },
    "aco_mmas_ls": {
        "small-graph":  dict(ants=10, beta=4.0, rho=0.3, ls_steps=100),  # -1,795,674
        "medium-graph": dict(ants=5,  beta=4.0, rho=0.1, ls_steps=200),  # -1,524,335
        "large-graph":  dict(ants=5,  beta=4.0, rho=0.1, ls_steps=200),  # -4,670,050
    },
    # Population-based MOEA chapter (NSGA-II / SMS-EMOA) LOCKED from the
    # full-grid seed-42 sweep (tuning_nsga2.csv / tuning_sms.csv).  HEADLINE:
    # SMS-EMOA wins small-graph outright (-1,818,395, ahead of GRASP+PR's
    # -1,816,409), and pure NSGA-II is 2nd-best on large-graph (-4,814,777,
    # ahead of VNS).  The family trails GRASP/VNS on medium-graph.
    "nsga2": {
        "small-graph":  dict(pop=40, pc=0.7, pm=0.2),  # -1,817,539
        "medium-graph": dict(pop=20, pc=0.9, pm=0.2),  # -1,594,342
        "large-graph":  dict(pop=20, pc=0.9, pm=0.4),  # -4,814,777
    },
    "sms": {
        "small-graph":  dict(pop=20, pc=0.9, pm=0.2),  # -1,818,395 (best small)
        "medium-graph": dict(pop=20, pc=0.9, pm=0.4),  # -1,590,049
        "large-graph":  dict(pop=20, pc=0.7, pm=0.2),  # -4,779,701
    },
    # Memetic '_ls' variants (population EA + per-offspring HV-descent) LOCKED
    # from tuning_nsga2_ls.csv / tuning_sms_ls.csv.  KEY FINDING: the descent
    # does NOT uniformly help at these budgets -- it costs evals, so pure
    # `sms` beats `sms_ls` on small and pure `nsga2` beats `nsga2_ls` on large.
    "nsga2_ls": {
        "small-graph":  dict(pop=20, pm=0.6, ls_steps=200),  # -1,816,959
        "medium-graph": dict(pop=20, pm=0.3, ls_steps=200),  # -1,594,821
        "large-graph":  dict(pop=20, pm=0.3, ls_steps=100),  # -4,776,072
    },
    "sms_ls": {
        "small-graph":  dict(pop=20, pm=0.3, ls_steps=50),   # -1,815,819
        "medium-graph": dict(pop=20, pm=0.3, ls_steps=200),  # -1,594,821
        "large-graph":  dict(pop=20, pm=0.3, ls_steps=100),  # -4,785,225
    },
}


# --- config registry --------------------------------------------------------
# Each invoker runs the algorithm at a throwaway stem with the LOCKED tuned
# hyperparameters for (config, problem); the caller re-scores the submission.
def _run_hc(name, problem, budget, seed, here):
    # hc9.run writes to the fixed canonical "hc9" stem and takes no --algo
    # override, so caller (run_one) protects the canonical file by backup.
    mod = importlib.import_module("algorithms.hill_climbing.hc9_hv_accept")
    mod.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20)


def _run_sa(name, problem, budget, seed, here, accept):
    cfg = TUNED["sa_amosa" if accept == "amosa" else "sa"][problem]
    sa.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20,
           accept=accept, algo=name, **cfg)


def _run_grasp(name, problem, budget, seed, here, pr=False, reactive=False):
    cfg = TUNED["grasp_pr" if pr else "grasp"][problem]
    grasp.run(problem, budget, seed, here, num_t_seeds=20,
              path_relinking=pr, reactive_alpha=reactive, algo=name, **cfg)


def _run_vns(name, problem, budget, seed, here, local_search="hv-descent"):
    cfg = TUNED["vns_vnd" if local_search == "vnd" else "vns"][problem]
    vns.run(problem, budget, seed, 10 ** 18, here, num_t_seeds=20,
            local_search=local_search, algo=name, **cfg)


def _run_aco(name, problem, budget, seed, here, variant="as",
             local_search=False):
    if local_search:
        key = "aco_mmas_ls" if variant == "mmas" else "aco_ls"
    else:
        key = "aco_mmas" if variant == "mmas" else "aco"
    cfg = TUNED[key][problem]
    aco.run(problem, budget, seed, here, num_t_seeds=20,
            variant=variant, local_search=local_search, algo=name, **cfg)


def _run_nsga2(name, problem, budget, seed, here, variant="nsga2",
               local_search=False):
    key = (variant + "_ls") if local_search else variant
    cfg = TUNED[key][problem]
    nsga2.run(problem, budget, seed, here, num_t_seeds=20,
              variant=variant, local_search=local_search, algo=name, **cfg)


CONFIGS = {
    "hc9":      lambda nm, p, b, s, h: _run_hc(nm, p, b, s, h),
    "sa":       lambda nm, p, b, s, h: _run_sa(nm, p, b, s, h, accept="scalar"),
    "sa_amosa": lambda nm, p, b, s, h: _run_sa(nm, p, b, s, h, accept="amosa"),
    "grasp":    lambda nm, p, b, s, h: _run_grasp(nm, p, b, s, h),
    "grasp_pr": lambda nm, p, b, s, h: _run_grasp(nm, p, b, s, h,
                                                  pr=True, reactive=True),
    "vns":      lambda nm, p, b, s, h: _run_vns(nm, p, b, s, h,
                                                local_search="hv-descent"),
    "vns_vnd":  lambda nm, p, b, s, h: _run_vns(nm, p, b, s, h,
                                                local_search="vnd"),
    "aco":      lambda nm, p, b, s, h: _run_aco(nm, p, b, s, h, variant="as"),
    "aco_mmas": lambda nm, p, b, s, h: _run_aco(nm, p, b, s, h,
                                                variant="mmas"),
    "aco_ls":   lambda nm, p, b, s, h: _run_aco(nm, p, b, s, h, variant="as",
                                                local_search=True),
    "aco_mmas_ls": lambda nm, p, b, s, h: _run_aco(nm, p, b, s, h,
                                                   variant="mmas",
                                                   local_search=True),
    "nsga2":    lambda nm, p, b, s, h: _run_nsga2(nm, p, b, s, h,
                                                  variant="nsga2"),
    "sms":      lambda nm, p, b, s, h: _run_nsga2(nm, p, b, s, h,
                                                  variant="sms"),
    "nsga2_ls": lambda nm, p, b, s, h: _run_nsga2(nm, p, b, s, h,
                                                  variant="nsga2",
                                                  local_search=True),
    "sms_ls":   lambda nm, p, b, s, h: _run_nsga2(nm, p, b, s, h,
                                                  variant="sms",
                                                  local_search=True),
}

DEFAULT_CONFIGS = ["hc9", "sa", "sa_amosa", "grasp", "grasp_pr",
                   "vns", "vns_vnd", "aco", "aco_mmas",
                   "aco_ls", "aco_mmas_ls", "nsga2", "sms",
                   "nsga2_ls", "sms_ls"]


def score_of(sub_path, gr_path):
    """Re-score a written submission with the canonical evaluator; the result
    equals the official verify score (-HV of the submitted top-20 front)."""
    with open(sub_path) as f:
        payload = json.load(f)
    entry = payload[0] if isinstance(payload, list) else payload
    n, adj = load_graph(gr_path)
    ab = build_adj_bitsets(n, adj)
    fits = [evaluate(dv[:-1], dv[-1], ab, n) for dv in entry["decisionVector"]]
    return -hypervolume_2d(fits, n)


def run_one(config, problem, seed, here):
    # hc9 writes the canonical "hc9" stem (no --algo override): back it up and
    # restore afterwards so the canonical submission is never disturbed.  Every
    # other config writes a throwaway "<name>_ms" stem we delete after scoring.
    canonical = (config == "hc9")
    stem = "hc9" if canonical else f"{config}_ms"
    sub_path = os.path.join(here, "submissions", problem, f"{stem}.json")
    backup = sub_path + ".ms.bak" if canonical else None
    if canonical and os.path.exists(sub_path):
        shutil.copy2(sub_path, backup)
    t0 = time.time()
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            CONFIGS[config](stem, problem, CANONICAL_BUDGET[problem], seed, here)
    except Exception as exc:   # noqa: BLE001
        print(f"  ! {config} {problem} seed={seed}: {exc!r}")
        if backup and os.path.exists(backup):
            shutil.move(backup, sub_path)
        return float("nan"), time.time() - t0
    elapsed = time.time() - t0
    gr = graph_path(here, problem)
    score = score_of(sub_path, gr) if os.path.exists(sub_path) else float("nan")
    if canonical:
        if backup and os.path.exists(backup):
            shutil.move(backup, sub_path)          # restore canonical hc9.json
    elif os.path.exists(sub_path):
        os.remove(sub_path)                         # drop throwaway stem
    return score, elapsed


# --- statistics -------------------------------------------------------------
def friedman_nemenyi(scores):
    """scores: dict[config] -> list of per-(problem,seed) scores, aligned by
    block (same index = same problem+seed).  Returns (chi2, dof, p_or_None,
    avg_ranks, cd_at_0.05) ranking configs from best (lowest score = most
    negative -HV) to worst.  Uses scipy if available for the p-value; the
    Friedman statistic and Nemenyi CD are computed directly so the script has
    no hard scipy dependency."""
    configs = list(scores.keys())
    k = len(configs)
    blocks = len(next(iter(scores.values())))
    # rank within each block: rank 1 = best = lowest (most negative) score
    rank_sums = {c: 0.0 for c in configs}
    for b in range(blocks):
        col = sorted(configs, key=lambda c: scores[c][b])
        # average ranks for ties
        i = 0
        vals = [(c, scores[c][b]) for c in col]
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[j + 1][1] == vals[i][1]:
                j += 1
            avg = sum(range(i + 1, j + 2)) / (j - i + 1)
            for t in range(i, j + 1):
                rank_sums[vals[t][0]] += avg
            i = j + 1
    avg_ranks = {c: rank_sums[c] / blocks for c in configs}
    N = blocks
    chi2 = (12.0 * N / (k * (k + 1))) * (
        sum(r * r for r in avg_ranks.values()) - k * (k + 1) ** 2 / 4.0)
    dof = k - 1
    p = None
    try:
        from scipy.stats import chi2 as _c   # noqa: WPS433
        p = float(_c.sf(chi2, dof))
    except Exception:   # noqa: BLE001
        p = None
    # Nemenyi critical distance at alpha=0.05; q_alpha from the studentized
    # range / sqrt(2) table (Demsar 2006, Table 5).
    Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}
    q = Q05.get(k, 3.164)
    cd = q * math.sqrt(k * (k + 1) / (6.0 * N))
    return chi2, dof, p, avg_ranks, cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                    help="comma-separated config names: "
                         + ", ".join(CONFIGS.keys()))
    ap.add_argument("--problems",
                    default="small-graph,medium-graph,large-graph")
    ap.add_argument("--seeds", default="1,2,3,42",
                    help="comma-separated seeds")
    ap.add_argument("--out", default="extra_instances/meta_multiseed.csv")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip Friedman/Nemenyi (just write the CSV)")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; choose from {list(CONFIGS)}")
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    here = repo_root()
    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = "a" if args.append else "w"
    write_header = (not args.append) or (not os.path.exists(out_path))
    f = open(out_path, mode, newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["config", "problem", "seed", "score", "elapsed_s"])

    # collected[config][(problem,seed)] = score  (for the stats block)
    collected = {c: {} for c in configs}
    total = len(configs) * len(problems) * len(seeds)
    done = 0
    t_start = time.time()
    print(f"Meta multi-seed head-to-head: {total} runs "
          f"({configs} x {problems} x seeds {seeds})\n")

    for problem in problems:
        for config in configs:
            for seed in seeds:
                done += 1
                score, elapsed = run_one(config, problem, seed, here)
                w.writerow([config, problem, seed, f"{score:.0f}",
                            f"{elapsed:.2f}"])
                f.flush()
                collected[config][(problem, seed)] = score
                print(f"  [{done:>3}/{total}] {config:<9} {problem:<13} "
                      f"seed={seed:<3}  score = {score:>14,.0f}  "
                      f"({elapsed:5.1f}s)")
    f.close()
    print(f"\nWrote {out_path}  ({time.time() - t_start:.1f}s)")

    if args.no_stats:
        return
    # Build aligned blocks (problem,seed) present for ALL configs.
    blocks = [(p, s) for p in problems for s in seeds
              if all(not math.isnan(collected[c].get((p, s), float("nan")))
                     for c in configs)]
    if len(blocks) < 2 or len(configs) < 2:
        print("\n(Not enough complete blocks for Friedman/Nemenyi.)")
        return
    aligned = {c: [collected[c][b] for b in blocks] for c in configs}
    chi2, dof, p, avg_ranks, cd = friedman_nemenyi(aligned)
    print("\n=== Friedman omnibus (lower -HV score = better) ===")
    print(f"  blocks (problem x seed) = {len(blocks)}, configs = {len(configs)}")
    pstr = f"{p:.3g}" if p is not None else "n/a (install scipy for p-value)"
    print(f"  chi^2 = {chi2:.3f}, dof = {dof}, p = {pstr}")
    print("\n=== Nemenyi average ranks (1 = best) ===")
    for c in sorted(avg_ranks, key=lambda c: avg_ranks[c]):
        print(f"  {c:<10} avg rank {avg_ranks[c]:.3f}")
    print(f"\n  critical distance (alpha=0.05) = {cd:.3f}")
    print("  -> two configs differ significantly iff their avg ranks differ "
          "by more than the CD.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gpu_search.py -- GPU-accelerated large-population CMA-ES for torso decomposition.

This is the search that the GPU evaluator unlocks. Instead of the CPU's ~14
evaluations per generation, it samples a LARGE population of policies
(--pop, e.g. 4096), decodes each to an elimination ordering by argsort over the
spectral node features, and scores the whole population on the GPU in one shot
(algorithms.continuous.gpu_eval.GpuEvaluator). The far larger population yields
a much better gradient estimate and far more exploration per unit wall-time --
the throughput step-change THESIS.md s9 identifies.

ADDITIVE / SAFE: writes only the `gpucma` submission stem (override with --algo).
It never touches canonical cmaes/gbdt/hc files. Harvested orderings are pooled
by tools/portfolio.py like any other source.

CORRECTNESS: the GPU evaluator is validated bit-for-bit against core.evaluate by
tools/validate_gpu.py -- run that FIRST. If numba/CUDA is unavailable this
script falls back to the (slow) numpy reference so it still runs anywhere.

Usage (on a Colab GPU runtime, after validate_gpu.py passes):
    python3 tools/gpu_search.py --problem large-graph --pop 4096 --budget 1800
    python3 tools/gpu_search.py --problem large-graph --pop 8192 --budget 3600 --seed 7
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, math, time, random, json
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, hypervolume_2d, MAX_TW, min_degree_perm,
                  submission_path, write_submission, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import get_features, SepCMAES
from algorithms.continuous.gpu_eval import (build_adj_words, cpu_eval_batch,
                                            staircase_widths)

INFEAS = 1e9


# --------------------------------------------------------------------------- #
def ridge_policy(F, perm, lam=1.0):
    """Least-squares policy whose argsort reproduces `perm` (warm start)."""
    pos = np.empty(F.shape[0]); pos[np.asarray(perm)] = np.arange(F.shape[0])
    Fb = np.hstack([F, np.ones((F.shape[0], 1))])
    A = Fb.T @ Fb + lam * np.eye(Fb.shape[1])
    w = np.linalg.solve(A, Fb.T @ pos)
    return w[:-1]


def row_penalties(deg, status, cap):
    """eval_fitness-exact penalty for mildly-infeasible (status 1) rows."""
    pen = np.zeros(deg.shape[0])
    for p in np.where(status == 1)[0]:
        over = deg[p][deg[p] > cap] - cap
        pen[p] = 501.0 + float(over.sum() - over[0])   # first overflow -> 501
    return pen


def make_evaluator(aw, n, W, cap, capacity):
    """Return (eval_fn, label). eval_fn(perms)->(deg,status)."""
    try:
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("no CUDA device")
        from algorithms.continuous.gpu_eval import GpuEvaluator
        ev = GpuEvaluator(aw, n, W, cap=cap, capacity=capacity)
        return (lambda perms: ev.eval(perms)), "GPU"
    except Exception as e:  # noqa: BLE001
        print(f"[GPU unavailable: {e}; using slow numpy reference]", flush=True)
        return (lambda perms: cpu_eval_batch(perms, aw, n, W, cap)), "CPU-ref"


def load_warmstart(path, n):
    """Return an int32[K,n] array of banked orderings, or None."""
    try:
        d = json.load(open(path))
        perms = [p for p in d["perms"] if len(p) == n and sorted(p) == list(range(n))]
        return np.array(perms, dtype=np.int32) if perms else None
    except Exception as e:  # noqa: BLE001
        print(f"[warmstart {path} not loaded: {e}]", flush=True)
        return None


# --------------------------------------------------------------------------- #
def run(problem, budget_s, seed, pop, here, k_eig=32, algo="gpucma",
        warmstart=None, cold=False):
    n, adj = load_graph(graph_path(here, problem))
    ab = build_adj_bitsets(n, adj)
    aw, W = build_adj_words(n, ab)
    target = LEADERBOARD_TARGETS.get(problem)
    t_grid = sorted({int(round(i * (n - 1) / 39)) for i in range(40)})

    print(f"\n=== gpu_search -- {problem} ===")
    print(f"n = {n}, edges = {sum(map(len, adj))//2}, W = {W}, pop = {pop}")
    if target is not None:
        print(f"leaderboard target = {target:,}  (more negative = better)")
    t0 = time.time()
    F, _ = get_features(here, problem, n, adj, k_eig)
    eval_fn, label = make_evaluator(aw, n, W, MAX_TW, pop)
    print(f"features {F.shape}, evaluator = {label}, built in {time.time()-t0:.1f}s",
          flush=True)

    archive = ParetoArchive()

    def harvest(perms, deg, status):
        """Add the t_grid front of every feasible ordering; return fitness[]."""
        widths = staircase_widths(deg)                 # P x n
        fit = np.full(perms.shape[0], INFEAS, dtype=np.float64)
        pen = row_penalties(deg, status, MAX_TW)
        feas = np.where(status == 0)[0]
        for p in feas:
            front = [(int(widths[p, t]), int(t)) for t in t_grid]
            for w, t in front:
                archive.try_add(w, t, perms[p].tolist())
            fit[p] = -hypervolume_2d(front, n)
        mild = np.where(status == 1)[0]
        fit[mild] = INFEAS + pen[mild]
        fit[status == 2] = INFEAS + 2000.0
        return fit

    # ----- warm start ---------------------------------------------------- #
    # Prefer CONTINUING from the banked best orderings (warmstart file); fall
    # back to a min-degree start only if none is available.
    if warmstart is None and not cold:
        cand = os.path.join(here, "data", f"warmstart_{problem}.json")
        warmstart = cand if os.path.exists(cand) else None
    banked = load_warmstart(warmstart, n) if warmstart else None

    best_perm = None; best_hv = -1.0
    if banked is not None and len(banked) > 0:
        print(f"warm start: continuing from {len(banked)} banked orderings "
              f"({os.path.basename(warmstart)})", flush=True)
        widths_all = None
        for i in range(0, len(banked), pop):                # batch through evaluator
            chunk = banked[i:i + pop]
            dC, sC = eval_fn(chunk)
            harvest(chunk, dC, sC)
            wC = staircase_widths(dC)
            for j in range(len(chunk)):
                if sC[j] == 0:
                    front = [(int(wC[j, t]), int(t)) for t in t_grid]
                    hv = hypervolume_2d(front, n)
                    if hv > best_hv:
                        best_hv = hv; best_perm = chunk[j]
    if best_perm is None:                                    # fallback
        print("warm start: min-degree (no banked orderings found)", flush=True)
        seed_perms = np.array(
            [min_degree_perm(n, ab, rng=random.Random(seed + s)) for s in range(4)],
            dtype=np.int32)
        sd, ss = eval_fn(seed_perms)
        harvest(seed_perms, sd, ss)
        best_perm = seed_perms[0]
    warm = ridge_policy(F, best_perm)

    dim = F.shape[1]
    opt = SepCMAES(dim, sigma0=0.5, seed=seed)
    # crank the population up to exploit the GPU
    opt.lam = int(pop); opt.mu = opt.lam // 2
    w = np.log(opt.mu + 0.5) - np.log(np.arange(1, opt.mu + 1))
    opt.w = w / w.sum(); opt.mueff = 1.0 / np.sum(opt.w ** 2)
    opt.mean = warm.copy()

    best = -archive.hypervolume(n)
    print(f"warm-start archive {len(archive)} -> score {best:,.0f}", flush=True)

    gen = 0; evals = len(archive); lastlog = t0
    while time.time() - t0 < budget_s:
        X = opt.ask()                                  # pop x dim
        scores = F @ X.T                               # n x pop
        perms = np.argsort(scores, axis=0).T.astype(np.int32)   # pop x n
        deg, status = eval_fn(perms)
        fit = harvest(perms, deg, status)
        opt.tell(fit)
        gen += 1; evals += perms.shape[0]
        cur = -archive.hypervolume(n)
        if cur < best:
            best = cur
        if time.time() - lastlog >= 5 or time.time() - t0 >= budget_s:
            feas = int(np.count_nonzero(status == 0))
            msg = (f"  gen {gen:4d} | evals {evals:8d} | feasible {feas:4d}/{pop} "
                   f"| archive {len(archive):3d} | score {best:,.0f} | t {time.time()-t0:5.0f}s")
            if target is not None:
                g = best - target
                msg += f" | gap {g:+,.0f}" + (" BEAT!" if g < 0 else "")
            print(msg, flush=True)
            lastlog = time.time()

    final = -archive.hypervolume(n)
    print(f"\nFinished in {time.time()-t0:.1f}s, {gen} gens, {evals:,} evals, archive {len(archive)}")
    print(f"Official score: {final:,.0f}")
    if target is not None:
        g = final - target
        print(f"Gap to target ({target:,}): {g:>+14,.0f}  "
              f"({'BEAT THE LEADER' if g < 0 else f'{abs(g):,.0f} short'})")

    top = archive.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = submission_path(here, problem, algo)
    write_submission(dvs, problem, out)
    print(f"Wrote submission: {out}  ({len(dvs)} vectors)")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=4096, help="population per generation (GPU batch)")
    ap.add_argument("--k-eig", type=int, default=32)
    ap.add_argument("--algo", default="gpucma", help="submission stem (additive)")
    ap.add_argument("--warmstart", default=None,
                    help="JSON of banked orderings to continue from "
                         "(default: data/warmstart_<problem>.json if present)")
    ap.add_argument("--cold", action="store_true",
                    help="ignore any warmstart file; start from min-degree")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, args.pop, repo_root(),
        k_eig=args.k_eig, algo=args.algo, warmstart=args.warmstart, cold=args.cold)


if __name__ == "__main__":
    main()

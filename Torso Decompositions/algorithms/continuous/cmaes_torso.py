#!/usr/bin/env python3
"""
cmaes_torso.py -- continuous-encoding search (the leaderboard-winning paradigm).

Why this chapter exists
-----------------------
Every prior family in this project searches DIRECTLY in permutation space
(swap/insert/reverse moves, min-degree construction).  The two top ESA
leaderboard solutions do something fundamentally different and beat us by a
wide margin on the dense instances:

  - `cuda-torso`  : a linear policy over spectral node features, scored per
                    vertex; the elimination order is `argsort(scores)`.  The
                    policy weights are evolved by GPU neuro-evolution.
  - `fast-cma-es` : the same continuous->argsort decode, optimised by
                    CMA-ES / MO-DE with parallel restarts.

Both lift the problem out of permutation space into a CONTINUOUS space and let
`argsort` decode the permutation.  This module replicates that paradigm on CPU:

    x  (continuous policy/priority vector)
        -> scores = features @ x        (or x itself, "direct" encoding)
        -> perm   = argsort(scores)
        -> ONE fill-in pass gives deg[i] for every elimination step
        -> width(t) = suffix-max(deg[t:])  for ALL t at once
        -> the whole (width, t) front from a single permutation
    optimise x to maximise the front's hypervolume (sep-CMA-ES, or fcmaes).

The single-pass "one permutation -> whole front" trick (the same one
`cuda-torso`'s CUDA kernel exploits) makes each fitness call yield a complete
front instead of one point, which is what makes this encoding efficient.

Encodings
---------
  --encoding spectral  (default) : x are policy weights over node features
        [degree profile (5) + top-K Laplacian eigenvectors].  Low-dimensional
        (~5+K), exactly cuda-torso's representation minus the GPU.
  --encoding direct              : x is a length-N per-vertex priority vector.
        High-dimensional; sep-CMA-ES handles it (diagonal covariance).

Optimiser
---------
  --engine builtin (default) : a self-contained separable CMA-ES (numpy only),
        so this runs anywhere -- no GPU, no fcmaes needed.
  --engine fcmaes            : use the uploaded fast-cma-es toolkit if it is
        importable (CMA-ES / BiteOpt / MO-DE with parallel retry).  This is the
        production path on the workstation.

ADDITIVE / SAFE: writes only the `cmaes` submission stem (overridable with
--algo).  No canonical submission is touched.

Usage
-----
    python3 algorithms/continuous/cmaes_torso.py --problem small-graph --budget 60
    python3 algorithms/continuous/cmaes_torso.py --problem large-graph \
        --encoding spectral --eigenvectors 32 --budget 600
    python3 algorithms/continuous/cmaes_torso.py --problem medium-graph \
        --encoding direct --engine fcmaes --budget 600
"""

from __future__ import annotations

# --- sys.path bootstrap ----------------------------------------------------
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

import argparse
import math
import time
from typing import List, Tuple

import numpy as np

from core import (
    LEADERBOARD_TARGETS,
    MAX_TW,
    ParetoArchive,
    build_adj_bitsets,
    build_warm_start,
    graph_path,
    hypervolume_2d,
    load_graph,
    repo_root,
    submission_path,
    write_submission,
)


# ---------------------------------------------------------------------------
# Node features (the spectral embedding cuda-torso uses)
# ---------------------------------------------------------------------------

def build_node_features(n: int, adj: List[set], k_eig: int) -> np.ndarray:
    """Return an (n x d) normalised feature matrix:
    [degree, neighbour-degree min/max/mean/std, top-k Laplacian eigenvectors].
    Pure-numpy (dense eigendecomposition); one-time cost."""
    deg = np.array([len(adj[v]) for v in range(n)], dtype=np.float64)
    feats = [deg]
    nmin = np.zeros(n); nmax = np.zeros(n); nmean = np.zeros(n); nstd = np.zeros(n)
    for v in range(n):
        if adj[v]:
            nd = deg[list(adj[v])]
            nmin[v], nmax[v], nmean[v], nstd[v] = nd.min(), nd.max(), nd.mean(), nd.std()
    feats += [nmin, nmax, nmean, nstd]

    if k_eig > 0:
        # normalised Laplacian L = I - D^-1/2 A D^-1/2 (dense; fine to n~2500)
        A = np.zeros((n, n), dtype=np.float64)
        for v in range(n):
            for u in adj[v]:
                A[v, u] = 1.0
        d = A.sum(1)
        dinv = np.zeros(n)
        nz = d > 0
        dinv[nz] = 1.0 / np.sqrt(d[nz])
        L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        pe = vecs[:, order[1:k_eig + 1]]   # skip the trivial constant vector
        for j in range(pe.shape[1]):
            feats.append(pe[:, j])

    F = np.vstack(feats).T  # (n, d)
    # z-normalise each feature column
    mu = F.mean(0); sd = F.std(0); sd[sd == 0] = 1.0
    return ((F - mu) / sd).astype(np.float64)


# ---------------------------------------------------------------------------
# Decode + single-pass full-front evaluation
# ---------------------------------------------------------------------------

_INFEAS = 1e7   # base offset so any infeasible fitness > any feasible (-HV <= 0)


def eval_fitness(perm: List[int], adj_bits: List[int], n: int,
                 t_grid: List[int], archive) -> float:
    """One fill-in elimination pass that returns a SCALAR fitness with a
    feasibility gradient -- the key to making the continuous encoding work on
    the dense instance (mirrors cuda-torso's `libeval.cu`):

      - feasible perm (no step exceeds the cap): seed `archive` with its whole
        (width, t) front (width(t) = suffix-max(deg[t:])) and return -HV(front)
        (<= 0, lower is better).
      - infeasible perm: return a POSITIVE graded penalty that grows with the
        amount of cap violation, so CMA-ES is pushed *toward* feasibility
        instead of seeing a flat wall.  We early-stop once the penalty blows
        up (the same trick that keeps `core.evaluate` fast on bad perms; its
        absence is what made early generations take ~45 s each on large).
    """
    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]
    temp = list(adj_bits)
    deg = [0] * n
    penalty = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        d = succ.bit_count()
        deg[i] = d
        if d > MAX_TW:
            penalty = 501 if penalty == 0 else penalty + (d - MAX_TW)
            if penalty > 1000:                       # deeply infeasible: bail
                return _INFEAS + penalty + 1000.0
        if succ:
            s = succ
            while s:
                vbit = s & -s
                s ^= vbit
                v = vbit.bit_length() - 1
                temp[v] |= succ ^ vbit
    if penalty > 0:                                   # mildly infeasible
        return _INFEAS + float(penalty)
    # feasible: suffix maximum width(t) = max(deg[t:])
    smax = [0] * n
    run = 0
    for i in range(n - 1, -1, -1):
        if deg[i] > run:
            run = deg[i]
        smax[i] = run
    front = [(smax[t], int(t)) for t in t_grid]
    for w, t in front:
        archive.try_add(w, t, perm)
    return -hypervolume_2d(front, n)


def decode(x: np.ndarray, features: np.ndarray) -> List[int]:
    """argsort decode.  spectral: scores = F @ x; direct: scores = x."""
    scores = features @ x if features is not None else x
    return [int(v) for v in np.argsort(scores, kind="stable")]


def get_features(here, problem, n, adj, k_eig):
    """Build the spectral feature matrix once and cache it to disk, keyed by
    (problem, k_eig).  The dense eigendecomposition is the expensive step; with
    a multi-seed sweep it must be paid ONCE per instance, not once per seed.
    Returns (features, was_cached)."""
    cache_dir = _os.path.join(here, ".feature_cache")
    _os.makedirs(cache_dir, exist_ok=True)
    cache = _os.path.join(cache_dir, f"{problem}_k{k_eig}.npy")
    if _os.path.exists(cache):
        try:
            F = np.load(cache)
            if F.shape[0] == n:
                return F, True
        except Exception:  # noqa: BLE001
            pass
    F = build_node_features(n, adj, k_eig)
    try:
        np.save(cache, F)
    except Exception:  # noqa: BLE001
        pass
    return F, False


# ---------------------------------------------------------------------------
# Separable CMA-ES (self-contained, numpy only)
# ---------------------------------------------------------------------------

class SepCMAES:
    """Minimal separable (diagonal-covariance) CMA-ES.  Scales to high-dim
    (the 'direct' encoding) and is trivial at low dim (the spectral policy)."""

    def __init__(self, dim: int, sigma0: float = 0.5, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.mean = self.rng.normal(0, 0.1, dim)
        self.sigma = sigma0
        self.lam = 4 + int(3 * math.log(dim))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        # separable c1/cmu (Ros & Hansen 2008 scaling)
        self.cc = (1 + 1 / dim + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff) * (dim + 2) / 3
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((dim + 2) ** 2 + self.mueff) * (dim + 2) / 3)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.C = np.ones(dim)          # diagonal covariance
        self.chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    def ask(self) -> np.ndarray:
        self.z = self.rng.standard_normal((self.lam, self.dim))
        self.y = self.z * np.sqrt(self.C)[None, :]
        return self.mean[None, :] + self.sigma * self.y

    def tell(self, fitness: np.ndarray) -> None:
        idx = np.argsort(fitness)            # minimise
        ysel = self.y[idx[:self.mu]]
        zsel = self.z[idx[:self.mu]]
        yw = (self.w[:, None] * ysel).sum(0)
        zw = (self.w[:, None] * zsel).sum(0)
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff) * zw
        hsig = (np.linalg.norm(self.ps) /
                math.sqrt(1 - (1 - self.cs) ** (2)) / self.chiN) < (1.4 + 2 / (self.dim + 1))
        self.pc = (1 - self.cc) * self.pc + (
            hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff)) * yw
        # diagonal C update
        cmu_term = (self.w[:, None] * (ysel ** 2)).sum(0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * cmu_term)
        self.C = np.maximum(self.C, 1e-20)
        self.sigma *= math.exp((self.cs / self.ds) *
                               (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, here, encoding="spectral", eigenvectors=32,
        num_t_seeds=40, engine="builtin", algo="cmaes", sigma0=0.5,
        no_warmstart=False, seed_banked=False):
    rng = np.random.default_rng(seed)
    n, adj = load_graph(graph_path(here, problem))
    adj_bits = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)

    t_grid = sorted({int(round(i * (n - 1) / (num_t_seeds - 1)))
                     for i in range(num_t_seeds)}) if num_t_seeds > 1 else [0]

    print(f"\n=== cmaes_torso -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj)//2}")
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"encoding = {encoding}, engine = {engine}, budget = {budget_s:.0f}s, "
          f"seed = {seed}, t-grid = {len(t_grid)}, "
          f"no-warmstart = {no_warmstart}, seed-banked = {seed_banked}")

    build_t0 = time.time()
    if encoding == "spectral":
        features, cached = get_features(here, problem, n, adj, eigenvectors)
        dim = features.shape[1]
        tag = "loaded from cache" if cached else f"built in {time.time()-build_t0:.1f}s"
        print(f"spectral features: {dim} dims "
              f"(5 degree + {eigenvectors} eigenvectors), {tag}", flush=True)
    else:
        features = None
        dim = n
        print(f"direct encoding: {dim} dims", flush=True)

    # IMPORTANT: start the optimisation budget clock AFTER the (one-time,
    # sometimes slow) feature build, so the eigendecomposition never eats the
    # search budget.  Previously t0 was set before this, which on large-graph
    # could consume the whole budget on feature construction alone.
    t0 = time.time()
    archive = ParetoArchive()

    # Optionally seed the archive with all banked orderings so the search
    # measures improvement from the current best portfolio, not from scratch.
    if seed_banked:
        from tools.gbfc import banked as _banked
        pool = _banked(here, problem, n, adj_bits)
        import random as _random
        for p in pool:
            eval_fitness(p, adj_bits, n, t_grid, archive)
        print(f"seeded archive from banked pool ({len(pool)} orderings), "
              f"score = {-archive.hypervolume(n):,.0f}", flush=True)

    if not no_warmstart:
        # Seed the archive with a min-degree/min-fill warm start so the
        # submission is never empty and the search has a feasible anchor --
        # the dense instance starts almost entirely infeasible otherwise.
        import random as _random
        ws_perm, ws_label = build_warm_start(n, adj_bits, rng=_random.Random(seed))
        eval_fitness(ws_perm, adj_bits, n, t_grid, archive)
        print(f"warm-start seed = {ws_label}; archive {len(archive)}, "
              f"score = {-archive.hypervolume(n):,.0f}", flush=True)
    else:
        print("warm-start skipped (--no-warmstart): CMA-ES starts from random x",
              flush=True)

    def fitness(x: np.ndarray) -> float:
        return eval_fitness(decode(x, features), adj_bits, n, t_grid, archive)

    if engine == "fcmaes":
        _run_fcmaes(fitness, dim, budget_s, seed)
    else:
        # With --no-warmstart, initialise x randomly to explore different
        # policy basins rather than anchoring near the min-degree solution.
        if no_warmstart:
            x0 = rng.normal(0.0, sigma0, dim)
        else:
            x0 = np.zeros(dim)
        opt = SepCMAES(dim, sigma0=sigma0, seed=seed)
        opt.mean = x0.copy()   # override default zero mean
        gen = 0
        last = 0.0
        while time.time() - t0 < budget_s:
            X = opt.ask()
            fits = np.array([fitness(X[i]) for i in range(X.shape[0])])
            opt.tell(fits)
            gen += 1
            if time.time() - last > 5:
                print(f"  gen {gen:5d} | evals {gen*opt.lam:7d} | "
                      f"archive {len(archive):3d} | "
                      f"score = {-archive.hypervolume(n):>14,.0f} | "
                      f"t = {time.time()-t0:5.1f}s", flush=True)
                last = time.time()
        print(f"  generations = {gen}, evals = {gen*opt.lam}")

    final = -archive.hypervolume(n)
    print(f"\nFinished in {time.time()-t0:.1f}s, archive {len(archive)}")
    print(f"Official score: {final:,.0f}")
    if target is not None:
        gap = final - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,.0f} short'})")

    top = archive.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = submission_path(here, problem, algo)
    write_submission(dvs, problem, out)
    print(f"Wrote submission: {out}  ({len(dvs)} vectors)")
    return final


def _run_fcmaes(fitness, dim, budget_s, seed):
    """Production path: full-covariance CMA-ES from the fast-cma-es toolkit
    (the optimiser the leaderboard winner used), if importable.  Same
    formulation as the builtin engine -- the archive is populated as a side
    effect of `fitness` -- but a stronger optimiser drives the policy search.

    fcmaes `minimize` has no wall-time argument; we stop via its
    `is_terminate(runid, iterations, best_value)` callback.  We keep
    `workers=1` so the run stays in-process and shares the archive (parallelism
    is across seeds in tools/cmaes_sweep.py, not inside one run)."""
    try:
        from fcmaes import cmaes
        from fcmaes.optimizer import Bounds
    except Exception as e:  # noqa: BLE001
        print(f"  [fcmaes unavailable: {e}; falling back to builtin sep-CMA-ES]",
              flush=True)
        opt = SepCMAES(dim, seed=seed)
        t0 = time.time()
        while time.time() - t0 < budget_s:
            X = opt.ask()
            opt.tell(np.array([fitness(X[i]) for i in range(X.shape[0])]))
        return

    start = time.time()

    def _terminate(runid, iterations, best_value):
        return (time.time() - start) > budget_s

    bounds = Bounds([-3.0] * dim, [3.0] * dim)
    cmaes.minimize(fitness, bounds, x0=np.zeros(dim), input_sigma=0.3,
                   max_evaluations=10 ** 9, max_iterations=10 ** 9,
                   popsize=4 + int(3 * math.log(dim)), workers=1,
                   is_terminate=_terminate, rg=np.random.default_rng(seed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoding", default="spectral", choices=["spectral", "direct"])
    ap.add_argument("--eigenvectors", type=int, default=32)
    ap.add_argument("--num-t-seeds", type=int, default=40)
    ap.add_argument("--engine", default="builtin", choices=["builtin", "fcmaes"])
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--algo", default="cmaes")
    ap.add_argument("--no-warmstart", action="store_true",
                    help="Skip min-degree warm start; init CMA-ES x ~ N(0, sigma0) "
                         "to explore different prefix basins.")
    ap.add_argument("--seed-banked", action="store_true",
                    help="Seed the archive from the current banked pool before search "
                         "so improvements are measured against the best known score.")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, repo_root(),
        encoding=args.encoding, eigenvectors=args.eigenvectors,
        num_t_seeds=args.num_t_seeds, engine=args.engine, algo=args.algo,
        sigma0=args.sigma0, no_warmstart=args.no_warmstart,
        seed_banked=args.seed_banked)


if __name__ == "__main__":
    main()

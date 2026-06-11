#!/usr/bin/env python3
"""
aco.py -- Ant Colony Optimization for the Torso-Decomposition bi-objective.

This is the "bridge" chapter between the trajectory/construction metaheuristics
(SA, GRASP, VNS) and the population-based evolutionary chapter that follows:
ACO is a *constructive, population-of-agents* method whose agents share a
global memory (the pheromone trail) rather than a genome.

References
----------
Dorigo, M. (1992).  "Optimization, Learning and Natural Algorithms" (PhD
thesis, Politecnico di Milano) -- the original Ant System.

Dorigo, M., Maniezzo, V. and Colorni, A. (1996).  "Ant System: optimization by
a colony of cooperating agents."  IEEE Trans. SMC-B 26(1): 29-41.

Stutzle, T. and Hoos, H.H. (2000).  "MAX-MIN Ant System."  Future Generation
Computer Systems 16(8): 889-914.  -- the enhanced variant (--variant mmas).

Dorigo, M. and Stutzle, T. (2004).  "Ant Colony Optimization."  MIT Press.

Method
------
Each ant constructs an elimination order one vertex at a time.  At step ``i``,
among the remaining vertices ``R`` the next vertex is drawn with probability

    p(v) proportional to  tau[i, v]**alpha  *  eta(v)**beta

where ``tau[i, v]`` is the pheromone for placing ``v`` at position ``i`` (a
position-indexed trail, as is standard for ACO on sequencing/ordering problems,
e.g. ACO for single-machine total weighted tardiness, den Besten & Stutzle),
and ``eta(v) = 1 / (1 + deg_R(v))`` is the dynamic **minimum-degree** heuristic
-- the *same* greedy signal GRASP uses, so at high ``beta`` (and uniform tau)
the ant reproduces GRASP's pure-greedy min-degree construction.  The fill-in
elimination bookkeeping is identical to ``grasp.construct_rcl`` for exact
comparability.

Each constructed order is evaluated at every threshold on the shared t-grid and
offered to the *same global Pareto archive* as every other chapter, so the
reported score is the HV of the union of all ants -- directly comparable.

Pheromone update
----------------
* **Ant System** (``--variant as``, the baseline): evaporate all trails by a
  factor ``(1 - rho)``, then every ant deposits ``Delta = quality`` on the
  (position, vertex) pairs it used, where ``quality`` is the ant's best feasible
  single-point dominated area normalised to ``(0, 1]``.

* **MAX-MIN Ant System** (``--variant mmas``, the additive enhancement):
  only the iteration-best ant deposits; trails are clamped to ``[tau_min,
  tau_max]`` with ``tau_max = quality_best / rho`` and ``tau_min = tau_max /
  (2 n)``; on ``stall_limit`` iterations without global improvement the trail is
  re-initialised to ``tau_max`` to escape stagnation.

Both variants optionally use the ACS pseudo-random-proportional rule: with
probability ``q0`` the ant takes the arg-max of ``tau**alpha * eta**beta``
instead of sampling (``q0 = 0`` recovers pure proportional selection).

Hyperparameters exposed for tuning: --ants, --alpha, --beta, --rho, --q0,
--variant, plus the shared --num-t-seeds.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# --- sys.path bootstrap -----------------------------------------------------
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

import random

from core import (
    LEADERBOARD_TARGETS,
    ParetoArchive,
    ensure_seeded,
    evaluate_full,
    repo_root,
)
from algorithms.meta_common import (
    OPERATORS,
    build_t_grid,
    call_op,
    finalize,
    hv_with_candidate,
    load_problem,
    seed_archive,
)


def construct_ant(n, adj_bits, tau, alpha, beta, q0, rng):
    """Build one elimination order under the pheromone+heuristic rule.

    ``tau`` is the (n, n) position-indexed pheromone matrix; ``rng`` is a
    numpy Generator.  Returns the order as a list of vertices.  The graph
    fill-in update mirrors ``grasp.construct_rcl`` exactly so the two
    constructions are comparable."""
    g = list(adj_bits)
    remaining = (1 << n) - 1
    order = []
    for i in range(n):
        # gather remaining vertices and their current (fill-in) degrees
        verts = []
        degs = []
        r = remaining
        while r:
            vbit = r & -r
            r ^= vbit
            v = vbit.bit_length() - 1
            verts.append(v)
            degs.append((g[v] & remaining).bit_count())
        eta = 1.0 / (1.0 + np.asarray(degs, dtype=np.float64))   # min-degree
        ph = tau[i, verts]
        weight = (ph ** alpha) * (eta ** beta)
        s = weight.sum()
        if not np.isfinite(s) or s <= 0.0:
            idx = rng.integers(len(verts))
        elif q0 > 0.0 and rng.random() < q0:
            idx = int(np.argmax(weight))            # ACS exploitation
        else:
            idx = int(rng.choice(len(verts), p=weight / s))
        best_v = verts[idx]
        order.append(best_v)
        remaining ^= 1 << best_v
        nbrs = g[best_v] & remaining
        sset = nbrs
        while sset:
            ubit = sset & -sset
            sset ^= ubit
            u = ubit.bit_length() - 1
            g[u] |= nbrs ^ ubit
    return order


def evaluate_order(archive, order, adj_bits, n, t_grid):
    """Offer ``order`` to the archive at every t-grid threshold; return the
    best feasible single-point dominated area found (0.0 if none feasible).
    The area doubles as the ant's deposit quality."""
    best_area = 0.0
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(order, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), order[:])
        if w < n and tt < n:
            area = (n - w) * (n - tt)
            if area > best_area:
                best_area = area
    return best_area


def descend(archive, perm, adj_bits, n, t_grid, max_steps, op_acc, op_att):
    """Step-bounded HV-improvement descent from ``perm``, feeding the global
    archive.  This is the *same* SMS-EMOA/IBEA criterion and the *same*
    15-operator move pool that GRASP and VNS use for local search -- a
    candidate is accepted iff it strictly increases the archive hypervolume --
    so the hybrid ACO's descent is directly comparable to the other chapters'.
    Unlike GRASP's wall-clock-bounded descent this one runs at most
    ``max_steps`` attempts so each ant gets an equal, budget-cheap polish.
    Returns the best feasible single-point dominated area found (the deposit
    quality for the polished order)."""
    seed_archive(archive, perm, adj_bits, n, t_grid)
    cur_w, cur_t, cur_perm = ensure_seeded(archive, perm, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    cur_hv = archive.hypervolume(n)
    for _ in range(max_steps):
        weights = [(op_acc[nm] + 1) / (op_att[nm] + 1) for nm, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
        cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                    cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1
        _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
            cand_perm, cand_t, adj_bits, n)
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            archive.try_add(int(cand_w), int(cand_t), cand_perm)
            op_acc[op_name] += 1
            cur_perm, cur_t = cand_perm, cand_t
            cur_bn_idx, cur_bn_mask = cand_bn_idx, cand_bn_mask
            cur_hv = archive.hypervolume(n)
    # best feasible area over the descent's settling point and its t-grid
    best_area = 0.0
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(cur_perm, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), cur_perm[:])
        if w < n and tt < n:
            area = (n - w) * (n - tt)
            if area > best_area:
                best_area = area
    return best_area


def run(problem, budget_s, seed, here, num_t_seeds=20, ants=10, alpha=1.0,
        beta=3.0, rho=0.1, q0=0.0, variant="as", stall_limit=8,
        local_search=False, ls_steps=150, algo="aco"):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    n, adj, adj_bits = load_problem(here, problem)
    area_max = float(n) * float(n)             # normaliser for quality in (0,1]

    print(f"\n=== aco (Ant Colony Optimization, variant={variant}) "
          f"-- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, t-grid = {num_t_seeds}, "
          f"ants = {ants}, alpha = {alpha}, beta = {beta}, rho = {rho}, "
          f"q0 = {q0}")
    print(f"local-search = {local_search}"
          + (f"  (HV descent, {ls_steps} steps/ant)" if local_search else
             "  (pure construction)"))
    print()

    t_grid = build_t_grid(n, num_t_seeds)
    archive = ParetoArchive()

    # position-indexed pheromone trail, initialised uniformly
    tau0 = 1.0
    tau = np.full((n, n), tau0, dtype=np.float64)
    tau_min = 0.0
    tau_max = float("inf")

    start = time.time()
    deadline = start + budget_s
    it = 0
    ants_built = 0
    best_q_global = 0.0          # best normalised quality ever (for MMAS bounds)
    best_hv = 0.0
    stall = 0

    while time.time() < deadline:
        iter_orders = []
        iter_quals = []
        ib_order = None
        ib_qual = 0.0
        for _ in range(ants):
            if time.time() >= deadline:
                break
            order = construct_ant(n, adj_bits, tau, alpha, beta, q0, rng)
            if local_search:
                # hybrid ACO: polish each ant with the shared HV descent so
                # the pheromone reinforces *locally optimal* orders (this is
                # what lets ACO clear the dense-graph feasibility wall).
                area = descend(archive, order, adj_bits, n, t_grid,
                               ls_steps, op_acc, op_att)
            else:
                area = evaluate_order(archive, order, adj_bits, n, t_grid)
            q = area / area_max                       # normalised to (0, 1]
            ants_built += 1
            iter_orders.append(order)
            iter_quals.append(q)
            if q > ib_qual:
                ib_qual = q
                ib_order = order
        if not iter_orders:
            break

        # --- pheromone update ----------------------------------------------
        tau *= (1.0 - rho)
        if variant == "mmas":
            # only the iteration-best ant deposits
            if ib_order is not None:
                for i, v in enumerate(ib_order):
                    tau[i, v] += ib_qual
            if ib_qual > best_q_global:
                best_q_global = ib_qual
            # MMAS trail bounds from the best-so-far quality
            if best_q_global > 0.0:
                tau_max = best_q_global / rho
                tau_min = tau_max / (2.0 * n)
                np.clip(tau, tau_min, tau_max, out=tau)
        else:   # Ant System: every ant deposits
            for order, q in zip(iter_orders, iter_quals):
                if q <= 0.0:
                    continue
                for i, v in enumerate(order):
                    tau[i, v] += q

        # --- stagnation handling (MMAS) ------------------------------------
        cur_hv = archive.hypervolume(n)
        improved = cur_hv > best_hv
        if improved:
            best_hv = cur_hv
            stall = 0
        else:
            stall += 1
        if variant == "mmas" and stall >= stall_limit and tau_max < float("inf"):
            tau.fill(tau_max)                          # smooth restart
            stall = 0

        it += 1
        print(f"  iter {it:>3d} | ants {ants_built:5d} | archive "
              f"{len(archive):3d} | score = {-cur_hv:>14,.0f} | "
              f"t = {time.time() - start:5.1f}s")

    out_path, nvec, score = finalize(archive, n, here, problem, algo)
    final_hv = -score
    print()
    print(f"Finished in {time.time() - start:.1f}s, iterations = {it}, "
          f"ants built = {ants_built}")
    print(f"Final archive size: {len(archive)}")
    print(f"Official score: {-final_hv:,.0f}")
    if target is not None:
        gap = -final_hv - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")
    print(f"\nWrote submission: {out_path}  ({nvec} vectors)")
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    ap.add_argument("--ants", type=int, default=10,
                    help="ants per iteration (colony size)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="pheromone exponent")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="heuristic (min-degree) exponent")
    ap.add_argument("--rho", type=float, default=0.1,
                    help="pheromone evaporation rate")
    ap.add_argument("--q0", type=float, default=0.0,
                    help="ACS pseudo-random-proportional exploitation prob "
                         "(0 = pure proportional selection)")
    ap.add_argument("--variant", choices=["as", "mmas"], default="as",
                    help="'as' = Ant System (baseline); "
                         "'mmas' = MAX-MIN Ant System (enhanced)")
    ap.add_argument("--stall-limit", type=int, default=8,
                    help="MMAS: iterations without improvement before a "
                         "smooth trail re-initialisation")
    ap.add_argument("--local-search", action="store_true",
                    help="hybrid ACO: polish each ant with the shared HV "
                         "descent (same move pool as GRASP/VNS) -- needed to "
                         "clear the dense-graph feasibility wall")
    ap.add_argument("--ls-steps", type=int, default=150,
                    help="max HV-descent attempts per ant (--local-search)")
    ap.add_argument("--algo", default="aco",
                    help="submission filename stem (override for tuning)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, repo_root(),
        num_t_seeds=args.num_t_seeds, ants=args.ants, alpha=args.alpha,
        beta=args.beta, rho=args.rho, q0=args.q0, variant=args.variant,
        stall_limit=args.stall_limit, local_search=args.local_search,
        ls_steps=args.ls_steps, algo=args.algo)


if __name__ == "__main__":
    main()

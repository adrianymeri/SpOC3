#!/usr/bin/env python3
"""
grasp.py -- Greedy Randomized Adaptive Search Procedure for the
Torso-Decomposition bi-objective.

References
----------
Feo, T.A. and Resende, M.G.C. (1995).  "Greedy randomized adaptive search
procedures."  Journal of Global Optimization 6(2): 109-133.

Resende, M.G.C. and Ribeiro, C.C. (2016).  "Optimization by GRASP: Greedy
Randomized Adaptive Search Procedures."  Springer.

Method
------
GRASP runs a sequence of independent restarts.  Each restart has two phases:

  1. **Greedy-randomized construction.**  An elimination order is built one
     vertex at a time using a minimum-degree criterion, but instead of always
     taking the lowest-degree vertex we form a Restricted Candidate List (RCL)
     of all vertices whose degree lies within
         [d_min,  d_min + alpha * (d_max - d_min)]
     and pick uniformly from it.  alpha = 0 is pure greedy (deterministic
     min-degree); alpha = 1 is fully random.  alpha thus tunes the
     construction's greediness/diversity trade-off.

  2. **Local search.**  The constructed order is improved with HV-improvement
     descent over the same 15-operator move pool used by the HC chapter: a
     candidate is accepted iff it strictly increases the archive hypervolume
     (the SMS-EMOA/IBEA criterion from hc9).

Every restart feeds the *same global Pareto archive*, so the reported score is
the HV of the union of all restarts -- directly comparable to every other
chapter.  The number of restarts is budget-driven (each gets an equal
wall-clock slice), with --restarts as an upper bound.

Hyperparameters exposed for tuning: --alpha, --restarts, plus the shared
--num-t-seeds.
"""

from __future__ import annotations

import argparse
import os
import random
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


def construct_rcl(n, adj_bits, alpha, rng):
    """Greedy-randomized minimum-degree elimination order with an
    alpha-controlled Restricted Candidate List."""
    g = list(adj_bits)
    remaining = (1 << n) - 1
    order = []
    for _ in range(n):
        degs = []
        r = remaining
        d_min = 1 << 60
        d_max = -1
        while r:
            vbit = r & -r
            r ^= vbit
            v = vbit.bit_length() - 1
            d = (g[v] & remaining).bit_count()
            degs.append((v, d))
            if d < d_min:
                d_min = d
            if d > d_max:
                d_max = d
        thresh = d_min + alpha * (d_max - d_min)
        rcl = [v for (v, d) in degs if d <= thresh]
        best_v = rng.choice(rcl)
        order.append(best_v)
        remaining ^= 1 << best_v
        nbrs = g[best_v] & remaining
        s = nbrs
        while s:
            ubit = s & -s
            s ^= ubit
            u = ubit.bit_length() - 1
            g[u] |= nbrs ^ ubit
    return order


def local_search(archive, perm, adj_bits, n, t_grid, deadline, rng_seed,
                 op_acc, op_att):
    """HV-improvement descent from `perm`, feeding the global archive.
    Returns (adds, final_perm, final_t): the number of archive additions
    made plus the local optimum the descent settled on (the latter is used
    to populate the elite pool for path-relinking)."""
    random.seed(rng_seed)
    seed_archive(archive, perm, adj_bits, n, t_grid)
    cur_w, cur_t, cur_perm = ensure_seeded(archive, perm, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, _ = evaluate_full(cur_perm, cur_t, adj_bits, n)
    cur_hv = archive.hypervolume(n)
    adds = 0
    while time.time() < deadline:
        weights = [(op_acc[nm] + 1) / (op_att[nm] + 1) for nm, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
        cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                    cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1
        _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
            cand_perm, cand_t, adj_bits, n)
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            if archive.try_add(int(cand_w), int(cand_t), cand_perm):
                adds += 1
            op_acc[op_name] += 1
            cur_perm = cand_perm
            cur_t = cand_t
            cur_bn_idx = cand_bn_idx
            cur_bn_mask = cand_bn_mask
            cur_hv = archive.hypervolume(n)
    return adds, list(cur_perm), cur_t


def path_relink(archive, start_perm, guide_perm, adj_bits, n, t_grid, deadline):
    """Permutation path-relinking (Glover 1997; Resende & Ribeiro 2016).

    Walks from ``start_perm`` toward ``guide_perm`` one position at a time:
    at each differing index i we swap the element currently at i with the one
    holding ``guide_perm[i]``, so each step is a single transposition that
    strictly reduces the Kendall distance to the guide.  Every intermediate
    permutation on the path is offered to the global archive at all t-grid
    thresholds (via ``seed_archive``), so any non-dominated decision lying
    *between* two good solutions is captured -- this is where PR adds value
    over independent restarts.  Returns the number of intermediate steps that
    grew the archive HV."""
    s = list(start_perm)
    pos = {v: i for i, v in enumerate(s)}
    adds = 0
    for i in range(n):
        if time.time() >= deadline:
            break
        gi = guide_perm[i]
        if s[i] == gi:
            continue
        j = pos[gi]
        s[i], s[j] = s[j], s[i]
        pos[s[i]] = i
        pos[s[j]] = j
        before = archive.hypervolume(n)
        seed_archive(archive, s, adj_bits, n, t_grid)
        if archive.hypervolume(n) > before:
            adds += 1
    return adds


REACTIVE_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5]


def run(problem, budget_s, seed, here, num_t_seeds=20, alpha=0.3,
        restarts=8, path_relinking=False, reactive_alpha=False,
        elite_size=5, algo="grasp"):
    random.seed(seed)
    np.random.seed(seed)

    n, adj, adj_bits = load_problem(here, problem)

    print(f"\n=== grasp (Greedy Randomized Adaptive Search) -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, t-grid = {num_t_seeds}, "
          f"alpha = {alpha}, max restarts = {restarts}")
    print(f"path-relinking = {path_relinking}, reactive-alpha = "
          f"{reactive_alpha}"
          + (f"  (alphas {REACTIVE_ALPHAS})" if reactive_alpha else ""))
    print()

    t_grid = build_t_grid(n, num_t_seeds)
    archive = ParetoArchive()
    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    # --- reactive-alpha bookkeeping (Prais & Ribeiro 2000) ------------------
    # Each alpha keeps a running mean of the archive-HV reached on restarts
    # that used it; selection probability is proportional to that mean, so
    # alphas that historically lead to better fronts are sampled more often.
    a_sum = {a: 0.0 for a in REACTIVE_ALPHAS}
    a_cnt = {a: 0 for a in REACTIVE_ALPHAS}

    def pick_alpha(rng):
        if not reactive_alpha:
            return alpha
        unseen = [a for a in REACTIVE_ALPHAS if a_cnt[a] == 0]
        if unseen:
            return rng.choice(unseen)
        means = [a_sum[a] / a_cnt[a] for a in REACTIVE_ALPHAS]
        lo = min(means)
        weights = [(m - lo) + 1e-9 for m in means]   # all >= 0, never all-zero
        return rng.choices(REACTIVE_ALPHAS, weights=weights, k=1)[0]

    # elite pool: best local optima seen, kept by single-point dominated area
    elites = []   # list of (area, perm)

    def offer_elite(perm, t):
        _, w, _, _, _ = evaluate_full(perm, t, adj_bits, n)
        if w >= n or t >= n:
            return
        area = (n - w) * (n - t)
        elites.append((area, list(perm)))
        elites.sort(key=lambda e: e[0], reverse=True)
        del elites[elite_size:]

    start = time.time()
    deadline = start + budget_s
    slice_s = budget_s / max(1, restarts)
    total_adds = 0
    pr_adds = 0
    done = 0

    for r in range(restarts):
        if time.time() >= deadline:
            break
        rng = random.Random(seed * 1000 + r)
        a_r = pick_alpha(rng)
        c_perm = construct_rcl(n, adj_bits, a_r, rng)
        ls_deadline = min(deadline, time.time() + slice_s)
        adds, lo_perm, lo_t = local_search(
            archive, c_perm, adj_bits, n, t_grid, ls_deadline,
            seed * 1000 + r, op_acc, op_att)
        total_adds += adds

        # path-relinking: relink the fresh local optimum with a random elite
        if path_relinking and elites and time.time() < deadline:
            guide = rng.choice(elites)[1]
            pr_deadline = min(deadline, time.time() + 0.25 * slice_s)
            d = path_relink(archive, lo_perm, guide, adj_bits, n, t_grid,
                            pr_deadline)
            pr_adds += d
            total_adds += d
        offer_elite(lo_perm, lo_t)

        if reactive_alpha:
            a_sum[a_r] += archive.hypervolume(n)
            a_cnt[a_r] += 1
        done += 1
        print(f"  restart {r:>3d} | alpha={a_r:<4} | adds {adds:5d} | "
              f"archive {len(archive):3d} | score = "
              f"{-archive.hypervolume(n):>14,.0f} | "
              f"t = {time.time() - start:5.1f}s")

    if reactive_alpha:
        print("\nReactive-alpha usage:")
        for a in REACTIVE_ALPHAS:
            mean = a_sum[a] / a_cnt[a] if a_cnt[a] else 0.0
            print(f"  alpha {a:<4}  used {a_cnt[a]:3d}x  "
                  f"mean archive-HV {mean:,.0f}")
    if path_relinking:
        print(f"\nPath-relinking archive growths: {pr_adds}")

    out_path, nvec, score = finalize(archive, n, here, problem, algo)
    final_hv = -score   # HV of the submitted top-20 front (== verify score)
    print()
    print(f"Finished in {time.time() - start:.1f}s, restarts = {done}, "
          f"archive adds = {total_adds}")
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
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="RCL greediness: 0 = pure greedy, 1 = fully random")
    ap.add_argument("--restarts", type=int, default=8,
                    help="upper bound on construction+local-search restarts")
    ap.add_argument("--path-relinking", action="store_true",
                    help="relink each local optimum with an elite (GRASP+PR)")
    ap.add_argument("--reactive-alpha", action="store_true",
                    help="adapt RCL alpha by historical front quality "
                         "(Prais & Ribeiro 2000)")
    ap.add_argument("--elite-size", type=int, default=5,
                    help="size of the elite pool used by path-relinking")
    ap.add_argument("--algo", default="grasp",
                    help="submission filename stem (override for tuning)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, repo_root(),
        num_t_seeds=args.num_t_seeds, alpha=args.alpha,
        restarts=args.restarts, path_relinking=args.path_relinking,
        reactive_alpha=args.reactive_alpha, elite_size=args.elite_size,
        algo=args.algo)


if __name__ == "__main__":
    main()

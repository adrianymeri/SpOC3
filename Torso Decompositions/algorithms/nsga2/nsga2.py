#!/usr/bin/env python3
"""
nsga2.py -- Population-based multi-objective evolutionary search for the
Torso-Decomposition bi-objective (NSGA-II, with an SMS-EMOA variant).

References
----------
Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002).  "A fast and
elitist multiobjective genetic algorithm: NSGA-II."  IEEE Transactions on
Evolutionary Computation 6(2): 182-197.

Beume, N., Naujoks, B. and Emmerich, M. (2007).  "SMS-EMOA: Multiobjective
selection based on dominated hypervolume."  EJOR 181(3): 1653-1669.

Knowles, J. and Corne, D. (2000).  "Approximating the nondominated front
using the Pareto archived evolution strategy."  Evolutionary Computation
8(2): 149-172.  (PAES -- the (1+1) archive-acceptance ancestor of SMS-EMOA,
already covered by the hc9 HV-acceptance hill-climber.)

Why this chapter
----------------
Every chapter so far has searched from a *single* working solution
(hill-climbing, SA, VNS) or a sequence of independent restarts (GRASP, ACO).
NSGA-II is the first genuinely *population-based* method: it keeps a set of N
decisions alive simultaneously and recombines them.  This tests two distinct
hypotheses against the leaderboard:

  1. **NSGA-II (default).**  Selection by non-dominated rank + crowding
     distance -- the canonical Pareto-EA.  Crowding distance is a *diversity*
     surrogate, NOT the HV indicator we actually score on.

  2. **SMS-EMOA (variant="sms").**  Steady-state (mu+1): the survivor culled
     each step is the one whose removal costs the least hypervolume.  This
     promotes the hc9 "accept iff HV improves" rule from an *acceptance*
     criterion to a *selection* criterion over a whole population -- the
     sharpest test of whether HV-guided selection beats GRASP's restart model.

Both variants reuse the shared substrate (meta_common): the identical 15
operators as *mutation*, the same Pareto archive, the same t-grid seeding,
the same -HV scoring and submission format, so they are directly comparable
to every prior chapter.  Recombination adds one new ingredient the move-pool
chapters lacked: **order crossover (OX)** on the elimination permutation.

Hyperparameters exposed for tuning: --pop, --pc (crossover prob),
--pm (mutation prob), --variant {nsga2,sms}, plus the shared --num-t-seeds.
"""

from __future__ import annotations

import argparse
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
    build_warm_start,
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


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------
class Individual:
    """One decision (perm, t) plus its objective vector and cached
    bottleneck masks (needed to drive the mutation operators)."""
    __slots__ = ("perm", "t", "w", "bn_idx", "bn_mask", "rank", "crowd")

    def __init__(self, perm, t, adj_bits, n):
        self.perm = perm
        self.t = int(t)
        _, w, bn_idx, bn_mask, _ = evaluate_full(perm, self.t, adj_bits, n)
        self.w = int(w)
        self.bn_idx = bn_idx
        self.bn_mask = bn_mask
        self.rank = 0
        self.crowd = 0.0

    def obj(self):
        # both objectives minimised
        return (self.w, self.t)


# ---------------------------------------------------------------------------
# Construction helpers for the initial population
# ---------------------------------------------------------------------------
def min_degree_order(n, adj_bits, rng, noise=0.0):
    """Greedy minimum-degree elimination order.  With noise>0 a random
    subset of decisions picks a random remaining vertex instead, giving a
    diversified-but-still-structured seed (cheap RCL-free perturbation)."""
    g = list(adj_bits)
    remaining = (1 << n) - 1
    order = []
    for _ in range(n):
        if noise and rng.random() < noise:
            # random pick among remaining
            r = remaining
            cnt = bin(remaining).count("1")
            k = rng.randrange(cnt)
            while k:
                r &= r - 1
                k -= 1
            best_v = (r & -r).bit_length() - 1
        else:
            r = remaining
            best_v = -1
            best_d = 1 << 60
            while r:
                vbit = r & -r
                r ^= vbit
                v = vbit.bit_length() - 1
                d = (g[v] & remaining).bit_count()
                if d < best_d:
                    best_d = d
                    best_v = v
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


def order_crossover(p1, p2, rng):
    """Order crossover (OX, Davis 1985) for permutations: copy a random
    contiguous slice from p1, then fill the remaining positions with the
    genes of p2 in their p2-order, skipping genes already taken.  Preserves
    relative order from the second parent while keeping an intact block of
    the first -- the standard recombination for permutation EAs."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    taken = set()
    for i in range(a, b + 1):
        child[i] = p1[i]
        taken.add(p1[i])
    fill = (g for g in p2 if g not in taken)
    for i in list(range(b + 1, n)) + list(range(0, a)):
        child[i] = next(fill)
    return child


# ---------------------------------------------------------------------------
# NSGA-II machinery: fast non-dominated sort + crowding distance
# ---------------------------------------------------------------------------
def dominates(a, b):
    """True iff objective vector a Pareto-dominates b (both minimised)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def fast_nondominated_sort(pop):
    """Deb's O(M N^2) fast non-dominated sort.  Assigns .rank in place and
    returns the list of fronts (each a list of Individuals)."""
    S = [[] for _ in pop]
    ndom = [0] * len(pop)
    fronts = [[]]
    objs = [ind.obj() for ind in pop]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return [[pop[idx] for idx in f] for f in fronts]


def crowding_distance(front):
    """Assign NSGA-II crowding distance (in .crowd) to a single front."""
    m = len(front)
    if m == 0:
        return
    for ind in front:
        ind.crowd = 0.0
    for obj_i in (0, 1):
        front.sort(key=lambda ind: ind.obj()[obj_i])
        lo = front[0].obj()[obj_i]
        hi = front[-1].obj()[obj_i]
        front[0].crowd = front[-1].crowd = float("inf")
        span = (hi - lo) or 1
        for k in range(1, m - 1):
            front[k].crowd += (front[k + 1].obj()[obj_i]
                               - front[k - 1].obj()[obj_i]) / span


def crowded_better(a, b):
    """NSGA-II crowded-comparison operator: lower rank wins; ties broken by
    larger crowding distance (prefer the less-crowded region)."""
    if a.rank != b.rank:
        return a.rank < b.rank
    return a.crowd > b.crowd


def tournament(pop, rng):
    a, b = rng.choice(pop), rng.choice(pop)
    return a if crowded_better(a, b) else b


# ---------------------------------------------------------------------------
# SMS-EMOA survivor selection: drop the least-HV-contributing member of the
# worst front
# ---------------------------------------------------------------------------
def hv_least_contributor(front, n):
    """Index (within `front`) of the member whose removal shrinks the
    front's 2-D hypervolume the least.  Used as the SMS-EMOA culling rule."""
    from core import hypervolume_2d
    pts = [ind.obj() for ind in front]
    full = hypervolume_2d(pts, n)
    worst_i = 0
    worst_loss = float("inf")
    for i in range(len(front)):
        sub = pts[:i] + pts[i + 1:]
        loss = full - hypervolume_2d(sub, n)
        if loss < worst_loss:
            worst_loss = loss
            worst_i = i
    return worst_i


# ---------------------------------------------------------------------------
# Offspring generation (shared by both variants)
# ---------------------------------------------------------------------------
def make_offspring(p1, p2, adj_bits, n, t_grid, pc, pm, rng):
    """One child from parents p1, p2: OX crossover (prob pc) then one
    mutation operator from the shared pool (prob pm).  t is inherited from a
    parent and occasionally re-seeded to a random t-grid value to keep the
    threshold objective mobile."""
    if rng.random() < pc:
        child_perm = order_crossover(p1.perm, p2.perm, rng)
    else:
        child_perm = list(rng.choice((p1, p2)).perm)
    child_t = rng.choice((p1.t, p2.t))
    if rng.random() < 0.15:
        child_t = rng.choice(t_grid)
    child = Individual(child_perm, child_t, adj_bits, n)
    if rng.random() < pm:
        op_name, op_fn = rng.choice(OPERATORS)
        m_perm, m_t = call_op(op_fn, child.perm, n, child.t,
                              child.bn_idx, child.bn_mask)
        child = Individual(m_perm, m_t, adj_bits, n)
    return child


def memetic_descent(archive, ind, adj_bits, n, steps, rng):
    """Brief HV-improvement descent on a single individual (the memetic /
    '_ls' variant).  Identical acceptance rule to GRASP/hc9: try operators
    from the shared 15-op pool, accept a move iff it strictly increases the
    *archive* hypervolume, and feed every improving move into the global
    archive.  This is the local-search component the pure Pareto-EA lacks --
    it is what actually pushes the torso width down, and it is exactly the
    ingredient that turned ACO from a -0 failure into a competitive front.
    Returns the descended individual (the local optimum the search settled
    on, used to replace the parent in the population)."""
    cur_perm = list(ind.perm)
    cur_t = ind.t
    cur_bn_idx, cur_bn_mask = ind.bn_idx, ind.bn_mask
    cur_hv = archive.hypervolume(n)
    for _ in range(steps):
        op_name, op_fn = rng.choice(OPERATORS)
        cand_perm, cand_t = call_op(op_fn, cur_perm, n, cur_t,
                                    cur_bn_idx, cur_bn_mask)
        _, cand_w, cand_bn_idx, cand_bn_mask, _ = evaluate_full(
            cand_perm, cand_t, adj_bits, n)
        new_hv = hv_with_candidate(archive.points(), (cand_w, cand_t), n)
        if new_hv > cur_hv:
            archive.try_add(int(cand_w), int(cand_t), cand_perm[:])
            cur_perm, cur_t = cand_perm, cand_t
            cur_bn_idx, cur_bn_mask = cand_bn_idx, cand_bn_mask
            cur_hv = archive.hypervolume(n)
    return Individual(cur_perm, cur_t, adj_bits, n)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(problem, budget_s, seed, here, num_t_seeds=20, pop=40,
        pc=0.9, pm=0.3, variant="nsga2", local_search=False, ls_steps=100,
        algo="nsga2", warm_start="auto"):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    n, adj, adj_bits = load_problem(here, problem)

    title = "SMS-EMOA (HV selection)" if variant == "sms" else "NSGA-II"
    if local_search:
        title = "memetic " + title + f" (+HV-descent x{ls_steps})"
    print(f"\n=== {title} -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, pop = {pop}, "
          f"pc = {pc}, pm = {pm}, t-grid = {num_t_seeds}, "
          f"variant = {variant}, local_search = {local_search}"
          + (f", ls_steps = {ls_steps}" if local_search else ""))
    print()

    t_grid = build_t_grid(n, num_t_seeds)
    archive = ParetoArchive()

    start = time.time()
    deadline = start + budget_s

    # --- initial population -------------------------------------------------
    # A spread of greedy / noisy-greedy / random orders, each at a random
    # t-grid threshold, all seeded into the global archive across the grid.
    # The single ELITE seed (i == 0) uses the shared build_warm_start: this
    # is min-fill on the affordable (small/medium) instances -- the project's
    # strongest warm start (docs/RESULTS.md 5.3) -- and auto-falls back to
    # min-degree on the dense large-graph.  The noisy/random seeds stay
    # min-degree for cheap diversity, so this change is purely additive: it
    # upgrades the one anchor the population is pulled toward.
    elite_perm, ws_label = build_warm_start(
        n, adj_bits, rng=random.Random(seed), method=warm_start)
    print(f"  elite warm start = {ws_label}")
    population = []
    for i in range(pop):
        if i == 0:
            perm = elite_perm[:]
        elif i < pop * 0.7:
            perm = min_degree_order(n, adj_bits, rng,
                                    noise=0.05 + 0.4 * rng.random())
        else:
            perm = list(range(n))
            rng.shuffle(perm)
        t0 = rng.choice(t_grid)
        ind = Individual(perm, t0, adj_bits, n)
        population.append(ind)
        seed_archive(archive, perm, adj_bits, n, t_grid)
        archive.try_add(ind.w, ind.t, ind.perm[:])
        if time.time() >= deadline:
            break

    fronts = fast_nondominated_sort(population)
    for f in fronts:
        crowding_distance(f)

    gen = 0
    evals = pop
    if variant == "sms":
        # --- steady-state (mu+1) loop --------------------------------------
        while time.time() < deadline:
            p1 = tournament(population, rng)
            p2 = tournament(population, rng)
            child = make_offspring(p1, p2, adj_bits, n, t_grid,
                                   pc, pm, rng)
            if local_search:
                child = memetic_descent(archive, child, adj_bits, n,
                                        ls_steps, rng)
            evals += 1
            archive.try_add(child.w, child.t, child.perm[:])
            population.append(child)
            fronts = fast_nondominated_sort(population)
            worst_front = fronts[-1]
            if len(worst_front) == 1:
                victim = worst_front[0]
            else:
                vi = hv_least_contributor(worst_front, n)
                victim = worst_front[vi]
            population.remove(victim)
            gen += 1
            if gen % 200 == 0:
                print(f"  iter {gen:>6d} | evals {evals:>7d} | "
                      f"archive {len(archive):3d} | score = "
                      f"{-archive.hypervolume(n):>14,.0f} | "
                      f"t = {time.time() - start:5.1f}s")
    else:
        # --- generational NSGA-II loop -------------------------------------
        while time.time() < deadline:
            offspring = []
            while len(offspring) < pop and time.time() < deadline:
                p1 = tournament(population, rng)
                p2 = tournament(population, rng)
                child = make_offspring(p1, p2, adj_bits, n, t_grid,
                                       pc, pm, rng)
                if local_search:
                    child = memetic_descent(archive, child, adj_bits, n,
                                            ls_steps, rng)
                evals += 1
                archive.try_add(child.w, child.t, child.perm[:])
                offspring.append(child)
            combined = population + offspring
            fronts = fast_nondominated_sort(combined)
            new_pop = []
            for f in fronts:
                crowding_distance(f)
                if len(new_pop) + len(f) <= pop:
                    new_pop.extend(f)
                else:
                    f.sort(key=lambda ind: ind.crowd, reverse=True)
                    new_pop.extend(f[:pop - len(new_pop)])
                    break
            population = new_pop
            gen += 1
            print(f"  gen {gen:>4d} | evals {evals:>7d} | "
                  f"archive {len(archive):3d} | score = "
                  f"{-archive.hypervolume(n):>14,.0f} | "
                  f"t = {time.time() - start:5.1f}s")

    # --- final harvest ------------------------------------------------------
    # Population selection (rank+crowding or HV) optimises which *orders* to
    # keep, but the archive only ever saw each survivor at its own single t.
    # Here we cash the discovered orders in across the entire threshold grid:
    # seeding every distinct final-population perm at all t-grid values exposes
    # each order's full (width, t) staircase.  This is a one-time O(mu * |t|)
    # cost and is exactly how GRASP earns the t-coverage of its front -- it is
    # what lifts the submitted hypervolume from the population's thin sample to
    # the dense Pareto staircase the scorer rewards.
    seen = set()
    harvested = 0
    for ind in population:
        key = tuple(ind.perm)
        if key in seen:
            continue
        seen.add(key)
        before = len(archive)
        seed_archive(archive, ind.perm, adj_bits, n, t_grid)
        harvested += len(archive) - before
    print(f"Final harvest: {len(seen)} distinct orders x {len(t_grid)} "
          f"thresholds -> +{harvested} archive points "
          f"(archive now {len(archive)})")

    out_path, nvec, score = finalize(archive, n, here, problem, algo)
    final_hv = -score
    print()
    print(f"Finished in {time.time() - start:.1f}s, "
          f"generations/iters = {gen}, evals = {evals}")
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
    ap.add_argument("--pop", type=int, default=40,
                    help="population size (mu)")
    ap.add_argument("--pc", type=float, default=0.9,
                    help="order-crossover probability")
    ap.add_argument("--pm", type=float, default=0.3,
                    help="mutation probability (shared 15-op pool)")
    ap.add_argument("--variant", default="nsga2", choices=["nsga2", "sms"],
                    help="nsga2 = rank+crowding; sms = SMS-EMOA HV selection")
    ap.add_argument("--local-search", action="store_true",
                    help="memetic: brief HV-descent on each offspring "
                         "(the '_ls' high-ceiling variant)")
    ap.add_argument("--ls-steps", type=int, default=100,
                    help="HV-descent operator trials per offspring")
    ap.add_argument("--warm-start", default="auto",
                    choices=["auto", "min-fill", "min-degree"],
                    help="elite seed construction (default auto: min-fill "
                         "where affordable, min-degree on dense large-graph)")
    ap.add_argument("--algo", default="nsga2",
                    help="submission filename stem (override for tuning)")
    args = ap.parse_args()
    run(args.problem, args.budget, args.seed, repo_root(),
        num_t_seeds=args.num_t_seeds, pop=args.pop, pc=args.pc, pm=args.pm,
        variant=args.variant, local_search=args.local_search,
        ls_steps=args.ls_steps, algo=args.algo, warm_start=args.warm_start)


if __name__ == "__main__":
    main()

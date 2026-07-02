#!/usr/bin/env python3
r"""
archive_evolve.py -- evolution where the INDIVIDUAL is a capped Pareto archive,
not a single ordering, and FITNESS is the exact competition objective.

Standard solvers optimise one elimination order pi and read HV(F(pi)).  But the
competition scores HV of the best <=20 points of the UNION of submitted orders.
So the right object to evolve is the archive A = {pi_1..pi_m}, with

        fitness(A) = HV( HSSP_20( union_i F(pi_i) ) )

i.e. the exact capped-20 hypervolume (core.top_k_by_hv_contribution is the exact
2-D HSSP DP).  This aligns the search 1:1 with the score, and -- unlike pooling
fronts after the fact -- it lets the search *generate* new orders judged by their
marginal contribution to the capped front, and recombine orders across archives.

Operators
  * seed   : the corpus's best orders (so we start at the current best capped HV)
  * mutate : generate a NEW order by perturbing a member around a visible width
             (op-relocate + min-fill repair) or a fresh min-fill restart, then
             keep it only if it raises the capped-20 HV -- generative, cap-aware
  * accept : strict improvement of the capped-20 HV (the exact score)

This is the reviewer's "evolve archives, not orderings" unified with the 20-point
cap discovery (THESIS s13.8a).  Additive: writes submissions/<problem>/
archive_evolve.json only on a verified capped-20 improvement.

    python3 tools/archive_evolve.py --problem medium-graph --iters 4000 \
        --pool 36 --seed 0
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, min_fill_in_perm)
from tools.fastwalk import IncEvalC


def order_front(perm, ev, n):
    """The (w, t) staircase points this ordering contributes (non-dominated)."""
    d = ev.full(perm); r = 0; pts = []
    a = ParetoArchive()
    for t in range(n - 1, -1, -1):
        c = int(d[t]); r = c if c > r else r
        if r <= MAX_TW: a.try_add(r, t, perm)
    return [(w, t) for (w, t, _) in a.entries()]


def capped_hv(members, n, k=20):
    """Exact capped-k competition score over the union of member fronts."""
    arc = ParetoArchive()
    for perm, pts in members:
        for (w, t) in pts:
            arc.try_add(w, t, perm)
    top = arc.top_k_by_hv_contribution(k, n)
    return -hypervolume_2d([(w, t) for (w, t, _) in top], n), top


def load_seed_orders(here, problem, n, ev, cap):
    """Distinct corpus orderings + their fronts, best (most band-wins) first."""
    seen = {}; arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = tuple(int(x) for x in dv[:-1])
                    if perm not in seen:
                        pts = order_front(list(perm), ev, n); seen[perm] = pts
                        for (w, t) in pts: arc.try_add(w, t, list(perm))
        except Exception:
            continue
    # MUST include the orderings that own the global HSSP-optimal 20 points, so
    # the pool starts exactly at the corpus's best valid capped-20 score.
    core = top_k_owners(arc, n)
    # then rank the rest by how many non-dominated bands they own (diversity)
    own = {}; by_w = {}
    for w, t, p in arc.entries():
        key = tuple(p)
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, key)
    for w, (t, key) in by_w.items():
        own[key] = own.get(key, 0) + 1
    ranked = [k for k, _ in sorted(seen.items(), key=lambda kv: -own.get(kv[0], 0))]
    out = []
    picked = set()
    for key in list(core) + ranked:
        if key in picked or key not in seen:
            continue
        picked.add(key); out.append((list(key), seen[key]))
        if len(out) >= max(cap, len(core)):
            break
    return out


def top_k_owners(arc, n, k=20):
    """Distinct orderings (as tuples) that own the HSSP-optimal k points."""
    return [tuple(p) for (_, _, p) in arc.top_k_by_hv_contribution(k, n)]


# --------------------------------------------------------------------------- #
# generative mutation: produce a NEW ordering that may grow a visible-width torso
# --------------------------------------------------------------------------- #
def crossover(p1, p2, n, rng):
    """Order-crossover: take a prefix from p1, fill the rest in p2's relative
    order.  Produces a NEW valid ordering blending two elimination strategies --
    the operator that makes archive evolution generative rather than local."""
    cut = rng.randint(n // 6, 5 * n // 6)
    head = p1[:cut]; hs = set(head)
    tail = [v for v in p2 if v not in hs]
    return head + tail


def twin_classes(n, adj):
    """True-twin classes (identical closed neighbourhoods). True twins induce a
    graph automorphism, so permuting them leaves every suffix width invariant --
    they are interchangeable UNITS. medium: 73% of vertices; large: 34% incl. a
    375-clique class at deg 499."""
    import collections
    h = collections.defaultdict(list)
    for v in range(n):
        h[frozenset(adj[v] | {v})].append(v)
    return [c for c in h.values() if len(c) > 1]


def twin_move(perm, classes, n, rng):
    """Coordinated multi-vertex move: relocate an entire twin class (or a random
    contiguous chunk of it) as ONE block. Single-vertex moves are provably frozen
    on the mature fronts; class-blocks are the symmetry-licensed moves that
    single-vertex search structurally lacks."""
    c = classes[rng.randrange(len(classes))]
    k = len(c) if len(c) <= 8 or rng.random() < 0.5 else rng.randint(2, len(c))
    cs = set(rng.sample(c, k))
    pos = [i for i, v in enumerate(perm) if v in cs]
    block = [perm[i] for i in pos]
    p = [v for v in perm if v not in cs]
    j = rng.choice([min(pos), max(pos) - len(block) + 1,
                    rng.randint(0, len(p)), rng.randint(len(p) // 2, len(p))])
    j = max(0, min(j, len(p)))
    return p[:j] + block + p[j:]


def mutate(perm, n, ab, rng, vis_widths, mate=None, classes=None):
    """Generate a NEW ordering by (a) order-crossover with a mate, (b) a short
    block relocation toward the torso side, or (c) a fresh randomised min-fill
    restart -- the three together give recombination + local + diversity."""
    r = rng.random()
    if classes and r < 0.30:
        return twin_move(list(perm), classes, n, rng)
    if mate is not None and r < 0.5:
        return crossover(list(perm), list(mate), n, rng)
    if r < 0.7:
        return [int(x) for x in min_fill_in_perm(n, ab, sample_size=48,
                                                 rng=random.Random(rng.randint(0, 1 << 30)))]
    p = list(perm)
    L = rng.randint(1, 4)
    i = rng.randint(0, n - L - 1)
    block = p[i:i + L]; del p[i:i + L]
    j = rng.randint(i, len(p))
    p[j:j] = block
    return p


def run(problem, here, iters, pool_cap, seed, twins=False):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    rng = random.Random(seed); nprng = np.random.default_rng(seed)

    classes = None
    if twins:
        classes = twin_classes(n, adj)
        cov = sum(len(c) for c in classes)
        print(f"    twin moves ON: {len(classes)} true-twin classes, "
              f"{cov}/{n} vertices ({100*cov//n}%)", flush=True)

    members = load_seed_orders(here, problem, n, ev, pool_cap)
    cur, top = capped_hv(members, n)
    vis = sorted(w for (w, _, _) in top)
    print(f"=== archive-evolve {problem} | seeded {len(members)} orders | "
          f"capped-20 {cur:,.0f}{f'  gap {cur-target:+,.0f}' if target else ''} ===",
          flush=True)
    print(f"    visible widths: {vis}", flush=True)

    t0 = time.time(); accepts = 0
    for it in range(iters):
        src = members[rng.randrange(len(members))][0]
        mate = members[rng.randrange(len(members))][0]
        mp = mutate(src, n, ab, rng, vis, mate, classes)
        pts = order_front(mp, ev, n)
        trial = members + [(mp, pts)]
        h, top = capped_hv(trial, n)
        if h < cur - 0.5:                         # strict capped-20 improvement
            cur = h; accepts += 1
            # keep the pool lean: retain only orders that own a selected point
            keep_perms = set(tuple(p) for (_, _, p) in top)
            members = [(p, q) for (p, q) in trial if tuple(p) in keep_perms]
            if (mp, pts) not in members: members.append((mp, pts))
            vis = sorted(w for (w, _, _) in top)
            print(f"  it {it}: *** capped-20 {cur:,.0f}"
                  f"{f'  gap {cur-target:+,.0f}' if target else ''} "
                  f"(accept #{accepts}, pool {len(members)}) ***", flush=True)
            # save additively on every improvement
            dvs = [list(p) + [int(t)] for (_, t, p) in top]
            out = os.path.join(here, "submissions", problem,
                               f"archive_evolve_s{seed}{'_tw' if twins else ''}.json")
            json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
                       "decisionVector": dvs}, open(out, "w"))
        if it % 500 == 0 and it:
            print(f"  [it {it}/{iters}  capped-20 {cur:,.0f}  {accepts} accepts  "
                  f"{time.time()-t0:.0f}s]", flush=True)
    print(f"\nfinal capped-20 {cur:,.0f}"
          f"{f'  gap {cur-target:+,.0f}' if target else ''} | {accepts} improvements",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--pool", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twins", action="store_true",
                    help="enable true-twin class-block moves (symmetry-licensed multi-vertex moves)")
    a = ap.parse_args()
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        a.iters, a.pool, a.seed, a.twins)


if __name__ == "__main__":
    main()

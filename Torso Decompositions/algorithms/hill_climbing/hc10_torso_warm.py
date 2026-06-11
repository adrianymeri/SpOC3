#!/usr/bin/env python3
"""
hc10_torso_warm.py -- Per-t torso-aware warm start.

References
----------
Bodlaender, H. L., Koster, A. M. C. A. and van der Hoeven, F. (2006).
"Treewidth: Computational Experiments."  Electronic Notes in Discrete
Mathematics 8: 54-57.

Seidman, S. B. (1983).  "Network structure and minimum degree."
Social Networks 5(3): 269-287.  (k-core decomposition)

Difference from hc3 / hc4
-------------------------
hc3 onwards builds ONE min-degree elimination order and evaluates it
across a t-grid.  But the same permutation is being reused for every
t -- a vertex placed near position 200 (because it's early in the
min-degree peeling) stays at position 200 whether t = 50 or t = 1200.
The "missing middle of the front" identified in `results.md §4` is a
direct consequence: intermediate-t points all share the same global
permutation, so the front there is whatever a single permutation can
give.

hc10 builds a DIFFERENT warm-start permutation per t-grid point:

    For each t in the grid:
        head = the t vertices with the HIGHEST original degree
        torso = the n - t vertices with the LOWEST original degree
        head_perm = min-degree elimination on the head subgraph
        torso_perm = min-degree continuation on the (full graph
                     with the head already eliminated)

The intuition: high-degree vertices are "harder" -- they will
generate more fill-in when eliminated.  Push them into the head so
they happen early, while the fill-in graph is still relatively clean.
Save the low-degree vertices for the torso, where their few
neighbours mean small fill-in degree at the time they are processed.

This is not a tight lower-bound argument -- it's a heuristic.  But
it produces a *different* permutation for every t, which is what we
need to fill the gap in the middle of the front.

The local-search phase is otherwise identical to hc4 (11 operators,
adaptive selection, Pareto archive).
"""

from __future__ import annotations

# --- sys.path bootstrap (added by restructure) ---
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(_os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

import argparse
import os
import random
import time
from typing import Callable, List, Tuple

import numpy as np

from core import (
    repo_root,
    ensure_seeded,
    LEADERBOARD_TARGETS,
    ParetoArchive,
    build_adj_bitsets,
    evaluate_full,
    graph_path,
    load_graph,
    submission_path,
    write_submission,
)


# ---------------------------------------------------------------------------
# Operators (identical to hc4)
# ---------------------------------------------------------------------------

def op_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i, j = random.sample(range(n), 2)
    out[i], out[j] = out[j], out[i]; return out, t

def op_adjacent_swap(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i = random.randrange(n - 1)
    out[i], out[i + 1] = out[i + 1], out[i]; return out, t

def op_insert(perm, n, t, bn_idx, bn_mask):
    out = perm[:]
    i, j = random.randrange(n), random.randrange(n)
    if i == j: return out, t
    out.insert(j, out.pop(i)); return out, t

def op_reverse(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; i, j = sorted(random.sample(range(n), 2))
    out[i:j + 1] = reversed(out[i:j + 1]); return out, t

def op_3opt(perm, n, t, bn_idx, bn_mask):
    """3-opt double-reverse: pick i < j < k and simultaneously reverse
    perm[i:j] AND perm[j:k].  A genuinely 3-opt neighbour."""
    if n < 6:
        return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    i = random.randint(0, n - 5)
    j = random.randint(i + 2, n - 3)
    k = random.randint(j + 2, n - 1)
    out[i:j] = out[i:j][::-1]
    out[j:k] = out[j:k][::-1]
    return out, t

def op_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; bs = random.randint(2, max(3, n // 20))
    s = random.randint(0, n - bs); block = out[s:s + bs]; del out[s:s + bs]
    ip = random.randint(0, len(out)); out[ip:ip] = block; return out, t

def op_long_block_move(perm, n, t, bn_idx, bn_mask):
    out = perm[:]; bs = random.randint(max(3, n // 50), max(4, n // 5))
    bs = min(bs, n - 1); s = random.randint(0, n - bs)
    block = out[s:s + bs]; del out[s:s + bs]
    ip = random.randint(0, len(out)); out[ip:ip] = block; return out, t

def op_bottleneck_to_head(perm, n, t, bn_idx, bn_mask):
    if bn_idx < 0 or t <= 0:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]
    if random.random() < 0.7 or bn_mask == 0:
        tv = out[bn_idx]; del out[bn_idx]
    else:
        bits = []; m = bn_mask
        while m:
            b = m & -m; bits.append(b.bit_length() - 1); m ^= b
        v = random.choice(bits); idx = out.index(v); tv = v; del out[idx]
    np_ = random.randint(0, max(0, t - 1)); out.insert(np_, tv); return out, t

def op_swap_head_tail(perm, n, t, bn_idx, bn_mask):
    if t <= 0 or t >= n:
        return op_swap(perm, n, t, bn_idx, bn_mask)
    out = perm[:]; i = random.randrange(0, t); j = random.randrange(t, n)
    out[i], out[j] = out[j], out[i]; return out, t

def op_2opt_torso(perm, n, t, bn_idx, bn_mask):
    if t >= n - 2: return op_reverse(perm, n, t, bn_idx, bn_mask)
    out = perm[:]; i = random.randint(t, n - 2); j = random.randint(i + 1, n - 1)
    out[i:j + 1] = reversed(out[i:j + 1]); return out, t

def op_t_shift(perm, n, t, bn_idx, bn_mask):
    span = max(1, n // 20)
    return perm[:], max(0, min(n - 1, t + random.randint(-span, span)))

def op_t_random(perm, n, t, bn_idx, bn_mask):
    return perm[:], random.randint(0, n - 1)


OPERATORS: List[Tuple[str, Callable]] = [
    ("swap", op_swap), ("adjacent_swap", op_adjacent_swap),
    ("insert", op_insert), ("reverse", op_reverse),
    ("3opt",              op_3opt),
    ("block_move", op_block_move), ("long_block_move", op_long_block_move),
    ("bottleneck->head", op_bottleneck_to_head),
    ("swap_head_tail", op_swap_head_tail), ("2opt_torso", op_2opt_torso),
    ("t_shift", op_t_shift), ("t_random", op_t_random),
]


def _call_op(op_fn, perm, n, t, bn_idx, bn_mask):
    out = op_fn(perm, n, t, bn_idx, bn_mask)
    return out if isinstance(out, tuple) and len(out) == 2 else (out, t)


# ---------------------------------------------------------------------------
# Torso-aware warm start
# ---------------------------------------------------------------------------

def min_degree_on_set(adj_bits, n, vertex_set, rng):
    """Run min-degree elimination on the induced subgraph defined by
    `vertex_set`.  Returns the elimination order, length |vertex_set|."""
    if not vertex_set:
        return []
    remaining_mask = 0
    for v in vertex_set:
        remaining_mask |= 1 << v
    g = list(adj_bits)
    # Restrict each adjacency to the set.
    for v in vertex_set:
        g[v] = g[v] & remaining_mask
    order: List[int] = []
    remaining = remaining_mask
    while remaining:
        best_d = 10 ** 18
        ties: List[int] = []
        r = remaining
        while r:
            vbit = r & -r
            r ^= vbit
            v = vbit.bit_length() - 1
            d = (g[v] & remaining).bit_count()
            if d < best_d:
                best_d = d
                ties = [v]
                if d == 0 and rng is None:
                    break
            elif d == best_d:
                ties.append(v)
        v = rng.choice(ties) if rng is not None else ties[0]
        order.append(v)
        remaining ^= 1 << v
        nbrs = g[v] & remaining
        s = nbrs
        while s:
            ubit = s & -s
            s ^= ubit
            u = ubit.bit_length() - 1
            g[u] |= nbrs ^ ubit
    return order


def torso_aware_warm(n, adj, adj_bits, t, rng):
    """Build a (perm, t) warm start tailored to this t.

    head  = t vertices with the HIGHEST original degree (in the input
            graph G), in min-degree-on-head-subgraph order.
    torso = n - t vertices with the LOWEST original degree, in
            min-degree continuation order.
    """
    if t <= 0:
        return min_degree_on_set(adj_bits, n, list(range(n)), rng)
    if t >= n:
        return min_degree_on_set(adj_bits, n, list(range(n)), rng)

    deg = [(len(adj[v]), v) for v in range(n)]
    # high-degree first
    deg.sort(key=lambda kv: (-kv[0], kv[1]))
    head_set = [v for _, v in deg[:t]]
    torso_set = [v for _, v in deg[t:]]

    head_perm = min_degree_on_set(adj_bits, n, head_set, rng)

    # For the torso, we want the order they'd be eliminated AFTER the head
    # has been processed (with fill-in cascading into the torso vertices).
    # Approximation: just min-degree the torso among themselves -- the
    # subsequent HC moves will fix this.
    torso_perm = min_degree_on_set(adj_bits, n, torso_set, rng)

    return head_perm + torso_perm


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, progress_every, here, num_t_seeds=20):
    random.seed(seed)
    np.random.seed(seed)

    gr = graph_path(here, problem)
    n, adj = load_graph(gr)
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc10_torso_warm -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}, "
          f"t-grid = {num_t_seeds}, operators = {len(OPERATORS)}")
    print("warm start = torso-aware (high-deg head, low-deg torso)")
    print()

    t_grid = sorted({int(round(i * (n - 1) / (num_t_seeds - 1)))
                     for i in range(num_t_seeds)}) if num_t_seeds > 1 else [0]

    md_rng = random.Random(seed)
    archive = ParetoArchive()
    warm_t0 = time.time()
    perms_built = 0
    for tt in t_grid:
        perm = torso_aware_warm(n, adj, adj_bits, tt, md_rng)
        perms_built += 1
        _, w, _, _, _ = evaluate_full(perm, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), perm)
    print(f"  built {perms_built} torso-aware permutations in "
          f"{time.time() - warm_t0:.1f}s")
    print(f"  seeded archive with {len(archive)} non-dominated points "
          f"(initial score = {-archive.hypervolume(n):,.0f})")

    cur_w, cur_t, cur_perm = ensure_seeded(archive, perm, adj_bits, n)
    cur_perm = list(cur_perm)
    _, _, cur_bn_idx, cur_bn_mask, cur_soft = evaluate_full(
        cur_perm, cur_t, adj_bits, n)

    op_acc = {name: 0 for name, _ in OPERATORS}
    op_att = {name: 0 for name, _ in OPERATORS}

    deadline = time.time() + budget_s
    iters = 0
    archive_adds = 0
    last_print = 0

    while time.time() < deadline:
        iters += 1
        weights = [(op_acc[n_] + 1) / (op_att[n_] + 1) for n_, _ in OPERATORS]
        op_name, op_fn = random.choices(OPERATORS, weights=weights, k=1)[0]
        cand_perm, cand_t = _call_op(op_fn, cur_perm, n, cur_t,
                                     cur_bn_idx, cur_bn_mask)
        op_att[op_name] += 1
        _, cand_w, cand_bn_idx, cand_bn_mask, cand_soft = evaluate_full(
            cand_perm, cand_t, adj_bits, n)
        if archive.try_add(int(cand_w), int(cand_t), cand_perm):
            archive_adds += 1
            op_acc[op_name] += 1
        if (cand_w < cur_w) or (cand_w == cur_w and cand_soft < cur_soft):
            cur_perm = cand_perm; cur_t = cand_t; cur_w = cand_w
            cur_soft = cand_soft
            cur_bn_idx = cand_bn_idx; cur_bn_mask = cand_bn_mask

        if iters - last_print >= progress_every:
            elapsed = budget_s - (deadline - time.time())
            print(f"  iter {iters:>8d} | archive {len(archive):3d} | "
                  f"adds {archive_adds:5d} | focus(w={cur_w}, t={cur_t}) | "
                  f"score = {-archive.hypervolume(n):>14,.0f} | "
                  f"t = {elapsed:5.1f}s")
            last_print = iters

    elapsed_total = budget_s - (deadline - time.time())
    print()
    print(f"Finished in {elapsed_total:.1f}s, iters = {iters:,}, "
          f"archive adds = {archive_adds}")
    print(f"Final archive size: {len(archive)}")
    final_hv = archive.hypervolume(n)
    print(f"Official score: {-final_hv:,.0f}")
    if target is not None:
        gap = -final_hv - target
        print(f"Gap to target ({target:,}): {gap:>+14,.0f}  "
              f"({'BEAT' if gap < 0 else f'{abs(gap):,} short'})")

    print(f"\nOperator stats:")
    for name, _ in OPERATORS:
        att = op_att[name]
        rate = op_acc[name] / att if att else 0
        print(f"  {name:18s}  {op_acc[name]:5d} / {att:6d}   ({rate:6.2%})")

    top = archive.top_k_by_hv_contribution(20, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, "hc10")
    write_submission(decision_vectors, problem, out_path)
    print(f"\nWrote submission: {out_path}  ({len(decision_vectors)} vectors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=38.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--num-t-seeds", type=int, default=20)
    args = ap.parse_args()
    here = repo_root()
    run(args.problem, args.budget, args.seed, args.progress_every, here,
        num_t_seeds=args.num_t_seeds)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
breakpoint_nudge.py  --  Targeted local search to push a specific breakpoint earlier.

The pooled staircase has t*(w) fixed for each w.  GBFC++ can't move them.
This script tries something different:

  For each breakpoint w, take the best ordering achieving that step and run a
  focused swap search in a NARROW WINDOW around the breakpoint:
    - Swap pairs of nodes in positions [t*(w)-K, t*(w)+K]
    - Accept any swap that decreases t*(w) by ≥ 1

Each 1-unit improvement = +1 HV.  We need 180 total.

Additionally checks:
  - EXACT treewidth lower bound of the suffix at t*(w)-1  (is improvement even possible?)
  - How far t*(w) can move based on graph structure

Usage:
    cd "Torso Decompositions"
    python3 tools/breakpoint_nudge.py --problem small-graph --width 14 --window 50
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, time, random
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  hypervolume_2d, load_decision_vectors, submission_path,
                  write_submission, LEADERBOARD_TARGETS, treewidth_lower_bound_mmd)


# ── staircase / evaluation ────────────────────────────────────────────────────

def staircase(perm, n, ab):
    tmp = list(ab); deg = np.zeros(n, dtype=np.int32)
    cur = 0; sm = [0] * n
    for i in range(n - 1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return np.maximum.accumulate(deg[::-1])[::-1]


def pareto_front(W, n):
    front = []; prev_t = n
    for w in range(n):
        idx = np.where(W <= w)[0]
        if not len(idx): continue
        t = int(idx[0])
        if t < prev_t: front.append((w, t)); prev_t = t
        if t == 0: break
    return front


def pooled_W(perms, n, ab):
    W = np.full(n, n, dtype=np.int64)
    for p in perms: W = np.minimum(W, staircase(p, n, ab))
    return W


# ── exact lower bound for a suffix ───────────────────────────────────────────

def suffix_lb(perm, t, n, adj):
    """Lower bound on treewidth of the INDUCED subgraph on perm[t..n-1]."""
    nodes = set(perm[t:])
    m = n - t
    sub = [0] * m
    idx = {v: i for i, v in enumerate(perm[t:])}
    for v in perm[t:]:
        for u in adj[v]:
            if u in nodes:
                sub[idx[v]] |= 1 << idx[u]
    return treewidth_lower_bound_mmd(m, sub)


# ── load best ordering achieving a specific width ─────────────────────────────

def load_pool(here, problem, n, ab):
    sub = os.path.join(here, "submissions", problem)
    fps = ([os.path.join(sub, "portfolio.json")]
           + sorted(glob.glob(os.path.join(sub, "*.json")))
           + sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))))
    perms = []; seen = set()
    for fp in fps:
        if not os.path.exists(fp): continue
        dvs = load_decision_vectors(fp)
        if not dvs: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = tuple(int(x) for x in dv[:-1])
                if len(p) == n and sorted(p) == list(range(n)) and p not in seen:
                    seen.add(p); perms.append(list(p))
    return perms


# ── targeted swap search around a breakpoint ─────────────────────────────────

def nudge_breakpoint(perm, target_w, target_t, n, ab, adj, window, budget_s, rng):
    """
    Try random swaps in window [target_t - window, target_t + window].
    Accept any swap that moves t*(target_w) earlier.
    Returns (best_perm, best_t_star).
    """
    best = list(perm)
    W = staircase(best, n, ab)
    best_t = int(np.where(W <= target_w)[0][0]) if (W <= target_w).any() else n
    lo = max(0, target_t - window)
    hi = min(n - 1, target_t + window)

    t0 = time.time()
    iters = 0; improvements = 0
    current = list(best)

    print(f"  Nudging w={target_w}: t*={target_t}, search window [{lo},{hi}], "
          f"budget={budget_s:.0f}s")

    while time.time() - t0 < budget_s:
        i = rng.randint(lo, hi); j = rng.randint(lo, hi)
        if i == j: continue
        # swap
        current[i], current[j] = current[j], current[i]
        W_new = staircase(current, n, ab)
        new_t = int(np.where(W_new <= target_w)[0][0]) if (W_new <= target_w).any() else n

        if new_t < best_t:
            best = list(current); best_t = new_t; improvements += 1
            print(f"    iter {iters:6d} | t*({target_w}) improved: {target_t} → {best_t} "
                  f"(Δ={target_t - best_t:+d}) | t={time.time()-t0:.1f}s", flush=True)
        else:
            # revert
            current[i], current[j] = current[j], current[i]
        iters += 1

    print(f"  Done: {iters} swaps, {improvements} improvements, "
          f"t*({target_w}): {target_t} → {best_t} (Δ={target_t - best_t:+d})")
    return best, best_t


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--width", type=int, default=None,
                    help="Target width level to nudge (default: all steps).")
    ap.add_argument("--window", type=int, default=100,
                    help="Half-window of positions to swap within.")
    ap.add_argument("--budget", type=float, default=120.0,
                    help="Time budget per breakpoint (seconds).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target_hv = -LEADERBOARD_TARGETS[args.problem]
    rng = random.Random(args.seed)

    print(f"\n{'='*60}")
    print(f"  BREAKPOINT NUDGE  --  {args.problem}")
    print(f"  Target HV = {target_hv:,}")
    print(f"{'='*60}\n")

    perms = load_pool(here, args.problem, n, ab)
    print(f"Loaded {len(perms)} orderings.\n")

    # Current pooled staircase
    W_pooled = pooled_W(perms, n, ab)
    front = pareto_front(W_pooled, n)
    our_hv = hypervolume_2d(front, n)
    print(f"Pooled HV = {our_hv:,.0f}  (gap = {target_hv - our_hv:.0f})\n")

    # Steps to try
    steps = front if args.width is None else [s for s in front if s[0] == args.width]

    print("=== Lower bound check (induced subgraph at t*(w)-1) ===")
    print(f"{'w':>5}  {'t*(w)':>8}  {'lb at t-1':>12}  {'improvable?':>14}")
    print("-" * 50)
    for (w, t) in steps:
        if t == 0:
            print(f"{w:>5}  {t:>8}  {'N/A':>12}  {'already 0':>14}")
            continue
        lb = suffix_lb(perms[0], t - 1, n, adj)   # lower bound on suffix tw
        improvable = "YES" if lb <= w else f"NO (lb={lb}>{w})"
        print(f"{w:>5}  {t:>8}  {lb:>12}  {improvable:>14}")
    print()

    # Nudge each step
    new_perms = []
    for (w, t) in steps:
        if t == 0: continue
        # best ordering for this width band
        best_in_band = min(perms, key=lambda p: staircase(p, n, ab)[t])
        new_perm, new_t = nudge_breakpoint(
            best_in_band, w, t, n, ab, adj, args.window, args.budget, rng)
        if new_t < t:
            new_perms.append(new_perm)

    # Re-evaluate pooled staircase with new orderings
    if new_perms:
        all_perms = perms + new_perms
        W_new = pooled_W(all_perms, n, ab)
        new_front = pareto_front(W_new, n)
        new_hv = hypervolume_2d(new_front, n)
        gain = new_hv - our_hv
        print(f"\nNew pooled HV = {new_hv:,.0f}  (Δ = {gain:+.0f}, "
              f"gap = {target_hv - new_hv:.0f})")
        if gain > 0:
            t_grid = list(range(0, n, max(1, n // 40))) + [n - 1]
            dvs = []
            for p in new_perms:
                for t in t_grid:
                    dvs.append(list(p) + [t])
            out = submission_path(here, args.problem, "nudge")
            write_submission(dvs, args.problem, out)
            print(f"Saved → {out}")
    else:
        print("\nNo breakpoints improved.")


if __name__ == "__main__":
    main()

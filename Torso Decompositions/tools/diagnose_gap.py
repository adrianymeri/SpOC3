#!/usr/bin/env python3
"""
diagnose_gap.py  --  Explain the 180 HV gap on small-graph.

For each width level w ∈ {0..MAX_W}, answers:
  "At what threshold t*(w) do we FIRST achieve width ≤ w?"

Then shows:
  - The exact Pareto front (t*, w) pairs
  - HV contribution of each step
  - What t*(w) would need to shift to to gain +1 .. +N HV
  - Which step(s) plausibly account for the full 180-HV gap

Usage:
    cd "Torso Decompositions"
    python3 tools/diagnose_gap.py --problem small-graph
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  hypervolume_2d, load_decision_vectors, LEADERBOARD_TARGETS)


# ── low-level width staircase ─────────────────────────────────────────────────

def staircase_for_perm(perm, n, ab):
    """W[t] = treewidth-upper-bound of suffix perm[t..n-1]."""
    tmp = list(ab)
    deg = np.zeros(n, dtype=np.int32)
    cur = 0
    suffix_mask = [0] * n
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]
    for i in range(n):
        s = tmp[perm[i]] & suffix_mask[i]
        deg[i] = s.bit_count()
        x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length() - 1; tmp[v] |= s ^ b
    return np.maximum.accumulate(deg[::-1])[::-1]   # suffix-max


def pooled_staircase(perms, n, ab):
    """Element-wise min across all orderings  (best achievable at each t)."""
    W = np.full(n, n, dtype=np.int64)
    for p in perms:
        W = np.minimum(W, staircase_for_perm(p, n, ab))
    return W


# ── Pareto front from staircase ───────────────────────────────────────────────

def pareto_front(W, n):
    """
    Extract Pareto-optimal (width, threshold) pairs from the pooled staircase.
    t*(w) = first index where W[t] <= w.
    A step is Pareto-optimal iff t*(w) < t*(w-1)  (strictly earlier).
    """
    front = []
    prev_t = n           # sentinel: nothing achieved yet
    for w in range(n):
        where = np.where(W <= w)[0]
        if len(where) == 0:
            continue
        t_star = int(where[0])
        if t_star < prev_t:
            front.append((w, t_star))   # (width, threshold)  -- hypervolume_2d format
            prev_t = t_star
        if t_star == 0:
            break
    return front          # ascending by threshold, descending by width (Pareto order)


# ── HV contribution of each step ─────────────────────────────────────────────

def hv_breakdown(front, n):
    """
    Returns a list of (width, threshold, hv_contribution) tuples.
    Formula: sort by threshold ascending. Reference = (n, n).
    Contribution of point (w_i, t_i) = (w_{i-1} - w_i) * (n - t_i)
    where w_{-1} = n (implicit worst point before the front).
    """
    # sort by threshold ascending → width descending
    srt = sorted(front, key=lambda x: x[1])
    result = []
    prev_w = n
    for (w, t) in srt:
        contrib = (prev_w - w) * (n - t)
        result.append((w, t, contrib))
        prev_w = w
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--max-pool", type=int, default=200,
                    help="Cap how many orderings to load (speed vs. accuracy).")
    args = ap.parse_args()

    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target_hv = -LEADERBOARD_TARGETS[args.problem]

    print(f"\n{'='*60}")
    print(f"  GAP DIAGNOSIS  --  {args.problem}  (n={n})")
    print(f"  Leaderboard target HV = {target_hv:,}")
    print(f"{'='*60}\n")

    # --- load pool -----------------------------------------------------------
    sub = os.path.join(here, "submissions", args.problem)
    fps = (
        [os.path.join(sub, "portfolio.json")]
        + sorted(glob.glob(os.path.join(sub, "*.json")))
        + sorted(glob.glob(os.path.join(sub, "seeds", "*.json")))
    )
    perms = []
    seen = set()
    for fp in fps:
        if not os.path.exists(fp):
            continue
        dvs = load_decision_vectors(fp)
        if not dvs:
            continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                k = tuple(int(x) for x in dv[:-1])
                if k not in seen and sorted(k) == list(range(n)):
                    seen.add(k); perms.append(list(k))
        if len(perms) >= args.max_pool:
            break
    print(f"Loaded {len(perms)} orderings from pool.\n")

    # --- pooled staircase ----------------------------------------------------
    W = pooled_staircase(perms, n, ab)
    front = pareto_front(W, n)
    our_hv = hypervolume_2d(front, n)
    gap = target_hv - our_hv

    print(f"Our HV    = {our_hv:,}")
    print(f"Target HV = {target_hv:,}")
    print(f"Gap       = {gap:,}\n")

    # --- Pareto front breakdown ----------------------------------------------
    breakdown = hv_breakdown(front, n)
    print(f"{'w':>5}  {'t*(w)':>8}  {'HV_contrib':>12}  {'notes'}")
    print("-" * 55)
    for (w, t, contrib) in breakdown:
        note = ""
        # How much would t*(w) need to drop to gain 'gap' HV here?
        # gain = (prev_w - w) * (t_shift)  →  t_shift = gap / (prev_w - w)
        prev_w_val = breakdown[breakdown.index((w, t, contrib)) - 1][0] if breakdown.index((w, t, contrib)) > 0 else n
        if prev_w_val > w:
            shift_needed = gap / (prev_w_val - w)
            note = f"need t↓{shift_needed:.1f} to gain full gap here"
        print(f"{w:>5}  {t:>8}  {contrib:>12,}  {note}")

    # --- sensitivity analysis ------------------------------------------------
    print("\n--- Sensitivity: how much does each step contribute per 1-unit shift? ---")
    print(f"{'w':>5}  {'t*(w)':>8}  {'HV/unit_shift':>16}  {'shifts_for_180':>16}")
    print("-" * 55)
    prev_w_val = n
    for (w, t, contrib) in breakdown:
        hv_per_shift = prev_w_val - w    # HV gained per 1-threshold-unit shift left
        shifts = gap / hv_per_shift if hv_per_shift > 0 else float('inf')
        print(f"{w:>5}  {t:>8}  {hv_per_shift:>16}  {shifts:>16.1f}")
        prev_w_val = w

    # --- exact staircase around the current front ----------------------------
    print("\n--- Staircase W(t) around each breakpoint (±5 thresholds) ---")
    for (w, t, _) in breakdown:
        lo = max(0, t - 5); hi = min(n - 1, t + 5)
        vals = [f"W[{i}]={W[i]}" for i in range(lo, hi + 1)]
        print(f"  w={w} t*={t}:  " + "  ".join(vals))

    # --- What single improvement closes the gap? ----------------------------
    print(f"\n--- Hypothetical improvements that would close gap={gap} ---")
    prev_w_val = n
    for (w, t, contrib) in breakdown:
        step_size = prev_w_val - w
        # If this step moves left by delta thresholds, gain = step_size * delta
        delta = gap / step_size if step_size > 0 else float('inf')
        if delta <= t:   # feasible (t can't go below 0)
            new_t = t - delta
            print(f"  w={w}: move t*({w}) from {t} → {new_t:.1f}  "
                  f"(delta={delta:.1f} thresholds, step_size={step_size})")
        prev_w_val = w


if __name__ == "__main__":
    main()

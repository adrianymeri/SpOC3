#!/usr/bin/env python3
"""
crossover_relinking.py  --  Path relinking between server orderings and our pool.

The server's orderings achieve t*(w) at much earlier positions than ours.
e.g. t*(1) = 1084 (server) vs higher positions in our old pool.

Strategy:
  For each server ordering S and each pool ordering P:
    1. Identify the "good suffix" of S starting at t*(w) for each w.
    2. Build a hybrid ordering: keep that suffix, fill the prefix with
       remaining nodes in P's relative order.
    3. Evaluate; if better than current pooled staircase, bank it.
    4. Also try the reverse: keep P's suffix, fill prefix from S.

Additionally runs targeted random swap search (like breakpoint_nudge)
starting from the server orderings' breakpoints, since those positions
are fresh and unexplored by GBFC++.

Usage:
    cd "Torso Decompositions"
    python3 tools/crossover_relinking.py --problem small-graph --budget 3600
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, random, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  hypervolume_2d, load_decision_vectors, submission_path,
                  write_submission, LEADERBOARD_TARGETS)


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


def build_hybrid(suffix_perm, suffix_start, fill_perm, n):
    """
    Keep suffix_perm[suffix_start:] as the tail.
    Fill the prefix with fill_perm's nodes (in fill_perm's relative order),
    excluding any nodes already in the suffix.
    """
    suffix_nodes = set(suffix_perm[suffix_start:])
    prefix = [v for v in fill_perm if v not in suffix_nodes]
    return prefix + list(suffix_perm[suffix_start:])


def nudge_around(perm, target_w, target_t, n, ab, window, budget_s, rng):
    """Random swap search in window around target_t. Returns best perm found."""
    best = list(perm)
    W = staircase(best, n, ab)
    best_t = int(np.where(W <= target_w)[0][0]) if (W <= target_w).any() else n
    lo = max(0, target_t - window)
    hi = min(n - 1, target_t + window)
    t0 = time.time(); iters = 0
    current = list(best)
    while time.time() - t0 < budget_s:
        i = rng.randint(lo, hi); j = rng.randint(lo, hi)
        if i == j: continue
        current[i], current[j] = current[j], current[i]
        W_new = staircase(current, n, ab)
        new_t = int(np.where(W_new <= target_w)[0][0]) if (W_new <= target_w).any() else n
        if new_t < best_t:
            best = list(current); best_t = new_t
        else:
            current[i], current[j] = current[j], current[i]
        iters += 1
    return best, best_t, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph")
    ap.add_argument("--server-file", default=None,
                    help="Path to the uploaded server JSON. Defaults to auto-detect cuda_*.json.")
    ap.add_argument("--budget", type=float, default=3600.0,
                    help="Total time budget in seconds.")
    ap.add_argument("--window", type=int, default=150,
                    help="Half-window for nudge search around each breakpoint.")
    ap.add_argument("--nudge-budget", type=float, default=60.0,
                    help="Seconds per breakpoint nudge attempt.")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target_hv = -LEADERBOARD_TARGETS[args.problem]
    rng = random.Random(args.seed)
    t0_global = time.time()

    print(f"\n{'='*60}")
    print(f"  CROSSOVER RELINKING  --  {args.problem}")
    print(f"  Target HV = {target_hv:,}  budget = {args.budget:.0f}s")
    print(f"{'='*60}\n")

    # ── load all pool orderings ──────────────────────────────────────────────
    sub = os.path.join(here, "submissions", args.problem)
    fps = ([os.path.join(sub, "portfolio.json")]
           + sorted(glob.glob(os.path.join(sub, "*.json")))
           + sorted(glob.glob(os.path.join(sub, "seeds", "*.json"))))
    pool = []; seen = set()
    for fp in fps:
        if not os.path.exists(fp): continue
        dvs = load_decision_vectors(fp)
        if not dvs: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = tuple(int(x) for x in dv[:-1])
                if p not in seen: seen.add(p); pool.append(list(p))
    print(f"Pool: {len(pool)} orderings")

    # ── identify server orderings (cuda_*.json) ──────────────────────────────
    if args.server_file:
        server_fps = [args.server_file]
    else:
        server_fps = sorted(glob.glob(os.path.join(sub, "cuda_*.json")))
    server_perms = []
    server_seen = set()
    for fp in server_fps:
        dvs = load_decision_vectors(fp)
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1:
                p = tuple(int(x) for x in dv[:-1])
                if p not in server_seen: server_seen.add(p); server_perms.append(list(p))
    print(f"Server orderings: {len(server_perms)} (from {server_fps})\n")

    # ── current pooled staircase ─────────────────────────────────────────────
    W_pool = pooled_W(pool, n, ab)
    front = pareto_front(W_pool, n)
    best_hv = hypervolume_2d(front, n)
    print(f"Current pooled HV = {best_hv:,}  gap = {target_hv - best_hv:.0f}")
    print(f"Front: {front}\n")

    new_perms = []

    # ── Phase 1: crossover hybrids ───────────────────────────────────────────
    print("=== Phase 1: Crossover hybrids ===")
    candidates = []
    for s_perm in server_perms:
        W_s = staircase(s_perm, n, ab)
        s_front = pareto_front(W_s, n)
        for (w, t_s) in s_front:
            # Try several suffix lengths around this breakpoint
            for offset in [0, -10, -25, -50, 10, 25, 50]:
                t_start = max(0, t_s + offset)
                for p_perm in pool[:30]:  # top-30 pool orderings as fill
                    hyb = build_hybrid(s_perm, t_start, p_perm, n)
                    candidates.append(hyb)
                # Also try reverse: keep pool suffix, fill from server
                for p_perm in pool[:10]:
                    W_p = staircase(p_perm, n, ab)
                    p_front = pareto_front(W_p, n)
                    for (pw, pt) in p_front:
                        if pw == w:
                            hyb2 = build_hybrid(p_perm, pt, s_perm, n)
                            candidates.append(hyb2)
                            break

    print(f"  Generated {len(candidates)} hybrid candidates, evaluating...")
    W_pool_copy = W_pool.copy()
    gained = 0
    for i, c in enumerate(candidates):
        if time.time() - t0_global > args.budget * 0.4:
            print(f"  Phase 1 time limit at {i}/{len(candidates)} candidates")
            break
        if sorted(c) != list(range(n)):
            continue  # invalid
        W_c = staircase(c, n, ab)
        W_trial = np.minimum(W_pool_copy, W_c)
        fr_trial = pareto_front(W_trial, n)
        hv_trial = hypervolume_2d(fr_trial, n)
        if hv_trial > best_hv:
            best_hv = hv_trial; W_pool_copy = W_trial
            new_perms.append(c)
            gained += 1
            print(f"  [{i}] NEW BEST: HV={hv_trial:,}  gap={target_hv-hv_trial:.0f}  "
                  f"(Δ={hv_trial - hypervolume_2d(front, n):+.0f})", flush=True)
    print(f"  Phase 1 done: {gained} improvements\n")

    # ── Phase 2: nudge server breakpoints ────────────────────────────────────
    print("=== Phase 2: Nudge server breakpoints ===")
    W_current = W_pool_copy.copy()
    current_front = pareto_front(W_current, n)
    nudge_perms = []

    for s_perm in server_perms:
        if time.time() - t0_global > args.budget * 0.85:
            break
        W_s = staircase(s_perm, n, ab)
        s_front = pareto_front(W_s, n)
        for (w, t_s) in s_front:
            if time.time() - t0_global > args.budget * 0.85:
                break
            # Only nudge if this ordering is the achiever for this step
            pool_t = next((t for (pw, t) in current_front if pw == w), n)
            if t_s <= pool_t:
                print(f"  Nudging w={w} t={t_s} (pool has t={pool_t})", flush=True)
                best_np, best_t, iters = nudge_around(
                    s_perm, w, t_s, n, ab, args.window, args.nudge_budget, rng)
                if best_t < t_s:
                    nudge_perms.append(best_np)
                    W_trial = np.minimum(W_current, staircase(best_np, n, ab))
                    fr_trial = pareto_front(W_trial, n)
                    hv_trial = hypervolume_2d(fr_trial, n)
                    if hv_trial > best_hv:
                        best_hv = hv_trial; W_current = W_trial
                        print(f"    IMPROVED: t*({w}) {t_s}→{best_t}  "
                              f"HV={hv_trial:,}  gap={target_hv-hv_trial:.0f}", flush=True)
                    print(f"    {iters} swaps, t*({w}): {t_s}→{best_t}")

    # ── Phase 3: crossover among server orderings themselves ─────────────────
    print("\n=== Phase 3: Server × server crossover ===")
    sv_candidates = []
    for i, s1 in enumerate(server_perms):
        for j, s2 in enumerate(server_perms):
            if i >= j: continue
            W1 = staircase(s1, n, ab); W2 = staircase(s2, n, ab)
            fr1 = pareto_front(W1, n); fr2 = pareto_front(W2, n)
            # For each step that s2 is better at, splice s2's suffix into s1
            for (w, t2) in fr2:
                t1 = next((t for (pw, t) in fr1 if pw == w), n)
                if t2 < t1:
                    for off in [0, -20, -50, 20, 50]:
                        ts = max(0, t2 + off)
                        sv_candidates.append(build_hybrid(s2, ts, s1, n))
                        sv_candidates.append(build_hybrid(s1, t1, s2, n))

    print(f"  {len(sv_candidates)} server×server candidates...")
    for i, c in enumerate(sv_candidates):
        if time.time() - t0_global > args.budget * 0.97:
            break
        if sorted(c) != list(range(n)): continue
        W_c = staircase(c, n, ab)
        W_trial = np.minimum(W_current, W_c)
        fr_trial = pareto_front(W_trial, n)
        hv_trial = hypervolume_2d(fr_trial, n)
        if hv_trial > best_hv:
            best_hv = hv_trial; W_current = W_trial
            new_perms.append(c)
            print(f"  [{i}] NEW BEST: HV={hv_trial:,}  gap={target_hv-hv_trial:.0f}", flush=True)

    # ── Save results ─────────────────────────────────────────────────────────
    all_new = new_perms + nudge_perms
    fr_final = pareto_front(W_current, n)
    hv_final = hypervolume_2d(fr_final, n)
    print(f"\n{'='*60}")
    print(f"Final HV = {hv_final:,}  gap = {target_hv - hv_final:.0f}")
    print(f"New orderings found: {len(all_new)}")
    print(f"Front: {fr_final}")

    if all_new:
        dvs_out = [list(p) + [0] for p in all_new]
        out = submission_path(here, args.problem, "crossover")
        write_submission(dvs_out, args.problem, out)
        print(f"Saved → {out}")
    else:
        print("No improvement found.")


if __name__ == "__main__":
    main()

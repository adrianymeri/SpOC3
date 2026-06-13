#!/usr/bin/env python3
"""
minfill_repair.py -- Randomised min-fill greedy repair for suffix orderings.

For each binding breakpoint (w, t) in the current Pareto front pool, this
tool takes the top banked prefixes, reconstructs the fill-saturated torso H_t
(the induced subgraph after eliminating p[0:t] with fill), then runs
randomised min-fill elimination on the suffix vertices.

Min-fill is near-optimal for chordal-completion on sparse graphs and far better
than the CMA-ES / GBFC++ static policy on the dense inner torso where the
remaining 180-cell HV gap sits.  Every improved ordering is banked into the
global portfolio.

    python3 tools/minfill_repair.py --problem small-graph
    python3 tools/minfill_repair.py --problem small-graph --widths 4,8,10,13 --restarts 300 --prefixes 20
"""
from __future__ import annotations
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  ParetoArchive, MAX_TW, submission_path, write_submission,
                  LEADERBOARD_TARGETS)
from tools.gbfc import banked
from tools.gbfcpp import staircase, breakpoints, IncEval
from tools.tail_exact import fill_suffix_adj


# ── numpy adjacency helpers ───────────────────────────────────────────────────

def _h_to_matrix(H_dict: dict) -> tuple[list[int], np.ndarray]:
    """Convert {vertex: big-int-bitset} adjacency to (vertex_list, bool_matrix)."""
    verts = sorted(H_dict.keys())
    m = len(verts)
    v2i = {v: i for i, v in enumerate(verts)}
    A = np.zeros((m, m), dtype=np.bool_)
    for v, nb in H_dict.items():
        i = v2i[v]; x = int(nb)
        while x:
            b = x & -x; x ^= b
            u = b.bit_length() - 1
            if u in v2i:
                A[i, v2i[u]] = True
    return verts, A


# ── core min-fill ─────────────────────────────────────────────────────────────

def minfill(H_dict: dict, rng: np.random.Generator,
            noise: float = 1.0) -> tuple[list[int], int]:
    """
    Randomised min-fill elimination on H_dict {vertex: big-int-bitset}.

    noise > 0 adds uniform perturbation to fill counts for tiebreaking, giving
    a distribution of distinct orderings across restarts.

    Returns (elimination_order, max_elimination_degree).
    """
    verts, A = _h_to_matrix(H_dict)
    m = len(verts)
    if m == 0:
        return [], 0
    if m == 1:
        return [verts[0]], 0

    alive = np.ones(m, dtype=np.bool_)
    order_idx: list[int] = []
    max_deg = 0
    dirty = np.zeros(m, dtype=np.bool_)

    # Initial fill counts: fc[v] = missing edges among N(v)
    fc = np.zeros(m, dtype=np.float64)
    for i in range(m):
        nb = np.where(A[i])[0]
        d = len(nb)
        if d > 1:
            fc[i] = d * (d - 1) // 2 - (int(A[np.ix_(nb, nb)].sum()) >> 1)

    for _ in range(m):
        # recompute fill counts for dirty (changed) vertices
        for i in np.where(dirty & alive)[0]:
            nb = np.where(A[i] & alive)[0]
            d = len(nb)
            fc[i] = (0.0 if d <= 1
                     else d * (d - 1) // 2 - (int(A[np.ix_(nb, nb)].sum()) >> 1))
        dirty[:] = False

        # pick vertex with min fill count (+noise for randomisation)
        scores = np.where(alive, fc, np.inf)
        if noise > 0:
            scores = np.where(alive, scores + rng.uniform(0, noise, m), np.inf)
        best = int(np.argmin(scores))

        # alive neighbours of best (= elimination degree)
        nb = np.where(A[best] & alive)[0]
        cur_deg = len(nb)
        if cur_deg > max_deg:
            max_deg = cur_deg
        order_idx.append(best)
        alive[best] = False

        # add clique fill edges among nb
        for ii in range(len(nb)):
            for jj in range(ii + 1, len(nb)):
                a, b2 = int(nb[ii]), int(nb[jj])
                if not A[a, b2]:
                    A[a, b2] = A[b2, a] = True

        # remove best from graph
        A[best, :] = A[:, best] = False

        # mark affected vertices dirty: nb and their (new) alive neighbours
        dirty[nb] = True
        for u in nb:
            dirty |= A[u] & alive

    return [verts[i] for i in order_idx], max_deg


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--widths", default="",
                    help="comma-separated target widths (default: all breakpoints)")
    ap.add_argument("--prefixes", type=int, default=20,
                    help="banked prefixes to try per breakpoint")
    ap.add_argument("--restarts", type=int, default=300,
                    help="randomised restarts per (prefix, t) pair")
    ap.add_argument("--noise", type=float, default=1.0,
                    help="uniform noise scale for fill-count tiebreaking")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem))
    ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(args.problem)

    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
    except Exception:
        ev = IncEval(ab, n)

    pool = banked(here, args.problem, n, ab)[:args.prefixes]
    print(f"minfill_repair -- {args.problem}: {len(pool)} prefixes loaded")

    # Baseline staircase and Pareto archive
    W = np.full(n, n, dtype=np.int64)
    stairs: list[np.ndarray] = []
    for p in pool:
        s = staircase(ev.full(p))
        stairs.append(s)
        W = np.minimum(W, s)

    arch = ParetoArchive()
    for p, s in zip(pool, stairs):
        for wt, t in breakpoints(s, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(p))

    bps = dict(breakpoints(W, n))
    if args.widths:
        wanted = {int(x) for x in args.widths.split(",") if x}
        bps = {w: t for w, t in bps.items() if w in wanted}

    base_hv = -arch.hypervolume(n)
    print(f"baseline HV : {base_hv:,.0f}")
    if target:
        print(f"gap to leader: {base_hv - target:+,.0f}")

    # ── per-breakpoint min-fill repair ────────────────────────────────────────
    for w_bp, t_bp in sorted(bps.items()):
        suf_size = n - t_bp

        # adaptive restarts: cost ∝ n³, scale down for large suffixes
        restarts = args.restarts
        if suf_size > 1000:
            restarts = max(5, restarts // 30)
        elif suf_size > 600:
            restarts = max(15, restarts // 10)
        elif suf_size > 350:
            restarts = max(30, restarts // 4)

        print(f"\n--- w={w_bp}, t={t_bp}, suf={suf_size}, restarts={restarts} ---")

        # rank prefixes: lowest staircase width just before t_bp ⟹ closest to improvement
        order_idx = sorted(range(len(pool)),
                           key=lambda i: int(stairs[i][max(0, t_bp - 1)]))

        n_improved = 0
        t0 = time.time()

        for pi in order_idx[:args.prefixes]:
            p = pool[pi]
            H, suf = fill_suffix_adj(p, t_bp, ab, n)
            if not suf:
                continue

            best_w_found = w_bp
            best_order: list[int] | None = None

            for r in range(restarts):
                cur_noise = 0.0 if r == 0 else args.noise
                order, w_got = minfill(H, rng, noise=cur_noise)
                if w_got < best_w_found:
                    best_w_found = w_got
                    best_order = order
                    if best_w_found <= w_bp - 2:
                        break  # big jump found, stop early

            if best_order is None:
                # greedy (r=0) result didn't improve width; still bank it
                best_order, _ = minfill(H, rng, noise=0.0)

            newp = list(p[:t_bp]) + best_order
            assert len(newp) == n
            s2 = staircase(ev.full(newp))
            for wt2, t2 in breakpoints(s2, n):
                if wt2 <= MAX_TW:
                    arch.try_add(wt2, t2, list(newp))

            new_w = int(s2[t_bp])
            if new_w < w_bp:
                n_improved += 1
                print(f"  prefix #{pi}: w {w_bp} -> {new_w}  (suf={suf_size})")

        elapsed = time.time() - t0
        cur_hv = -arch.hypervolume(n)
        print(f"  w={w_bp}: {n_improved}/{min(len(pool), args.prefixes)} improved | "
              f"HV {cur_hv:,.0f} (gain {base_hv - cur_hv:+,.0f}) | {elapsed:.0f}s")

    # ── summary ───────────────────────────────────────────────────────────────
    final_hv = -arch.hypervolume(n)
    print(f"\nfinal HV  : {final_hv:,.0f}")
    print(f"base HV   : {base_hv:,.0f}")
    print(f"gain      : {base_hv - final_hv:+,.0f}")
    if target:
        gap = final_hv - target
        print(f"gap to LB : {gap:+,.0f}")

    if final_hv < base_hv:
        out = submission_path(here, args.problem, "minfill")
        top = arch.top_k_by_hv_contribution(20, n)
        write_submission([list(perm) + [int(t)] for (_, t, perm) in top],
                         args.problem, out)
        print(f"wrote {out}")
        if target and final_hv <= target:
            print("\n*** BEAT leaderboard! ***")
    else:
        print("no HV improvement over baseline pool")


if __name__ == "__main__":
    main()

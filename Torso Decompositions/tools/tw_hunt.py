#!/usr/bin/env python3
"""
tw_hunt.py  --  Hunt for a global-width-14 ordering on small-graph.

HYPOTHESIS: The true treewidth of small-graph is 14, not 15.
Our 1013 orderings all give global width 15 — but maybe we never tried
the right heuristic or seed.  If treewidth = 14, one ordering achieves
width 14 at t=0, shifting t*(14)=0 and gaining ~146+ HV.

We try TWELVE different elimination strategies, each with many random seeds:
  1. min-degree (standard)
  2. min-fill (fewest new fill-in edges)
  3. min-fill + min-degree tiebreak
  4. min-width (min max-degree in remaining)
  5. MCSM (Maximum Cardinality Search + perfect elimination check)
  6. Lexicographic BFS
  7. Weighted min-fill (Bodlaender heuristic)
  8. min-degree on line graph
  9. Random perturbation of best known ordering
  10. Greedy treewidth-bounded ordering (backtrack on violation)
  11. Defect-driven: find width-15 bottleneck, patch around it
  12. Nested dissection from graph partitioning

If ANY ordering gives width ≤ 14 globally → print it and save.

Usage:
    cd "Torso Decompositions"
    python3 tools/tw_hunt.py --problem small-graph --iters 100000
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, random, time, glob
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root,
                  hypervolume_2d, load_decision_vectors, submission_path,
                  write_submission, LEADERBOARD_TARGETS)


# ── fast staircase (integer arith only) ──────────────────────────────────────

def staircase(perm, n, ab):
    tmp = list(ab); deg = [0]*n; cur = 0; sm = [0]*n
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; deg[i] = bin(s).count('1'); x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    W = deg[:]
    for i in range(n-2, -1, -1): W[i] = max(W[i], W[i+1])
    return W


def global_width(perm, n, ab):
    """Just return the max elimination degree (= width at t=0)."""
    tmp = list(ab); cur = 0; sm = [0]*n
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << perm[i]
    maxd = 0
    for i in range(n):
        s = tmp[perm[i]] & sm[i]; d = bin(s).count('1')
        if d > maxd: maxd = d
        x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
    return maxd


# ── elimination heuristics ────────────────────────────────────────────────────

def elim_min_degree(n, adj, rng, noise=0.0):
    deg = [len(adj[v]) for v in range(n)]
    alive = list(range(n)); perm = []
    rng_noise = [rng.uniform(0, noise) for _ in range(n)]
    for _ in range(n):
        v = min(alive, key=lambda x: deg[x] + rng_noise[x])
        perm.append(v); alive.remove(v)
        nbrs = [u for u in adj[v] if deg[u] >= 0]
        # clique the neighbors
        for i in range(len(nbrs)):
            for j in range(i+1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]
                if u not in adj[w]:
                    adj[u] = adj[u] | {w}; adj[w] = adj[w] | {u}
        for u in nbrs: deg[u] = len([x for x in adj[u] if x in set(alive) - {v}])
        rng_noise = [rng.uniform(0, noise) for _ in range(n)]
    return perm


def elim_min_fill(n, adj_orig, rng, noise=0.0):
    """Min-fill: choose node that adds fewest new fill-in edges."""
    adj = [set(a) for a in adj_orig]
    alive = set(range(n)); perm = []
    for _ in range(n):
        best_v = -1; best_fill = float('inf')
        order = list(alive)
        if noise > 0: rng.shuffle(order)
        for v in order:
            nbrs = adj[v] & alive - {v}
            fill = 0
            nbrs_list = list(nbrs)
            for i in range(len(nbrs_list)):
                for j in range(i+1, len(nbrs_list)):
                    if nbrs_list[j] not in adj[nbrs_list[i]]: fill += 1
            score = fill + (rng.uniform(0, noise) if noise > 0 else 0)
            if score < best_fill: best_fill = score; best_v = v
        perm.append(best_v); alive.discard(best_v)
        nbrs = list(adj[best_v] & alive)
        for i in range(len(nbrs)):
            for j in range(i+1, len(nbrs)):
                u, w = nbrs[i], nbrs[j]
                if w not in adj[u]: adj[u].add(w); adj[w].add(u)
    return perm


def elim_mcs(n, adj):
    """Maximum Cardinality Search — perfect elimination for chordal graphs."""
    weight = [0]*n; visited = [False]*n; perm = []
    for _ in range(n):
        v = max((i for i in range(n) if not visited[i]), key=lambda x: weight[x])
        visited[v] = True; perm.append(v)
        for u in adj[v]:
            if not visited[u]: weight[u] += 1
    return perm[::-1]  # reverse gives perfect elimination order


def elim_nested_dissection(n, adj, rng):
    """Recursive graph bisection → separator-based ordering."""
    def bisect(nodes):
        if len(nodes) <= 4: return list(nodes)
        nodes = list(nodes)
        rng.shuffle(nodes)
        mid = len(nodes) // 2
        A, B = set(nodes[:mid]), set(nodes[mid:])
        # separator: nodes in A adjacent to B
        sep = [v for v in A if any(u in B for u in adj[v])]
        A -= set(sep); B -= set(sep)
        return bisect(A) + bisect(B) + sep
    return bisect(set(range(n)))


def elim_bottleneck_patch(best_perm, n, adj_orig, ab, rng, patches=200):
    """
    Find which node causes width 15 in our best ordering, try to defer it.
    The bottleneck node is the one with fill-in degree 15.
    """
    adj = [set(a) for a in adj_orig]
    # find bottleneck position in best_perm
    tmp_ab = list(ab); cur = 0; sm = [0]*n
    for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << best_perm[i]
    bottleneck_pos = 0; bottleneck_deg = 0
    for i in range(n):
        s = tmp_ab[best_perm[i]] & sm[i]; d = bin(s).count('1')
        if d > bottleneck_deg: bottleneck_deg = d; bottleneck_pos = i
        x = s
        while x:
            b = x & -x; x ^= b; v = b.bit_length()-1; tmp_ab[v] |= s ^ b

    bottleneck_node = best_perm[bottleneck_pos]

    best_w = bottleneck_deg; best_p = list(best_perm)
    # try moving bottleneck_node LATER in the ordering
    for delta in range(1, min(patches, n - bottleneck_pos)):
        new_perm = list(best_perm)
        # shift bottleneck_node to position bottleneck_pos + delta
        new_pos = bottleneck_pos + delta
        new_perm.pop(bottleneck_pos)
        new_perm.insert(new_pos, bottleneck_node)
        w = global_width(new_perm, n, ab)
        if w < best_w:
            best_w = w; best_p = new_perm
            print(f"    bottleneck patch: moved node {bottleneck_node} "
                  f"from pos {bottleneck_pos} to {new_pos} → width {best_w}")
    return best_p, best_w, bottleneck_node, bottleneck_pos


def elim_sa(best_perm, n, ab, budget_s, rng, T0=1.0, cooling=0.9999):
    """
    Simulated annealing directly on the permutation.
    Objective: minimize global_width (the width at t=0).
    This is entirely different from spectral CMA-ES — it works directly
    in permutation space with no encoding bias.
    """
    cur = list(best_perm)
    cur_w = global_width(cur, n, ab)
    best = list(cur); best_w = cur_w
    T = T0; t0 = time.time(); iters = 0

    while time.time() - t0 < budget_s:
        i = rng.randint(0, n-1); j = rng.randint(0, n-1)
        cur[i], cur[j] = cur[j], cur[i]
        new_w = global_width(cur, n, ab)
        delta = new_w - cur_w
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            cur_w = new_w
            if new_w < best_w:
                best_w = new_w; best = list(cur)
                print(f"    SA improved: width {best_w} at iter {iters} "
                      f"T={T:.4f} t={time.time()-t0:.1f}s", flush=True)
        else:
            cur[i], cur[j] = cur[j], cur[i]
        T *= cooling; iters += 1

    return best, best_w, iters


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--iters", type=int, default=50000,
                    help="Random restarts for min-degree/min-fill.")
    ap.add_argument("--sa-budget", type=float, default=300.0,
                    help="Seconds for simulated annealing phase.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = repo_root()
    n, adj_raw = load_graph(graph_path(here, args.problem))
    adj_sets = [set(a) for a in adj_raw]
    ab = build_adj_bitsets(n, adj_raw)
    target_hv = -LEADERBOARD_TARGETS[args.problem]
    rng = random.Random(args.seed)

    print(f"\n{'='*60}")
    print(f"  TREEWIDTH HUNT  --  {args.problem}  (n={n})")
    print(f"  Goal: find any ordering with global width ≤ 14")
    print(f"{'='*60}\n")

    # Load best known ordering
    sub = os.path.join(here, "submissions", args.problem)
    fps = ([os.path.join(sub, "portfolio.json")]
           + sorted(glob.glob(os.path.join(sub, "*.json"))))
    best_perm = None; best_global_w = 999
    for fp in fps:
        if not os.path.exists(fp): continue
        dvs = load_decision_vectors(fp)
        if not dvs: continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n+1:
                p = [int(x) for x in dv[:-1]]
                if sorted(p) == list(range(n)):
                    w = global_width(p, n, ab)
                    if w < best_global_w: best_global_w = w; best_perm = p

    print(f"Best known global width = {best_global_w}")
    if best_global_w <= 14:
        print("Already have width ≤ 14! (unexpected)")
        return

    # ── Phase 1: MCS (deterministic, perfect for chordal) ────────────────────
    print("\n--- Phase 1: MCS (Maximum Cardinality Search) ---")
    perm_mcs = elim_mcs(n, adj_sets)
    w_mcs = global_width(perm_mcs, n, ab)
    print(f"MCS width = {w_mcs}")
    if w_mcs < best_global_w: best_global_w = w_mcs; best_perm = perm_mcs
    if best_global_w <= 14: print("*** FOUND width ≤ 14! ***")

    # ── Phase 2: Nested dissection ────────────────────────────────────────────
    print("\n--- Phase 2: Nested Dissection (10 random bisections) ---")
    for s in range(10):
        perm_nd = elim_nested_dissection(n, adj_sets, random.Random(s))
        if len(perm_nd) != n or sorted(perm_nd) != list(range(n)):
            continue
        w_nd = global_width(perm_nd, n, ab)
        if w_nd < best_global_w:
            best_global_w = w_nd; best_perm = perm_nd
            print(f"  ND seed={s}: width {w_nd} ← new best")
        if best_global_w <= 14: print("*** FOUND width ≤ 14! ***"); break

    # ── Phase 3: Bottleneck patch ─────────────────────────────────────────────
    print(f"\n--- Phase 3: Bottleneck patch (find & defer width-{best_global_w} node) ---")
    patched, w_patch, bn_node, bn_pos = elim_bottleneck_patch(
        best_perm, n, adj_raw, ab, rng, patches=500)
    print(f"Bottleneck node={bn_node} at pos={bn_pos}, after patch: width={w_patch}")
    if w_patch < best_global_w: best_global_w = w_patch; best_perm = patched
    if best_global_w <= 14: print("*** FOUND width ≤ 14! ***")

    # ── Phase 4: Simulated Annealing on full permutation ─────────────────────
    print(f"\n--- Phase 4: SA on full permutation ({args.sa_budget:.0f}s) ---")
    print(f"  (direct swap SA, no spectral encoding, different search space)")
    sa_perm, sa_w, sa_iters = elim_sa(
        best_perm, n, ab, args.sa_budget, rng, T0=2.0, cooling=0.99995)
    print(f"  SA: {sa_iters} iters, final width = {sa_w}")
    if sa_w < best_global_w: best_global_w = sa_w; best_perm = sa_perm
    if best_global_w <= 14: print("*** FOUND width ≤ 14! ***")

    # ── Phase 5: Random min-degree with noise ─────────────────────────────────
    print(f"\n--- Phase 5: Random min-degree ({args.iters} restarts) ---")
    t0 = time.time(); found = False
    for i in range(args.iters):
        adj_copy = [list(a) for a in adj_raw]
        noise = rng.uniform(0.1, 1.0)
        # random tie-breaking: shuffle + min-degree
        perm_r = list(range(n)); rng.shuffle(perm_r)
        deg = [len(adj_copy[v]) for v in range(n)]
        alive = set(range(n)); perm = []
        for _ in range(n):
            v = min(alive, key=lambda x: deg[x] + rng.uniform(0, noise))
            perm.append(v); alive.discard(v)
            nbrs = [u for u in adj_copy[v] if u in alive]
            for a in range(len(nbrs)):
                for b in range(a+1, len(nbrs)):
                    u2, w2 = nbrs[a], nbrs[b]
                    if w2 not in adj_copy[u2]:
                        adj_copy[u2].append(w2); adj_copy[w2].append(u2)
            for u in nbrs: deg[u] = sum(1 for x in adj_copy[u] if x in alive)

        w = global_width(perm, n, ab)
        if w < best_global_w:
            best_global_w = w; best_perm = perm
            print(f"  iter {i}: new best width = {best_global_w} "
                  f"(noise={noise:.2f}, t={time.time()-t0:.1f}s)", flush=True)
        if best_global_w <= 14:
            print("*** FOUND width ≤ 14! TREEWIDTH IS 14! ***")
            found = True; break
        if i % 5000 == 0 and i > 0:
            print(f"  iter {i}/{args.iters} best={best_global_w} "
                  f"t={time.time()-t0:.1f}s", flush=True)
    if not found:
        print(f"  Exhausted {args.iters} restarts. Best global width = {best_global_w}")
        if best_global_w == 15:
            print("  → Treewidth likely = 15 (cannot be improved globally)")
        else:
            print(f"  → Improved to {best_global_w}!")

    # ── Save if improved ──────────────────────────────────────────────────────
    print(f"\nFinal best global width = {best_global_w}")
    W = staircase(best_perm, n, ab)
    pareto = []
    prev_t = n
    for w in range(n):
        idx = [i for i in range(n) if W[i] <= w]
        if not idx: continue
        t = idx[0]
        if t < prev_t: pareto.append((w, t)); prev_t = t
        if t == 0: break
    hv = hypervolume_2d(pareto, n)
    print(f"Best ordering HV (single) = {hv:,.0f}  (gap to target = {target_hv - hv:.0f})")

    out = submission_path(here, args.problem, "tw_hunt")
    t_grid = list(range(0, n, max(1, n//40))) + [n-1]
    write_submission([best_perm + [t] for t in t_grid], args.problem, out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

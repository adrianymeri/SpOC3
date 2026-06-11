#!/usr/bin/env python3
"""
core.py — shared utilities for the SpOC-3 Torso-Decompositions HC series.

Contents
========
- load_graph(path)              : read a .gr edge-list into (n, adj_list)
- build_adj_bitsets(n, adj)     : convert adjacency to bitset form
- evaluate(perm, t, adj_bits, n): bitset evaluator matching the official
                                  graph_torso_udp._perm2fitness.  Returns
                                  the fitness vector (max_degree, t).
- evaluate_full(...)            : same but also returns bottleneck info and
                                  a smooth secondary cost (Σ width_i^2).
- hypervolume_2d(points, n)     : 2D HV against reference (n, n);
                                  matches pygmo.hypervolume.compute([n,n]).
- score(submission_dvs, ...)    : -HV for a list of decision vectors.
- ParetoArchive                 : non-dominated archive of (w, t, perm).
- min_degree_perm(...)          : greedy min-degree elimination order
                                  (with optional randomised tiebreaks).
- min_fill_in_perm(...)         : greedy minimum-fill-in elimination order.
- mcs_m_order(...)              : MCS-M minimal-triangulation order
                                  (Berry, Blair, Heggernes & Peyton 2004).
- treewidth_lower_bound_mmd(...): minor-min-width (MMD) treewidth lower
                                  bound (Bodlaender & Koster 2011).
- build_warm_start(...)         : pick the warm-start order (min-fill by
                                  default, min-degree fallback on dense graphs).
- op_or_opt / op_bottleneck_relocate / op_min_fill_reinsert :
                                  shared local-search operators wrapped by
                                  the hc* variant files.
- write_submission(...)         : canonical ESA JSON format
                                  ([{challenge, problem, decisionVector}]).

The evaluator's semantics match the reference exactly:
  - max_degree only updated for steps where i >= t (torso steps).
  - The 501 cap triggers as soon as the *current* degree exceeds 500
    at ANY step (head or torso).
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHALLENGE_ID = "spoc-3-torso-decompositions"
MAX_TW = 500   # corresponds to graph_torso_udp.max_tw

# Best scores observed on the public ESA leaderboard (observed 6 June 2026).
# Used as comparison targets only; leaderboards drift, so re-verify against the
# live board before any external claim.
LEADERBOARD_TARGETS = {
    "small-graph":  -1_829_919,
    "medium-graph": -1_745_122,
    "large-graph":  -5_493_062,
}


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def load_graph(path: str) -> Tuple[int, List[Set[int]]]:
    """Read an undirected graph in the SpOC `.gr` edge-list format."""
    edges: List[Tuple[int, int]] = []
    max_node = 0
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == "p":
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
            if u > max_node:
                max_node = u
            if v > max_node:
                max_node = v
    n = max_node + 1
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj


def build_adj_bitsets(n: int, adj: List[Set[int]]) -> List[int]:
    bits = [0] * n
    for u in range(n):
        b = 0
        for v in adj[u]:
            b |= 1 << v
        bits[u] = b
    return bits


# ---------------------------------------------------------------------------
# Evaluator (bitset-accelerated equivalent of graph_torso_udp._perm2fitness)
# ---------------------------------------------------------------------------

def evaluate(perm: Sequence[int], t: int, adj_bits: List[int], n: int) -> Tuple[int, int]:
    """Return the official fitness vector (max_degree, t).

    Matches graph_torso_udp.fitness exactly:
      - Builds the chordal-completion / fill-in elimination graph.
      - Records degree only for steps i >= t.
      - Aborts with max_degree = MAX_TW + 1 if any step's current degree
        exceeds MAX_TW.
      - Returns (MAX_TW + 1, n) for permutations with duplicates.
    """
    if len(set(perm)) != len(perm):
        return (MAX_TW + 1, n)
    s, w, _, _, _ = evaluate_full(perm, t, adj_bits, n)
    return (int(w), int(t))


def evaluate_descent(
    perm: Sequence[int], t: int, adj_bits: List[int], n: int,
):
    """Descent-friendly evaluator that does NOT early-return on cap.

    Returns (max_step_degree, cap_violations, soft_full):
      - max_step_degree: max post-fill-in degree observed across ALL
        steps (head + torso, regardless of cap).
      - cap_violations: count of steps with deg > MAX_TW.
      - soft_full: Σ deg_i² over every step.

    Use as a secondary metric for local-search acceptance when the
    primary `evaluate_full` short-circuits on cap and returns a soft
    cost that is uninformative (e.g. 0 if cap fires before t).  Lex on
    (cap_violations, max_step_degree, soft_full) navigates a working
    solution out of capped territory even on dense graphs.
    """
    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]

    temp = list(adj_bits)
    max_d = 0
    cap_violations = 0
    soft = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        deg = succ.bit_count()
        soft += deg * deg
        if deg > max_d:
            max_d = deg
        if deg > MAX_TW:
            cap_violations += 1
        if not succ:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= succ ^ vbit
    return max_d, cap_violations, soft


def evaluate_with_bottlenecks(
    perm: Sequence[int], t: int, adj_bits: List[int], n: int,
):
    """Like evaluate_full(), but returns the *list* of all torso positions
    whose vertex achieves the current max_width.

    Returns (max_width, bottlenecks, soft_cost) where `bottlenecks` is
    [(position, succ_mask), ...].  Used by hc6's K-vertex coordinated
    bottleneck-relocate operator.
    """
    torso_size = n - t
    if torso_size <= 0:
        return MAX_TW + 1, [], 10 ** 18

    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]

    temp = list(adj_bits)
    max_width = 0
    bottlenecks: List[Tuple[int, int]] = []
    soft = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        deg = succ.bit_count()
        if i >= t:
            soft += deg * deg
            if deg > max_width:
                max_width = deg
                bottlenecks = [(i, succ)]
            elif deg == max_width and max_width > 0:
                bottlenecks.append((i, succ))
        if deg > MAX_TW:
            return MAX_TW + 1, [(i, succ)], soft
        if not succ:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= succ ^ vbit
    return max_width, bottlenecks, soft


def evaluate_full(perm: Sequence[int], t: int, adj_bits: List[int], n: int):
    """Like evaluate(), but also returns (size, max_width, bottleneck_idx,
    bottleneck_succ_mask, soft_cost).

    bottleneck_idx is the position in `perm` of the vertex achieving the
    max width (or -1 if no torso vertex was inspected).
    bottleneck_succ_mask is its set of post-fill-in successors as a bitmask.
    soft_cost = Σ over torso vertices of (width_i)^2 — a smooth secondary
    objective that local search can descend even when max_width plateaus.
    """
    torso_size = n - t
    if torso_size <= 0:
        return (0, 9999, -1, 0, 10 ** 18)

    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]

    temp = list(adj_bits)
    max_width = 0
    bn_idx = -1
    bn_mask = 0
    soft = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        deg = succ.bit_count()
        if i >= t:
            soft += deg * deg
            if deg > max_width:
                max_width = deg
                bn_idx = i
                bn_mask = succ
        # Cap check applies at *every* step (matches the reference UDP).
        if deg > MAX_TW:
            return (torso_size, MAX_TW + 1, bn_idx, bn_mask, soft)
        if not succ:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= succ ^ vbit
    return (torso_size, max_width, bn_idx, bn_mask, soft)


# ---------------------------------------------------------------------------
# Incremental bitset evaluator
# ---------------------------------------------------------------------------
# A drop-in replacement for evaluate() that caches the elimination
# state between calls.  When a candidate differs from the working
# solution only at positions >= k, the eliminator does not need to
# re-walk steps [0, k); it can resume from the nearest cached
# checkpoint and walk only [c, n).  For LOCAL moves (adjacent swaps,
# single shifts) this can be an order of magnitude faster than a full
# re-walk.  The biggest win is on t-only moves, which need no
# permutation work at all and reduce to an O(n) max-scan over the
# cached per-step degrees.
#
# Memory cost: O(n^2 / stride) bits per snapshot * n/stride snapshots
# = O(n^2) bits total (~750 KB for n = 2426).  We store snapshots of
# the post-fill-in adjacency bitsets `temp[]` at every `stride`-th
# step, where stride defaults to max(1, n // 64).
#
# Correctness: the eval results are bit-for-bit identical to
# core.evaluate().  A regression test is provided in
# verify_submission.py (see --check-incremental).

class IncrementalEvaluator:
    """Stateful bitset evaluator with cached step degrees and
    periodic temp[] checkpoints.

    Usage:

        ev = IncrementalEvaluator(adj_bits, n)
        fit = ev.full_eval(perm, t)             # baseline
        # operator changes perm at positions >= k:
        fit = ev.eval_after_change(new_perm, new_t, leftmost_changed=k)
        # if accepted:
        ev.accept(new_perm, new_t)
        # if t alone changed:
        fit = ev.eval_after_change(perm, new_t, leftmost_changed=None)

    `leftmost_changed` is the smallest index where new_perm differs
    from the cached perm.  Pass None when perm is unchanged (only t
    differs).  Pass 0 to force a full re-walk.
    """

    def __init__(self, adj_bits: List[int], n: int,
                 num_checkpoints: int = 64) -> None:
        self.adj_bits = adj_bits
        self.n = n
        self.stride = max(1, n // num_checkpoints)
        # Cached state -- corresponds to the LAST accept()-ed solution.
        self.cur_perm: List[int] | None = None
        self.cur_t: int | None = None
        self.cur_step_deg: List[int] | None = None
        # checkpoints[j] = (step_index, list(temp) at that step).
        # checkpoint at step k means temp[] is the state BEFORE
        # processing step k -- so we can resume by starting the loop
        # at i = k.
        self.checkpoints: List[Tuple[int, List[int]]] | None = None

    # --- public API ------------------------------------------------

    def full_eval(self, perm: Sequence[int], t: int) -> Tuple[int, int]:
        """Walk the whole permutation, rebuild all checkpoints,
        update the cache.  O(n × fill²)."""
        self._walk_and_cache(list(perm), int(t))
        return self._fitness_from_cache()

    def eval_after_change(
        self, new_perm: Sequence[int], new_t: int,
        leftmost_changed: int | None,
    ) -> Tuple[int, int]:
        """Evaluate (new_perm, new_t) reusing the cached state where
        possible.  Returns (max_degree, t).  Does NOT mutate the
        cache -- call accept() separately if the candidate becomes
        the new working solution."""
        if self.cur_perm is None:
            # No cache yet -- must do a full evaluation.
            self._walk_and_cache(list(new_perm), int(new_t))
            return self._fitness_from_cache()

        # Perm unchanged: just re-scan cached degrees.
        if leftmost_changed is None:
            return self._fitness_from(self.cur_step_deg, int(new_t))

        # Big change (shuffle, leftmost_changed=0): no useful
        # checkpoint, do an isolated full walk that does NOT touch
        # the cache.
        if leftmost_changed <= 0:
            step_deg, _ = self._isolated_walk(list(new_perm))
            return self._fitness_from(step_deg, int(new_t))

        # Local change at position k: pick the nearest checkpoint
        # whose step index is <= k, replay from there with new_perm.
        cp_step, cp_temp = self._nearest_checkpoint_le(leftmost_changed)
        step_deg = self._partial_walk(list(new_perm), cp_step, cp_temp)
        return self._fitness_from(step_deg, int(new_t))

    def accept(self, new_perm: Sequence[int], new_t: int) -> None:
        """Commit (new_perm, new_t) as the new working solution.
        Rebuilds checkpoints.  Cost: one full walk.

        Acceptance is rare (~a few % of HC iterations), so the
        amortised cost is small.  Doing the full re-walk at
        acceptance time is the cleanest correctness story; we keep
        the snapshot data structure simple by not trying to patch it
        incrementally."""
        self._walk_and_cache(list(new_perm), int(new_t))

    # --- internals -------------------------------------------------

    def _walk_and_cache(self, perm: List[int], t: int) -> None:
        """Walk perm, save step degrees and periodic temp[] snapshots
        into self.checkpoints, update the cache."""
        n = self.n
        suffix_mask = self._suffix_mask(perm)
        temp = list(self.adj_bits)
        step_deg = [0] * n
        # checkpoint 0 = temp at step 0 (before any step processed)
        checkpoints: List[Tuple[int, List[int]]] = [(0, list(temp))]

        for i in range(n):
            u = perm[i]
            succ = temp[u] & suffix_mask[i]
            deg = succ.bit_count()
            step_deg[i] = deg
            if deg > MAX_TW:
                # over-width: stamp remaining steps and stop
                for j in range(i + 1, n):
                    step_deg[j] = deg
                break
            if succ:
                s = succ
                while s:
                    vbit = s & -s
                    s ^= vbit
                    v = vbit.bit_length() - 1
                    temp[v] |= succ ^ vbit
            # save a checkpoint every `stride` steps
            if (i + 1) % self.stride == 0 and (i + 1) < n:
                checkpoints.append((i + 1, list(temp)))

        self.cur_perm = perm
        self.cur_t = t
        self.cur_step_deg = step_deg
        self.checkpoints = checkpoints

    def _partial_walk(
        self, perm: List[int], start_step: int, start_temp: List[int],
    ) -> List[int]:
        """Walk perm from `start_step` to n-1, using `start_temp` as
        the initial state.  Returns the resulting step_deg array.
        Positions [0, start_step) are copied from the cache (they are
        unchanged because perm matches cur_perm there)."""
        n = self.n
        suffix_mask = self._suffix_mask(perm)
        temp = list(start_temp)
        step_deg = list(self.cur_step_deg)  # type: ignore[arg-type]

        for i in range(start_step, n):
            u = perm[i]
            succ = temp[u] & suffix_mask[i]
            deg = succ.bit_count()
            step_deg[i] = deg
            if deg > MAX_TW:
                for j in range(i + 1, n):
                    step_deg[j] = deg
                return step_deg
            if succ:
                s = succ
                while s:
                    vbit = s & -s
                    s ^= vbit
                    v = vbit.bit_length() - 1
                    temp[v] |= succ ^ vbit
        return step_deg

    def _isolated_walk(
        self, perm: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Full walk that does NOT touch the cache.  Used for
        big-change candidates that are evaluated speculatively."""
        n = self.n
        suffix_mask = self._suffix_mask(perm)
        temp = list(self.adj_bits)
        step_deg = [0] * n
        for i in range(n):
            u = perm[i]
            succ = temp[u] & suffix_mask[i]
            deg = succ.bit_count()
            step_deg[i] = deg
            if deg > MAX_TW:
                for j in range(i + 1, n):
                    step_deg[j] = deg
                return step_deg, temp
            if succ:
                s = succ
                while s:
                    vbit = s & -s
                    s ^= vbit
                    v = vbit.bit_length() - 1
                    temp[v] |= succ ^ vbit
        return step_deg, temp

    def _suffix_mask(self, perm: List[int]) -> List[int]:
        n = self.n
        suffix_mask = [0] * n
        cur = 0
        for i in range(n - 1, -1, -1):
            suffix_mask[i] = cur
            cur |= 1 << perm[i]
        return suffix_mask

    def _nearest_checkpoint_le(self, position: int) -> Tuple[int, List[int]]:
        """Return the (step, temp_snapshot) checkpoint whose step is
        the largest value <= position."""
        best = self.checkpoints[0]  # type: ignore[index]
        for cp in self.checkpoints:  # type: ignore[union-attr]
            if cp[0] <= position:
                best = cp
            else:
                break
        return best

    def _fitness_from_cache(self) -> Tuple[int, int]:
        return self._fitness_from(self.cur_step_deg, self.cur_t)  # type: ignore[arg-type]

    def _fitness_from(
        self, step_deg: List[int], t: int,
    ) -> Tuple[int, int]:
        n = self.n
        if t >= n:
            return (0, t)
        max_deg = 0
        for i in range(t, n):
            d = step_deg[i]
            if d > max_deg:
                max_deg = d
        if max_deg > MAX_TW:
            max_deg = MAX_TW + 1
        return (int(max_deg), int(t))


# ---------------------------------------------------------------------------
# Hypervolume (2-D, both axes minimised)
# ---------------------------------------------------------------------------

def hypervolume_2d(points: List[Tuple[int, int]], n: int) -> float:
    """Hypervolume against reference (n, n).  Matches pygmo for 2-D min/min.

    A point (x, y) is valid iff x < n AND y < n; it then dominates the
    rectangle [x, n] × [y, n].  HV = area of the union of those rectangles.
    """
    valid = [(x, y) for (x, y) in points if x < n and y < n]
    if not valid:
        return 0.0
    valid.sort(key=lambda p: (p[0], p[1]))
    nd: List[Tuple[int, int]] = []
    best_y = n
    for x, y in valid:
        if y < best_y:
            nd.append((x, y))
            best_y = y
    hv = 0.0
    for i, (x, y) in enumerate(nd):
        next_x = nd[i + 1][0] if i + 1 < len(nd) else n
        hv += (next_x - x) * (n - y)
    return hv


def score(decision_vectors: List[List[int]],
          adj_bits: List[int],
          n: int) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute the official leaderboard score = -HV for a list of decision
    vectors.  Returns (score, list of (max_degree, t) fitness points).
    """
    fits: List[Tuple[int, int]] = []
    for dv in decision_vectors:
        perm = dv[:-1]
        t = int(dv[-1])
        fits.append(evaluate(perm, t, adj_bits, n))
    hv = hypervolume_2d(fits, n)
    return -hv, fits


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------

class ParetoArchive:
    """Stores non-dominated (max_degree, t) points and their permutations.

    Both objectives minimised.  A point is dominated by another if both
    coordinates are <= and at least one is strictly <.  Capped solutions
    (max_degree > MAX_TW) are rejected entirely.
    """

    def __init__(self) -> None:
        # Each entry: (width, t, perm).  Sorted by (width asc, t asc).
        self._pts: List[Tuple[int, int, List[int]]] = []

    def __len__(self) -> int:
        return len(self._pts)

    def points(self) -> List[Tuple[int, int]]:
        return [(w, t) for (w, t, _) in self._pts]

    def entries(self) -> List[Tuple[int, int, List[int]]]:
        return list(self._pts)

    def try_add(self, w: int, t: int, perm: List[int],
                allow_overwidth: bool = False) -> bool:
        """Attempt to add (w, t, perm).  Returns True if accepted.

        Over-width points (w > MAX_TW) are rejected by default, matching
        the ESA contract that caps such solutions.  ``allow_overwidth``
        is an escape hatch used only by :func:`ensure_seeded` to inject a
        last-resort scoring point on instances where no feasible torso
        exists at any seeded threshold; it must never be set on the
        normal search path.
        """
        if w > MAX_TW and not allow_overwidth:
            return False
        for w2, t2, _ in self._pts:
            if w2 <= w and t2 <= t and (w2 < w or t2 < t):
                return False
        kept: List[Tuple[int, int, List[int]]] = []
        for w2, t2, p2 in self._pts:
            # Evict incumbents that the NEW point (w, t) dominates or equals.
            # (Fixed: the prior condition tested the reverse direction and so
            #  never pruned anything -- see docs/AUDIT.md BUG-1. HV is invariant
            #  to this, but the archive is now a proper Pareto set, which keeps
            #  top_k_by_hv_contribution's precondition valid and bounds growth.)
            if w2 >= w and t2 >= t and (w2 > w or t2 > t):
                continue
            if w == w2 and t == t2:
                continue
            kept.append((w2, t2, p2))
        kept.append((w, t, perm))
        kept.sort(key=lambda e: (e[0], e[1]))
        self._pts = kept
        return True

    def hypervolume(self, n: int) -> float:
        return hypervolume_2d(self.points(), n)

    def top_k_by_hv_contribution(
        self, k: int, n: int
    ) -> List[Tuple[int, int, List[int]]]:
        """Return up to `k` entries that maximise the hypervolume of the
        chosen subset.

        For 2-D non-dominated fronts this is the Hypervolume Subset Selection
        Problem (HSSP), which admits an optimal O(K · m²) dynamic program:

            f(i, j) = best HV using exactly j points whose leftmost (smallest
                      x = width) is point i.
            f(i, 1) = (n − x_i)(n − y_i)
            f(i, j) = max_{i' > i}  (x_{i'} − x_i)(n − y_i) + f(i', j − 1)

        The answer is max_i f(i, k).  We track successors to reconstruct the
        chosen subset.

        The previous greedy heuristic — drop the single point with the
        smallest individual contribution — can be sub-optimal whenever
        removing two cheap points to keep one slightly more expensive one
        would have raised total HV; the DP avoids that pitfall.
        """
        valid = [
            (w, t, p) for (w, t, p) in self._pts if w < n and t < n
        ]
        if len(valid) <= k:
            return valid
        pts = sorted(valid, key=lambda e: (e[0], e[1]))
        m = len(pts)
        NEG = -1.0
        dp = [[NEG] * m for _ in range(k + 1)]
        succ = [[-1] * m for _ in range(k + 1)]
        for i, (x, y, _) in enumerate(pts):
            dp[1][i] = (n - x) * (n - y)
        for j in range(2, k + 1):
            for i in range(m - j + 1):
                x_i, y_i, _ = pts[i]
                strip_h = n - y_i
                best = NEG
                best_ip = -1
                for ip in range(i + 1, m):
                    if dp[j - 1][ip] < 0:
                        continue
                    x_ip = pts[ip][0]
                    val = (x_ip - x_i) * strip_h + dp[j - 1][ip]
                    if val > best:
                        best = val
                        best_ip = ip
                dp[j][i] = best
                succ[j][i] = best_ip
        best_i = 0
        best_val = NEG
        for i in range(m):
            if dp[k][i] > best_val:
                best_val = dp[k][i]
                best_i = i
        chosen: List[Tuple[int, int, List[int]]] = []
        j = k
        cur = best_i
        while cur != -1 and j > 0:
            chosen.append(pts[cur])
            cur = succ[j][cur]
            j -= 1
        return chosen


def ensure_seeded(
    archive: "ParetoArchive",
    perm: List[int],
    adj_bits: List[int],
    n: int,
    t_fallback: int = 0,
) -> Tuple[int, int, List[int]]:
    """Return the lex-best archive entry to start a hill climb from.

    On the official instances the warm-start t-grid always seeds at least
    one feasible (width <= MAX_TW) point, so this is just
    ``archive.entries()[0]``.

    On pathologically dense instances every seeded torso can exceed
    MAX_TW, leaving the archive empty.  Indexing ``entries()[0]`` then
    raises ``IndexError`` -- the cause of the ``nan`` scores observed for
    every warm-started variant on the densest synthetic instances
    (e.g. inst_16_n750_d35).  In that case we inject a single over-width
    fallback point ``(width(perm, t_fallback), t_fallback)`` -- exactly
    the kind of point hc1/hc2 submit -- so the run still yields a scoring
    submission instead of crashing.  ``t_fallback = 0`` maximises the HV
    rectangle among the (all equally capped) over-width options.
    """
    if len(archive) == 0:
        w = evaluate(perm, t_fallback, adj_bits, n)[0]
        archive.try_add(int(w), int(t_fallback), list(perm),
                        allow_overwidth=True)
    return archive.entries()[0]


# ---------------------------------------------------------------------------
# Warm-start heuristics
# ---------------------------------------------------------------------------

def min_degree_perm(
    n: int, adj_bits: List[int], rng: random.Random | None = None
) -> List[int]:
    """Greedy minimum-degree elimination order.  Optional randomised tiebreaks."""
    g = list(adj_bits)
    remaining = (1 << n) - 1
    order: List[int] = []
    for _ in range(n):
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
        best_v = rng.choice(ties) if rng is not None else ties[0]
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


def min_fill_in_perm(
    n: int, adj_bits: List[int],
    sample_size: int | None = 64,
    rng: random.Random | None = None,
) -> List[int]:
    """Greedy minimum fill-in elimination order.  Picks the vertex whose
    elimination introduces the fewest new edges.  Bound per-step cost via
    `sample_size` (only the lowest-degree candidates are inspected)."""
    if rng is None:
        rng = random.Random(0)
    g = list(adj_bits)
    remaining = (1 << n) - 1
    order: List[int] = []

    for _ in range(n):
        candidates: List[Tuple[int, int]] = []
        r = remaining
        while r:
            vbit = r & -r
            r ^= vbit
            v = vbit.bit_length() - 1
            d = (g[v] & remaining).bit_count()
            candidates.append((d, v))
            if d == 0:
                break
        if candidates and candidates[-1][0] == 0:
            best_v = candidates[-1][1]
        else:
            candidates.sort(key=lambda x: x[0])
            if sample_size is not None and len(candidates) > sample_size:
                candidates = candidates[:sample_size]
            best_fill = 10 ** 18
            ties: List[int] = []
            for d, v in candidates:
                nbrs = g[v] & remaining
                existing = 0
                s = nbrs
                while s:
                    ubit = s & -s
                    s ^= ubit
                    u = ubit.bit_length() - 1
                    existing += (g[u] & nbrs).bit_count()
                existing //= 2
                fill = d * (d - 1) // 2 - existing
                if fill < best_fill:
                    best_fill = fill
                    ties = [v]
                    if fill == 0:
                        break
                elif fill == best_fill:
                    ties.append(v)
            best_v = rng.choice(ties)
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


def mcs_m_order(n: int, adj_bits: List[int]) -> List[int]:
    """MCS-M minimal-triangulation elimination order (Berry, Blair,
    Heggernes & Peyton 2004, *Maximum cardinality search for computing
    minimal triangulations of graphs*).

    Produces a *minimal* (not minimum) triangulation: no fill edge can be
    removed while keeping the graph chordal.  Returns an elimination order
    where index 0 is eliminated first (the reverse of the MCS-M numbering).

    The reachability rule ("u joins the fill set of the just-numbered v iff
    there is a v–u path through unnumbered vertices each of weight < w[u]")
    is evaluated with a Dijkstra-style relaxation in which the cost of a
    vertex is the maximum weight of any strictly-internal vertex on the
    cheapest path; u is reached iff that cost is < w[u].  Direct neighbours
    have cost −∞ and are always reached.  O(n·(m log n)).

    EMPIRICAL NOTE (this project): on the dense SpOC-3 synthetic cells
    inst_16 / inst_19 / inst_20 the minimal triangulation is *wider* than
    plain `min_fill_in_perm` (607/690/808 vs 576/635/781), so MCS-M does
    NOT break the 500-width feasibility wall there.  It is provided as a
    construction baseline; `min_fill_in_perm` remains the best warm start
    in the family.  See docs/RESULTS.md §5.3 and docs/FUTURE.md §2.3.
    """
    NEG = float("-inf")
    INF = float("inf")
    import heapq
    w = [0] * n
    numbered = bytearray(n)
    alpha = [0] * n
    for k in range(n, 0, -1):
        # choose the unnumbered vertex of maximum weight
        v, bw = -1, -1
        for x in range(n):
            if not numbered[x] and w[x] > bw:
                bw = w[x]
                v = x
        numbered[v] = 1
        alpha[v] = k
        dist = [INF] * n
        pq: List[Tuple[float, int]] = []
        m = adj_bits[v]
        while m:
            zb = m & -m
            m ^= zb
            z = zb.bit_length() - 1
            if not numbered[z]:
                dist[z] = NEG
                heapq.heappush(pq, (NEG, z))
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist[x]:
                continue
            base = d if d > w[x] else w[x]   # max(d, w[x])
            mx = adj_bits[x]
            while mx:
                zb = mx & -mx
                mx ^= zb
                z = zb.bit_length() - 1
                if not numbered[z] and base < dist[z]:
                    dist[z] = base
                    heapq.heappush(pq, (base, z))
        for u in range(n):
            if not numbered[u] and u != v and dist[u] < w[u]:
                w[u] += 1
    return sorted(range(n), key=lambda x: alpha[x])


def treewidth_lower_bound_mmd(n: int, adj_bits: List[int]) -> int:
    """Minor-min-width (MMD) treewidth lower bound (Bodlaender & Koster
    2011, *Treewidth computations II: Lower bounds*).

    Repeatedly take the minimum-degree vertex v, record deg(v) as a lower-
    bound candidate, then contract v into its lowest-degree neighbour.  The
    maximum recorded degree is a valid lower bound on the treewidth (and
    hence on the achievable `max_degree`).  Cheap but loose on dense random
    graphs.  Returns the lower bound as an int.
    """
    g = list(adj_bits)
    remaining = (1 << n) - 1
    lb = 0
    for _ in range(n):
        best_d = 1 << 62
        v = -1
        r = remaining
        while r:
            vb = r & -r
            r ^= vb
            x = vb.bit_length() - 1
            d = (g[x] & remaining).bit_count()
            if d < best_d:
                best_d = d
                v = x
        if v < 0:
            break
        if best_d > lb:
            lb = best_d
        nb = g[v] & remaining
        if nb:
            best_nd = 1 << 62
            u = -1
            s = nb
            while s:
                ub = s & -s
                s ^= ub
                y = ub.bit_length() - 1
                nd = (g[y] & remaining).bit_count()
                if nd < best_nd:
                    best_nd = nd
                    u = y
            g[u] |= (g[v] & remaining) & ~(1 << u)
        remaining ^= 1 << v
    return lb


# ---------------------------------------------------------------------------
# Warm-start selection
# ---------------------------------------------------------------------------

def build_warm_start(
    n: int,
    adj_bits: List[int],
    rng: random.Random | None = None,
    method: str = "auto",
) -> Tuple[List[int], str]:
    """Construct the warm-start elimination order, returning (perm, label).

    `method="auto"` (the family default): use minimum-fill-in
    (`min_fill_in_perm`) -- the strongest warm start in this family, see
    docs/RESULTS.md §5.3 -- on graphs where it is affordable, and fall back
    to minimum-degree on large *dense* graphs where min-fill's per-step fill
    scan is too slow and, empirically, no better (the dense regime is
    dominated by structural moves, not by the warm start).  The fallback
    fires when ``n * average_degree`` exceeds ~2e5 (e.g. the official
    large-graph: 2426 * 209 ≈ 5.1e5 -> min-degree).

    `method="min-fill"` / `method="min-degree"` force the choice.
    """
    if method == "min-degree":
        return min_degree_perm(n, adj_bits, rng=rng), "min-degree"
    if method == "min-fill":
        return min_fill_in_perm(n, adj_bits, rng=rng), "min-fill"
    if method != "auto":
        raise ValueError(f"unknown warm-start method: {method!r}")
    total_deg = sum(b.bit_count() for b in adj_bits)
    avg_deg = total_deg / max(1, n)
    if n * avg_deg > 200_000:
        return min_degree_perm(n, adj_bits, rng=rng), "min-degree (auto: dense)"
    return min_fill_in_perm(n, adj_bits, rng=rng), "min-fill (auto)"


# ---------------------------------------------------------------------------
# Local-search operators (single source of truth; variant files wrap these).
#
# Each core operator is a pure function returning a NEW permutation list and
# the (possibly updated) threshold t.  Variant files adapt them to their
# local (perm, n, t, bn_idx, bn_mask) calling convention with thin wrappers;
# op_min_fill_reinsert additionally takes the adjacency bitsets.
# ---------------------------------------------------------------------------

def op_or_opt(perm: List[int], n: int, t: int) -> Tuple[List[int], int]:
    """Or-opt / segment-shift: relocate a contiguous block of length 1-3 to
    a different position WITHOUT reversing it.  Complements 2-opt/3-opt
    (which only reverse) and block_move (which uses larger blocks): the
    short, order-preserving shift is the classic Or-opt neighbourhood
    (Or 1976).  No-op-safe for tiny n."""
    if n < 4:
        return perm[:], t
    block_len = random.randint(1, min(3, n - 1))
    start = random.randint(0, n - block_len)
    out = perm[:]
    block = out[start:start + block_len]
    del out[start:start + block_len]
    insert_pos = random.randint(0, len(out))
    out[insert_pos:insert_pos] = block
    return out, t


def op_bottleneck_relocate(
    perm: List[int], n: int, t: int, bn_idx: int,
) -> Tuple[List[int], int]:
    """Relocate the *bottleneck* vertex -- the one eliminated at the step
    achieving the max torso width (position ``bn_idx``) -- to a new position.

    Unlike ``bottleneck->head`` (which only moves it into the head
    ``[0, t)``), this can move it anywhere, biased (60%) toward an *earlier*
    position so the vertex is eliminated sooner, before it accumulates
    fill-in.  This ties the move causally to the objective instead of
    picking positions blindly.  Falls back to a random swap when no
    bottleneck is known."""
    if bn_idx is None or bn_idx < 0 or n < 3:
        i, j = random.sample(range(n), 2)
        out = perm[:]
        out[i], out[j] = out[j], out[i]
        return out, t
    out = perm[:]
    v = out.pop(bn_idx)
    if bn_idx > 0 and random.random() < 0.6:
        new_pos = random.randint(0, bn_idx - 1)
    else:
        new_pos = random.randint(0, len(out))
    out.insert(new_pos, v)
    return out, t


def op_min_fill_reinsert(
    perm: List[int], n: int, t: int, bn_idx: int, adj_bits: List[int],
) -> Tuple[List[int], int]:
    """Min-fill-guided reinsertion.  Remove one vertex (the bottleneck if
    known, else random) and reinsert it at the position -- among a small
    sampled candidate set -- that minimises the fill-in it would induce.

    A vertex eliminated before its neighbours forces those *later* neighbours
    to become a clique, so the fill it introduces is
    ``C(k, 2) - existing_edges`` where ``k`` is the number of its neighbours
    that end up after it in the order.  We evaluate this exactly for a handful
    of candidate positions (8 random samples plus the slots just before the
    first and just after the last neighbour) and keep the cheapest.  Cost is
    bounded by the small candidate set, so the operator is affordable even on
    dense graphs.  Falls back to a random swap if adjacency is unavailable."""
    if adj_bits is None or n < 3:
        i, j = random.sample(range(n), 2)
        out = perm[:]
        out[i], out[j] = out[j], out[i]
        return out, t
    out = perm[:]
    if bn_idx is not None and bn_idx >= 0 and random.random() < 0.5:
        src = bn_idx
    else:
        src = random.randrange(n)
    v = out.pop(src)
    nb = adj_bits[v]
    m = len(out)
    neigh_positions = [idx for idx, u in enumerate(out) if (nb >> u) & 1]
    cand_positions = {random.randint(0, m) for _ in range(8)}
    if neigh_positions:
        cand_positions.add(neigh_positions[0])
        cand_positions.add(neigh_positions[-1] + 1)
    best_pos = 0
    best_fill = None
    for p in cand_positions:
        later_bits = 0
        for idx in range(p, m):
            u = out[idx]
            if (nb >> u) & 1:
                later_bits |= 1 << u
        k = later_bits.bit_count()
        if k <= 1:
            fill = 0
        else:
            existing = 0
            s = later_bits
            while s:
                ub = s & -s
                s ^= ub
                u = ub.bit_length() - 1
                existing += (adj_bits[u] & later_bits).bit_count()
            existing //= 2
            fill = k * (k - 1) // 2 - existing
        if best_fill is None or fill < best_fill:
            best_fill = fill
            best_pos = p
    out.insert(best_pos, v)
    return out, t


# ---------------------------------------------------------------------------
# Submission output
# ---------------------------------------------------------------------------

def write_submission(
    decision_vectors: List[List[int]],
    problem: str,
    out_path: str,
) -> None:
    """Write a submission JSON in the canonical ESA format:
    [{challenge, problem, decisionVector: [...]}].
    For multi-objective problems decisionVector is a list of vectors.
    """
    payload = [{
        "challenge": CHALLENGE_ID,
        "problem": problem,
        "decisionVector": [list(map(int, dv)) for dv in decision_vectors],
    }]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def repo_root() -> str:
    """Absolute path to the repository root.

    core.py lives at the repo root, so the root is simply this file's
    directory.  Using this keeps the per-algorithm path bootstrap
    independent of how deep a variant file is nested under algorithms/
    (e.g. algorithms/hill_climbing/, algorithms/simulated_annealing/).
    """
    return os.path.dirname(os.path.abspath(__file__))


def graph_path(here: str, problem: str) -> str:
    return os.path.join(here, "data", f"{problem}.gr")


def _is_seed_stem(algo: str) -> bool:
    """A per-seed sweep stem ends in `_s<digits>` (e.g. grasp_s26, cmaes_s4,
    cmaesf_s6, sms_ls_s19).  These are routed into a `seeds/` subfolder so the
    handful of canonical method submissions stay uncluttered."""
    i = algo.rfind("_s")
    return i != -1 and algo[i + 2:].isdigit() and len(algo) > i + 2


def submission_path(here: str, problem: str, algo: str) -> str:
    base = os.path.join(here, "submissions", problem)
    if _is_seed_stem(algo):
        return os.path.join(base, "seeds", f"{algo}.json")
    return os.path.join(base, f"{algo}.json")

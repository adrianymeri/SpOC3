#!/usr/bin/env python3
"""
esa_eval.py -- the official scoring, ported unchanged from the main project.

This is the authoritative evaluator. It is a verbatim port of core.py from
the `test/gbdt-novelty` branch, which is documented as matching ESA's
`graph_torso_udp._perm2fitness`, and is the code every leaderboard-verified
score in that project was computed with.

Three things live here and nothing else touches them:

    evaluate(perm, t, adj_bits, n)   -> (width, t)
    hypervolume_2d(points, n)        -> area against the reference (n, n)
    ParetoArchive.top_k_by_hv_contribution(k, n)
                                     -> the optimal k-point subset (HSSP)

A caveat worth stating plainly: this is not literally ESA's own source file,
which was never published with the challenge materials. It is the evaluator
the main project used and validated against the live leaderboard over
several hundred submissions. If the official file ever becomes available,
swapping it in here is the only change required.

Keep this file boring. Everything else in the repository can be rewritten
for clarity; this one exists to be identical to what it was ported from.
"""

from __future__ import annotations


# corresponds to graph_torso_udp.max_tw
MAX_TW = 500


def evaluate(perm, t, adj_bits, n):
    """The official fitness vector (width, t).

    Builds the chordal-completion / fill-in elimination graph, records the
    degree only for steps i >= t, and aborts with width = MAX_TW + 1 if any
    step's degree exceeds the cap. Permutations with duplicates score
    (MAX_TW + 1, n).
    """
    if len(set(perm)) != len(perm):
        return (MAX_TW + 1, n)

    torso_size = n - t
    if torso_size <= 0:
        return (9999, t)

    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]

    temp = list(adj_bits)
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        deg = succ.bit_count()
        if i >= t and deg > max_width:
            max_width = deg
        # the cap applies at EVERY step, head included
        if deg > MAX_TW:
            return (MAX_TW + 1, t)
        if not succ:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= succ ^ vbit
    return (int(max_width), int(t))


def step_degrees(perm, adj_bits, n):
    """Every step's degree in one pass: deg[i] for i in 0..n-1.

    Not part of the official interface, but it is the same walk as
    evaluate() with the per-step numbers kept instead of maxed. One call
    gives the whole (width, t) staircase for a permutation.
    """
    suffix_mask = [0] * n
    cur = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = cur
        cur |= 1 << perm[i]

    temp = list(adj_bits)
    deg = [0] * n
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        deg[i] = succ.bit_count()
        if not succ:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= succ ^ vbit
    return deg


def build_adj_bitsets(n, adj):
    """Adjacency as one integer per vertex, bit u set when u is a neighbour."""
    bits = [0] * n
    for v in range(n):
        mask = 0
        for u in adj[v]:
            mask |= 1 << u
        bits[v] = mask
    return bits


def hypervolume_2d(points, n):
    """Hypervolume against the reference (n, n). Matches pygmo for 2-D min/min.

    A point (x, y) is valid iff x < n AND y < n; it then dominates the
    rectangle [x, n] x [y, n]. HV is the area of the union of those
    rectangles.
    """
    valid = [(x, y) for (x, y) in points if x < n and y < n]
    if not valid:
        return 0.0
    valid.sort(key=lambda p: (p[0], p[1]))
    nd = []
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


class ParetoArchive:
    """Non-dominated (width, t) points and the permutation behind each.

    Both objectives are minimised. A point is dominated when another is
    <= on both coordinates and < on at least one. Points over the width cap
    are rejected outright, matching the ESA contract.
    """

    def __init__(self):
        self._pts = []                      # (width, t, perm), sorted

    def __len__(self):
        return len(self._pts)

    def points(self):
        return [(w, t) for (w, t, _) in self._pts]

    def entries(self):
        return list(self._pts)

    def try_add(self, w, t, perm):
        """Add (w, t, perm) if it is not dominated. True if accepted."""
        if w > MAX_TW:
            return False
        for w2, t2, _ in self._pts:
            if w2 <= w and t2 <= t and (w2 < w or t2 < t):
                return False
        kept = []
        for w2, t2, p2 in self._pts:
            if w2 >= w and t2 >= t and (w2 > w or t2 > t):
                continue                    # the new point dominates this one
            if w == w2 and t == t2:
                continue
            kept.append((w2, t2, p2))
        kept.append((w, t, perm))
        kept.sort(key=lambda e: (e[0], e[1]))
        self._pts = kept
        return True

    def hypervolume(self, n):
        return hypervolume_2d(self.points(), n)

    def top_k_by_hv_contribution(self, k, n):
        """The k points maximising the hypervolume of the chosen subset.

        For 2-D non-dominated fronts this is the Hypervolume Subset Selection
        Problem, which has an optimal O(k * m^2) dynamic program:

            f(i, 1) = (n - x_i)(n - y_i)
            f(i, j) = max over i' > i of  (x_i' - x_i)(n - y_i) + f(i', j-1)

        The answer is max over i of f(i, k); successors are tracked so the
        chosen subset can be reconstructed.

        A greedy rule -- repeatedly drop the point with the smallest
        individual contribution -- can be beaten whenever discarding two
        cheap points to keep one costlier point would have raised the total.
        The DP does not have that failure mode.
        """
        valid = [(w, t, p) for (w, t, p) in self._pts if w < n and t < n]
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
                    val = (pts[ip][0] - x_i) * strip_h + dp[j - 1][ip]
                    if val > best:
                        best = val
                        best_ip = ip
                if best_ip >= 0:
                    dp[j][i] = best
                    succ[j][i] = best_ip

        best_val = NEG
        best_start = -1
        for i in range(m):
            if dp[k][i] > best_val:
                best_val = dp[k][i]
                best_start = i
        if best_start < 0:
            return pts[:k]

        chosen = []
        i = best_start
        for j in range(k, 0, -1):
            chosen.append(pts[i])
            i = succ[j][i]
            if i < 0:
                break
        return chosen

#!/usr/bin/env python3
"""
hc1_initial.py -- Hill Climbing with Random Restarts (Algorithm 10).

This is the very first file to read in this project.  It is intended
to be primitive: a single ``Solution`` class with explicit
``off_torso`` / ``in_torso`` lists, four simple operators, and
Algorithm 10 followed line by line.  It is the foundation that the
optimisations in hc1 ... hc6 (and the incremental evaluator in hc12)
build on top of.

A ``Solution`` carries:

    perm          the elimination order (a permutation of 0..n-1)
    torso_index   = t, the threshold separating off-torso from in-torso
    off_torso     list of vertices BEFORE the torso     ( = perm[:t] )
    in_torso      dict { v : original neighbours of v }
                  for every v in perm[t:]
                  -- the FULL original adjacency list, no torso filter
    fitness       = (max_degree, t)  -- the official two objectives

For the toy example from the README (10 vertices, t = 6,
perm = [0, 1, 2, 4, 5, 3, 6, 8, 9, 7]) printing a Solution gives:

    Solution: t = 6, fitness (max_degree, t) = (2, 6)
      off_torso (6 vertices): [0, 1, 2, 4, 5, 3]
      in_torso  (4 vertices, original neighbours):
        vertex 6: [1, 4]
        vertex 8: [2, 3]
        vertex 9: [3]
        vertex 7: [4]
"""

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

import numpy as np

from core import (
    repo_root,
    LEADERBOARD_TARGETS,
    MAX_TW,
    build_adj_bitsets,
    evaluate,
    graph_path,
    load_graph,
    submission_path,
    write_submission,
)


# ---------------------------------------------------------------------------
# Solution class
# ---------------------------------------------------------------------------

class Solution:
    """A single candidate solution.

    Builds three derived views in the constructor:
        off_torso  -- list of head vertices    ( = perm[:t] )
        in_torso   -- dict { v : original neighbours }
                      for every torso vertex v
        fitness    -- (max_degree, t)

    Whenever perm or t changes you build a new Solution (operators
    below do exactly that -- see line 7 of Algorithm 10:
    "R <- Tweak(Copy(S))").
    """

    def __init__(self, perm, t, adj, adj_bits, n):
        # store the raw decision variables
        self.perm = list(perm)
        self.torso_index = t

        # off-torso = vertices that get eliminated BEFORE the torso
        # starts.  These are the head of the permutation.
        self.off_torso = list(perm[:t])

        # in-torso = the actual torso.  Each vertex is shown together
        # with its ORIGINAL neighbours (no fill-in, no torso filter --
        # just the input graph adjacency, easiest to read).
        self.in_torso = {}
        for v in perm[t:]:
            self.in_torso[v] = sorted(adj[v])

        # fitness = (max_degree, t).  This call runs the official
        # bitset-accelerated chordal-completion evaluator on (perm, t).
        self.fitness = evaluate(perm, t, adj_bits, n)

    def quality(self):
        """Algorithm 10's Quality(S).

        In our problem, SMALLER (max_degree, t) is better (lex order).
        So Algorithm 10's  "Quality(R) > Quality(S)"  -- meaning "R is
        BETTER than S" -- becomes  R.fitness < S.fitness  in code.
        """
        return self.fitness

    def is_over_width(self):
        """True if the elimination hit the 500-width limit."""
        return self.fitness[0] > MAX_TW

    def show(self, max_lines=8):
        """Pretty-print the solution.  Truncates long lists."""
        print(f"Solution: t = {self.torso_index}, "
              f"fitness (max_degree, t) = {self.fitness}")

        head = self.off_torso
        head_str = str(head[:max_lines])
        if len(head) > max_lines:
            head_str = head_str[:-1] + ", ...]"
        print(f"  off_torso ({len(head)} vertices): {head_str}")

        print(f"  in_torso  ({len(self.in_torso)} vertices, "
              f"original neighbours):")
        shown = 0
        for v, nbrs in self.in_torso.items():
            if shown >= max_lines:
                remaining = len(self.in_torso) - max_lines
                print(f"    ... ({remaining} more)")
                break
            print(f"    vertex {v}: {nbrs}")
            shown += 1


# ---------------------------------------------------------------------------
# Operators -- the four simplest tweaks you can write.
#
# Each one takes (perm, t, n) and returns a NEW (perm, t) pair.
# Algorithm 10 line 7 says "R <- Tweak(Copy(S))" so we always work on
# a copy and never modify the caller's perm in place.
# ---------------------------------------------------------------------------

def op_shift_left(perm, t, n):
    # pick a random index i > 0 and swap perm[i-1] with perm[i]
    new_perm = list(perm)
    i = random.randrange(1, n)
    new_perm[i - 1], new_perm[i] = new_perm[i], new_perm[i - 1]
    return new_perm, t


def op_shift_right(perm, t, n):
    # pick a random index i < n-1 and swap perm[i] with perm[i+1]
    new_perm = list(perm)
    i = random.randrange(0, n - 1)
    new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
    return new_perm, t


def op_shuffle(perm, t, n):
    # shuffle the whole permutation
    new_perm = list(perm)
    random.shuffle(new_perm)
    return new_perm, t


def op_t_shift(perm, t, n):
    # move the threshold t by +1 or -1 (clamped to [0, n-1])
    delta = random.choice([-1, +1])
    new_t = max(0, min(n - 1, t + delta))
    return list(perm), new_t


OPERATORS = [
    ("shift_left",  op_shift_left),
    ("shift_right", op_shift_right),
    ("shuffle",     op_shuffle),
    ("t_shift",     op_t_shift),
]


def tweak(S, adj, adj_bits, n):
    """Algorithm 10 line 7: R <- Tweak(Copy(S)).

    Pick one of the four operators uniformly at random, apply it to a
    copy of S's perm/t, build the resulting Solution, and return it
    together with the operator name (just for our stats reporting).
    """
    name, op = random.choice(OPERATORS)
    new_perm, new_t = op(S.perm, S.torso_index, n)
    R = Solution(new_perm, new_t, adj, adj_bits, n)
    return name, R


def random_solution(adj, adj_bits, n):
    """Algorithm 10 line 2 / line 13: a fresh random candidate."""
    perm = list(range(n))
    random.shuffle(perm)
    t = random.randint(0, n - 1)
    return Solution(perm, t, adj, adj_bits, n)


# ---------------------------------------------------------------------------
# Algorithm 10 -- Hill Climbing with Random Restarts
# Line numbers in the comments match the textbook pseudocode.
# ---------------------------------------------------------------------------

def hc7(adj, adj_bits, n, total_budget_s, interval_min, interval_max):
    deadline = time.time() + total_budget_s

    # line 1: T = distribution of possible time intervals
    #         (we use uniform(interval_min, interval_max) seconds)

    # line 2: S <- some initial random candidate solution
    S = random_solution(adj, adj_bits, n)

    # line 3: Best <- S
    Best = S
    print(f"  initial S: fitness = {S.fitness}")

    # bookkeeping (not part of the algorithm; just for reporting)
    iters = 0
    accepts = 0
    restarts = 0
    op_calls   = {name: 0 for name, _ in OPERATORS}
    op_accepts = {name: 0 for name, _ in OPERATORS}

    # line 4: repeat (outer loop -- runs until total time is up)
    while time.time() < deadline:

        # line 5: time <- random time in the near future, chosen from T
        interval = random.uniform(interval_min, interval_max)
        phase_deadline = min(time.time() + interval, deadline)

        # line 6: repeat (inner loop -- climb until phase time is up)
        while time.time() < phase_deadline:
            iters += 1

            # line 7: R <- Tweak(Copy(S))
            op_name, R = tweak(S, adj, adj_bits, n)
            op_calls[op_name] += 1

            # line 8: if Quality(R) > Quality(S) then
            # (smaller is better in our problem, so the comparison
            #  flips to '<' -- see Solution.quality docstring)
            if R.fitness < S.fitness:
                # line 9: S <- R
                S = R
                accepts += 1
                op_accepts[op_name] += 1
        # line 10: until time is up

        # line 11: if Quality(S) > Quality(Best) then
        if S.fitness < Best.fitness:
            # line 12: Best <- S
            Best = S
            t_left = max(0.0, deadline - time.time())
            print(f"  new Best after {restarts} restart(s): "
                  f"fitness = {Best.fitness}  ({t_left:.1f}s left)")

        # line 13: S <- some random candidate solution
        if time.time() < deadline:
            S = random_solution(adj, adj_bits, n)
            restarts += 1
    # line 14: until total time is up

    # line 15: return Best
    stats = {
        "iters":      iters,
        "accepts":    accepts,
        "restarts":   restarts,
        "op_calls":   op_calls,
        "op_accepts": op_accepts,
    }
    return Best, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(problem, budget_s, seed, here,
        interval_min, interval_max, show_final):
    random.seed(seed)
    np.random.seed(seed)

    n, adj = load_graph(graph_path(here, problem))
    adj_bits = build_adj_bitsets(n, adj)

    print(f"\n=== hc1_initial -- {problem} ===")
    print(f"n = {n}, edges = {sum(len(a) for a in adj) // 2}")
    target = LEADERBOARD_TARGETS.get(problem)
    if target is not None:
        print(f"leaderboard target = {target:,}")
    print(f"budget = {budget_s:.0f}s, seed = {seed}")
    print(f"T = uniform({interval_min:.1f}, {interval_max:.1f}) "
          f"seconds per phase")
    print(f"operators = {[name for name, _ in OPERATORS]}")
    print()

    t0 = time.time()
    Best, stats = hc7(adj, adj_bits, n, budget_s,
                      interval_min, interval_max)
    elapsed = time.time() - t0

    print()
    print(f"iters = {stats['iters']:,}, accepts = {stats['accepts']:,}, "
          f"restarts = {stats['restarts']}, elapsed = {elapsed:.1f}s")

    print(f"\nOperator stats (calls, accepts, accept rate):")
    for name, _ in OPERATORS:
        c = stats['op_calls'][name]
        a = stats['op_accepts'][name]
        rate = a / c if c else 0.0
        print(f"  {name:12s}  {c:7,d} calls   {a:5,d} accepts   "
              f"({rate:6.2%})")

    print()
    print("Best solution found:")
    Best.show(max_lines=8)
    print()

    max_w, t = Best.fitness
    if Best.is_over_width():
        print(f"NOTE: max_degree = {max_w} -- elimination hit the "
              f"500-width limit.  Heavy HV penalty.")

    # single-point HV against the reference (n, n)
    if max_w < n and t < n:
        hv = (n - max_w) * (n - t)
    else:
        hv = 0
    score = -hv
    print(f"Single-point HV (vs ref ({n}, {n})): {hv:,}")
    print(f"Official score (-HV):                {score:,}")
    if target is not None:
        gap = score - target
        verdict = "BEAT" if gap < 0 else f"{abs(gap):,} short"
        print(f"Gap to target ({target:,}): {gap:+,}  ({verdict})")

    if show_final:
        print("\n--- full Best (no truncation) ---")
        Best.show(max_lines=10 ** 9)

    # write the submission
    decision_vector = list(Best.perm) + [int(Best.torso_index)]
    out_path = submission_path(here, problem, "hc1")
    write_submission([decision_vector], problem, out_path)
    print(f"\nWrote submission: {out_path}  (1 vector)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph",
                    choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--budget", type=float, default=30.0,
                    help="total wall-time budget in seconds")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--interval-min", type=float, default=1.0,
                    help="lower bound of T (seconds per inner phase)")
    ap.add_argument("--interval-max", type=float, default=3.0,
                    help="upper bound of T (seconds per inner phase)")
    ap.add_argument("--show-final", action="store_true",
                    help="print the full Best solution at the end "
                         "(no truncation)")
    args = ap.parse_args()

    here = repo_root()
    run(args.problem, args.budget, args.seed, here,
        args.interval_min, args.interval_max, args.show_final)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
meta.py -- shared pieces for simulated_annealing.py, vns.py and grasp.py.

Mirrors `algorithms/meta_common.py` in the main project, where the three
metaheuristics reuse the hill-climbing chapter's mature move pool rather than
re-deriving one each. Same idea here, with one important difference recorded
below.

The state is a (perm, t) pair
-----------------------------
hill_climbing.py searches permutations alone and reads a whole staircase off
each one. These three carry a **threshold as well**, because their acceptance
rules are scalar: they need one number for "how good is where I am standing",
and that is the area the single point (width_at(t), t) dominates against the
reference (n, n):

    E = -(n - width) * (n - t)          lower is better

Every candidate still donates its *entire* staircase to the shared front, so
the score is computed exactly as it is for every other solver here.

Why the move pool is bigger than hill climbing's four
-----------------------------------------------------
Measured on small-graph from a min-degree start, 400 random moves:

    the four generic operators     97.0% neutral    0.0% improving
    a bottleneck-targeted move      0.3% neutral

The torso width is `max(deg[t:])` -- a maximum over a suffix. One vertex
holds that maximum, and unless a move touches *that* vertex the objective
does not move at all. A generic swap almost never does, so the landscape a
generic pool sees is essentially flat, and no acceptance rule can search a
flat landscape: simulated annealing drifts, VNS cannot recover from a kick,
GRASP's descent has nothing to descend.

The reference implementations solve this with bottleneck-aware operators
(`bottleneck->head`, `bottleneck_relocate`, `min_fill_reinsert` in their
15-operator pool). The two below are the simple versions of that idea, plus
two moves on t, which hill_climbing.py has no need of.

This means the three metaheuristics search a **richer neighbourhood** than
hill_climbing.py. That is deliberate and it matches the original, where they
reuse hc9's pool while the teaching hill climber keeps its own. When reading
the comparison, remember that part of any gap is the move pool, not the
acceptance rule.

How much of it is the move pool: measured
-----------------------------------------
That caveat went unquantified for most of this chapter, so it was closed by
experiment. `move_perm` below gives hill_climbing.py this richer pool and
changes nothing else -- same min-degree construction, same greedy acceptance,
same eight target widths, same budget -- which isolates the move pool exactly
as grasp_front isolates the construction. Run as `--solver hc_bottleneck`.

    the move pool is worth  +9,562 HV   over ten instances, W-T-L 2-7-1

For scale, the baseline differs from an independent run of *itself* by 21,519
HV, so +9,562 is less than half of nothing. The confounding this docstring
warned about is real but small: it cannot account for any gap worth
discussing, and the front-aware variants (`solve_front`) avoid it entirely by
calling hill_climbing.Operators.ALL directly. See README Part 13.
"""

from __future__ import annotations

import random

from torso import Solution, MAX_WIDTH
import hill_climbing


def energy(solution, t, n):
    """Negative area dominated by the single point (width_at(t), t).

    Zero when the ordering breaks the width cap -- the worst possible value,
    since every legal point gives something strongly negative.
    """
    width = solution.width_at(t)
    if width > MAX_WIDTH:
        return 0.0
    return -float((n - width) * (n - t))


def cost_at(solution, target_width, n):
    """Smallest t reaching `target_width`; n if that width is unreachable.

    This is hill_climbing.HillClimber.cost. The front-aware variants of the
    three metaheuristics use it instead of `energy` above, so that they
    optimise the same thing Hill Climbing does -- one front region at a time,
    eight regions per run -- rather than the area of a single point. The
    score is the hypervolume of a staircase, so a method aimed at eight
    places on that staircase is optimising what is measured; one aimed at a
    single rectangle is optimising a proxy. Keeping the objective identical
    is what makes the acceptance rule the only thing that differs.
    """
    t = solution.best_t_for_width(target_width)
    return n if t is None else t


def bottleneck_position(solution, t):
    """Where the vertex sitting on max(deg[t:]) is -- the one setting width."""
    deg = solution.degrees()
    best, at = -1, t
    for i in range(t, len(deg)):
        if deg[i] > best:
            best, at = deg[i], i
    return at


# --- the moves. Each takes (graph, perm, t, rng) and returns (perm, t) -----

def t_shift(graph, perm, t, rng, solution=None):
    """Nudge the threshold by a small amount."""
    step = rng.randint(1, max(1, graph.n // 50))
    if rng.random() < 0.5:
        step = -step
    return list(perm), max(0, min(graph.n - 1, t + step))


def t_jump(graph, perm, t, rng, solution=None):
    """Put the threshold somewhere else entirely."""
    return list(perm), rng.randrange(graph.n)


def generic(graph, perm, t, rng, solution=None):
    """One of hill climbing's four permutation moves, t untouched."""
    return rng.choice(hill_climbing.Operators.ALL)(perm, rng), t


def bottleneck_earlier(graph, perm, t, rng, solution=None):
    """Pull the bottleneck vertex to a random earlier position.

    Eliminating it sooner means fewer of its neighbours are still around to
    be joined up, which is the direct way to attack the width.
    """
    i = bottleneck_position(solution or Solution(graph, perm), t)
    out = list(perm)
    v = out.pop(i)
    out.insert(rng.randrange(0, i + 1), v)
    return out, t


def bottleneck_relocate(graph, perm, t, rng, solution=None):
    """Move the bottleneck vertex anywhere at all."""
    i = bottleneck_position(solution or Solution(graph, perm), t)
    out = list(perm)
    v = out.pop(i)
    out.insert(rng.randrange(0, len(out) + 1), v)
    return out, t


# Weighted so most moves actually change the objective. The generic four are
# 97% neutral, so giving them half the pool meant half of every walk was
# accepted unconditionally regardless of temperature -- simulated annealing
# reported 91% acceptance and never cooled into a search. They are kept at a
# fifth because they are cheap and do occasionally help; the bottleneck and
# threshold moves carry the search.
MOVES = [
    (generic,             0.20),
    (bottleneck_earlier,  0.35),
    (bottleneck_relocate, 0.20),
    (t_shift,             0.20),
    (t_jump,              0.05),
]
_FNS = [m for m, _ in MOVES]
_WEIGHTS = [w for _, w in MOVES]


def move(graph, perm, t, rng, solution=None):
    """Apply one random move from the pool. Returns (perm, t).

    Pass the caller's Solution for `perm` when it already has one. The
    bottleneck moves need the degree array to find the vertex holding
    max(deg[t:]); without this they rebuild it, costing a second full
    evaluation per candidate -- measured at 1.5x the cost of a plain
    hill-climbing step.
    """
    fn = rng.choices(_FNS, weights=_WEIGHTS, k=1)[0]
    return fn(graph, perm, t, rng, solution)


# --- the width-aware pool, for the move-pool arm of the experiment --------

PERM_MOVES = [(generic,             0.267),
              (bottleneck_earlier,  0.466),
              (bottleneck_relocate, 0.267)]
_PFNS = [m for m, _ in PERM_MOVES]
_PWEIGHTS = [w for _, w in PERM_MOVES]


def move_perm(graph, perm, t, rng, solution=None):
    """One move from the bottleneck-aware pool, returning only a permutation.

    hill_climbing searches permutations at a *fixed* target width, so it has
    no free threshold to move: t is derived from the permutation, not chosen
    alongside it. This drops the two threshold moves from MOVES above and
    renormalises what is left, giving a pool that is hill_climbing's four
    generic operators plus the two that can actually see the vertex holding
    max(deg[t:]).

    That is the only difference between the hc_bottleneck arm of the
    controlled experiment and plain Hill Climbing -- same construction, same
    acceptance rule, same objective, same budget, better moves.
    """
    fn = rng.choices(_PFNS, weights=_PWEIGHTS, k=1)[0]
    return fn(graph, perm, t, rng, solution)[0]


def start_state(graph, rng, start="min_degree"):
    """Initial (perm, t): a construction, and the t its best point sits at."""
    perm = (hill_climbing.Starts.random_order(graph, rng) if start == "random"
            else hill_climbing.Starts.min_degree(graph, rng))
    solution = Solution(graph, perm)
    stairs = solution.staircase()
    if not stairs:
        return perm, 0, solution
    n = graph.n
    _, best_t = max(stairs, key=lambda wt: (n - wt[0]) * (n - wt[1]))
    return perm, best_t, solution

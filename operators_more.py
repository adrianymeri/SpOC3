#!/usr/bin/env python3
"""
operators_more.py -- additional move operators for hill_climbing.py.

Nothing in hill_climbing.py, torso.py, meta.py or esa_eval.py is modified.
`install()` adds names to Operators.NAMES / BY_NAME, so every existing
caller keeps working and `harness/ablation.py --operator <name>` can run a
new operator at exactly the protocol the current four were measured at.

WHY THESE MOVES
---------------
Two facts about this problem decide what a useful operator looks like.
Both follow from one observation: eliminating a SET of vertices leaves a
torso that does not depend on the order that set was eliminated in, so
deg[i] is a function only of the vertex at position i and the SET of
vertices before it.

(1) A move that rearranges positions [lo, hi] cannot change deg[i] for any
    i > hi -- the set before i contains the whole window either way -- nor
    for i < lo. So its effect is confined to the window, and its cost under
    an incremental evaluator is O(hi - lo). Picking two uniform positions
    gives a window of about n/3; a bounded window is far cheaper.

(2) Only a few positions can ever set the score. deg[i] matters to the
    staircase exactly when deg[i] > max(deg[i+1:]) -- those are the
    staircase corners, the positions where the running maximum steps up.
    A move whose window misses every corner leaves the staircase unchanged
    and therefore cannot improve the cost at ANY target width.

    `reverse_segment` spans 2-10 positions out of n, so it covers a corner
    only rarely. That is consistent with what the ablation already found:
    swap_neighbours accepted 1 move in ~1.5M attempts.

The families below are the two answers to those facts, plus two that use
graph structure the current four ignore entirely:

  bounded k-opt   or_opt, three_opt, k_opt_rotate
                  k-opt confined to a window. Plain TSP k-opt does not
                  transfer: there its appeal is O(1) delta from exchanging
                  k edges, but here a span-L rearrangement costs O(L)
                  whatever k is. reverse_segment is already a bounded
                  2-opt, so this extends a direction the file started.

  corner-aimed    *_aimed, and corner_shuffle
                  the same mechanisms with the window placed over a
                  staircase corner instead of at random. Mechanism and
                  placement are independent choices, so pairing each move
                  with its aimed twin isolates placement as one factor.

  min-fill        min_fill_window
                  destroy-and-repair: discard the order inside a window
                  and rebuild it greedily by fill-in (Markowitz 1957;
                  George & Liu 1981) against the live graph the climber
                  has actually reached.

  structure       simplicial_pull, degeneracy_window
                  a vertex whose surviving neighbours already form a
                  clique eliminates with zero fill-in; classical treewidth
                  preprocessing.

Standard library only, matching the rest of the project.
"""

from __future__ import annotations

import hill_climbing
from esa_eval import step_degrees

SPAN = 32          # bound on how many positions a windowed move disturbs

# Bound graph plus a one-entry deg[] memo. The climber calls operators many
# times with the SAME current.perm between accepts, so an O(n) list compare
# is a cheap way to avoid repeating the O(n x fill) walk.
_GRAPH = None
_MEMO_PERM = None
_MEMO_DEG = None


def bind(graph):
    """Point the structure-aware operators at the graph being solved."""
    global _GRAPH, _MEMO_PERM, _MEMO_DEG
    _GRAPH = graph
    _MEMO_PERM = None
    _MEMO_DEG = None


def _deg(perm):
    global _MEMO_PERM, _MEMO_DEG
    if _MEMO_PERM is not None and _MEMO_PERM == perm:
        return _MEMO_DEG
    _MEMO_DEG = step_degrees(perm, _GRAPH.bits, _GRAPH.n)
    _MEMO_PERM = list(perm)
    return _MEMO_DEG


def corners(deg):
    """Positions where the running maximum from the right steps up.

    These are exactly the staircase corners: the only positions whose
    degree can set the cost at any target width. One backward pass.
    """
    out = []
    run = -1
    for i in range(len(deg) - 1, -1, -1):
        if deg[i] > run:
            run = deg[i]
            out.append(i)
    return out


def _aim(perm):
    """A staircase corner to aim a window at, or None if unavailable."""
    if _GRAPH is None:
        return None
    spots = corners(_deg(perm))
    return spots[0] if spots else None


def _window(rng, n, span=SPAN, anchor=None, minimum=3):
    """[start, stop) with minimum <= length <= span, covering `anchor`."""
    minimum = max(2, min(minimum, n))
    hi = max(minimum, min(span, n))
    length = rng.randint(minimum, hi) if hi > minimum else minimum
    length = min(length, n)
    if anchor is None:
        start = rng.randrange(0, max(1, n - length + 1))
    else:
        lo = max(0, anchor - length + 1)
        up = min(anchor, n - length)
        start = rng.randint(lo, up) if lo <= up else lo
    start = max(0, min(start, n - length))
    return start, start + length


def _live(position, perm):
    """(work, alive) just before `position`: adjacency with fill-in, and the
    bitset of vertices not yet eliminated. Walks from the start, so it is
    only used by operators that need the real graph state."""
    n = _GRAPH.n
    later = [0] * n
    seen = 0
    for i in range(n - 1, -1, -1):
        later[i] = seen
        seen |= 1 << perm[i]
    work = list(_GRAPH.bits)
    for i in range(min(position, n)):
        u = perm[i]
        succ = work[u] & later[i]
        rest = succ
        while rest:
            low = rest & -rest
            rest ^= low
            v = low.bit_length() - 1
            work[v] |= succ ^ low
    alive = later[position - 1] if position > 0 else (1 << n) - 1
    return work, alive


# ---------------------------------------------------------------------------
# bounded k-opt
# ---------------------------------------------------------------------------

def _or_opt(perm, rng, anchor=None):
    n = len(perm)
    if n < 5:
        return hill_climbing.Operators.move_vertex(perm, rng)
    block = rng.randint(2, 3)
    if anchor is None:
        start = rng.randrange(0, n - block)
        offset = rng.randint(1, min(SPAN, n - block))
        if rng.random() < 0.5:
            offset = -offset
    else:
        start = max(0, min(n - block, anchor - rng.randint(0, block - 1)))
        offset = -rng.randint(1, min(SPAN, n - block))
    dest = max(0, min(n - block, start + offset))
    if dest == start:
        return list(perm)
    new = list(perm)
    chunk = new[start:start + block]
    del new[start:start + block]
    new[dest:dest] = chunk
    return new


def _three_opt(perm, rng, anchor=None):
    n = len(perm)
    if n < 6:
        return hill_climbing.Operators.reverse_segment(perm, rng)
    start, stop = _window(rng, n, max(6, SPAN), anchor, minimum=4)
    cut = rng.randint(start + 2, stop - 2) if stop - start >= 4 else start + 2
    new = list(perm)
    new[start:cut] = reversed(new[start:cut])
    new[cut:stop] = reversed(new[cut:stop])
    return new


def _k_opt_rotate(perm, rng, anchor=None):
    n = len(perm)
    if n < 4:
        return hill_climbing.Operators.swap_any(perm, rng)
    start, stop = _window(rng, n, SPAN, anchor)
    length = stop - start
    if length < 3:
        return list(perm)
    shift = rng.randrange(1, length)
    new = list(perm)
    chunk = new[start:stop]
    new[start:stop] = chunk[shift:] + chunk[:shift]
    return new


def or_opt(perm, rng):
    return _or_opt(perm, rng)


def three_opt(perm, rng):
    return _three_opt(perm, rng)


def k_opt_rotate(perm, rng):
    return _k_opt_rotate(perm, rng)


# ---------------------------------------------------------------------------
# corner-aimed
# ---------------------------------------------------------------------------

def or_opt_aimed(perm, rng):
    return _or_opt(perm, rng, _aim(perm))


def three_opt_aimed(perm, rng):
    return _three_opt(perm, rng, _aim(perm))


def k_opt_aimed(perm, rng):
    return _k_opt_rotate(perm, rng, _aim(perm))


def corner_shuffle(perm, rng):
    """Shuffle a bounded window guaranteed to contain a staircase corner.

    Makes no claim about WHICH rearrangement helps, but unlike a random
    shuffle it is at least capable of changing the staircase.
    """
    n = len(perm)
    start, stop = _window(rng, n, SPAN, _aim(perm))
    new = list(perm)
    chunk = new[start:stop]
    rng.shuffle(chunk)
    new[start:stop] = chunk
    return new


def corner_earlier(perm, rng):
    """Move the vertex at a staircase corner earlier in the order.

    Positions before the threshold are free, so pushing a wide step toward
    the front can drop the smallest t that stays within a target width --
    the quantity the climber minimises.
    """
    anchor = _aim(perm)
    if anchor is None or anchor == 0:
        return hill_climbing.Operators.move_vertex(perm, rng)
    dest = max(0, anchor - rng.randint(1, min(SPAN, anchor)))
    new = list(perm)
    new.insert(dest, new.pop(anchor))
    return new


# ---------------------------------------------------------------------------
# min-fill window
# ---------------------------------------------------------------------------

def _min_fill(perm, rng, anchor=None):
    n = len(perm)
    if _GRAPH is None:
        return hill_climbing.Operators.move_vertex(perm, rng)
    start, stop = _window(rng, n, SPAN, anchor, minimum=4)
    work, alive = _live(start, perm)
    work = list(work)
    pool = perm[start:stop]
    bits = 0
    for v in pool:
        bits |= 1 << v
    order = []
    live = alive
    while bits:
        best_v, best_cost = -1, None
        rest = bits
        while rest:
            low = rest & -rest
            rest ^= low
            v = low.bit_length() - 1
            nb = work[v] & live & ~(1 << v)
            cost = 0
            r2 = nb
            while r2:
                l2 = r2 & -r2
                r2 ^= l2
                u = l2.bit_length() - 1
                cost += (nb & ~work[u] & ~l2).bit_count()
            if best_cost is None or cost < best_cost:
                best_v, best_cost = v, cost
        order.append(best_v)
        bits &= ~(1 << best_v)
        nb = work[best_v] & live & ~(1 << best_v)
        r2 = nb
        while r2:
            l2 = r2 & -r2
            r2 ^= l2
            u = l2.bit_length() - 1
            work[u] |= nb & ~l2
        live &= ~(1 << best_v)
    new = list(perm)
    new[start:stop] = order
    return new


def min_fill_window(perm, rng):
    return _min_fill(perm, rng)


def min_fill_aimed(perm, rng):
    return _min_fill(perm, rng, _aim(perm))


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

def _degeneracy(perm, rng, anchor=None):
    n = len(perm)
    if _GRAPH is None:
        return hill_climbing.Operators.swap_any(perm, rng)
    start, stop = _window(rng, n, SPAN, anchor, minimum=4)
    work, alive = _live(start, perm)
    chunk = perm[start:stop]
    chunk.sort(key=lambda v: (work[v] & alive).bit_count())
    new = list(perm)
    new[start:stop] = chunk
    return new


def degeneracy_window(perm, rng):
    return _degeneracy(perm, rng)


def degeneracy_aimed(perm, rng):
    return _degeneracy(perm, rng, _aim(perm))


def simplicial_pull(perm, rng):
    """Pull a vertex that eliminates for free to the front of its window.

    Simplicial means the surviving neighbourhood is already a clique, so
    eliminating it adds no edges. Isolated vertices count -- vacuously, the
    empty set is a clique -- and they are most of the candidates near the
    end of the order, so excluding them makes the operator a no-op.
    """
    n = len(perm)
    if _GRAPH is None:
        return hill_climbing.Operators.move_vertex(perm, rng)
    start, stop = _window(rng, n, SPAN, _aim(perm), minimum=4)
    work, alive = _live(start, perm)
    movable = []
    for pos in range(start, stop):
        v = perm[pos]
        nb = work[v] & alive & ~(1 << v)
        clique = True
        rest = nb
        while rest and clique:
            low = rest & -rest
            rest ^= low
            u = low.bit_length() - 1
            if nb & ~work[u] & ~low:
                clique = False
        if clique and pos > start:
            movable.append(pos)
    if not movable:
        return list(perm)
    new = list(perm)
    new.insert(start, new.pop(rng.choice(movable)))
    return new


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

NEW = [
    ("or_opt", or_opt),
    ("three_opt", three_opt),
    ("k_opt_rotate", k_opt_rotate),
    ("or_opt_aimed", or_opt_aimed),
    ("three_opt_aimed", three_opt_aimed),
    ("k_opt_aimed", k_opt_aimed),
    ("corner_shuffle", corner_shuffle),
    ("corner_earlier", corner_earlier),
    ("min_fill_window", min_fill_window),
    ("min_fill_aimed", min_fill_aimed),
    ("degeneracy_window", degeneracy_window),
    ("degeneracy_aimed", degeneracy_aimed),
    ("simplicial_pull", simplicial_pull),
]

FAMILY = {
    "or_opt": "kopt", "three_opt": "kopt", "k_opt_rotate": "kopt",
    "or_opt_aimed": "aimed", "three_opt_aimed": "aimed",
    "k_opt_aimed": "aimed", "min_fill_aimed": "aimed",
    "degeneracy_aimed": "aimed",
    "corner_shuffle": "corner", "corner_earlier": "corner",
    "min_fill_window": "minfill",
    "degeneracy_window": "structure", "simplicial_pull": "structure",
}


def install(graph=None):
    """Add the new operators to hill_climbing.Operators, idempotently.

    The four originals keep their positions at the front of NAMES, so a run
    restricted to them draws the identical random stream it always did and
    the existing CSVs still reproduce.
    """
    ops = hill_climbing.Operators
    for name, fn in NEW:
        if name not in ops.BY_NAME:
            ops.NAMES.append(name)
            ops.ALL.append(fn)
            ops.BY_NAME[name] = fn
    if graph is not None:
        bind(graph)
    return [name for name, _ in NEW]

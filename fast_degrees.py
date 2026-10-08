#!/usr/bin/env python3
"""
fast_degrees.py -- delta evaluation: re-walk only the part that changed.

Addresses the first review point. Nothing is modified: `install()` swaps
`torso.step_degrees` for the function below, which returns exactly what
`esa_eval.step_degrees` returns and is checked against it. Because
`Solution.degrees()` looks the name up on the torso module, every caller --
hill_climbing.py, ablation.py, bench_one.py -- picks it up with no change
of their own.

THE PROBLEM
-----------
`step_degrees` walks all n steps. The climber builds a fresh Solution for
every candidate, so every candidate pays a full walk even when the operator
moved two positions.

THE KEY FACT
------------
Eliminating a SET S of vertices leaves a torso on the survivors that does
not depend on the ORDER S was eliminated in: two survivors end up adjacent
exactly when some path joins them through S, which says nothing about
order. So deg[i] is a function of just two things -- the vertex at position
i, and the SET of vertices at positions below i.

For a move that rearranges a window [lo, hi] without changing which
vertices occupy it (every permutation move is of that form, with lo and hi
the outermost changed positions):

    i <  lo    the set below i is untouched              -> deg[i] same
    lo<=i<=hi  the set below i differs                   -> may change
    i >  hi    the set below i contains the whole window
               either way, and perm[i] is unchanged      -> deg[i] SAME

That last line is the one that pays. The obvious incremental scheme replays
[lo, n), everything from the first change to the end -- but the walk's cost
peaks in the MIDDLE positions, because succ = temp[u] & suffix_mask[i] is
small early (little fill-in yet) and small late (few survivors left). So
replaying to the end redoes the expensive part and saves almost nothing.
Stopping at hi is the whole trick.

Cost: O(stride + (hi - lo)) instead of O(n), where stride = n / snapshots.

HOW IT KNOWS THE WINDOW
-----------------------
It does not need to be told. It caches the last permutation it walked, and
diffs the incoming one against it. Consecutive candidates in a climb are
both derived from the same current solution, so the diff is bounded by the
operators' spans -- which is why the bounded-span moves in
operators_more.py are cheap here and the unbounded ones are not.

A NOTE ON THE 500 CAP
---------------------
`esa_eval.step_degrees` computes every step degree and never checks the
cap; the cap is applied afterwards by `Solution.width_at` and
`staircase()`. So this must not short-circuit on it either, or it would
agree on every candidate under the cap and disagree on every candidate
over it -- which the wide-span moves breach often on medium-graph.

Standard library only.
"""

from __future__ import annotations

from esa_eval import step_degrees as reference_step_degrees

_SNAPSHOTS = 64

_state = {
    "n": 0,
    "bits": None,
    "perm": None,
    "deg": None,
    "later": None,
    "snaps": [],          # [(step, work[] before that step), ...]
    "stride": 1,
}

stats = {"full": 0, "window": 0, "steps_walked": 0, "steps_saved": 0}


def changed_window(a, b):
    """Outermost positions where two permutations differ, as (lo, hi).

    (None, None) when identical. Costs O(lo + n - hi), so it is cheapest
    exactly when the window is wide and the replay is expensive anyway.
    """
    n = len(a)
    lo = 0
    while lo < n and a[lo] == b[lo]:
        lo += 1
    if lo == n:
        return None, None
    hi = n - 1
    while hi > lo and a[hi] == b[hi]:
        hi -= 1
    return lo, hi


def _later_from(perm, n, start, base):
    """later[i] = bitset of vertices after position i, rebuilt from start-1.

    For i below start-1 the SET after i is invariant under a window move at
    lo >= start, so those entries are reused from the cache.
    """
    later = [0] * n if base is None else list(base)
    seen = 0
    lowest = max(start, 1) - 1
    for i in range(n - 1, lowest - 1, -1):
        later[i] = seen
        seen |= 1 << perm[i]
    return later


def _full_walk(perm, bits, n):
    """Walk everything, rebuilding the cache. Mirrors esa_eval.step_degrees."""
    later = _later_from(perm, n, 0, None)
    work = list(bits)
    deg = [0] * n
    stride = max(1, n // _SNAPSHOTS)
    snaps = [(0, list(work))]
    for i in range(n):
        u = perm[i]
        succ = work[u] & later[i]
        deg[i] = succ.bit_count()
        if succ:
            rest = succ
            while rest:
                low = rest & -rest
                rest ^= low
                v = low.bit_length() - 1
                work[v] |= succ ^ low
        if (i + 1) % stride == 0 and (i + 1) < n:
            snaps.append((i + 1, list(work)))
    _state.update(n=n, bits=bits, perm=list(perm), deg=deg, later=later,
                  snaps=snaps, stride=stride)
    stats["full"] += 1
    stats["steps_walked"] += n
    return deg


def _snapshot_at_or_before(position):
    best = _state["snaps"][0]
    for snap in _state["snaps"]:
        if snap[0] <= position:
            best = snap
        else:
            break
    return best


def step_degrees(perm, adj_bits, n):
    """Drop-in for esa_eval.step_degrees, returning the identical deg[].

    Falls back to a full walk whenever the cache cannot be reused: a
    different graph, a first call, or a change spanning the whole order.
    """
    if (_state["perm"] is None or _state["n"] != n
            or _state["bits"] is not adj_bits):
        return _full_walk(perm, adj_bits, n)

    lo, hi = changed_window(_state["perm"], perm)
    if lo is None:
        return _state["deg"]
    if lo <= 0 and hi >= n - 1:
        return _full_walk(perm, adj_bits, n)

    start, base_work = _snapshot_at_or_before(lo)
    later = _later_from(perm, n, start, _state["later"])
    work = list(base_work)
    deg = list(_state["deg"])
    stride = _state["stride"]
    new_snaps = []

    for i in range(start, hi + 1):
        u = perm[i]
        succ = work[u] & later[i]
        deg[i] = succ.bit_count()
        if succ:
            rest = succ
            while rest:
                low = rest & -rest
                rest ^= low
                v = low.bit_length() - 1
                work[v] |= succ ^ low
        if (i + 1) % stride == 0 and (i + 1) < n:
            new_snaps.append((i + 1, list(work)))

    # Snapshots past hi stay valid: the elimination state there depends only
    # on the SET of earlier vertices, which a window move leaves alone. They
    # may differ in bits belonging to already-eliminated vertices, but every
    # read is masked by later[i], so those bits are never observable.
    kept_low = [s for s in _state["snaps"] if s[0] <= start]
    kept_high = [s for s in _state["snaps"] if s[0] > hi]
    _state.update(perm=list(perm), deg=deg, later=later,
                  snaps=kept_low + new_snaps + kept_high)
    stats["window"] += 1
    stats["steps_walked"] += hi - start + 1
    stats["steps_saved"] += n - (hi - start + 1)
    return deg


def reset():
    """Drop the cache. Call between instances."""
    _state.update(n=0, bits=None, perm=None, deg=None, later=None,
                  snaps=[], stride=1)


def install():
    """Point torso.step_degrees at the delta version. Idempotent."""
    import torso
    if getattr(torso.step_degrees, "__module__", None) != __name__:
        torso.step_degrees = step_degrees
    return True


def uninstall():
    """Restore the original, for A/B measurement."""
    import torso
    torso.step_degrees = reference_step_degrees
    reset()
    return True


def report():
    total = stats["steps_walked"] + stats["steps_saved"]
    frac = stats["steps_saved"] / total if total else 0.0
    return (f"delta evaluator: {stats['window']:,} windowed, "
            f"{stats['full']:,} full, "
            f"{stats['steps_walked']:,} steps walked, "
            f"{frac:.1%} of step-work skipped")

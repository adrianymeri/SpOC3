#!/usr/bin/env python3
"""hri_plain.py -- PLAIN HRI: the winners' MO-LNS as THEY ran it.

Every hri_lns arm in this campaign is (a) warm-started from our pooled archive
and (b) accepts on the exact capped-20 objective.  HRI did NEITHER: they ran
from scratch and (being a multi-objective LNS) accepted on ARCHIVE-FRONT
improvement, not on a 20-point truncation.  Consequence: all our arms inherit
-- and polish -- the same deep local basin.  A from-scratch full-front run
explores a DIFFERENT basin, which is exactly what a frozen capped score calls
for (cf. small-graph: its wall fell to a fresh basin draw, not to force).

Same operators as hri_lns (Limmer p.c.; balanced repair = MEDIAN PLACEMENT,
Biedl et al. DAM 148 (2005) Sec. 5), but:
  * population seeded from randomised min-degree orderings (no pool),
  * acceptance on full-front hypervolume of a PERSISTENT archive (no cap),
  * writes hri_plain_s<seed>.json so it never clobbers the pooled arms; the
    normal cap_submit pooling picks it up like any other source.

v2 (2026-08-02): the archive is now an incremental staircase (numpy suffix-min)
instead of a per-iteration ParetoArchive rebuild -- the rebuild cost O(pop*n)
ParetoArchive.try_add calls per trial and held the loop to ~1 it/s, which is
far too slow to reach a leaderboard front.  Same acceptance semantics, ~100x
the throughput.  Verify the incremental HV against core.hypervolume_2d with
    python3 tools/hri_plain.py --problem medium-graph --iters 300 --verify

    python3 tools/hri_plain.py --problem medium-graph --iters 100000000 --seed 1
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC
from tools.hri_lns import neighbor_destroy, balanced_repair, random_destroy_repair


# ---------------------------------------------------------------- staircase

def staircase(perm, ev, n):
    """Compressed Pareto staircase of one ordering: [(w, min t), ...],
    strictly increasing in w and decreasing in t.  None if head is over-cap."""
    df = np.asarray(ev.full(perm), dtype=np.int64)
    r = np.maximum.accumulate(df[::-1])          # r[i] = width for t = n-1-i
    if r[-1] > MAX_TW:
        return None
    idx = np.flatnonzero(np.append(np.diff(r) != 0, True))   # last i per level
    return list(zip(r[idx].tolist(), ((n - 1) - idx).tolist()))


class Front:
    """Persistent uncapped archive as a monotone staircase over w = 0..MAX_TW.

    bestt[w] = smallest t reachable with width <= w  (n = 'not reachable').
    HV against reference (n, n) equals sum_w mult[w] * (n - bestt[w]), where
    mult accounts for the widths above MAX_TW that inherit bestt[MAX_TW]
    (points with w > MAX_TW are rejected outright by the ESA contract)."""

    def __init__(self, n):
        self.n = n
        self.bestt = np.full(MAX_TW + 1, n, dtype=np.int64)
        self.owner = np.full(MAX_TW + 1, -1, dtype=np.int64)
        self.mult = np.ones(MAX_TW + 1, dtype=np.int64)
        self.mult[MAX_TW] = n - MAX_TW
        self.hv = 0.0

    def gain(self, pts):
        """Would these points improve the front?  (cheap accept test)"""
        ws = np.fromiter((w for w, _ in pts), dtype=np.int64, count=len(pts))
        ts = np.fromiter((t for _, t in pts), dtype=np.int64, count=len(pts))
        return bool(np.any(ts < self.bestt[ws])), ws, ts

    def add(self, ws, ts, oid):
        new = self.bestt.copy()
        np.minimum.at(new, ws, ts)
        np.minimum.accumulate(new, out=new)      # a point at w covers all w' > w
        d = self.bestt - new
        self.hv += float((d * self.mult).sum())
        self.owner[d > 0] = oid
        self.bestt = new

    def steps(self):
        """Distinct staircase corners: [(w, t, owner_id), ...]."""
        out, prev = [], self.n
        for w in range(MAX_TW + 1):
            t = int(self.bestt[w])
            if t < prev:
                out.append((w, t, int(self.owner[w])))
                prev = t
        return out


# ------------------------------------------------------------------- starts

def mindeg_perm(n, adj_sets, rng):
    """Randomised min-degree ordering (a legal, non-pool starting point)."""
    deg = {v: len(adj_sets[v]) for v in range(n)}
    remaining = set(range(n))
    nbr = {v: set(adj_sets[v]) for v in range(n)}
    order = []
    while remaining:
        m = min(deg[v] for v in remaining)
        cands = [v for v in remaining if deg[v] <= m + 1]
        v = rng.choice(cands)
        order.append(v); remaining.discard(v)
        nb = nbr[v] & remaining
        for u in nb:
            nbr[u] |= (nb - {u})
            deg[u] = len(nbr[u] & remaining)
    return order


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--iters", type=int, default=100000000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pop", type=int, default=20, help="population size")
    ap.add_argument("--destroy-cap", type=int, default=220)
    ap.add_argument("--small-k", type=int, default=6)
    ap.add_argument("--stall", type=int, default=3000)
    ap.add_argument("--start", default="mindeg", choices=["mindeg", "random"])
    ap.add_argument("--save-every", type=float, default=60.0, help="seconds")
    ap.add_argument("--verify", action="store_true",
                    help="cross-check incremental HV against core.hypervolume_2d")
    a = ap.parse_args()

    n, adj_sets = load_graph(graph_path(HERE, a.problem))
    adj = [sorted(s) for s in adj_sets]
    ab = build_adj_bitsets(n, adj_sets)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS.get(a.problem)
    rng = random.Random(a.seed)

    # ---- FROM-SCRATCH population (no pool!) ----
    front = Front(n)
    perms, members = {}, []          # perms: id -> ordering (owners + population)
    nid = 0
    for mode_start in ([a.start, "mindeg"] if a.start == "random" else [a.start]):
        tries = 0
        while len(members) < a.pop and tries < a.pop * 20:
            tries += 1
            p = (mindeg_perm(n, adj_sets, rng) if mode_start == "mindeg"
                 else rng.sample(range(n), n))
            pts = staircase(p, ev, n)
            if pts is None:
                continue
            ok, ws, ts = front.gain(pts)
            perms[nid] = p
            if ok:
                front.add(ws, ts, nid)
            members.append(nid); nid += 1
        if members:
            if mode_start != a.start:
                print(f"  [--start {a.start} gave no feasible ordering "
                      f"(width > {MAX_TW}); fell back to mindeg]", flush=True)
            break
    if not members:
        print("could not build a feasible starting population"); return

    cur = -front.hv
    print(f"=== PLAIN HRI-LNS {a.problem} | from-scratch (pop {len(members)}, "
          f"seed {a.seed}) | full-front HV {cur:,.0f}"
          f"{f'  vs target {target:,.0f}' if target else ''} ===", flush=True)

    out = os.path.join(HERE, "submissions", a.problem, f"hri_plain_s{a.seed}.json")

    def save():
        arc = ParetoArchive()
        for w, t, oid in front.steps():
            arc.try_add(w, t, perms[oid])
        top = arc.top_k_by_hv_contribution(60, n)
        json.dump({"challenge": "spoc-3-torso-decompositions",
                   "problem": a.problem,
                   "decisionVector": [list(p) + [int(t)] for (_, t, p) in top]},
                  open(out, "w"))

    accepts = 0; since = 0; mode = "B"; t0 = time.time(); last_save = 0.0
    for it in range(a.iters):
        src = perms[members[rng.randrange(len(members))]]
        if mode == "B":
            d = neighbor_destroy(src, adj, rng, a.destroy_cap)
            cand = balanced_repair(src, d, adj, rng)
        else:
            cand = random_destroy_repair(src, rng, a.small_k)
        since += 1
        pts = staircase(cand, ev, n)
        ok = False
        if pts is not None:
            ok, ws, ts = front.gain(pts)
        if ok:
            front.add(ws, ts, nid)
            perms[nid] = cand; members.append(nid); nid += 1
            cur = -front.hv; accepts += 1; since = 0

            # population management: keep the orderings owning the most corners
            if len(members) > a.pop:
                own = {}
                for o in front.owner.tolist():
                    if o >= 0:
                        own[o] = own.get(o, 0) + 1
                keep = sorted(members[:-1], key=lambda m: -own.get(m, 0))[:a.pop - 1]
                members = keep + [members[-1]]
            live = set(members) | {int(o) for o in front.owner.tolist() if o >= 0}
            if len(perms) > len(live) + 64:
                perms = {k: v for k, v in perms.items() if k in live}

            print(f"  it {it} [{mode}]: *** front HV {cur:,.0f} (accept #{accepts})"
                  f"{f'  gap {cur-target:+,.0f}' if target else ''} ***", flush=True)
            now = time.time()
            if now - last_save >= a.save_every:
                save(); last_save = now

            if a.verify and accepts % 25 == 0:
                arc = ParetoArchive()
                for w, t, oid in front.steps():
                    arc.try_add(w, t, perms[oid])
                ref = -hypervolume_2d(arc.points(), n)
                flag = "OK" if abs(ref - cur) < 0.5 else f"MISMATCH ref={ref:,.0f}"
                print(f"  [verify @ accept {accepts}: {flag}]", flush=True)

        if since >= a.stall:
            mode = "A" if mode == "B" else "B"; since = 0
            print(f"  [stall -> operator set {mode}]", flush=True)
        if it % 2000 == 0 and it:
            el = time.time() - t0
            print(f"  [it {it} [{mode}] front HV {cur:,.0f}  {accepts} accepts  "
                  f"{el:.0f}s  {it/max(el,1e-9):.1f} it/s]", flush=True)
    save()


if __name__ == "__main__":
    main()

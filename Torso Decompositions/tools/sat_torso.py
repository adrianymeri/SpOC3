#!/usr/bin/env python3
r"""
sat_torso.py -- SAT-based exact treewidth for the torso bands (a second, engine-
independent exact oracle, distinct from our branch-and-bound and from Tamaki PID).

Strategy: treewidth is invariant under *simplicial reductions* -- repeatedly
eliminating a vertex of degree <= k whose neighbourhood is already a clique adds
no fill and cannot raise the width.  We peel those for free (they carry no
treewidth), shrinking each torso to its small irreducible CORE, then decide
"tw(core) <= k" with the Samer-Veith SAT encoding solved by a modern CDCL solver
(python-sat).  This is O(m^3) in the core size m, so it is only practical once the
core is small (<~150) -- which is exactly what the reductions deliver on these
sparse torsos.

Uses:
  # second-engine exact confirmation of each band's width (rigor)
  python3 tools/sat_torso.py --problem small-graph --verify --bands 1,2,3,4,5,6,7
  # exact treewidth of the whole graph (settles the MMD-loose gap, §5.2)
  python3 tools/sat_torso.py --problem small-graph --whole --tw-timeout 3600
  # attempt a bigger torso at a band (the beat): single-vertex grows, SAT-checked
  python3 tools/sat_torso.py --problem small-graph --grow --bands 3,4,5 --tw-timeout 120

Requires:  pip install python-sat[pblib,aiger]
Honest expectation: strong for the low/mid bands and rigor; the high-band cores
are the same hard structure that times out PID, so a beat here is a long shot.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import load_graph, build_adj_bitsets, graph_path, ParetoArchive, MAX_TW, LEADERBOARD_TARGETS
import tools.torso_deletion as td
from tools.fastwalk import IncEvalC


# ---------- treewidth-preserving reductions (peel the easy periphery) ----------
def reduce_core(adj, k):
    """adj: dict v->set(neighbours).  Eliminate simplicial / almost-simplicial
    vertices of degree <= k (no fill added, width <= k locally).  Returns
    (core_adj, ok) where ok=False means a vertex of degree <= k was forced to add
    fill that *exceeds* a clique -> inconclusive by reduction alone (go to SAT);
    ok=True with empty core means tw <= k proven by reductions."""
    a = {v: set(ns) for v, ns in adj.items()}
    changed = True
    while changed:
        changed = False
        for v in list(a):
            d = len(a[v])
            if d == 0:
                del a[v]; changed = True; break
            if d <= k:
                ns = a[v]
                # simplicial: neighbours already form a clique -> free elimination
                if all(ns - {u} <= a[u] for u in ns):
                    for u in ns:
                        a[u] |= ns; a[u].discard(u); a[u].discard(v)
                    del a[v]; changed = True; break
    return a


# ---------- Samer-Veith SAT encoding of "tw(core) <= k" ----------
def tw_le_sat(adj, k, deadline):
    """Exact: does the graph (dict v->set) have treewidth <= k?  Reductions then
    SAT on the core.  Returns True / False, or None if the SAT call is cut off."""
    core = reduce_core(adj, k)
    if not core:
        return True                      # reductions eliminated everything
    V = sorted(core)
    if len(V) <= k + 1:
        return True                      # <=k+1 vertices -> tw <= k trivially
    idx = {v: i for i, v in enumerate(V)}
    m = len(V)
    E = [(idx[u], idx[v]) for u in V for v in core[u] if idx[u] < idx[v] and v in idx]
    try:
        from pysat.formula import IDPool
        from pysat.card import CardEnc, EncType
        from pysat.solvers import Cadical153 as Solver
    except Exception as e:
        raise SystemExit(f"python-sat not installed ({e}); run: pip install python-sat[pblib,aiger]")
    vp = IDPool()
    def O(i, j):            # order var: True => i before j  (stored for i<j)
        return vp.id(("o", i, j)) if i < j else -vp.id(("o", j, i))
    def A(i, j):            # arc var i->j  (i before j and adjacent at elim)
        return vp.id(("a", i, j))
    cls = []
    # arc only in the 'before' direction
    for i in range(m):
        for j in range(m):
            if i != j:
                cls.append([-A(i, j), O(i, j)])
    # each original edge induces an arc in the order direction
    for (i, j) in E:
        cls.append([-O(i, j), A(i, j)])
        cls.append([-O(j, i), A(j, i)])
    # transitivity of the order
    for i in range(m):
        for j in range(m):
            if j == i: continue
            for l in range(m):
                if l == i or l == j: continue
                cls.append([-O(i, j), -O(j, l), O(i, l)])
    # fill rule: common predecessor i -> j,l become adjacent
    for i in range(m):
        for j in range(m):
            if j == i: continue
            for l in range(j + 1, m):
                if l == i: continue
                cls.append([-A(i, j), -A(i, l), -O(j, l), A(j, l)])
                cls.append([-A(i, j), -A(i, l), -O(l, j), A(l, j)])
        if time.time() > deadline:
            return None
    s = Solver(bootstrap_with=cls)
    # width: each vertex has <= k outgoing arcs
    top = vp.top + 1
    for i in range(m):
        lits = [A(i, j) for j in range(m) if j != i]
        cnf = CardEnc.atmost(lits, bound=k, top_id=top, encoding=EncType.seqcounter)
        top = cnf.nv + 1
        for c in cnf.clauses:
            s.add_clause(c)
    if time.time() > deadline:
        s.delete(); return None
    res = s.solve()              # Cadical; fast on the small reduced cores
    s.delete()
    return bool(res)


def load_front(here, problem, n, ev):
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            p = json.load(open(fp)); e = p[0] if isinstance(p, list) else p
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; d = ev.full(perm); r = 0
                    for t in range(n - 1, -1, -1):
                        c = int(d[t]); r = c if c > r else r
                        if r <= MAX_TW: arc.try_add(r, t, perm)
        except Exception:
            continue
    by_w = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, list(p))
    return by_w


def adjd(Slist, ab, n):
    tor, _ = td.torso_adj(Slist, ab, n); a = {s: set() for s in Slist}
    for s in Slist:
        x = tor[s]
        while x:
            b = x & -x; x &= x - 1; u = b.bit_length() - 1
            if u in a: a[s].add(u)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--grow", action="store_true")
    ap.add_argument("--bands", default="1,2,3,4,5")
    ap.add_argument("--tw-timeout", type=float, default=120.0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n)
    bands = [int(x) for x in a.bands.split(",") if x.strip()]

    if a.whole:
        full = adjd(list(range(n)), ab, n)
        print(f"whole-graph: reducing core then SAT (heuristic tw = 15) ...", flush=True)
        for k in range(2, 16):
            t0 = time.time()
            r = tw_le_sat(full, k, time.time() + a.tw_timeout)
            print(f"  tw <= {k}?  {r}   [{time.time()-t0:.0f}s, core after reduction varies]", flush=True)
            if r is True:
                print(f"  -> exact treewidth = {k}"); break
            if r is None:
                print("  -> SAT cut off; core too large at this k"); break
        return

    by_w = load_front(here, a.problem, n, ev)

    if a.verify:
        for W in bands:
            if W not in by_w: continue
            S = list(by_w[W][1][by_w[W][0]:]); core = reduce_core(adjd(S, ab, n), W)
            t0 = time.time()
            le = tw_le_sat(adjd(S, ab, n), W, time.time() + a.tw_timeout)
            lt = tw_le_sat(adjd(S, ab, n), W - 1, time.time() + a.tw_timeout) if le else False
            exact = (W if le and not lt else ('<=%d' % (W-1) if lt else '>%d' % W))
            print(f"  band w={W}: |S|={len(S)} core={len(core)} -> tw {exact} "
                  f"(expected {W})  [{time.time()-t0:.0f}s]", flush=True)
        return

    if a.grow:
        full_set = set(range(n))
        for W in bands:
            if W not in by_w: continue
            t, perm = by_w[W]; S = set(perm[t:]); Sm = 0
            for s in S: Sm |= 1 << s
            X = sorted((u for u in range(n) if not (Sm >> u) & 1),
                       key=lambda u: (ab[u] & Sm).bit_count())
            won = False; tested = 0; t0 = time.time()
            for v in X:
                if time.time() - t0 > a.tw_timeout * 20: break
                r = tw_le_sat(adjd(list(S) + [v], ab, n), W, time.time() + a.tw_timeout)
                tested += 1
                if r is True:
                    print(f"  band {W}: *** SAT WIN -- grown torso {len(S)+1} has tw<={W} (v={v}) ***",
                          flush=True); won = True; break
            if not won:
                print(f"  band {W}: {tested} grows SAT-checked, none reach tw<={W}", flush=True)


if __name__ == "__main__":
    main()

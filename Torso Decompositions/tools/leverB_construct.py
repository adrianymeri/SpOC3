#!/usr/bin/env python3
"""
leverB_construct.py -- Lever B: min-union K500-external head composition.

Certificate (docs/PLANTED_STRUCTURE_CERTIFICATES.md) reduces the large-graph
unlock region w in [82,180) to a per-width selection problem: to build a
width-w head we need (499-w) vertices of K500.  375 are the zero-fill twins;
the remaining (124-w) must be K500 *externals*, whose edges OUT of the clique
land on the 25 planted low-treewidth components and cause the fill that keeps
those components locked out of the torso until high w.

clique_prefix.py picks those externals by outside-degree (the v0 ranker, which
§15.6 showed loses).  §6f pinpointed the real objective: choose the externals
that MINIMISE THE UNION of touched planted-component vertices -- a weighted
min-union, small enough for greedy-with-exact-tiebreak.  Fewer distinct
component vertices dirtied  ->  components stay low-fill  ->  their unlock
breakpoint shifts left  ->  the capped-20 front captures more area.

Everything else (twin quota, tail templates, eval, banking) mirrors
clique_prefix.py so the two are directly comparable -- min-union is the only
change, which is exactly the ablation the thesis wants (§6/§15.6).

    python3 tools/leverB_construct.py --wmin 82 --wmax 180 --templates cap20.json
    python3 tools/cap_submit.py --problem large-graph | grep -iE "best-20|residual"
"""
from __future__ import annotations
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"


def find_cliques(n, adj):
    """The three planted cliques = closed nbhds of the big true-twin classes."""
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    out = []
    for vs in h.values():
        if len(vs) >= 50:
            K = set(vs) | set(adj[vs[0]])
            out.append((sorted(vs), sorted(K)))
    out.sort(key=lambda ck: -len(ck[1]))
    assert [len(k) for _, k in out] == [500, 400, 300], \
        f"unexpected clique sizes {[len(k) for _, k in out]}"
    ks = [set(k) for _, k in out]
    assert not (ks[0] & ks[1]) and not (ks[0] & ks[2]) and not (ks[1] & ks[2])
    return out


def planted_components(n, adj, glue):
    """Connected components of G minus the 1200 glue vertices."""
    comp = {}
    cid = 0
    for s in range(n):
        if s in glue or s in comp:
            continue
        stack = [s]; comp[s] = cid
        while stack:
            u = stack.pop()
            for w2 in adj[u]:
                if w2 not in glue and w2 not in comp:
                    comp[w2] = cid; stack.append(w2)
        cid += 1
    return comp, cid


def minunion_select(externals, attach, k, adj):
    """Greedy min-union: pick k externals minimising |union of attachments|.
    Tie-break by smaller outside-degree (fewer component edges)."""
    chosen, U, pool = [], set(), list(externals)
    for _ in range(k):
        best, best_gain, best_deg = None, None, None
        for v in pool:
            gain = len(attach[v] - U)
            deg = len(attach[v])
            if best is None or gain < best_gain or (gain == best_gain and deg < best_deg):
                best, best_gain, best_deg = v, gain, deg
        chosen.append(best); U |= attach[best]; pool.remove(best)
    return chosen, U


def head_for_width(w, cliques, adj, ext500, attach500, rank):
    """Quota head for width w. K500 externals by min-union (rank='minunion')
    or outside-degree (rank='outdeg', = clique_prefix baseline)."""
    degs = [499, 399, 299]
    parts = []
    for i, ((twins, K), d) in enumerate(zip(cliques, degs)):
        q = max(0, d - w)
        Kset = set(K); twinset = set(twins)
        take = list(twins[:q])
        if q > len(twins):
            need = q - len(twins)
            if i == 0:  # K500: the binding clique (§6f)
                if rank == "minunion":
                    sel, _ = minunion_select(ext500, attach500, need, adj)
                else:
                    sel = sorted(ext500, key=lambda v: (len(attach500[v]), len(adj[v])))[:need]
            else:       # K400/K300: sacrifice-free at low w, outdeg is fine
                ext = [v for v in K if v not in twinset]
                ext.sort(key=lambda v: (len(adj[v] - Kset), len(adj[v])))
                sel = ext[:need]
            take = list(twins) + sel
        parts.append((take, twinset))
    tw_part = [v for take, twinset in parts for v in take if v in twinset]
    Kall = {}
    for (twins, K) in cliques:
        for v in K:
            Kall[v] = set(K)
    ex_part = [v for take, twinset in parts for v in take if v not in twinset]
    ex_part.sort(key=lambda v: len(adj[v] - Kall[v]))
    return tw_part + ex_part


def load_templates(names, n):
    perms = []
    for name in names:
        fp = os.path.join(HERE, "submissions", PROBLEM, name)
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1:
                    p = [int(x) for x in dv[:-1]]
                    if sorted(p) == list(range(n)):
                        perms.append((int(dv[-1]), p))
        except Exception as ex:
            print(f"  template {name}: skipped ({ex})")
    seen, out = set(), []
    for t, p in sorted(perms):
        if t not in seen:
            seen.add(t); out.append((t, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmin", type=int, default=82)
    ap.add_argument("--wmax", type=int, default=180)
    ap.add_argument("--templates", default="cap20.json")
    ap.add_argument("--max-templates", type=int, default=8)
    ap.add_argument("--rank", default="minunion", choices=["minunion", "outdeg"])
    ap.add_argument("--algo", default="leverB")
    a = ap.parse_args()

    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ab = build_adj_bitsets(n, adj_l)
    ev = IncEvalC(ab, n)
    target = LEADERBOARD_TARGETS[PROBLEM]
    cliques = find_cliques(n, adj)
    glue = set()
    for twins, K in cliques:
        glue |= set(K)
    comp, ncomp = planted_components(n, adj, glue)
    print(f"cliques {[len(k) for _,k in cliques]} | glue {len(glue)} | "
          f"planted components {ncomp} ({n-len(glue)} vertices)")

    twins500, K500 = cliques[0]
    twinset500 = set(twins500)
    ext500 = [v for v in K500 if v not in twinset500]       # 125 externals
    # attachment set: the planted-component vertices each K500 external touches
    attach500 = {v: (adj[v] - glue) for v in ext500}
    tot = sum(len(attach500[v]) for v in ext500)
    print(f"K500 externals {len(ext500)} | total attachment slots {tot} "
          f"(min-union works this set)")

    templates = load_templates(a.templates.split(","), n)[:a.max_templates]
    print(f"{len(templates)} tail templates (t={[t for t,_ in templates]})")
    if not templates:
        print("NO templates loaded -- pass --templates with a valid submission file"); return

    arc = ParetoArchive()
    t0 = time.time(); results = []
    for w in range(a.wmin, a.wmax):
        H = head_for_width(w, cliques, adj, ext500, attach500, a.rank)
        Hset = set(H); tpos = len(H); best = None
        for tt, tp in templates:
            perm = H + [v for v in tp if v not in Hset]
            df = ev.full(perm)
            if max(int(x) for x in df) > MAX_TW:
                continue
            r = 0
            for t in range(n - 1, -1, -1):
                c = int(df[t]); r = c if c > r else r
                if r <= MAX_TW:
                    arc.try_add(r, t, perm)
            wt = max(int(df[i]) for i in range(tpos, n))
            if best is None or wt < best:
                best = wt
        results.append((w, tpos, best))
        if w % 10 == 0 or w == a.wmax - 1:
            print(f"  w={w:3d} quota-t={tpos:3d} achieved-width={best} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    hv = -arc.hypervolume(n)
    top = arc.top_k_by_hv_contribution(20, n)
    cap = -hypervolume_2d([(x, t) for x, t, _ in top], n)
    print(f"\n[{a.rank}] constructed-archive alone: envelope {hv:,.0f} | "
          f"cap20 {cap:,.0f} (target {target:,.0f})")
    out = os.path.join(HERE, "submissions", PROBLEM, f"{a.algo}.json")
    top60 = arc.top_k_by_hv_contribution(60, n)
    write_submission([list(p) + [int(t)] for (_, t, p) in top60], PROBLEM, out)
    print(f"wrote {out} ({len(top60)} points) -- now: python3 tools/cap_submit.py "
          f"--problem large-graph")


if __name__ == "__main__":
    main()

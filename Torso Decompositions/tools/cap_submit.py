#!/usr/bin/env python3
r"""
cap_submit.py -- build the VALID (<=20-point) leaderboard submission and report
where the search effort is worth spending.

ESA scores at most 20 front points (docs/EXPLAINER.md, PROBLEM.md).  Our engines
optimise the full per-width envelope (medium: 249 points), but only the best 20
count.  This tool:

  1. pools every ordering in submissions/<problem>/ into the per-width envelope;
  2. selects the HV-optimal <=20-point subset (greedy is optimal here -- HV is
     submodular over a 2-D staircase; verified within +1 HV of local search);
  3. writes that as the canonical valid submission  cap20.json;
  4. reports, for each of the 20 chosen widths, its current torso size and its
     MARGINAL HV VALUE = (width-gap to the next chosen point) -- i.e. how much
     one extra torso-vertex at that width is worth to the submitted score.  The
     widths are printed sorted by value: that is the priority list for a
     width-focused search (gbfcpp --only-widths).

This makes the search cap-aware: spend effort where it converts to leaderboard HV,
not on the ~229 invisible bands.

    python3 tools/cap_submit.py --problem medium-graph --k 20
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC


def build_envelope(here, problem, n, ev):
    arc = ParetoArchive()
    for fp in glob.glob(os.path.join(here, "submissions", problem, "*.json")):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e["decisionVector"]:
                if isinstance(dv, list) and len(dv) == n + 1 and \
                        sorted(int(x) for x in dv[:-1]) == list(range(n)):
                    perm = [int(x) for x in dv[:-1]]; df = ev.full(perm); r = 0
                    for t in range(n - 1, -1, -1):
                        c = int(df[t]); r = c if c > r else r
                        if r <= MAX_TW: arc.try_add(r, t, perm)
        except Exception:
            continue
    # per-width best (smallest t = largest torso)
    by_w = {}
    for w, t, p in arc.entries():
        if w not in by_w or t < by_w[w][0]: by_w[w] = (t, list(p))
    return by_w


def greedy_select(pts, n, k):
    """pts: list of (w,t,perm).  Return indices of the HV-optimal <=k subset."""
    def hv(idx): return hypervolume_2d([(pts[i][0], pts[i][1]) for i in idx], n)
    sel, rest = [], set(range(len(pts)))
    for _ in range(min(k, len(pts))):
        best, bg = None, -1.0
        for j in rest:
            g = hv(sel + [j])
            if g > bg: bg, best = g, j
        sel.append(best); rest.discard(best)
    # one local-search polish pass (cheap; usually +0)
    cur = hv(sel); selset = set(sel); improved = True
    while improved:
        improved = False
        for i in list(selset):
            for j in range(len(pts)):
                if j in selset: continue
                cand = set(selset); cand.discard(i); cand.add(j)
                h = hv(list(cand))
                if h > cur + 1e-9:
                    selset, cur = cand, h; improved = True; break
            if improved: break
    return sorted(selset), cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="medium-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(a.problem)

    by_w = build_envelope(here, a.problem, n, ev)
    pts = [(w, t, p) for w, (t, p) in sorted(by_w.items())]
    full_hv = hypervolume_2d([(w, t) for w, t, _ in pts], n)
    # EXACT selection: the 2-D HSSP dynamic program (same as the search arms).
    arc = ParetoArchive()
    for w, tt, p in pts:
        arc.try_add(w, tt, p)
    top = arc.top_k_by_hv_contribution(a.k, n)
    exact_hv = hypervolume_2d([(w, tt) for (w, tt, _) in top], n)
    sel_g, cap_hv_g = greedy_select(pts, n, a.k)
    if exact_hv >= cap_hv_g:
        cap_hv = exact_hv
        key = {(w, tt) for (w, tt, _) in top}
        sel = [i for i, (w, tt, _) in enumerate(pts) if (w, tt) in key][:a.k]
    else:                                   # defensive: keep whichever is better
        sel, cap_hv = sel_g, cap_hv_g
    print(f"=== cap_submit {a.problem} | envelope {len(pts)} pts ===")
    print(f"full envelope (uncapped, NOT submittable): {-full_hv:,.0f}")
    print(f"best-{a.k} submission (VALID):              {-cap_hv:,.0f}"
          f"{f'   gap {-cap_hv-target:+,.0f}' if target else ''}")
    print(f"cap cost (envelope - best{a.k}):            {full_hv-cap_hv:,.0f} HV")
    if target:
        print(f"residual to target (needs bigger torsos):  {(-full_hv)-target:+,.0f} HV "
              f"even with UNLIMITED points")

    # marginal value of each chosen width = width-gap to the next chosen point
    sw = sorted(sel, key=lambda i: pts[i][0])
    print(f"\n  width | torso |  marginal HV value (gap x 1 vertex) -- attack priority")
    rows = []
    for r, i in enumerate(sw):
        w = pts[i][0]; tor = n - pts[i][1]
        nxt = pts[sw[r + 1]][0] if r + 1 < len(sw) else n
        val = nxt - w
        rows.append((val, w, tor))
    for val, w, tor in sorted(rows, reverse=True):
        print(f"   {w:4d} | {tor:5d} |  {val:5d}")

    # write the valid submission
    dvs = [list(pts[i][2]) + [int(pts[i][1])] for i in sel]
    out = os.path.join(here, "submissions", a.problem, "cap20.json")
    json.dump({"challenge": "spoc-3-torso-decompositions", "problem": a.problem,
               "decisionVector": dvs}, open(out, "w"))
    print(f"\nwrote VALID submission -> {out}  ({len(dvs)} points)")
    # official Optimize-platform wrapper (top-level array; README format)
    pout = out.replace(".json", "_platform.json")
    json.dump([{"decisionVector": dvs, "problem": a.problem,
                "challenge": "spoc-3-torso-decompositions"}], open(pout, "w"))
    print(f"wrote platform-format copy   -> {pout}")
    prio = ",".join(str(w) for _, w, _ in sorted(rows, reverse=True))
    print(f"\nattack-priority widths (feed to gbfcpp --only-widths):\n{prio}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gaps_ablation_stats.py -- multi-seed statistics for the GAPS GBDT ablation (§10).

Turns the single-seed +24,766 HV result into a statistically grounded one. Score
every with-GBDT run (`gaps`, `gaps_s*`) and every without-GBDT control
(`gaps_nogbdt`, `gaps_nogbdt_s*`) with the OFFICIAL top-20 submission HV, and
report mean ± std for each arm plus the paired contribution.

Run the seeds on a GPU runtime first, e.g.:
    for s in 1 2 3 4 5: gaps_search.py --algo gaps_s$s        --gbdt-every 8 --seed $s ...
    for s in 1 2 3 4 5: gaps_search.py --algo gaps_nogbdt_s$s --gbdt-every 0 --seed $s ...
then:
    python3 tools/gaps_ablation_stats.py --problem large-graph
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, json, re, statistics
from core import load_graph, build_adj_bitsets, graph_path, repo_root, ParetoArchive, MAX_TW


def official_top20(fp, ab, n):
    try: dvs = json.load(open(fp))[0]["decisionVector"]
    except Exception: return None
    arch = ParetoArchive(); perms = set()
    for dv in dvs:
        if isinstance(dv, list) and len(dv) == n + 1 and \
                sorted(int(x) for x in dv[:-1]) == list(range(n)):
            perms.add(tuple(int(x) for x in dv[:-1]))
    for p in perms:
        sm = [0]*n; cur = 0
        for i in range(n-1, -1, -1): sm[i] = cur; cur |= 1 << p[i]
        tmp = list(ab); deg = [0]*n; cap = False
        for i in range(n):
            s = tmp[p[i]] & sm[i]; d = s.bit_count(); deg[i] = d
            if d > MAX_TW: cap = True; break
            x = s
            while x:
                b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
        if cap: continue
        run = 0
        for t in range(n-1, -1, -1):
            if deg[t] > run: run = deg[t]
            arch.try_add(int(run), t, list(p))
    top = arch.top_k_by_hv_contribution(20, n)
    sub = ParetoArchive()
    for w, t, p in top: sub.try_add(w, t, p)
    return -sub.hypervolume(n)


def collect(here, prob, ab, n, with_re, without_re):
    sub = os.path.join(here, "submissions", prob)
    w, wo = {}, {}
    files = glob.glob(os.path.join(sub, "*.json")) + \
            glob.glob(os.path.join(sub, "seeds", "*.json"))   # seeded stems route to seeds/
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if re.fullmatch(without_re, stem):
            s = official_top20(fp, ab, n); wo[stem] = s if s else None
        elif re.fullmatch(with_re, stem):
            s = official_top20(fp, ab, n); w[stem] = s if s else None
    return {k: v for k, v in w.items() if v}, {k: v for k, v in wo.items() if v}


def summarize(name, d):
    vals = list(d.values())
    if not vals:
        print(f"  {name}: (no runs found)"); return None
    mean = statistics.mean(vals); sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  {name}: n={len(vals)}  mean={mean:,.0f}  std={sd:,.0f}  "
          f"min={min(vals):,.0f}  max={max(vals):,.0f}")
    for k in sorted(d): print(f"      {k:<18}{d[k]:,.0f}")
    return mean, sd, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    args = ap.parse_args()
    here = repo_root()
    n, adj = load_graph(graph_path(here, args.problem)); ab = build_adj_bitsets(n, adj)
    # match gaps / gaps_s<d> (with) and gaps_nogbdt / gaps_nogbdt_s<d> (without)
    w, wo = collect(here, args.problem, ab, n,
                    with_re=r"gaps(_s\d+)?", without_re=r"gaps_nogbdt(_s\d+)?")
    print(f"\n=== GAPS GBDT ablation, multi-seed ({args.problem}) ===")
    print("WITH GBDT:")
    sw = summarize("with-GBDT", w)
    print("WITHOUT GBDT (control):")
    swo = summarize("without-GBDT", wo)
    if sw and swo:
        contrib = sw[0] - swo[0]
        print("\n" + "="*70)
        print(f"GBDT contribution in GAPS:  {contrib:>+12,.0f} HV "
              f"(mean with-GBDT {sw[0]:,.0f}  −  mean without {swo[0]:,.0f})")
        if len(sw[2]) > 1 and len(swo[2]) > 1:
            pooled = (statistics.pstdev(sw[2])**2 + statistics.pstdev(swo[2])**2) ** 0.5
            print(f"separation: {abs(contrib)/ (pooled+1e-9):.1f}× the combined run-to-run std "
                  f"({pooled:,.0f}) — {'clearly significant' if abs(contrib) > 2*pooled else 'check more seeds'}")
        print("="*70)


if __name__ == "__main__":
    main()

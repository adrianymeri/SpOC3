#!/usr/bin/env python3
"""paired_stats.py -- shared readout for the paired GBDT ablations.

Reads a RESULT file (one line per arm) and reports the paired comparison with
BOTH tests:

  sign test      -- uses only the direction of each pair. Robust, but weak:
                    with 8 pairs it cannot reach p < 0.05 unless the result is
                    a perfect 8-0 (7-1 gives p = 0.070).
  Wilcoxon
  signed-rank    -- uses the MAGNITUDE of each pair's difference as well, so a
                    consistent direction with a couple of small reversals can
                    still be significant at n = 8. Computed EXACTLY here by
                    enumerating all 2^m sign assignments (no scipy, no normal
                    approximation), which is valid for the small m we have.

Report the Wilcoxon result as primary and the sign test as the robustness
check; say which was pre-specified.

    python3 tools/paired_stats.py ablation_grow_medium-graph.txt --metric delta_hv
    python3 tools/paired_stats.py ablation_endpoint_small-graph.txt --metric endpoint
"""
from __future__ import annotations
import argparse, itertools, sys


def parse(path):
    rows = {}
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        d = dict(kv.split("=", 1) for kv in line.split()[1:])
        rows.setdefault(int(d["seed"]), {})[d["mode"]] = d
    return rows


def exact_sign_p(w, l):
    from math import comb
    n = w + l
    if n == 0:
        return None
    k = min(w, l)
    return min(sum(comb(n, i) for i in range(k + 1)) * 2 / (2 ** n), 1.0)


def exact_wilcoxon_p(diffs):
    """Exact two-sided Wilcoxon signed-rank p-value. diffs: nonzero floats,
    positive = treatment better."""
    d = [x for x in diffs if x != 0]
    m = len(d)
    if m == 0:
        return None, None
    if m > 18:                       # enumeration would blow up; caller warned
        return None, m
    order = sorted(range(m), key=lambda i: abs(d[i]))
    rank = [0.0] * m
    i = 0
    while i < m:                     # average ranks within ties on |d|
        j = i
        while j + 1 < m and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    obs = sum(rank[i] for i in range(m) if d[i] > 0)
    total = sum(rank)
    count = 0
    for signs in itertools.product((0, 1), repeat=m):
        wp = sum(rank[i] for i in range(m) if signs[i])
        # two-sided: as or more extreme than observed, either direction
        if wp >= max(obs, total - obs) or wp <= min(obs, total - obs):
            count += 1
    return min(count / (2 ** m), 1.0), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--metric", default="delta_hv",
                    help="delta_hv (grow ablation) or endpoint (endpoint ablation)")
    a = ap.parse_args()

    rows = parse(a.path)
    if a.metric == "endpoint":
        A, B, better = "gbdt", "random", "lower"
    else:
        A, B, better = "gbdt", "nogbdt", "higher"

    diffs = []
    w = l = t = 0
    zero = 0
    print(f"  {'seed':>4} | {'GBDT':>14} | {'control':>14} | winner")
    for s in sorted(rows):
        p = rows[s]
        if A not in p or B not in p:
            continue
        if a.metric == "endpoint":
            g = (int(p[A]["best_width"]), int(p[A]["best_bottleneck"]))
            r = (int(p[B]["best_width"]), int(p[B]["best_bottleneck"]))
            # lower is better; difference in bottleneck at equal width
            dv = (r[0] - g[0]) * 10000 + (r[1] - g[1])
            gs, rs = str(g), str(r)
        else:
            gv = float(p[A]["delta_hv"]); rv = float(p[B]["delta_hv"])
            dv = gv - rv
            if gv == 0 and rv == 0:
                zero += 1
            gs, rs = f"{gv:,.0f}", f"{rv:,.0f}"
        if dv > 0:
            win = "GBDT"; w += 1
        elif dv < 0:
            win = "control"; l += 1
        else:
            win = "tie"; t += 1
        diffs.append(dv)
        print(f"  {s:>4} | {gs:>14} | {rs:>14} | {win}")

    print(f"\n  pairs: {w+l+t}   GBDT wins {w} | losses {l} | ties {t}")
    ps = exact_sign_p(w, l)
    if ps is not None:
        print(f"  sign test (exact, two-sided):      p = {ps:.4f}"
              f"   {'SIGNIFICANT' if ps < 0.05 else 'n.s.'}")
    else:
        print("  sign test: all pairs tied -- inconclusive")
    pw, m = exact_wilcoxon_p(diffs)
    if pw is not None:
        print(f"  Wilcoxon signed-rank (exact):      p = {pw:.4f}"
              f"   {'SIGNIFICANT' if pw < 0.05 else 'n.s.'}   (m={m} non-tied)")
        if pw < 0.05:
            print("  ->", "LEARNED ranking causally better" if w > l
                  else "CLASSICAL control better -- report honestly")
    elif m:
        print(f"  Wilcoxon: m={m} too large for exact enumeration here")
    if zero and zero == t and t:
        print(f"\n  NOTE: {zero} pair(s) had BOTH arms gain 0 -- the starting front")
        print("  is too mature to discriminate. INCONCLUSIVE, not negative:")
        print("  rerun from a weakened snapshot so both arms have room to climb.")


if __name__ == "__main__":
    main()

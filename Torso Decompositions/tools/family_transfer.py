#!/usr/bin/env python3
"""
family_transfer.py -- P3(b): train a GBDT ordering policy ACROSS planted-family
instances and decode it (with noise) on a real instance, banking valid points.

Pipeline (self-contained, no core-tool edits):

  gen     : make K family instances at --scale via family_gen.py;
  train   : for each instance, build a certificate-guided teacher ordering
            (clique quota prefixes by width sweep + min-degree suffix), label
            every vertex with its normalized teacher position, extract
            STRUCTURE features (twin-class size, clique rank / is-twin /
            is-external, #attachments, component size, degrees), and fit one
            LightGBM regressor across ALL instances -> .feature_cache/family_policy.txt;
  decode  : compute the same features on a REAL instance, predict positions,
            sample S orderings via argsort(pred + Gumbel*temp), exact-evaluate,
            bank valid ones -> submissions/<problem>/family_policy.json.

The scientific claim under test: if the cross-instance policy transfers to the
competition instances, the GBDT has learned the CONSTRUCTION, not the instance
(extends THESIS §6b.5 cross-density generalization to the disclosed family).

    python3 tools/family_transfer.py --mode gen --k 12 --scale 0.25
    python3 tools/family_transfer.py --mode train
    python3 tools/family_transfer.py --mode decode --problem large-graph --samples 200
"""
from __future__ import annotations
import argparse, collections, glob, json, math, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

FAMDIR = os.path.join(HERE, "extra_instances", "family")
MODEL = os.path.join(HERE, ".feature_cache", "family_policy.txt")


def load_edges(path):
    adj = collections.defaultdict(set)
    mx = -1
    for line in open(path):
        a = line.split()
        if len(a) == 2:
            u, v = int(a[0]), int(a[1])
            mx = max(mx, u, v)
            if u != v:
                adj[u].add(v); adj[v].add(u)
    return mx + 1, adj


def features_all(n, adj):
    """structure features per vertex, computable on ANY family-like graph."""
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    cls_of, cls_sz = {}, {}
    for vs in h.values():
        for v in vs:
            cls_of[v] = vs[0]; cls_sz[v] = len(vs)
    big = sorted(([vs for vs in h.values() if len(vs) >= max(8, n // 60)]),
                 key=lambda vs: -len(adj[vs[0]]))
    Ks = []
    for vs in big[:3]:
        Ks.append(set(vs) | set(adj[vs[0]]))
    glue = set().union(*Ks) if Ks else set()
    who = {}
    for i, K in enumerate(Ks):
        for v in K:
            who.setdefault(v, i)
    # components of non-glue
    comp_sz = {}
    seen = set()
    for s0 in range(n):
        if s0 in glue or s0 in seen:
            continue
        stack, c = [s0], []
        seen.add(s0)
        while stack:
            u = stack.pop(); c.append(u)
            for x in adj[u]:
                if x not in glue and x not in seen:
                    seen.add(x); stack.append(x)
        for v in c:
            comp_sz[v] = len(c)
    F = []
    for v in range(n):
        i = who.get(v, -1)
        is_glue = 1 if v in glue else 0
        is_twin = 0
        if is_glue and i >= 0:
            K = Ks[i]
            is_twin = 1 if cls_sz[v] >= 8 else 0
        att = len([u for u in adj[v] if u not in glue]) if is_glue else 0
        F.append([len(adj[v]) / max(1, n), cls_sz[v],
                  is_glue, i, is_twin, att,
                  comp_sz.get(v, 0) / max(1, n),
                  len([u for u in adj[v] if u in glue])])
    return F


def teacher(n, adj):
    """certificate-guided ordering: glue twins (largest clique first), glue
    externals by attachment count desc, then components by min-degree."""
    F = features_all(n, adj)
    glue_tw = [v for v in range(n) if F[v][2] and F[v][4]]
    glue_ex = [v for v in range(n) if F[v][2] and not F[v][4]]
    comps = [v for v in range(n) if not F[v][2]]
    glue_tw.sort(key=lambda v: (F[v][3], v))
    glue_ex.sort(key=lambda v: -F[v][5])
    # min-degree order for components (static approximation)
    comps.sort(key=lambda v: len(adj[v]))
    return glue_tw + glue_ex + comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gen", "train", "decode"])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.mode == "gen":
        os.makedirs(FAMDIR, exist_ok=True)
        for s in range(a.k):
            out = os.path.join(FAMDIR, f"fam_{int(a.scale*100)}_{s}.gr")
            subprocess.run([sys.executable,
                            os.path.join(HERE, "tools", "family_gen.py"),
                            "--scale", str(a.scale), "--seed", str(s),
                            "--out", out], check=True)
        print(f"{a.k} family instances in {FAMDIR}")
        return

    if a.mode == "train":
        X, y = [], []
        for fp in sorted(glob.glob(os.path.join(FAMDIR, "*.gr"))):
            n, adj = load_edges(fp)
            order = teacher(n, adj)
            rank = {v: i / n for i, v in enumerate(order)}
            F = features_all(n, adj)
            for v in range(n):
                X.append(F[v]); y.append(rank[v])
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, num_leaves=63,
                              learning_rate=0.05, seed=a.seed)
        m.fit(X, y)
        os.makedirs(os.path.dirname(MODEL), exist_ok=True)
        m.booster_.save_model(MODEL)
        print(f"trained on {len(X)} vertices across "
              f"{len(glob.glob(os.path.join(FAMDIR, '*.gr')))} instances -> {MODEL}")
        return

    # decode on a real instance
    import lightgbm as lgb
    booster = lgb.Booster(model_file=MODEL)
    n, adj_l = load_graph(graph_path(HERE, a.problem))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    F = features_all(n, adj)
    base = booster.predict(F)
    arc = ParetoArchive()
    valid = 0
    t0 = time.time()
    for s in range(a.samples):
        noise = [-a.temp * math.log(-math.log(max(1e-12, rng.random())))
                 for _ in range(n)]
        perm = sorted(range(n), key=lambda v: base[v] + noise[v])
        df = ev.full(perm)
        if int(max(df)) > MAX_TW:
            continue
        valid += 1
        r = 0
        for t in range(n - 1, -1, -1):
            c = int(df[t]); r = c if c > r else r
            if r <= MAX_TW:
                arc.try_add(r, t, perm)
        if (s + 1) % 50 == 0:
            print(f"  {s+1}/{a.samples} sampled, {valid} valid "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    out = os.path.join(HERE, "submissions", a.problem, "family_policy.json")
    top = arc.top_k_by_hv_contribution(60, n)
    if top:
        write_submission([list(p) + [int(t)] for (_, t, p) in top],
                         a.problem, out)
    print(f"decode: {valid}/{a.samples} valid; banked {len(top)} points -> {out}")


if __name__ == "__main__":
    main()

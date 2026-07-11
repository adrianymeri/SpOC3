#!/usr/bin/env python3
"""
rank_externals.py -- the GBDT head-admission ranker (thesis centerpiece).

The clique-packing certificate (docs/PLANTED_STRUCTURE_CERTIFICATES.md) reduces
large-graph at widths w in [125,299) to ONE open choice: WHICH K400/K300/K500
externals join the head (twins are forced and free). clique_prefix.py's v0
heuristic (ascending out-of-clique degree) overshoots the bound by ~30-100
width. This tool learns the choice with a GBDT:

  gen       random-subset constructions -> training rows (also banks any point
            that helps the envelope, so data generation is never wasted compute)
  train     LightGBM (fallback: sklearn HistGradientBoosting, still a GBDT)
            regressor: vertex features + target w -> achieved width
  construct heads ranked by the model; banks results; prints the ablation table
            random / outdeg / gbdt  achieved-width per target w

    python3 tools/rank_externals.py --mode gen --samples 300
    python3 tools/rank_externals.py --mode train
    python3 tools/rank_externals.py --mode construct
"""
from __future__ import annotations
import argparse, collections, json, os, random, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

PROBLEM = "large-graph"
DATA = os.path.join(HERE, ".feature_cache", "rank_externals.jsonl")
MODEL = os.path.join(HERE, ".feature_cache", "rank_externals.model.txt")
DEGS = [499, 399, 299]


def setup():
    n, adj_l = load_graph(graph_path(HERE, PROBLEM))
    adj = {v: set(adj_l[v]) for v in range(n)}
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)
    h = {}
    for v in range(n):
        h.setdefault(frozenset(adj[v] | {v}), []).append(v)
    cliques = []
    for vs in h.values():
        if len(vs) >= 50:
            cliques.append((sorted(vs), sorted(set(vs) | set(adj[vs[0]]))))
    cliques.sort(key=lambda ck: -len(ck[1]))
    glue = set()
    for _, K in cliques:
        glue |= set(K)
    # component id per non-glue vertex
    comp, cid = {}, 0
    for s in range(n):
        if s in glue or s in comp:
            continue
        cid += 1
        stack = [s]; comp[s] = cid
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in glue and w not in comp:
                    comp[w] = cid; stack.append(w)
    return n, adj, ev, cliques, comp


def features(v, ci, adj, cliques, comp):
    K = set(cliques[ci][1])
    out = adj[v] - K
    comps = collections.Counter(comp.get(u, 0) for u in out)
    n_comp_edges = sum(c for k, c in comps.items() if k != 0)
    n_glue_edges = comps.get(0, 0)
    return [ci, len(adj[v]), len(out), n_comp_edges, n_glue_edges,
            len([k for k in comps if k != 0]),
            max(comps.values()) if comps else 0]


F_NAMES = ["clique", "deg", "outdeg", "comp_edges", "glue_edges",
           "n_comps", "max_one_comp"]


def build_head(w, cliques, order_ext):
    head_tw, head_ex = [], []
    for ci, (twins, K) in enumerate(cliques):
        q = max(0, DEGS[ci] - w)
        tw = list(twins[:q])
        head_tw += tw
        if q > len(twins):
            ext = order_ext(ci, [v for v in K if v not in set(twins)])
            head_ex += ext[:q - len(twins)]
    return head_tw + head_ex


def evaluate(head, template, n, ev, arc):
    Hset = set(head); t = len(head)
    perm = head + [v for v in template if v not in Hset]
    df = ev.full(perm)
    if max(int(x) for x in df) > MAX_TW:
        return None
    r = 0
    for tt in range(n - 1, -1, -1):
        c = int(df[tt]); r = c if c > r else r
        if r <= MAX_TW:
            arc.try_add(r, tt, perm)
    return max(int(df[i]) for i in range(t, n))


def load_templates(n, k=8):
    fp = os.path.join(HERE, "submissions", PROBLEM, "cap20.json")
    d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
    out = []
    for dv in sorted(e["decisionVector"], key=lambda dv: dv[-1]):
        p = [int(x) for x in dv[:-1]]
        if len(p) == n and sorted(p) == list(range(n)):
            out.append(p)
        if len(out) >= k:
            break
    return out


def get_model():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "sklearn-histgbdt", HistGradientBoostingRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gen", "train", "construct"])
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--wmin", type=int, default=170)
    ap.add_argument("--wmax", type=int, default=299)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    rng = random.Random(a.seed)
    n, adj, ev, cliques, comp = setup()
    templates = load_templates(n)
    arc = ParetoArchive()

    if a.mode == "gen":
        t0 = time.time()
        with open(DATA, "a") as fh:
            for s in range(a.samples):
                w = rng.randrange(a.wmin, a.wmax)
                chosen = {}
                def order_ext(ci, ext, chosen=chosen):
                    rng.shuffle(ext); chosen[ci] = ext; return ext
                head = build_head(w, cliques, order_ext)
                wt = evaluate(head, templates[s % len(templates)], n, ev, arc)
                if wt is None:
                    continue
                q = {ci: max(0, DEGS[ci] - len(cliques[ci][0]) - (0 if DEGS[ci] - w <= len(cliques[ci][0]) else 0)) for ci in range(3)}
                used = {ci: chosen.get(ci, [])[:max(0, DEGS[ci] - w - len(cliques[ci][0]))] for ci in range(3)}
                fh.write(json.dumps({"w": w, "achieved": wt,
                                     "ext": {str(ci): used[ci] for ci in range(3)}}) + "\n")
                if (s + 1) % 20 == 0:
                    print(f"  gen {s+1}/{a.samples} last(w={w} got {wt})"
                          f" [{time.time()-t0:.0f}s]", flush=True)
        out = os.path.join(HERE, "submissions", PROBLEM, "cp_rand.json")
        top = arc.top_k_by_hv_contribution(60, n)
        if top:
            write_submission([list(p) + [int(t)] for (_, t, p) in top], PROBLEM, out)
            print(f"banked search byproduct -> {out}")
        print(f"training rows appended to {DATA}")
        return

    # ---- build vertex-level dataset from the jsonl
    X, y = [], []
    feat_cache = {}
    for line in open(DATA):
        r = json.loads(line)
        for ci_s, vs in r["ext"].items():
            ci = int(ci_s)
            for v in vs:
                if (v, ci) not in feat_cache:
                    feat_cache[(v, ci)] = features(v, ci, adj, cliques, comp)
                X.append(feat_cache[(v, ci)] + [r["w"]])
                y.append(r["achieved"] - r["w"])          # overshoot as label
    print(f"dataset: {len(X)} rows from {DATA}")

    name, M = get_model()
    if a.mode == "train":
        if name == "lightgbm":
            model = M.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=63, seed=a.seed)
            model.fit(X, y)
            model.booster_.save_model(MODEL)
        else:
            import pickle
            model = M(max_iter=400)
            model.fit(X, y)
            pickle.dump(model, open(MODEL + ".pkl", "wb"))
        print(f"trained {name} on {len(X)} rows -> {MODEL}")
        return

    # ---- construct with the trained ranker + print ablation
    if name == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=MODEL)
        pred = lambda rows: booster.predict(rows)
    else:
        import pickle
        m = pickle.load(open(MODEL + ".pkl", "rb"))
        pred = lambda rows: m.predict(rows)

    print("  w | quota-t | rand | outdeg | GBDT   (achieved width; lower=better, ==w is the bound)")
    out_arc = ParetoArchive()
    for w in range(a.wmin, a.wmax, 4):
        row = [w, None, None, None, None]
        Kext = {}
        def mk(mode):
            def order_ext(ci, ext):
                if mode == "rand":
                    r2 = random.Random(w * 7 + ci); e = list(ext); r2.shuffle(e); return e
                if mode == "outdeg":
                    return sorted(ext, key=lambda v: len(adj[v] - set(cliques[ci][1])))
                rows = [features(v, ci, adj, cliques, comp) + [w] for v in ext]
                sc = pred(rows)
                return [v for _, v in sorted(zip(sc, ext))]
            return order_ext
        for j, mode in enumerate(["rand", "outdeg", "gbdt"]):
            best = None
            for tp in templates:
                wt = evaluate(build_head(w, cliques, mk(mode)), tp, n, ev,
                              out_arc if mode == "gbdt" else ParetoArchive())
                if wt is not None and (best is None or wt < best):
                    best = wt
            row[2 + j] = best
        row[1] = sum(max(0, d - w) for d in DEGS)
        print("  {:3d} | {:5d} | {} | {} | {}".format(*row), flush=True)
    top = out_arc.top_k_by_hv_contribution(60, n)
    if top:
        out = os.path.join(HERE, "submissions", PROBLEM, "cp_gbdt.json")
        write_submission([list(p) + [int(t)] for (_, t, p) in top], PROBLEM, out)
        print(f"banked GBDT constructions -> {out}")


if __name__ == "__main__":
    main()

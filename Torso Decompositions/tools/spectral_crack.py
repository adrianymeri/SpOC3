#!/usr/bin/env python3
r"""
spectral_crack.py -- the reverse-engineered attack on the small-graph wall.

The landscape probe (THESIS s13.7, tools/landscape_gbdt.py) proved that on the
HARD bands (w>=8, where the +6 HV gap provably lives behind a (w+2)-clique) the
optimal torso membership is decided by GLOBAL low-Laplacian (separator/community)
structure -- NOT by the local boundary count that every prior attack
(band_climb, clique_break, SAT single-vertex grows) used to pick its candidates.
So all our exact-verification budget was spent on candidates drawn from the
wrong distribution.

This tool changes the candidate distribution to match the landscape:

  * ADD vertices that are SPECTRALLY INTERIOR to one community -- their torso
    neighbours cluster in a single low-eigenvector nodal domain, so the induced
    fill stays inside that cluster instead of welding a cross-community clique.
  * DROP vertices that are SPECTRAL BRIDGES / separators (high eigenvector
    gradient across their neighbourhood) and/or sit in the live obstruction
    clique -- the exact vertices that hold the clique together.

Every candidate is exact-verified with the same branch-and-bound treewidth oracle
as clique_break (tw_le), so any acceptance is a PROVEN +1 HV.  Two prongs:

  --nd     : evaluate a pure spectral nested-dissection elimination ordering as a
             fresh, globally-structured front source (cheap; may dominate a band
             outright).
  restructure (default): the spectral-interior/bridge net-+1 search on the
             exact-tractable bands (8-10; 11-14 share the ~1108-vertex core that
             defeats all exact methods, so they are out of reach here too).

Additive / safe: writes submissions/<problem>/spectral_crack.json ONLY on a
verified win.  Honest expectation: three exact engines agree the wall is real;
this is the one representation we had never searched, so it is the right
experiment, not a likely beat.

    python3 tools/spectral_crack.py --problem small-graph --bands 8,9,10 \
        --budget 14400 --tw-timeout 3.0 --keig 8 --nd
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(400000)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC
from tools.clique_break import tw_le, max_clique, adjdict
from algorithms.continuous.cmaes_torso import get_features


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


def eig_low(here, problem, n, adj, keig):
    """Return the first `keig` non-trivial Laplacian eigenvectors (n x keig),
    pulled from the cached feature matrix (cols 5.. are the eigenvectors)."""
    F, _ = get_features(here, problem, n, adj, max(32, keig))
    F = np.asarray(F)
    return F[:, 5:5 + keig]                   # [deg, nbr{min,max,mean,std}, eig...]


def front_hv(by_w, n):
    a = ParetoArchive()
    for w in by_w: a.try_add(w, by_w[w][0], None)
    return -hypervolume_2d(a.points(), n)


def save_front(by_w, here, problem, n):
    """Persist the current front additively (called on every verified win so a
    long, interruptible server run never loses an improvement)."""
    a = ParetoArchive()
    for w in by_w:
        if by_w[w][1] is not None: a.try_add(w, by_w[w][0], by_w[w][1])
    top = a.top_k_by_hv_contribution(20, n)
    dvs = [list(p) + [int(t)] for (_, t, p) in top]
    out = os.path.join(here, "submissions", problem, "spectral_crack.json")
    json.dump({"challenge": "spoc-3-torso-decompositions", "problem": problem,
               "decisionVector": dvs}, open(out, "w"))
    return out


# --------------------------------------------------------------------------- #
# prong 1: spectral elimination orderings (numpy-only, no scipy) as a fresh,
# globally-structured front source.  Each ordering is the vertices sorted along a
# low-Laplacian direction (Fiedler & friends) or a random combination of them --
# a spectral linear arrangement that minimises global edge-cut.  Adopting a band
# whose torso comes out STRICTLY larger under the exact evaluator IS a verified
# win (ev.full is exact), so no extra tw check is needed here.
# --------------------------------------------------------------------------- #
def try_spectral_orders(E, n, ev, by_w, target, seed, max_w=15, combos=24):
    rng = np.random.default_rng(seed)
    keig = E.shape[1]
    dirs = [E[:, j] for j in range(keig)]                       # each eigenvector
    for _ in range(combos):                                     # random low-eig combos
        w = rng.normal(0, 1, min(keig, 6)); w /= np.linalg.norm(w) + 1e-12
        dirs.append(E[:, :len(w)] @ w)
    # best torso size found per relevant band across all spectral orderings
    best_band = {}                                              # w -> (t, perm)
    for vec in dirs:
        for perm in (np.argsort(vec), np.argsort(-vec)):
            perm = [int(x) for x in perm]
            d = ev.full(perm); r = 0
            for t in range(n - 1, -1, -1):
                c = int(d[t]); r = c if c > r else r
                if r > max_w:            # only the score-relevant bands (0..15)
                    continue
                if r not in best_band or t < best_band[r][0]:
                    best_band[r] = (t, list(perm))
    raised = 0
    for w, (t, perm) in best_band.items():
        cur_t = by_w[w][0] if w in by_w else n
        if t < cur_t:                    # strictly larger torso at band w -> real win
            by_w[w] = (t, perm); raised += 1
    c = front_hv(by_w, n)
    print(f"  [spectral-orders] {len(dirs)*2} orderings evaluated | front {c:,.0f}"
          f"{f'  gap {c-target:+,.0f}' if target else ''} | {raised} band(s) strictly raised", flush=True)
    return raised


# --------------------------------------------------------------------------- #
# prong 2: spectral-interior ADD / spectral-bridge DROP, exact-verified
# --------------------------------------------------------------------------- #
def spectral_scores(E, adj, S):
    """interiority[v] (high = neighbours in S share one nodal domain) and
    bridgeness[v] (high = neighbours straddle communities)."""
    Sset = S if isinstance(S, set) else set(S)
    interior = np.full(len(adj), -1e9); bridge = np.zeros(len(adj))
    for v in range(len(adj)):
        ns = [u for u in adj[v] if u in Sset]
        if len(ns) < 2:
            continue
        sub = E[ns]                       # (|ns| x keig)
        spread = float(sub.std(0).mean()) # spread of neighbours in eigen-space
        bridge[v] = spread
        interior[v] = -spread
    return interior, bridge


def run(problem, here, bands, budget_s, tw_timeout, keig, kick, seed, do_nd):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); target = LEADERBOARD_TARGETS.get(problem)
    by_w = load_front(here, problem, n, ev)
    E = eig_low(here, problem, n, adj, keig)
    start_hv = front_hv(by_w, n)
    print(f"=== spectral-crack {problem} | front {start_hv:,.0f}"
          f"{f'  gap {start_hv-target:+,.0f}' if target else ''} "
          f"| keig={keig} ===", flush=True)

    wins = 0
    if do_nd:
        wins += try_spectral_orders(E, n, ev, by_w, target, seed)

    rng = np.random.default_rng(seed); t0 = time.time(); full = set(range(n))
    for W in bands:
        if W not in by_w: continue
        t_star, perm = by_w[W]; S = set(perm[t_star:]); best = len(S)
        interior, bridge = spectral_scores(E, adj, S)
        # the live obstruction clique on the canonical local grow
        Sm = 0
        for s in S: Sm |= 1 << s
        v0 = min((u for u in range(n) if not (Sm >> u) & 1),
                 key=lambda u: (ab[u] & Sm).bit_count())
        C0 = max_clique(adjdict(list(S) + [v0], ab, n))
        print(f"  w={W}: T_w={len(S)} | obstruction clique {len(C0)} "
              f"| spectral-guided restructure (avoid {W+2}-cliques)", flush=True)
        tested = to = 0; tw0 = time.time()
        while time.time() - t0 < budget_s and time.time() - tw0 < budget_s / max(1, len(bands)):
            Sset = S; Sm = 0
            for s in Sset: Sm |= 1 << s
            interior, bridge = spectral_scores(E, adj, Sset)
            # DROP candidates: live clique members + top spectral bridges in S
            vg = min((u for u in range(n) if not (Sm >> u) & 1),
                     key=lambda u: (ab[u] & Sm).bit_count())
            C = set(max_clique(adjdict(list(Sset) + [vg], ab, n))) & Sset
            in_list = list(Sset)
            br_rank = sorted(in_list, key=lambda u: -bridge[u])[:max(8, kick * 6)]
            drop_pool = list(C) + br_rank
            drop = set(int(x) for x in rng.choice(drop_pool,
                       size=min(kick, len(drop_pool)), replace=False)) if drop_pool else set()
            base = Sset - drop; bm = 0
            for s in base: bm |= 1 << s
            # ADD candidates: spectrally INTERIOR out-vertices (not in clique C)
            outs = [u for u in range(n) if u not in base and u not in C]
            outs.sort(key=lambda u: (-interior[u], (ab[u] & bm).bit_count()))
            pool = outs[:max(kick * 6, 24)]
            rng.shuffle(pool)
            add = pool[:kick + 1]
            cand = base | set(int(x) for x in add)
            if len(cand) <= best:
                tested += 1; continue
            ad = adjdict(list(cand), ab, n)
            if len(max_clique(ad)) >= W + 2:     # still obstructed -> skip exact
                tested += 1; continue
            try:
                if tw_le(ad, W, time.time() + tw_timeout):
                    best = len(cand); S = set(cand)
                    permnew = [u for u in full if u not in S] + list(S)
                    by_w[W] = (n - len(S), permnew); wins += 1
                    c = front_hv(by_w, n)
                    sp = save_front(by_w, here, problem, n)
                    print(f"  w={W}: *** SPECTRAL WIN torso {best} *** front {c:,.0f}"
                          f"{f'  gap {c-target:+,.0f}' if target else ''}  saved->{os.path.basename(sp)}",
                          flush=True)
                tested += 1
            except TimeoutError:
                to += 1
        print(f"  w={W}: done best {best} (was {n-t_star}) "
              f"[{tested} spectral exact tests, {to} timeouts, {time.time()-tw0:.0f}s]", flush=True)

    fin = front_hv(by_w, n)
    print(f"\nfinal front {fin:,.0f}"
          f"{f'  gap {fin-target:+,.0f}' if target else ''} | {wins} improvement(s)", flush=True)
    if wins and fin < start_hv - 0.5:        # only persist a GENUINE HV gain
        out = save_front(by_w, here, problem, n)
        print(f"saved -> {out}  (VERIFY with tools/verify_submission.py before trusting)", flush=True)
    else:
        print("no verified improvement -> nothing saved (front unchanged)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--bands", default="8,9,10")
    ap.add_argument("--budget", type=float, default=7200.0)
    ap.add_argument("--tw-timeout", type=float, default=3.0)
    ap.add_argument("--keig", type=int, default=8)
    ap.add_argument("--kick", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nd", action="store_true", help="also try spectral nested-dissection orderings")
    a = ap.parse_args()
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    run(a.problem, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        bands, a.budget, a.tw_timeout, a.keig, a.kick, a.seed, a.nd)


if __name__ == "__main__":
    main()

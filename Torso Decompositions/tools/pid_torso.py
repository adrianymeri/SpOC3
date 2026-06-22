#!/usr/bin/env python3
r"""
pid_torso.py -- exact high-band torso search using Tamaki's PID solver.

Our own branch-and-bound treewidth times out past ~700 vertices, so bands 8-14
were never exactly settled.  Tamaki's PID (PACE-2017 exact-treewidth winner) is
the scalable exact oracle: this wrapper exports a torso to PACE .gr, runs PID,
parses the exact treewidth, and uses it to (a) verify our claimed band widths,
(b) confirm the whole-graph treewidth, and (c) search for a SMALLER deletion set
(= larger torso) at a high band -- any success is a proven +1 HV.

    # one-shot: exact treewidth of an arbitrary .gr
    python3 tools/pid_torso.py --jar ~/PACE2017-TrackA --whole

    # verify our current high-band torsos are exactly their claimed widths
    python3 tools/pid_torso.py --jar ~/PACE2017-TrackA --verify --bands 8,9,10,11,12,13,14

    # search for a bigger torso at the high bands (exact)
    python3 tools/pid_torso.py --jar ~/PACE2017-TrackA --search --bands 12,13,14 \
        --budget 14400 --pid-timeout 120 --kick 4

--jar points at the built PACE2017-TrackA dir (containing build/classes/java/main)
or a direct path to a runnable tw-exact jar.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS)
import tools.torso_deletion as td
from tools.fastwalk import IncEvalC


def pid_cmd(jar):
    """Return the command prefix to run Tamaki's exact decomposer."""
    jar = os.path.expanduser(jar)
    classes = os.path.join(jar, "build", "classes", "java", "main")
    if os.path.isdir(classes):
        return ["java", "-Xmx8g", "-classpath", classes, "tw.exact.MainDecomposer"]
    if jar.endswith(".jar"):
        return ["java", "-Xmx8g", "-jar", jar]
    # fallback: tw-exact.jar inside the dir
    for cand in ("tw-exact.jar", "tw.jar"):
        p = os.path.join(jar, cand)
        if os.path.isfile(p):
            return ["java", "-Xmx8g", "-jar", p]
    raise SystemExit(f"could not locate PID solver under {jar}")


def torso_to_pace(Slist, ab, n):
    """Build a PACE .gr string for torso(Slist), relabelled 1..|S|."""
    tor, _ = td.torso_adj(Slist, ab, n)
    idx = {v: i + 1 for i, v in enumerate(Slist)}
    edges = []
    for v in Slist:
        x = tor[v] & ((1 << (max(Slist) + 1)) - 1)
        while x:
            b = x & -x; x &= x - 1; u = b.bit_length() - 1
            if u in idx and u > v:
                edges.append((idx[v], idx[u]))
    head = f"p tw {len(Slist)} {len(edges)}\n"
    return head + "".join(f"{a} {b}\n" for a, b in edges)


def exact_tw(cmd, gr_text, timeout):
    """Run PID on a PACE graph string; return exact treewidth or None on timeout."""
    try:
        r = subprocess.run(cmd, input=gr_text, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    best = None
    for line in r.stdout.splitlines():
        # PACE .td output:  's td <#bags> <max_bag_size> <#vertices>'
        if line.startswith("s td"):
            parts = line.split()
            best = int(parts[3]) - 1            # treewidth = max bag size - 1
            break
    return best


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--jar", required=True, help="PACE2017-TrackA dir or tw-exact.jar")
    ap.add_argument("--whole", action="store_true", help="exact treewidth of the whole graph")
    ap.add_argument("--verify", action="store_true", help="verify current band torsos exactly")
    ap.add_argument("--search", action="store_true", help="search bigger high-band torsos")
    ap.add_argument("--bands", default="12,13,14")
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--pid-timeout", type=float, default=120.0)
    ap.add_argument("--kick", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n, adj = load_graph(graph_path(here, a.problem)); ab = build_adj_bitsets(n, adj)
    ev = IncEvalC(ab, n); cmd = pid_cmd(a.jar)
    bands = [int(x) for x in a.bands.split(",") if x.strip()]
    target = LEADERBOARD_TARGETS.get(a.problem)

    if a.whole:
        gr = torso_to_pace(list(range(n)), ab, n)
        t0 = time.time(); tw = exact_tw(cmd, gr, a.pid_timeout * 10)
        print(f"whole-graph exact treewidth = {tw}  ({time.time()-t0:.1f}s)  "
              f"[heuristic said 15]")
        return

    by_w = load_front(here, a.problem, n, ev)

    if a.verify:
        for W in bands:
            if W not in by_w: continue
            t, perm = by_w[W]; S = list(perm[t:])
            gr = torso_to_pace(S, ab, n)
            t0 = time.time(); tw = exact_tw(cmd, gr, a.pid_timeout)
            tag = "OK" if tw == W else (f"!! claims {W}" if tw is not None else "timeout")
            print(f"  band w={W}: |S|={len(S)}  exact tw={tw}  {tag}  ({time.time()-t0:.0f}s)",
                  flush=True)
        return

    if a.search:
        rng = np.random.default_rng(a.seed); t0 = time.time(); wins = 0; full = set(range(n))

        def front_hv():
            ar = ParetoArchive()
            for w in by_w: ar.try_add(w, by_w[w][0], None)
            return -hypervolume_2d(ar.points(), n)
        print(f"=== pid-search {a.problem} | front {front_hv():,.0f}"
              f"{f'  gap {front_hv()-target:+,.0f}' if target else ''} ===", flush=True)
        for W in bands:
            if W not in by_w: continue
            t_star, perm = by_w[W]; S = set(perm[t_star:]); best = len(S)
            tested = 0; tw0 = time.time()
            while time.time() - t0 < a.budget and time.time() - tw0 < a.budget / max(1, len(bands)):
                Sm = 0
                for s in S: Sm |= 1 << s
                X = sorted([u for u in range(n) if u not in S],
                           key=lambda u: (ab[u] & Sm).bit_count())
                pool = X[:max(a.kick * 4, 16)]; rng.shuffle(pool)
                add = pool[:int(rng.integers(1, a.kick + 1))]
                S2 = list(S | set(int(v) for v in add))
                gr = torso_to_pace(S2, ab, n)
                tw = exact_tw(cmd, gr, a.pid_timeout); tested += 1
                if tw is not None and tw <= W and len(S2) > best:
                    best = len(S2); S = set(S2)
                    permnew = [u for u in full if u not in S] + S2
                    by_w[W] = (n - len(S), permnew); wins += 1
                    c = front_hv()
                    print(f"  w={W}: WIN torso {best}  front {c:,.0f}"
                          f"{f'  gap {c-target:+,.0f}' if target else ''}", flush=True)
            print(f"  w={W}: done best {best} (was {n-t_star}) [{tested} exact PID checks, "
                  f"{time.time()-tw0:.0f}s]", flush=True)
        print(f"\nfinal front {front_hv():,.0f}"
              f"{f'  gap {front_hv()-target:+,.0f}' if target else ''} | {wins} improved")
        if wins:
            ar = ParetoArchive()
            for w in by_w: ar.try_add(w, by_w[w][0], by_w[w][1])
            top = ar.top_k_by_hv_contribution(20, n)
            dvs = [list(p) + [int(t)] for (_, t, p) in top]
            out = os.path.join(here, "submissions", a.problem, "pid_torso.json")
            json.dump({"challenge": "spoc-3-torso-decompositions", "problem": a.problem,
                       "decisionVector": dvs}, open(out, "w"))
            print(f"saved -> {out}")


if __name__ == "__main__":
    main()

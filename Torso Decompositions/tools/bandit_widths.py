#!/usr/bin/env python3
"""
bandit_widths.py -- P2: UCB compute allocation over the open widths.

With 6/20 large-graph points certificate-optimal, the score is a weighted sum
of ~14 independent max-torso problems whose marginal values (zone widths) are
known from the HSSP geometry and whose per-width productivity is measurable.
This driver replaces a static-width gbfcpp arm: every stint it

  1. pools the envelope, runs the exact HSSP-20 selection, derives the open
     selected widths (< 299; the certified region is never fed);
  2. credits each width with the head-size improvement since the last stint,
     weighted by its zone value (exact capped-HV units);
  3. picks the top-K widths by  value_rate + c * sqrt(ln T / n_w)  (UCB1),
     always including one random open width for exploration;
  4. runs gbfcpp --cap20 --only-widths <picks> for --stint seconds (the arm
     resumes its own state files across stints), then repeats.

    python3 tools/bandit_widths.py --problem large-graph --stint 1800
    python3 tools/bandit_widths.py --once          # one selection, no launch
"""
from __future__ import annotations
import argparse, json, math, os, random, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  MAX_TW, LEADERBOARD_TARGETS)
from tools.fastwalk import IncEvalC
from tools.cap_submit import build_envelope

CERT_MIN = 299          # certified-optimal region on large: never allocate


def selection(problem, n, ev):
    """[(width, t, zone_value)] for the open HSSP-selected widths."""
    by_w = build_envelope(HERE, problem, n, ev)
    arc = ParetoArchive()
    for w, (t, p) in by_w.items():
        arc.try_add(w, t, [0])
    top = sorted((w, t) for w, t, _ in arc.top_k_by_hv_contribution(20, n))
    out = []
    for i, (w, t) in enumerate(top):
        nxt = top[i + 1][0] if i + 1 < len(top) else MAX_TW + 1
        if w < CERT_MIN:
            out.append((w, t, nxt - w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph")
    ap.add_argument("--stint", type=int, default=1800)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--c", type=float, default=1.0, help="UCB exploration weight")
    ap.add_argument("--algo", default="gbfcpp_slack")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n, adj_l = load_graph(graph_path(HERE, a.problem))
    ev = IncEvalC(build_adj_bitsets(n, adj_l), n)

    st_fp = os.path.join(HERE, "submissions", a.problem, ".bandit_state.json")
    st = {"T": 0, "arms": {}, "last_t": {}}
    if os.path.exists(st_fp):
        st = json.load(open(st_fp))

    child = None

    def stop(*_):
        if child and child.poll() is None:
            child.terminate()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while True:
        sel = selection(a.problem, n, ev)
        st["T"] += 1
        # credit rewards: capped-HV units gained per width since last stint
        for w, t, zv in sel:
            k = str(w)
            arm = st["arms"].setdefault(k, {"n": 0, "r": 0.0})
            if k in st["last_t"]:
                gained = max(0, st["last_t"][k] - t) * zv
                arm["r"] += gained
            st["last_t"][k] = t
        # UCB over the open selected widths
        scored = []
        for w, t, zv in sel:
            arm = st["arms"][str(w)]
            rate = arm["r"] / max(1, arm["n"])
            bonus = a.c * zv * math.sqrt(math.log(max(2, st["T"])) / max(1, arm["n"]))
            scored.append((rate + bonus, w, zv, arm["n"]))
        scored.sort(reverse=True)
        picks = [w for _, w, _, _ in scored[:a.k - 1]]
        rest = [w for _, w, _, _ in scored[a.k - 1:]]
        if rest:
            picks.append(rng.choice(rest))       # forced exploration slot
        picks = sorted(set(picks))
        for w in picks:
            st["arms"][str(w)]["n"] += 1
        json.dump(st, open(st_fp, "w"))
        print(f"[bandit T={st['T']}] open widths {[w for _,w,_,_ in scored]} "
              f"-> picks {picks}", flush=True)
        if a.once:
            for s, w, zv, nn in scored:
                print(f"   w={w:3d} zone={zv:3d} plays={nn} score={s:,.1f}")
            return
        cmd = [sys.executable, os.path.join(HERE, "tools", "gbfcpp.py"),
               "--problem", a.problem, "--algo", a.algo, "--cap20",
               "--only-widths", ",".join(map(str, picks)),
               "--rounds", "1000", "--round-budget", "120"]
        child = subprocess.Popen(cmd, cwd=HERE)
        t0 = time.time()
        while time.time() - t0 < a.stint:
            if child.poll() is not None:
                break
            time.sleep(5)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        print(f"[bandit] stint done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

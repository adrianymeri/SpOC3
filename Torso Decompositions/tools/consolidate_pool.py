#!/usr/bin/env python3
"""
consolidate_pool.py -- losslessly shrink a submissions pool.

Pools every ordering in submissions/<problem>/*.json into the per-width
envelope, writes ONE file (full_envelope.json) containing a deduplicated set
of achiever orderings that reproduces the ENTIRE envelope bit-for-bit, then
(only after verifying HV equality) optionally archives every other file to
submissions/archived/<problem>/.

Why: the server pool grew to ~1800 files / 74 MB (accumulated cuda-torso
dumps), which bloats sync_pool.sh and makes every cap_submit re-evaluate
everything. Naive `ls -t | tail | mv` pruning measurably lost envelope HV
(residual +1,335 -> +2,365 on large, 2026-07-11). This tool cannot lose HV:
it refuses to archive unless the consolidated envelope matches exactly.

    python3 tools/consolidate_pool.py --problem large-graph            # dry run
    python3 tools/consolidate_pool.py --problem large-graph --archive  # do it
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core import (load_graph, build_adj_bitsets, graph_path, ParetoArchive,
                  hypervolume_2d, MAX_TW, LEADERBOARD_TARGETS, write_submission)
from tools.fastwalk import IncEvalC

# never archived, on top of full_envelope/cap20: live arms rewrite these stems
DEFAULT_KEEP = "cap20,full_envelope,gbfcpp,hri_lns,archive_evolve,gbdt,portfolio"


def envelope_of_files(files, n, ev, verbose=True):
    """{width: (t, perm)} per-width best over all valid decision vectors."""
    by_w, t0 = {}, time.time()
    for i, fp in enumerate(files):
        try:
            d = json.load(open(fp)); e = d[0] if isinstance(d, list) else d
            for dv in e.get("decisionVector", []):
                if not (isinstance(dv, list) and len(dv) == n + 1):
                    continue
                perm = [int(x) for x in dv[:-1]]
                if sorted(perm) != list(range(n)):
                    continue
                df = ev.full(perm); r = 0
                for t in range(n - 1, -1, -1):
                    c = int(df[t]); r = c if c > r else r
                    if r > MAX_TW:
                        break
                    cur = by_w.get(r)
                    if cur is None or t < cur[0]:
                        by_w[r] = (t, perm)
        except Exception:
            pass
        if verbose and (i + 1) % 100 == 0:
            print(f"  ..{i+1}/{len(files)} files  [{time.time()-t0:.0f}s]", flush=True)
    return by_w


def pareto(by_w):
    out, bt = [], 10 ** 9
    for w in sorted(by_w):
        t = by_w[w][0]
        if t < bt:
            out.append((w, t)); bt = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="large-graph", choices=list(LEADERBOARD_TARGETS))
    ap.add_argument("--src", default=None, help="pool dir (default submissions/<problem>)")
    ap.add_argument("--archive", action="store_true", help="actually move redundant files")
    ap.add_argument("--keep-stems", default=DEFAULT_KEEP,
                    help="comma list: files whose name starts with any stem are never archived")
    a = ap.parse_args()

    src = a.src or os.path.join(HERE, "submissions", a.problem)
    n, adj = load_graph(graph_path(HERE, a.problem))
    ev = IncEvalC(build_adj_bitsets(n, adj), n)
    target = LEADERBOARD_TARGETS.get(a.problem)

    files = sorted(f for f in glob.glob(os.path.join(src, "*.json"))
                   if not f.endswith("_platform.json"))
    print(f"pooling {len(files)} files from {src} ...")
    by_w = envelope_of_files(files, n, ev)
    front = pareto(by_w)
    full_hv = hypervolume_2d(front, n)
    print(f"envelope: {len(front)} pareto points | uncapped {-full_hv:,.0f}"
          + (f" (residual {(-full_hv)-target:+,.0f})" if target else ""))

    # dedupe achiever perms (one perm may cover many widths via its staircase)
    uniq, dvs = {}, []
    for w, t in front:
        tt, perm = by_w[w]
        key = tuple(perm)
        if key not in uniq:
            uniq[key] = True
            dvs.append(list(perm) + [int(tt)])
    out = os.path.join(src, "full_envelope.json")
    write_submission(dvs, a.problem, out)
    print(f"wrote {out}: {len(dvs)} unique orderings covering all {len(front)} points")

    # verify: envelope from the ONE file must match exactly
    by_w2 = envelope_of_files([out], n, ev, verbose=False)
    hv2 = hypervolume_2d(pareto(by_w2), n)
    if abs(hv2 - full_hv) > 1e-6:
        print(f"VERIFY FAILED: {-hv2:,.0f} != {-full_hv:,.0f} -- NOT archiving anything")
        sys.exit(1)
    print(f"VERIFIED: full_envelope.json alone reproduces {-hv2:,.0f}")

    keep = tuple(s.strip() for s in a.keep_stems.split(",") if s.strip())
    redundant = [f for f in sorted(glob.glob(os.path.join(src, "*.json")))
                 if not os.path.basename(f).startswith(keep)]
    print(f"{len(redundant)} files are now redundant (envelope lives in full_envelope.json)")
    if a.archive:
        dst = os.path.join(HERE, "submissions", "archived", a.problem)
        os.makedirs(dst, exist_ok=True)
        for f in redundant:
            shutil.move(f, os.path.join(dst, os.path.basename(f)))
        print(f"archived {len(redundant)} files -> {dst}")
        print("re-run cap_submit now; best-20 and residual must be unchanged.")
    else:
        print("dry run: re-run with --archive to move them (verification already passed).")


if __name__ == "__main__":
    main()

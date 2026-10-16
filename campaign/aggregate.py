#!/usr/bin/env python3
"""
aggregate.py -- turns every run's files into the campaign's results.

Run by the supervisor every 10 minutes (and by status.sh / snapshot.sh).
Reads runs/<instance>/<method>/{status.json,best.json,anytime.csv} and writes

    results/summary.csv         one row per run: score, budget, 24 h change,
                                last improvement, ESA virtual rank, checks
    results/ola2027_table.csv   the OLA2027 tab's layout, one row per instance
    results/fair_synthetic.csv  every synthetic run's score at the last round
                                level all synthetic runs have completed
    results/anytime_official.csv  score over time, official instances
    results/dashboard.html      all of the above on one self-contained page
    results/STATUS.txt          the terminal view (campaign/status.sh)

Every best.json is re-scored with the campaign engine on each pass (exact,
fast). The authoritative evaluator esa_eval.py re-scores a few submissions per
pass, oldest first, so every reported number gets checked by it over time;
`--final` checks all of them.
"""

from __future__ import annotations

import csv
import html
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import config as C  # noqa: E402

VERIFY = os.path.join(C.RESULTS, "verify")
ESA_PER_PASS = 4


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_anytime(path):
    rows = []
    try:
        with open(path) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 4:
                    continue
                try:
                    rows.append((float(row[0]), row[1], int(float(row[2])), row[5] if len(row) > 5 else ""))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def wall_epoch(s):
    """UTC timestamp string -> epoch seconds (independent of the server's time zone and DST)."""
    import calendar
    try:
        return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def score_at_budget(rows, b, tol=0.0):
    """Best score recorded at a budget <= b + tol."""
    best = None
    for bud, _, sc, _ in rows:
        if bud <= b + tol + 1e-6:
            best = sc if best is None else min(best, sc)
    return best


def score_at_wall(rows, epoch):
    best = None
    for _, w, sc, _ in rows:
        e = wall_epoch(w)
        if e is not None and e <= epoch:
            best = sc if best is None else min(best, sc)
    return best


def board():
    out = {}
    path = os.path.join(C.REPO, "leaderboard-references", "esa_leaderboard.csv")
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                out.setdefault(row["problem"], []).append((int(row["score"]), row["username"]))
    except OSError:
        pass
    for k in out:
        out[k].sort()
    return out


def virtual_rank(score, entries):
    if score is None or not entries:
        return None
    return 1 + sum(1 for s, _ in entries if s < score)


# ------------------------------------------------------------------- checks

_engines = {}


def engine_check(inst, best):
    import numpy as np
    import torso_engine as te
    eng = _engines.get(inst)
    if eng is None:
        eng = _engines[inst] = te.Engine(C.instance_path(inst))
    pts = []
    for dv in best.get("decisionVector", []):
        p = np.asarray(dv[:-1], dtype=np.int32)
        if len(p) != eng.n or not eng.is_perm(p):
            return None
        pts.append((eng.fitness(p, int(dv[-1])), int(dv[-1])))
    if len(pts) > 20:
        return None
    return -int(te.hv_points(pts, eng.n))


def esa_check(inst, best):
    import esa_eval
    from torso import Graph
    g = Graph.load(C.instance_path(inst))
    pts = []
    for dv in best.get("decisionVector", []):
        perm, t = [int(x) for x in dv[:-1]], int(dv[-1])
        pts.append(esa_eval.evaluate(perm, t, g.bits, g.n))
    if len(pts) > 20:
        return None
    return -int(esa_eval.hypervolume_2d(pts, g.n))


def verify(inst, method, final=False, budget_left=None):
    os.makedirs(VERIFY, exist_ok=True)
    bpath = os.path.join(C.run_dir(inst, method), "best.json")
    if not os.path.exists(bpath):
        return None
    vpath = os.path.join(VERIFY, f"{inst}__{method}.json")
    v = read_json(vpath, {})
    mt = os.path.getmtime(bpath)
    best = read_json(bpath)
    if best is None:
        return v or None
    if v.get("mtime") != mt:
        es = engine_check(inst, best)
        v = {"mtime": mt, "claimed": best.get("score"), "engine": es,
             "engine_ok": es is not None and es == best.get("score"),
             "esa": None, "esa_ok": None, "esa_checked": None}
    if v.get("esa_ok") is None and (final or (budget_left is not None and budget_left[0] > 0)):
        if budget_left is not None:
            budget_left[0] -= 1
        try:
            e = esa_check(inst, best)
            v["esa"] = e
            v["esa_ok"] = e is not None and e == best.get("score")
        except Exception as ex:          # never let a check kill the aggregation
            v["esa_error"] = repr(ex)
        v["esa_checked"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = vpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(v, f)
    os.replace(tmp, vpath)
    return v


# ------------------------------------------------------------------- tables

def collect(final=False):
    B = board()
    now = time.time()
    left = [10 ** 9 if final else ESA_PER_PASS]
    # esa checks go to the oldest unchecked submissions first
    runs = []
    for inst in C.INSTANCES:
        for m in C.METHODS:
            rd = C.run_dir(inst, m)
            st = read_json(os.path.join(rd, "status.json"))
            if st is None:
                runs.append({"instance": inst, "method": m, "state": "not started"})
                continue
            rows = read_anytime(os.path.join(rd, "anytime.csv"))
            score = st.get("score")
            s24 = score_at_wall(rows, now - 86400)
            last_imp = None
            prev = None
            for bud, w, sc, kind in rows:
                if prev is None or sc < prev:
                    last_imp = w
                    prev = sc
            hb = os.path.join(rd, "heartbeat")
            runs.append({
                "instance": inst, "method": m, "family": C.FAMILY[inst],
                "official": inst in C.OFFICIAL, "state": st.get("state"),
                "score": score, "budget_h": round(st.get("budget_s", 0) / 3600, 3),
                "budget_s": float(st.get("budget_s", 0)),
                "evaluations": st.get("evaluations"),
                "change_24h": (score - s24) if (score is not None and s24 is not None) else None,
                "last_improvement_utc": last_imp,
                "heartbeat_age_min": round((now - os.path.getmtime(hb)) / 60, 1) if os.path.exists(hb) else None,
                "esa_rank": virtual_rank(score, B.get(inst)) if inst in C.OFFICIAL else None,
                "endpoint": (st.get("front") or [[None]])[-1][0],
                "rows": rows,
            })
    order = sorted([r for r in runs if "score" in r and r.get("score") is not None],
                   key=lambda r: os.path.getmtime(os.path.join(C.run_dir(r["instance"], r["method"]), "best.json"))
                   if os.path.exists(os.path.join(C.run_dir(r["instance"], r["method"]), "best.json")) else 0)
    checks = {}
    for r in order:
        checks[(r["instance"], r["method"])] = verify(r["instance"], r["method"], final, left)
    for r in runs:
        v = checks.get((r["instance"], r["method"])) or {}
        r["engine_ok"] = v.get("engine_ok")
        r["esa_ok"] = v.get("esa_ok")
    return runs, B


def last_complete_level(runs):
    synth = [r for r in runs if r["instance"] in C.SYNTHETIC and r["method"] in C.CPU_METHODS]
    if not synth or any(r.get("score") is None for r in synth):
        return None
    m = min(r["budget_s"] for r in synth)
    done = [L for L in C.SYNTH_LEVELS_H if L * 3600 <= m + 5]
    return done[-1] if done else None


def write_csv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def fmt(x):
    return "" if x is None else f"{x:,}"


def build(final=False):
    os.makedirs(C.RESULTS, exist_ok=True)
    runs, B = collect(final)
    by = {(r["instance"], r["method"]): r for r in runs}
    sup = read_json(os.path.join(C.STATE, "supervisor_status.json"), {})

    # summary.csv
    hdr = ["instance", "family", "method", "label", "state", "score", "budget_h", "evaluations",
           "change_24h", "last_improvement_utc", "endpoint_width", "esa_virtual_rank", "engine_check",
           "esa_eval_check", "heartbeat_age_min"]
    rows = []
    for r in runs:
        if r.get("score") is None:
            rows.append([r["instance"], C.FAMILY[r["instance"]], r["method"], C.LABEL[r["method"]],
                         r.get("state")] + [""] * 10)
            continue
        rows.append([r["instance"], r["family"], r["method"], C.LABEL[r["method"]], r["state"], r["score"],
                     r["budget_h"], r["evaluations"], r["change_24h"], r["last_improvement_utc"],
                     r["endpoint"], r["esa_rank"], r["engine_ok"], r["esa_ok"], r["heartbeat_age_min"]])
    write_csv(os.path.join(C.RESULTS, "summary.csv"), hdr, rows)

    # the OLA2027 tab layout (+ one column per method of ours)
    ours = ["ea1", "ea2", "nsga2", "ne1", "ne2"]
    hdr = (["#", "Instance name", "family", "n", "Best Known (ESA board)", "Team",
            "fcmaes", "Team HRI", "Spacekangaroos", "Hill Climbing"]
           + [C.LABEL[m] for m in ours] + ["Best of ours", "Best method of ours", "Best of all methods",
                                           "Gap of ours to best known (%, negative = better)"])
    table = []
    for k, inst in enumerate(C.INSTANCES, 1):
        n = next((read_json(os.path.join(C.run_dir(inst, m), "status.json"), {}).get("n")
                  for m in C.METHODS if read_json(os.path.join(C.run_dir(inst, m), "status.json"))), "")
        bk = B.get(inst, [(None, "")])[0] if inst in C.OFFICIAL else (None, "")
        sc = {m: (by.get((inst, m)) or {}).get("score") for m in C.METHODS}
        ours_sc = [(sc[m], m) for m in ours if sc.get(m) is not None]
        best_ours = min(ours_sc) if ours_sc else (None, "")
        allv = [v for v in sc.values() if v is not None]
        best_all = min(allv) if allv else None
        ref = bk[0] if bk[0] is not None else best_all
        # signed: positive = ours below the reference, negative = ours better
        gap = ((best_ours[0] - ref) / abs(ref) * 100) if (best_ours[0] is not None and ref) else None
        table.append([k, inst, C.FAMILY[inst], n, bk[0] if bk[0] is not None else "", bk[1],
                      sc.get("fcmaes"), sc.get("hri"), sc.get("cuda_torso"), sc.get("hc")] + [sc.get(m) for m in ours]
                     + [best_ours[0], C.LABEL.get(best_ours[1], ""), best_all,
                        None if gap is None else round(gap, 4)])
    write_csv(os.path.join(C.RESULTS, "ola2027_table.csv"), hdr, table)

    # fair synthetic table at the last level every synthetic CPU run completed
    L = last_complete_level(runs)
    fair = []
    wins = {m: 0 for m in C.CPU_METHODS}
    ranks = {m: [] for m in C.CPU_METHODS}
    if L is not None:
        for inst in C.SYNTHETIC:
            # a slice stops at the first step past the level, so its closing
            # checkpoint row sits a few seconds above it: count it
            tol = max(60.0, 0.002 * L * 3600)
            vals = {m: score_at_budget(by[(inst, m)]["rows"], L * 3600, tol) for m in C.CPU_METHODS}
            got = sorted((v, m) for m, v in vals.items() if v is not None)
            for pos, (v, m) in enumerate(got, 1):
                ranks[m].append(sum(1 for w, _ in got if w < v) + 1)
            # a tie for the best score is a win for every tied method
            for v, m in got:
                if v == got[0][0]:
                    wins[m] += 1
            fair.append([inst, C.FAMILY[inst], L] + [vals[m] for m in C.CPU_METHODS])
        write_csv(os.path.join(C.RESULTS, "fair_synthetic.csv"),
                  ["instance", "family", "level_h"] + [C.LABEL[m] for m in C.CPU_METHODS], fair)

    # anytime, official instances (wall time vs score)
    at = []
    for inst in C.OFFICIAL:
        for m in C.METHODS:
            r = by.get((inst, m))
            if r and r.get("rows"):
                for bud, w, sc, kind in r["rows"]:
                    at.append([inst, m, round(bud / 3600, 4), w, sc])
    write_csv(os.path.join(C.RESULTS, "anytime_official.csv"), ["instance", "method", "budget_h", "wall_utc", "score"], at)

    status_txt(runs, B, sup, L, wins, ranks)
    dashboard(runs, B, sup, L, fair, wins, ranks, table)


def status_txt(runs, B, sup, L, wins, ranks):
    by = {(r["instance"], r["method"]): r for r in runs}
    end = time.mktime(time.strptime(C.END, "%Y-%m-%dT%H:%M:%S"))
    days_left = (end - time.time()) / 86400
    out = []
    out.append(f"OLA 2027 campaign -- {time.strftime('%Y-%m-%d %H:%M')} (server time), "
               f"end {C.END.replace('T', ' ')} ({days_left:.1f} days left)")
    if os.path.exists(os.path.join(C.STATE, "STOP")):
        out.append("STOPPED (campaign/state/STOP exists; campaign/start.sh resumes)")
    if sup:
        out.append(f"supervisor pid {sup.get('pid')}, updated {sup.get('updated')}; {len(sup.get('running', {}))} CPU runs active; "
                   f"GPU: {sup.get('gpu') or '-'}; synthetic round level {sup.get('synthetic_level_h') or '-'} h; "
                   f"GPU round level {sup.get('gpu_level_h') or '-'} h")
    out.append("")
    cols = C.METHODS
    w0 = 14
    out.append("OFFICIAL INSTANCES  score  [change in 24 h]  (virtual ESA rank)")
    out.append("instance".ljust(w0) + "".join(C.LABEL[m][:13].rjust(15) for m in cols) + "   ESA #1")
    for inst in C.OFFICIAL:
        line = inst.ljust(w0)
        for m in cols:
            r = by.get((inst, m), {})
            line += (fmt(r.get("score")) or "-").rjust(15)
        b1 = B.get(inst, [(None, "")])[0]
        line += f"   {fmt(b1[0])} {b1[1]}"
        out.append(line)
        line = "".ljust(w0)
        for m in cols:
            r = by.get((inst, m), {})
            ch = r.get("change_24h")
            rk = r.get("esa_rank")
            s = (f"[{ch:+,}]" if ch is not None else "") + (f"(#{rk})" if rk else "")
            line += s.rjust(15)
        out.append(line)
    out.append("")
    if L is None:
        out.append("SYNTHETIC: no round completed by every synthetic run yet")
    else:
        out.append(f"SYNTHETIC at the last completed round, {L:g} h CPU per run: wins / mean rank "
                   f"({len(C.SYNTHETIC)} instances, {len(C.CPU_METHODS)} CPU methods)")
        for m in C.CPU_METHODS:
            rk = ranks[m]
            out.append(f"  {C.LABEL[m]:<20} wins {wins[m]:>2}   mean rank {sum(rk) / len(rk):.2f}" if rk else f"  {C.LABEL[m]:<20} -")
    out.append("")
    alerts = []
    for r in runs:
        if r.get("engine_ok") is False:
            alerts.append(f"engine re-score MISMATCH: {r['instance']}/{r['method']}")
        if r.get("esa_ok") is False:
            alerts.append(f"esa_eval re-score MISMATCH: {r['instance']}/{r['method']}")
        if (r.get("heartbeat_age_min") or 0) > 30 and r.get("state") in ("running", "periodic", "initial"):
            alerts.append(f"no heartbeat for {r['heartbeat_age_min']:.0f} min: {r['instance']}/{r['method']}")
    for p in (sup or {}).get("parked", []):
        alerts.append(f"parked after repeated crashes: {p}")
    checked = sum(1 for r in runs if r.get("esa_ok"))
    total = sum(1 for r in runs if r.get("score") is not None)
    out.append(f"checks: engine re-score on every pass; esa_eval confirmed {checked}/{total} current submissions")
    out.append("ALERTS: " + ("none" if not alerts else ""))
    out += ["  " + a for a in alerts]
    txt = "\n".join(out) + "\n"
    tmp = os.path.join(C.RESULTS, "STATUS.txt.tmp")
    with open(tmp, "w") as f:
        f.write(txt)
    os.replace(tmp, os.path.join(C.RESULTS, "STATUS.txt"))
    return txt


def _spark(rows, width=260, height=60):
    pts = [(b, s) for b, _, s, _ in rows]
    if len(pts) < 2:
        return ""
    xs = [p[0] for p in pts]
    ys = [-p[1] for p in pts]
    x0, x1 = min(xs), max(xs) or 1
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    path = []
    for x, y in zip(xs, ys):
        px = 4 + (x - x0) / max(x1 - x0, 1e-9) * (width - 8)
        py = height - 4 - (y - y0) / (y1 - y0) * (height - 8)
        path.append(f"{px:.1f},{py:.1f}")
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">'
            f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{" ".join(path)}"/></svg>')


def dashboard(runs, B, sup, L, fair, wins, ranks, table):
    by = {(r["instance"], r["method"]): r for r in runs}
    e = html.escape
    css = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--good:#1f7a4d;--bad:#b3261e;--hi:#eef4ff}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ecebe6;--muted:#a3a29c;--line:#2c2c2a;--good:#5cc28d;--bad:#ff8a80;--hi:#1d2533}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)} table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin:6px 0}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left} .wrap{overflow-x:auto}
td.best{background:var(--hi);font-weight:600} .up{color:var(--good)} .bad{color:var(--bad)}
.cards{display:flex;flex-wrap:wrap;gap:12px} .card{border:1px solid var(--line);border-radius:8px;padding:8px 10px}
"""
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>OLA 2027 campaign</title><style>{css}</style></head><body>"]
    out.append("<h1>OLA 2027 campaign</h1>")
    out.append(f"<div class='muted'>Updated {e(time.strftime('%Y-%m-%d %H:%M'))} server time · ends {e(C.END.replace('T', ' '))}"
               + (f" · supervisor pid {sup.get('pid')}, synthetic level {sup.get('synthetic_level_h')} h, GPU level {sup.get('gpu_level_h')} h"
                  if sup else "") + "</div>")
    # official table
    out.append("<h2>Official instances</h2><div class='wrap'><table><tr><th>instance</th>"
               + "".join(f"<th>{e(C.LABEL[m])}</th>" for m in C.METHODS) + "<th>ESA #1</th></tr>")
    for inst in C.OFFICIAL:
        vals = [by.get((inst, m), {}).get("score") for m in C.METHODS]
        best = min([v for v in vals if v is not None], default=None)
        out.append(f"<tr><td>{e(inst)}</td>")
        for m, v in zip(C.METHODS, vals):
            r = by.get((inst, m), {})
            ch = r.get("change_24h")
            rk = r.get("esa_rank")
            cls = " class='best'" if v is not None and v == best else ""
            sub = []
            if rk:
                sub.append(f"#{rk}")
            if ch:
                sub.append(f"<span class='up'>{ch:+,}</span>")
            out.append(f"<td{cls}>{fmt(v) or '–'}<div class='muted'>{' · '.join(sub)}</div></td>")
        b1 = B.get(inst, [(None, "")])[0]
        out.append(f"<td>{fmt(b1[0])}<div class='muted'>{e(b1[1])}</div></td></tr>")
    out.append("</table></div><div class='muted'>Score (lower is better), virtual rank on the ESA board, change in the last 24 h.</div>")
    # anytime sparklines
    out.append("<h2>Official instances over the run</h2><div class='cards'>")
    for inst in C.OFFICIAL:
        for m in C.METHODS:
            r = by.get((inst, m))
            if r and len(r.get("rows", [])) > 1:
                out.append(f"<div class='card'><div>{e(inst)} · {e(C.LABEL[m])}</div>{_spark(r['rows'])}"
                           f"<div class='muted'>{fmt(r.get('score'))} after {r.get('budget_h', 0):.1f} h</div></div>")
    out.append("</div>")
    # synthetic fair table
    out.append("<h2>Synthetic instances, equal budget</h2>")
    if L is None:
        out.append("<div class='muted'>No round completed by every synthetic run yet.</div>")
    else:
        out.append(f"<div class='muted'>Every synthetic CPU run at {L:g} h of CPU (the last round all of them completed).</div>")
        out.append("<div class='wrap'><table><tr><th>method</th><th>wins</th><th>mean rank</th></tr>")
        for m in sorted(C.CPU_METHODS, key=lambda m: (sum(ranks[m]) / len(ranks[m])) if ranks[m] else 99):
            rk = ranks[m]
            out.append(f"<tr><td>{e(C.LABEL[m])}</td><td>{wins[m]}</td><td>{(sum(rk) / len(rk)):.2f}</td></tr>" if rk else "")
        out.append("</table></div>")
        out.append("<div class='wrap'><table><tr><th>instance</th><th>family</th>"
                   + "".join(f"<th>{e(C.LABEL[m])}</th>" for m in C.CPU_METHODS) + "</tr>")
        for row in fair:
            vals = row[3:]
            best = min([v for v in vals if v is not None], default=None)
            out.append(f"<tr><td>{e(row[0])}</td><td>{e(row[1])}</td>" + "".join(
                f"<td{' class=best' if v is not None and v == best else ''}>{fmt(v)}</td>" for v in vals) + "</tr>")
        out.append("</table></div>")
    # current scores, all instances
    out.append("<h2>All instances, current best (runs at different budgets)</h2><div class='wrap'><table><tr><th>instance</th>"
               + "".join(f"<th>{e(C.LABEL[m])}</th>" for m in C.METHODS) + "</tr>")
    for inst in C.INSTANCES:
        vals = [by.get((inst, m), {}).get("score") for m in C.METHODS]
        best = min([v for v in vals if v is not None], default=None)
        out.append(f"<tr><td>{e(inst)}</td>" + "".join(
            f"<td{' class=best' if v is not None and v == best else ''}>{fmt(v) or '–'}"
            f"<div class='muted'>{by.get((inst, m), {}).get('budget_h', 0) or 0:.1f} h</div></td>"
            for m, v in zip(C.METHODS, vals)) + "</tr>")
    out.append("</table></div>")
    out.append("<h2>Status</h2><pre>" + e(open(os.path.join(C.RESULTS, "STATUS.txt")).read()) + "</pre>")
    out.append("</body></html>")
    tmp = os.path.join(C.RESULTS, "dashboard.html.tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(out))
    os.replace(tmp, os.path.join(C.RESULTS, "dashboard.html"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="check every submission with esa_eval")
    ap.add_argument("--print", action="store_true", help="print STATUS.txt afterwards")
    a = ap.parse_args()
    # one aggregation at a time (the supervisor's pass, status.sh --refresh,
    # snapshot.sh and stop.sh can overlap); a second caller waits
    import fcntl
    os.makedirs(C.STATE, exist_ok=True)
    with open(os.path.join(C.STATE, "aggregate.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        build(final=a.final)
    if a.print:
        print(open(os.path.join(C.RESULTS, "STATUS.txt")).read())


if __name__ == "__main__":
    main()

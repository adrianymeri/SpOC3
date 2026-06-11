#!/usr/bin/env python3
"""
make_figures.py -- generate the thesis figures into docs/figures/.

All numbers are the canonical-scorer-verified results of record; the Pareto
front is recomputed from the written portfolio submissions via core.evaluate.
Run: python3 tools/make_figures.py
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import (load_graph, build_adj_bitsets, graph_path, evaluate,
                  repo_root, MAX_TW)

HERE = repo_root()
FIG = os.path.join(HERE, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.family": "DejaVu Sans",
})
C_OLD, C_NEW, C_TOP = "#9aa7b4", "#2563eb", "#16a34a"
INSTANCES = ["small-graph", "medium-graph", "large-graph"]
N = {"small-graph": 1357, "medium-graph": 1399, "large-graph": 2426}

# --- canonical results of record ------------------------------------------
PERM = {"small-graph": -1_819_283, "medium-graph": -1_617_086, "large-graph": -5_033_531}
CMAES = {"small-graph": -1_828_451, "medium-graph": -1_711_954, "large-graph": -5_399_072}
TOP = {"small-graph": -1_829_919, "medium-graph": -1_745_122, "large-graph": -5_493_062}
# GBDT independent contribution (controlled ablation: WITH minus WITHOUT gbdt orderings)
ABLATION = {"small-graph": 0, "medium-graph": 0, "large-graph": 2524}
# Synthetic-benchmark generalisation: GBDT-adaptive contribution over a min-degree
# baseline, by graph size -> {density: HV added} (LightGBM; infeasible cells omitted).
GEN = {
    200:  {3: 165, 8: 489, 18: 672, 35: 814},
    350:  {3: 370, 8: 1389, 18: 2106, 35: 2414},
    500:  {3: 771, 8: 2986, 18: 4354, 35: 5235},
    750:  {3: 1902, 8: 5982, 18: 9580},
    1000: {3: 3087, 8: 10618},
}
EIG = {16: -5_292_737, 32: -5_376_007, 48: -5_353_725}          # large best single
ENGINE = {  # best single per instance, MATCHED 8 seeds / 32 eig / 600 s
    "builtin": {"small-graph": -1_827_610, "medium-graph": -1_693_446, "large-graph": -5_358_345},
    "fcmaes":  {"small-graph": -1_826_951, "medium-graph": -1_689_268, "large-graph": -5_335_648},
}
RANKS = [("grasp",3.848),("hc9",4.061),("vns",4.182),("vns_vnd",4.455),
         ("grasp_pr",5.000),("sms",6.530),("sa",6.803),("nsga2",6.924),
         ("sms_ls",7.803),("nsga2_ls",8.167),("sa_amosa",8.227),
         ("aco_mmas",13.333),("aco_mmas_ls",13.394),("aco_ls",13.455),("aco",13.818)]
CD = 3.483
SHORT = {"small-graph": "small", "medium-graph": "medium", "large-graph": "large"}


def pct_of_top(score, top):           # how close to leaderboard, %
    return 100.0 * score / top


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p, HERE))


# 1 ── % of leaderboard top: old vs new -------------------------------------
def fig_gap_closing():
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); w = 0.38
    old = [pct_of_top(PERM[i], TOP[i]) for i in INSTANCES]
    new = [pct_of_top(CMAES[i], TOP[i]) for i in INSTANCES]
    ax.bar(x - w/2, old, w, label="Permutation-space portfolio (4 chapters)", color=C_OLD)
    ax.bar(x + w/2, new, w, label="Continuous-encoding portfolio (this work)", color=C_NEW)
    for xi, v in zip(x - w/2, old): ax.text(xi, v + .1, f"{v:.1f}%", ha="center", fontsize=9, color="#555")
    for xi, v in zip(x + w/2, new): ax.text(xi, v + .1, f"{v:.1f}%", ha="center", fontsize=9, color=C_NEW, fontweight="bold")
    ax.axhline(100, ls="--", lw=1, color=C_TOP); ax.text(2.45, 100.15, "leaderboard top", color=C_TOP, fontsize=9, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in INSTANCES])
    ax.set_ylabel("hypervolume as % of leaderboard top"); ax.set_ylim(88, 102)
    ax.set_title("Closing the gap: permutation search vs. continuous encoding")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=9, frameon=False)
    save(fig, "fig1_gap_closing.png")


# 2 ── remaining gap (HV short of top), old vs new, log scale ---------------
def fig_remaining_gap():
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3); w = 0.38
    # score = -HV; gap to top = score - TOP (positive, in thousands of HV)
    old = [(PERM[i] - TOP[i]) / 1000 for i in INSTANCES]
    new = [(CMAES[i] - TOP[i]) / 1000 for i in INSTANCES]
    ax.bar(x - w/2, old, w, label="Permutation portfolio", color=C_OLD)
    ax.bar(x + w/2, new, w, label="Continuous encoding", color=C_NEW)
    top_lim = max(old) * 1.12
    for xi, v in zip(x - w/2, old): ax.text(xi, v + top_lim*.012, f"{v*1000:,.0f}", ha="center", fontsize=8, color="#555")
    for xi, v in zip(x + w/2, new): ax.text(xi, v + top_lim*.012, f"{v*1000:,.0f}", ha="center", fontsize=8, color=C_NEW, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in INSTANCES])
    ax.set_ylabel("HV short of leaderboard top  (thousands)")
    ax.set_ylim(0, top_lim)
    ax.set_title("Remaining gap to the top (lower is better)")
    ax.legend(fontsize=9)
    save(fig, "fig2_remaining_gap.png")


# 3 ── Pareto front of the large-graph portfolio ----------------------------
def fig_pareto_large():
    prob = "large-graph"; n = N[prob]
    nn, adj = load_graph(graph_path(HERE, prob)); ab = build_adj_bitsets(nn, adj)
    dvs = json.load(open(os.path.join(HERE, "submissions", prob, "portfolio.json")))[0]["decisionVector"]
    pts = []
    for dv in dvs:
        perm, t = [int(x) for x in dv[:-1]], int(dv[-1])
        w, tr = evaluate(perm, t, ab, nn)
        if w <= MAX_TW: pts.append((w, tr))
    pts.sort()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # staircase
    sx, sy = [], []
    best = n
    for w, t in pts:
        sx += [w, w]; sy += [best, t]; best = t
    ax.step([p[0] for p in pts] + [n], [p[1] for p in pts] + [pts[-1][1]], where="post",
            color=C_NEW, lw=1.5, alpha=.45, zorder=2)
    ax.scatter(xs, ys, color=C_NEW, zorder=5, s=34, label="submitted front (20 points)")
    ax.scatter([n], [n], marker="*", s=240, color=C_TOP, zorder=6,
               label=f"reference point (n, n) = ({n}, {n})")
    ax.axvline(MAX_TW, color="#dc2626", ls=":", lw=1.2, zorder=1)
    ax.text(MAX_TW - 30, n*0.5, "width cap = 500", color="#dc2626", rotation=90,
            va="center", ha="right", fontsize=9)
    ax.set_xlabel("torso width  (max fill-in degree)"); ax.set_ylabel("threshold t")
    ax.set_xlim(-40, n + 60); ax.set_ylim(-80, n + 90)
    ax.set_title("large-graph submitted Pareto front  (−HV = 5,383,985)")
    ax.legend(fontsize=9, loc="center right", framealpha=.95)
    save(fig, "fig3_pareto_large.png")


# 4 ── eigenvector trade-off (large best single vs K) -----------------------
def fig_eig_tradeoff():
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ks = sorted(EIG); ys = [-EIG[k] for k in ks]   # plot +HV (higher better)
    lo, hi = min(ys), max(ys); span = hi - lo
    ax.set_ylim(lo - span*0.18, hi + span*0.32)   # generous headroom for labels + title
    ax.plot(ks, ys, "-o", color=C_NEW, lw=2.2, ms=9, zorder=3)
    # peak label BELOW the point (keeps clear of the title); others above
    lbl = {16: (0, 11, "bottom"), 32: (0, -17, "top"), 48: (0, 11, "bottom")}
    for k in ks:
        dx, dy, va = lbl[k]
        ax.annotate(f"{-EIG[k]:,.0f}", (k, -EIG[k]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va=va, fontsize=9.5,
                    fontweight="bold" if k == 32 else "normal",
                    color=C_NEW if k == 32 else "#333")
    ax.annotate("knee:\nrich enough, still converges", (32, -EIG[32]),
                textcoords="offset points", xytext=(40, -6), fontsize=9, color="#333",
                ha="left", va="center", arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xticks(ks); ax.set_xlim(11, 53)
    ax.set_xlabel("spectral features K  (Laplacian eigenvectors)")
    ax.set_ylabel("large-graph HV (best single seed)")
    ax.set_title("Why K = 32: the features-vs-convergence trade-off", pad=12)
    save(fig, "fig4_eigenvector_tradeoff.png")


# 5 ── engine comparison: builtin vs fcmaes ---------------------------------
def fig_engine():
    # Instances differ ~3x in HV, so plot the *advantage* (builtin HV - fcmaes HV)
    # per instance: all positive => diagonal wins everywhere, on a common scale.
    fig, ax = plt.subplots(figsize=(7, 4.3))
    adv = [(-ENGINE["builtin"][i]) - (-ENGINE["fcmaes"][i]) for i in INSTANCES]
    x = np.arange(3)
    ax.bar(x, adv, 0.5, color=C_NEW, zorder=3)
    for xi, v in zip(x, adv):
        ax.text(xi, v + max(adv)*0.03, f"+{v:,.0f} HV", ha="center",
                fontsize=11, fontweight="bold", color=C_NEW)
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in INSTANCES])
    ax.set_ylabel("builtin advantage  (HV$_{builtin}$ − HV$_{fcmaes}$)")
    ax.set_ylim(0, max(adv)*1.20)
    ax.set_title("Optimiser is not the bottleneck:\ndiagonal CMA-ES beats full-covariance fcmaes on every instance\n(matched 8 seeds · 32 eigenvectors · 600 s)",
                 fontsize=11)
    save(fig, "fig5_engine_comparison.png")


# 6 ── Friedman/Nemenyi average ranks ---------------------------------------
def fig_ranks():
    fig, ax = plt.subplots(figsize=(7, 5))
    names = [r[0] for r in RANKS][::-1]; vals = [r[1] for r in RANKS][::-1]
    cols = ["#dc2626" if "aco" in nme else (C_NEW if v <= RANKS[0][1] + CD else C_OLD)
            for nme, v in zip(names, vals)]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    best = RANKS[0][1]
    ax.axvline(best, color=C_NEW, ls="-", lw=1)
    ax.axvspan(best, best + CD, color=C_NEW, alpha=.08)
    ax.axvline(best + CD, color=C_NEW, ls="--", lw=1)
    ax.text(best + CD, len(names)-0.5, f"  critical distance = {CD}", color=C_NEW, fontsize=8, va="top")
    ax.set_xlabel("Friedman average rank  (lower = better, 33 blocks)")
    ax.set_title("Permutation-family ranking (ACO red, beyond CD = significantly worse)")
    save(fig, "fig6_method_ranks.png")


def fig_ablation():
    """GBDT independent contribution per instance (controlled ablation)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(3); vals = [ABLATION[i] for i in INSTANCES]
    cols = [C_OLD if v == 0 else C_NEW for v in vals]
    ax.bar(x, vals, 0.5, color=cols, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.03 + 8, f"+{v:,} HV" if v else "+0",
                ha="center", fontsize=11,
                fontweight="bold" if v else "normal",
                color=C_NEW if v else "#777")
    ax.axhline(0, color="#888", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[i] for i in INSTANCES])
    ax.set_ylabel("GBDT contribution to portfolio HV\n(WITH − WITHOUT gbdt orderings)")
    ax.set_ylim(-50, max(vals) * 1.25)
    ax.set_title("Boosted-tree learning helps where adaptivity matters:\n"
                 "+2,524 HV on the dense instance, neutral on the saturated ones",
                 fontsize=11)
    save(fig, "fig7_gbdt_ablation.png")


def fig_importance():
    """GBDT feature importances on large-graph — dynamic features dominate."""
    data = [("elim_nbr", 21.5, "dyn"), ("cur_deg", 19.3, "dyn"), ("fill", 10.7, "dyn"),
            ("eig1", 3.2, "spec"), ("nbr_deg_mean", 2.7, "struct"), ("eig2", 2.6, "spec"),
            ("nbr_deg_max", 2.5, "struct"), ("eig3", 2.4, "spec"), ("eig14", 2.1, "spec"),
            ("nbr_deg_std", 1.7, "struct")]
    col = {"dyn": C_NEW, "spec": "#16a34a", "struct": "#9aa7b4"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = [d[0] for d in data][::-1]; vals = [d[1] for d in data][::-1]
    cols = [col[d[2]] for d in data][::-1]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("LightGBM feature importance (%)")
    ax.set_title("What the heuristic learned: dynamic live-graph features dominate\n"
                 "(dynamic 51.5% · spectral ≈15–20% · degree stats secondary)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col["dyn"], label="dynamic (live graph)"),
                       Patch(color=col["spec"], label="spectral (Laplacian)"),
                       Patch(color=col["struct"], label="static degree stats")],
              fontsize=9, loc="lower right")
    save(fig, "fig9_feature_importance.png")


def fig_generalization():
    """GBDT-adaptive contribution vs density, one line per graph size — the
    honest within-size view (the cross-density mean is confounded by infeasible
    dense-large cells and by size scaling)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sizes = sorted(GEN)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(sizes)))
    for c, sz in zip(cmap, sizes):
        ds = sorted(GEN[sz]); ys = [GEN[sz][d] for d in ds]
        ax.plot(ds, ys, "-o", color=c, lw=2, ms=7, label=f"n = {sz}")
    ax.set_xticks([3, 8, 18, 35])
    ax.set_xlabel("graph density  d  (avg degree proxy)")
    ax.set_ylabel("GBDT-adaptive contribution (HV added over min-degree)")
    ax.set_title("Boosted-tree learning helps in proportion to density\n"
                 "(strictly monotone within every graph size; 17/17 feasible "
                 "instances positive)", fontsize=11)
    ax.legend(title="graph size", fontsize=9, loc="upper left")
    save(fig, "fig8_generalization.png")


# --- GBFC / GBFC++ results of record (THESIS §11 / §11.6) -------------------
GBFC = {"small-graph": -1_828_994, "medium-graph": -1_712_688, "large-graph": -5_431_924}
GBFCPP_SMALL = -1_829_735          # verified 11 Jun 2026, tools/portfolio.py re-score
# paired same-seed single-round ablation, small-graph, 30 s, pre-GBFC++ pool
GBFCPP_ABL = [("seed 42\n(bp w=13)", 52, 34), ("seed 7\n(bp w=11)", 40, 22),
              ("seed 13\n(bp w=14)", 42, 0)]


def fig_gbfcpp():
    """(a) small-graph: % of leaderboard top by paradigm, ending at GBFC++.
    (b) the paired GBDT-proposal ablation (same seed => same breakpoint/zone)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                  gridspec_kw={"width_ratios": [1.25, 1]})
    prob = "small-graph"
    steps = [("permutation\nportfolio (§2)", PERM[prob], C_OLD),
             ("continuous\nCMA-ES (§5)", CMAES[prob], C_NEW),
             ("GBFC (§11)", GBFC[prob], "#7c3aed"),
             ("GBFC++ (§11.6)", GBFCPP_SMALL, "#dc2626")]
    xs = np.arange(len(steps))
    vals = [pct_of_top(s, TOP[prob]) for _, s, _ in steps]
    ax.bar(xs, vals, 0.6, color=[c for _, _, c in steps])
    for x, v, (_, s, c) in zip(xs, vals, steps):
        ax.text(x, v + .004, f"{v:.3f}%", ha="center", fontsize=9,
                color=c, fontweight="bold")
        ax.text(x, 99.255, f"{s:,.0f}".replace(",", " "), ha="center",
                fontsize=7.5, color="#fff", rotation=90, va="bottom")
    ax.axhline(100, ls="--", lw=1, color=C_TOP)
    ax.text(len(steps)-0.55, 100.005, "leaderboard top (−1 829 919)",
            color=C_TOP, fontsize=8.5, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _, _ in steps], fontsize=9)
    ax.set_ylabel("hypervolume as % of leaderboard top")
    ax.set_ylim(99.25, 100.03)
    ax.set_title("(a) small-graph: every paradigm step, to 99.990 %")

    labels = [a for a, _, _ in GBFCPP_ABL]
    xs2 = np.arange(len(labels)); w = 0.36
    g = [b for _, b, _ in GBFCPP_ABL]; ng = [c for _, _, c in GBFCPP_ABL]
    ax2.bar(xs2 - w/2, g, w, color="#dc2626", label="GBDT move proposals")
    ax2.bar(xs2 + w/2, ng, w, color=C_OLD, label="uniform proposals (--no-gbdt)")
    for x, v in zip(xs2 - w/2, g):
        ax2.text(x, v + 1, f"+{v}", ha="center", fontsize=9, color="#dc2626",
                 fontweight="bold")
    for x, v in zip(xs2 + w/2, ng):
        ax2.text(x, v + 1, f"+{v}", ha="center", fontsize=9, color="#555")
    ax2.set_xticks(xs2); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("HV gained in one paired 30 s round")
    ax2.set_title("(b) the GBDT proposal is load-bearing (3/3 pairs)")
    ax2.legend(fontsize=9, frameon=False)
    save(fig, "fig13_gbfcpp_small.png")


GBFCPP_SMALL_LIVE = -1_829_735     # verified 11 Jun 2026 (swarm, still improving)
QNE_REPRO_SMALL = -1_828_493       # winner's engine, 49 CPU gens / 30 s


def fig_gbdt_ledger():
    """(a) % of leaderboard top by method family x instance (THESIS §12.1);
    (b) controlled GBDT contributions by mechanism, log scale (§12.3)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    fams = [("permutation\n(§2)", PERM, C_OLD),
            ("CMA-ES+GBDT\nconstructor (§5–6b)", CMAES, C_NEW),
            ("GBFC (§11)", GBFC, "#7c3aed")]
    xs = np.arange(3); w = 0.2
    for k, (label, scores, color) in enumerate(fams):
        vals = [pct_of_top(scores[i], TOP[i]) for i in INSTANCES]
        ax.bar(xs + (k - 1.5) * w, vals, w, label=label, color=color)
    # GBFC++ exists on small only -- the red cap
    ax.bar([xs[0] + 1.5 * w], [pct_of_top(GBFCPP_SMALL_LIVE, TOP["small-graph"])],
           w, label="GBFC++ (§11.6, small)", color="#dc2626")
    ax.axhline(100, ls="--", lw=1, color=C_TOP)
    ax.text(2.45, 100.2, "cuda-torso (leaderboard top)", color=C_TOP,
            fontsize=8.5, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[i] for i in INSTANCES])
    ax.set_ylabel("hypervolume as % of leaderboard top"); ax.set_ylim(88, 102)
    ax.set_title("(a) every paradigm, every instance")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    mech = [("constructor §6b\n(large)", 2524, "#2563eb"),
            ("GAPS decode §10\n(large)", 24766, "#0891b2"),
            ("GBFC weak learner §11\n(small)", 688, "#7c3aed"),
            ("GBFC weak learner §11\n(medium)", 734, "#7c3aed"),
            ("GBFC weak learner §11\n(large)", 329, "#7c3aed"),
            ("GBFC++ machinery §11.6\n(small, over GBFC)", 741, "#dc2626")]
    ys = np.arange(len(mech))[::-1]
    ax2.barh(ys, [m[1] for m in mech], color=[m[2] for m in mech])
    for y, (_, v, c) in zip(ys, mech):
        ax2.text(v * 1.12, y, f"+{v:,}", va="center", fontsize=9, color=c,
                 fontweight="bold")
    ax2.set_yticks(ys); ax2.set_yticklabels([m[0] for m in mech], fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlim(100, 90000)
    ax2.set_xlabel("verified GBDT contribution (HV, log scale)")
    ax2.set_title("(b) the GBDT mechanisms, measured")
    save(fig, "fig14_gbdt_ledger.png")


if __name__ == "__main__":
    fig_gap_closing(); fig_remaining_gap(); fig_pareto_large()
    fig_eig_tradeoff(); fig_engine(); fig_ranks(); fig_ablation()
    fig_generalization(); fig_importance(); fig_gbfcpp(); fig_gbdt_ledger()
    print("done ->", os.path.relpath(FIG, HERE))

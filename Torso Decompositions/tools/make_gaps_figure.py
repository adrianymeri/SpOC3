#!/usr/bin/env python3
"""
make_gaps_figure.py -- fig11 for THESIS.md §10: GAPS (the novel method) on
large-graph, with the controlled GBDT ablation.

Left: the with-GBDT GAPS run's best-score trajectory (single Tesla T4, seed 42,
warm-started from the banked portfolio). The score is flat until generation 8 --
the first GBDT retrain -- then breaks loose: the improvement is time-locked to
the GBDT mechanism engaging. Right: the verified, official top-20 submission
scores (re-scored by tools/portfolio.py): the linear GPU baseline, GAPS with the
GBDT column DISABLED (poly-only control), and full GAPS. The gap between the last
two is GBDT's controlled contribution in the novel method (+24,766 HV).

    python3 tools/make_gaps_figure.py   # writes docs/figures/fig11_gaps_ablation.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# with-GBDT trajectory (gen, best score) -- sparse points from the 90-min run log
TRAJ = [(1,-5398113),(6,-5398113),(7,-5398547),(8,-5398858),(9,-5407012),
        (10,-5420195),(11,-5422807),(14,-5423551),(33,-5423675),(37,-5423861),
        (40,-5424481),(44,-5426286),(47,-5427775),(50,-5428463),(56,-5430206),
        (60,-5431200),(68,-5432506),(77,-5433812),(90,-5435120),(100,-5435993),
        (120,-5436491),(145,-5437116)]

# verified official top-20 submission HV (tools/portfolio.py methodology)
LINEAR   = -5404348      # gpucma (linear GPU baseline)
NOGBDT   = -5406555      # GAPS, GBDT column disabled (poly-only control)
GAPS     = -5431321      # GAPS with GBDT
LEADER   = -5493062


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(here, "docs", "figures"); os.makedirs(outdir, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    # ---- left: trajectory, GBDT onset locked at gen 8 ----
    xs = [g for g, _ in TRAJ]; ys = [s for _, s in TRAJ]
    axL.plot(xs, ys, "-o", ms=3.5, lw=1.8, color="#1f77b4", label="GAPS (with GBDT)")
    axL.axhline(NOGBDT, ls="--", lw=1.3, color="#7f7f7f",
                label="GAPS without GBDT (control)")
    axL.axhline(LINEAR, ls=":", lw=1.3, color="#2ca02c", label="linear GPU baseline")
    axL.axvline(8, color="#d62728", lw=1.2, alpha=0.7)
    axL.text(8.6, -5435000, "GBDT column\nfirst trains (gen 8)", color="#d62728",
             fontsize=8.5, va="center")
    axL.set_xlabel("generation (4,096 GPU-scored orderings each)")
    axL.set_ylabel("best score  (−HV, more negative = better)")
    axL.set_title("GAPS improves only once GBDT engages", fontsize=10.5)
    axL.legend(frameon=False, fontsize=8.5, loc="upper right")
    axL.grid(True, alpha=0.25)
    axL.ticklabel_format(axis="y", style="plain")

    # ---- right: verified ablation bars ----
    labels = ["linear\nbaseline", "GAPS\nno-GBDT", "GAPS\n+GBDT"]
    vals = [LINEAR, NOGBDT, GAPS]
    base = -5_400_000  # bar baseline so differences are visible
    heights = [base - v for v in vals]   # positive HV below the baseline
    colors = ["#2ca02c", "#7f7f7f", "#1f77b4"]
    bars = axR.bar(labels, heights, color=colors)
    for b, v in zip(bars, vals):
        axR.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:,.0f}",
                 ha="center", va="bottom", fontsize=8.5)
    axR.annotate("", xy=(2, base - GAPS), xytext=(2, base - NOGBDT),
                 arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.5))
    axR.text(2.05, base - (GAPS+NOGBDT)/2, "GBDT\n+24,766 HV", color="#d62728",
             fontsize=9, va="center")
    axR.set_ylabel(f"HV below {base:,.0f}")
    axR.set_title("Controlled GBDT contribution (verified)", fontsize=10.5)
    axR.grid(True, axis="y", alpha=0.25)

    fig.suptitle("GAPS — a learned nonlinear (GBDT-driven) decode beats the linear one (large-graph)",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(outdir, "fig11_gaps_ablation.png")
    fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()

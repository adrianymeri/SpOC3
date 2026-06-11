#!/usr/bin/env python3
"""
make_gpu_figure.py -- fig10 for THESIS.md §9: the GPU search's
improve-then-plateau signature on large- and medium-graph.

Data are the per-generation best scores logged by tools/gpu_search.py on a
single Tesla T4 (Colab), warm-started from the banked portfolio. Plotting the
HV *gained over the warm start* puts both instances on a common axis and makes
the shared signature visible: a burst of improvement, then a hard plateau that
~10^5 further evaluations do not move -- the evidence that the residual gap is a
decode-expressiveness limit, not a throughput one.

    python3 tools/make_gpu_figure.py     # writes docs/figures/fig10_gpu_plateau.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (generation, best score) -- sparse points from the run logs (monotone fill).
LARGE = [(0,-5398113),(5,-5398113),(6,-5398175),(7,-5398857),(8,-5399415),
         (17,-5399539),(18,-5399601),(20,-5399973),(22,-5400407),(24,-5401461),
         (25,-5402280),(26,-5403458),(28,-5404078),(33,-5405016),(34,-5406020),
         (36,-5406454),(37,-5406517),(47,-5406643),(54,-5406768),(55,-5406768)]
MEDIUM = [(0,-1711379),(30,-1711379),(31,-1711559),(36,-1711773),(44,-1711915),
          (46,-1712059),(48,-1712166),(53,-1712310),(56,-1712346),(67,-1712490),
          (78,-1712562),(88,-1712598),(101,-1712633),(108,-1712668),(139,-1712668)]


def gained(series):
    g0 = series[0][1]
    xs = [g for g, _ in series]
    ys = [(s - g0) for _, s in series]   # s and g0 negative; improvement = s-g0 < 0
    return xs, [-y for y in ys]          # plot positive HV gained


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(here, "docs", "figures")
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for series, label, color in [(LARGE, "large-graph", "#1f77b4"),
                                 (MEDIUM, "medium-graph", "#d62728")]:
        xs, ys = gained(series)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=color, label=label)

    ax.axvspan(0, 8, color="0.92", zorder=0)
    ax.text(4, ax.get_ylim()[1]*0.96, "JIT /\nwarm-up", ha="center", va="top",
            fontsize=8, color="0.4")
    ax.annotate("plateau — ~10^5 further evaluations\nadd almost nothing",
                xy=(120, 1289), xytext=(70, 4200), fontsize=9, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
    ax.set_xlabel("generation (4,096 GPU-scored orderings each)")
    ax.set_ylabel("HV gained over warm start (search-internal estimate)")
    ax.set_title("GPU search improves, then plateaus — a decode limit, not a throughput limit",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    fig.tight_layout()
    out = os.path.join(outdir, "fig10_gpu_plateau.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()

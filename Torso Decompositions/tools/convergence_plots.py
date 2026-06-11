#!/usr/bin/env python3
"""
convergence_plots.py -- generate archive-HV-vs-wall-time figures.

Two-phase to fit sandbox time limits:

  Phase 1 (per-(algo, problem) collection):
      python3 tools/convergence_plots.py collect --algo hc9 --problem small-graph
    Runs the algorithm at a given seed, parses the
    `iter ... | score = -X | t = Xs` log lines, saves a CSV at
    extra_instances/conv_{problem}_{algo}.csv.

  Phase 2 (plotting after all CSVs exist):
      python3 tools/convergence_plots.py plot [--problem small-graph]
    Reads the CSVs and writes
    extra_instances/convergence_{small,medium,large}.png.
"""

from __future__ import annotations

import sys
import os
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

ALGOS = {
    "hc5":  ("algorithms/hill_climbing/hc5_operators.py",   "hc5 (lex, 12 ops)",          "#1f77b4"),
    "hc7":  ("algorithms/hill_climbing/hc7_kbottleneck.py", "hc7 (K-bottleneck)",         "#9467bd"),
    "hc9":  ("algorithms/hill_climbing/hc9_hv_accept.py",   "hc9 (HV-accept, 12 ops)",    "#d62728"),
    "hc11": ("algorithms/hill_climbing/hc11_ils.py",        "hc11 (ILS, 12 ops)",         "#2ca02c"),
}

BUDGETS = {
    # (wall budget seconds, progress-every iters)
    "small-graph":  (25.0,  50),
    "medium-graph": (12.0,  20),
    "large-graph":  (25.0,  10),
}

PROGRESS_RE = re.compile(
    r"iter\s+(\d+)\s*\|.*?score\s*=\s*([-\d,]+).*?t\s*=\s*([\d.]+)s"
)


def collect(algo: str, problem: str, seed: int) -> None:
    script, _, _ = ALGOS[algo]
    budget, progress_every = BUDGETS[problem]

    sub_path = os.path.join(_HERE, "submissions", problem, f"{algo}.json")
    backup_path = sub_path + ".convbak"
    if os.path.exists(sub_path):
        shutil.copy2(sub_path, backup_path)

    cmd = [
        sys.executable, script,
        "--problem", problem,
        "--budget", str(budget),
        "--seed", str(seed),
        "--progress-every", str(progress_every),
    ]
    proc = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)

    samples: list[tuple[float, float]] = []
    for line in proc.stdout.splitlines():
        m = PROGRESS_RE.search(line)
        if m:
            score = -float(m.group(2).replace(",", ""))
            t_s = float(m.group(3))
            samples.append((t_s, score))

    # Restore canonical submission
    if os.path.exists(backup_path):
        shutil.move(backup_path, sub_path)

    out_path = os.path.join(_HERE, "extra_instances",
                            f"conv_{problem}_{algo}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "hv"])
        for t, h in samples:
            w.writerow([f"{t:.3f}", f"{h:.0f}"])
    print(f"Wrote {out_path}  ({len(samples)} samples, "
          f"final HV = {samples[-1][1]:,.0f}" if samples else
          f"Wrote {out_path}  (0 samples!)")


def plot(problem: str, seed: int, inline: bool = False,
         algos_subset: list | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget, _ = BUDGETS[problem]
    figsize = (5.0, 2.8) if inline else (7.5, 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    use = algos_subset if algos_subset else list(ALGOS.keys())
    for algo in use:
        if algo not in ALGOS:
            continue
        _, label, colour = ALGOS[algo]
        csv_path = os.path.join(_HERE, "extra_instances",
                                f"conv_{problem}_{algo}.csv")
        if not os.path.exists(csv_path):
            print(f"  ! missing {csv_path}; run `collect --algo {algo} "
                  f"--problem {problem}` first")
            continue
        ts, ys = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ts.append(float(row["t_s"]))
                ys.append(float(row["hv"]))
        if not ts:
            print(f"  ! {csv_path} empty"); continue
        ax.plot(ts, ys, marker="o", markersize=2 if inline else 3,
                linewidth=1.0 if inline else 1.3,
                label=label, color=colour)
        print(f"  {label}: {len(ts)} samples, final HV = {ys[-1]:,.0f}")

    ax.set_xlabel("Wall time (s)" if inline else "Wall time (seconds)",
                  fontsize=8 if inline else 11)
    ax.set_ylabel("Archive HV (higher better)" if inline else
                  "Pareto-archive hypervolume (higher is better)",
                  fontsize=8 if inline else 11)
    if not inline:
        ax.set_title(f"Convergence on {problem}  "
                     f"(seed = {seed}, budget = {budget:.0f}s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7 if inline else 9)
    ax.tick_params(axis="both", labelsize=7 if inline else 10)
    fig.tight_layout()

    if inline:
        out_dir = os.path.join(_HERE, "docs", "figures")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f"convergence_{problem.split('-')[0]}_inline.png")
    else:
        out_path = os.path.join(_HERE, "extra_instances",
                                f"convergence_{problem.split('-')[0]}.png")
    fig.savefig(out_path, dpi=200 if inline else 140, bbox_inches="tight")
    print(f"  Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p_c = sub.add_parser("collect")
    p_c.add_argument("--algo", required=True, choices=ALGOS.keys())
    p_c.add_argument("--problem", required=True, choices=BUDGETS.keys())
    p_c.add_argument("--seed", type=int, default=42)

    p_p = sub.add_parser("plot")
    p_p.add_argument("--problem", default=None,
                     help="default: plot all three")
    p_p.add_argument("--seed", type=int, default=42)
    p_p.add_argument("--inline", action="store_true",
                     help="paper-ready compact PNG → Figures/convergence_<problem>_inline.png")
    p_p.add_argument("--algos", default=None,
                     help="comma-separated subset of algos to include")

    args = ap.parse_args()
    if args.mode == "collect":
        collect(args.algo, args.problem, args.seed)
    else:
        problems = [args.problem] if args.problem else list(BUDGETS.keys())
        algos_subset = (
            [a.strip() for a in args.algos.split(",") if a.strip()]
            if args.algos else None
        )
        for p in problems:
            print(f"\n--- {p} ---")
            plot(p, args.seed, inline=args.inline, algos_subset=algos_subset)


if __name__ == "__main__":
    main()

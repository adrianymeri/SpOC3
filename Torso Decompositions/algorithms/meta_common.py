#!/usr/bin/env python3
"""
meta_common.py -- shared scaffolding for the metaheuristic chapter
(Simulated Annealing, GRASP, VNS).

Design rationale
----------------
The Hill-Climbing chapter (algorithms/hill_climbing/) already contains a
mature, well-tuned move pool of 15 operators plus the archive-seeding,
warm-start and submission-writing plumbing.  To keep the new techniques
*directly comparable* to the HC chapter we reuse exactly that machinery
rather than re-deriving it: same operators, same instances/seeds/budgets,
same Pareto archive, same -HV scoring, same submission format.

This module exposes the common pieces so each metaheuristic only has to
implement its distinctive search logic:

  - OPERATORS / call_op   : the 15-operator move pool from hc9
  - hv_with_candidate     : marginal-HV helper (linear time)
  - set_adj_bits          : point the min-fill operator at the live graph
  - load_problem          : graph + adjacency bitsets
  - build_t_grid          : threshold seed grid (identical formula to hc9)
  - seed_archive          : seed the archive across the t-grid
  - finalize              : top-k-by-HV-contribution + write submission
  - energy                : scalar single-point HV (rectangle area) for SA

Reusing the hc9 operator module also means the min-fill-reinsert operator
sees the correct adjacency bitsets (a subtle bug that bit hc14, which
imported operators without setting the module global).
"""

from __future__ import annotations

# --- sys.path bootstrap -----------------------------------------------------
import sys as _sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root != _os.path.dirname(_root) and not _os.path.exists(
        _os.path.join(_root, "core.py")):
    _root = _os.path.dirname(_root)
_sys.path.insert(0, _root)

from typing import Callable, List, Tuple

# Reuse the entire move pool and HV helper from hc9 so the metaheuristics
# search the identical neighbourhood the HC chapter used.
import algorithms.hill_climbing.hc9_hv_accept as _hc9
from core import (
    ParetoArchive,
    build_adj_bitsets,
    ensure_seeded,
    evaluate_full,
    graph_path,
    hypervolume_2d,
    load_graph,
    submission_path,
    write_submission,
)

__all__ = [
    "OPERATORS", "call_op", "hv_with_candidate", "set_adj_bits",
    "load_problem", "build_t_grid", "seed_archive", "energy", "finalize",
]

OPERATORS: List[Tuple[str, Callable]] = _hc9.OPERATORS
call_op = _hc9._call_op
hv_with_candidate = _hc9.hv_with_candidate


def set_adj_bits(adj_bits: List[int]) -> None:
    """Point the (module-global) min-fill-reinsert operator at the current
    graph.  Must be called once per run before any operator is applied."""
    _hc9._ADJ_BITS = adj_bits


def load_problem(here: str, problem: str):
    """Return (n, adj, adj_bits) for `problem` and wire up the operators."""
    n, adj = load_graph(graph_path(here, problem))
    adj_bits = build_adj_bitsets(n, adj)
    set_adj_bits(adj_bits)
    return n, adj, adj_bits


def build_t_grid(n: int, num_t_seeds: int) -> List[int]:
    """Threshold seed grid -- identical formula to hc9 for comparability."""
    if num_t_seeds <= 1:
        return [0]
    return sorted({int(round(i * (n - 1) / (num_t_seeds - 1)))
                   for i in range(num_t_seeds)})


def seed_archive(archive: ParetoArchive, perm: List[int], adj_bits, n: int,
                 t_grid: List[int]) -> None:
    """Seed `archive` with `perm` evaluated at every threshold in t_grid."""
    for tt in t_grid:
        _, w, _, _, _ = evaluate_full(perm, tt, adj_bits, n)
        archive.try_add(int(w), int(tt), perm[:])


def energy(w: int, t: int, n: int) -> float:
    """Scalar SA energy: the *negative* single-point hypervolume of the
    decision (w, t) against the reference point (n, n).  A move that
    enlarges the dominated rectangle lowers the energy.

        HV_point = (n - w) * (n - t)        (area dominated by one point)
        E        = -(n - w) * (n - t)

    Lower energy == larger dominated area == better.  Width-cap violations
    (w >= n, i.e. the 501 over-width fallback) give zero/positive energy and
    are therefore strongly penalised, exactly as the archive would reject
    them.
    """
    return -float((n - w) * (n - t))


def finalize(archive: ParetoArchive, n: int, here: str, problem: str,
             algo: str, top_k: int = 20) -> Tuple[str, int, float]:
    """Write the top-k-by-HV-contribution submission and return
    (out_path, n_vectors, final_score).

    The returned score is the HV of *exactly the written subset* (the
    top-k front), not the full runtime archive.  Only the top-k vectors
    are submitted, so this is the score `tools/verify_submission.py`
    reproduces end-to-end; reporting the full-archive HV here would
    overstate the submitted score whenever the archive holds more than
    `top_k` points.
    """
    top = archive.top_k_by_hv_contribution(top_k, n)
    decision_vectors = [list(p) + [int(t)] for (_, t, p) in top]
    out_path = submission_path(here, problem, algo)
    write_submission(decision_vectors, problem, out_path)
    submitted_hv = hypervolume_2d([(w, t) for (w, t, _) in top], n)
    return out_path, len(decision_vectors), -submitted_hv

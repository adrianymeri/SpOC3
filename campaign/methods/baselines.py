"""
baselines.py -- the three CPU baselines of the OLA2027 table, on the shared engine.

Each keeps the search logic of the repository's own implementation (master,
used for the AlbShkenca 2026 study) and changes only what a 3-week resumable
run needs. Every change is listed in the class docstring.

  HRI      Team HRI's multi-objective LNS      leaderboard-references/hri_lns.py
  HC       Hill Climbing (this group's)        hill_climbing.py
  FCMAES   the fcmaes-style CMA-ES stand-in    leaderboard-references/cmaes.py
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

from common import Method, np_rng_state, np_rng_from_state
import torso_engine as te

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


# ============================================================== Team HRI

class HRI(Method):
    """Team HRI's MO-LNS (hri_lns.solve, start="min_degree"), multi-start.

    Operator set B: neighbour destroy (seed + neighbours, capped at 220) and
    balanced repair (median placement, Biedl et al. 2005 §5). Operator set A:
    6 random removals reinserted at random. Switch sets after 3,000 iterations
    without improvement. A candidate becomes the current ordering when it
    improves its run's capped 20-point score, exactly as hri_lns.py.

    Multi-start: a single run stops improving within about 15 CPU-minutes (on
    the server, no change for hours on all three official graphs). From a
    fixed ordering, set B offers about one move per vertex, because the repair
    is deterministic. So after 24,000 iterations (eight set switches) without
    an accepted move, a new independent run starts from a fresh min-degree
    ordering, with a front of its own for acceptance. Every candidate of every
    run is banked in the method's archive, which gives the submission. Each
    run is hri_lns.py's procedure; restarts only let the baseline use a
    budget of weeks instead of minutes.

    Acceptance: Limmer's description gives the operators, not the acceptance
    rule. The other reading, accepting any improvement of the whole Pareto
    front (accept="front"), was measured much weaker (medium-graph, 240 CPU-s:
    -1,601,073 against -1,678,877), so the baseline keeps the stronger rule.
    Changes: evaluation, destroy and repair run in C (same logic, checked by
    tests/test_engine.py); random streams differ; the state is checkpointed.
    Start: min-degree with random tie-breaks, as benchmark.py gave it, because a
    random start is void on the medium and large families.
    """

    name = "hri"
    defaults = {"destroy_cap": 220, "small_k": 6, "stall": 3000, "iters_per_step": 64, "accept": "capped",
                "restart_after": 24000}

    def _new_run(self):
        perm, deg, bad = self.eng.greedy(self.rng, 0)
        self.current = perm
        self.bank_deg(perm, deg)
        self.run_arch = te.Archive(self.n)
        self.run_arch.add_deg(perm, deg)
        self.best = self._run_score()
        self.mode = "B"
        self.since = 0
        self.idle = 0
        self.runs += 1

    def _run_score(self):
        return self.run_arch.front_hv() if self.p["accept"] == "front" else -self.run_arch.score()

    def init(self):
        self.rng = te.Rng(self.ctx.seed * 7919 + 1)
        self.iterations = 0
        self.accepts = 0
        self.switches = 0
        self.runs = 0
        self._new_run()

    def step(self):
        p = self.p
        for _ in range(p["iters_per_step"]):
            self.iterations += 1
            if self.mode == "B":
                hole = te.hri_neighbour_destroy(self.eng, self.current, p["destroy_cap"], self.rng)
                cand = te.hri_balanced_repair(self.eng, self.current, hole, self.rng)
            else:
                cand = te.hri_random_destroy_repair(self.eng, self.current, p["small_k"], self.rng)
            imp, deg, w, t = self.bank(cand)
            self.run_arch.add_deg(cand, deg)
            self.since += 1
            self.idle += 1
            # accept on improvement of the run's capped 20-point score (or, with
            # accept="front", of the run's whole Pareto front)
            score = self._run_score()
            if score > self.best:
                self.best = score
                self.current = cand
                self.accepts += 1
                self.since = 0
                self.idle = 0
            if self.since >= p["stall"]:
                self.mode = "A" if self.mode == "B" else "B"
                self.since = 0
                self.switches += 1
            if self.idle >= p["restart_after"]:
                self._new_run()

    def get_state(self):
        return {"rng": self.rng.state, "current": self.current, "best": self.best,
                "mode": self.mode, "since": self.since, "iterations": self.iterations,
                "accepts": self.accepts, "switches": self.switches,
                "run_arch": self.run_arch.export(), "idle": self.idle, "runs": self.runs}

    def set_state(self, d):
        self.rng = te.Rng(0)
        self.rng.state = d["rng"]
        self.current = np.asarray(d["current"], dtype=np.int32)
        self.best = d["best"]
        self.mode = d["mode"]
        self.since = d["since"]
        self.iterations = d["iterations"]
        self.accepts = d["accepts"]
        self.switches = d["switches"]
        # checkpoints from before multi-start: the run so far is run 1, and its
        # front is the archive (the old acceptance)
        self.run_arch = te.Archive.restore(d["run_arch"] if "run_arch" in d else self.ctx.arch.export())
        self.idle = d.get("idle", 0)
        self.runs = d.get("runs", 1)

    def extra_status(self):
        return {"iterations": self.iterations, "accepts": self.accepts, "mode": self.mode,
                "switches": self.switches, "runs": self.runs}


# ========================================================= Hill Climbing

class HC(Method):
    """Hill Climbing (hill_climbing.solve, start="min_degree", 8 target widths).

    A min-degree probe fixes 8 target widths spread over the widths it reaches.
    Each target gets its own climb from a fresh min-degree start: one of four
    moves (swap neighbours, swap any two, move a vertex, reverse a segment of
    2-10), accepted when the smallest t within the target width strictly
    drops. Every candidate is banked.

    Changes: the original runs the 8 climbs one after another with budget/8
    each, which needs the total budget in advance. Here the 8 climbs advance in
    turn (one step of each per round), so at any budget B every climb has had
    B/8 -- the same split, available at any stopping time. Moves are evaluated
    with the windowed incremental evaluator (exact; the moves only rearrange a
    window). Random streams differ; state is checkpointed.
    Multi-start: a hill climber stops at its first local optimum (on the
    server, no change for hours on all three official graphs). So a climb
    with 200,000 tries since its last improvement starts again from a fresh
    min-degree ordering, same target. The archive keeps every climb's
    orderings, which lets the baseline use a budget of weeks.
    """

    name = "hc"
    defaults = {"widths": 8, "iters_per_step": 400, "restart_after": 200000}

    def init(self):
        self.rng = random.Random(self.ctx.seed)
        self.crng = te.Rng(self.ctx.seed * 104729 + 3)
        # the probe: one min-degree ordering banked, its staircase spans the targets
        probe, deg, _ = self.eng.greedy(te.Rng(0), 0)
        self.bank_deg(probe, deg)
        w, t = self.eng.staircase(deg)
        count = self.p["widths"]
        if len(w) == 0:
            targets = sorted({max(1, (i + 1) * te.MAX_TW // count) for i in range(count)})
        else:
            low, high = int(w.min()), int(w.max())
            if low == high:
                targets = [low]
            else:
                stepw = (high - low) / max(1, count - 1)
                targets = sorted({int(round(low + i * stepw)) for i in range(count)})
        self.targets = targets
        self.climbs = []
        for tw in targets:
            perm, deg, _ = self.eng.greedy(self.crng, 0)
            self.bank_deg(perm, deg)
            cost = self.eng.best_t_for_width(deg, tw)
            self.climbs.append({"target": tw, "perm": perm, "cost": cost, "tries": 0, "accepts": 0,
                                "stale": 0, "restarts": 0})
        self.turn = 0
        self.inc = te.Inc(self.eng, 32)
        self.inc_owner = -1

    def _move(self, perm):
        """One of the four moves; returns (new perm, lo, hi) with the window it touched."""
        n = self.n
        r = self.rng
        op = r.randrange(4)
        new = perm.copy()
        if op == 0:                                   # swap_neighbours
            i = r.randrange(n - 1)
            new[i], new[i + 1] = new[i + 1], new[i]
            return new, i, i + 1
        if op == 1:                                   # swap_any
            i, j = r.randrange(n), r.randrange(n)
            new[i], new[j] = new[j], new[i]
            return new, min(i, j), max(i, j)
        if op == 2:                                   # move_vertex
            i = r.randrange(n)
            v = new[i]
            rest = np.delete(new, i)
            j = r.randrange(n)                       # insert position in the n-1 list (0..n-1)
            new = np.insert(rest, j, v).astype(np.int32)
            return new, min(i, j), max(i, j)
        i = r.randrange(n - 1)                       # reverse_segment
        j = min(n, i + r.randint(2, 10))
        new[i:j] = new[i:j][::-1]
        return new, i, j - 1

    def step(self):
        c = self.climbs[self.turn]
        if self.inc_owner != self.turn:
            self.inc.set(c["perm"])
            self.inc_owner = self.turn
        tw = c["target"]
        for _ in range(self.p["iters_per_step"]):
            new, lo, hi = self._move(c["perm"])
            deg = self.inc.eval(new, lo, hi)
            self.bank_deg(new, deg)
            c["tries"] += 1
            cost = self.eng.best_t_for_width(deg, tw)
            if cost < c["cost"]:
                c["perm"] = new
                c["cost"] = cost
                c["accepts"] += 1
                c["stale"] = 0
                self.inc.accept(new, lo, hi)
            else:
                c["stale"] += 1
        if c["stale"] >= self.p["restart_after"]:
            # local optimum: the same target from a fresh min-degree ordering
            perm, deg, _ = self.eng.greedy(self.crng, 0)
            self.bank_deg(perm, deg)
            c["perm"] = perm
            c["cost"] = self.eng.best_t_for_width(deg, tw)
            c["stale"] = 0
            c["restarts"] += 1
            self.inc_owner = -1
        self.turn = (self.turn + 1) % len(self.climbs)

    def get_state(self):
        return {"rng": self.rng.getstate(), "crng": self.crng.state, "targets": self.targets,
                "climbs": self.climbs, "turn": self.turn}

    def set_state(self, d):
        self.rng = random.Random()
        self.rng.setstate(d["rng"])
        self.crng = te.Rng(0)
        self.crng.state = d["crng"]
        self.targets = d["targets"]
        self.climbs = d["climbs"]
        for c in self.climbs:               # checkpoints from before multi-start
            c.setdefault("stale", 0)
            c.setdefault("restarts", 0)
        self.turn = d["turn"]
        self.inc = te.Inc(self.eng, 32)
        self.inc_owner = -1

    def extra_status(self):
        return {"targets": self.targets, "costs": [c["cost"] for c in self.climbs],
                "tries": sum(c["tries"] for c in self.climbs),
                "accepts": sum(c["accepts"] for c in self.climbs),
                "restarts": sum(c.get("restarts", 0) for c in self.climbs)}


# ============================================================ fcmaes stand-in

def build_features(eng, eigenvectors=8):
    """neuroevo_cpu.build_features, vectorised (same numbers): degree profile
    (5), the first `eigenvectors` non-trivial eigenvectors of the normalised
    Laplacian, all squares and pairwise products, each column standardised.
    Returns (n, dim) float64."""
    n = eng.n
    deg = eng.deg0.astype(np.float64)
    raw = np.zeros((n, 5))
    for v in range(n):
        nb = deg[eng.neighbours(v)]
        if len(nb) == 0:
            nb = np.zeros(1)
        raw[v] = [deg[v], nb.min(), nb.max(), nb.mean(), nb.std()]
    if eigenvectors > 0:
        A = np.zeros((n, n))
        for v in range(n):
            A[v, eng.neighbours(v)] = 1.0
        d = A.sum(1)
        d[d == 0] = 1.0
        dm = 1.0 / np.sqrt(d)
        lap = np.eye(n) - (A * dm).T * dm
        vals, vecs = np.linalg.eigh(lap)
        pe = np.real(vecs[:, vals.argsort()][:, 1:eigenvectors + 1])
        raw = np.hstack([raw, pe])
    width = raw.shape[1]
    cols = [raw, raw * raw]
    prods = [raw[:, i] * raw[:, j] for i in range(width) for j in range(i + 1, width)]
    feats = np.hstack(cols + [np.stack(prods, 1)]) if prods else np.hstack(cols)
    mean = feats.mean(0)
    sd = feats.std(0)
    sd[sd == 0] = 1.0
    return (feats - mean) / sd


class SepCMAES:
    """Separable CMA-ES (Ros & Hansen 2008), as leaderboard-references/cmaes.py."""

    def __init__(self, dim, sigma0, rng, mean=None, lam=None):
        self.dim = dim
        self.rng = rng
        self.mean = rng.normal(0, 0.1, dim) if mean is None else np.array(mean, dtype=np.float64)
        self.sigma = sigma0
        self.lam = int(lam) if lam else 4 + int(3 * math.log(dim))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.cc = (1 + 1 / dim + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff) * (dim + 2) / 3
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff)
                       / ((dim + 2) ** 2 + self.mueff) * (dim + 2) / 3)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        self.C = np.ones(dim)
        self.chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.dim))
        self.y = self.z * np.sqrt(self.C)[None, :]
        return self.mean[None, :] + self.sigma * self.y

    def tell(self, fitness):
        idx = np.argsort(fitness, kind="stable")
        ysel, zsel = self.y[idx[:self.mu]], self.z[idx[:self.mu]]
        yw = (self.w[:, None] * ysel).sum(0)
        zw = (self.w[:, None] * zsel).sum(0)
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** 2) / self.chiN) < (1.4 + 2 / (self.dim + 1))
        self.pc = (1 - self.cc) * self.pc + (hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff)) * yw
        cmu_term = (self.w[:, None] * (ysel ** 2)).sum(0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * cmu_term)
        self.C = np.maximum(self.C, 1e-20)
        self.sigma *= math.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(np.clip(self.sigma, 1e-12, 1e6))

    def get_state(self):
        return {k: getattr(self, k) for k in ("mean", "sigma", "ps", "pc", "C")}

    def set_state(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class FCMAES(Method):
    """The fcmaes-style stand-in (leaderboard-references/cmaes.py, built-in
    separable CMA-ES), the column the AlbShkenca 2026 tab calls fast-cma-es.

    A candidate is a weight vector x over static per-vertex features (degree
    profile + 8 Laplacian eigenvectors + squares and pairwise products, 104
    values); the ordering is argsort(features @ x). Separable CMA-ES, sigma0
    0.5, population 4 + 3 ln(dim). Every candidate is banked.

    Changes:
      * fitness. cmaes.py scores a candidate by stairs[-1][1], which is the t of
        the threshold-0 point and therefore 0 for every feasible ordering: the
        optimiser could only tell feasible from infeasible. Here a feasible
        candidate scores -HV of its own best 20 points (the problem's score for
        that one ordering) and an infeasible one scores n - (first step over
        the cap), so later breaches rank better.
      * restarts. cmaes.py's built-in engine runs one CMA-ES to the deadline;
        its fcmaes engine restarts from a uniform(-1, 1) point whenever a run
        ends. Over weeks a single run would stall, so this restarts the same way
        when sigma has collapsed (< 1e-8 sigma0) or after 300 generations without
        improving that run's best fitness.
      * random streams differ; state is checkpointed.
    """

    name = "fcmaes"
    defaults = {"eigenvectors": 8, "sigma0": 0.5, "stall_generations": 300}

    def _features(self):
        if not hasattr(self, "F"):
            self.F = build_features(self.eng, self.p["eigenvectors"])
        return self.F

    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        F = self._features()
        self.dim = F.shape[1]
        self.es = SepCMAES(self.dim, self.p["sigma0"], self.g)
        self.generations = 0
        self.restarts = 0
        self.run_best = math.inf
        self.run_best_gen = 0

    def _fitness(self, x):
        perm = np.argsort(self.F @ x, kind="stable").astype(np.int32)
        imp, deg, w, t = self.bank(perm)
        if len(w) == 0:
            bad = int(np.argmax(deg > te.MAX_TW)) if (deg > te.MAX_TW).any() else 0
            return float(self.n - bad)
        return -float(self.eng.stair_hv(deg, 20))

    def step(self):
        F = self._features()
        pop = self.es.ask()
        vals = np.array([self._fitness(x) for x in pop])
        self.es.tell(vals)
        self.generations += 1
        b = float(vals.min())
        if b < self.run_best:
            self.run_best = b
            self.run_best_gen = self.generations
        if (self.es.sigma < 1e-8 * self.p["sigma0"]
                or self.generations - self.run_best_gen >= self.p["stall_generations"]):
            self.restarts += 1
            self.es = SepCMAES(self.dim, self.p["sigma0"], self.g, mean=self.g.uniform(-1.0, 1.0, self.dim))
            self.run_best = math.inf
            self.run_best_gen = self.generations

    def get_state(self):
        return {"g": np_rng_state(self.g), "es": self.es.get_state(), "generations": self.generations,
                "restarts": self.restarts, "run_best": self.run_best, "run_best_gen": self.run_best_gen}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        F = self._features()
        self.dim = F.shape[1]
        self.es = SepCMAES(self.dim, self.p["sigma0"], self.g)
        self.es.set_state(d["es"])
        self.generations = d["generations"]
        self.restarts = d["restarts"]
        self.run_best = d["run_best"]
        self.run_best_gen = d["run_best_gen"]

    def extra_status(self):
        return {"generations": self.generations, "restarts": self.restarts, "sigma": self.es.sigma}

"""
neuro.py -- the two neuroevolution methods. Both evolve the weights of a small
neural decoder rather than orderings.

The decoder (te_policy_decode, C) builds an ordering step by step: at every
step it scores the alive vertices with a one-hidden-layer ReLU MLP (12 hidden
units, 169 weights) over live features -- current degree, its excess over the
target width, the share of original neighbours already eliminated, the fill
gained so far, progress, the target -- and six static ones (original degree,
core number, twin-class size, big-clique size, two spectral coordinates), and
eliminates the highest-scoring vertex whose current degree is <= 500. So a
decoded ordering never breaks the cap by choice, and the policy reacts to the
fill-in it has caused, which a fixed argsort of static scores (cuda-torso, the
fcmaes stand-in) cannot. The target width is an input: one policy can behave
differently for different widths.

NE-1  ANE: one generalist policy, trained by separable CMA-ES. A candidate is
      scored by decoding it at 4 target widths drawn from the archive's 20
      selected widths (always including the endpoint, stratified over the
      rest; the same targets for a whole generation) and summing, over those
      widths w_j, strip_j * (n - t_j): the hypervolume the candidate would
      earn on the strips the score keeps. Restarts from the best policy found
      when the search stalls.

NE-2  AQD-NE: quality diversity over widths, the per-threshold-elite recipe of
      the small and large tops (cuda-torso) made adaptive. A policy archive
      keeps, for every width, the policy (and the target it was decoded with)
      that reached the smallest t there. Offspring: Gaussian mutation of half
      the weights (sigma 0.3, as cuda-torso) plus gene flow (each weight
      copied from another elite with p 0.2, CoSyNE style), decoded at their
      parent's target; and two CMA-ME improvement emitters (separable CMA-ES
      around an elite, ranked by the hypervolume their decodings add to the
      archive, restarted from another elite when they stop adding).
      Differences from cuda-torso: adaptive decoding instead of a static
      argsort, effort on the widths the score keeps instead of uniform over
      all thresholds, exact degrees (no +1), exact 20-point selection.

Both start from a policy equivalent to min-degree (one hidden unit reading
the current degree) and bank the structure-aware starting portfolio.
"""

from __future__ import annotations

import math

import numpy as np

from common import Method, np_rng_state, np_rng_from_state, staircase_t_at
from methods.baselines import SepCMAES
import seeding
import structure
import torso_engine as te

HIDDEN = 12


def min_degree_policy(ns, hidden=HIDDEN):
    """Weights whose argmax is the min-degree choice."""
    nf = ns + 6
    w = np.zeros(te.policy_nweights(ns, hidden), dtype=np.float64)
    w[0 * nf + ns + 0] = -1.0          # unit 0 reads the current degree (/100) negatively
    w[hidden * nf + 0] = 10.0          # b1[0]: stays in the linear part of the ReLU
    w[hidden * nf + hidden + 0] = 1.0  # w2[0]
    return w


class _Neuro(Method):
    def _setup(self):
        S = structure.load(self.eng)
        self.stat = np.ascontiguousarray(S["stat"][:, : 4 + self.p["n_spectral"]])
        self.ns = self.stat.shape[1]
        self.dim = te.policy_nweights(self.ns, HIDDEN)

    def _decode(self, theta, target):
        perm, deg, bad = te.policy_decode(self.eng, self.stat, theta.astype(np.float32), HIDDEN,
                                          int(target), self.p["rescore_every"])
        imp = self.bank_deg(perm, deg)
        w, t = self.eng.staircase(deg)
        return perm, deg, w, t

    def _strips(self):
        ws, ts = self.ctx.arch.selected()
        ws = [int(x) for x in ws]
        strip = {w: ((ws[j + 1] - w) if j + 1 < len(ws) else (self.n - w)) for j, w in enumerate(ws)}
        return ws, strip

    def _seed_archive(self):
        for perm, deg in seeding.portfolio(self.eng, self.crng, self.g):
            self.bank_deg(perm, deg)


class ANE(_Neuro):
    name = "ne1"
    defaults = {"sigma0": 0.3, "targets": 4, "rescore_every": 64, "stall_generations": 150, "n_spectral": 2}

    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        self.crng = te.Rng(self.ctx.seed * 7 + 99)
        self._setup()
        self._seed_archive()
        self.theta0 = min_degree_policy(self.ns)
        self.es = SepCMAES(self.dim, self.p["sigma0"], self.g, mean=self.theta0.copy())
        self.best_theta = self.theta0.copy()
        self.best_fit = math.inf
        self.gen = 0
        self.restarts = 0
        self.stall = 0

    def _targets(self):
        ws, strip = self._strips()
        if not ws:
            return [], strip
        K = self.p["targets"]
        endpoint = ws[-1]
        interior = ws[:-1]
        picks = [endpoint]
        if interior:
            parts = np.array_split(np.array(interior), max(1, K - 1))
            for part in parts:
                if len(part):
                    picks.append(int(part[self.g.integers(len(part))]))
        return picks, strip

    def step(self):
        targets, strip = self._targets()
        if not targets:
            return
        pop = self.es.ask()
        fits = np.zeros(len(pop))
        for i, theta in enumerate(pop):
            val = 0.0
            for w in targets:
                perm, deg, sw, st = self._decode(theta, w)
                tw = staircase_t_at(sw, st, w, self.n)
                val += strip.get(w, 1) * (self.n - tw)
            fits[i] = -val / (self.n * self.n)
        self.es.tell(fits)
        self.gen += 1
        b = int(np.argmin(fits))
        # the targets move with the archive, so "best" is per-generation evidence
        if fits[b] < self.best_fit - 1e-12:
            self.best_fit = float(fits[b])
            self.best_theta = pop[b].copy()
            self.stall = 0
        else:
            self.stall += 1
        if self.stall >= self.p["stall_generations"] or self.es.sigma < 1e-4:
            self.restarts += 1
            self.es = SepCMAES(self.dim, self.p["sigma0"], self.g, mean=self.best_theta.copy())
            self.stall = 0
            self.best_fit = math.inf      # re-measure against the current targets

    def get_state(self):
        return {"g": np_rng_state(self.g), "crng": self.crng.state, "es": self.es.get_state(),
                "best_theta": self.best_theta, "best_fit": self.best_fit, "gen": self.gen,
                "restarts": self.restarts, "stall": self.stall}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        self.crng = te.Rng(0)
        self.crng.state = d["crng"]
        self._setup()
        self.es = SepCMAES(self.dim, self.p["sigma0"], self.g, mean=d["best_theta"])
        self.es.set_state(d["es"])
        self.best_theta = d["best_theta"]
        self.best_fit = d["best_fit"]
        self.gen = d["gen"]
        self.restarts = d["restarts"]
        self.stall = d["stall"]

    def extra_status(self):
        return {"generations": self.gen, "restarts": self.restarts, "sigma": self.es.sigma}


class AQDNE(_Neuro):
    name = "ne2"
    defaults = {"rescore_every": 64, "mut_sigma": 0.3, "mut_prob": 0.5, "flow_prob": 0.2, "n_spectral": 2,
                "emitters": 2, "p_emit": 0.3, "offspring_per_step": 8, "emitter_patience": 20,
                "p_selected": 0.75, "span": 6}

    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        self.crng = te.Rng(self.ctx.seed * 13 + 7)
        self._setup()
        self._seed_archive()
        self.offspring = 0
        self.emitter_restarts = 0
        self.cells = {}                       # width -> [theta, target, t]
        theta0 = min_degree_policy(self.ns)
        pw, pt = self.ctx.arch.pareto()
        lo, hi = (int(pw[0]), int(pw[-1])) if len(pw) else (0, te.MAX_TW)
        for target in np.unique(np.linspace(lo, hi, 12).astype(int)):
            self._offer(theta0, int(target))
        for _ in range(16):
            th = theta0 + self.g.normal(0, 0.5, self.dim)
            self._offer(th, int(self.g.integers(lo, hi + 1)))
        self.emit = [None] * self.p["emitters"]
        self.emit_turn = 0

    def _offer(self, theta, target):
        """Decode, bank, and update the policy archive. Returns (archive HV gain, cells improved)."""
        hv0 = self.ctx.arch.hv()
        perm, deg, w, t = self._decode(theta, target)
        gain = self.ctx.arch.hv() - hv0
        improved = 0
        for wi, ti in zip(w.tolist(), t.tolist()):
            c = self.cells.get(wi)
            if c is None or ti < c[2]:
                self.cells[wi] = [theta.copy(), int(target), int(ti)]
                improved += 1
        self.offspring += 1
        return gain, improved

    def _pick_cell(self):
        keys = sorted(self.cells)
        if not keys:
            return None
        ws, _ = self.ctx.arch.selected()
        if len(ws) and self.g.random() < self.p["p_selected"]:
            w = int(ws[self.g.integers(len(ws))]) - int(self.g.integers(0, self.p["span"] + 1))
            # nearest cell at or below w, else the smallest above
            below = [k for k in keys if k <= w]
            return below[-1] if below else keys[0]
        return keys[self.g.integers(len(keys))]

    def _mutant(self, theta):
        child = theta.copy()
        mask = self.g.random(self.dim) < self.p["mut_prob"]
        child[mask] += self.g.normal(0, self.p["mut_sigma"], int(mask.sum()))
        keys = list(self.cells)
        flow = self.g.random(self.dim) < self.p["flow_prob"]
        if flow.any() and len(keys) > 1:
            donors = self.g.integers(len(keys), size=int(flow.sum()))
            idx = np.where(flow)[0]
            for j, dnr in zip(idx, donors):
                child[j] = self.cells[keys[dnr]][0][j]
        return child

    def _emitter_step(self, k):
        em = self.emit[k]
        if em is None:
            cell = self._pick_cell()
            if cell is None:
                return
            theta, target, _ = self.cells[cell]
            es = SepCMAES(self.dim, self.p["mut_sigma"], self.g, mean=theta.copy())
            em = {"es": es, "target": target, "stale": 0}
            self.emit[k] = em
        pop = em["es"].ask()
        fits = np.zeros(len(pop))
        any_gain = False
        for i, th in enumerate(pop):
            gain, improved = self._offer(th, em["target"])
            fits[i] = -(gain + 1e-3 * improved)
            any_gain |= (gain > 0 or improved > 0)
        em["es"].tell(fits)
        em["stale"] = 0 if any_gain else em["stale"] + 1
        if em["stale"] >= self.p["emitter_patience"] or em["es"].sigma < 1e-4:
            self.emit[k] = None
            self.emitter_restarts += 1

    def step(self):
        if self.g.random() < self.p["p_emit"] and self.p["emitters"] > 0:
            k = self.emit_turn
            self.emit_turn = (self.emit_turn + 1) % self.p["emitters"]
            self._emitter_step(k)
            return
        for _ in range(self.p["offspring_per_step"]):
            cell = self._pick_cell()
            if cell is None:
                return
            theta, target, _ = self.cells[cell]
            child = self._mutant(theta)
            tgt = target if self.g.random() < 0.5 else cell
            self._offer(child, tgt)

    def get_state(self):
        cells = {int(k): (v[0].astype(np.float32), int(v[1]), int(v[2])) for k, v in self.cells.items()}
        emit = []
        for em in self.emit:
            emit.append(None if em is None else {"es": em["es"].get_state(), "target": em["target"],
                                                  "stale": em["stale"]})
        return {"g": np_rng_state(self.g), "crng": self.crng.state, "cells": cells, "emit": emit,
                "emit_turn": self.emit_turn, "offspring": self.offspring,
                "emitter_restarts": self.emitter_restarts}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        self.crng = te.Rng(0)
        self.crng.state = d["crng"]
        self._setup()
        self.cells = {int(k): [np.asarray(v[0], dtype=np.float64), int(v[1]), int(v[2])]
                      for k, v in d["cells"].items()}
        self.emit = []
        for em in d["emit"]:
            if em is None:
                self.emit.append(None)
            else:
                es = SepCMAES(self.dim, self.p["mut_sigma"], self.g)
                es.set_state(em["es"])
                self.emit.append({"es": es, "target": em["target"], "stale": em["stale"]})
        self.emit_turn = d["emit_turn"]
        self.offspring = d["offspring"]
        self.emitter_restarts = d["emitter_restarts"]

    def extra_status(self):
        return {"cells": len(self.cells), "offspring": self.offspring,
                "emitter_restarts": self.emitter_restarts}

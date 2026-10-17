"""
nsga2.py -- EA-3: NSGA-II (Deb et al. 2002) on the ESA encoding itself.

An individual is a decision vector as ESA defines it: an elimination ordering
perm plus a threshold t. Its objectives are the official fitness
(width(t), t), both minimised; the 500 cap is a constraint handled by Deb's
constrained domination with violation CV = sum_i max(0, d_i - 500):

    feasible beats infeasible; two infeasible: smaller CV wins;
    two feasible: Pareto dominance.

Textbook configuration (pymoo's operators for permutation problems):
    population 100, 100 offspring per generation, (mu + lambda) survival by
    non-dominated rank then crowding distance;
    binary tournament on (rank, crowding);
    ordering: order crossover (pymoo ox, shift=False) with p = 0.9 and
              inversion mutation (reverse a uniformly random segment);
    threshold: SBX (eta 15, p 0.9) and polynomial mutation (eta 20), rounded
               and clipped to [0, n-1].

Shared with every method: the start (min-degree constructions with random
tie-breaks), the engine, and the archive -- each evaluated ordering's whole
staircase is banked, and the submission is the archive's exact best 20.
"""

from __future__ import annotations

import numpy as np

from common import Method, np_rng_state, np_rng_from_state
import torso_engine as te


def ox(receiver, donor, start, end):
    """pymoo's order crossover (shift=False): donor[start:end+1] stays in place,
    the other positions take receiver's remaining genes in receiver order."""
    donation = donor[start:end + 1]
    keep = np.isin(receiver, donation, invert=True)
    rest = receiver[keep]
    return np.concatenate([rest[:start], donation, rest[start:]]).astype(np.int32)


def sbx_int(a, b, lo, hi, eta, rng):
    """SBX on one bounded variable (Deb & Agrawal), children rounded."""
    if abs(a - b) < 1e-12:
        return float(a), float(b)
    y1, y2 = min(a, b), max(a, b)
    u = rng.random()
    beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
    alpha = 2.0 - beta ** -(eta + 1)
    bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
    c1 = 0.5 * ((y1 + y2) - bq * (y2 - y1))
    beta = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
    alpha = 2.0 - beta ** -(eta + 1)
    bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
    c2 = 0.5 * ((y1 + y2) + bq * (y2 - y1))
    c1, c2 = min(max(c1, lo), hi), min(max(c2, lo), hi)
    if rng.random() < 0.5:
        c1, c2 = c2, c1
    return c1, c2


def pm_int(x, lo, hi, eta, rng):
    """Polynomial mutation on one bounded variable."""
    if hi <= lo:
        return x
    d1, d2 = (x - lo) / (hi - lo), (hi - x) / (hi - lo)
    u = rng.random()
    mp = 1.0 / (eta + 1)
    if u < 0.5:
        xy = 1.0 - d1
        val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1)
        dq = val ** mp - 1.0
    else:
        xy = 1.0 - d2
        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1)
        dq = 1.0 - val ** mp
    return min(max(x + dq * (hi - lo), lo), hi)


def constrained_sort(F, CV):
    """Fast non-dominated sorting under constrained domination.
    F: (N, 2) objectives (min), CV: (N,) violation. Returns rank (N,)."""
    N = len(F)
    feas = CV <= 0
    # dom[i, j] = i constrained-dominates j
    le = (F[:, None, :] <= F[None, :, :]).all(-1)
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    pareto = le & lt
    both_feas = feas[:, None] & feas[None, :]
    dom = (feas[:, None] & ~feas[None, :]) | (~feas[:, None] & ~feas[None, :] & (CV[:, None] < CV[None, :])) \
        | (both_feas & pareto)
    count = dom.sum(0)                       # how many dominate j
    rank = np.full(N, -1, dtype=np.int64)
    current = np.where(count == 0)[0]
    r = 0
    remaining = count.copy()
    while len(current):
        rank[current] = r
        dec = dom[current].sum(0)
        remaining = remaining - dec
        remaining[rank >= 0] = 1 << 30
        current = np.where(remaining == 0)[0]
        r += 1
    return rank


def crowding(F, idx):
    """Crowding distance of the individuals idx (one front)."""
    m = len(idx)
    d = np.zeros(m)
    if m <= 2:
        d[:] = np.inf
        return d
    for k in range(F.shape[1]):
        order = np.argsort(F[idx, k], kind="stable")
        fk = F[idx[order], k]
        span = fk[-1] - fk[0]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (fk[2:] - fk[:-2]) / span
    return d


class NSGA2(Method):
    name = "nsga2"
    defaults = {"pop": 100, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0, "pm_t": 1.0}

    def _evaluate(self, perm, t):
        deg, bad = self.eng.eval(perm, abort_cap=False)
        self.bank_deg(perm, deg)          # banks nothing if the cap is broken
        cv = float(np.maximum(deg - te.MAX_TW, 0).sum())
        width = int(deg[t:].max()) if t < self.n else 0
        return width, cv

    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        crng = te.Rng(self.ctx.seed * 31 + 5)
        N = self.p["pop"]
        self.P, self.T, self.F, self.CV = [], [], [], []
        for i in range(N):
            perm, deg, bad = self.eng.greedy(crng, i % 4)
            self.bank_deg(perm, deg)
            w, ts = self.eng.staircase(deg)
            t = int(ts[self.g.integers(len(ts))]) if len(ts) else int(self.g.integers(self.n))
            cv = float(np.maximum(deg - te.MAX_TW, 0).sum())
            self.P.append(perm)
            self.T.append(t)
            self.F.append((int(deg[t:].max()), t))
            self.CV.append(cv)
        self.generation = 0
        self._rank_crowd()

    def _rank_crowd(self):
        F = np.array(self.F, dtype=float)
        CV = np.array(self.CV, dtype=float)
        self.rank = constrained_sort(F, CV)
        self.crowd = np.zeros(len(F))
        for r in np.unique(self.rank):
            idx = np.where(self.rank == r)[0]
            self.crowd[idx] = crowding(F, idx)

    def _tournament(self):
        N = len(self.P)
        a, b = self.g.integers(N), self.g.integers(N)
        if self.rank[a] != self.rank[b]:
            return a if self.rank[a] < self.rank[b] else b
        if self.crowd[a] != self.crowd[b]:
            return a if self.crowd[a] > self.crowd[b] else b
        return a if self.g.random() < 0.5 else b

    def _inversion(self, perm):
        i, j = np.sort(self.g.choice(self.n, 2, replace=False))
        out = perm.copy()
        out[i:j + 1] = out[i:j + 1][::-1]
        return out

    def step(self):
        p = self.p
        N = p["pop"]
        n = self.n
        children, ct = [], []
        while len(children) < N:
            a, b = self._tournament(), self._tournament()
            pa, pb = self.P[a], self.P[b]
            if self.g.random() < p["pc"]:
                s, e = np.sort(self.g.choice(n, 2, replace=False))
                c1, c2 = ox(pa, pb, s, e), ox(pb, pa, s, e)
                t1, t2 = sbx_int(float(self.T[a]), float(self.T[b]), 0.0, float(n - 1), p["eta_c"], self.g)
            else:
                c1, c2 = pa.copy(), pb.copy()
                t1, t2 = float(self.T[a]), float(self.T[b])
            for c, tt in ((c1, t1), (c2, t2)):
                if len(children) >= N:
                    break
                c = self._inversion(c)
                if self.g.random() < p["pm_t"]:
                    tt = pm_int(tt, 0.0, float(n - 1), p["eta_m"], self.g)
                children.append(c)
                ct.append(int(round(tt)))
        for c, tt in zip(children, ct):
            w, cv = self._evaluate(c, tt)
            self.P.append(c); self.T.append(tt); self.F.append((w, tt)); self.CV.append(cv)
        # (mu + lambda) survival
        F = np.array(self.F, dtype=float)
        CV = np.array(self.CV, dtype=float)
        rank = constrained_sort(F, CV)
        chosen = []
        for r in range(int(rank.max()) + 1):
            idx = np.where(rank == r)[0]
            if len(chosen) + len(idx) <= N:
                chosen.extend(idx.tolist())
            else:
                cd = crowding(F, idx)
                order = np.argsort(-cd, kind="stable")
                chosen.extend(idx[order[:N - len(chosen)]].tolist())
                break
        self.P = [self.P[i] for i in chosen]
        self.T = [self.T[i] for i in chosen]
        self.F = [self.F[i] for i in chosen]
        self.CV = [self.CV[i] for i in chosen]
        self._rank_crowd()
        self.generation += 1

    def get_state(self):
        return {"g": np_rng_state(self.g), "P": np.stack(self.P), "T": list(self.T),
                "F": list(self.F), "CV": list(self.CV), "generation": self.generation}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        self.P = [np.asarray(x, dtype=np.int32) for x in d["P"]]
        self.T = list(d["T"]); self.F = [tuple(x) for x in d["F"]]; self.CV = list(d["CV"])
        self.generation = d["generation"]
        self._rank_crowd()

    def extra_status(self):
        feas = sum(1 for c in self.CV if c <= 0)
        return {"generation": self.generation, "feasible": feas,
                "rank0": int((self.rank == 0).sum()) if hasattr(self, "rank") else 0}

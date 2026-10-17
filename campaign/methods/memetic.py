"""
memetic.py -- EA-1, MEMETIC-PERM: a memetic island GA on elimination orderings.

Population search on orderings, built around the operator that wins medium
(Team HRI's neighbour destroy + median-placement repair) and adding what a
single LNS trajectory lacks: a population, crossover, and local search aimed
at the step that blocks each threshold.

Islands (steady state, 12 orderings each):
    E    the endpoint: minimise the threshold-0 width W, then the number of
         steps at W, then the sum of the top degrees
    F1   the low third of the archive's currently selected widths
    F2   the middle third
    F3   the high third (the endpoint itself belongs to E)
         An F island maximises sum_j s_j (n - t_j(perm)) over its widths w_j,
         with s_j the strip each width owns in the archive's 20-point front:
         the hypervolume that ordering alone would earn on those strips.

One child per step, islands in turn:
    1. two parents by binary tournament on the island fitness
    2. crossover (p 0.7): order crossover or rank-average crossover
    3. mutation, one of: guided greedy completion (p 0.2: keep a random
       prefix, rebuild the rest by min-degree with a frontier preference,
       steered by the child's own order), HRI neighbour destroy + balanced
       repair (p 0.25), twin-class consolidation (p 0.1), a shuffled window of
       3-40 positions (p 0.25), or none; on the endpoint island the child then
       gets an iterated-greedy burst (8 completions from random cuts, the best
       kept)
    4. targeted local search (te_ls_target) at one of the island's widths --
       E: W-1; F: a selected width of its band -- with incremental evaluation
    5. the child replaces the island's worst if it is better and not a copy
Every 25 children an island sends its best to the next island; every 50 it
takes in the shared archive's elite for one of its widths. An island whose
best has not improved for 2,000 children re-seeds its worse half. The islands
start from the structure-aware portfolio (seeding.py).
Every ordering evaluated anywhere is banked in the shared archive.
"""

from __future__ import annotations

import numpy as np

from common import Method, np_rng_state, np_rng_from_state, staircase_t_at
import seeding
import structure
import torso_engine as te


def rank_average(pa, pb, g):
    n = len(pa)
    ra = np.empty(n); ra[pa] = np.arange(n)
    rb = np.empty(n); rb[pb] = np.arange(n)
    key = (ra + rb) * 0.5 + g.uniform(-0.5, 0.5, n)
    return np.argsort(key, kind="stable").astype(np.int32)


def order_crossover(pa, pb, g):
    n = len(pa)
    s, e = np.sort(g.choice(n, 2, replace=False))
    donation = pb[s:e + 1]
    rest = pa[np.isin(pa, donation, invert=True)]
    return np.concatenate([rest[:s], donation, rest[s:]]).astype(np.int32)


class MemeticPerm(Method):
    name = "ea1"
    defaults = {"islands": 4, "island_size": 12, "p_cross": 0.7, "ls_iters": 400,
                "ls_window": 24, "migrate_every": 25, "inject_every": 50, "stall_children": 2000, "ig_burst": 8,
                "destroy_cap": 220}

    # ------------------------------------------------------------ helpers
    def _struct(self):
        if not hasattr(self, "S"):
            self.S = structure.load(self.eng)
            tw = self.S["twin"]
            self.twin_groups = [g for g in structure.groups(tw) if len(g) > 1]
        return self.S

    def _new(self, slack):
        perm, deg, _ = self.eng.greedy(self.crng, slack)
        self.bank_deg(perm, deg)
        return self._ind(perm, deg)

    def _ind(self, perm, deg):
        w, t = self.eng.staircase(deg)
        return {"perm": perm, "deg": deg, "w": w, "t": t}

    def _bands(self):
        ws, ts = self.ctx.arch.selected()
        if len(ws) == 0:
            return [[], [], []], {}
        ws = [int(x) for x in ws]
        strip = {}
        for j, w in enumerate(ws):
            strip[w] = (ws[j + 1] - w) if j + 1 < len(ws) else (self.n - w)
        interior = ws[:-1]
        k = len(interior)
        bands = [interior[: k // 3], interior[k // 3: 2 * k // 3], interior[2 * k // 3:]]
        bands = [b if b else interior[-1:] for b in bands]
        return bands, strip

    def _fit(self, island, ind):
        if len(ind["w"]) == 0:                      # breaks the cap
            return (10 ** 9,)
        if island == 0:
            deg = ind["deg"]
            W = int(deg.max())
            top = np.sort(deg)[-20:]
            return (W, int((deg == W).sum()), int(top.sum()))
        band = self.bands[island - 1]
        if not band:
            return (0,)
        val = 0
        for w in band:
            tw = staircase_t_at(ind["w"], ind["t"], w, self.n)
            val += self.strip.get(w, 1) * (self.n - tw)
        return (-val,)

    def _refresh(self):
        sel = tuple(self.ctx.arch.selected()[0].tolist())
        if sel != getattr(self, "_sel", None):
            self._sel = sel
            self.bands, self.strip = self._bands()
            for k in range(1, len(self.pop)):
                for ind in self.pop[k]:
                    ind["fit"] = self._fit(k, ind)
                if hasattr(self, "best_fit"):
                    # the band moved: the island's record restarts on the new scale
                    self.best_fit[k] = min(i["fit"] for i in self.pop[k])

    def _tournament(self, k):
        isl = self.pop[k]
        a, b = self.g.integers(len(isl)), self.g.integers(len(isl))
        return isl[a] if isl[a]["fit"] <= isl[b]["fit"] else isl[b]

    # -------------------------------------------------------------- method
    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        self.crng = te.Rng(self.ctx.seed * 2654435761 + 17)
        self._struct()
        seeds = seeding.portfolio(self.eng, self.crng, self.g)
        for perm, deg in seeds:
            self.bank_deg(perm, deg)
        order = self.g.permutation(len(seeds))
        seeds = [seeds[i] for i in order]
        self.pop = []
        it = iter(seeds)
        for k in range(self.p["islands"]):
            isl = []
            for i in range(self.p["island_size"]):
                nxt = next(it, None)
                isl.append(self._ind(*nxt) if nxt is not None else self._new(i % 4))
            self.pop.append(isl)
        self.bands, self.strip = self._bands()
        self._sel = None
        for k in range(len(self.pop)):
            for ind in self.pop[k]:
                ind["fit"] = self._fit(k, ind)
        self.turn = 0
        self.children = [0] * len(self.pop)
        self.best_fit = [min(i["fit"] for i in isl) for isl in self.pop]
        self.last_gain = [0] * len(self.pop)
        self.reseeds = 0
        self.inc = te.Inc(self.eng, 32)

    def _complete(self, perm, cut):
        """Keep perm[:cut], rebuild the rest greedily, steered by perm's own order."""
        n = self.n
        rank = np.empty(n, dtype=np.float32)
        rank[perm] = np.arange(n, dtype=np.float32)
        prio = 10.0 * (1.0 - rank / n)
        lam = float(self.g.choice([0.25, 0.5, 1.0, 2.0, 4.0, 8.0]))
        alpha = float(self.g.uniform(0.0, 3.0))
        new, deg, bad = te.greedy_complete(self.eng, perm[:cut], prio, lam, alpha, self.crng)
        self.bank_deg(new, deg)
        return new, deg, bad

    def _ig_burst(self, perm, k):
        """Iterated greedy on the endpoint: k completions from random cuts, best kept."""
        best, bdeg, _ = perm, None, None
        deg0, _ = self.eng.eval(perm, abort_cap=False)
        bkey = self._fit(0, self._ind(perm, deg0))
        for _ in range(k):
            cut = int(self.g.integers(0, self.n))
            new, deg, bad = self._complete(best, cut)
            if bad >= 0:
                continue
            key = self._fit(0, self._ind(new, deg))
            if key <= bkey:
                best, bkey = new, key
        return best

    def _mutate(self, perm):
        r = self.g.random()
        if r < 0.2:
            new, deg, bad = self._complete(perm, int(self.g.integers(0, self.n)))
            return new if bad < 0 else perm
        if r < 0.45:
            hole = te.hri_neighbour_destroy(self.eng, perm, self.p["destroy_cap"], self.crng)
            return te.hri_balanced_repair(self.eng, perm, hole, self.crng)
        if r < 0.55 and self.twin_groups:
            # twin-class consolidation: the class's members follow its first one
            grp = self.twin_groups[self.g.integers(len(self.twin_groups))]
            pos = np.empty(self.n, dtype=np.int64)
            pos[perm] = np.arange(self.n)
            first = int(pos[grp].min())
            rest = perm[np.isin(perm, grp, invert=True)]
            return np.concatenate([rest[:first], grp[np.argsort(pos[grp])], rest[first:]]).astype(np.int32)
        if r < 0.80:
            n = self.n
            L = int(self.g.integers(3, 41))
            i = int(self.g.integers(0, max(1, n - L)))
            out = perm.copy()
            seg = out[i:i + L].copy()
            self.g.shuffle(seg)
            out[i:i + L] = seg
            return out
        return perm

    def step(self):
        self._refresh()
        k = self.turn
        self.turn = (self.turn + 1) % len(self.pop)
        pa = self._tournament(k)
        pb = self._tournament(k)
        if self.g.random() < self.p["p_cross"] and pa is not pb:
            child = order_crossover(pa["perm"], pb["perm"], self.g) if self.g.random() < 0.5 \
                else rank_average(pa["perm"], pb["perm"], self.g)
        else:
            child = pa["perm"].copy()
        child = self._mutate(child)
        if k == 0:
            child = self._ig_burst(child, self.p["ig_burst"])
        bad = self.inc.set(child)
        perm0, deg0 = self.inc.get()
        self.bank_deg(perm0, deg0)
        if bad < 0:
            if k == 0:
                target = int(deg0.max()) - 1
            else:
                band = self.bands[k - 1]
                target = int(band[self.g.integers(len(band))]) if band else int(deg0.max()) - 1
            if target >= 0:
                te.ls_target(self.eng, self.inc, self.ctx.arch, target, self.p["ls_iters"], self.crng,
                             window=self.p["ls_window"])
                self.ctx.evals += self.p["ls_iters"]
        perm, deg = self.inc.get()
        ind = self._ind(perm, deg)
        ind["fit"] = self._fit(k, ind)
        isl = self.pop[k]
        worst = max(range(len(isl)), key=lambda i: isl[i]["fit"])
        dup = any(ind["fit"] == x["fit"] and np.array_equal(ind["w"], x["w"]) for x in isl)
        if ind["fit"] < isl[worst]["fit"] and not dup:
            isl[worst] = ind
        self.children[k] += 1
        bf = min(i["fit"] for i in isl)
        if bf < self.best_fit[k]:
            self.best_fit[k] = bf
            self.last_gain[k] = self.children[k]
        if self.children[k] % self.p["inject_every"] == 0:
            # archive injection: the shared archive's elite for one of the
            # island's widths (E: the endpoint's) replaces the island's worst
            ws, _ = self.ctx.arch.selected()
            if len(ws):
                band = self.bands[k - 1] if k > 0 else [int(ws[-1])]
                wsel = int(band[self.g.integers(len(band))]) if band else int(ws[-1])
                ep = self.ctx.arch.get_perm(wsel)
                if ep is not None and not any(np.array_equal(ep, x["perm"]) for x in isl):
                    ed, _ = self.eng.eval(ep, abort_cap=False)
                    ind2 = self._ind(ep, ed)
                    ind2["fit"] = self._fit(k, ind2)
                    w3 = max(range(len(isl)), key=lambda i: isl[i]["fit"])
                    isl[w3] = ind2
        if self.children[k] % self.p["migrate_every"] == 0:
            dst = (k + 1) % len(self.pop)
            best = min(isl, key=lambda i: i["fit"])
            mig = dict(best)
            mig["fit"] = self._fit(dst, mig)
            disl = self.pop[dst]
            w2 = max(range(len(disl)), key=lambda i: disl[i]["fit"])
            if not any(np.array_equal(mig["perm"], x["perm"]) for x in disl):
                disl[w2] = mig
        if self.children[k] - self.last_gain[k] >= self.p["stall_children"]:
            order = sorted(range(len(isl)), key=lambda i: isl[i]["fit"])
            for i in order[len(isl) // 2:]:
                ind2 = self._new(int(self.g.integers(0, 4)))
                ind2["fit"] = self._fit(k, ind2)
                isl[i] = ind2
            self.last_gain[k] = self.children[k]
            self.best_fit[k] = min(i["fit"] for i in isl)
            self.reseeds += 1

    def get_state(self):
        return {"g": np_rng_state(self.g), "crng": self.crng.state,
                "pop": [[{"perm": i["perm"], "deg": i["deg"]} for i in isl] for isl in self.pop],
                "turn": self.turn, "children": self.children, "best_fit": self.best_fit,
                "last_gain": self.last_gain, "reseeds": self.reseeds}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        self.crng = te.Rng(0)
        self.crng.state = d["crng"]
        self._struct()
        self.pop = [[self._ind(np.asarray(i["perm"], dtype=np.int32), np.asarray(i["deg"], dtype=np.int32))
                     for i in isl] for isl in d["pop"]]
        self.bands, self.strip = self._bands()
        self._sel = None
        for k in range(len(self.pop)):
            for ind in self.pop[k]:
                ind["fit"] = self._fit(k, ind)
        self.turn = d["turn"]
        self.children = d["children"]
        self.best_fit = [tuple(x) for x in d["best_fit"]]
        self.last_gain = d["last_gain"]
        self.reseeds = d["reseeds"]
        self.inc = te.Inc(self.eng, 32)

    def extra_status(self):
        return {"children": self.children, "reseeds": self.reseeds,
                "endpoint_best": [int(x) for x in self.best_fit[0]] if self.best_fit else None}

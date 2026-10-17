"""
torso_qd.py -- EA-2, TORSO-QD: a width-indexed quality-diversity EA over head sets.

The torso left after eliminating a head set H does not depend on the order in
which H is eliminated, so the search works on H directly. The archive is the
QD map: one cell per width, holding the smallest head (threshold t) found
whose torso fits in that width, and the ordering behind it. It is the shared
archive itself, so every ordering any operator evaluates can become an elite.

One child per iteration:
  1. target width w*: with p 0.7 one of the archive's 20 selected widths minus
     a small geometric offset (the cells the score uses and those that could
     enter it); otherwise uniform over the archive's width range
  2. parent: the elite with the smallest t among cells of width <= w*
     (p 0.75), or an elite of a cell up to 6 wider (p 0.25)
  3. one head-set move on H = parent[:t]:
       evict     1 or 2 head vertices move to the torso, each at the median
                 position of its torso neighbours (t' = t - k)
       swap      evict one, and move the torso's blocking vertex (the highest
                 step degree in the torso) to the end of the head (t' = t)
       block     evict a whole twin class or attached block lying in the head
       quota     big cliques only: evict a vertex of one clique, admit a vertex
                 of another (t' = t)
       cross     head = intersection of the heads of two elites of nearby
                 widths plus a random part of their difference
     endpoint  the threshold-0 cell has no head: the ordering is rebuilt from
                 a random cut (iterated greedy, steered by the parent's order)
  4. with p 0.3 the torso is rebuilt greedily from the new head (min-degree
     with a frontier preference, steered by the inherited torso order)
  5. memetic repair: targeted local search at w* on the child (te_ls_target),
     moves around the step that blocks the threshold, evaluated incrementally
Every evaluated ordering is banked. Seeds: the structure-aware portfolio
(min-degree orderings and clique-quota constructions).
"""

from __future__ import annotations

import numpy as np

from common import Method, np_rng_state, np_rng_from_state, staircase_t_at
import seeding
import structure
import torso_engine as te


class TorsoQD(Method):
    name = "ea2"
    defaults = {"ls_iters": 200, "ls_window": 24, "p_selected": 0.7, "span": 6,
                "children_per_step": 8, "p_regrow": 0.3}

    def _struct(self):
        if not hasattr(self, "S"):
            S = structure.load(self.eng)
            self.S = S
            self.blocks = [g for g in structure.groups(S["twin"]) if len(g) > 1]
            self.blocks += [g for g in structure.groups(S["block"]) if 1 < len(g) <= 200]
            self.cliques = structure.groups(S["clique"])
            self.clique_of = S["clique"]

    def init(self):
        self.g = np.random.default_rng(self.ctx.seed)
        self.crng = te.Rng(self.ctx.seed * 40503 + 11)
        self._struct()
        for perm, deg in seeding.portfolio(self.eng, self.crng, self.g):
            self.bank_deg(perm, deg)
        self.inc = te.Inc(self.eng, 32)
        self.children = 0
        self.moves = {k: [0, 0] for k in ("evict", "swap", "block", "quota", "cross", "endpoint")}

    # ------------------------------------------------------------ selection
    def _target(self):
        arch = self.ctx.arch
        ws, ts = arch.selected()
        pw, pt = arch.pareto()
        if len(pw) == 0:
            return None
        if self.g.random() < self.p["p_selected"] and len(ws):
            w = int(ws[self.g.integers(len(ws))])
            off = 0
            while off < self.p["span"] and self.g.random() < 0.5:
                off += 1
            w = max(int(pw[0]), w - off)
        else:
            w = int(self.g.integers(int(pw[0]), int(pw[-1]) + 1))
        return w

    def _parent(self, w):
        arch = self.ctx.arch
        best = arch.best_t_all()
        w = int(min(max(w, 0), len(best) - 1))
        if self.g.random() < 0.75:
            cand = best[: w + 1]
            wp = int(np.argmin(cand))
            if cand[wp] >= self.n:
                return None
        else:
            hi = min(len(best) - 1, w + self.p["span"])
            choices = [x for x in range(w, hi + 1) if best[x] < self.n]
            if not choices:
                cand = best[: w + 1]
                wp = int(np.argmin(cand))
                if cand[wp] >= self.n:
                    return None
            else:
                wp = int(choices[self.g.integers(len(choices))])
        perm = arch.get_perm(wp)
        return perm

    # ---------------------------------------------------------------- moves
    def _median_insert(self, torso, v):
        """Insert v into the torso list at the median of its torso neighbours."""
        nb = self.eng.neighbours(v)
        pos = self._pos_in(torso)
        p = pos[nb]
        p = np.sort(p[p >= 0])
        j = int(p[len(p) // 2]) if len(p) else len(torso)
        return np.insert(torso, j, v).astype(np.int32)

    def _pos_in(self, arr):
        pos = -np.ones(self.n, dtype=np.int64)
        pos[arr] = np.arange(len(arr))
        return pos

    def _pick_evict(self, head, torso):
        if len(head) == 0:
            return None
        k = min(8, len(head))
        cand = head[self.g.choice(len(head), k, replace=False)]
        tpos = self._pos_in(torso)
        cost = np.array([int((tpos[self.eng.neighbours(v)] >= 0).sum()) for v in cand])
        cost = cost + self.g.random(k) * 0.5
        return int(cand[int(np.argmin(cost))])

    def _child(self, perm, t, w):
        head, torso = perm[:t].copy(), perm[t:].copy()
        r = self.g.random()
        kind = "evict"
        if r < 0.45:
            kk = 1 if self.g.random() < 0.7 else 2
            for _ in range(kk):
                v = self._pick_evict(head, torso)
                if v is None:
                    break
                head = head[head != v]
                torso = self._median_insert(torso, v)
        elif r < 0.65:
            kind = "swap"
            v = self._pick_evict(head, torso)
            if v is not None:
                head = head[head != v]
                torso = self._median_insert(torso, v)
            # the torso's blocking vertex joins the head
            deg = self._deg_of(np.concatenate([head, torso]))
            if len(torso):
                i = int(np.argmax(deg[len(head):]))
                x = torso[i]
                torso = np.delete(torso, i)
                head = np.append(head, x).astype(np.int32)
        elif r < 0.75 and self.blocks:
            kind = "block"
            hset = np.zeros(self.n, dtype=bool)
            hset[head] = True
            inside = [b for b in self.blocks[:: max(1, len(self.blocks) // 64)] if hset[b].all()]
            if not inside:
                inside = [b for b in self.blocks if hset[b].all()]
            if inside:
                b = inside[self.g.integers(len(inside))]
                head = head[~np.isin(head, b)]
                for v in b:
                    torso = self._median_insert(torso, int(v))
        elif r < 0.85 and len(self.cliques) >= 2:
            kind = "quota"
            ca, cb = self.g.choice(len(self.cliques), 2, replace=False)
            hset = np.zeros(self.n, dtype=bool)
            hset[head] = True
            A = self.cliques[ca][hset[self.cliques[ca]]]
            B = self.cliques[cb][~hset[self.cliques[cb]]]
            if len(A) and len(B):
                v = int(A[self.g.integers(len(A))])
                x = int(B[self.g.integers(len(B))])
                head = head[head != v]
                torso = self._median_insert(torso[torso != x], v)
                head = np.append(head, x).astype(np.int32)
        else:
            kind = "cross"
            other = self._parent(min(te.MAX_TW, w + int(self.g.integers(-3, 4))))
            if other is not None:
                ot = self.eng.best_t_for_width(self._deg_of(other), w)
                hb = set(other[:ot].tolist())
                ha = head.tolist()
                inter = [v for v in ha if v in hb]
                diff = [v for v in ha if v not in hb] + [v for v in other[:ot].tolist() if v not in set(ha)]
                keep = int(self.g.integers(0, len(diff) + 1)) if diff else 0
                extra = [diff[i] for i in self.g.choice(len(diff), keep, replace=False)] if keep else []
                new_head = np.array(inter + extra, dtype=np.int32)
                hpos = self._pos_in(perm)
                new_head = new_head[np.argsort(hpos[new_head], kind="stable")]
                rest = perm[~np.isin(perm, new_head)]
                head, torso = new_head, rest
        child = np.concatenate([head, torso]).astype(np.int32)
        if self.g.random() < self.p["p_regrow"] and len(head) < self.n:
            # rebuild the torso greedily (min-degree with a frontier preference),
            # steered by the inherited torso order; the head set is kept
            rank = np.empty(self.n, dtype=np.float32)
            rank[child] = np.arange(self.n, dtype=np.float32)
            prio = 10.0 * (1.0 - rank / self.n)
            lam = float(self.g.choice([0.5, 1.0, 2.0, 4.0, 8.0]))
            alpha = float(self.g.uniform(0.0, 3.0))
            new, deg, bad = te.greedy_complete(self.eng, head, prio, lam, alpha, self.crng)
            self.bank_deg(new, deg)
            if bad < 0:
                child = new
                kind = kind + "+regrow"
        return child, kind

    def _deg_of(self, perm):
        deg, _ = self.eng.eval(perm, abort_cap=False)
        return deg

    # ----------------------------------------------------------------- step
    def step(self):
        for _ in range(self.p["children_per_step"]):
            w = self._target()
            if w is None:
                return
            parent = self._parent(w)
            if parent is None:
                continue
            pdeg = self._deg_of(parent)
            t = self.eng.best_t_for_width(pdeg, w)
            if t >= self.n:
                t = self.n - 1
            if t == 0:
                # the endpoint cell has an empty head: rebuild from a random cut
                # instead (iterated greedy), steered by the parent's order
                cut = int(self.g.integers(0, self.n))
                rank = np.empty(self.n, dtype=np.float32)
                rank[parent] = np.arange(self.n, dtype=np.float32)
                child, deg, bad = te.greedy_complete(self.eng, parent[:cut], 10.0 * (1.0 - rank / self.n),
                                                     float(self.g.choice([0.5, 1.0, 2.0, 4.0, 8.0])),
                                                     float(self.g.uniform(0.0, 3.0)), self.crng)
                self.bank_deg(child, deg)
                kind = "endpoint"
                if bad >= 0:
                    continue
            else:
                child, kind = self._child(parent, t, w)
            before = self.ctx.arch.best_t(w)
            bad = self.inc.set(child)
            p0, d0 = self.inc.get()
            self.bank_deg(p0, d0)
            if bad < 0:
                te.ls_target(self.eng, self.inc, self.ctx.arch, w, self.p["ls_iters"], self.crng,
                             window=self.p["ls_window"])
                self.ctx.evals += self.p["ls_iters"]
            self.children += 1
            st = self.moves.setdefault(kind, [0, 0])
            st[0] += 1
            if self.ctx.arch.best_t(w) < before:
                st[1] += 1

    def get_state(self):
        return {"g": np_rng_state(self.g), "crng": self.crng.state, "children": self.children,
                "moves": self.moves}

    def set_state(self, d):
        self.g = np_rng_from_state(d["g"])
        self.crng = te.Rng(0)
        self.crng.state = d["crng"]
        self._struct()
        self.children = d["children"]
        self.moves = d["moves"]
        self.inc = te.Inc(self.eng, 32)

    def extra_status(self):
        return {"children": self.children, "moves": self.moves}

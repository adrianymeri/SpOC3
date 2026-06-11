#!/usr/bin/env python3
"""
np_gbdt.py -- pure-numpy histogram gradient-boosted regression trees.

A dependency-free GBDT backend so every GBDT-central method in this repo
(gbdt_torso construct mode, GAPS' decode column, GBFC's weak learner, GBFC++'s
move-proposal policy) runs on any machine with numpy alone -- no LightGBM /
XGBoost / scikit-learn wheel required.  On the reference workstation LightGBM
remains the default; this backend slots into `make_gbdt`'s auto chain just
before the linear ridge fallback, so the *boosted-tree* character of the
methods is preserved even in minimal environments.

Design: classic second-order-free gradient boosting on the squared loss
(residual fitting), histogram splits (quantile bins, default 32), depth-limited
trees grown breadth-first, sample weights supported.  Vectorised over features
via flat bincounts; fitting ~20k rows x 40 features x 200 trees takes a few
seconds.

    from algorithms.continuous.np_gbdt import NpGBDT
    model = NpGBDT(n_estimators=200, learning_rate=0.1, max_depth=4, seed=0)
    model.fit(X, y, sample_weight=w); yhat = model.predict(X)
"""
from __future__ import annotations

import numpy as np


class NpGBDT:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=4,
                 n_bins=32, min_samples_leaf=8, subsample=0.8, seed=0):
        self.n_estimators = int(n_estimators)
        self.lr = float(learning_rate)
        self.max_depth = int(max_depth)
        self.n_bins = int(n_bins)
        self.min_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.seed = int(seed)
        self.trees_ = []          # each: list of node dicts (array-packed)
        self.bin_edges_ = None
        self.base_ = 0.0

    # ------------------------------------------------------------------ #
    def _bin(self, X):
        """Quantile-bin each feature; returns uint8 codes and stores edges."""
        n, d = X.shape
        if self.bin_edges_ is None:
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.bin_edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(d)]
        codes = np.empty((n, d), dtype=np.int16)
        for j in range(d):
            codes[:, j] = np.searchsorted(self.bin_edges_[j], X[:, j], side="right")
        return codes

    def _fit_tree(self, codes, resid, sw, idx):
        """Grow one depth-limited tree breadth-first on rows `idx`.
        Returns packed nodes: (feature, bin_thr, left, right, value)."""
        d = codes.shape[1]
        nodes = []                      # dicts; leaves have feature == -1
        stack = [(idx, 0, 0)]           # (rows, depth, node_id)
        nodes.append({})
        while stack:
            rows, depth, nid = stack.pop()
            w = sw[rows]
            r = resid[rows]
            wsum = w.sum()
            value = float((r * w).sum() / max(wsum, 1e-12))
            if depth >= self.max_depth or len(rows) < 2 * self.min_leaf:
                nodes[nid].update(feature=-1, value=value)
                continue
            c = codes[rows]                                  # (m, d)
            flat = (c + np.arange(d, dtype=np.int16) * self.n_bins).ravel()
            hist_w = np.bincount(flat, weights=np.repeat(w, d),
                                 minlength=d * self.n_bins).reshape(d, self.n_bins)
            hist_rw = np.bincount(flat, weights=np.repeat(r * w, d),
                                  minlength=d * self.n_bins).reshape(d, self.n_bins)
            cw = hist_w.cumsum(1)[:, :-1]                    # left weight per (f, thr)
            crw = hist_rw.cumsum(1)[:, :-1]
            tw, trw = cw[:, -1:] + hist_w[:, -1:], crw[:, -1:] + hist_rw[:, -1:]
            rw_ = tw - cw                                    # right weight
            rrw = trw - crw
            ok = (cw > 1e-12) & (rw_ > 1e-12)
            gain = np.where(ok, crw**2 / np.maximum(cw, 1e-12)
                            + rrw**2 / np.maximum(rw_, 1e-12), -np.inf)
            f, thr = np.unravel_index(int(np.argmax(gain)), gain.shape)
            if not np.isfinite(gain[f, thr]) or gain[f, thr] <= (trw[0, 0]**2 / max(tw[0, 0], 1e-12)) + 1e-12:
                nodes[nid].update(feature=-1, value=value)
                continue
            mask = codes[rows, f] <= thr
            lrows, rrows = rows[mask], rows[~mask]
            if len(lrows) < self.min_leaf or len(rrows) < self.min_leaf:
                nodes[nid].update(feature=-1, value=value)
                continue
            lid, rid = len(nodes), len(nodes) + 1
            nodes.append({}); nodes.append({})
            nodes[nid].update(feature=int(f), thr=int(thr), left=lid, right=rid,
                              value=value)
            stack.append((lrows, depth + 1, lid))
            stack.append((rrows, depth + 1, rid))
        return nodes

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        sw = (np.ones(n) if sample_weight is None
              else np.asarray(sample_weight, dtype=np.float64))
        self.bin_edges_ = None
        codes = self._bin(X)
        self.base_ = float((y * sw).sum() / max(sw.sum(), 1e-12))
        pred = np.full(n, self.base_)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        for _ in range(self.n_estimators):
            resid = y - pred
            if self.subsample < 1.0:
                idx = rng.choice(n, size=max(2 * self.min_leaf,
                                             int(self.subsample * n)), replace=False)
            else:
                idx = np.arange(n)
            tree = self._fit_tree(codes, resid, sw, idx)
            self.trees_.append(tree)
            pred += self.lr * self._predict_codes(codes, tree)
        return self

    # ------------------------------------------------------------------ #
    def _predict_codes(self, codes, tree):
        out = np.empty(len(codes))
        node_ids = np.zeros(len(codes), dtype=np.int32)
        active = np.arange(len(codes))
        while len(active):
            done = []
            for nid in np.unique(node_ids[active]):
                rows = active[node_ids[active] == nid]
                nd = tree[nid]
                if nd["feature"] == -1:
                    out[rows] = nd["value"]
                    done.extend(rows.tolist())
                else:
                    mask = codes[rows, nd["feature"]] <= nd["thr"]
                    node_ids[rows[mask]] = nd["left"]
                    node_ids[rows[~mask]] = nd["right"]
            if done:
                active = np.setdiff1d(active, np.asarray(done, dtype=active.dtype),
                                      assume_unique=False)
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        codes = np.empty(X.shape, dtype=np.int16)
        for j in range(X.shape[1]):
            codes[:, j] = np.searchsorted(self.bin_edges_[j], X[:, j], side="right")
        pred = np.full(len(X), self.base_)
        for tree in self.trees_:
            pred += self.lr * self._predict_codes(codes, tree)
        return pred

    @property
    def feature_importances_(self):
        d = len(self.bin_edges_) if self.bin_edges_ is not None else 0
        imp = np.zeros(d)
        for tree in self.trees_:
            for nd in tree:
                if nd.get("feature", -1) >= 0:
                    imp[nd["feature"]] += 1.0
        s = imp.sum()
        return imp / s if s > 0 else imp


if __name__ == "__main__":
    # smoke test: learn y = x0*x1 + noise (interaction => beyond ridge)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8))
    y = X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=4000)
    m = NpGBDT(n_estimators=120, max_depth=4, seed=0).fit(X, y)
    r = np.corrcoef(m.predict(X), y)[0, 1]
    print(f"train corr (interaction target): {r:.3f}  (ridge would be ~0)")
    assert r > 0.85

#!/usr/bin/env python3
"""
gbfcpp.py -- GBFC++: breakpoint-residual boosting with a GBDT move-proposal
policy (extension of GBFC, tools/gbfc.py).

Why GBFC saturates
------------------
GBFC's weak learner is decoded by a band-restricted CMA over `argsort` scores.
An argsort decode moves MANY vertices at once: it cannot express the move the
front actually needs near convergence -- "shift THIS breakpoint one t-step to
the left", i.e. relocate one or two specific vertices. On small-graph the
remaining gap to the leaderboard top is 925 HV = 925 unit staircase cells
(HV = n^2 - sum_t width(t)); capturing them needs surgical, single-vertex
moves at specific breakpoints. GBFC++ keeps the boosting loop (residual ->
weak learner -> pool) and replaces the weak learner's *decoder* with a
breakpoint-targeted local search whose move proposals are sampled from a GBDT.

The boost round
---------------
  1. RESIDUAL: pooled staircase W(t); breakpoints (w, t_w). The marginal HV of
     moving breakpoint w left is 1 per t-step, bounded by t_w - t_{w+1}; pick
     the breakpoint with the largest remaining potential (decayed by failures).
  2. WEAK LEARNER: fit a GBDT (LightGBM / XGBoost / numpy fallback) on the pool
     elites best in the target zone: F[v] -> normalised elimination position.
     The model's *disagreement* with the current ordering (predicted-earlier
     vs placed-later) defines a move-proposal distribution: a learned policy
     for WHICH vertex to relocate WHERE.
  3. DECODE: incremental-evaluation local search from the best-in-zone pool
     ordering. Moves: GBDT-disagreement relocation, offender relocation
     (the positions whose fill degree blocks the breakpoint), offender
     deferral, neighbour-of-offender promotion, segment reverse. Acceptance:
     lexicographic (pooled-cells gained, zone pressure).
  4. POOL: archive the improved staircase points; re-pool; the residual
     shifts; repeat. Submission checkpointed every round (timeout-safe).

Ablation: --no-gbdt replaces the learned proposal with uniform random
relocation (same budget, same seeds) -- isolates the GBDT contribution.

    python3 tools/gbfcpp.py --problem small-graph --rounds 16 --round-budget 60
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, math, random, time
import numpy as np
from core import (load_graph, build_adj_bitsets, graph_path, repo_root, ParetoArchive,
                  hypervolume_2d, MAX_TW, submission_path, write_submission,
                  load_decision_vectors, LEADERBOARD_TARGETS)
from algorithms.continuous.cmaes_torso import get_features
from algorithms.continuous.gbdt_torso import make_gbdt, training_set
from tools.gbfc import banked, standardize


# --------------------------------------------------------------------------- #
# Incremental staircase evaluator (prefix-checkpointed elimination game)
# --------------------------------------------------------------------------- #
class IncEval:
    """Full elimination pass with checkpoints of the fill state every C steps;
    a move that changes positions [L, ...] re-walks only from the last
    checkpoint <= L. deg/staircase prefixes are reused."""

    def __init__(self, ab, n, C=64):
        self.ab = ab; self.n = n; self.C = C
        self.perm = None; self.deg = None
        self._ckpt = []          # list of (i, tmp_snapshot)

    def full(self, perm):
        n, C = self.n, self.C
        sm = [0]*n; cur = 0
        for i in range(n-1, -1, -1):
            sm[i] = cur; cur |= 1 << perm[i]
        tmp = list(self.ab); deg = np.zeros(n, dtype=np.int64)
        self._ckpt = []
        for i in range(n):
            if i % C == 0:
                self._ckpt.append((i, list(tmp)))
            s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
            while x:
                b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
        self.perm = list(perm); self.deg = deg
        return deg

    def move(self, perm, L):
        """Evaluate `perm` whose positions < L are identical to the cached
        perm. Returns the full deg array (cached prefix + recomputed suffix).
        Does NOT update the cache; call commit() to keep it."""
        n = self.n
        ci = min(L // self.C, len(self._ckpt) - 1)
        i0, snap = self._ckpt[ci]
        sm = [0]*n; cur = 0
        for i in range(n-1, -1, -1):
            sm[i] = cur; cur |= 1 << perm[i]
        tmp = list(snap); deg = self.deg.copy()
        new_ck = []
        for i in range(i0, n):
            if i % self.C == 0:
                new_ck.append((i, list(tmp)))
            s = tmp[perm[i]] & sm[i]; deg[i] = s.bit_count(); x = s
            while x:
                b = x & -x; x ^= b; v = b.bit_length()-1; tmp[v] |= s ^ b
        self._pending = (list(perm), deg, ci, new_ck)
        return deg

    def commit(self):
        perm, deg, ci, new_ck = self._pending
        self.perm = perm; self.deg = deg
        self._ckpt = self._ckpt[:ci] + new_ck if new_ck else self._ckpt[:ci+1]
        # ensure checkpoint 0 exists
        if not self._ckpt or self._ckpt[0][0] != 0:
            self.full(perm)


def staircase(deg):
    return np.maximum.accumulate(deg[::-1])[::-1]


# --------------------------------------------------------------------------- #
# GBDT move-proposal policy
# --------------------------------------------------------------------------- #
def fit_proposal(F, pool_scored, n, seed):
    """Train GBDT on zone-best pool elites: F[v] -> normalised position.
    Returns predicted positions (continuous, scaled to [0, n))."""
    top = pool_scored[:max(4, len(pool_scored)//3)]
    X, y, w = training_set([(1.0, p) for _, p in top], F, n)
    name, gb = make_gbdt("auto", seed)
    gb.fit(X, y, sample_weight=w)
    pred = np.asarray(gb.predict(F), dtype=np.float64)
    r = pred.argsort().argsort().astype(np.float64)   # rank-normalise
    return name, r


# --------------------------------------------------------------------------- #
# Breakpoint-targeted local search (the weak learner's decoder)
# --------------------------------------------------------------------------- #
def ls_breakpoint(ev, perm0, Wpool, lo, hi, target_w, pred_pos, rng, budget,
                  arch, no_gbdt=False, mates=None, t0=1.5):
    """Push breakpoint (target_w, hi) leftwards. Fitness (maximise, lexico):
       gain  = cells improved vs pooled staircase (global)
       press = - sum over [lo, hi) of (width - target_w)+   (zone pressure)
    Every accepted state's improved points are archived."""
    n = ev.n
    deg = ev.full(list(perm0))
    w = staircase(deg)

    def fitness(w_):
        """Scalar: pooled cells gained dominate (x1024); zone pressure breaks
        ties and provides gradient toward the target breakpoint."""
        gain = int(np.maximum(Wpool - w_, 0).sum())
        press = int(np.maximum(w_[lo:hi] - target_w, 0).sum())
        return 1024.0 * gain - press

    best_f = cur_f = fitness(w)
    cur = list(perm0)
    pos = np.empty(n, dtype=np.int64); pos[np.asarray(cur)] = np.arange(n)
    t0 = time.time(); evals = 0; accepts = 0; last_improve = time.time()
    # ACHIEVER POPULATION: distinct orderings that hold the current best
    # fitness at this breakpoint. The offender census showed the binding
    # breakpoints are each held by a SINGLE pool ordering -- one positive
    # example, i.e. supervision collapse for the GBDT weak learner. Collecting
    # equal-fitness committed states gives the next round's weak learner (and
    # this round's restarts) a real population.
    achievers = {tuple(cur[max(0, lo-20):]): list(cur)}

    def archive_improved(w_, p_):
        idxs = np.where(w_ < Wpool)[0]
        # archive only breakpoint rows (first t of each improved width)
        seen_w = set()
        for t in idxs:
            ww = int(w_[t])
            if ww not in seen_w:
                seen_w.add(ww)
                arch.try_add(ww, int(t), list(p_))

    archive_improved(w, cur)

    def relocate(i, j):
        """perm with vertex at i moved to index j; leftmost changed."""
        p2 = list(ev.perm); vv = p2.pop(i); p2.insert(j, vv)
        return p2, min(i, j)

    # ---------------- phase A: exact boundary scan ----------------------- #
    # While the budget allows, try EVERY suffix vertex (GBDT-ordered; random
    # under the ablation) as the new boundary vertex at hi-1. Each trial is a
    # short suffix walk; if one clears (all degrees <= target_w from hi-1 on),
    # the breakpoint provably moves one t-step left. Repeat.
    bnd = hi
    while time.time() - t0 < budget * 0.7 and bnd > lo:
        j = bnd - 1
        cands = [ev.perm[i] for i in range(j, n)]
        if no_gbdt:
            cands = [cands[k] for k in rng.permutation(len(cands))]
        else:
            cands.sort(key=lambda v: -pred_pos[v])     # predicted-late first
        cleared = False
        for v in cands:
            if time.time() - t0 > budget * 0.7:
                break
            i = int(pos[v])
            if i == j: continue
            p2, L = relocate(i, j)
            deg2 = ev.move(p2, L); evals += 1
            w2 = staircase(deg2)
            f2 = fitness(w2)
            if f2 > cur_f:
                ev.commit(); cur_f = f2
                pos[np.asarray(ev.perm)] = np.arange(n)
                if f2 > best_f:
                    best_f = f2; last_improve = time.time()
                    archive_improved(w2, ev.perm)
                accepts += 1
                if int(w2[j]) <= target_w:
                    cleared = True
                break
        if cleared:
            bnd -= 1                                   # breakpoint shifted left
        else:
            break                                      # no single move clears

    # ---------------- phase B: stochastic GBDT-guided LS ----------------- #
    # Annealed acceptance: small regressions are accepted with Metropolis
    # probability (temperature decays over the round) so the walk can cross
    # the shallow valleys that strict descent + sideways cannot.
    T = t0
    stall_t = max(8.0, budget * 0.25)
    while time.time() - t0 < budget:
        # offenders: steps inside the zone whose fill degree blocks target_w
        off = np.where(ev.deg[lo:hi] > target_w)[0] + lo
        m = rng.random()
        if len(off) and m < 0.45:
            i = int(off[-1] if rng.random() < 0.6 else rng.choice(off))
            v = ev.perm[i]
            r2 = rng.random()
            if r2 < 0.4:
                # relocate the offender: GBDT-predicted position (learned
                # proposal) or uniform (ablation)
                j = (int(np.clip(pred_pos[v] + rng.normal(0, n*0.03), 0, n-1))
                     if not no_gbdt else int(rng.integers(n)))
                if i == j: continue
                p2, L = relocate(i, j)
            elif r2 < 0.75:
                j = int(rng.integers(max(1, lo)))      # promote into the prefix
                if i == j: continue
                p2, L = relocate(i, j)
            else:
                # compound boundary repair: offender -> just-left-of-zone,
                # then a (GBDT-late / random) suffix vertex -> offender's slot
                q = int(rng.integers(max(1, lo - 80), max(2, lo)))
                p2 = list(ev.perm); p2.pop(i); p2.insert(q, v)
                suf = p2[i+1:] if i+1 < n else p2[lo:]
                if not suf: continue
                if no_gbdt:
                    u = suf[int(rng.integers(len(suf)))]
                else:
                    u = max(suf[:200] if len(suf) > 200 else suf,
                            key=lambda z: pred_pos[z])
                p2.remove(u); p2.insert(i, u); L = q
        elif m < 0.75:
            # GBDT-disagreement relocation: sample vertex by |pred - pos|
            if no_gbdt:
                v = int(rng.integers(n)); j = int(rng.integers(n))
            else:
                dis = np.abs(pred_pos - pos).astype(np.float64)
                s = dis.sum()
                v = int(rng.choice(n, p=dis/s)) if s > 0 else int(rng.integers(n))
                j = int(np.clip(pred_pos[v] + rng.normal(0, n*0.02), 0, n-1))
            i = int(pos[v])
            if i == j: continue
            p2, L = relocate(i, j)
        elif m < 0.88:
            # adjacent transposition near the breakpoint boundary
            i = int(np.clip(hi - 1 + rng.integers(-30, 30), 0, n-2))
            p2 = list(ev.perm); p2[i], p2[i+1] = p2[i+1], p2[i]; L = i
        elif m < 0.91 or not mates:
            # short segment reverse in/near the zone
            a = int(rng.integers(max(0, lo-50), n-2))
            b = min(n, a + int(rng.integers(2, 40)))
            p2 = ev.perm[:a] + ev.perm[a:b][::-1] + ev.perm[b:]; L = a
        elif m < 0.96:
            # block relocation: a contiguous zone block moved as one unit
            # (to the prefix, or to the blocks' mean GBDT-predicted position)
            blen = int(rng.integers(3, 13))
            a = int(rng.integers(max(0, lo - 30), max(1, hi - blen)))
            blk = ev.perm[a:a+blen]
            rest = ev.perm[:a] + ev.perm[a+blen:]
            if no_gbdt or rng.random() < 0.5:
                j = int(rng.integers(max(1, lo)))
            else:
                j = int(np.clip(np.mean([pred_pos[v] for v in blk]), 0, n - blen))
            p2 = rest[:j] + blk + rest[j:]; L = min(a, j)
        else:
            # path relinking: keep prefix [0,c), order the rest as in a mate
            mate = mates[int(rng.integers(len(mates)))]
            c = int(rng.integers(max(1, lo - 50), min(n - 1, hi + 50)))
            head = ev.perm[:c]; hs = set(head)
            p2 = head + [v for v in mate if v not in hs]; L = c
        deg2 = ev.move(p2, max(0, L)); evals += 1
        w2 = staircase(deg2)
        f2 = fitness(w2)
        if f2 >= cur_f or rng.random() < math.exp((f2 - cur_f) / T):
            ev.commit(); cur_f = f2
            pos[np.asarray(ev.perm)] = np.arange(n)
            if f2 > best_f:
                best_f = f2; last_improve = time.time()
                archive_improved(w2, ev.perm)
                achievers = {tuple(ev.perm[max(0, lo-20):]): list(ev.perm)}
            elif f2 == best_f and len(achievers) < 12:
                k = tuple(ev.perm[max(0, lo-20):])
                if k not in achievers:
                    achievers[k] = list(ev.perm)        # new distinct achiever
            accepts += 1
        T = max(0.25, T * 0.99995)                      # anneal
        # stuck: kick the current solution (ILS), or restart from a distinct
        # ACHIEVER (same breakpoint, different basin), or from perm0/mates
        if time.time() - last_improve > stall_t:
            r = rng.random()
            if r < 0.4:
                pk = list(ev.perm)
                for _ in range(int(rng.integers(4, 12))):
                    i = int(rng.integers(max(0, lo - 50), n))
                    j = int(rng.integers(max(1, lo - 50), n))
                    vv = pk.pop(i); pk.insert(j, vv)
                ev.full(pk)
            elif r < 0.75 and len(achievers) > 1:
                ach = list(achievers.values())
                ev.full(list(ach[int(rng.integers(len(ach)))]))
            else:
                bases = [perm0] + (mates or [])
                ev.full(list(bases[int(rng.integers(len(bases)))]))
            cur_f = fitness(staircase(ev.deg))
            pos[np.asarray(ev.perm)] = np.arange(n)
            last_improve = time.time(); T = t0          # reheat
    return best_f, evals, accepts, list(achievers.values())


# --------------------------------------------------------------------------- #
def pooled_staircase(pool, ev):
    W = np.full(ev.n, ev.n, dtype=np.int64)
    for p in pool:
        W = np.minimum(W, staircase(ev.full(p)))
    return W


def breakpoints(W, n):
    """[(w, first_t)] sorted by w ascending."""
    front = {}
    for t in range(n):
        w = int(W[t])
        if w not in front:
            front[w] = t
        front[w] = min(front[w], t)
    return sorted(front.items())


def banked_excluding(here, problem, n, ab, stems):
    """banked() but skipping submission files whose stem starts with any of
    `stems` -- used by the ablation to start from the pre-GBFC++ pool."""
    import glob as _glob
    out, seen = [], set()
    sub = os.path.join(here, "submissions", problem)
    files = ([os.path.join(sub, "portfolio.json")]
             + sorted(_glob.glob(os.path.join(sub, "*.json")))
             + sorted(_glob.glob(os.path.join(sub, "seeds", "*.json"))))
    for fp in files:
        stem = os.path.basename(fp).rsplit(".", 1)[0]
        if any(stem.startswith(s) for s in stems):
            continue
        if not os.path.exists(fp):
            continue
        dvs = load_decision_vectors(fp)
        if not dvs:
            continue
        for dv in dvs:
            if isinstance(dv, list) and len(dv) == n + 1 and \
                    sorted(int(x) for x in dv[:-1]) == list(range(n)):
                k = tuple(int(x) for x in dv[:-1])
                if k not in seen:
                    seen.add(k); out.append(list(k))
    return out


def run(problem, rounds, round_budget, seed, here, algo="gbfcpp", no_gbdt=False,
        k_eig=32, exclude_stems=(), t0=1.5, only_widths=()):
    n, adj = load_graph(graph_path(here, problem)); ab = build_adj_bitsets(n, adj)
    target = LEADERBOARD_TARGETS.get(problem)
    tag = "GBFC++(no-gbdt ablation)" if no_gbdt else "GBFC++"
    print(f"\n=== {tag} -- {problem} (rounds={rounds}, {round_budget}s/round, seed={seed}) ===")
    F = np.asarray(get_features(here, problem, n, adj, k_eig)[0])
    try:
        from tools.fastwalk import IncEvalC
        ev = IncEvalC(ab, n)
        print("evaluator: C kernel (fastwalk)", flush=True)
    except Exception as e:  # noqa: BLE001
        ev = IncEval(ab, n)
        print(f"evaluator: python bitset [fastwalk unavailable: {e}]", flush=True)
    rng = np.random.default_rng(seed)

    # resume-safe pooling: our own previous checkpoint FIRST (so the cap never
    # drops earlier GBFC++ progress), then the banked corpus
    pool = []
    own = submission_path(here, problem, algo)
    if os.path.exists(own) and not exclude_stems:
        for dv in load_decision_vectors(own):
            if isinstance(dv, list) and len(dv) == n + 1:
                pool.append([int(x) for x in dv[:-1]])
    seen = {tuple(p) for p in pool}
    corpus = (banked_excluding(here, problem, n, ab, exclude_stems)
              if exclude_stems else banked(here, problem, n, ab))
    for p in corpus:
        if tuple(p) not in seen:
            seen.add(tuple(p)); pool.append(p)
    # Build full archive from ALL banked orderings first, then select the
    # top-60 by HV contribution so we always start from the best achievable
    # pooled front (not just the first 60 in file order).
    arch_all = ParetoArchive()
    for p in pool:
        w = staircase(ev.full(p))
        for wt, t in breakpoints(w, n):
            if wt <= MAX_TW:
                arch_all.try_add(wt, t, list(p))
    top_all = arch_all.top_k_by_hv_contribution(60, n)
    pool = [list(p) for (_, _, p) in top_all]

    arch = ParetoArchive()
    for p in pool:
        w = staircase(ev.full(p))
        for wt, t in breakpoints(w, n):
            if wt <= MAX_TW:
                arch.try_add(wt, t, list(p))
    best = -arch.hypervolume(n)
    print(f"pooled {len(pool)} banked -> {best:,.0f}", flush=True)

    out = submission_path(here, problem, algo)
    def save_sub():
        top = arch.top_k_by_hv_contribution(60, n)   # save 60, not 20
        write_submission([list(p)+[int(t)] for (_, t, p) in top], problem, out)
    save_sub()

    # failure decay persists across (resumed) invocations
    state_fp = os.path.join(here, "submissions", problem, f".{algo}_state.json")
    fails = {}
    if os.path.exists(state_fp):
        try:
            fails = {int(k): v for k, v in json.load(open(state_fp)).items()}
        except Exception:
            fails = {}
    # achiever populations persist too: without this, the anti-supervision-
    # collapse population dies with each (wave) process and every restart
    # re-collapses to a single positive example per breakpoint
    ach_fp = os.path.join(here, "submissions", problem, f".{algo}_achievers.json")
    ach_store = {}
    if os.path.exists(ach_fp):
        try:
            ach_store = {k: [list(map(int, p)) for p in v]
                         for k, v in json.load(open(ach_fp)).items()}
        except Exception:
            ach_store = {}
    for r in range(rounds):
        W = pooled_staircase(pool, ev)
        bps = breakpoints(W, n)
        # potential of pushing breakpoint w left = room to the next smaller
        # width's breakpoint (bps sorted by w asc => t desc), decayed by the
        # round failures on that breakpoint
        cand = []
        for k in range(len(bps) - 1):
            w_, t_ = bps[k]                       # breakpoint (w, first t)
            room = t_ - bps[k+1][1]               # next LARGER width starts left
            if room <= 0 or t_ == 0:
                continue
            decay = 0.5 ** fails.get(w_, 0)
            cand.append((room * decay, w_, t_, bps[k+1][1]))
        if not cand:
            print("no addressable breakpoints; stopping"); break
        # swarm partitioning: restrict this worker to its assigned widths
        # (fall back to all if its widths have no addressable breakpoints)
        if only_widths:
            mine = [c for c in cand if c[1] in only_widths]
            if mine:
                cand = mine
        # sample a breakpoint ~ room x failure-decay (rotation beats argmax:
        # a +1-trickle breakpoint must not monopolise the rounds)
        wts = np.array([c[0] for c in cand], dtype=np.float64)
        pick = int(rng.choice(len(cand), p=wts / wts.sum()))
        _, tw, hi, lo = cand[pick]                # push (tw, hi) left toward lo
        # weak learner on zone-best elites (achieve tw at hi; low zone width)
        scored = []
        for p in pool:
            wp = staircase(ev.full(p))
            scored.append(((int(wp[min(hi, n-1)]), int(wp[lo:hi].max())), p))
        scored.sort(key=lambda z: z[0])
        if no_gbdt:
            name, pred = "none", np.arange(n, dtype=np.float64)
        else:
            name, pred = fit_proposal(F, scored, n, seed + r)
        base_p = scored[0][1]
        prev = best
        # mates: zone elites PLUS two random pool members -- path relinking
        # between near-clones does nothing, diversity gives it real material
        mates = [p for _, p in scored[1:5]]
        if len(pool) > 6:
            mates += [pool[int(rng.integers(5, len(pool)))] for _ in range(2)]
        mates += ach_store.get(str(tw), [])[:6]   # stored breakpoint population
        bf, evs, acc, achievers = ls_breakpoint(
            ev, base_p, W, lo, hi, tw, pred, rng, round_budget, arch,
            no_gbdt=no_gbdt, mates=mates, t0=t0)
        # merge + persist this breakpoint's achiever population
        seen_a = set(); merged = []
        for p in achievers + ach_store.get(str(tw), []):
            k = tuple(p[max(0, lo-20):][:60])
            if k not in seen_a:
                seen_a.add(k); merged.append(list(p))
        ach_store[str(tw)] = merged[:12]
        try:
            json.dump(ach_store, open(ach_fp, "w"))
        except Exception:
            pass
        cur = -arch.hypervolume(n)
        if cur >= prev - 0.5:
            fails[tw] = fails.get(tw, 0) + 1
        else:
            fails[tw] = 0
        try:
            json.dump({str(k): v for k, v in fails.items()}, open(state_fp, "w"))
        except Exception:
            pass
        best = min(best, cur)
        top = arch.top_k_by_hv_contribution(40, n)
        # achievers enter the pool FIRST: next round's weak learner trains on
        # a population at the breakpoint instead of a single positive example
        pool = achievers + [list(p) for _, _, p in top] + pool[:40]
        save_sub()
        msg = (f"  round {r+1:2d} | bp w={tw} t={hi} zone[{lo},{hi}) | {name} | "
               f"{evs} evals {acc} acc | ach {len(ach_store.get(str(tw), []))} "
               f"| score {best:,.0f}")
        if target is not None:
            msg += f" | gap {best-target:+,.0f}" + (" BEAT!" if best < target else "")
        print(msg, flush=True)

    final = -arch.hypervolume(n)
    print(f"\nFinished. Official score: {final:,.0f}")
    if target is not None:
        g = final - target
        print(f"Gap to target ({target:,}): {g:+,.0f} ({'BEAT' if g < 0 else f'{abs(g):,.0f} short'})")
    save_sub()
    print(f"Wrote {out}")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="small-graph", choices=list(LEADERBOARD_TARGETS.keys()))
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--round-budget", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algo", default="gbfcpp")
    ap.add_argument("--no-gbdt", action="store_true",
                    help="ablation: uniform move proposals instead of the GBDT policy")
    ap.add_argument("--exclude-stems", default="",
                    help="comma-separated submission stems to EXCLUDE from the "
                         "pool (ablation from the pre-GBFC++ plateau: gbfcpp)")
    ap.add_argument("--t0", type=float, default=1.5,
                    help="initial annealing temperature (swarm varies this)")
    ap.add_argument("--only-widths", default="",
                    help="comma-separated breakpoint widths this worker may "
                         "attack (swarm partitioning); empty = all")
    args = ap.parse_args()
    ex = tuple(s for s in args.exclude_stems.split(",") if s)
    ow = tuple(int(s) for s in args.only_widths.split(",") if s)
    run(args.problem, args.rounds, args.round_budget, args.seed, repo_root(),
        algo=args.algo, no_gbdt=args.no_gbdt, exclude_stems=ex,
        t0=args.t0, only_widths=ow)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
solver_final.py

Single-file, CPU-only memetic NSGA-II-style solver with:
 - robust sanitizers
 - per-process LRU cached fast bitset evaluator
 - targeted offender repair, iterated greedy rebuild (IG)
 - local search, tabu intensification for elites
 - immigrant injection for diversity
 - Pareto archive + hypervolume tracking and persistence
 - adaptive mutation & aggressive restarts

Usage:
  python solver_final.py --problem easy --pop_size 220 --generations 600 --intensity 28
  python solver_final.py --problem easy --generations 1 --pop_size 40 --diagnostic
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import random
import time
import urllib.request
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm
import multiprocessing

# -------------------------
# CONFIG default
# -------------------------
DEFAULT_CONFIG = {
    "general": {
        "mutation_rate": 0.55,
        "crossover_rate": 0.95,
        "checkpoint_interval": 10,
        "elite_count": 6,
        "elite_ls_multiplier": 4,
        "stagnation_limit": 12,
        "mutation_boost_factor": 1.8,
        "elite_tabu_iters": 400,
        "elite_tabu_tenure": (6, 20),
        "elite_tabu_count": 4,
        "archive_max_size": 800,
        "archive_pls_freq": 4,
        "pareto_local_neighbor_samples": 600,
        "immigrant_every": 12,
        "immigrant_fraction": 0.06,
        "restart_on_stagnation": True,
        "restart_stagnation_threshold": 80,
        "restart_keep_fraction": 0.10,
    },
    "easy": {"pop_size": 220, "generations": 600, "local_search_intensity": 28},
    "medium": {"pop_size": 220, "generations": 500, "local_search_intensity": 28},
    "hard": {"pop_size": 300, "generations": 1000, "local_search_intensity": 40},
}

PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# -------------------------
# Worker globals for pool
# -------------------------
WORKER_ADJ_BITS: List[int] | None = None
WORKER_N: int | None = None

# sanitizer warn counter as multiprocessing.Value
_SAN_WARN = multiprocessing.Value('i', 0)

# -------------------------
# Utilities: download/load graph
# -------------------------
def download_graph(problem_id: str, target_dir: str = "data/spoc3/torso") -> str:
    url = PROBLEMS[problem_id]
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, f"{problem_id}.gr")
    if not os.path.exists(local_path):
        print(f"📥 Downloading graph {problem_id} ...")
        with urllib.request.urlopen(url) as resp, open(local_path, "wb") as out:
            out.write(resp.read())
    return local_path

def load_graph_from_file(local_path: str) -> Tuple[int, List[Set[int]]]:
    edges = []
    max_node = -1
    with open(local_path, "rb") as f:
        for line in f:
            if line.startswith(b'#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except Exception:
                continue
            edges.append((u, v))
            if u > max_node: max_node = u
            if v > max_node: max_node = v
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Loaded graph with {n} nodes and {len(edges)} edges from {local_path}")
    return n, adj

def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    adj_bits = [0] * n
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits

# -------------------------
# Pool initializer
# -------------------------
def _init_worker(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

# -------------------------
# Sanitizer for solutions
# -------------------------
def repair_perm_list(perm: List[Any], n: int) -> List[int]:
    seen = set()
    fixed = []
    for x in perm:
        try:
            xi = int(x)
        except Exception:
            continue
        if 0 <= xi < n and xi not in seen:
            seen.add(xi)
            fixed.append(xi)
    for v in range(n):
        if v not in seen:
            fixed.append(v)
    return fixed[:n]

def sanitize_solution_tuple(sol_tpl: Tuple[Any, ...], n: int) -> Tuple[int, ...]:
    """
    Ensure tuple length == n+1, valid permutation and t in range.
    """
    with _SAN_WARN.get_lock():
        warn_allowed = _SAN_WARN.value < 40
    if len(sol_tpl) == n + 1:
        perm_raw = sol_tpl[:-1]
        t_raw = sol_tpl[-1]
        try:
            perm_ints = [int(x) for x in perm_raw]
            if len(set(perm_ints)) == n and all(0 <= x < n for x in perm_ints):
                t = int(t_raw)
                t = max(0, min(n - 1, t))
                return tuple(perm_ints) + (t,)
        except Exception:
            pass
    # repair
    perm_raw = list(sol_tpl[:-1]) if len(sol_tpl) >= 1 else []
    t_raw = sol_tpl[-1] if len(sol_tpl) >= 1 else 0
    fixed_perm = repair_perm_list(perm_raw, n)
    try:
        t = int(t_raw)
    except Exception:
        t = 0
    t = max(0, min(n - 1, t))
    with _SAN_WARN.get_lock():
        if _SAN_WARN.value < 40:
            print("⚠️  Warning: repaired malformed permutation passed to evaluator (length/dup/out-of-range).")
            _SAN_WARN.value += 1
    return tuple(int(x) for x in fixed_perm) + (t,)

# -------------------------
# Fast evaluator with LRU cache
# -------------------------
def bitcount(x: int) -> int:
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count('1')

@lru_cache(maxsize=400_000)
def evaluate_solution_bitset_cached(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker not initialized")
    n = WORKER_N
    sol = sanitize_solution_tuple(solution_tuple, n)
    t = int(sol[-1])
    perm = list(sol[:-1])
    size = n - t
    if size <= 0:
        return (0, 501)
    adj_bits = WORKER_ADJ_BITS
    suffix_mask = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr
        curr |= (1 << perm[i])
    temp = adj_bits[:]  # copy
    max_width = 0
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount(succ)
        if out_deg > max_width:
            max_width = out_deg
            if max_width >= 500:
                return (size, 501)
        if succ == 0:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ (1 << v))
    return (size, max_width)

def eval_wrapper(solution_tuple):
    try:
        return (solution_tuple, evaluate_solution_bitset_cached(solution_tuple))
    except Exception:
        # sanitize and try again
        n = WORKER_N if WORKER_N else 0
        if n > 0:
            st = sanitize_solution_tuple(solution_tuple, n)
            try:
                return (st, evaluate_solution_bitset_cached(st))
            except Exception:
                pass
        return (solution_tuple, (0, 501))

# -------------------------
# Offender analysis & targeted repair
# -------------------------
def compute_maxwidth_offenders(solution_tuple: Tuple[int, ...]) -> Tuple[int, int, List[int], Dict[int, int]]:
    global WORKER_ADJ_BITS, WORKER_N
    if WORKER_ADJ_BITS is None or WORKER_N is None:
        raise RuntimeError("Worker not initialized")
    n = WORKER_N
    sol = sanitize_solution_tuple(solution_tuple, n)
    t = int(sol[-1])
    perm = list(sol[:-1])
    size = n - t
    if size <= 0:
        return (0, 501, [], {})
    adj_bits = WORKER_ADJ_BITS
    suffix_mask = [0] * n
    curr = 0
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr
        curr |= (1 << perm[i])
    temp = adj_bits[:]
    max_width = 0
    offenders = []
    succ_masks = {}
    for i in range(n):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount(succ)
        if out_deg > max_width:
            max_width = out_deg
            offenders = [u]
            succ_masks = {u: succ}
        elif out_deg == max_width and out_deg > 0:
            offenders.append(u)
            succ_masks[u] = succ
        if succ == 0:
            continue
        s = succ
        while s:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            temp[v] |= (succ ^ (1 << v))
    return (size, max_width, offenders, succ_masks)

def targeted_repair(solution: List[int], max_tries: int = 200) -> List[int]:
    n = WORKER_N
    if n is None:
        return solution
    sol_tpl = tuple(int(x) for x in solution)
    size, maxw, offenders, succ_masks = compute_maxwidth_offenders(sol_tpl)
    if maxw <= 1 or not offenders:
        return solution
    perm = solution[:-1]
    t = solution[-1]
    best = tuple(int(x) for x in solution)
    best_score = evaluate_solution_bitset_cached(best)
    tries = 0
    random.shuffle(offenders)
    window = max(4, int(n * 0.06))
    for u in offenders:
        if tries >= max_tries:
            break
        try:
            pos_u = perm.index(u)
        except ValueError:
            continue
        for shift in [1, 2, 4, 8, 16]:
            if tries >= max_tries or shift > window:
                break
            new_pos = max(0, pos_u - shift)
            if new_pos == pos_u:
                continue
            new_perm = perm[:]
            new_perm.pop(pos_u)
            new_perm.insert(new_pos, u)
            candidate = tuple(new_perm + [t])
            cand_score = evaluate_solution_bitset_cached(candidate)
            tries += 1
            if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
                return list(candidate)
        succ = succ_masks.get(u, 0)
        succ_nodes = []
        s = succ
        limit_succs = 12
        while s and len(succ_nodes) < limit_succs:
            vbit = s & -s
            s ^= vbit
            v = vbit.bit_length() - 1
            succ_nodes.append(v)
        for v in succ_nodes:
            if tries >= max_tries:
                break
            if v not in perm:
                continue
            pos_v = perm.index(v)
            for shift in [1, 2, 4, 8]:
                if tries >= max_tries:
                    break
                new_pos = max(0, pos_v - shift)
                new_perm = perm[:]
                new_perm.pop(pos_v)
                new_perm.insert(new_pos, v)
                candidate = tuple(new_perm + [t])
                cand_score = evaluate_solution_bitset_cached(candidate)
                tries += 1
                if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
                    return list(candidate)
    return solution

# -------------------------
# Iterated Greedy rebuild (IG)
# -------------------------
def iterated_greedy_rebuild(solution: List[int], k_min=18, k_max_pct=0.20) -> List[int]:
    n = WORKER_N
    if n is None:
        return solution
    perm = solution[:-1]
    t = solution[-1]
    m = len(perm)
    k_max = max(k_min, int(m * k_max_pct))
    k = random.randint(k_min, min(k_max, max(1, m // 2)))
    suffix_start = max(0, int(m * 0.35))
    candidate_indices = list(range(suffix_start, m))
    if len(candidate_indices) <= k:
        removed_indices = candidate_indices
    else:
        removed_indices = sorted(random.sample(candidate_indices, k))
    removed_nodes = [perm[i] for i in removed_indices]
    base = [perm[i] for i in range(m) if i not in removed_indices]
    for node in removed_nodes:
        best_pos = None
        best_score = None
        # test early positions + few random mid positions
        candidate_positions = list(range(min(200, len(base) + 1)))
        if len(base) + 1 > 0:
            try:
                rnds = random.sample(range(len(base) + 1), min(80, len(base) + 1))
                candidate_positions += rnds
            except Exception:
                pass
        candidate_positions = list(dict.fromkeys(candidate_positions))
        for pos in candidate_positions:
            cand_perm = base[:]
            cand_perm.insert(pos, node)
            cand_tpl = tuple(cand_perm + [t])
            score = evaluate_solution_bitset_cached(cand_tpl)
            if best_score is None or (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best_pos = pos
        if best_pos is None:
            base.append(node)
        else:
            base.insert(best_pos, node)
    candidate = tuple(base + [t])
    cand_score = evaluate_solution_bitset_cached(candidate)
    orig_score = evaluate_solution_bitset_cached(tuple(perm + [t]))
    if (cand_score[0] > orig_score[0]) or (cand_score[0] == orig_score[0] and cand_score[1] < orig_score[1]):
        return list(candidate)
    return solution

# -------------------------
# Local search worker
# -------------------------
def block_move(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    perm = neighbor[:-1]
    block_size = random.randint(2, max(3, int(n * 0.03)))
    if n > block_size:
        start = random.randint(0, n - block_size)
        block = perm[start:start + block_size]
        del perm[start:start + block_size]
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm
    return neighbor

def inversion_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2:
        return perm[:]
    a, b = sorted(random.sample(range(len(perm)), 2))
    perm = perm[:]
    perm[a:b+1] = reversed(perm[a:b+1])
    return perm

def swap_mutation(perm: List[int]) -> List[int]:
    if len(perm) < 2:
        return perm[:]
    perm = perm[:]
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return perm

def smart_torso_shift(solution: List[int], n: int) -> List[int]:
    neighbor = solution[:]
    t = neighbor[-1]
    shift = int(max(1, n * 0.06))
    neighbor[-1] = max(0, min(n - 1, t + random.randint(-shift, shift)))
    return neighbor

def local_search_worker(args):
    sol, intensity = args
    try:
        n = WORKER_N
        best = tuple(int(x) for x in sol)
        best_score = evaluate_solution_bitset_cached(best)
        for _ in range(intensity):
            r = random.random()
            if r < 0.22:
                neigh = block_move(list(best), n)
            elif r < 0.6:
                perm = list(best[:-1])
                perm = inversion_mutation(perm)
                neigh = perm + [best[-1]]
            elif r < 0.92:
                neigh = smart_torso_shift(list(best), n)
            else:
                neigh = targeted_repair(list(best), max_tries=16)
            neigh_t = tuple(int(x) for x in neigh)
            neigh_score = evaluate_solution_bitset_cached(neigh_t)
            if (neigh_score[0] > best_score[0]) or (neigh_score[0] == best_score[0] and neigh_score[1] < best_score[1]):
                best = neigh_t
                best_score = neigh_score
        # occasional stronger IG
        if random.random() < 0.18:
            cand = iterated_greedy_rebuild(list(best), k_min=18, k_max_pct=0.20)
            cand_t = tuple(int(x) for x in cand)
            cand_score = evaluate_solution_bitset_cached(cand_t)
            if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
                best = cand_t
        return list(best)
    except Exception as e:
        # sanitize fallback
        if WORKER_N is not None:
            good = sanitize_solution_tuple(tuple(int(x) for x in sol), WORKER_N)
            return list(good)
        return list(sol)

# -------------------------
# Tabu intensification worker
# -------------------------
def tabu_search_worker(args):
    sol, iters, tenure_range = args
    n = WORKER_N
    current = tuple(int(x) for x in sol)
    best = current
    best_score = evaluate_solution_bitset_cached(best)
    curr_score = best_score
    tabu = {}
    for it in range(iters):
        candidates = []
        for _ in range(12):
            r = random.random()
            if r < 0.42:
                i, j = random.sample(range(n), 2)
                perm = list(current[:-1])
                perm[i], perm[j] = perm[j], perm[i]
                move_sig = ('swap', min(i, j), max(i, j))
                neigh = tuple(perm + [current[-1]])
            elif r < 0.84:
                a, b = sorted(random.sample(range(n), 2))
                perm = list(current[:-1])
                perm[a:b+1] = reversed(perm[a:b+1])
                move_sig = ('inv', a, b)
                neigh = tuple(perm + [current[-1]])
            else:
                perm = list(current[:-1])
                tnew = max(0, min(n - 1, current[-1] + random.randint(-max(1, int(n*0.04)), max(1, int(n*0.04)))))
                move_sig = ('t', tnew)
                neigh = tuple(perm + [tnew])
            candidates.append((move_sig, neigh))
        cand_scores = []
        for move_sig, neigh in candidates:
            sc = evaluate_solution_bitset_cached(neigh)
            cand_scores.append((move_sig, neigh, sc))
        cand_scores.sort(key=lambda x: (-x[2][0], x[2][1]))
        selected = None
        for move_sig, neigh, sc in cand_scores:
            in_tabu = move_sig in tabu and tabu[move_sig] > 0
            if (not in_tabu) or (sc[0] > best_score[0]) or (sc[0] == best_score[0] and sc[1] < best_score[1]):
                selected = (move_sig, neigh, sc)
                break
        if selected is None:
            selected = cand_scores[0]
        move_sig, neigh, sc = selected
        current = neigh
        curr_score = sc
        tenure = random.randint(tenure_range[0], tenure_range[1])
        tabu[move_sig] = tenure
        to_del = []
        for k in list(tabu.keys()):
            tabu[k] -= 1
            if tabu[k] <= 0:
                to_del.append(k)
        for k in to_del:
            del tabu[k]
        if (curr_score[0] > best_score[0]) or (curr_score[0] == best_score[0] and curr_score[1] < best_score[1]):
            best = current
            best_score = curr_score
    res = list(best)
    if random.random() < 0.8:
        res = targeted_repair(res, max_tries=64)
    if random.random() < 0.2:
        res = iterated_greedy_rebuild(res, k_min=18, k_max_pct=0.20)
    return res

# -------------------------
# NSGA-II style crowding selection
# -------------------------
def dominates_internal(p, q):
    return (p[0] >= q[0] and p[1] < q[1]) or (p[0] > q[0] and p[1] <= q[1])

def crowding_selection(population: List[Dict[str, Any]], pop_size: int) -> List[Dict[str, Any]]:
    for p in population:
        p.setdefault('dominates_set', [])
        p.setdefault('dominated_by_count', 0)
    fronts: List[List[Dict[str, Any]]] = [[]]
    for p in population:
        p['dominates_set'] = []
        p['dominated_by_count'] = 0
    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if p is q:
                continue
            if dominates_internal(p['score'], q['score']):
                p['dominates_set'].append(q)
            elif dominates_internal(q['score'], p['score']):
                p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[Dict[str, Any]] = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    new_pop: List[Dict[str, Any]] = []
    for front in fronts:
        if not front:
            continue
        if len(new_pop) + len(front) > pop_size:
            for p in front:
                p['distance'] = 0.0
            for i_obj in range(2):
                front.sort(key=lambda p: p['score'][i_obj])
                front[0]['distance'] = front[-1]['distance'] = float('inf')
                f_min = front[0]['score'][i_obj]
                f_max = front[-1]['score'][i_obj]
                if f_max > f_min:
                    for j in range(1, len(front) - 1):
                        prev_v = front[j - 1]['score'][i_obj]
                        next_v = front[j + 1]['score'][i_obj]
                        front[j]['distance'] += (next_v - prev_v) / (f_max - f_min)
            front.sort(key=lambda p: p['distance'], reverse=True)
            need = pop_size - len(new_pop)
            new_pop.extend(front[:need])
            break
        new_pop.extend(front)
    return new_pop

# -------------------------
# Hypervolume utilities (2D minimization transform)
# -------------------------
def transform_for_hv(score_tuple: Tuple[int, int], n_nodes: int) -> Tuple[int, int]:
    size, width = score_tuple
    return (width, int(n_nodes - size))

def nondominated_2d_minimization(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    nondom = []
    best_y = float('inf')
    for x, y in pts:
        if y < best_y:
            nondom.append((x, y))
            best_y = y
    return nondom

def hypervolume_2d_minimization(points: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    rx, ry = ref
    pts = [(float(x), float(y)) for x, y in points if x <= rx and y <= ry]
    if not pts:
        return 0.0
    nd = nondominated_2d_minimization(pts)
    nd_desc = sorted(nd, key=lambda p: p[0], reverse=True)
    hv = 0.0
    best_y = ry
    for x, y in nd_desc:
        if y < best_y:
            hv += (rx - x) * (best_y - y)
            best_y = y
    return float(hv)

def compute_hv_for_archive(archive: List[Dict[str, Any]], n_nodes: int) -> float:
    if not archive:
        return 0.0
    pts = [transform_for_hv(a['score'], n_nodes) for a in archive]
    ref = (n_nodes, n_nodes)
    return hypervolume_2d_minimization(pts, ref)

# -------------------------
# Archive management
# -------------------------
def add_to_archive(archive: List[Dict[str, Any]], candidate: Dict[str, Any], archive_max_size: int, n_nodes: int) -> List[Dict[str, Any]]:
    cand_score = candidate['score']
    for a in archive:
        if dominates_internal(a['score'], cand_score):
            return archive
    new_archive = [a for a in archive if not dominates_internal(cand_score, a['score'])]
    new_archive.append({'solution': candidate['solution'], 'score': cand_score})
    if len(new_archive) > archive_max_size:
        pts = [transform_for_hv(a['score'], n_nodes) for a in new_archive]
        import numpy as _np
        dmat = _np.zeros((len(pts), len(pts)))
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = _np.linalg.norm(_np.array(pts[i]) - _np.array(pts[j]))
                dmat[i, j] = dmat[j, i] = d
        sumd = [(i, dmat[i].sum()) for i in range(len(pts))]
        sumd.sort(key=lambda x: x[1], reverse=True)
        keep_idx = set(idx for idx, _ in sumd[:archive_max_size])
        new_archive = [new_archive[i] for i in range(len(new_archive)) if i in keep_idx]
    return new_archive

# -------------------------
# Pareto Local Search (PLS)
# -------------------------
def pareto_local_search(archive: List[Dict[str, Any]], n_nodes: int, budget_samples: int, pool=None) -> List[Dict[str, Any]]:
    if not archive:
        return archive
    candidates = []
    per_member = max(1, budget_samples // len(archive))
    for a in archive:
        base = a['solution']
        for _ in range(per_member):
            r = random.random()
            if r < 0.36:
                perm = base[:-1][:]
                i, j = random.sample(range(n_nodes), 2)
                perm[i], perm[j] = perm[j], perm[i]
                neigh = perm + [base[-1]]
            elif r < 0.72:
                perm = base[:-1][:]
                aidx, bidx = sorted(random.sample(range(n_nodes), 2))
                perm[aidx:bidx+1] = reversed(perm[aidx:bidx+1])
                neigh = perm + [base[-1]]
            else:
                perm = base[:-1][:]
                tnew = max(0, min(n_nodes-1, base[-1] + random.randint(-max(1,int(n_nodes*0.04)), max(1,int(n_nodes*0.04)))))
                neigh = perm + [tnew]
            candidates.append(tuple(int(x) for x in neigh))
    if pool:
        results = pool.map(eval_wrapper, candidates)
    else:
        results = [eval_wrapper(c) for c in candidates]
    for sol_tuple, score in results:
        cand = {'solution': list(sol_tuple), 'score': score}
        archive = add_to_archive(archive, cand, DEFAULT_CONFIG['general']['archive_max_size'], n_nodes)
    # targeted repairs on a few extremes
    extremes = sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[:16]
    for e in extremes:
        cand_sol = targeted_repair(e['solution'], max_tries=80)
        cand_tpl = tuple(int(x) for x in cand_sol)
        sc = evaluate_solution_bitset_cached(cand_tpl)
        archive = add_to_archive(archive, {'solution': cand_sol, 'score': sc}, DEFAULT_CONFIG['general']['archive_max_size'], n_nodes)
        if random.random() < 0.35:
            ig = iterated_greedy_rebuild(e['solution'], k_min=18, k_max_pct=0.20)
            ig_tpl = tuple(int(x) for x in ig)
            sc2 = evaluate_solution_bitset_cached(ig_tpl)
            archive = add_to_archive(archive, {'solution': ig, 'score': sc2}, DEFAULT_CONFIG['general']['archive_max_size'], n_nodes)
    return archive

# -------------------------
# Persistence helpers
# -------------------------
def persist_archive(archive: List[Dict[str, Any]], n_nodes: int, problem_id: str, combined_score: float):
    fn_pkl = f"best_archive_{problem_id}.pkl"
    fn_json = f"best_archive_{problem_id}.json"
    try:
        with open(fn_pkl, "wb") as f:
            pickle.dump({'archive': archive, 'hv': combined_score}, f)
    except Exception:
        pass
    try:
        decs = [[int(x) for x in a['solution']] for a in archive]
        with open(fn_json, "w") as f:
            json.dump({"decisionVector": decs, "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)
    except Exception:
        pass

def persist_best_solution(best_solution: List[int], best_score: Tuple[int, int], problem_id: str):
    pkl_name = f"best_solution_{problem_id}.pkl"
    json_name = f"best_submission_{problem_id}.json"
    try:
        with open(pkl_name, "wb") as f:
            pickle.dump({'solution': best_solution, 'score': best_score}, f)
    except Exception:
        pass
    try:
        with open(json_name, "w") as f:
            json.dump({"decisionVector": [[int(x) for x in best_solution]], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=2)
    except Exception:
        pass

# -------------------------
# Seeding heuristics
# -------------------------
def min_degree_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    degree = [len(adj_list[i]) for i in range(n)]
    neighbors = [set(s) for s in adj_list]
    import heapq
    heap = [(degree[i], i) for i in range(n)]
    heapq.heapify(heap)
    removed = [False] * n
    order = []
    while heap:
        d, v = heapq.heappop(heap)
        if removed[v] or degree[v] != d:
            continue
        removed[v] = True
        order.append(v)
        for u in list(neighbors[v]):
            if not removed[u]:
                neighbors[u].remove(v)
                degree[u] -= 1
                heapq.heappush(heap, (degree[u], u))
        neighbors[v].clear()
    return order

def min_fill_order(adj_list: List[Set[int]]) -> List[int]:
    n = len(adj_list)
    neighbors = [set(s) for s in adj_list]
    alive = set(range(n))
    order = []
    for _ in range(n):
        best_v = None
        best_fill = None
        for v in alive:
            neigh = neighbors[v]
            k = len(neigh)
            if k <= 1:
                fill = 0
            else:
                existing = 0
                for u in neigh:
                    existing += sum(1 for w in neighbors[u] if w in neigh)
                existing = existing // 2
                total_pairs = k * (k - 1) // 2
                fill = total_pairs - existing
            if best_fill is None or fill < best_fill:
                best_fill = fill
                best_v = v
        order.append(best_v)
        neigh = neighbors[best_v]
        for a in list(neigh):
            for b in list(neigh):
                if a != b:
                    neighbors[a].add(b)
        for u in neigh:
            neighbors[u].discard(best_v)
        neighbors[best_v].clear()
        alive.remove(best_v)
    return order

# -------------------------
# Aggressive restart / immigrant injection
# -------------------------
def aggressive_restart(population: List[Dict[str, Any]], archive: List[Dict[str, Any]], pop_size: int, n_nodes: int) -> List[Dict[str, Any]]:
    keep = max(2, int(pop_size * 0.06))
    pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
    new_pop = pop_sorted[:keep]
    archive_pool = [a['solution'] for a in sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[:min(len(archive), 200)]]
    while len(new_pop) < pop_size:
        if archive_pool and random.random() < 0.7:
            a = random.choice(archive_pool)
            b = random.choice(archive_pool)
            perm1 = a[:-1]
            perm2 = b[:-1]
            child_perm = pmx_crossover(perm1, perm2)
            for _ in range(random.randint(1, 4)):
                if random.random() < 0.6:
                    child_perm = inversion_mutation(child_perm)
                else:
                    child_perm = swap_mutation(child_perm)
            c_t = int((a[-1] + b[-1]) / 2)
            if random.random() < 0.6:
                c_t = max(0, min(n_nodes - 1, c_t + random.randint(-int(n_nodes*0.06), int(n_nodes*0.06))))
            new_pop.append({'solution': child_perm + [c_t]})
        else:
            perm = list(np.random.permutation(n_nodes))
            t = random.randint(int(n_nodes * 0.2), int(n_nodes * 0.8))
            for _ in range(random.randint(0, 3)):
                perm = inversion_mutation(perm)
                perm = swap_mutation(perm)
            new_pop.append({'solution': perm + [t]})
    return new_pop[:pop_size]

# -------------------------
# Crossover
# -------------------------
def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b+1):
        val = p2[i]
        if val in child:
            continue
        pos = i
        while True:
            mapped = p1[pos]
            try:
                pos = p2.index(mapped)
            except ValueError:
                break
            if child[pos] == -1:
                child[pos] = val
                break
    for i in range(n):
        if child[i] == -1:
            child[i] = p2[i]
    return child

# -------------------------
# Main memetic algorithm
# -------------------------
def memetic_algorithm(n: int, adj_list: List[Set[int]], config: Dict[str, Any], problem_id: str, seed_solutions: List[List[int]] | None = None, diagnostic: bool = False) -> List[List[int]]:
    checkpoint_file = f"checkpoint_{problem_id}.pkl"
    adj_bits = build_adj_bitsets(n, adj_list)
    pop_size = config['pop_size']
    local_search_intensity = config['local_search_intensity']

    # seeded initial population
    population: List[Dict[str, Any]] = []
    seed_orders: List[List[int]] = []
    try:
        seed_orders.append(min_fill_order(adj_list))
    except Exception:
        pass
    try:
        seed_orders.append(min_degree_order(adj_list))
    except Exception:
        pass
    for base in list(seed_orders):
        seed_orders.append(list(reversed(base)))
        for _ in range(4):
            p = base[:]
            for _ in range(max(1, len(p)//200)):
                i, j = random.sample(range(len(p)), 2)
                p[i], p[j] = p[j], p[i]
            seed_orders.append(p)
    while len(seed_orders) < 20:
        seed_orders.append(list(np.random.permutation(n)))

    # If user provided seed_solutions (from results.json), include them
    if seed_solutions:
        for s in seed_solutions:
            try:
                # attempt to sanitize and include
                if len(s) >= n + 1:
                    t = int(s[-1])
                    perm = [int(x) for x in s[:-1]]
                    perm_fixed = repair_perm_list(perm, n)
                    population.append({'solution': perm_fixed + [max(0, min(n - 1, t))]})
            except Exception:
                pass

    # fill to pop_size
    while len(population) < pop_size:
        base = random.choice(seed_orders)
        perm = base[:]
        for _ in range(random.randint(0, 6)):
            perm = inversion_mutation(perm)
            perm = swap_mutation(perm)
        t = random.randint(int(n * 0.18), int(n * 0.82))
        population.append({'solution': perm + [t]})

    start_gen = 0
    archive: List[Dict[str, Any]] = []
    best_hv = None
    best_combined_score = None

    # set worker globals in main thread for direct calls
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n

    # Stats
    hv_no_improve_gens = 0
    stagnation_counter = 0

    # prepare pool
    with multiprocessing.Pool(initializer=_init_worker, initargs=(adj_bits, n)) as pool:
        # evaluate initial population
        if 'score' not in population[0]:
            sols = [tuple(int(x) for x in p['solution']) for p in population]
            results = list(pool.map(eval_wrapper, sols))
            sol_to_score = {sol: score for sol, score in results}
            for p in population:
                p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))

        # build initial archive
        for p in population:
            archive = add_to_archive(archive, p, config.get('archive_max_size', DEFAULT_CONFIG['general']['archive_max_size']), n)
        hv = compute_hv_for_archive(archive, n)
        best_hv = hv
        best_combined_score = -hv
        persist_archive(archive, n, problem_id, best_combined_score)
        persist_best_solution(sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[0]['solution'] if archive else population[0]['solution'],
                              sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[0]['score'] if archive else population[0]['score'],
                              problem_id)
        print(f"Initial archive size: {len(archive)} | HV: {hv:.6f}")

        base_mutation = config.get('mutation_rate', DEFAULT_CONFIG['general']['mutation_rate'])

        # main generational loop
        for gen in tqdm(range(start_gen, config['generations']), desc="🧬 Evolving", initial=start_gen, total=config['generations']):
            # adaptive mutation
            mutation_rate = base_mutation
            if stagnation_counter >= DEFAULT_CONFIG['general']['stagnation_limit']:
                mutation_rate = min(0.99, base_mutation * DEFAULT_CONFIG['general']['mutation_boost_factor'])

            mating_pool = crowding_selection(population, pop_size)

            # generate offspring
            offspring_sols = []
            while len(offspring_sols) < pop_size:
                p1 = random.choice(mating_pool)
                p2 = random.choice(mating_pool)
                perm1 = list(p1['solution'][:-1])
                perm2 = list(p2['solution'][:-1])
                if random.random() < config.get('crossover_rate', DEFAULT_CONFIG['general']['crossover_rate']):
                    child_perm = pmx_crossover(perm1, perm2)
                else:
                    child_perm = perm1[:]
                if random.random() < mutation_rate:
                    if random.random() < 0.7:
                        child_perm = inversion_mutation(child_perm)
                    else:
                        child_perm = swap_mutation(child_perm)
                c_t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                if random.random() < 0.55:
                    c_t = max(0, min(n - 1, c_t + random.randint(-int(n*0.06), int(n*0.06))))
                offspring_sols.append(child_perm + [c_t])

            # local search parallel
            ls_args = [(sol, local_search_intensity) for sol in offspring_sols]
            improved_offspring = pool.map(local_search_worker, ls_args)

            # optionally exact reorder / sanitize improvements quick pass (light)
            for idx, sol in enumerate(improved_offspring):
                # ensure length
                if len(sol) != n + 1:
                    sol = repair_perm_list(sol[:-1] if len(sol) > 0 else [], n) + [sol[-1] if len(sol) > 0 else 0]
                    improved_offspring[idx] = sol

            # evaluate offspring in batch (parallel)
            eval_tuples = [tuple(int(x) for x in sol) for sol in improved_offspring]
            eval_results = list(pool.map(eval_wrapper, eval_tuples))
            offspring_pop = [{'solution': list(sol), 'score': score} for sol, score in eval_results]

            # combine and select
            population = crowding_selection(population + offspring_pop, pop_size)

            # immigrant injection for diversity occasionally
            if gen > 0 and (gen % DEFAULT_CONFIG['general']['immigrant_every'] == 0):
                retain = int(pop_size * (1.0 - DEFAULT_CONFIG['general']['immigrant_fraction']))
                if retain < 2: retain = 2
                pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                new_pop = pop_sorted[:retain]
                while len(new_pop) < pop_size:
                    perm = list(np.random.permutation(n))
                    t = random.randint(int(n * 0.18), int(n * 0.82))
                    for _ in range(random.randint(0, 3)):
                        perm = inversion_mutation(perm)
                        perm = swap_mutation(perm)
                    new_pop.append({'solution': perm + [t], 'score': (0, 999)})
                population = crowding_selection(new_pop, pop_size)

            # Elite intensification
            E = config.get('elite_count', DEFAULT_CONFIG['general']['elite_count'])
            elite_ls = config.get('elite_ls_multiplier', DEFAULT_CONFIG['general']['elite_ls_multiplier'])
            pop_sorted = sorted(population, key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
            elites = pop_sorted[:E]
            elite_args = []
            for e in elites:
                extra_int = int(max(1, local_search_intensity * elite_ls))
                elite_args.append((e['solution'], extra_int))
            if elite_args:
                improved_elites = pool.map(local_search_worker, elite_args)
                # targeted repair + eval
                repaired = []
                for sol in improved_elites:
                    if random.random() < 0.80:
                        sol = targeted_repair(sol, max_tries=120)
                    if random.random() < 0.25:
                        sol = iterated_greedy_rebuild(sol, k_min=18, k_max_pct=0.20)
                    repaired.append(sol)
                eval_tuples = [tuple(int(x) for x in sol) for sol in repaired]
                eval_results = list(pool.map(eval_wrapper, eval_tuples))
                for sol, score in eval_results:
                    population.append({'solution': list(sol), 'score': score})
                population = crowding_selection(population, pop_size)

            # Tabu intensification on top elites (smaller set)
            tabu_count = min(DEFAULT_CONFIG['general']['elite_tabu_count'], len(elites))
            if tabu_count > 0:
                tabu_args = []
                for e in elites[:tabu_count]:
                    tenure_range = DEFAULT_CONFIG['general']['elite_tabu_tenure']
                    iters = DEFAULT_CONFIG['general']['elite_tabu_iters']
                    tabu_args.append((e['solution'], iters, tenure_range))
                tabu_results = pool.map(tabu_search_worker, tabu_args)
                eval_tuples = [tuple(int(x) for x in sol) for sol in tabu_results]
                eval_results = list(pool.map(eval_wrapper, eval_tuples))
                for sol, score in eval_results:
                    population.append({'solution': list(sol), 'score': score})
                population = crowding_selection(population, pop_size)

            # merge into archive
            for p in population:
                archive = add_to_archive(archive, p, config.get('archive_max_size', DEFAULT_CONFIG['general']['archive_max_size']), n)

            # PLS occasionally
            if (gen + 1) % DEFAULT_CONFIG['general']['archive_pls_freq'] == 0:
                archive = pareto_local_search(archive, n, DEFAULT_CONFIG['general']['pareto_local_neighbor_samples'], pool=pool)

            # compute HV
            hv = compute_hv_for_archive(archive, n)
            combined_score = -hv
            improved = False
            if best_hv is None or hv > best_hv + 1e-12:
                best_hv = hv
                best_combined_score = combined_score
                persist_archive(archive, n, problem_id, best_combined_score)
                persist_best_solution(sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[0]['solution'],
                                      sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)[0]['score'],
                                      problem_id)
                improved = True
                stagnation_counter = 0
                hv_no_improve_gens = 0
                tqdm.write(f"✨ Gen {gen+1}: HV improved -> {hv:.6f} | archive_size {len(archive)}")
            else:
                stagnation_counter += 1
                hv_no_improve_gens += 1

            # restart if stagnation
            if DEFAULT_CONFIG['general']['restart_on_stagnation'] and hv_no_improve_gens >= DEFAULT_CONFIG['general']['restart_stagnation_threshold']:
                tqdm.write(f"🚨 HV stagnation for {hv_no_improve_gens} gens -> aggressive restart")
                population = aggressive_restart(population, archive, pop_size, n)
                sols = [tuple(int(x) for x in p['solution']) for p in population]
                results = list(pool.map(eval_wrapper, sols))
                sol_to_score = {sol: score for sol, score in results}
                for p in population:
                    p['score'] = sol_to_score.get(tuple(p['solution']), (0, 501))
                stagnation_counter = 0
                hv_no_improve_gens = 0

            # logging best single
            best_single = None
            for p in population:
                if best_single is None or dominates_internal(p['score'], best_single['score']):
                    best_single = p
            best_single_score = best_single['score'] if best_single else (0, 999)
            tqdm.write(f"Gen {gen+1}: best_single(size,width)={best_single_score} archive_size={len(archive)} stagn={stagnation_counter} mut={mutation_rate:.3f}")

            # checkpoint
            if (gen + 1) % config.get('checkpoint_interval', DEFAULT_CONFIG['general']['checkpoint_interval']) == 0:
                tqdm.write(f"\n💾 Saving checkpoint at generation {gen + 1}...")
                to_save = {'pop': population, 'gen': gen, 'archive': archive, 'best_hv': best_hv}
                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(to_save, f)
                except Exception:
                    pass

            if diagnostic:
                # break early for diagnostics
                break

    final_archive_sorted = sorted(archive, key=lambda a: (a['score'][0], -a['score'][1]), reverse=True)
    return [a['solution'] for a in final_archive_sorted[:20]]

# -------------------------
# CLI / main
# -------------------------
def create_submission_file(decision_vectors: List[List[int]], problem_id: str):
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}
    final_vectors = [[int(val) for val in vec] for vec in decision_vectors]
    submission = {
        "decisionVector": final_vectors,
        "problem": problem_name_map.get(problem_id, problem_id),
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename} with {len(decision_vectors)} solutions.")

def main():
    parser = argparse.ArgumentParser(description="Solver final - CPU-only memetic solver")
    parser.add_argument("--problem", type=str, default="easy", choices=list(PROBLEMS.keys()))
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--intensity", type=int, default=None, help="local search intensity per offspring")
    parser.add_argument("--seed_file", type=str, default=None, help="optional JSON results file with seed vectors")
    parser.add_argument("--no_seed", action="store_true", help="do not load any seed file")
    parser.add_argument("--diagnostic", action="store_true", help="run a single diagnostic generation and exit")
    args = parser.parse_args()

    problem_id = args.problem
    cfg = deepcopy(DEFAULT_CONFIG['general'])
    local_cfg = deepcopy(DEFAULT_CONFIG.get(problem_id, {}))
    # override defaults with CLI
    if args.pop_size is not None:
        local_cfg['pop_size'] = args.pop_size
    if args.generations is not None:
        local_cfg['generations'] = args.generations
    if args.intensity is not None:
        local_cfg['local_search_intensity'] = args.intensity
    cfg.update(local_cfg)

    graph_local = download_graph(problem_id)
    n, adj = load_graph_from_file(graph_local)
    seed_sols = None
    if (args.seed_file or (not args.no_seed and os.path.exists("results.json"))):
        sf = args.seed_file if args.seed_file else "results.json"
        try:
            with open(sf, "r") as f:
                j = json.load(f)
            decs = j.get("decisionVector", j.get("decision_vector", j.get("vectors", [])))
            seed_sols = decs if isinstance(decs, list) else None
            if seed_sols:
                print(f"✅ Loaded {len(seed_sols)} seed vectors from {sf}")
        except Exception:
            seed_sols = None

    random.seed(42)
    np.random.seed(42)

    start = time.time()
    final = memetic_algorithm(n, adj, cfg, problem_id, seed_solutions=seed_sols, diagnostic=args.diagnostic)
    elapsed = time.time() - start
    print(f"\n⏱️ Total time: {elapsed:.2f}s")
    create_submission_file(final, problem_id)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

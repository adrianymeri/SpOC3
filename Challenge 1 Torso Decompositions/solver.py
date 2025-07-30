# solver.py

import json
import random
import time
import math
import numpy as np
from typing import List
import urllib.request  # <--- FIXED: Added the missing import

# --- All of your original helper functions remain here ---

problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

def load_graph(problem_id: str) -> List[List[int]]:
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    edges = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b"#"): continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
    print(f"✅ Loaded graph with {len(edges)} edges")
    return edges

def evaluate_solution(decision_vector: List[int], n: int, adj_list: List[set]) -> List[float]:
    t = decision_vector[-1]
    size = n - t
    if size <= 0: return [0, 501, []]

    max_width = 0
    critical_nodes = []
    for i in range(t, n):
        node = decision_vector[i]
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
            critical_nodes = [node]
        elif current_width == max_width:
            critical_nodes.append(node)
    
    return [size, max_width if max_width < 500 else 501, critical_nodes]

def dominates(score1: List[float], score2: List[float]) -> bool:
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
           (score1[0] >= score2[0] and score1[1] < score2[1])

def smart_torso_shift(current: List[int], n: int, adj_list: List[set]) -> List[int]:
    neighbor = current[:]
    t = neighbor[-1]
    score = evaluate_solution(neighbor, n, adj_list)
    if score[1] > 400:
        new_t = min(n - 1, t + random.randint(5, 10))
    else:
        shift = int(n * 0.04) + 1
        new_t = max(0, min(n - 1, t + random.randint(-shift, shift)))
    neighbor[-1] = new_t
    return neighbor

def block_move(current: List[int], n: int, adj_list: List[set]) -> List[int]:
    neighbor = current[:]
    perm = neighbor[:-1]
    score = evaluate_solution(neighbor, n, adj_list)
    base_size = 3 if score[1] < 100 else 5
    block_size = random.randint(base_size, base_size + 2)
    if n > block_size:
        start = random.randint(0, n - block_size)
        block = perm[start:start + block_size]
        del perm[start:start + block_size]
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm
    return neighbor

neighbor_operators = [smart_torso_shift, block_move]

def initialize_solution(n: int, adj_list: List[set]) -> List[int]:
    perm = sorted(range(n), key=lambda x: -len(adj_list[x]))
    if random.random() < 0.5:
        perm.reverse()
    t = int(n * np.random.beta(1.5, 2.5))
    return perm + [t]

def hill_climbing(
    problem_id: str,
    max_iterations: int,
    num_restarts: int,
    cooling_rate: float,
    initial_temp: float
) -> List[dict]:
    """
    Main solver function, now parameterized for tuning.
    Returns a list of dictionaries, each representing a non-dominated solution.
    """
    edges = load_graph(problem_id)
    n = max(max(edge) for edge in edges) + 1 if edges else 0
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    pareto_front_solutions = []

    for _ in range(num_restarts):
        current = initialize_solution(n, adj_list)
        current_score = evaluate_solution(current, n, adj_list)[:2]
        
        best_local = current[:]
        best_local_score = current_score[:]
        
        T = initial_temp
        last_improvement = 0

        for iteration in range(max_iterations):
            T *= cooling_rate
            op = random.choice(neighbor_operators)
            neighbor = op(current, n, adj_list)
            
            neighbor_score = evaluate_solution(neighbor, n, adj_list)[:2]
            
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                size_gain = neighbor_score[0] - current_score[0]
                width_diff = current_score[1] - neighbor_score[1]
                delta = 2 * size_gain + width_diff
                if T > 1e-6:
                    accept_prob = math.exp(delta / T)
                    if random.random() < accept_prob:
                        accept = True

            if accept:
                current = neighbor
                current_score = neighbor_score
                if dominates(current_score, best_local_score):
                    best_local = current[:]
                    best_local_score = current_score[:]
                    last_improvement = iteration
            
            if iteration - last_improvement > 1500: # Early restart
                break
        
        pareto_front_solutions.append({'solution': best_local, 'score': best_local_score})

    # Filter for the final non-dominated front
    final_pareto_front = []
    for candidate in pareto_front_solutions:
        is_dominated = False
        # Create a copy of the list to iterate over while potentially modifying the original list indirectly
        for other in list(pareto_front_solutions):
            if dominates(other['score'], candidate['score']):
                is_dominated = True
                break
        if not is_dominated:
            final_pareto_front.append(candidate)
            
    # Remove duplicate solutions
    unique_solutions = []
    seen_solutions = set()
    for item in final_pareto_front:
        sol_tuple = tuple(item['solution'])
        if sol_tuple not in seen_solutions:
            unique_solutions.append(item)
            seen_solutions.add(sol_tuple)

    return unique_solutions

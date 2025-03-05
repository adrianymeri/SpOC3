import json
import random
import time
import math
import numpy as np
from typing import List, Set
import urllib.request

# Problem configurations
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# Precompute degrees for faster evaluation
def precompute_degrees(edges: List[List[int]], n: int) -> List[int]:
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees

def evaluate_solution(decision_vector: List[int], degrees: List[int]) -> List[float]:
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    size = n - t
    torso_nodes = decision_vector[t:]
    
    if not torso_nodes:
        return [0, float('inf'), []]
    
    max_width = max(degrees[node] for node in torso_nodes)
    critical_nodes = [node for node in torso_nodes if degrees[node] == max_width]
    return [size, max_width, critical_nodes]

# Enhanced neighborhood operators
def critical_node_replacement(current: List[int], degrees: List[int]) -> List[int]:
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    torso_nodes = perm[t:]
    
    if not torso_nodes:
        return current
    
    # Find highest degree node in torso
    max_degree = max(degrees[node] for node in torso_nodes)
    candidates = [node for node in torso_nodes if degrees[node] == max_degree]
    to_remove = random.choice(candidates)
    
    # Find best replacement in non-torso
    non_torso = [node for node in perm[:t] if degrees[node] < max_degree]
    if not non_torso:
        return current
    
    to_add = min(non_torso, key=lambda x: degrees[x])
    
    neighbor = current.copy()
    idx1 = perm.index(to_remove)
    idx2 = perm.index(to_add)
    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    return neighbor

def adaptive_torso_expansion(current: List[int], degrees: List[int]) -> List[int]:
    n = len(current) - 1
    t = current[-1]
    best_t = t
    best_width = float('inf')
    
    # Test expansion candidates with width check
    for delta in range(-5, 1):
        new_t = max(0, t + delta)
        if new_t == t:
            continue
        
        test_torso = current[new_t:-1]
        if not test_torso:
            continue
            
        current_width = max(degrees[node] for node in test_torso)
        if current_width < best_width:
            best_t = new_t
            best_width = current_width

    neighbor = current.copy()
    neighbor[-1] = best_t
    return neighbor

# Updated operator configuration
neighbor_operators = [
    critical_node_replacement,
    adaptive_torso_expansion,
    block_move,
    community_shuffle
]

def hill_climbing(edges: List[List[int]], max_iterations: int = 50000, num_restarts: int = 100) -> List[List[int]]:
    n = max(max(edge) for edge in edges) + 1
    degrees = precompute_degrees(edges, n)
    pareto_front = []
    global_best = [0, float('inf')]
    start_time = time.time()

    # Enhanced cooling parameters
    T0 = 10000.0
    cooling_rate = 0.9997

    for restart in range(num_restarts):
        current = initialize_solution(n, "community", edges, degrees)
        current_score = evaluate_solution(current, degrees)
        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0

        print(f"\n🌀 Restart {restart + 1}/{num_restarts} | Initial Score: {current_score[:2]}")

        for iteration in range(max_iterations):
            T *= cooling_rate

            # Adaptive operator selection
            if current_score[1] > 200:
                op = critical_node_replacement
            else:
                op = random.choice(neighbor_operators)

            neighbor = op(current, degrees)
            neighbor_score = evaluate_solution(neighbor, degrees)

            # Enhanced acceptance criteria
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                # Favor width reduction more aggressively
                width_diff = current_score[1] - neighbor_score[1]
                size_diff = neighbor_score[0] - current_score[0]
                delta = width_diff * 3 + size_diff
                accept_prob = math.exp(delta / (T + 1e-6))
                accept = random.random() < accept_prob

            if accept:
                current = neighbor
                current_score = neighbor_score
                update_operator_performance(op, True)

                if dominates(current_score, best_local_score):
                    best_local = current.copy()
                    best_local_score = current_score.copy()
                    last_improvement = iteration

                    if dominates(best_local_score, global_best):
                        global_best = best_local_score.copy()
                        print(f"\n🔥 NEW GLOBAL BEST @ Iter {iteration}: "
                              f"Size={global_best[0]} Width={global_best[1]} "
                              f"Time={time.time() - start_time:.1f}s")

            # Intensification every 500 iterations
            if iteration % 500 == 499:
                current = best_local.copy()
                current_score = best_local_score.copy()
                T = max(T, 1000)  # Reset temperature

            # Adaptive restart
            if iteration - last_improvement > 2000:
                break

        pareto_front.append((best_local, best_local_score))
        print(f"✅ Restart {restart + 1} Completed | Best: {best_local_score[:2]}")

    # Filter and return best solutions
    return sorted([sol for sol, _ in pareto_front],
                 key=lambda x: (-evaluate_solution(x, degrees)[0], 
                            evaluate_solution(x, degrees)[1]))[:5]

# Modified initialization with degree awareness
def initialize_solution(n: int, strategy: str, edges: List[List[int]], degrees: List[int]) -> List[int]:
    if strategy == "community":
        communities = detect_communities(list(range(n)), edges)
        communities.sort(key=lambda c: (-sum(degrees[node] for node in c)))
        permutation = []
        for comm in communities:
            comm_sorted = sorted(comm, key=lambda x: degrees[x])
            permutation.extend(comm_sorted)
    else:
        permutation = sorted(range(n), key=lambda x: -degrees[x])

    t = int(n * np.random.beta(1.2, 2.8))
    return permutation + [t]

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

def load_graph(problem_id: str) -> List[List[int]]:
    """Loads the graph data for the given problem ID."""
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    with urllib.request.urlopen(url) as f:
        edges = []
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
    print(f"✅ Loaded graph with {len(edges)} edges")
    return edges

def precompute_degrees(edges: List[List[int]], n: int) -> List[int]:
    """Precompute node degrees for faster evaluation"""
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees

def evaluate_solution(decision_vector: List[int], degrees: List[int]) -> List[float]:
    """Optimized evaluation using precomputed degrees"""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    size = n - t
    torso_nodes = decision_vector[t:]
    
    if not torso_nodes:
        return [0, float('inf'), []]
    
    max_width = max(degrees[node] for node in torso_nodes)
    critical_nodes = [node for node in torso_nodes if degrees[node] == max_width]
    return [size, max_width, critical_nodes]

# ------------------- ENHANCED NEIGHBOR OPERATORS -------------------
def degree_based_torso_reduction(current: List[int], degrees: List[int]) -> List[int]:
    """Swap high-degree torso nodes with low-degree non-torso nodes"""
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    torso_nodes = perm[t:]
    non_torso_nodes = perm[:t]
    
    if not torso_nodes or not non_torso_nodes:
        return current
    
    # Find highest degree node in torso
    max_degree = max(degrees[node] for node in torso_nodes)
    candidates = [node for node in torso_nodes if degrees[node] == max_degree]
    to_remove = random.choice(candidates)
    
    # Find lowest degree node in non-torso
    min_degree = min(degrees[node] for node in non_torso_nodes)
    replacements = [node for node in non_torso_nodes if degrees[node] == min_degree]
    to_add = random.choice(replacements)
    
    # Perform swap
    neighbor = current.copy()
    idx1 = perm.index(to_remove)
    idx2 = perm.index(to_add)
    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    return neighbor

def adaptive_torso_expansion(current: List[int], degrees: List[int]) -> List[int]:
    """Intelligently expand torso while monitoring width"""
    n = len(current) - 1
    t = current[-1]
    best_t = t
    best_width = float('inf')
    
    # Test potential expansion candidates
    for delta in range(-5, 0):
        new_t = max(0, t + delta)
        if new_t == t:
            continue
        
        test_torso = current[new_t:-1]
        current_width = max(degrees[node] for node in test_torso) if test_torso else 0
        
        if current_width < best_width:
            best_t = new_t
            best_width = current_width
    
    neighbor = current.copy()
    neighbor[-1] = best_t
    return neighbor

def community_aware_swap(current: List[int], communities: List[List[int]], degrees: List[int]) -> List[int]:
    """Swap nodes within communities to reduce torso width"""
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    neighbor = current.copy()
    
    for comm in communities:
        comm_nodes = [node for node in comm if node in perm]
        if len(comm_nodes) < 2:
            continue
        
        # Prioritize swapping high-degree torso nodes with low-degree non-torso nodes in same community
        torso_in_comm = [node for node in comm_nodes if perm.index(node) >= t]
        non_torso_in_comm = [node for node in comm_nodes if perm.index(node) < t]
        
        if torso_in_comm and non_torso_in_comm:
            to_remove = max(torso_in_comm, key=lambda x: degrees[x])
            to_add = min(non_torso_in_comm, key=lambda x: degrees[x])
            
            idx1 = perm.index(to_remove)
            idx2 = perm.index(to_add)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
    
    return neighbor

# ------------------- OPTIMIZED ALGORITHM CORE -------------------
def hill_climbing(edges: List[List[int]], max_iterations: int = 50000, num_restarts: int = 100) -> List[List[int]]:
    n = max(max(edge) for edge in edges) + 1
    degrees = precompute_degrees(edges, n)
    communities = detect_communities(list(range(n)), edges)
    pareto_front = []
    start_time = time.time()

    # Enhanced cooling schedule
    T0 = 5000.0
    cooling_rate = 0.9995
    restart_improvement_threshold = 50

    for restart in range(num_restarts):
        current = initialize_solution(n, "community", edges, degrees)
        current_score = evaluate_solution(current, degrees)[:2]
        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0
        
        print(f"\n🌀 Restart {restart + 1}/{num_restarts} | Initial Score: {current_score}")

        for iteration in range(max_iterations):
            T *= cooling_rate

            # Dynamic operator selection
            if random.random() < 0.7 and current_score[1] > 175:
                neighbor = degree_based_torso_reduction(current, degrees)
            else:
                neighbor = adaptive_torso_expansion(current, degrees)
            
            neighbor_score = evaluate_solution(neighbor, degrees)[:2]

            # Acceptance criteria with width focus
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                width_diff = current_score[1] - neighbor_score[1]
                size_diff = neighbor_score[0] - current_score[0]
                improvement = width_diff * 5 + size_diff  # Weight width reduction more
                accept_prob = math.exp(improvement / (T + 1e-6))
                accept = random.random() < accept_prob

            if accept:
                current = neighbor
                current_score = neighbor_score
                
                if dominates(current_score, best_local_score):
                    best_local = current.copy()
                    best_local_score = current_score.copy()
                    last_improvement = iteration

            # Community-aware intensification
            if iteration % 200 == 0:
                community_neighbor = community_aware_swap(current, communities, degrees)
                community_score = evaluate_solution(community_neighbor, degrees)[:2]
                if dominates(community_score, current_score):
                    current = community_neighbor
                    current_score = community_score

            # Early restart if stuck
            if iteration - last_improvement > restart_improvement_threshold:
                break

        # Update Pareto front
        pareto_front.append((best_local, best_local_score))
        print(f"✅ Restart {restart + 1} Completed | Best: {best_local_score}")

    # Filter Pareto front
    filtered = []
    for sol, score in pareto_front:
        if not any(dominates(other, score) for _, other in pareto_front):
            filtered.append(sol)
    
    return sorted(filtered,
                 key=lambda x: (-evaluate_solution(x, degrees)[0], 
                               evaluate_solution(x, degrees)[1]))[:3]

# ------------------- SUPPORTING FUNCTIONS -------------------
def detect_communities(nodes: List[int], edges: List[List[int]]) -> List[List[int]]:
    """Community detection using label propagation (optimized)"""
    adj_list = [[] for _ in range(len(nodes))]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    labels = list(range(len(nodes)))
    changed = True
    iterations = 0

    while changed and iterations < 10:
        changed = False
        order = list(range(len(nodes)))
        random.shuffle(order)
        
        for i in order:
            neighbor_labels = [labels[neighbor] for neighbor in adj_list[i]]
            if not neighbor_labels:
                continue
            
            # Count label frequencies
            freq = {}
            for label in neighbor_labels:
                freq[label] = freq.get(label, 0) + 1
            
            # Find most frequent label
            max_freq = max(freq.values())
            candidates = [label for label, count in freq.items() if count == max_freq]
            new_label = random.choice(candidates)
            
            if labels[i] != new_label:
                labels[i] = new_label
                changed = True
        iterations += 1

    # Form communities
    communities = {}
    for i, label in enumerate(labels):
        communities.setdefault(label, []).append(nodes[i])
    
    return list(communities.values())

def initialize_solution(n: int, strategy: str, edges: List[List[int]], degrees: List[int]) -> List[int]:
    """Degree-aware initialization with community structure"""
    permutation = sorted(range(n), key=lambda x: degrees[x])  # Start with low-degree nodes
    
    # Initialize threshold to include 30% lowest degree nodes in torso
    t = int(n * 0.3)
    return permutation + [t]

def create_submission_file(decision_vector, problem_id, filename="submission.json"):
    submission = {
        "decisionVector": [decision_vector],
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"📄 Created submission file: {filename}")

if __name__ == "__main__":
    random.seed(42)
    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    while problem_id not in problems:
        problem_id = input("❌ Invalid! Choose easy/medium/hard: ").lower()

    edges = load_graph(problem_id)
    start_time = time.time()
    solutions = hill_climbing(edges)
    
    print(f"\n⏱️ Optimization completed in {time.time() - start_time:.2f} seconds")
    best_solution = min(solutions, key=lambda x: (x[1], -x[0]))
    
    create_submission_file(best_solution, problem_id, "optimized_solution.json")

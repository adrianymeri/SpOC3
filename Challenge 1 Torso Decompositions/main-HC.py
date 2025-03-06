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

def precompute_degrees(edges: List[List[int]]) -> List[int]:
    """Precompute node degrees for faster evaluation"""
    n = max(max(u, v) for u, v in edges) + 1
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

def dominates(score1: List[float], score2: List[float]) -> bool:
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
           (score1[0] >= score2[0] and score1[1] < score2[1])

# ------------------- OPTIMIZED NEIGHBORHOOD OPERATORS -------------------
def smart_torso_shift(current: List[int], degrees: List[int]) -> List[int]:
    n = len(current) - 1
    t = current[-1]
    current_score = evaluate_solution(current, degrees)
    
    # Dynamic shift size based on current width
    if current_score[1] > 300:
        shift = random.randint(5, 15)
    else:
        shift = random.randint(1, 5)
        
    new_t = t + shift if current_score[1] > 200 else t - shift
    new_t = max(0, min(n-1, new_t))
    
    neighbor = current.copy()
    neighbor[-1] = new_t
    return neighbor

def block_move(current: List[int], degrees: List[int]) -> List[int]:
    neighbor = current.copy()
    n = len(neighbor) - 1
    current_score = evaluate_solution(current, degrees)
    
    # Adaptive block size based on problem size
    base_size = max(2, int(n * 0.01))
    block_size = random.randint(base_size, base_size + 3)
    
    start = random.randint(0, n - block_size)
    insert_pos = random.choice([
        random.randint(0, n - block_size),
        random.randint(max(0, start - 5), min(n - block_size, start + 5))
    ])
    
    if insert_pos != start:
        block = neighbor[start:start + block_size]
        del neighbor[start:start + block_size]
        neighbor[insert_pos:insert_pos] = block
    return neighbor

def critical_swap(current: List[int], degrees: List[int]) -> List[int]:
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    score_data = evaluate_solution(current, degrees)
    
    if score_data[1] >= 500 or len(score_data[2]) == 0:
        return current
    
    # Find top 5% critical nodes
    critical_nodes = sorted(score_data[2], key=lambda x: -degrees[x])[:max(1, len(score_data[2])//20)]
    to_remove = random.choice(critical_nodes)
    
    # Find best possible replacement
    non_torso = [node for node in perm[:t] if degrees[node] < score_data[1]]
    if not non_torso:
        return current
    
    replacement = min(non_torso, key=lambda x: degrees[x])
    
    neighbor = current.copy()
    crit_idx = perm.index(to_remove)
    rep_idx = perm.index(replacement)
    neighbor[crit_idx], neighbor[rep_idx] = neighbor[rep_idx], neighbor[crit_idx]
    return neighbor

def community_shuffle(current: List[int], edges: List[List[int]]) -> List[int]:
    n = len(current) - 1
    perm = current[:-1]
    
    # Use efficient community detection
    communities = detect_communities(perm, edges)
    neighbor = current.copy()
    new_perm = neighbor[:-1]
    
    for comm in communities:
        indices = [i for i, node in enumerate(new_perm) if node in comm]
        if len(indices) >= 2:
            # Swap random pair within community
            i, j = random.sample(indices, 2)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    
    neighbor[:-1] = new_perm
    return neighbor

def detect_communities(nodes: List[int], edges: List[List[int]]) -> List[List[int]]:
    """Fast approximate community detection using label propagation"""
    node_map = {node: idx for idx, node in enumerate(nodes)}
    adj_list = [[] for _ in nodes]
    
    # Build local adjacency list
    for u, v in edges:
        if u in node_map and v in node_map:
            adj_list[node_map[u]].append(node_map[v])
            adj_list[node_map[v]].append(node_map[u])
    
    labels = list(range(len(nodes)))
    changed = True
    iterations = 0
    
    while changed and iterations < 5:
        changed = False
        order = random.sample(range(len(nodes)), len(nodes))
        
        for i in order:
            if not adj_list[i]:
                continue
                
            # Count neighboring labels
            counts = {}
            for neighbor in adj_list[i]:
                counts[labels[neighbor]] = counts.get(labels[neighbor], 0) + 1
            
            # Find most frequent label
            max_count = max(counts.values(), default=0)
            candidates = [lbl for lbl, cnt in counts.items() if cnt == max_count]
            new_label = random.choice(candidates)
            
            if labels[i] != new_label:
                labels[i] = new_label
                changed = True
                
        iterations += 1
    
    # Map back to original nodes
    communities = {}
    for idx, lbl in enumerate(labels):
        communities.setdefault(lbl, []).append(nodes[idx])
    
    return list(communities.values())

# Initialize operator tracking
neighbor_operators = [
    smart_torso_shift,
    block_move,
    critical_swap,
    community_shuffle
]

op_weights = [3.0, 2.0, 4.0, 1.5]  # Initial weights based on operator effectiveness

def update_operator_performance(op_idx: int, success: bool):
    """Adaptive weight adjustment with momentum"""
    global op_weights
    learning_rate = 0.15
    if success:
        op_weights[op_idx] *= (1 + learning_rate)
    else:
        op_weights[op_idx] *= (1 - learning_rate)
    
    # Softmax normalization
    total = sum(math.exp(w) for w in op_weights)
    op_weights = [math.exp(w)/total for w in op_weights]

def initialize_solution(n: int, strategy: str, edges: List[List[int]]) -> List[int]:
    """Improved initialization with hybrid strategy"""
    if strategy == "hybrid":
        # Combine community and degree information
        communities = detect_communities(list(range(n)), edges)
        comm_sizes = {i: len(comm) for i, comm in enumerate(communities)}
        
        permutation = []
        for comm in sorted(communities, key=lambda c: (-comm_sizes[id(c)], -sum(1 for _ in c))):
            # Add nodes sorted by degree within community
            permutation.extend(sorted(comm, key=lambda x: -precompute_degrees(edges)[x]))
        
        # Initial torso size based on community structure
        t = n - int(n * 0.25)
    elif strategy == "degree":
        degrees = precompute_degrees(edges)
        permutation = sorted(range(n), key=lambda x: -degrees[x])
        t = n - int(n * 0.3)
    else:
        permutation = list(range(n))
        random.shuffle(permutation)
        t = n // 2
    
    return permutation + [t]

def hill_climbing(edges: List[List[int]], max_iterations: int = 40000, num_restarts: int = 150) -> List[List[int]]:
    n = max(max(u, v) for u, v in edges) + 1
    degrees = precompute_degrees(edges)
    pareto_front = []
    global_best = [0, float('inf')]
    start_time = time.time()
    
    # Adaptive cooling parameters
    T0 = 5000.0
    cooling_rate = 0.9995
    
    for restart in range(num_restarts):
        strategy = "hybrid" if restart % 3 == 0 else "degree"
        current = initialize_solution(n, strategy, edges)
        current_score = evaluate_solution(current, degrees)
        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0
        
        print(f"\n🌀 Restart {restart+1}/{num_restarts} | Initial: {current_score[:2]}")
        
        for iteration in range(max_iterations):
            T *= cooling_rate
            
            # Select operator based on adaptive weights
            op_idx = random.choices(range(len(neighbor_operators)), weights=op_weights)[0]
            op = neighbor_operators[op_idx]
            
            # Generate neighbor solution
            if op == community_shuffle:
                neighbor = op(current, edges)
            else:
                neighbor = op(current, degrees)
            
            neighbor_score = evaluate_solution(neighbor, degrees)
            
            # Acceptance criteria
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                # Energy difference calculation favoring width reduction
                delta_width = current_score[1] - neighbor_score[1]
                delta_size = neighbor_score[0] - current_score[0]
                accept_prob = math.exp((delta_width * 3 + delta_size) / T)
                accept = random.random() < accept_prob
            
            if accept:
                current = neighbor
                current_score = neighbor_score
                update_operator_performance(op_idx, True)
                
                if dominates(current_score, best_local_score):
                    best_local = current.copy()
                    best_local_score = current_score.copy()
                    last_improvement = iteration
                    
                    if dominates(best_local_score, global_best):
                        global_best = best_local_score.copy()
                        print(f"🔥 NEW GLOBAL BEST @ Iter {iteration}: "
                              f"Size={global_best[0]} Width={global_best[1]}")
            
            # Intensification phase
            if iteration % 500 == 499:
                # Local search around best solution
                for _ in range(50):
                    candidate = critical_swap(best_local, degrees)
                    candidate_score = evaluate_solution(candidate, degrees)
                    if dominates(candidate_score, best_local_score):
                        best_local = candidate.copy()
                        best_local_score = candidate_score.copy()
            
            # Adaptive restart
            if iteration - last_improvement > 2000:
                break
        
        pareto_front.append((best_local, best_local_score))
        print(f"✅ Restart {restart+1} Completed | Best: {best_local_score[:2]}")
    
    # Extract Pareto front
    filtered = []
    for sol, score in pareto_front:
        if not any(dominates(other, score) for _, other in pareto_front):
            filtered.append(sol)
    
    return sorted(filtered,
                 key=lambda x: (-evaluate_solution(x, degrees)[0],
                            evaluate_solution(x, degrees)[1]))[:5]

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

    # Select best solution considering both size and width
    best_solution = max(solutions,
                       key=lambda x: (evaluate_solution(x, precompute_degrees(edges))[0] * 1000 -
                                   evaluate_solution(x, precompute_degrees(edges))[1])
    
    create_submission_file(best_solution, problem_id, "optimized_solution.json")

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

# ------------------- NEIGHBORHOOD OPERATORS -------------------
def smart_torso_shift(current: List[int], degrees: List[int]) -> List[int]:
    n = len(current) - 1
    t = current[-1]
    current_score = evaluate_solution(current, degrees)
    new_t = t

    if current_score[1] > 400:
        new_t = min(n - 1, t + random.randint(5, 10))
    else:
        shift = int(n * 0.04) + 1
        new_t = max(0, min(n - 1, t + random.randint(-shift, shift)))

    neighbor = current.copy()
    neighbor[-1] = new_t
    return neighbor

def block_move(current: List[int], degrees: List[int]) -> List[int]:
    neighbor = current.copy()
    n = len(neighbor) - 1
    current_score = evaluate_solution(current, degrees)

    base_size = 3 if current_score[1] < 100 else 5
    block_size = random.randint(base_size, base_size + 2)

    start = random.randint(0, n - block_size)
    block = neighbor[start:start + block_size]
    insert_pos = random.randint(0, n - block_size)

    if insert_pos != start:
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

    critical_node = random.choice(score_data[2])
    non_torso = [node for node in perm[:t] if degrees[node] < score_data[1]]

    if not non_torso:
        return current

    replacement = min(non_torso, key=lambda x: degrees[x])
    neighbor = current.copy()
    crit_idx = perm.index(critical_node)
    rep_idx = perm.index(replacement)
    neighbor[crit_idx], neighbor[rep_idx] = neighbor[rep_idx], neighbor[crit_idx]
    return neighbor

def community_shuffle(current: List[int], edges: List[List[int]]) -> List[int]:
    n = len(current) - 1
    perm = current[:-1]
    communities = detect_communities(perm, edges)
    neighbor = current.copy()
    new_perm = neighbor[:-1]

    for comm in communities:
        indices = [i for i, node in enumerate(new_perm) if node in comm]
        if len(indices) < 2:
            continue

        community_nodes = [new_perm[i] for i in indices]
        random.shuffle(community_nodes)
        for i, idx in enumerate(indices):
            new_perm[idx] = community_nodes[i]

    neighbor[:-1] = new_perm
    return neighbor

def detect_communities(nodes: List[int], edges: List[List[int]]) -> List[List[int]]:
    adj_list = [[] for _ in range(len(nodes))]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    labels = list(range(len(nodes)))
    changed = True

    while changed:
        changed = False
        order = list(range(len(nodes)))
        random.shuffle(order)

        for i in order:
            counts = {}
            for neighbor in adj_list[i]:
                counts[labels[neighbor]] = counts.get(labels[neighbor], 0) + 1

            if counts:
                max_label = max(counts, key=lambda k: (counts[k], -k))
                if labels[i] != max_label:
                    labels[i] = max_label
                    changed = True

    communities = {}
    for i, label in enumerate(labels):
        communities.setdefault(label, []).append(nodes[i])
    return list(communities.values())

# Initialize operator tracking
neighbor_operators = [
    smart_torso_shift,
    block_move,
    critical_swap,
    community_shuffle
]

op_scores = {op: 1.0 for op in neighbor_operators}

def update_operator_performance(op, success: bool):
    if success:
        op_scores[op] = min(op_scores[op] * 1.25, 15.0)
    else:
        op_scores[op] = max(op_scores[op] * 0.85, 0.05)

    total = sum(math.exp(s) for s in op_scores.values())
    for o in op_scores:
        op_scores[o] = math.exp(op_scores[o]) / total

def initialize_solution(n: int, strategy: str, edges: List[List[int]]) -> List[int]:
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    if strategy == "community":
        communities = detect_communities(list(range(n)), edges)
        communities.sort(key=lambda c: (-len(c), -sum(len(adj_list[n]) for n in c)))
        permutation = []
        for comm in communities:
            comm_sorted = sorted(comm, key=lambda x: -len(adj_list[x]))
            permutation.extend(comm_sorted)
    elif strategy == "degree":
        permutation = sorted(range(n), key=lambda x: -len(adj_list[x]))
    else:
        permutation = list(range(n))
        random.shuffle(permutation)

    t = int(n * np.random.beta(1.5, 2.5))
    return permutation + [t]

def hill_climbing(edges: List[List[int]], max_iterations: int = 25000, num_restarts: int = 250) -> List[List[int]]:
    n = max(max(edge) for edge in edges) + 1
    degrees = precompute_degrees(edges)
    pareto_front = []
    global_best = [0, float('inf')]
    start_time = time.time()

    T0 = 3000.0
    cooling_rate = 0.9992

    for restart in range(num_restarts):
        init_strategy = random.choice(["community", "degree"])
        current = initialize_solution(n, init_strategy, edges)
        current_score = evaluate_solution(current, degrees)
        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0

        print(f"\n🌀 Restart {restart + 1}/{num_restarts} | Initial: {current_score[:2]}")

        for iteration in range(max_iterations):
            T *= cooling_rate

            op = random.choices(neighbor_operators, weights=[op_scores[o] for o in neighbor_operators])[0]
            neighbor = op(current, degrees) if op != community_shuffle else op(current, edges)
            neighbor_score = evaluate_solution(neighbor, degrees)

            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                size_gain = neighbor_score[0] - current_score[0]
                width_diff = current_score[1] - neighbor_score[1]
                delta = 2 * size_gain + width_diff
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
                        print(f"\n🔥 NEW BEST @ Iter {iteration}: Size={global_best[0]} Width={global_best[1]}")

            if iteration % 300 == 299:
                for _ in range(25):
                    candidate = random.choice(neighbor_operators)(best_local, degrees) 
                    candidate_score = evaluate_solution(candidate, degrees)
                    if dominates(candidate_score, best_local_score):
                        best_local = candidate.copy()
                        best_local_score = candidate_score.copy()

            if iteration - last_improvement > 1500:
                break

        pareto_front.append((best_local, best_local_score))
        print(f"✅ Restart {restart + 1} Completed | Best: {best_local_score[:2]}")

    filtered = []
    for sol, score in pareto_front:
        if not any(dominates(other, score) for _, other in pareto_front):
            filtered.append(sol)

    return sorted(filtered, key=lambda x: (-evaluate_solution(x, degrees)[0], evaluate_solution(x, degrees)[1]))[:5]

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

    for idx, sol in enumerate(solutions):
        create_submission_file(sol, problem_id, f"best_solution_{idx + 1}.json")

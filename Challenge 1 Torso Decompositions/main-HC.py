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


def evaluate_solution(decision_vector: List[int], edges: List[List[int]]) -> List[float]:
    """Optimized evaluation with early termination and width tracking"""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    size = n - t

    # Build adjacency list
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)

    max_width = 0
    critical_nodes = []
    for i in range(t, n):
        node = decision_vector[i]
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
            critical_nodes = [node]
            if max_width >= 500:
                return [size, 501, critical_nodes]
        elif current_width == max_width:
            critical_nodes.append(node)

    return [size, max_width, critical_nodes]


# ------------------- ENHANCED NEIGHBOR OPERATORS -------------------
def smart_torso_shift(current: List[int], edges: List[List[int]]) -> List[int]:
    """Adaptive threshold adjustment with width awareness"""
    n = len(current) - 1
    t = current[-1]
    current_score = evaluate_solution(current, edges)
    new_t = t

    if current_score[1] > 400:
        # Aggressive reduction for high width
        new_t = min(n - 1, t + random.randint(5, 10))
    else:
        # Balanced exploration
        shift = int(n * 0.04) + 1
        new_t = max(0, min(n - 1, t + random.randint(-shift, shift)))

    neighbor = current.copy()
    neighbor[-1] = new_t
    return neighbor


def block_move(current: List[int], edges: List[List[int]]) -> List[int]:
    """Move a block of nodes with size based on current width"""
    neighbor = current.copy()
    n = len(neighbor) - 1
    current_score = evaluate_solution(current, edges)

    # Dynamic block size based on width
    base_size = 3 if current_score[1] < 100 else 5
    block_size = random.randint(base_size, base_size + 2)

    start = random.randint(0, n - block_size)
    block = neighbor[start:start + block_size]
    insert_pos = random.randint(0, n - block_size)

    if insert_pos != start:
        del neighbor[start:start + block_size]
        neighbor[insert_pos:insert_pos] = block
    return neighbor


def critical_swap(current: List[int], edges: List[List[int]]) -> List[int]:
    """Swap critical high-degree nodes with low-degree alternatives"""
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    score_data = evaluate_solution(current, edges)

    if score_data[1] >= 500 or len(score_data[2]) == 0:
        return current

    # Build degree list
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    degrees = [len(nb) for nb in adj_list]

    # Find swap candidates
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


def greedy_expand(current: List[int], edges: List[List[int]]) -> List[int]:
    """Systematically test potential expansions of the torso"""
    n = len(current) - 1
    t = current[-1]
    original_score = evaluate_solution(current, edges)
    best_t = t
    best_score = original_score

    # Test expansion candidates
    for delta in range(-3, 0):
        new_t = max(0, t + delta)
        if new_t == t:
            continue

        test_sol = current.copy()
        test_sol[-1] = new_t
        test_score = evaluate_solution(test_sol, edges)

        if (test_score[0] > best_score[0] or
                (test_score[0] == best_score[0] and test_score[1] < best_score[1])):
            best_t = new_t
            best_score = test_score

    neighbor = current.copy()
    neighbor[-1] = best_t
    return neighbor


def community_shuffle(current: List[int], edges: List[List[int]]) -> List[int]:
    """Shuffle nodes within detected communities"""
    n = len(current) - 1
    perm = current[:-1]

    # Detect communities using label propagation
    communities = detect_communities(perm, edges)

    neighbor = current.copy()
    new_perm = neighbor[:-1]

    for comm in communities:
        # Find all indices of community members in the permutation
        indices = [i for i, node in enumerate(new_perm) if node in comm]
        if len(indices) < 2:
            continue

        # Extract and shuffle community nodes
        community_nodes = [new_perm[i] for i in indices]
        random.shuffle(community_nodes)

        # Replace nodes while maintaining positions
        for i, idx in enumerate(indices):
            new_perm[idx] = community_nodes[i]

    neighbor[:-1] = new_perm
    return neighbor


def detect_communities(nodes: List[int], edges: List[List[int]]) -> List[List[int]]:
    """Community detection using label propagation"""
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
    greedy_expand,
    community_shuffle
]

op_scores = {op: 1.0 for op in neighbor_operators}
op_usage = {op: 0 for op in neighbor_operators}


def dominates(score1: List[float], score2: List[float]) -> bool:
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
        (score1[0] >= score2[0] and score1[1] < score2[1])


def update_operator_performance(op, success: bool):
    """Adaptive weight updates with momentum"""
    if success:
        op_scores[op] = min(op_scores[op] * 1.25, 15.0)
    else:
        op_scores[op] = max(op_scores[op] * 0.85, 0.05)

    # Softmax normalization
    total = sum(math.exp(s) for s in op_scores.values())
    for o in op_scores:
        op_scores[o] = math.exp(op_scores[o]) / total


def initialize_solution(n: int, strategy: str, edges: List[List[int]]) -> List[int]:
    """Advanced initialization with community detection"""
    # Build adjacency list
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    if strategy == "community":
        communities = detect_communities(list(range(n)), edges)
        communities.sort(key=lambda c: (-len(c), -sum(len(adj_list[n]) for n in c)))
        permutation = []
        for comm in communities:
            # Sort community nodes by degree (high to low)
            comm_sorted = sorted(comm, key=lambda x: -len(adj_list[x]))
            permutation.extend(comm_sorted)
    elif strategy == "degree":
        permutation = sorted(range(n), key=lambda x: -len(adj_list[x]))
    elif strategy == "reverse_degree":
        permutation = sorted(range(n), key=lambda x: len(adj_list[x]))
    else:
        permutation = list(range(n))
        random.shuffle(permutation)

    # Initialize threshold with exploration bias
    t = int(n * np.random.beta(1.5, 2.5))  # Favor larger initial torso
    return permutation + [t]


def hill_climbing(edges: List[List[int]], max_iterations: int = 25000, num_restarts: int = 250) -> List[List[int]]:
    n = max(max(edge) for edge in edges) + 1
    pareto_front = []
    global_best = [0, float('inf')]
    start_time = time.time()

    # Adaptive cooling parameters
    T0 = 3000.0
    cooling_rate = 0.9992

    for restart in range(num_restarts):
        # Initialize with progress tracking
        init_strategy = random.choice(["community", "degree", "reverse_degree"])
        current = initialize_solution(n, init_strategy, edges)
        current_score = evaluate_solution(current, edges)[:2]

        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0
        tabu = set()
        tabu.add(tuple(current))

        print(f"\n🌀 Restart {restart + 1}/{num_restarts} | Initial Score: {current_score}")

        for iteration in range(max_iterations):
            T *= cooling_rate

            # Dynamic operator selection
            op = random.choices(
                neighbor_operators,
                weights=[op_scores[o] for o in neighbor_operators]
            )[0]

            neighbor = op(current, edges)
            if tuple(neighbor) in tabu:
                continue

            neighbor_score = evaluate_solution(neighbor, edges)[:2]

            # Update tabu list (adaptive size)
            tabu.add(tuple(neighbor))
            if len(tabu) > 50 + 10 * (restart % 5):
                tabu.pop()

            # Enhanced acceptance criteria
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                # Weighted acceptance favoring size
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
                        print(f"\n🔥 NEW GLOBAL BEST @ Iter {iteration}: "
                              f"Size={global_best[0]} Width={global_best[1]} "
                              f"Time={time.time() - start_time:.1f}s")
            else:
                update_operator_performance(op, False)

            # Intensification every 300 iterations
            if iteration % 300 == 299:
                for _ in range(25):
                    candidate = random.choice(neighbor_operators)(best_local, edges)
                    candidate_score = evaluate_solution(candidate, edges)[:2]
                    if dominates(candidate_score, best_local_score):
                        best_local = candidate.copy()
                        best_local_score = candidate_score.copy()

            # Progress reporting
            if iteration % 200 == 0:
                print(f"Iter {iteration:5d} | Temp: {T:7.1f} | "
                      f"Current: {current_score} | Best: {best_local_score}")

            # Early restart if stuck
            if iteration - last_improvement > 1500:
                print(f"🔄 Early restart at iteration {iteration}")
                break

        # Update Pareto front
        pareto_front.append((best_local, best_local_score))
        print(f"\n✅ Restart {restart + 1} Completed | Best: {best_local_score} | "
              f"Time: {time.time() - start_time:.1f}s")

    # Filter and sort Pareto front
    filtered = []
    for sol, score in pareto_front:
        if not any(dominates(other, score) for _, other in pareto_front):
            filtered.append(sol)

    return sorted(filtered,
                  key=lambda x: (-evaluate_solution(x, edges)[0], evaluate_solution(x, edges)[1]))[:5]


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

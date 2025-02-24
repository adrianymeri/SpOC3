import json
import random
import time
import math
import numpy as np
from typing import List, Set, Dict
import urllib.request
from collections import defaultdict

# Problem configurations
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}


def load_graph(problem_id: str) -> List[List[int]]:
    """Loads the graph data."""
    url = problems[problem_id]
    print(f"📥 Loading graph data from: {url}")
    edges = []
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
    print(f"✅ Loaded graph with {len(edges)} edges")
    return edges


def evaluate_solution(decision_vector: List[int], adj_list: List[Set[int]]) -> List[float]:
    """Optimized evaluation with early termination."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    size = n - t

    max_width = 0
    critical_nodes = []
    for i in range(t, n):
        node = decision_vector[i]
        current_width = len(adj_list[node])
        if current_width > max_width:
            max_width = current_width
            critical_nodes = [node]
            if max_width >= 500:
                return [size, 501, critical_nodes]  # Early termination
        elif current_width == max_width:
            critical_nodes.append(node)

    return [size, max_width, critical_nodes]


def build_adjacency_list(n: int, edges: List[List[int]]) -> List[Set[int]]:
    """Builds an adjacency list from the edge list."""
    adj_list = [set() for _ in range(n)]
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    return adj_list


def louvain_community_detection(adj_list: List[Set[int]]) -> List[List[int]]:
    """Community detection using the Louvain algorithm."""

    n = len(adj_list)
    communities = {i: [i] for i in range(n)}  # Initially, each node is its own community
    node_to_comm = {i: i for i in range(n)}  # Maps node to community ID
    degrees = [len(adj) for adj in adj_list]
    m = sum(degrees) / 2  # Total number of edges (divided by 2)

    def modularity_gain(node: int, comm_id: int) -> float:
        """Calculates the modularity gain of moving node to comm_id."""
        comm = communities[comm_id]
        k_i = degrees[node]
        k_i_in = sum(1 for neighbor in adj_list[node] if node_to_comm[neighbor] == comm_id)
        sigma_tot = sum(degrees[j] for j in comm)
        return (k_i_in - (sigma_tot * k_i) / (2 * m)) / m

    def move_node(node: int, new_comm_id: int) -> None:
        """Moves a node between communities and updates mappings."""
        old_comm_id = node_to_comm[node]
        communities[old_comm_id].remove(node)
        communities[new_comm_id].append(node)
        node_to_comm[node] = new_comm_id

        # Clean up empty communities: Important for efficiency
        if not communities[old_comm_id]:
            del communities[old_comm_id]

    improved = True
    while improved:
        improved = False
        for node in random.sample(range(n), n):  # Important: Iterate in random order
            best_comm = node_to_comm[node]
            best_gain = 0

            neighbor_comms = {node_to_comm[neighbor] for neighbor in adj_list[node] if
                              node_to_comm[neighbor] != node_to_comm[node]}  # only check neighbor communities

            for comm_id in neighbor_comms:
                gain = modularity_gain(node, comm_id)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = comm_id

            if best_comm != node_to_comm[node]:
                move_node(node, best_comm)
                improved = True
    return list(communities.values())


def community_shuffle(current: List[int], adj_list: List[Set[int]], degrees=None) -> List[int]:
    """Shuffle nodes within detected communities (Louvain)."""
    n = len(current) - 1
    perm = current[:-1]
    communities = louvain_community_detection([adj_list[node] for node in perm])  # Communities of nodes indexes

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


def smart_torso_shift(current: List[int], adj_list, degrees: List[int]) -> List[int]:
    """Adaptive threshold adjustment with width awareness"""
    n = len(current) - 1
    t = current[-1]
    current_score = evaluate_solution(current, adj_list)
    new_t = t
    neighbor = current.copy()

    if current_score[1] > 400:
        # Aggressive reduction for high width
        new_t = min(n - 1, t + random.randint(5, 10))
    else:
        # Try expansion and contraction
        if (random.random() < 0.7):
            # Contraction. Nodes in torso. Prioritize nodes that reduce width
            candidates = []

            for i in range(max(0, t - 5), min(n - 1, t + 5)):  # Limit search to nearby t
                node_to_remove = current[i]

                connections_in_torso = 0
                for neighbor_node in adj_list[node_to_remove]:
                    if current.index(neighbor_node) >= t:
                        connections_in_torso += 1

                if (t < n - 1):
                    candidates.append((node_to_remove, len(
                        adj_list[node_to_remove]) - 2 * connections_in_torso))  # Heuristic, lower is better

            if (len(candidates) > 0):
                candidates.sort(key=lambda x: x[1])
                best_node_to_remove, _ = candidates[0]  # take best

                # Swap with last

                best_index = current.index(best_node_to_remove)
                last_index = n - 1
                neighbor[best_index], neighbor[last_index] = neighbor[last_index], neighbor[best_index]

                new_t = max(0, min(t + 1, n - 1))  # Increase t
        else:
            # Expansion

            candidates = []
            # Nodes not in torso.  Prioritize nodes that increase connectivity to torso.
            for i in range(max(0, t - 5), min(n - 1, t + 5)):  # Limit search to nearby t
                node_to_add = current[i]

                connections_in_torso = 0
                for neighbor_node in adj_list[node_to_add]:
                    if current.index(neighbor_node) >= t:
                        connections_in_torso += 1

                candidates.append((node_to_add, -connections_in_torso))  # Heuristic, lower its better

            if (len(candidates) > 0):
                candidates.sort(key=lambda x: x[1])
                best_node_to_add, _ = candidates[0]  # take best

                best_index = current.index(best_node_to_add)
                first_index = t
                neighbor[best_index], neighbor[first_index] = neighbor[first_index], neighbor[best_index]
                new_t = max(0, min(t - 1, n - 1))

    neighbor[-1] = new_t
    return neighbor


def block_move(current: List[int], adj_list: List[Set[int]], degrees: List[int]) -> List[int]:
    """Move a block of nodes."""
    neighbor = current.copy()
    n = len(neighbor) - 1
    t = neighbor[-1]
    current_score = evaluate_solution(current, adj_list)

    # Dynamic block size based on width AND position: larger blocks if width is low
    base_size = 3 if current_score[1] < 200 else 2  # Smaller blocks if high width
    block_size = random.randint(base_size, base_size + 1)

    # Prioritize moving blocks *around* the torso boundary
    start_range_start = max(0, t - block_size)
    start_range_end = min(n - block_size, t + block_size)

    if start_range_start >= start_range_end:  # Handle edge cases
        start = random.randint(0, n - block_size)
    else:
        start = random.randint(start_range_start, start_range_end)  # Bias to move t

    block = neighbor[start:start + block_size]

    # Insert at a different position, also biased towards the torso boundary:
    insert_range_start = max(0, t - 5)
    insert_range_end = min(n - block_size, t + 5)
    if insert_range_start >= insert_range_end:
        insert_pos = random.randint(0, n - block_size)
    else:
        insert_pos = random.randint(insert_range_start, insert_range_end)

    if insert_pos != start:
        del neighbor[start:start + block_size]
        neighbor[insert_pos:insert_pos] = block  # Insert the block

    return neighbor


def critical_swap(current: List[int], adj_list: List[Set[int]], degrees: List[int]) -> List[int]:
    """Swap critical high-degree nodes with low-degree alternatives."""
    n = len(current) - 1
    t = current[-1]
    perm = current[:-1]
    score_data = evaluate_solution(current, adj_list)
    neighbor = current.copy()

    if score_data[1] >= 500 or len(score_data[2]) == 0:
        return current  # No critical nodes or width too high
    if (len(score_data[2]) == 0):
        return neighbor

    # Find swap candidates
    critical_node = random.choice(score_data[2])

    # Prioritize nodes close to the torso boundary:
    candidates = []
    for i in range(max(0, t - 5), min(t, n)):  # Look at the 5 first of the torso
        node = perm[i]
        if degrees[node] < score_data[1]:
            # Calculate *potential* width reduction:
            current_width = len(adj_list[critical_node])
            potential_width = len(adj_list[node])
            candidates.append((node, current_width - potential_width))  # Higher gain is better.

    if not candidates:
        return current

    # Select the replacement that offers the *best* width reduction
    candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by potential gain
    replacement, _ = candidates[0]  # Take best candidate based on gain

    # Perform the swap
    crit_idx = perm.index(critical_node)
    rep_idx = perm.index(replacement)

    neighbor[crit_idx], neighbor[rep_idx] = neighbor[rep_idx], neighbor[crit_idx]
    return neighbor


def greedy_expand(current: List[int], adj_list: List[Set[int]], degrees: List[int]) -> List[int]:
    return current


def initialize_solution(n: int, strategy: str, adj_list: List[Set[int]], degrees: List[int]) -> List[int]:
    """Advanced initialization with community detection and degree sorting."""

    if strategy == "community":
        communities = louvain_community_detection(adj_list)
        communities.sort(key=lambda c: (-len(c), -sum(degrees[n] for n in c)))  # Sort by size, then total degree
        permutation = []
        for comm in communities:
            # Sort community nodes by degree (high to low) *within* the community
            comm_sorted = sorted(comm, key=lambda x: -degrees[x])
            permutation.extend(comm_sorted)

    elif strategy == "degree":
        permutation = sorted(range(n), key=lambda x: -degrees[x])
    elif strategy == "reverse_degree":
        permutation = sorted(range(n), key=lambda x: degrees[x])
    else:  # random
        permutation = list(range(n))
        random.shuffle(permutation)

    # Initialize threshold (t) with exploration bias, but a reasonable default.
    t = int(n * np.random.beta(1.0, 2.0))  # Favor larger initial torso
    t = max(0, min(t, n - 1))  # Ensure t is valid
    return permutation + [t]


def dominates(score1: List[float], score2: List[float]) -> bool:
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or \
        (score1[0] >= score2[0] and score1[1] < score2[1])


def update_operator_performance(op, success: bool, op_scores, op_usage):
    if success:
        op_scores[op] += (1.0 - op_scores[op]) * 0.2  # Increase score

    op_usage[op] += 1


# Initialize operator tracking
neighbor_operators = [
    smart_torso_shift,
    block_move,
    critical_swap,
    community_shuffle,
    greedy_expand
]


def hill_climbing(edges: List[List[int]], max_iterations: int = 30000, num_restarts: int = 300) -> List[List[int]]:
    n = max(max(edge) for edge in edges) + 1
    adj_list = build_adjacency_list(n, edges)
    degrees = [len(adj) for adj in adj_list]  # Precompute degrees

    op_scores = {op: 0.0 for op in neighbor_operators}  # equal weights
    op_usage = {op: 0 for op in neighbor_operators}

    pareto_front = []
    global_best_score = [0, float('inf')]  # Track the overall best (for printing)
    global_best_solution = []
    start_time = time.time()

    # Tabu management:
    tabu_list = set()  # Set for efficient membership checking
    max_tabu_size = 50  # Base tabu size
    tabu_hits = 0

    # Adaptive cooling parameters (start faster)
    T0 = 100.0  # Initial temperature - lower T0 = less exploration
    cooling_rate = 0.995  # Faster cooling

    for restart in range(num_restarts):
        # Initialize solution and track scores
        init_strategy = random.choice(["community", "degree", "reverse_degree"])  # Explore different parts of space
        current = initialize_solution(n, init_strategy, adj_list, degrees)
        current_score = evaluate_solution(current, adj_list)[:2]

        best_local = current.copy()
        best_local_score = current_score.copy()
        T = T0
        last_improvement = 0

        print(f"\n🌀 Restart {restart + 1}/{num_restarts} | Initial Score: {current_score} | Strategy: {init_strategy}")

        for iteration in range(max_iterations):
            T *= cooling_rate

            # Operator selection based on weights
            # probs = np.array([math.exp(op_scores[op]) for op in neighbor_operators]) # Use exp to ensure positivity
            # probs /= np.sum(probs)  # Normalize to probabilities
            # op = random.choices(neighbor_operators, weights=probs)[0]
            probs = np.array([op_scores[op] for op in neighbor_operators])  # Use exp to ensure positivity
            probs = np.maximum(probs, 0)  # Ensure no negative probabilities
            probs_sum = np.sum(probs)

            if probs_sum > 0:
                probs /= probs_sum  # Normalize only it is greater than zero, to avoid division by zero
                op = random.choices(neighbor_operators, weights=probs)[0]
            else:
                op = random.choice(neighbor_operators)

            neighbor = op(current, adj_list, degrees)
            neighbor_score = evaluate_solution(neighbor, adj_list)[:2]
            neighbor_tuple = tuple(neighbor)  # Convert to tuple for hashing in tabu list

            # Tabu check:  If in tabu, skip to the next iteration.
            if neighbor_tuple in tabu_list:
                tabu_hits += 1
                continue

            # Update tabu list (FIFO):
            tabu_list.add(neighbor_tuple)
            if len(tabu_list) > max_tabu_size + restart // 5:  # dynamic tabu size
                tabu_list.pop()

            # Acceptance criteria (dominates or simulated annealing)
            accept = False
            if dominates(neighbor_score, current_score):
                accept = True
            else:
                # Weighted acceptance function (favoring size)
                size_gain = neighbor_score[0] - current_score[0]
                width_diff = current_score[1] - neighbor_score[1]  # *Decrease* in width is good!
                delta = 2.0 * size_gain + width_diff
                if T > 1e-06:
                    accept_prob = math.exp(delta / T)  # Standard SA acceptance
                    accept = random.random() < accept_prob

            if accept:
                current = neighbor
                current_score = neighbor_score
                update_operator_performance(op, True, op_scores, op_usage)
                if dominates(current_score, best_local_score):
                    best_local = current.copy()
                    best_local_score = current_score.copy()
                    last_improvement = iteration
            else:
                update_operator_performance(op, False, op_scores, op_usage)

            # Intensification (every 200 iterations):
            if iteration % 200 == 199:
                for _ in range(25):  # Try 25 local moves
                    candidate_op = random.choice(neighbor_operators)
                    candidate = candidate_op(best_local, adj_list, degrees)
                    candidate_score = evaluate_solution(candidate, adj_list)[:2]
                    if dominates(candidate_score, best_local_score):
                        best_local = candidate.copy()
                        best_local_score = candidate_score.copy()

            # Global best tracking (for reporting purposes)
            if dominates(best_local_score, global_best_score):
                global_best_score = best_local_score.copy()
                global_best_solution = best_local
                print(
                    f"🔥 New Global Best @ Iter {iteration}: Size={global_best_score[0]}, Width={global_best_score[1]}, Time={time.time() - start_time:.1f}s")

            # Progress reporting
            if iteration % 500 == 0:
                print(
                    f"Iter {iteration}/{max_iterations}, Temp: {T:.3f}, Current: {current_score}, Best Local: {best_local_score}, Global Best: {global_best_score}, Tabu Hits: {tabu_hits}")

            # Early restart:
            if iteration - last_improvement > 2500:
                print(f"🔄 Early Restart at iteration {iteration} (no improvement)")
                break

        # Local search post-processing (before adding to Pareto front)
        best_local = deterministic_local_search(best_local, adj_list)
        best_local_score = evaluate_solution(best_local, adj_list)[:2]

        # Update Pareto front (add if non-dominated)
        pareto_front.append((best_local, best_local_score))
        print(f"✅ Restart {restart + 1} Done | Best: {best_local_score} | Time: {time.time() - start_time:.1f}s")

    # Filter Pareto front to keep only non-dominated solutions
    filtered_front = []
    for sol, score in pareto_front:
        if not any(dominates(other_score, score) for other_sol, other_score in pareto_front if other_sol != sol):
            filtered_front.append(sol)

    return sorted(filtered_front,
                  key=lambda x: (-evaluate_solution(x, adj_list)[0], evaluate_solution(x, adj_list)[1]))[
           :5]  # Return top 5 by size, then width


def deterministic_local_search(solution: List[int], adj_list: List[Set[int]]) -> List[int]:
    """
    Performs a deterministic local search to improve a solution.
    Tries to add nodes to the torso (if width doesn't increase) and
    remove nodes to improve width.
    """
    n = len(solution) - 1
    t = solution[-1]
    improved = True
    while improved:
        improved = False
        # 1. Try to *add* nodes to the torso (increase size):
        for i in range(t):
            node_to_add = solution[i]
            # Check if adding this node would increase the width:

            current_width = 0
            for j in range(t, n):
                current_width = max(current_width, len(adj_list[solution[j]]))

            new_width = max(current_width, len(adj_list[node_to_add]))
            if new_width <= current_width:
                # Add the node to torso
                solution[i], solution[t] = solution[t], solution[i]
                t = t - 1
                solution[-1] = t  # update t
                improved = True
                break  # Go back to the beginning

        # 2. Try to *remove* nodes from the torso (decrease width)
        if not improved:  # only if additions did not help
            for i in range(n - 1, t - 1, -1):  # iterate from n-1 to t
                node_to_remove = solution[i]

                connections_in_torso = 0
                for neighbor in adj_list[node_to_remove]:
                    if solution.index(neighbor) >= t:
                        connections_in_torso += 1

                if (len(adj_list[node_to_remove]) > connections_in_torso):
                    # Remove
                    solution[i], solution[t] = solution[t], solution[i]
                    t = t + 1
                    solution[-1] = t
                    improved = True
                    break
    return solution


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
    random.seed(42)  # For reproducibility
    problem_id = input("🔍 Select problem (easy/medium/hard): ").lower()
    while problem_id not in problems:
        problem_id = input("❌ Invalid! Choose easy/medium/hard: ").lower()

    edges = load_graph(problem_id)

    start_time = time.time()
    solutions = hill_climbing(edges)
    print(f"\n⏱️ Optimization completed in {time.time() - start_time:.2f} seconds")

    for idx, sol in enumerate(solutions):
        create_submission_file(sol, problem_id, f"best_solution_{idx + 1}.json")

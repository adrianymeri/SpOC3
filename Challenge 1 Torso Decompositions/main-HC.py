import json
import random
import time
from typing import List, Tuple
import numpy as np
import urllib.request
import math

# Problem configurations
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}


def load_graph(problem_id: str) -> List[List[int]]:
    """Loads the graph data for the given problem ID."""
    url = problems[problem_id]
    print(f"Loading graph data from: {url}")
    with urllib.request.urlopen(url) as f:
        edges = []
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
    print(f"Loaded graph with {len(edges)} edges.")
    return edges


def evaluate_solution(decision_vector: List[int], edges: List[List[int]]) -> List[float]:
    """
    Evaluates the given solution and returns [torso_size, torso_width].
    Torso size = n - t  (we want this to be as high as possible, so t should be small).
    Torso width is the maximum degree (number of adjacent nodes) for nodes in the torso.
    Width > 500 is penalized (set to 501).
    """
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    size = n - t
    adj_list = {i: set() for i in range(n)}
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    max_width = 0
    for i in range(n):
        if i >= t:
            max_width = max(max_width, len(adj_list[i]))
    if max_width > 500:
        max_width = 501
    return [size, max_width]


# Define neighbor operators
def neighbor_swap(decision_vector: List[int]) -> List[int]:
    """Swap two random nodes in the permutation part."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i, j = random.sample(range(n), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def neighbor_shuffle(decision_vector: List[int]) -> List[int]:
    """Shuffle a random sublist of the permutation part."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    sub_len = max(2, int(0.1 * n))
    start = random.randint(0, n - sub_len)
    sub = neighbor[start:start + sub_len]
    random.shuffle(sub)
    neighbor[start:start + sub_len] = sub
    return neighbor


def neighbor_inversion(decision_vector: List[int]) -> List[int]:
    """Invert a random segment of the permutation part."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor


def neighbor_insert(decision_vector: List[int]) -> List[int]:
    """Remove a node and insert it at a random position in the permutation part."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 1)
    node = neighbor.pop(i)
    j = random.randint(0, n - 1)
    neighbor.insert(j, node)
    return neighbor


def neighbor_torso_shift(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    """Shift the torso threshold by a random amount."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    shift = int(perturbation_rate * n)
    neighbor[-1] = max(0, min(n - 1, neighbor[-1] + random.randint(-shift, shift)))
    return neighbor


neighbor_operators = [
    neighbor_swap,
    neighbor_shuffle,
    neighbor_inversion,
    neighbor_insert,
    neighbor_torso_shift,
]


def dominates(score1: List[float], score2: List[float]) -> bool:
    """
    Checks if score1 dominates score2.
    We want a higher torso size (size) and a lower torso width.
    """
    return (score1[0] > score2[0] and score1[1] <= score2[1]) or (score1[0] >= score2[0] and score1[1] < score2[1])


def scalar_objective(decision_vector: List[int], score: List[float]) -> float:
    """
    Scalar objective for simulated annealing:
    We use f(sol) = t + torso_width, where t is the threshold.
    Lower values are better.
    """
    return decision_vector[-1] + score[1]


def hill_climbing(edges: List[List[int]], max_iterations: int = 10000, num_restarts: int = 100) -> List[List[int]]:
    """
    Performs hill climbing with multiple restarts and simulated annealing acceptance
    to approximate the Pareto frontier.
    """
    n = max(node for edge in edges for node in edge) + 1
    pareto_front = []

    # Adaptive operator weights
    op_weights = {op: 1.0 for op in neighbor_operators}

    # Simulated annealing parameters
    T0 = 1000.0  # initial temperature
    cooling_rate = 0.9995  # cooling factor per iteration

    for restart in range(num_restarts):
        # Use two kinds of initialization: random and heuristic (by degree)
        if restart % 10 == 0:  # every 10th restart, use heuristic init
            # heuristic: sort nodes by ascending degree so that low-degree nodes appear in the torso part
            degree = {}
            # Build degree dictionary from edges
            for u, v in edges:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1
            permutation = sorted(range(n), key=lambda x: degree.get(x, 0))
        else:
            permutation = list(range(n))
            random.shuffle(permutation)
        initial_threshold = n // 2
        decision_vector = permutation + [initial_threshold]
        current_score = evaluate_solution(decision_vector, edges)
        best_solution, best_score = decision_vector[:], current_score[:]

        T = T0
        for iteration in range(max_iterations):
            T *= cooling_rate  # cool down
            # Choose operator adaptively
            op = random.choices(neighbor_operators, weights=[op_weights[o] for o in neighbor_operators])[0]
            neighbor = op(decision_vector)
            neighbor_score = evaluate_solution(neighbor, edges)

            # If neighbor dominates or is equal, accept it
            if dominates(neighbor_score, current_score) or neighbor_score == current_score:
                decision_vector, current_score = neighbor[:], neighbor_score[:]
                op_weights[op] *= 1.05
            else:
                # Compute scalar objectives
                f_current = scalar_objective(decision_vector, current_score)
                f_neighbor = scalar_objective(neighbor, neighbor_score)
                delta = f_neighbor - f_current
                if delta < 0 or random.random() < math.exp(-delta / T):
                    decision_vector, current_score = neighbor[:], neighbor_score[:]
                    op_weights[op] *= 1.02
            # Normalize operator weights
            total_weight = sum(op_weights[o] for o in neighbor_operators)
            for o in neighbor_operators:
                op_weights[o] /= total_weight

            if dominates(current_score, best_score):
                best_solution, best_score = decision_vector[:], current_score[:]

        pareto_front.append((best_solution, best_score))
        print(
            f"Restart {restart + 1}: Best score: {best_score} (scalar = {scalar_objective(best_solution, best_score):.2f})")

    # Filter non-dominated solutions across all restarts
    filtered = []
    for sol1, score1 in pareto_front:
        if not any(dominates(score2, score1) for _, score2 in pareto_front if score2 != score1):
            filtered.append(sol1)

    return filtered


def create_submission_file(decision_vector, problem_id, filename="submission.json"):
    """Creates a valid submission file."""
    submission = {
        "decisionVector": [decision_vector],
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")


if __name__ == "__main__":
    random.seed(42)
    problem_id = input("Select a problem instance (easy, medium, hard): ").lower()
    while problem_id not in problems:
        problem_id = input("Invalid problem ID. Please choose from 'easy', 'medium', or 'hard': ")
    edges = load_graph(problem_id)

    start_time = time.time()
    pareto_front = hill_climbing(edges, max_iterations=10000, num_restarts=100)
    end_time = time.time()
    print(f"Hill climbing completed in {end_time - start_time:.2f} seconds.")

    for i, solution in enumerate(pareto_front):
        create_submission_file(solution, problem_id, f"final_solution_{i + 1}.json")

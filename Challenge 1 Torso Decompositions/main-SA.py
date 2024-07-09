import json
import random
import time
from typing import List, Tuple

import numpy as np
import urllib.request
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed

# Define the problem instances
problems = {
    "supereasy": "data/supereasy.gr",
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# Define a scorer function for multi-objective optimization
def torso_scorer(y_true, y_pred):"""Combines torso size and width into a single score for optimization."""size_weight = -1  # Prioritize minimizing size
    width_weight = -0.5  # Penalize width but less than size
    return size_weight * y_pred[:, 0] + width_weight * y_pred[:, 1]


def load_graph(problem_id: str) -> List[List[int]]:"""Loads the graph data for the given problem ID."""file_path = problems[problem_id]
    print(f"Loading graph data from: {file_path}")
    edges = []
    if file_path.startswith("http"):
        with urllib.request.urlopen(file_path) as f:
            for line in f:
                if line.startswith(b"#"):
                    continue
                u, v = map(int, line.strip().split())
                edges.append([u, v])
    else:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                u, v = map(int, line.strip().split())
                edges.append([u, v])
    print(f"Loaded graph with {len(edges)} edges.")
    return edges


def calculate_torso_size(decision_vector: List[int]) -> int:"""Calculates the size of the torso for the given decision vector."""n = len(decision_vector) - 1
    t = decision_vector[-1]
    return n - t


def calculate_torso_width(decision_vector: List[int], edges: List[List[int]]) -> int:"""Calculates the width of the torso for the given decision vector and edges."""n = len(decision_vector) - 1
    t = decision_vector[-1]
    permutation = decision_vector[:-1]

    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    oriented_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] in adj_list[permutation[j]]:
                oriented_edges.append((permutation[i], permutation[j]))

    outdegrees = [0 for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (permutation[i], permutation[j]) in oriented_edges:
                for k in range(j + 1, n):
                    if (
                        (permutation[i], permutation[k]) in oriented_edges
                        and (permutation[j], permutation[k]) not in oriented_edges
                    ):
                        if permutation[j] >= t and permutation[k] >= t:
                            outdegrees[permutation[j]] += 1

    return max(outdegrees)


def evaluate_solution(
    decision_vector: List[int], edges: List[List[int]]
) -> List[float]:"""Evaluates the given solution and returns the torso size and width."""torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, edges)
    return [torso_size, torso_width]


# --- Neighborhood Operators ---

def generate_neighbor_swap(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:"""Generates a neighbor solution by swapping two random elements."""neighbor = decision_vector[:]
    n = len(neighbor) - 1
    num_perturbations = max(1, int(perturbation_rate * n))
    swap_indices = random.sample(range(n), num_perturbations * 2)
    for i in range(0, len(swap_indices), 2):
        neighbor[swap_indices[i]], neighbor[swap_indices[i + 1]] = (
            neighbor[swap_indices[i + 1]],
            neighbor[swap_indices[i]],
        )
    return neighbor


def generate_neighbor_2opt(decision_vector: List[int]) -> List[int]:"""Generates a neighbor solution using the 2-opt operator."""n = len(decision_vector) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor = decision_vector[:i] + decision_vector[i:j][::-1] + decision_vector[j:]
    return neighbor


def generate_neighbor_shuffle(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:"""Generates a neighbor solution by shuffling a portion of the decision vector."""neighbor = decision_vector[:]
    n = len(neighbor) - 1
    shuffle_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - shuffle_length)
    neighbor[start_index : start_index + shuffle_length] = random.sample(
        neighbor[start_index : start_index + shuffle_length], shuffle_length
    )
    return neighbor


def generate_neighbor_torso_shift(decision_vector: List[int]) -> List[int]:"""Generates a neighbor solution by shifting the torso position."""neighbor = decision_vector[:]
    n = len(neighbor) - 1
    new_torso_position = random.randint(0, n - 1)
    neighbor[-1] = new_torso_position
    return neighbor


# --- End of Neighborhood Operators ---


def dominates(score1: List[float], score2: List[float]) -> bool:"""Checks if score1 dominates score2 in multi-objective optimization."""return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )


def acceptance_probability(
    old_score: List[float], new_score: List[float], temperature: float
) -> float:"""Calculates the acceptance probability in Simulated Annealing."""# Use torso_scorer to combine multiple objectives into a single value
    delta_score = torso_scorer([[0, 0]], np.array(new_score)) - torso_scorer(
        [[0, 0]], np.array(old_score)
    )
    return np.exp(delta_score / temperature)


def simulated_annealing_single_restart(
    edges: List[List[int]],
    restart: int,
    max_iterations: int = 1000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    neighbor_operators: List[str] = ["swap", "2opt", "shuffle", "torso_shift"],
    save_interval: int = 50,
) -> Tuple[List[int], List[float]]:"""Performs a single restart of the Simulated Annealing algorithm."""n = max(node for edge in edges for node in edge) + 1

    # Generate random initial solution
    current_solution = [i for i in range(n)] + [random.randint(0, n - 1)]
    current_score = evaluate_solution(current_solution, edges)

    best_solution = current_solution[:]
    best_score = current_score[:]

    temperature = initial_temperature

    # Initialize operator weights and success counts
    operator_weights = {op: 1.0 for op in neighbor_operators}
    operator_success_counts = {op: 0 for op in neighbor_operators}

    for i in range(max_iterations):
        # Choose a neighbor operator based on weights
        operator = random.choices(
            list(operator_weights.keys()), list(operator_weights.values())
        )[0]

        # Generate neighbor solution using the selected operator
        if operator == "swap":
            neighbor = generate_neighbor_swap(current_solution)
        elif operator == "2opt":
            neighbor = generate_neighbor_2opt(current_solution)
        elif operator == "shuffle":
            neighbor = generate_neighbor_shuffle(current_solution)
        elif operator == "torso_shift":
            neighbor = generate_neighbor_torso_shift(current_solution)
        else:
            raise ValueError(f"Invalid neighbor operator: {operator}")

        neighbor_score = evaluate_solution(neighbor, edges)

        # Accept the neighbor if it's better or based on probability
        if dominates(neighbor_score, current_score) or random.random() < acceptance_probability(
            current_score, neighbor_score, temperature
        ):
            current_solution = neighbor[:]
            current_score = neighbor_score[:]

            # Update best solution if current solution is better
            if dominates(current_score, best_score):
                best_solution = current_solution[:]
                best_score = current_score[:]
                print(
                    f"Restart {restart+1} - Iteration {i+1}: New best solution found - {best_solution}, Score: {best_score}"
                )

            # Update operator success count
            operator_success_counts[operator] += 1

        # Update operator weights (e.g., every 10 iterations)
        if (i + 1) % 10 == 0:
            total_successes = sum(operator_success_counts.values())
            if total_successes > 0:
                for op in operator_weights:
                    operator_weights[op] = (
                        0.8 * operator_weights[op]
                        + 0.2 * operator_success_counts[op] / total_successes
                    )
                # Normalize weights
                total_weight = sum(operator_weights.values())
                operator_weights = {
                    op: w / total_weight for op, w in operator_weights.items()
                }

        # Cool down the temperature
        temperature *= cooling_rate

        # Save intermediate solutions
        if (i + 1) % save_interval == 0:
            create_submission_file(
                best_solution, problem_id, f"intermediate_solution_{restart+1}_iter_{i+1}.json"
            )

    print(
        f"Restart {restart+1} completed. Best solution: {best_solution}, Score: {best_score}"
    )
    return best_solution, best_score


def simulated_annealing(
    edges: List[List[int]],
    max_iterations: int = 1000,
    num_restarts: int = 10,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    neighbor_operators: List[str] = ["swap", "2opt", "shuffle", "torso_shift"],
    save_interval: int = 50,
    n_jobs: int = -1,
) -> List[List[int]]:"""Performs Simulated Annealing to find a set of Pareto optimal solutions."""pareto_front = []
    start_time = time.time()

    # Run Simulated Annealing with multiple restarts in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulated_annealing_single_restart)(
            edges,
            restart,
            max_iterations,
            initial_temperature,
            cooling_rate,
            neighbor_operators,
            save_interval,
        )
        for restart in range(num_restarts)
    )

    pareto_front = results  # Results now contain solutions from all restarts

    print(f"Pareto Front: {pareto_front}")

    # Filter for non-dominated solutions
    filtered_pareto_front = []
    for i in range(len(pareto_front)):
        solution1, score1 = pareto_front[i]
        dominated = False
        for j in range(len(pareto_front)):
            if i != j:
                solution2, score2 = pareto_front[j]
                if dominates(score2, score1):
                    dominated = True
                    break
        if not dominated:
            filtered_pareto_front.append(solution1)

    print(f"Filtered Pareto Front: {filtered_pareto_front}")
    return filtered_pareto_front


def create_submission_file(
    decision_vector, problem_id, filename="submission.json"
):"""Creates a valid submission file."""submission = {
        "decisionVector": [decision_vector],  # Wrap in a list for multiple solutions
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")


if __name__ == "__main__":
    random.seed(42)
    problem_id = input(
        "Select problem instance (supereasy, easy, medium, hard): "
    )  # Get input from the user
    edges = load_graph(problem_id)

    # Simulated Annealing
    print("Starting Simulated Annealing...")
    pareto_front = simulated_annealing(
        edges,
        max_iterations=1000,  # Adjust as needed
        num_restarts=20,  # Adjust as needed
        initial_temperature=100.0,  # Adjust as needed
        cooling_rate=0.95,  # Adjust as needed
        neighbor_operators=[
            "swap",
            "2opt",
            "shuffle",
            "torso_shift",
        ],  # Choose operators to use
        save_interval=50,  # Save every 50 iterations
    )

    # Create Final Submission File
    for i, solution in enumerate(pareto_front):
        create_submission_file(solution, problem_id, f"final_solution_{i+1}.json")
    print("All submission files created successfully!")

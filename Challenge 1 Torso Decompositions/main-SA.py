import json
import random
import time
from typing import List, Tuple

import numpy as np
import urllib.request
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from joblib import Parallel, delayed
from collections import defaultdict

# Define the problem instances
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# Define a scorer function for multi-objective optimization
def torso_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Combines torso size and width into a single score for optimization."""
    size_weight = -1  # Prioritize minimizing size
    width_weight = -0.5  # Penalize width but less than size
    return size_weight * y_pred[:, 0] + width_weight * y_pred[:, 1]


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

def calculate_torso_size(decision_vector: List[int]) -> int:
    """Calculates the size of the torso for the given decision vector."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    return n - t


def calculate_torso_width(decision_vector: List[int], edges: List[List[int]]) -> int:
    """Calculates the width of the torso for the given decision vector and edges."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    permutation = decision_vector[:-1]

    # Create adjacency list
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Calculate outdegrees for nodes in the torso
    outdegrees = [0 for _ in range(n)]
    for i in range(t, n):
        for j in adj_list[permutation[i]]:
            if j in permutation[t:] and permutation.index(j) > i:
                outdegrees[i] += 1

    return max(outdegrees)


def evaluate_solution(
    decision_vector: List[int], edges: List[List[int]]
) -> List[float]:
    """Evaluates the given solution and returns the torso size and width."""
    torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, edges)
    return [torso_size, torso_width]

def generate_neighbor_swap(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by swapping two random elements within the torso."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    t = neighbor[-1]
    num_perturbations = max(1, int(perturbation_rate * (n - t)))
    
    # Only swap elements within the torso
    swap_indices = random.sample(range(t, n), num_perturbations * 2) 
    for i in range(0, len(swap_indices), 2):
        neighbor[swap_indices[i]], neighbor[swap_indices[i + 1]] = (
            neighbor[swap_indices[i + 1]],
            neighbor[swap_indices[i]],
        )
    return neighbor


def generate_neighbor_shuffle(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution by shuffling a portion of the decision vector within the torso."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    t = neighbor[-1]
    start = random.randint(t, max(t, n - 2))  # Ensure start is within or at the beginning of the torso
    end = random.randint(start + 1, n)
    random.shuffle(neighbor[start:end])
    return neighbor


def generate_neighbor_torso_shift(decision_vector: List[int], max_shift: int = 5) -> List[int]:
    """Generates a neighbor by shifting the torso position within a limited range."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    t = neighbor[-1]
    new_t = random.randint(max(0, t - max_shift), min(n - 1, t + max_shift))
    neighbor[-1] = new_t
    return neighbor


def generate_neighbor_2opt(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution using 2-opt within the torso."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    t = neighbor[-1]
    i = random.randint(t, max(t, n - 3))  # Ensure i is within or at the beginning of the torso
    j = random.randint(i + 2, n - 1)
    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor

def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )


def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'lgbm') -> MultiOutputRegressor:
    """Trains an LGBM or XGBoost model with hyperparameter tuning."""
    print(f"Training {model_type} model...")
    best_model = None
    best_score = float("inf")

    if model_type == 'lgbm':
        model = MultiOutputRegressor(LGBMRegressor(random_state=42, n_jobs=-1))
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7],
            "estimator__num_leaves": [15, 31, 63],
        }
    else:  # XGBoost
        model = MultiOutputRegressor(XGBRegressor(random_state=42, n_jobs=-1))
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7],
            "estimator__subsample": [0.8, 0.9, 1.0],
            "estimator__colsample_bytree": [0.8, 0.9, 1.0],
        }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,  # Increased iterations for better exploration
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),  # Increased folds
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    best_score = -random_search.best_score_
    print(f"Best {model_type} model: {best_model}")
    print(f"{model_type} training completed with best score: {best_score}")
    return best_model

def generate_neighbor_ml(
    decision_vector: List[int],
    edges: List[List[int]],
    model: MultiOutputRegressor,
    num_neighbors: int = 100,
) -> List[int]:
    """Generates neighbors using ML models for prediction and selects the best."""
    n = len(decision_vector) - 1

    neighbors = []
    for _ in range(num_neighbors // 5):
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.1))
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.2))
        neighbors.append(generate_neighbor_shuffle(decision_vector.copy()))
        neighbors.append(generate_neighbor_torso_shift(decision_vector.copy()))
        neighbors.append(generate_neighbor_2opt(decision_vector.copy()))

    # Convert neighbors to a NumPy array for prediction
    neighbors_np = np.array(neighbors)
    predictions = model.predict(neighbors_np)

    # Calculate the score for each prediction
    scores = [torso_scorer(np.array([[0, 0]]), pred.reshape(1, -1)) for pred in predictions]

    # Get the index of the best neighbor
    best_neighbor_idx = np.argmin(scores)
    return neighbors[best_neighbor_idx]


def choose_neighbor_generation_method(
    current_iteration: int,
    initial_exploration_iterations: int,
    ml_switch_interval: int,
    operator_weights: List[float],
) -> str:
    """Chooses the neighbor generation method based on exploration/exploitation strategy."""

    if current_iteration < initial_exploration_iterations:
        # Exploration phase: Use all operators with equal probability
        return random.choice(["swap", "shuffle", "torso_shift", "2opt"])
    else:
        # Exploitation phase: Use operator weights
        return random.choices(
            ["swap", "shuffle", "torso_shift", "2opt"], weights=operator_weights
        )[0]


def simulated_annealing_single_restart(
    edges: List[List[int]],
    restart: int,
    problem_id: str,
    max_iterations: int = 1000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    initial_exploration_iterations: int = 20,
    ml_switch_interval: int = 25,
    save_interval: int = 50,
    operator_change_interval: int = 100,
) -> Tuple[List[int], List[float]]:
    """Performs a single restart of the simulated annealing algorithm."""
    n = max(node for edge in edges for node in edge) + 1

    # Start with a random valid solution
    current_solution = [i for i in range(n)] + [random.randint(0, n - 1)]
    current_score = evaluate_solution(current_solution, edges)

    best_solution = current_solution[:]
    best_score = current_score[:]

    temperature = initial_temperature

    initial_perturbation_rate = 0.5
    perturbation_rate_decay = 0.99

    # Data collection for ML models
    X = []
    y = []

    lgbm_model = None
    xgboost_model = None

    # Operator weights for exploitation phase
    operator_weights = [1.0, 1.0, 1.0, 1.0]  # Initial weights are equal

    for i in range(max_iterations):
        # Choose neighbor generation method (exploration vs. exploitation)
        neighbor_generation_method = choose_neighbor_generation_method(
            i, initial_exploration_iterations, ml_switch_interval, operator_weights
        )
        print(
            f"Restart {restart+1} - Iteration {i+1}: Using {neighbor_generation_method} operator."
        )

        if i >= initial_exploration_iterations and i % operator_change_interval == 0:
            # Update operator weights based on their performance
            operator_weights = update_operator_weights(
                X, y, operator_weights, initial_exploration_iterations, ml_switch_interval
            )

        if neighbor_generation_method == "swap":
            neighbor = generate_neighbor_swap(current_solution, initial_perturbation_rate)
            initial_perturbation_rate *= perturbation_rate_decay
        elif neighbor_generation_method == "shuffle":
            neighbor = generate_neighbor_shuffle(current_solution)
        elif neighbor_generation_method == "torso_shift":
            neighbor = generate_neighbor_torso_shift(current_solution)
        elif neighbor_generation_method == "2opt":
            neighbor = generate_neighbor_2opt(current_solution)

        neighbor_score = evaluate_solution(neighbor, edges)

        # Collect data for ML training (even if not using ML in this iteration)
        X.append(neighbor)
        y.append(neighbor_score)

        # Phase 2: Alternate between basic operators and ML models
        if i >= initial_exploration_iterations and i % ml_switch_interval == 0:
            # Train/retrain models
            if lgbm_model is None or i % (2 * ml_switch_interval) == 0:
                lgbm_model = train_model(np.array(X), np.array(y), 'lgbm')
            if xgboost_model is None or i % (2 * ml_switch_interval) == ml_switch_interval:
                xgboost_model = train_model(np.array(X), np.array(y), 'xgboost')

            # Choose ML model (LGBM, XGBoost, or hybrid)
            model_choice = random.choice(["lgbm", "xgboost", "hybrid"])
            print(f"Restart {restart+1} - Iteration {i+1}: Using {model_choice} model.")

            if model_choice == "lgbm":
                neighbor = generate_neighbor_ml(current_solution, edges, lgbm_model)
            elif model_choice == "xgboost":
                neighbor = generate_neighbor_ml(current_solution, edges, xgboost_model)
            else:  # hybrid
                neighbor = generate_neighbor_ml(current_solution, edges, lgbm_model if i % (2 * ml_switch_interval) < ml_switch_interval else xgboost_model)

            neighbor_score = evaluate_solution(neighbor, edges)

        # Calculate acceptance probability
        delta_score = (
            (neighbor_score[0] - current_score[0])
            + 0.5 * (neighbor_score[1] - current_score[1])
        )
        acceptance_probability = np.exp(min(0, delta_score) / temperature)

        if delta_score < 0 or random.random() < acceptance_probability:
            current_solution = neighbor[:]
            current_score = neighbor_score[:]

        # Update best solution
        if dominates(current_score, best_score):
            best_solution = current_solution[:]
            best_score = current_score[:]
            print(
                f"Restart {restart+1} - Iteration {i+1}: New best solution found - Score: {best_score}"
            )

        temperature *= cooling_rate

        # Save intermediate solutions
        if (i + 1) % save_interval == 0:
            create_submission_file(best_solution, problem_id, f"intermediate_solution_{restart+1}_iter_{i+1}.json")

    print(f"Restart {restart+1} completed. Best solution: {best_solution}, Score: {best_score}")
    return best_solution, best_score


def simulated_annealing(
    edges: List[List[int]],
    problem_id: str,
    max_iterations: int = 1000,
    num_restarts: int = 10,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    initial_exploration_iterations: int = 20,
    ml_switch_interval: int = 25,
    save_interval: int = 50,
    operator_change_interval: int = 100,
    n_jobs: int = -1,
) -> List[List[int]]:
    """Performs simulated annealing to find a set of Pareto optimal solutions."""
    n = max(node for edge in edges for node in edge) + 1
    pareto_front = []
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulated_annealing_single_restart)(
            edges,
            restart,
            problem_id,
            max_iterations,
            initial_temperature,
            cooling_rate,
            initial_exploration_iterations,
            ml_switch_interval,
            save_interval,
            operator_change_interval,
        )
        for restart in range(num_restarts)
    )

    for result in results:
        solution, score = result
        pareto_front.append((solution, score))

    print(f"Pareto Front: {pareto_front}")

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


def create_submission_file(decision_vector: List[int], problem_id: str, filename: str = "submission.json"):
    """Creates a valid submission file."""
    submission = {
        "decisionVector": [decision_vector],
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")


def update_operator_weights(
    X: List[List[int]],
    y: List[List[float]],
    operator_weights: List[float],
    initial_exploration_iterations: int,
    ml_switch_interval: int,
) -> List[float]:
    """Updates the operator weights based on their past performance."""
    # Calculate the improvement made by each operator
    operator_improvements = [0.0, 0.0, 0.0, 0.0]
    for i in range(initial_exploration_iterations, len(X) - 1):
        if i % ml_switch_interval != 0:  # Only consider iterations where basic operators were used
            previous_score = y[i - 1]
            current_score = y[i]
            improvement = (previous_score[0] - current_score[0]) + 0.5 * (
                previous_score[1] - current_score[1]
            )

            # Determine which operator was used
            if X[i] != X[i - 1]:  # Operator was applied
                if X[i][-1] != X[i - 1][-1]:
                    operator_index = 2  # torso_shift
                elif is_shuffled(X[i], X[i - 1]):
                    operator_index = 1  # shuffle
                elif is_2opt(X[i], X[i - 1]):
                    operator_index = 3  # 2-opt
                else:
                    operator_index = 0  # swap

                operator_improvements[operator_index] += improvement

    # Normalize the improvements to get weights
    total_improvement = sum(operator_improvements)
    if total_improvement > 0:
        operator_weights = [
            imp / total_improvement for imp in operator_improvements
        ]
    else:
        # If no improvement, maintain equal weights
        operator_weights = [1.0, 1.0, 1.0, 1.0]
    return operator_weights


def is_shuffled(list1: List[int], list2: List[int]) -> bool:
    """Checks if two lists are permutations of each other."""
    return sorted(list1) == sorted(list2)


def is_2opt(list1: List[int], list2: List[int]) -> bool:
    """Checks if two lists are different by a single 2-opt move."""
    differences = sum(1 for a, b in zip(list1, list2) if a != b)
    return differences == 4  # 2-opt changes 4 elements


if __name__ == "__main__":
    random.seed(42)

    chosen_problem = input("Choose problem difficulty (easy, medium, hard): ").lower()

    while chosen_problem not in problems:
        print("Invalid problem difficulty. Please choose from 'easy', 'medium', or 'hard'.")
        chosen_problem = input(
            "Choose problem difficulty (easy, medium, hard): "
        ).lower()

    print(f"Processing problem: {chosen_problem}")
    edges = load_graph(chosen_problem)

    pareto_front = simulated_annealing(
        edges,
        chosen_problem,
        max_iterations=10000,  # Increased iterations for better exploration
        num_restarts=20,     # Increased restarts for better exploration
        save_interval=250,     # Save less frequently to reduce I/O
        operator_change_interval=50,  # Change operators more frequently
        initial_exploration_iterations=100,  # Increased iterations for initial exploration
        ml_switch_interval=50,     # Adjusted interval for switching
        n_jobs=-1,          # Use all available cores for parallel processing
    )

    for i, solution in enumerate(pareto_front):
        create_submission_file(
            solution, chosen_problem, f"{chosen_problem}_final_solution_{i+1}.json"
        )

    print(f"All submission files created successfully!")

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

# Define the problem instances
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# Define a scorer function for multi-objective optimization
def torso_scorer(y_true, y_pred):
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
) -> List[float]:
    """Evaluates the given solution and returns the torso size and width."""
    torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, edges)
    return [torso_size, torso_width]


def generate_neighbor_swap(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by swapping two random elements."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    num_perturbations = max(1, int(perturbation_rate * n))
    swap_indices = random.sample(range(n), num_perturbations * 2)
    for i in range(0, len(swap_indices), 2):
        neighbor[swap_indices[i]], neighbor[swap_indices[i + 1]] = (
            neighbor[swap_indices[i + 1]],
            neighbor[swap_indices[i]],
        )
    return neighbor


def generate_neighbor_shuffle(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution by shuffling a portion of the decision vector."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    start = random.randint(0, n - 2)
    end = random.randint(start + 1, n)
    random.shuffle(neighbor[start:end])
    return neighbor


def generate_neighbor_torso_shift(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor by shifting the torso position."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    t = neighbor[-1]
    new_t = random.randint(0, n - 1)
    neighbor[-1] = new_t
    return neighbor


def generate_neighbor_2opt(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution using 2-opt."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 3)
    j = random.randint(i + 2, n - 1)
    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor


def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )


def train_model(X, y, model_type='lgbm'):
    """Trains an LGBM or XGBoost model with hyperparameter tuning."""
    print(f"Training {model_type} model...")
    best_model = None
    best_score = float("inf")

    if model_type == 'lgbm':
        model = MultiOutputRegressor(LGBMRegressor(random_state=42))
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7],
            "estimator__num_leaves": [15, 31, 63],  # Added num_leaves
        }
    else:  # XGBoost
        model = MultiOutputRegressor(XGBRegressor(random_state=42))
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [3, 5, 7],
            "estimator__subsample": [0.8, 0.9, 1.0],
            "estimator__colsample_bytree": [0.8, 0.9, 1.0],
        }

    # Use RandomizedSearchCV for more efficient hyperparameter search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,  # Number of random combinations to try
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,  # Use all available cores for faster training
    )

    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    best_score = -random_search.best_score_  # Use negative mean for minimization
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

    # Generate diverse neighbors using different operators
    neighbors = []
    for _ in range(num_neighbors // 5):
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.1))  # Smaller perturbation
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.2))
        neighbors.append(generate_neighbor_shuffle(decision_vector.copy()))
        neighbors.append(generate_neighbor_torso_shift(decision_vector.copy()))
        neighbors.append(generate_neighbor_2opt(decision_vector.copy()))

    # Predict scores for potential neighbors
    predictions = model.predict(np.array(neighbors))

    # Select the best neighbor based on predictions
    best_neighbor_idx = np.argmin(
        [torso_scorer([[0, 0]], pred) for pred in predictions]
    )
    return neighbors[best_neighbor_idx]


def simulated_annealing_single_restart(
    edges: List[List[int]],
    restart: int,
    problem_id: str,  # Add problem_id here
    max_iterations: int = 1000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    neighbor_generation_method: str = "swap",
    lgbm_model=None,
    xgboost_model=None,
    ml_switch_interval: int = 25,
    save_interval: int = 50,
) -> Tuple[List[int], List[float]]:
    """Performs a single restart of the simulated annealing algorithm."""
    n = max(node for edge in edges for node in edge) + 1

    # Generate random initial solution
    current_solution = [i for i in range(n)] + [random.randint(0, n - 1)]
    current_score = evaluate_solution(current_solution, edges)

    best_solution = current_solution[:]
    best_score = current_score[:]

    temperature = initial_temperature

    # Adaptive parameters
    initial_perturbation_rate = 0.5
    perturbation_rate_decay = 0.99

    for i in range(max_iterations):
        # Generate neighbor solution using the selected method
        if neighbor_generation_method == "lgbm_ml":
            neighbor = generate_neighbor_ml(current_solution, edges, lgbm_model)
        elif neighbor_generation_method == "xgboost_ml":
            neighbor = generate_neighbor_ml(current_solution, edges, xgboost_model)
        elif neighbor_generation_method == "hybrid_ml":
            if i % ml_switch_interval < ml_switch_interval // 2:
                model = lgbm_model
                model_name = "LGBM"
            else:
                model = xgboost_model
                model_name = "XGBoost"
            neighbor = generate_neighbor_ml(current_solution, edges, model)
            print(f"Iteration {i+1}: Used {model_name} to generate neighbor.")
        else:  # Default to 'swap'
            # Adaptively adjust perturbation rate during the search
            neighbor = generate_neighbor_swap(current_solution, initial_perturbation_rate)
            initial_perturbation_rate *= perturbation_rate_decay

        neighbor_score = evaluate_solution(neighbor, edges)

        # Calculate acceptance probability
        delta_score = (
            (neighbor_score[0] - current_score[0])
            + 0.5 * (neighbor_score[1] - current_score[1])
        )  # Prioritize torso size
        acceptance_probability = np.exp(min(0, delta_score) / temperature)

        # Accept neighbor based on probability
        if delta_score < 0 or random.random() < acceptance_probability:
            current_solution = neighbor[:]
            current_score = neighbor_score[:]

        # Update best solution
        if dominates(current_score, best_score):
            best_solution = current_solution[:]
            best_score = current_score[:]
            print(
                f"Restart {restart+1} - Iteration {i+1}: New best solution found - {best_solution}, Score: {best_score}"
            )

        # Decrease temperature
        temperature *= cooling_rate

        # Save intermediate solutions
        if (i + 1) % save_interval == 0:
            create_submission_file(best_solution, problem_id, f"intermediate_solution_{restart+1}_iter_{i+1}.json")

    print(f"Restart {restart+1} completed. Best solution: {best_solution}, Score: {best_score}")
    return best_solution, best_score


def simulated_annealing(
    edges: List[List[int]],
    problem_id: str,  # Add problem_id as an argument
    max_iterations: int = 1000,
    num_restarts: int = 10,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    neighbor_generation_method: str = "swap",
    use_hybrid_ml: bool = False,
    ml_switch_interval: int = 25,
    save_interval: int = 50,
    n_jobs: int = -1,
) -> List[List[int]]:
    """Performs simulated annealing to find a set of Pareto optimal solutions."""
    n = max(node for edge in edges for node in edge) + 1
    pareto_front = []
    start_time = time.time()

    # Train models only once at the beginning if using ML
    if neighbor_generation_method.endswith("ml"):
        print("Training models for the first time...")
        X = []
        y = []
        for _ in range(5000):  # Increased data size for better model training
            decision_vector = [i for i in range(n)] + [random.randint(0, n - 1)]
            X.append(decision_vector)
            y.append(evaluate_solution(decision_vector, edges))
        X = np.array(X)
        y = np.array(y)
        lgbm_model = train_model(X, y, 'lgbm')
        xgboost_model = train_model(X, y, 'xgboost')
    else:
        lgbm_model = None
        xgboost_model = None

    # Run simulated annealing with multiple restarts in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(simulated_annealing_single_restart)(
            edges,
            restart,
            problem_id,  # Pass problem_id to the function
            max_iterations,
            initial_temperature,
            cooling_rate,
            neighbor_generation_method,
            lgbm_model,
            xgboost_model,
            ml_switch_interval,
            save_interval,
        )
        for restart in range(num_restarts)
    )

    # Aggregate results from all restarts
    for result in results:
        solution, score = result
        pareto_front.append((solution, score))

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

    # Get problem difficulty from user input
    chosen_problem = input("Choose problem difficulty (easy, medium, hard): ").lower()

    # Validate user input
    while chosen_problem not in problems:
        print("Invalid problem difficulty. Please choose from 'easy', 'medium', or 'hard'.")
        chosen_problem = input(
            "Choose problem difficulty (easy, medium, hard): "
        ).lower()

    print(f"Processing problem: {chosen_problem}")
    edges = load_graph(chosen_problem)

    # Define the neighbor generation methods to use
    methods = ["swap", "shuffle", "torso_shift", "2opt", "lgbm_ml", "xgboost_ml", "hybrid_ml"]

    # Iterate over the methods and run simulated annealing
    for method in methods:
        print(f"Starting Simulated Annealing with {method}...")
        pareto_front = simulated_annealing(
            edges,
            chosen_problem,  # Pass chosen_problem to the function
            neighbor_generation_method=method,
            max_iterations=5000,  # Increased iterations
            num_restarts=50,  # Increased restarts
            save_interval=200,  # Increased save interval
        )

        # Create Final Submission Files for the current problem and method
        for i, solution in enumerate(pareto_front):
            create_submission_file(
                solution, chosen_problem, f"{chosen_problem}_{method}_final_solution_{i+1}.json"
            )

        print(f"All submission files for {method} created successfully!")

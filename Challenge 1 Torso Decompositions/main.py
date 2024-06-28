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

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }

    for n_estimators in param_grid["n_estimators"]:
        for learning_rate in param_grid["learning_rate"]:
            for max_depth in param_grid["max_depth"]:
                if model_type == 'lgbm':
                    model = MultiOutputRegressor(
                        LGBMRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42,
                        )
                    )
                else: # XGBoost
                    model = MultiOutputRegressor(
                        XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42,
                        )
                    )
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=KFold(n_splits=3, shuffle=True, random_state=42),
                    scoring="neg_mean_squared_error",
                )
                mean_score = -cv_scores.mean()  # Use negative mean for minimization
                print(f"Model with {n_estimators}, {learning_rate}, {max_depth}: {mean_score}") # Verbose output
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model
    print(f"Best {model_type} model: {best_model}") # Verbose output
    print(f"{model_type} training completed.")
    return best_model


def generate_neighbor_ml(
    decision_vector: List[int],
    edges: List[List[int]],
    model: MultiOutputRegressor,
) -> List[int]:
    """Generates neighbors using ML models for prediction."""
    n = len(decision_vector) - 1

    # Prepare data for training
    X = []
    y = []
    for _ in range(100):  # Generate some training data
        neighbor = generate_neighbor_swap(decision_vector, 0.2)
        X.append(neighbor)
        y.append(evaluate_solution(neighbor, edges))

    X = np.array(X)
    y = np.array(y)

    # Predict scores for potential neighbors
    neighbors = [generate_neighbor_swap(decision_vector, 0.2) for _ in range(100)]
    predictions = model.predict(np.array(neighbors))

    # Select the best neighbor based on predictions
    best_neighbor_idx = np.argmin(
        [torso_scorer([[0, 0]], pred) for pred in predictions]
    )
    return neighbors[best_neighbor_idx]


def hill_climbing_single_restart(
    edges: List[List[int]],
    restart: int,
    max_iterations: int = 100,
    perturbation_rate: float = 0.2,
    neighbor_generation_method: str = "swap",
    lgbm_model=None,
    xgboost_model=None,
    ml_switch_interval: int = 5,
    save_interval: int = 50, # Save every 50 iterations
) -> Tuple[List[int], List[float]]:
    """Performs a single restart of the hill climbing algorithm."""
    n = max(node for edge in edges for node in edge) + 1

    # Generate random initial solution
    decision_vector = [i for i in range(n)] + [random.randint(0, n - 1)]

    # Evaluate initial solution
    current_score = evaluate_solution(decision_vector, edges)
    print(f"Restart {restart+1} - Initial solution: {decision_vector}, Score: {current_score}")

    best_decision_vector = decision_vector[:]
    best_score = current_score[:]

    for i in range(max_iterations):
        # Generate neighbor solution using the selected method
        if neighbor_generation_method == "lgbm_ml":
            neighbor = generate_neighbor_ml(decision_vector, edges, lgbm_model)
        elif neighbor_generation_method == "xgboost_ml":
            neighbor = generate_neighbor_ml(decision_vector, edges, xgboost_model)
        elif neighbor_generation_method == "hybrid_ml":
            if i % ml_switch_interval < ml_switch_interval // 2:
                model = lgbm_model
                model_name = "LGBM"
            else:
                model = xgboost_model
                model_name = "XGBoost"
            neighbor = generate_neighbor_ml(decision_vector, edges, model)
            print(f"Iteration {i+1}: Used {model_name} to generate neighbor.")  # Indicate which model was used
        else:  # Default to 'swap'
            neighbor = generate_neighbor_swap(decision_vector, perturbation_rate)

        # Evaluate neighbor solution
        neighbor_score = evaluate_solution(neighbor, edges)

        # Update current solution if neighbor is better or equal
        if dominates(neighbor_score, current_score) or neighbor_score == current_score:
            decision_vector = neighbor[:]
            current_score = neighbor_score[:]
            print(f"Restart {restart+1} - Iteration {i+1}: Found better solution - {decision_vector}, Score: {current_score}")

        # Update best solution if current solution is better
        if dominates(current_score, best_score):
            best_decision_vector = decision_vector[:]
            best_score = current_score[:]
            print(f"Restart {restart+1} - Iteration {i+1}: New best solution found -  {best_decision_vector}, Score: {best_score}")

        # Save intermediate solutions
        if (i + 1) % save_interval == 0:
            create_submission_file(best_decision_vector, problem_id, f"intermediate_solution_{restart+1}_iter_{i+1}.json")

    print(f"Restart {restart+1} completed. Best solution: {best_decision_vector}, Score: {best_score}")
    return best_decision_vector, best_score

def hill_climbing(
    edges: List[List[int]],
    max_iterations: int = 100,
    num_restarts: int = 10,
    perturbation_rate: float = 0.2,
    neighbor_generation_method: str = "swap",
    use_hybrid_ml: bool = False,
    ml_switch_interval: int = 5,
    save_interval: int = 50, # Save every 50 iterations
    n_jobs: int = -1,  # Use all available cores for parallelization
) -> List[List[int]]:
    """Performs hill climbing to find a set of Pareto optimal solutions."""
    n = max(node for edge in edges for node in edge) + 1
    pareto_front = []
    start_time = time.time()

    # Train models only once at the beginning if using ML
    if neighbor_generation_method.endswith("ml"):
        print("Training models for the first time...")
        X = []
        y = []
        for _ in range(100):
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

    # Run hill climbing with multiple restarts in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(hill_climbing_single_restart)(
            edges,
            restart,
            max_iterations,
            perturbation_rate,
            neighbor_generation_method,
            lgbm_model,
            xgboost_model,
            ml_switch_interval,
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

def create_submission_file(decision_vector, problem_id, filename="submission.json"):
    """Creates a valid submission file."""
    submission = {
        "decisionVector": [decision_vector], # Wrap in a list for multiple solutions
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")

if __name__ == "__main__":
    random.seed(42)
    problem_id = "easy"  # You can change this to 'medium' or 'hard'
    edges = load_graph(problem_id)

    # Hill Climbing with Swap
    print("Starting Hill Climbing with Swap...")
    pareto_front_swap = hill_climbing(
        edges,
        neighbor_generation_method="swap",
        max_iterations=500,  # Adjust as needed
        num_restarts=20,  # Adjust as needed
        save_interval=50, # Save every 50 iterations
    )

    # Hill Climbing with LGBM
    print("Starting Hill Climbing with LGBM...")
    pareto_front_lgbm = hill_climbing(
        edges,
        neighbor_generation_method="lgbm_ml",
        max_iterations=500,  # Adjust as needed
        num_restarts=20,  # Adjust as needed
        save_interval=50, # Save every 50 iterations
    )

    # Hill Climbing with XGBoost
    print("Starting Hill Climbing with XGBoost...")
    pareto_front_xgboost = hill_climbing(
        edges,
        neighbor_generation_method="xgboost_ml",
        max_iterations=500,  # Adjust as needed
        num_restarts=20,  # Adjust as needed
        save_interval=50, # Save every 50 iterations
    )

    # Hill Climbing with Hybrid LGBM/XGBoost
    print("Starting Hill Climbing with Hybrid LGBM/XGBoost...")
    pareto_front_hybrid = hill_climbing(
        edges,
        neighbor_generation_method="hybrid_ml",
        ml_switch_interval=25,  # Switch between models every 25 iterations
        max_iterations=500,  # Adjust as needed
        num_restarts=20,  # Adjust as needed
        save_interval=50, # Save every 50 iterations
    )

    # Example: Select the Pareto front from the hybrid approach
    best_pareto_front = pareto_front_hybrid

    # Create Final Submission File
    for i, solution in enumerate(best_pareto_front):
        create_submission_file(solution, problem_id, f"final_solution_{i+1}.json")
    print("All submission files created successfully!")

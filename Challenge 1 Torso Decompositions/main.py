import json
import random
from typing import List

import numpy as np
import urllib.request
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from scipy.optimize import minimize

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
    return size_weight * y_pred[0] + width_weight * y_pred[1]

# Create a scorer object
torso_scorer_obj = make_scorer(torso_scorer, greater_is_better=False)

def load_graph(problem_id: str) -> List[List[int]]:
    """Loads the graph data for the given problem ID."""
    url = problems[problem_id]
    with urllib.request.urlopen(url) as f:
        edges = []
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
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
    """
    Checks if score1 dominates score2 in multi-objective optimization.

    Args:
        score1: The first score vector.
        score2: The second score vector.

    Returns:
        True if score1 dominates score2, False otherwise.
    """
    return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )


def train_lgbm_model(X, y):
    """Trains an LGBM model with hyperparameter tuning."""
    model = LGBMRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    # (Add more sophisticated hyperparameter tuning here if needed)
    best_model = model  # Start with default model
    best_score = float('inf')
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=torso_scorer_obj)
                mean_score = -cv_scores.mean()  # Use negative mean for minimization
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model
    return best_model


def train_xgboost_model(X, y):
    """Trains an XGBoost model with hyperparameter tuning."""
    model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }
    # (Add more sophisticated hyperparameter tuning here if needed)
    best_model = model  # Start with default model
    best_score = float('inf')
    for n_estimators in param_grid['n_estimators']:
        for learning_rate in param_grid['learning_rate']:
            for max_depth in param_grid['max_depth']:
                model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=torso_scorer_obj)
                mean_score = -cv_scores.mean()  # Use negative mean for minimization
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model
    return best_model


def generate_neighbor_ml(
    decision_vector: List[int],
    edges: List[List[int]],
    model_type: str = 'lgbm',
    use_hybrid: bool = False,
    iteration: int = 0,
    switch_interval: int = 5
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

    # Train models (only once at the beginning)
    if iteration == 0:
        global lgbm_model, xgboost_model  # Use global models for efficiency
        lgbm_model = train_lgbm_model(X, y)
        xgboost_model = train_xgboost_model(X, y)

    # Select model based on hybrid strategy
    if use_hybrid:
        if iteration % switch_interval < switch_interval // 2:
            model = lgbm_model
        else:
            model = xgboost_model
    else:
        model = lgbm_model if model_type == 'lgbm' else xgboost_model

    # Predict scores for potential neighbors
    neighbors = [generate_neighbor_swap(decision_vector, 0.2) for _ in range(100)]
    predictions = model.predict(neighbors)

    # Select the best neighbor based on predictions
    best_neighbor_idx = np.argmin(
        [torso_scorer([0, 0], pred) for pred in predictions]
    )
    return neighbors[best_neighbor_idx]

def generate_neighbor_multi_objective(decision_vector: List[int], edges: List[List[int]]) -> List[int]:
    """Generates a neighbor solution using multi-objective optimization."""
    n = len(decision_vector) - 1
    current_score = evaluate_solution(decision_vector, edges)

    def objective_function(x):
        """Objective function for minimization (combines size and width)."""
        neighbor = [int(round(i)) for i in x[:-1]] + [int(round(x[-1]))]
        return torso_scorer(current_score, evaluate_solution(neighbor, edges))

    # Use a global optimization method like differential evolution
    result = minimize(objective_function, decision_vector, method='Nelder-Mead')

    # Ensure the solution is valid
    neighbor = [int(round(i)) for i in result.x[:-1]] + [int(round(result.x[-1]))]
    neighbor[-1] = max(0, min(neighbor[-1], n - 1))  # Keep t within bounds
    return neighbor


def hill_climbing(
    edges: List[List[int]],
    max_iterations: int = 100,
    num_restarts: int = 10,
    perturbation_rate: float = 0.2,
    neighbor_generation_method: str = 'swap',
    use_hybrid_ml: bool = False,
    ml_switch_interval: int = 5,
) -> List[List[int]]:
    """Performs hill climbing to find a set of Pareto optimal solutions."""
    n = max(node for edge in edges for node in edge) + 1
    pareto_front = []

    for restart in range(num_restarts):
        print(f"Restart: {restart + 1}")

        # Generate random initial solution
        decision_vector = [i for i in range(n)] + [random.randint(0, n - 1)]

        # Evaluate initial solution
        current_score = evaluate_solution(decision_vector, edges)

        best_decision_vector = decision_vector[:]
        best_score = current_score[:]

        for i in range(max_iterations):
            print(f"Iteration: {i}, Best Score: {best_score}")

            # Generate neighbor solution using the selected method
            if neighbor_generation_method == 'lgbm':
                neighbor = generate_neighbor_ml(decision_vector, edges, 'lgbm', use_hybrid_ml, i, ml_switch_interval)
            elif neighbor_generation_method == 'xgboost':
                neighbor = generate_neighbor_ml(decision_vector, edges, 'xgboost', use_hybrid_ml, i, ml_switch_interval)
            elif neighbor_generation_method == 'multi_objective':
                neighbor = generate_neighbor_multi_objective(decision_vector, edges)
            else:  # Default to 'swap'
                neighbor = generate_neighbor_swap(decision_vector, perturbation_rate)

            # Evaluate neighbor solution
            neighbor_score = evaluate_solution(neighbor, edges)

            # Update current solution if neighbor is better or equal
            if dominates(neighbor_score, current_score) or neighbor_score == current_score:
                decision_vector = neighbor[:]
                current_score = neighbor_score[:]

            # Update best solution if current solution is better
            if dominates(current_score, best_score):
                best_decision_vector = decision_vector[:]
                best_score = current_score[:]

        pareto_front.append((best_decision_vector, best_score))

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

    return filtered_pareto_front


if __name__ == "__main__":
    random.seed(42)

    problem_id = "easy"
    edges = load_graph(problem_id)

    # Hill Climbing with different configurations
    # 1. Basic Swap
    pareto_front_swap = hill_climbing(edges, neighbor_generation_method='swap')

    # 2. LGBM
    pareto_front_lgbm = hill_climbing(edges, neighbor_generation_method='lgbm')

    # 3. XGBoost
    pareto_front_xgboost = hill_climbing(edges, neighbor_generation_method='xgboost')

    # 4. Hybrid LGBM/XGBoost
    pareto_front_hybrid = hill_climbing(edges, neighbor_generation_method='lgbm', use_hybrid_ml=True)

    # 5. Multi-objective Optimization (replace with your chosen algorithm)
    pareto_front_multi_objective = hill_climbing(edges, neighbor_generation_method='multi_objective')

    # Compare Results (add more comprehensive comparison and analysis)
    print("Pareto Front (Swap):", pareto_front_swap)
    print("Pareto Front (LGBM):", pareto_front_lgbm)
    print("Pareto Front (XGBoost):", pareto_front_xgboost)
    print("Pareto Front (Hybrid):", pareto_front_hybrid)
    print("Pareto Front (Multi-objective):", pareto_front_multi_objective)

    # ... (Choose the best Pareto front based on your evaluation)

    best_pareto_front = pareto_front_lgbm  # Example: Select LGBM results

    # Create Submission File
    submission = {
        "decisionVector": best_pareto_front,
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump([submission], f, indent=4)
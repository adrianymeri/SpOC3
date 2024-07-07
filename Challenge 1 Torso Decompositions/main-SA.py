import json
import random
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import urllib.request
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import os
import networkx as nx  # Add NetworkX for graph analysis

# Determine the number of available cores
num_cores = os.cpu_count()

# Use a fraction of the available cores to avoid oversubscription
n_jobs = max(1, num_cores // 3)  # Ensure at least one core is used

# Define the problem instances
problems = {
    "supereasy": "data/supereasy.gr",  # Add your local path here
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}


# Define a scorer function for multi-objective optimization
def torso_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Combines torso size and width into a single score for optimization."""
    size_weight = -1  # Prioritize minimizing size
    width_weight = -0.2  # Penalize width less than size
    return (size_weight * y_pred[:, 0] + width_weight * y_pred[:, 1]).mean()


def load_graph(problem_id: str) -> nx.Graph:
    """Loads the graph data for the given problem ID."""
    url = problems[problem_id]
    print(f"Loading graph data from: {url}")
    if url.startswith("http"):  # Load from URL
        with urllib.request.urlopen(url) as f:
            # Don't split lines here, pass them directly to parse_adjlist
            graph = nx.parse_adjlist(
                (line.decode("utf-8").strip() for line in f if not line.startswith(b"#")),
                nodetype=int
            )
    else:  # Load from local file
        with open(url, "r") as f:
            # Same here, don't split lines prematurely
            graph = nx.parse_adjlist(
                (line.strip() for line in f if not line.strip().startswith("#")),
                nodetype=int
            )
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def calculate_torso_size(decision_vector: List[int]) -> int:
    """Calculates the size of the torso for the given decision vector."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    return n - t


def calculate_torso_width(decision_vector: List[int], graph: nx.Graph) -> int:
    """Calculates the width of the torso for the given decision vector and graph."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    permutation = decision_vector[:-1]

    # Use NetworkX's efficient neighborhood access
    max_outdegree = 0
    for i in range(t, n):
        outdegree = sum(1 for j in graph.neighbors(permutation[i]) if j in permutation[t:] and permutation.index(j) > i)
        max_outdegree = max(max_outdegree, outdegree)

    return max_outdegree


def evaluate_solution(
    decision_vector: List[int], graph: nx.Graph
) -> List[float]:
    """Evaluates the given solution and returns the torso size and width."""
    torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, graph)
    return [torso_size, torso_width]


def generate_neighbor_swap(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by swapping two random elements within the torso."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    num_perturbations = max(1, int(perturbation_rate * (n - t)))

    neighbor = decision_vector.copy()

    for _ in range(num_perturbations):
        i = random.randint(t, n - 1)
        j = random.randint(t, n - 1)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def generate_neighbor_insert(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor by randomly inserting an element from the torso to another position within the torso."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    neighbor = decision_vector.copy()

    i = random.randint(t, n - 1)  # Index of element to move
    j = random.randint(t, n)  # Target index (inclusive)

    # Insert at j, shifting elements to the right
    neighbor.insert(j, neighbor.pop(i))

    # Recalculate torso size (t) - This is crucial!
    neighbor[-1] = len(neighbor) - 1 - neighbor[:-1].index(neighbor[-2])

    return neighbor

def generate_neighbor_shuffle(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution by shuffling a portion of the decision vector within the torso."""
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    t = neighbor[-1]
    start = random.randint(t, max(t, n - 2))
    end = random.randint(start + 1, n)
    random.shuffle(neighbor[start:end])
    return neighbor


def generate_neighbor_torso_shift(
    decision_vector: List[int], max_shift: int = 5
) -> List[int]:
    """Generates a neighbor by shifting the torso position within a limited range."""
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    t = neighbor[-1]
    new_t = random.randint(max(0, t - max_shift), min(n - 1, t + max_shift))
    neighbor[-1] = new_t
    return neighbor


def generate_neighbor_2opt(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution using 2-opt within the torso."""
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    t = neighbor[-1]

    i = random.randint(t, max(t, n - 3))
    j = random.randint(i + 1, n)

    neighbor[i:j] = neighbor[i:j][::-1]
    return neighbor


def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )

def generate_initial_solution(graph: nx.Graph) -> List[int]:
    """Generates an initial solution using a greedy approach."""
    n = graph.number_of_nodes()
    unplaced_nodes = set(graph.nodes())
    placement_order = []

    while unplaced_nodes:
        # Find node with the fewest unplaced neighbors
        best_node = min(
            unplaced_nodes, key=lambda node: sum(1 for neighbor in graph.neighbors(node) if neighbor in unplaced_nodes)
        )
        placement_order.append(best_node)
        unplaced_nodes.remove(best_node)

    # Initial torso size is set to a random value for now
    t = random.randint(0, n - 1) 
    return placement_order + [t] 

def train_model(
    X: np.ndarray, y: np.ndarray, model_type: str = "lgbm"
) -> MultiOutputRegressor:
    print(f"Training {model_type} model...")
    if model_type == "lgbm":
        model = MultiOutputRegressor(
            LGBMRegressor(random_state=42, n_jobs=n_jobs, verbose=-1)
        )
        param_grid = {
            "estimator__n_estimators": [200, 500, 1000],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [7, 9, 11],
            "estimator__num_leaves": [31, 50, 75],
            "estimator__min_data_in_leaf": [10, 20, 30],
        }
    else:  # XGBoost
        model = MultiOutputRegressor(
            XGBRegressor(random_state=42, n_jobs=n_jobs, verbose=-1)
        )
        param_grid = {
            "estimator__n_estimators": [200, 500, 1000],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__max_depth": [7, 9, 11],
            "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
            "estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        }

    kfold = KFold(
        n_splits=3, shuffle=True, random_state=42
    )  # Reduced splits for speed
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=5,  # Reduced iterations for speed
        scoring=make_scorer(torso_scorer, greater_is_better=False),
        cv=kfold,
        random_state=42,
        n_jobs=n_jobs,
    )
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    best_score = -random_search.best_score_
    print(f"Best {model_type} model: {best_model}")
    print(f"{model_type} training completed with best score: {best_score}")
    return best_model


def generate_neighbor_ml(
    decision_vector: List[int],
    graph: nx.Graph,
    model: MultiOutputRegressor,
    num_neighbors: int = 100,
) -> List[int]:
    """Generates neighbors using ML models for prediction and selects the best."""
    neighbors = []
    for _ in range(num_neighbors // 5):  # Generate diverse neighbors
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.1))
        neighbors.append(generate_neighbor_swap(decision_vector.copy(), 0.2))
        neighbors.append(generate_neighbor_shuffle(decision_vector.copy()))
        neighbors.append(generate_neighbor_torso_shift(decision_vector.copy()))
        neighbors.append(generate_neighbor_2opt(decision_vector.copy()))

    neighbors_np = np.array(neighbors)
    predictions = model.predict(neighbors_np)
    scores = [
        torso_scorer(np.array([[0, 0]]), pred.reshape(1, -1)) for pred in predictions
    ]
    best_neighbor_idx = np.argmin(scores)
    return neighbors[best_neighbor_idx]


def choose_neighbor_generation_method(
    current_iteration: int,
    initial_exploration_iterations: int,
    ml_switch_iteration: int,
    operator_weights: List[float],
) -> str:
    """Chooses the neighbor generation method based on exploration/exploitation strategy."""
    if current_iteration < initial_exploration_iterations:
        method = random.choices(
            ["swap", "shuffle", "torso_shift", "2opt", "insert"],
            weights=[0.3, 0.3, 0.2, 0.1, 0.1],
        )[0]
        print(f"Iteration {current_iteration}: Exploring with {method}")
        return method
    elif current_iteration >= ml_switch_iteration:
        print(f"Iteration {current_iteration}: Exploiting with ML model")
        return "ml"
    else:
        method = random.choices(
            ["swap", "shuffle", "torso_shift", "2opt", "insert"],
            weights=operator_weights,
        )[0]
        print(f"Iteration {current_iteration}: Using weighted operator: {method}")
        return method


def evaluate_neighbors_parallel(
    neighbors: List[List[int]], graph: nx.Graph
) -> List[List[float]]:
    """Evaluates a list of neighbor solutions in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_solution)(neighbor, graph) for neighbor in neighbors
    )
    return results


def simulated_annealing_single_restart(
    graph: nx.Graph,
    restart: int,
    problem_id: str,
    max_iterations: int = 20000,
    initial_temperature: float = 500.0,
    cooling_rate: float = 0.99,
    initial_exploration_iterations: int = 2000,
    ml_switch_iteration: int = 4000,  # Iteration to start using ML model
    save_interval: int = 500,
    operator_change_interval: int = 1000,
    neighbor_batch_size: int = 20,
    lgbm_model: MultiOutputRegressor = None,  # Pass models as arguments
    xgboost_model: MultiOutputRegressor = None,
) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    """Performs a single restart of the simulated annealing algorithm."""
    n = graph.number_of_nodes()

    current_solution = generate_initial_solution(graph)
    current_score = evaluate_solution(current_solution, graph)

    best_solution = current_solution[:]
    best_score = current_score[:]

    temperature = initial_temperature

    initial_perturbation_rate = 0.5
    perturbation_rate_decay = 0.99

    X = []
    y = []

    operator_weights = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # Added weight for 'insert'

    for i in range(max_iterations):
        if i % 100 == 0:
            print(f"Restart {restart + 1}, Iteration {i + 1}...")

        neighbor_generation_method = choose_neighbor_generation_method(
            i, initial_exploration_iterations, ml_switch_iteration, operator_weights
        )

        if (
            i >= initial_exploration_iterations
            and i % operator_change_interval == 0
        ):
            operator_weights = update_operator_weights(
                X,
                y,
                operator_weights,
                initial_exploration_iterations,
                ml_switch_iteration,
            )
            print(
                f"Restart {restart + 1}, Iteration {i + 1}: Updated operator weights: {operator_weights}"
            )

        # Generate neighbors based on the chosen method
        if neighbor_generation_method == "ml":
            if lgbm_model is None or xgboost_model is None:
                print(
                    f"Restart {restart + 1}, Iteration {i + 1}: ML models not ready yet, using random operator"
                )
                # If models haven't been trained yet, use a regular operator
                neighbor_generation_method = random.choices(
                    ["swap", "shuffle", "torso_shift", "2opt", "insert"],
                    weights=[0.3, 0.3, 0.2, 0.1, 0.1],
                )[0]
            else:
                # Use ML models to generate a neighbor
                model_choice = random.choice(["lgbm", "xgboost", "hybrid"])
                if model_choice == "lgbm":
                    neighbor = generate_neighbor_ml(
                        current_solution, graph, lgbm_model
                    )
                elif model_choice == "xgboost":
                    neighbor = generate_neighbor_ml(
                        current_solution, graph, xgboost_model
                    )
                else:  # Hybrid - choose model based on iteration
                    neighbor = generate_neighbor_ml(
                        current_solution,
                        graph,
                        lgbm_model if i % 2 == 0 else xgboost_model,
                    )
                neighbor_score = evaluate_solution(neighbor, graph)

                # Acceptance based on Metropolis Hastings (always accept better)
                delta_score = (neighbor_score[0] - current_score[0]) + 0.5 * (
                    neighbor_score[1] - current_score[1]
                )
                acceptance_probability = np.exp(-delta_score / temperature)
                if delta_score < 0 or random.random() < acceptance_probability:
                    current_solution = neighbor[:]
                    current_score = neighbor_score[:]
                    print(
                        f"Restart {restart + 1}, Iteration {i + 1}: Accepted ML-generated neighbor with score {neighbor_score}"
                    )

        # If not using ML for neighbor generation in this iteration
        if neighbor_generation_method != "ml":
            neighbors = []
            for _ in range(neighbor_batch_size):
                if neighbor_generation_method == "swap":
                    neighbor = generate_neighbor_swap(
                        current_solution, initial_perturbation_rate
                    )
                    initial_perturbation_rate *= perturbation_rate_decay
                elif neighbor_generation_method == "shuffle":
                    neighbor = generate_neighbor_shuffle(current_solution)
                elif neighbor_generation_method == "torso_shift":
                    neighbor = generate_neighbor_torso_shift(current_solution)
                elif neighbor_generation_method == "2opt":
                    neighbor = generate_neighbor_2opt(current_solution)
                elif neighbor_generation_method == "insert":
                    neighbor = generate_neighbor_insert(current_solution)
                neighbors.append(neighbor)

            neighbor_scores = evaluate_neighbors_parallel(neighbors, graph)
            X.extend(neighbors)
            y.extend(neighbor_scores)

            for neighbor, neighbor_score in zip(neighbors, neighbor_scores):
                if dominates(neighbor_score, best_score):
                    best_solution = neighbor[:]
                    best_score = neighbor_score[:]
                    print(
                        f"Restart {restart + 1} - Iteration {i + 1}: New best solution found - Score: {best_score}"
                    )

                # Metropolis Hastings acceptance
                delta_score = (neighbor_score[0] - current_score[0]) + 0.5 * (
                    neighbor_score[1] - current_score[1]
                )
                acceptance_probability = np.exp(-delta_score / temperature)
                if delta_score < 0 or random.random() < acceptance_probability:
                    current_solution = neighbor[:]
                    current_score = neighbor_score[:]

        temperature *= cooling_rate

        if (i + 1) % save_interval == 0:
            create_submission_file(
                best_solution,
                problem_id,
                f"intermediate_solution_{restart + 1}_iter_{i + 1}.json",
            )

    X_np = np.array(X)
    y_np = np.array(y)

    print(
        f"Restart {restart + 1} completed. Best solution: {best_solution}, Score: {best_score}"
    )
    return (
        best_solution,
        best_score,
        X_np,
        y_np,
    )  # Return X_np and y_np


def simulated_annealing(
    graph: nx.Graph,
    problem_id: str,
    max_iterations: int = 1000,
    num_restarts: int = 10,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    initial_exploration_iterations: int = 20,
    ml_switch_iteration: int = 25,  # When to switch to ML
    save_interval: int = 50,
    operator_change_interval: int = 100,
    n_jobs: int = n_jobs,  # Use the calculated n_jobs
    neighbor_batch_size: int = 10,
) -> List[List[int]]:
    """Performs simulated annealing to find a set of Pareto optimal solutions."""
    pareto_front = []
    start_time = time.time()

    # Initialize ML models outside the parallel loop
    lgbm_model = None
    xgboost_model = None

    # Collect training data from all restarts
    all_X = []
    all_y = []

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulated_annealing_single_restart)(
            graph,
            restart,
            problem_id,
            max_iterations,
            initial_temperature,
            cooling_rate,
            initial_exploration_iterations,
            ml_switch_iteration,
            save_interval,
            operator_change_interval,
            neighbor_batch_size,
            lgbm_model,  # Pass models to restarts
            xgboost_model,
        )
        for restart in range(num_restarts)
    )
    for result in results:
        (
            solution,
            score,
            restart_X,
            restart_y,
        ) = result  # Get X and y from each restart
        pareto_front.append((solution, score))
        all_X.extend(restart_X)
        all_y.extend(restart_y)

    print(f"Pareto Front: {pareto_front}")

    # Train models after all restarts have finished
    if all_X:
        print("Training ML models with data from all restarts...")
        X_np = np.array(all_X)
        y_np = np.array(all_y)
        lgbm_model = train_model(X_np, y_np, "lgbm")
        xgboost_model = train_model(X_np, y_np, "xgboost")

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
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    return filtered_pareto_front


def create_submission_file(
    decision_vector: List[int],
    problem_id: str,
    filename: str = "submission.json",
):
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
    ml_switch_iteration: int,
) -> List[float]:
    """Updates the operator weights based on their past performance."""
    operator_improvements = defaultdict(lambda: {"count": 0, "total_improvement": 0})
    for i in range(initial_exploration_iterations, len(X) - 1):
        if (
            i < ml_switch_iteration
        ):  # Only update weights based on non-ML iterations
            previous_score = y[i - 1]
            current_score = y[i]
            improvement = (previous_score[0] - current_score[0]) + 0.5 * (
                previous_score[1] - current_score[1]
            )
            if X[i] != X[i - 1]:
                if X[i][-1] != X[i - 1][-1]:
                    operator_index = 2  # torso_shift
                elif is_shuffled(X[i], X[i - 1]):
                    operator_index = 1  # shuffle
                elif is_2opt(X[i], X[i - 1]):
                    operator_index = 3  # 2-opt
                elif is_insert(X[i], X[i - 1]):
                    operator_index = 4  # insert
                else:
                    operator_index = 0  # swap

                operator_improvements[operator_index]["count"] += 1
                operator_improvements[operator_index][
                    "total_improvement"
                ] += improvement

    for i in range(5):  # Update weights for all 5 operators
        op_data = operator_improvements[i]
        if op_data["count"] > 0:
            operator_weights[i] = op_data["total_improvement"] / op_data["count"]
        else:
            operator_weights[i] = 1.0  # Default if no data yet

    total_weight = sum(operator_weights)
    operator_weights = [w / total_weight for w in operator_weights]

    return operator_weights


def is_shuffled(list1: List[int], list2: List[int]) -> bool:
    """Checks if two lists are permutations of each other."""
    return sorted(list1) == sorted(list2)


def is_2opt(list1: List[int], list2: List[int]) -> bool:
    """Checks if two lists are different by a single 2-opt move."""
    differences = sum(1 for a, b in zip(list1, list2) if a != b)
    return differences == 4  # 2-opt changes 4 elements


def is_insert(list1: List[int], list2: List[int]) -> bool:
    """Checks if two lists differ by a single insert operation."""
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(list1, list2)) if a != b]
    if len(differences) != 2:  # One element moved to another position
        return False
    i1, a1, b1 = differences[0]
    i2, a2, b2 = differences[1]
    return a1 == b2 and (i1 + 1) == i2  # Check if elements were indeed inserted


if __name__ == "__main__":
    random.seed(42)

    chosen_problem = (
        input("Choose problem difficulty (supereasy, easy, medium, hard): ")
        .lower()
        .strip()
    )

    while chosen_problem not in problems:
        print(
            "Invalid problem difficulty. Please choose from 'supereasy', 'easy', 'medium', or 'hard'."
        )
        chosen_problem = (
            input("Choose problem difficulty (supereasy, easy, medium, hard): ")
            .lower()
            .strip()
        )

    print(f"Processing problem: {chosen_problem}")
    graph = load_graph(chosen_problem) 

    pareto_front = simulated_annealing(
        graph, # Pass the NetworkX graph 
        chosen_problem,
        max_iterations=10000,  # Significantly increased iterations
        num_restarts=20,  # Increased restarts
        save_interval=2000,  # Save less frequently
        operator_change_interval=500,  # Update operator weights less frequently
        initial_exploration_iterations=1000,  # Longer exploration
        ml_switch_iteration=2000,  # Switch to ML after more exploration
        n_jobs=n_jobs,  # Use calculated n_jobs
        neighbor_batch_size=20,  # Increased batch size
    )

    for i, solution in enumerate(pareto_front):
        create_submission_file(
            solution, chosen_problem, f"{chosen_problem}_final_solution_{i + 1}.json"
        )
    print(f"All submission files created successfully!")

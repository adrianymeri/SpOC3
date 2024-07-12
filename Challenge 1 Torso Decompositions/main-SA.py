import json
import random
import time
from typing import List, Tuple, Dict

import numpy as np
import urllib.request
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from joblib import Parallel, delayed

# Define the problem instances
problems = {
    "supereasy": "data/toy.gr",
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Helper Functions ---


def load_graph(problem_id: str) -> List[List[int]]:
    """Loads the graph data for the given problem ID."""
    file_path = problems[problem_id]
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


def calculate_torso_size(decision_vector: List[int]) -> int:
    """Calculates the size of the torso for the given decision vector."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    return n - t


def calculate_torso_width(decision_vector: List[int], edges: List[List[int]]) -> int:
    """Calculates the width of the torso."""
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    permutation = decision_vector[:-1]

    # Find the maximum node index for correct adj_list size
    max_node_index = max(node for edge in edges for node in edge) + 1 

    # Initialize adj_list with the correct size
    adj_list = [[] for _ in range(max_node_index)] 

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


def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return all(x <= y for x, y in zip(score1, score2)) and any(
        x < y for x, y in zip(score1, score2)
    )


def create_submission_file(
    decision_vector, problem_id, filename="submission.json"
):
    """Creates a valid submission file."""
    submission = {
        "decisionVector": [decision_vector],  # Wrap in a list for multiple solutions
        "problem": problem_id,  # Use the provided problem_id
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")

# --- Neighborhood Operators ---


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


def generate_neighbor_2opt(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution using the 2-opt operator."""
    n = len(decision_vector) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor = decision_vector[:i] + decision_vector[i:j][::-1] + decision_vector[j:]
    return neighbor


def generate_neighbor_shuffle(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by shuffling a portion of the decision vector."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    shuffle_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - shuffle_length)
    neighbor[start_index : start_index + shuffle_length] = random.sample(
        neighbor[start_index : start_index + shuffle_length], shuffle_length
    )
    return neighbor


def generate_neighbor_torso_shift(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor solution by shifting the torso position."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    new_torso_position = random.randint(0, n - 1)
    neighbor[-1] = new_torso_position
    return neighbor


def generate_neighbor_insert(decision_vector: List[int]) -> List[int]:
    """Generates a neighbor by randomly inserting an element at a different position."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 1)
    j = random.randint(0, n)
    element = neighbor.pop(i)
    neighbor.insert(j, element)
    return neighbor


def generate_neighbor_inversion(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor by reversing a subsequence of the decision vector."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    inversion_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - inversion_length)
    neighbor[start_index : start_index + inversion_length] = reversed(
        neighbor[start_index : start_index + inversion_length]
    )
    return neighbor

# --- End of Neighborhood Operators ---


def acceptance_probability(
    old_score: List[float], new_score: List[float], temperature: float
) -> float:
    """Calculates the acceptance probability in Simulated Annealing."""
    # Directly compare scores using weighted sum (smaller is better)
    size_weight = -1  # Prioritize minimizing size
    width_weight = -0.5  # Penalize width but less than size
    old_weighted_score = size_weight * old_score[0] + width_weight * old_score[1]
    new_weighted_score = size_weight * new_score[0] + width_weight * new_score[1]
    delta_score = new_weighted_score - old_weighted_score
    return np.exp(delta_score / temperature)


def simulated_annealing_single_restart(
    edges: List[List[int]],
    restart: int,
    initial_solution: List[int] = None,  # Optional initial solution
    max_iterations: int = 10000,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    neighbor_operators: List[str] = [
        "swap",
        "2opt",
        "shuffle",
        "torso_shift",
        "insert",
        "inversion",
    ],
    save_interval: int = 50,
) -> Tuple[List[int], List[float]]:
    """Performs a single restart of the Simulated Annealing algorithm."""
    n = max(node for edge in edges for node in edge) + 1

    # Use provided initial solution or generate random one
    if initial_solution is not None:
        current_solution = initial_solution[:]
    else:
        current_solution = [i for i in range(n)] + [random.randint(0, n - 1)]
    current_score = evaluate_solution(current_solution, edges)

    best_solution = current_solution[:]
    best_score = current_score[:]

    temperature = initial_temperature

    print(
        f"Restart {restart+1} - Initial Solution: {current_solution}, Score: {current_score}"
    )

    for i in range(max_iterations):
        # Adaptive Perturbation: Adjust perturbation rate based on temperature
        perturbation_rate = 0.2 * (temperature / initial_temperature)

        # Choose a random neighbor operator
        operator = random.choice(neighbor_operators)

        # Generate neighbor solution using the selected operator
        if operator == "swap":
            neighbor = generate_neighbor_swap(current_solution, perturbation_rate)
        elif operator == "2opt":
            neighbor = generate_neighbor_2opt(current_solution)
        elif operator == "shuffle":
            neighbor = generate_neighbor_shuffle(current_solution, perturbation_rate)
        elif operator == "torso_shift":
            neighbor = generate_neighbor_torso_shift(current_solution)
        elif operator == "insert":
            neighbor = generate_neighbor_insert(current_solution)
        elif operator == "inversion":
            neighbor = generate_neighbor_inversion(current_solution, perturbation_rate)
        else:
            raise ValueError(f"Invalid neighbor operator: {operator}")

        neighbor_score = evaluate_solution(neighbor, edges)

        # Accept the neighbor if it's better or based on probability
        if (
            dominates(neighbor_score, current_score)
            or random.random()
            < acceptance_probability(current_score, neighbor_score, temperature)
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

        # Cool down the temperature
        temperature *= cooling_rate

        # Save intermediate solutions
        if (i + 1) % save_interval == 0:
            create_submission_file(
                best_solution,
                problem_id,
                f"intermediate_solution_{restart+1}_iter_{i+1}.json",
            )

    print(
        f"Restart {restart+1} completed. Best solution: {best_solution}, Score: {best_score}"
    )
    return best_solution, best_score


def simulated_annealing(
    edges: List[List[int]],
    initial_solutions: List[List[int]] = None,  # Optional list of initial solutions
    max_iterations: int = 1000,
    num_restarts: int = 10,
    initial_temperature: float = 400.0,
    cooling_rate: float = 0.95,
    neighbor_operators: List[str] = [
        "swap",
        "2opt",
        "shuffle",
        "torso_shift",
        "insert",
        "inversion",
    ],
    save_interval: int = 50,
    n_jobs: int = -1,
) -> List[List[int]]:
    """Performs Simulated Annealing to find a set of Pareto optimal solutions."""
    pareto_front = []
    start_time = time.time()

    # If initial solutions are provided, use them for restarts
    if initial_solutions is not None:
        num_restarts = len(initial_solutions)
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulated_annealing_single_restart)(
                edges,
                restart,
                initial_solution=initial_solutions[restart],  # Use provided solution
                max_iterations=max_iterations,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                neighbor_operators=neighbor_operators,
                save_interval=save_interval,
            )
            for restart in range(num_restarts)
        )
    else:
        # Run Simulated Annealing with multiple restarts in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulated_annealing_single_restart)(
                edges,
                restart,
                max_iterations=max_iterations,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                neighbor_operators=neighbor_operators,
                save_interval=save_interval,
            )
            for restart in range(num_restarts)
        )

    # Results now contain solutions from all restarts
    for result in results:
        pareto_front.append(result)

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


# --- Genetic Algorithm using pymoo ---


class TorsoProblem(ElementwiseProblem):
    def __init__(self, edges: List[List[int]]):
        super().__init__(
            n_var=len(edges) + 1,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(len(edges) + 1),
            xu=np.array([len(edges)] * len(edges) + [len(edges) - 1]),
        )
        self.edges = edges

    def _evaluate(self, x, out, *args, **kwargs):
        decision_vector = x.astype(int).tolist()
        torso_size = calculate_torso_size(decision_vector)
        torso_width = calculate_torso_width(decision_vector, self.edges)
        out["F"] = [torso_size, torso_width]


def run_genetic_algorithm(
    edges: List[List[int]], pop_size: int = 1500, n_gen: int = 1500
):
    """Runs the genetic algorithm to find a set of Pareto optimal solutions."""
    problem = TorsoProblem(edges)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0, eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        save_history=True,
        verbose=True,
    )

    # Extract and return Pareto front solutions
    pareto_front = [s.X.astype(int).tolist() for s in res.pop.get("X")]
    return pareto_front


# --- Main Execution ---

if __name__ == "__main__":
    random.seed(42)
    problem_id = input(
        "Select problem instance (supereasy, easy, medium, hard): "
    )  # Get input from the user
    edges = load_graph(problem_id)

    # --- Hybrid Approach: GA to initialize SA ---

    # 1. Run GA for a short time to get initial solutions
    print("Starting Genetic Algorithm for initialization...")
    ga_initial_solutions = run_genetic_algorithm(edges, pop_size=100, n_gen=50)

    # 2. Use GA solutions to initialize SA
    print("Starting Simulated Annealing with GA initialization...")
    sa_pareto_front = simulated_annealing(
        edges,
        initial_solutions=ga_initial_solutions,  # Pass GA solutions to SA
        max_iterations=2000,  # Adjust as needed
        num_restarts=len(ga_initial_solutions),  # Use the number of GA solutions
        initial_temperature=200.0,  # Adjust as needed
        cooling_rate=0.98,  # Adjust as needed
        neighbor_operators=[
            "swap",
            "2opt",
            "shuffle",
            "torso_shift",
            "insert",
            "inversion",
        ],
        save_interval=100,
    )

    # --- End of Hybrid Approach ---

    # Create Final Submission Files for the combined Pareto front
    for i, solution in enumerate(sa_pareto_front):
        create_submission_file(
            solution, problem_id, f"final_solution_hybrid_{i+1}.json"
        )
    print("All hybrid submission files created successfully!")

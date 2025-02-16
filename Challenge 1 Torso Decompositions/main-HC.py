import json
import random
import time
from typing import List, Tuple

import numpy as np
import pygmo as pg  # For hypervolume calculation
from loguru import logger

# Logger configuration
log_format = "<green>{time:YYYY-MM-DD@HH:mm:ss}</green> | <level>{message}</level>"
log_config = {"handlers": [{"sink": sys.stdout, "format": log_format}]}
logger.configure(**log_config)

# ... (graph_torso_udp class definition - same as before) ...

def generate_neighbor_swap(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    i, j = random.sample(range(n), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def generate_neighbor_shuffle(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    sublist_length = random.randint(1, int(0.2 * n))  # Adaptive sublist length
    start_index = random.randint(0, n - sublist_length)
    sublist = neighbor[start_index: start_index + sublist_length].copy() #Important: Create a copy to avoid modifying original array
    random.shuffle(sublist)
    neighbor[start_index: start_index + sublist_length] = sublist
    return neighbor

def generate_neighbor_torso_shift(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    shift_amount = random.randint(1, int(0.2 * n)) # Adaptive shift amount
    neighbor[-1] = max(0, min(n - 1, neighbor[-1] + random.choice([-shift_amount, shift_amount])))
    return neighbor

def generate_neighbor_2opt(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor[i:j+1] = neighbor[i:j+1][::-1]  # Corrected 2-opt
    return neighbor

def generate_neighbor_insert(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    index1 = random.randint(0, n - 1)
    index2 = random.randint(0, n)
    value = neighbor[index1]
    neighbor = np.delete(neighbor, index1)
    neighbor = np.insert(neighbor, index2, value)
    return neighbor

def generate_neighbor_inversion(decision_vector: np.ndarray) -> np.ndarray:
    neighbor = decision_vector.copy()
    n = len(neighbor) - 1
    sublist_length = random.randint(1, int(0.2 * n))  # Adaptive sublist length
    start_index = random.randint(0, n - sublist_length)
    neighbor[start_index: start_index + sublist_length] = neighbor[start_index: start_index + sublist_length][::-1]
    return neighbor

def generate_neighbor_all(decision_vector: np.ndarray) -> np.ndarray:
    operators = [
        generate_neighbor_swap,
        generate_neighbor_shuffle,
        generate_neighbor_torso_shift,
        generate_neighbor_2opt,
        generate_neighbor_insert,
        generate_neighbor_inversion
    ]
    op = random.choice(operators)
    return op(decision_vector)

def generate_random_solution(n: int) -> np.ndarray:
    """Generates a random solution (permutation + threshold)."""
    perm = np.arange(n)
    np.random.shuffle(perm)
    return np.concatenate([perm, [random.randint(0, n - 1)]])

def dominates(score1: List[float], score2: List[float]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return all(x <= y for x, y in zip(score1, score2)) and any(x < y for x, y in zip(score1, score2))

def update_pareto_front(pareto_front: List[np.ndarray], solution: np.ndarray, udp: graph_torso_udp) -> List[np.ndarray]:
    """Updates the Pareto front with a new solution."""
    fitness = np.array(udp.fitness(solution))  # Calculate fitness only once
    new_solution_entry = np.concatenate([solution, fitness])  # Store solution and fitness together

    if not pareto_front:  # Initialize Pareto front if it's empty
        return [new_solution_entry]
    
    updated_pareto_front = []
    is_dominated = False

    for existing_solution_entry in pareto_front:
        existing_fitness = existing_solution_entry[-2:]  # Extract existing fitness
        if all(fitness <= existing_fitness) and any(fitness < existing_fitness):
            is_dominated = True
            break  # New solution is dominated, no need to add it
        elif all(fitness >= existing_fitness) and any(fitness > existing_fitness):
            continue  # Existing solution is dominated by the new one
        else:
            updated_pareto_front.append(existing_solution_entry)  # Keep non-dominated existing solutions
    
    if not is_dominated:
      updated_pareto_front.append(new_solution_entry)

    return updated_pareto_front

def combine_scores(points: np.ndarray, udp: graph_torso_udp) -> float:
    """Combines Pareto front scores into a single hypervolume score."""
    ref_point = np.array([udp.n_nodes, udp.n_nodes])  # Reference point for hypervolume
    if len(points) == 0:
        return 0.0
    hv = pg.hypervolume(points)
    return -hv.compute(ref_point)  # Negative because we want to maximize


def hill_climbing(udp: graph_torso_udp, max_iterations: int = 1000, num_restarts: int = 20):
    """Hill climbing with Pareto front and hypervolume calculation."""

    best_hypervolume = -np.inf
    best_pareto_front = []

    for restart in range(num_restarts):
        pareto_front = []  # Initialize Pareto front for each restart
        n = udp.n_nodes
        S = generate_random_solution(n)
        pareto_front = update_pareto_front(pareto_front, S, udp)

        for i in range(max_iterations):
            R = generate_neighbor_all(S)  # Generate a neighbor
            pareto_front = update_pareto_front(pareto_front, R, udp) # Update pareto front

            S = R.copy()  # Move to the neighbor

            if (i+1) % 100 == 0:  # Log every 100 iterations
                hv = combine_scores(np.array([p[-2:] for p in pareto_front]), udp) #Hypervolume calculation
                logger.info(f"Restart {restart+1}, Iteration {i+1}: Pareto front size = {len(pareto_front)}, Hypervolume = {hv:.3f}")

        hv = combine_scores(np.array([p[-2:] for p in pareto_front]), udp) #Hypervolume calculation
        logger.info(f"Restart {restart+1} finished. Pareto front size = {len(pareto_front)}, Hypervolume = {hv:.3f}")

        if hv > best_hypervolume:
            best_hypervolume = hv
            best_pareto_front = pareto_front

    return best_pareto_front, best_hypervolume


if __name__ == "__main__":
    import sys
    random.seed(42)
    np.random.seed(42)

    problem_id = input("Select a problem instance (easy, medium, hard): ").lower()
    udp = graph_torso_udp(f"data/spoc3/torso/{problem_id}.gr")  # Load graph using UDP class

    best_pareto_front, best_hypervolume = hill_climbing(udp, max_iterations=5

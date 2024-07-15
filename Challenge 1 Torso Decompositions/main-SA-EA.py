import json
import random
import time
from typing import List, Tuple

import numpy as np
import urllib.request
from multiprocessing import Pool

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
) -> Tuple[int, int]:
    """Evaluates the given solution and returns the torso size and width."""
    torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, edges)
    return torso_size, torso_width


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


def generate_neighbor_shuffle(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by shuffling a sublist of elements."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    sublist_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - sublist_length)
    sublist = neighbor[start_index : start_index + sublist_length]
    random.shuffle(sublist)
    neighbor[start_index : start_index + sublist_length] = sublist
    return neighbor


def generate_neighbor_torso_shift(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by shifting the torso threshold."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    shift_amount = int(perturbation_rate * n)
    neighbor[-1] = max(
        0, min(n - 1, neighbor[-1] + random.randint(-shift_amount, shift_amount))
    )
    return neighbor


def generate_neighbor_2opt(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution using the 2-opt heuristic."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor[i:j] = neighbor[i:j][::-1]  # Reverse the sublist
    return neighbor


def generate_neighbor_insert(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by inserting an element at a random position."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    index1 = random.randint(0, n - 1)
    index2 = random.randint(0, n)
    value = neighbor.pop(index1)
    neighbor.insert(index2, value)
    return neighbor


def generate_neighbor_inversion(
    decision_vector: List[int], perturbation_rate: float = 0.2
) -> List[int]:
    """Generates a neighbor solution by inverting a sublist of elements."""
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    sublist_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - sublist_length)
    neighbor[start_index : start_index + sublist_length] = neighbor[
        start_index : start_index + sublist_length
    ][::-1]
    return neighbor


def dominates(score1: Tuple[int, int], score2: Tuple[int, int]) -> bool:
    """Checks if score1 dominates score2 in multi-objective optimization."""
    return (score1[0] <= score2[0] and score1[1] <= score2[1]) and (
        score1[0] < score2[0] or score1[1] < score2[1]
    )


def generate_neighbor(
    decision_vector: List[int],
    edges: List[List[int]],
    temperature: float,
    perturbation_rate: float = 0.2,
) -> List[int]:
    """Generates a neighbor solution using a combination of operators
    and Simulated Annealing acceptance criteria.
    """
    neighbor_functions = [
        generate_neighbor_swap,
        generate_neighbor_shuffle,
        generate_neighbor_torso_shift,
        generate_neighbor_2opt,
        generate_neighbor_insert,
        generate_neighbor_inversion,
    ]

    current_size, current_width = evaluate_solution(decision_vector, edges)
    current_score = (current_size, current_width)

    # Try different operators with decreasing probability based on temperature
    for operator in sorted(neighbor_functions, key=lambda x: random.random()):
        neighbor = operator(decision_vector, perturbation_rate)
        neighbor_size, neighbor_width = evaluate_solution(neighbor, edges)
        neighbor_score = (neighbor_size, neighbor_width)

        # Metropolis acceptance criterion
        if dominates(neighbor_score, current_score) or random.random() < np.exp(
            -(
                (neighbor_size - current_size)
                + 0.5 * (neighbor_width - current_width)
            )
            / temperature
        ):
            print(
                f"    Neighbor accepted - Size: {neighbor_size}, Width: {neighbor_width}"
            )
            return neighbor

    print(
        f"    No better neighbor found - Size: {current_size}, Width: {current_width}"
    )
    return decision_vector  # Return current solution if no better neighbor is found


def simulated_annealing(
    edges: List[List[int]],
    initial_temperature: float = 1000,
    cooling_rate: float = 0.95,
    max_iterations: int = 1000,
    perturbation_rate: float = 0.2,
    early_stopping_iterations: int = 100,
) -> Tuple[List[int], Tuple[int, int]]:
    """Performs Simulated Annealing to find a good solution."""
    n = max(node for edge in edges for node in edge) + 1

    # Generate random initial solution
    decision_vector = [i for i in range(n)] + [random.randint(0, n - 1)]
    best_decision_vector = decision_vector[:]
    best_size, best_width = evaluate_solution(decision_vector, edges)
    best_score = (best_size, best_width)

    temperature = initial_temperature

    iterations_without_improvement = 0
    for i in range(max_iterations):
        print(f"Iteration {i+1}, Temperature: {temperature:.2f}")
        # Generate neighbor solution
        decision_vector = generate_neighbor(
            decision_vector, edges, temperature, perturbation_rate
        )
        current_size, current_width = evaluate_solution(decision_vector, edges)
        current_score = (current_size, current_width)

        # Update best solution
        if dominates(current_score, best_score):
            best_decision_vector = decision_vector[:]
            best_score = current_score
            best_size, best_width = best_score
            iterations_without_improvement = 0
            print(
                f"  New best solution found - Size: {best_size}, Width: {best_width}"
            )
        else:
            iterations_without_improvement += 1

        # Cool down the temperature
        temperature *= cooling_rate

        # Early stopping
        if iterations_without_improvement >= early_stopping_iterations:
            print(f"  Early stopping at iteration {i+1}.")
            break

    print(
        f"Simulated Annealing completed. Best solution: Size: {best_size}, Width: {best_width}"
    )
    return best_decision_vector, best_score


def evaluate_population(population, edges):
    """Evaluates a population of solutions in parallel."""
    with Pool() as pool:
        scores = pool.starmap(
            evaluate_solution, zip(population, [edges] * len(population))
        )
    return scores


def evolutionary_algorithm(
    edges: List[List[int]],
    population_size: int = 100,
    generations: int = 500,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    early_stopping_generations: int = 50,
) -> List[List[int]]:
    """Performs an Evolutionary Algorithm to find a set of Pareto optimal solutions."""
    n = max(node for edge in edges for node in edge) + 1

    def create_individual():
        return [i for i in range(n)] + [random.randint(0, n - 1)]

    def mutate(individual):
        return generate_neighbor_swap(individual, mutation_rate)

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    generations_without_improvement = 0
    best_pareto_front_score = float("inf")
    for generation in range(generations):
        start_time = time.time()

        # Evaluate population in parallel
        scores = evaluate_population(population, edges)

        # Select parents for the next generation using tournament selection
        parents = []
        for i in range(population_size):
            tournament = random.sample(range(population_size), 2)
            if dominates(scores[tournament[0]], scores[tournament[1]]):
                parents.append(population[tournament[0]])
            else:
                parents.append(population[tournament[1]])

        # Create offspring
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            if random.random() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1[:], parent2[:]])

        # Mutate offspring
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i])

        # Combine parents and offspring
        population = parents + offspring

        # Select the best individuals for the next generation using non-dominated sorting
        ranked_population = non_dominated_sort(population, scores)
        population = []
        rank_idx = 0
        while len(population) < population_size:
            population.extend(ranked_population[rank_idx])
            rank_idx += 1

        # Extract Pareto front
        pareto_front = []
        for i in range(len(population)):
            solution1 = population[i]
            score1 = evaluate_solution(solution1, edges)
            dominated = False
            for j in range(len(population)):
                if i != j:
                    solution2 = population[j]
                    score2 = evaluate_solution(solution2, edges)
                    if dominates(score2, score1):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(solution1)

        # Calculate the average score of the Pareto front
        pareto_front_scores = [
            evaluate_solution(sol, edges) for sol in pareto_front
        ]
        avg_pareto_front_score = np.mean(pareto_front_scores, axis=0)

        # Early stopping
        if (
            avg_pareto_front_score[0] < best_pareto_front_score
            or avg_pareto_front_score[1] < best_pareto_front_score
        ):
            best_pareto_front_score = min(
                avg_pareto_front_score[0], avg_pareto_front_score[1]
            )
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= early_stopping_generations:
            print(f"Early stopping at generation {generation+1}.")
            break

        end_time = time.time()
        generation_time = end_time - start_time
        print(
            f"Generation {generation+1}: Best front score: {best_pareto_front_score:.4f}, Time: {generation_time:.2f}s"
        )

    print(f"Final Pareto Front: {pareto_front}")
    return pareto_front


def non_dominated_sort(population, scores):
    """Performs non-dominated sorting on the population."""
    dominating_counts = [0 for _ in range(len(population))]
    dominated_by = [[] for _ in range(len(population))]

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if dominates(scores[i], scores[j]):
                dominating_counts[i] += 1
                dominated_by[j].append(i)
            elif dominates(scores[j], scores[i]):
                dominating_counts[j] += 1
                dominated_by[i].append(j)

    ranked_population = [[] for _ in range(len(population))]
    front = [i for i in range(len(population)) if dominating_counts[i] == 0]
    rank = 0
    while front:
        ranked_population[rank] = [population[i] for i in front]
        next_front = []
        for i in front:
            for j in dominated_by[i]:
                dominating_counts[j] -= 1
                if dominating_counts[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1

    return ranked_population


def hybrid_algorithm(
    edges: List[List[int]],
    population_size: int = 50,
    generations: int = 250,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    sa_iterations: int = 100,
    initial_temperature: float = 1000,
    cooling_rate: float = 0.95,
    early_stopping_generations: int = 25,
) -> List[List[int]]:
    """
    Hybrid algorithm combining elements of Simulated Annealing and Evolutionary Algorithm.

    Args:
        edges (List[List[int]]): List of edges in the graph.
        population_size (int): Size of the population for the EA.
        generations (int): Number of generations for the EA.
        mutation_rate (float): Mutation rate for the EA.
        crossover_rate (float): Crossover rate for the EA.
        sa_iterations (int): Number of iterations for the SA local search.
        initial_temperature (float): Initial temperature for SA.
        cooling_rate (float): Cooling rate for SA.
        early_stopping_generations (int): Number of generations without improvement to trigger early stopping.

    Returns:
        List[List[int]]: Pareto front of non-dominated solutions.
    """
    n = max(node for edge in edges for node in edge) + 1

    def create_individual():
        return [i for i in range(n)] + [random.randint(0, n - 1)]

    def mutate(individual):
        return generate_neighbor_swap(individual, mutation_rate)

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 2)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    generations_without_improvement = 0
    best_pareto_front_score = float("inf")
    for generation in range(generations):
        start_time = time.time()

        # Evaluate population in parallel
        scores = evaluate_population(population, edges)

        # Select parents for the next generation using tournament selection
        parents = []
        for i in range(population_size):
            tournament = random.sample(range(population_size), 2)
            if dominates(scores[tournament[0]], scores[tournament[1]]):
                parents.append(population[tournament[0]])
            else:
                parents.append(population[tournament[1]])

        # Create offspring
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            if random.random() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1[:], parent2[:]])

        # Mutate offspring
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = mutate(offspring[i])

        # Apply Simulated Annealing as local search to improve offspring
        for i in range(len(offspring)):
            print(f"Generation {generation+1}, Offspring {i+1}:")
            offspring[i], _ = simulated_annealing(
                edges,
                initial_temperature=initial_temperature,
                cooling_rate=cooling_rate,
                max_iterations=sa_iterations,
            )

        # Combine parents and offspring
        population = parents + offspring

        # Select the best individuals for the next generation using non-dominated sorting
        ranked_population = non_dominated_sort(population, scores)
        population = []
        rank_idx = 0
        while len(population) < population_size:
            population.extend(ranked_population[rank_idx])
            rank_idx += 1

        # Extract Pareto front
        pareto_front = []
        for i in range(len(population)):
            solution1 = population[i]
            score1 = evaluate_solution(solution1, edges)
            dominated = False
            for j in range(len(population)):
                if i != j:
                    solution2 = population[j]
                    score2 = evaluate_solution(solution2, edges)
                    if dominates(score2, score1):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(solution1)

        # Calculate the average score of the Pareto front
        pareto_front_scores = [
            evaluate_solution(sol, edges) for sol in pareto_front
        ]
        avg_pareto_front_score = np.mean(pareto_front_scores, axis=0)

        # Early stopping
        if (
            avg_pareto_front_score[0] < best_pareto_front_score
            or avg_pareto_front_score[1] < best_pareto_front_score
        ):
            best_pareto_front_score = min(
                avg_pareto_front_score[0], avg_pareto_front_score[1]
            )
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= early_stopping_generations:
            print(f"Early stopping at generation {generation+1}.")
            break

        end_time = time.time()
        generation_time = end_time - start_time
        print(
            f"Generation {generation+1}: Best front score: {best_pareto_front_score:.4f}, Time: {generation_time:.2f}s"
        )

    print(f"Final Pareto Front: {pareto_front}")
    return pareto_front


def create_submission_file(
    decision_vector, problem_id, filename="submission.json"
):
    """Creates a valid submission file."""
    submission = {
        "decisionVector": [
            decision_vector
        ],  # Wrap in a list for multiple solutions
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)
    print(f"Submission file '{filename}' created successfully!")


if __name__ == "__main__":
    random.seed(42)

    # Get problem ID from user input
    while True:
        problem_id = input(
            "Select a problem instance (easy, medium, hard): "
        ).lower()
        if problem_id in problems:
            break
        else:
            print(
                "Invalid problem ID. Please choose from 'easy', 'medium', or 'hard'."
            )

    edges = load_graph(problem_id)

    # Choose an algorithm: 'simulated_annealing', 'evolutionary_algorithm', or 'hybrid_algorithm'
    algorithm = input(
        "Choose an algorithm (simulated_annealing, evolutionary_algorithm, or hybrid_algorithm): "
    )

    if algorithm == "simulated_annealing":
        best_solution, best_score = simulated_annealing(
            edges,
            initial_temperature=1000,  # Adjust as needed
            cooling_rate=0.95,  # Adjust as needed
            max_iterations=1000,  # Adjust as needed
            perturbation_rate=0.2,  # Adjust as needed
            early_stopping_iterations=100,  # Adjust as needed
        )
        create_submission_file(
            best_solution, problem_id, f"simulated_annealing_solution.json"
        )

    elif algorithm == "evolutionary_algorithm":
        pareto_front = evolutionary_algorithm(
            edges,
            population_size=100,  # Adjust as needed
            generations=500,  # Adjust as needed
            mutation_rate=0.2,  # Adjust as needed
            crossover_rate=0.8,  # Adjust as needed
            early_stopping_generations=50,  # Adjust as needed
        )
        for i, solution in enumerate(pareto_front):
            create_submission_file(
                solution,
                problem_id,
                f"evolutionary_algorithm_solution_{i+1}.json",
            )

    elif algorithm == "hybrid_algorithm":
        pareto_front = hybrid_algorithm(
            edges,
            population_size=50,  # Adjust as needed
            generations=250,  # Adjust as needed
            mutation_rate=0.2,  # Adjust as needed
            crossover_rate=0.8,  # Adjust as needed
            sa_iterations=100,  # Adjust as needed
            initial_temperature=1000,  # Adjust as needed
            cooling_rate=0.95,  # Adjust as needed
            early_stopping_generations=25,  # Adjust as needed
        )
        for i, solution in enumerate(pareto_front):
            create_submission_file(
                solution, problem_id, f"hybrid_algorithm_solution_{i+1}.json"
            )

    else:
        print(
            "Invalid algorithm choice. Please choose from 'simulated_annealing', 'evolutionary_algorithm', or 'hybrid_algorithm'."
        )

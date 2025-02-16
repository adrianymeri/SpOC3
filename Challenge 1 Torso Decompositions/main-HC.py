import json
import random
import sys
from typing import List
import urllib.request
from loguru import logger

# Logger configuration
log_format = "<green>{time:YYYY-MM-DD@HH:mm:ss}</green> | <level>{message}</level>"
log_config = {"handlers": [{"sink": sys.stdout, "format": log_format}]}
logger.configure(**log_config)

problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}


def load_graph(problem_id: str) -> List[List[int]]:
    url = problems[problem_id]
    logger.info(f"Loading graph data from: {url}")
    with urllib.request.urlopen(url) as f:
        edges = []
        for line in f:
            if line.startswith(b"#"):
                continue
            u, v = map(int, line.strip().split())
            edges.append([u, v])
    logger.info(f"Loaded graph with {len(edges)} edges.")
    return edges


def calculate_torso_size(decision_vector: List[int]) -> int:
    n = len(decision_vector) - 1
    t = decision_vector[-1]
    return n - t


def calculate_torso_width(decision_vector: List[int], edges: List[List[int]]) -> int:
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
                    if ((permutation[i], permutation[k]) in oriented_edges and
                            (permutation[j], permutation[k]) not in oriented_edges):
                        if permutation[j] >= t and permutation[k] >= t:
                            outdegrees[permutation[j]] += 1
    return max(outdegrees)


def evaluate_solution(decision_vector: List[int], edges: List[List[int]]) -> List[float]:
    torso_size = calculate_torso_size(decision_vector)
    torso_width = calculate_torso_width(decision_vector, edges)
    return [torso_size, torso_width]


def quality(solution: List[int], edges: List[List[int]]) -> float:
    score = evaluate_solution(solution, edges)
    return - (0.7 * score[0] + 0.3 * score[1])


def ideal(solution: List[int], edges: List[List[int]]) -> bool:
    return False


def generate_neighbor_swap(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
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


def generate_neighbor_shuffle(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    sublist_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - sublist_length)
    sublist = neighbor[start_index: start_index + sublist_length]
    random.shuffle(sublist)
    neighbor[start_index: start_index + sublist_length] = sublist
    return neighbor


def generate_neighbor_torso_shift(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    shift_amount = int(perturbation_rate * n)
    neighbor[-1] = max(0, min(n - 1, neighbor[-1] + random.randint(-shift_amount, shift_amount)))
    return neighbor


def generate_neighbor_2opt(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    neighbor[i: j] = neighbor[i: j][::-1]
    return neighbor


def generate_neighbor_insert(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    index1 = random.randint(0, n - 1)
    index2 = random.randint(0, n)
    value = neighbor.pop(index1)
    neighbor.insert(index2, value)
    return neighbor


def generate_neighbor_inversion(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    neighbor = decision_vector[:]
    n = len(neighbor) - 1
    sublist_length = max(1, int(perturbation_rate * n))
    start_index = random.randint(0, n - sublist_length)
    neighbor[start_index: start_index + sublist_length] = neighbor[start_index: start_index + sublist_length][::-1]
    return neighbor


def generate_neighbor_all(decision_vector: List[int], perturbation_rate: float = 0.2) -> List[int]:
    operators = [
        generate_neighbor_swap,
        generate_neighbor_shuffle,
        generate_neighbor_torso_shift,
        generate_neighbor_2opt,
        generate_neighbor_insert,
        generate_neighbor_inversion
    ]
    op = random.choice(operators)
    return op(decision_vector, perturbation_rate)


def generate_random_solution(n: int) -> List[int]:
    return [i for i in range(n)] + [random.randint(0, n - 1)]


def hill_climbing_rr(edges: List[List[int]], max_total_time: int = 1000, perturbation_rate: float = 0.2):
    T_intervals = [10, 20, 30]
    n = max(node for edge in edges for node in edge) + 1
    S = generate_random_solution(n)
    Best = S[:]
    total_time = 0
    restart_count = 0
    while total_time < max_total_time and not ideal(Best, edges):
        restart_count += 1
        time_interval = random.choice(T_intervals)
        inner_time = 0
        logger.info(f"Restart {restart_count}: Starting with quality = {quality(S, edges):.3f}")
        while inner_time < time_interval and total_time < max_total_time and not ideal(S, edges):
            R = generate_neighbor_all(S, perturbation_rate)
            if quality(R, edges) > quality(S, edges):
                S = R[:]
                logger.info(f"Restart {restart_count}, step {inner_time}: Improved quality = {quality(S, edges):.3f}")
            inner_time += 1
            total_time += 1
        if quality(S, edges) > quality(Best, edges):
            Best = S[:]
            logger.info(f"Restart {restart_count}: New best solution with quality = {quality(Best, edges):.3f}")
        logger.info(f"Restart {restart_count}: Total time elapsed = {total_time}")
        S = generate_random_solution(n)
    return Best


if __name__ == "__main__":
    random.seed(42)
    while True:
        problem_id = input("Select a problem instance (easy, medium, hard): ").lower()
        if problem_id in problems:
            break
        else:
            print("Invalid problem ID. Please choose from 'easy', 'medium', or 'hard'.")
    edges = load_graph(problem_id)
    best_solution = hill_climbing_rr(edges, max_total_time=1000, perturbation_rate=0.2)
    logger.info(f"Best solution found: {best_solution}")
    logger.info(f"Solution evaluation (torso size, torso width): {evaluate_solution(best_solution, edges)}")
    logger.info(f"Quality: {quality(best_solution, edges)}")


    def create_submission_file(decision_vector, problem_id, filename="submission.json"):
        submission = {
            "decisionVector": [decision_vector],
            "problem": problem_id,
            "challenge": "spoc-3-torso-decompositions",
        }
        with open(filename, "w") as f:
            json.dump(submission, f, indent=4)
        logger.info(f"Submission file '{filename}' created successfully!")


    create_submission_file(best_solution, problem_id, "final_solution.json")

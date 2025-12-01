#!/usr/bin/env python3
import json
import random
import time
import numpy as np
import urllib.request
import sys
from typing import List, Set, Tuple
from functools import lru_cache

# -------------------------
# Configuration
# -------------------------
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

WORKER_ADJ_BITS = None
WORKER_N = None


# -------------------------
# Step 0: Setup & Helper Functions
# -------------------------

def load_graph(problem_id: str) -> Tuple[int, List[Set[int]]]:
    """Downloads the graph or loads it from a local file."""
    if problem_id not in PROBLEMS:
        filename = f"data/{problem_id}.gr"
        print(f"📂 Reading local file: '{filename}'...")
        edges = []
        max_node = 0
        try:
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.split()
                    # Skip metadata lines
                    if not parts or parts[0] == 'p': continue

                    u, v = int(parts[0]), int(parts[1])
                    edges.append((u, v))
                    max_node = max(max_node, u, v)
        except FileNotFoundError:
            print(f"❌ Error: File {filename} not found.")
            sys.exit(1)

        n = max_node + 1
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        print(f"✅ Graph loaded! Nodes: {n}, Edges: {len(edges)}")
        return n, adj

    url = PROBLEMS[problem_id]
    print(f"📥 Downloading graph data for '{problem_id}'...")
    edges = []
    max_node = 0
    try:
        with urllib.request.urlopen(url) as f:
            for line in f:
                if line.startswith(b'#'): continue
                parts = line.strip().split()
                if not parts: continue
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node = max(max_node, u, v)
    except Exception as e:
        print(f"⚠️ Download failed ({e}). Trying local file fallback...")
        return load_graph(problem_id)  # Recursive fallback

    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    print(f"✅ Graph loaded! Nodes: {n}, Edges: {len(edges)}")
    return n, adj


def build_adj_bitsets(n: int, adj_list: List[Set[int]]) -> List[int]:
    """
    Converts the graph into a list of massive integers (Bitsets).
    Each bit '1' at position 'k' means there is a connection to node 'k'.
    This allows us to check connections using binary AND/OR, which is super fast.
    """
    adj_bits = [0] * n
    for u in range(n):
        bits = 0
        for v in adj_list[u]:
            bits |= (1 << v)
        adj_bits[u] = bits
    return adj_bits


def init_globals(adj_bits: List[int], n: int):
    global WORKER_ADJ_BITS, WORKER_N
    WORKER_ADJ_BITS = adj_bits
    WORKER_N = n


def bitcount(x: int) -> int:
    """Counts how many '1's are in a binary number (Population Count)."""
    try:
        return int(x).bit_count()
    except Exception:
        return bin(int(x)).count('1')


@lru_cache(maxsize=1000000)
def evaluate_solution(solution_tuple: Tuple[int, ...]) -> Tuple[int, int]:
    """
    The 'Scorekeeper'.
    Input: A solution (permutation + split point).
    Output: A score tuple (Torso Size, Max Width).

    Goal: We want BIG Torso Size, and SMALL Max Width.
    """
    global WORKER_ADJ_BITS, WORKER_N
    n = WORKER_N
    adj_bits = WORKER_ADJ_BITS

    # Unpack the solution
    # The last number is the split point 't'
    # Everything before it is the order of nodes 'perm'
    t = int(solution_tuple[-1])
    perm = solution_tuple[:-1]

    # Calculate Torso Size
    torso_size = n - t
    if torso_size <= 0: return (0, 9999)  # Invalid solution

    # 1. Build a map of "Future Nodes"
    # We need to know which nodes appear AFTER a specific point to calculate width.
    suffix_mask = [0] * n
    curr_mask = 0

    # We walk backwards from the end of the list to the start
    for i in range(n - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << int(perm[i]))

    # 2. Simulate the Graph Processing
    # We make a copy of the graph because we will be adding "shortcut" edges
    temp_graph = list(adj_bits)
    max_width = 0

    for i in range(n):
        u = int(perm[i])

        # 'succ' contains all neighbors of 'u' that appear LATER in the list
        succ = temp_graph[u] & suffix_mask[i]

        # Only check width if we are inside the Torso (past the split point t)
        if i >= t:
            current_width = bitcount(succ)
            if current_width > max_width:
                max_width = current_width
                # Optimization: If width is already bad (>500), stop calculating exactly
                if max_width > 500:
                    return (torso_size, 501 + (max_width - 500))

        if succ == 0: continue

        # 3. Add shortcut edges (The Clique Property)
        # If 'u' is connected to 'v' and 'w' (who are both in the future),
        # we must connect 'v' to 'w'.
        s = succ
        while s:
            # Extract the next neighbor 'v' from the bitset
            vbit = s & -s
            s ^= vbit
            v = int(vbit).bit_length() - 1

            # Connect 'v' to everyone else that 'u' was connected to
            temp_graph[v] |= (succ ^ vbit)

    return (torso_size, max_width)


def is_better(new_score, old_score):
    """
    Returns True if new_score is better than old_score.
    Priorities:
    1. Higher Torso Size is always better.
    2. If Torso Size is equal, Lower Width is better.
    """
    new_size, new_width = new_score
    old_size, old_width = old_score

    if new_size > old_size:
        return True
    if new_size == old_size:
        if new_width < old_width:
            return True
    return False


# -------------------------
# Step 3 (Helper): Mutation
# -------------------------
def create_stochastic_neighbor(solution: List[int], n: int) -> List[int]:
    """
    Creates a 'Neighbor' by making a small random change to the solution.
    This is the 'Stochastic' part of Stochastic Hill Climbing.
    """
    neighbor = list(solution)  # Make a copy
    chance = random.random()

    # Mutation A: Shift the split point (20% chance)
    if chance < 0.2:
        t = neighbor[-1]
        # Move the split point left or right by up to 5% of the total nodes
        shift = random.randint(-int(n * 0.05), int(n * 0.05))
        neighbor[-1] = max(0, min(n - 1, t + shift))

    # Mutation B: Move a Block of nodes (30% chance)
    elif chance < 0.5:
        perm = neighbor[:-1]
        # Pick a random block size
        block_size = random.randint(2, max(3, int(n * 0.05)))
        start = random.randint(0, n - block_size)

        # Cut the block out
        block = perm[start:start + block_size]
        del perm[start:start + block_size]

        # Paste it back in somewhere else
        insert_pos = random.randint(0, len(perm))
        perm[insert_pos:insert_pos] = block
        neighbor[:-1] = perm

    # Mutation C: Swap two random nodes (25% chance)
    elif chance < 0.75:
        idx1, idx2 = random.sample(range(n), 2)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

    # Mutation D: Invert a section (25% chance)
    else:
        idx1, idx2 = sorted(random.sample(range(n), 2))
        neighbor[idx1:idx2 + 1] = reversed(neighbor[idx1:idx2 + 1])

    return neighbor


# -------------------------
# MAIN ALGORITHM: Hill Climbing
# -------------------------
def run_algorithm_10(n: int, adj_list: List[Set[int]], problem_id: str):
    # Prepare the fast bitsets
    adj_bits = build_adj_bitsets(n, adj_list)
    init_globals(adj_bits, n)

    print("\n=== ⛰️  Starting Algorithm 10: Hill Climbing ===")

    # --- Step 1: Initialization ---
    # Create a random permutation and a random split point (t)
    # Note: We convert to Python 'int' to avoid numpy issues
    perm = [int(x) for x in np.random.permutation(n)]
    t = int(random.randint(int(n * 0.2), int(n * 0.8)))

    current_solution = perm + [t]

    # --- Step 2: Initial Evaluation ---
    current_score = evaluate_solution(tuple(current_solution))

    best_solution = current_solution
    best_score = current_score

    step = 0
    stagnation_counter = 0

    print(f"Step {step}: Initial Random Score -> Size: {current_score[0]}, Width: {current_score[1]}")

    # --- Step 6: The Infinite Loop ---
    while True:
        step += 1

        # --- Step 3: Neighbor Generation (Mutation) ---
        neighbor = create_stochastic_neighbor(current_solution, n)

        # --- Step 4: Evaluation ---
        neighbor_score = evaluate_solution(tuple(neighbor))

        # --- Step 5: Selection (The "Climb") ---
        # If the neighbor is better, we move there.
        if is_better(neighbor_score, current_score):
            current_solution = neighbor
            current_score = neighbor_score
            stagnation_counter = 0  # Reset stagnation because we found an improvement

            print(f"Step {step}: ✅ Improvement! Size: {current_score[0]}, Width: {current_score[1]}")

            # Save if it's the global best we've seen so far
            if is_better(current_score, best_score):
                best_score = current_score
                best_solution = current_solution
                save_submission(best_solution, problem_id)

        else:
            # If neighbor was NOT better, we stay put.
            stagnation_counter += 1
            if step % 5000 == 0:
                print(f"Step {step}: ... searching ... (Best: {best_score})")

        # --- Step 7: Restart (Escaping Local Optima) ---
        # If we haven't improved in 20,000 tries, we are likely stuck.
        # We "jump" to a completely new random spot.
        if stagnation_counter > 20000:
            print(f"⚠️  Stuck in Local Optima (Step {step}). Restarting search...")

            # Re-Initialize Randomly
            perm = [int(x) for x in np.random.permutation(n)]
            t = int(random.randint(int(n * 0.2), int(n * 0.8)))
            current_solution = perm + [t]
            current_score = evaluate_solution(tuple(current_solution))

            stagnation_counter = 0


def save_submission(solution, problem_id):
    """Saves the best result to a JSON file for submission."""
    filename = f"submission_{problem_id}.json"
    problem_name_map = {"easy": "small-graph", "medium": "medium-graph", "hard": "large-graph"}

    submission = {
        "decisionVector": [[int(x) for x in solution]],
        "problem": problem_name_map.get(problem_id, problem_id),
        "challenge": "spoc-3-torso-decompositions",
    }

    with open(filename, "w") as f:
        json.dump(submission, f, indent=4)


if __name__ == "__main__":
    problem_id = "medium"
    if len(sys.argv) > 1:
        problem_id = sys.argv[1]

    try:
        n, adj = load_graph(problem_id)
        run_algorithm_10(n, adj, problem_id)
    except KeyboardInterrupt:
        print("\n🛑 Execution stopped by user.")

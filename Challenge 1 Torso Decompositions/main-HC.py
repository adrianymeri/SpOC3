import json
import random
import math
import time
import urllib.request
from typing import List, Tuple, Dict

# Configuration
problems = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

class TorsoOptimizer:
    def __init__(self, edges: List[List[int]]):
        self.edges = edges
        self.n = max(max(u, v) for u, v in edges) + 1
        self.adj_list = [[] for _ in range(self.n)]
        self.degrees = [0] * self.n
        for u, v in edges:
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
            self.degrees[u] += 1
            self.degrees[v] += 1
        
        self.communities = self._detect_communities()
        self.best_ever = (0, float('inf'), [])

    def _detect_communities(self) -> List[List[int]]:
        """Fast label propagation with degree-based initialization"""
        labels = list(range(self.n))
        changed = True
        for _ in range(10):
            changed = False
            order = sorted(range(self.n), key=lambda x: -self.degrees[x])
            for node in order:
                counts = {}
                for neighbor in self.adj_list[node]:
                    lbl = labels[neighbor]
                    counts[lbl] = counts.get(lbl, 0) + 1
                if counts:
                    max_lbl = max(counts, key=lambda k: (counts[k], -k))
                    if labels[node] != max_lbl:
                        labels[node] = max_lbl
                        changed = True
        communities = {}
        for node, lbl in enumerate(labels):
            communities.setdefault(lbl, []).append(node)
        return list(communities.values())

    def evaluate(self, solution: List[int]) -> Tuple[int, int]:
        t = solution[-1]
        torso = solution[t:-1]
        if not torso:
            return (0, float('inf'))
        max_degree = max(self.degrees[node] for node in torso)
        return (len(torso), max_degree)

    def initialize_solution(self) -> List[int]:
        """Degree-community balanced initialization"""
        solution = []
        remaining = set(range(self.n))
        
        # Add low-degree nodes from each community
        for comm in self.communities:
            comm_nodes = sorted(comm, key=lambda x: self.degrees[x])
            solution.extend(comm_nodes[:len(comm)//3])
            remaining -= set(comm_nodes[:len(comm)//3])
        
        # Add remaining nodes sorted by descending degree
        solution += sorted(remaining, key=lambda x: -self.degrees[x])
        t = len(solution) - int(self.n * 0.25)
        return solution + [t]

    def generate_neighbor(self, current: List[int]) -> List[int]:
        """Hybrid neighborhood operator with 5 different perturbation strategies"""
        t = current[-1]
        new_sol = current.copy()
        
        # Strategy 1: Critical node replacement
        if random.random() < 0.6:
            torso_nodes = new_sol[t:-1]
            if torso_nodes:
                max_deg = max(self.degrees[node] for node in torso_nodes)
                candidates = [node for node in torso_nodes if self.degrees[node] == max_deg]
                to_remove = random.choice(candidates)
                non_torso = [node for node in new_sol[:t] if self.degrees[node] < max_deg]
                if non_torso:
                    to_add = min(non_torso, key=lambda x: self.degrees[x])
                    idx_rm = new_sol.index(to_remove)
                    idx_add = new_sol.index(to_add)
                    new_sol[idx_rm], new_sol[idx_add] = to_add, to_remove

        # Strategy 2: Community-based expansion
        if random.random() < 0.4:
            comm = random.choice(self.communities)
            candidates = [node for node in comm if node not in new_sol[t:-1]]
            if candidates:
                best = min(candidates, key=lambda x: self.degrees[x])
                new_t = max(0, t - 1)
                new_sol.insert(new_t, best)
                new_sol.pop(new_sol.index(best, 0, t))
                new_sol[-1] = new_t

        # Strategy 3: Degree-weighted threshold shift
        delta = int(math.copysign(1, (self.n//2 - t))) if random.random() < 0.5 else -1
        new_sol[-1] = max(0, min(len(new_sol)-1, t + delta))

        # Strategy 4: Block swap within community
        if random.random() < 0.3:
            comm = random.choice(self.communities)
            comm_nodes = [node for node in comm if node in new_sol]
            if len(comm_nodes) >= 2:
                i, j = random.sample(range(len(comm_nodes)), 2)
                idx_i = new_sol.index(comm_nodes[i])
                idx_j = new_sol.index(comm_nodes[j])
                new_sol[idx_i], new_sol[idx_j] = new_sol[idx_j], new_sol[idx_i]

        # Strategy 5: Greedy width reduction
        if random.random() < 0.2:
            torso_nodes = new_sol[t:-1]
            if torso_nodes:
                worst = max(torso_nodes, key=lambda x: self.degrees[x])
                new_sol.remove(worst)
                new_sol.insert(random.randint(0, t), worst)

        return new_sol

    def optimize(self, iterations: int = 100000) -> List[int]:
        """Enhanced Simulated Annealing with adaptive cooling and intensification"""
        current = self.initialize_solution()
        current_score = self.evaluate(current)
        best = current.copy()
        self.best_ever = current_score + (best,)
        
        T = 5000.0
        cooling = 0.9995
        accept_rate = 1.0
        improvements = []
        
        for i in range(iterations):
            # Adaptive temperature adjustment
            if len(improvements) > 100:
                avg_improve = sum(improvements[-100:])/100
                if avg_improve < 0.1:
                    T *= 1.05  # Escape local minima
                improvements.pop(0)
            
            neighbor = self.generate_neighbor(current)
            neighbor_score = self.evaluate(neighbor)
            
            # Acceptance criteria with width priority
            delta_width = current_score[1] - neighbor_score[1]
            delta_size = neighbor_score[0] - current_score[0]
            score_diff = delta_width * 5 + delta_size  # Strong width bias
            
            if score_diff > 0 or math.exp(score_diff / T) > random.random():
                accept_rate = 0.9 * accept_rate + 0.1
                current = neighbor
                current_score = neighbor_score
                
                if (neighbor_score[0] > self.best_ever[0] or 
                   (neighbor_score[0] == self.best_ever[0] and neighbor_score[1] < self.best_ever[1])):
                    best = neighbor.copy()
                    self.best_ever = neighbor_score + (best,)
                    improvements.append(1)
                else:
                    improvements.append(0)
            else:
                accept_rate = 0.9 * accept_rate - 0.1
            
            # Dynamic cooling adjustment
            T *= cooling
            if accept_rate < 0.2:
                T *= 1.1
                accept_rate = 0.5
            elif accept_rate > 0.8:
                T *= 0.9
            
            # Intensification phase
            if i % 1000 == 999:
                current = best.copy()
                current_score = self.best_ever[:2]
                T = max(T, 500)
                
            # Progress reporting
            if i % 500 == 0:
                print(f"Iter {i}: Temp={T:.1f} Best={self.best_ever[:2]} Current={current_score}")

        return best

def create_submission(solution: List[int], problem_id: str, filename: str):
    submission = {
        "decisionVector": [solution],
        "problem": problem_id,
        "challenge": "spoc-3-torso-decompositions",
    }
    with open(filename, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"Created submission file: {filename}")

if __name__ == "__main__":
    problem_id = input("Enter problem (easy/medium/hard): ").lower()
    while problem_id not in problems:
        problem_id = input("Invalid! Choose easy/medium/hard: ").lower()
    
    # Load data
    with urllib.request.urlopen(problems[problem_id]) as f:
        edges = [list(map(int, line.strip().split())) 
                for line in f if not line.startswith(b'#')]
    
    start = time.time()
    optimizer = TorsoOptimizer(edges)
    best_solution = optimizer.optimize(iterations=50000)
    
    final_score = optimizer.evaluate(best_solution)
    print(f"\nOptimization completed in {time.time()-start:.1f}s")
    print(f"Final score: Size={final_score[0]} Width={final_score[1]}")
    
    create_submission(best_solution, problem_id, "optimized_solution.json")

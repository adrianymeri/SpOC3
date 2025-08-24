#!/usr/bin/env python3
import os, sys, time, json, urllib.request, subprocess, random
from typing import List, Set, Tuple, Dict
import numpy as np
from tqdm import tqdm
import multiprocessing
import pygmo as pg

# --- Auto-compile Cython module ---
def compile_cython_module():
    module_name = "solver_cython"
    pyx_file = "solver_cython.pyx"
    try: import importlib.util; ext_suffix = importlib.util.EXTENSIONS[0]
    except (ImportError, AttributeError): import sysconfig; ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    
    cython_so_file = f"{module_name}{ext_suffix}"
    if not os.path.exists(cython_so_file) or os.path.getmtime(pyx_file) > os.path.getmtime(cython_so_file):
        print(f"🚀 Building/updating Cython module...")
        try:
            subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
            print("✅ Cython module built successfully.")
        except Exception as e:
            print(f"❌ Failed to build Cython module: {e}"); sys.exit(1)

compile_cython_module()
import solver_cython

# CONFIGURATION
# ==============================================================================
CONFIG = {
    "general": {
        "num_islands": os.cpu_count() or 8,
        "migration_interval": 25, 
        "migration_size": 5,
        "stagnation_limit": 50, 
        "mutation_boost_factor": 2.0,
        "restart_stagnation_trigger": 150, 
        "restart_fraction": 0.3,
        "elite_count": 8, 
        "elite_ls_multiplier": 4,
        "crossover_rate": 0.9,
        "mutation_rate": 0.6 # <-- THE MISSING KEY
    },
    "easy": {"pop_size_per_island": 100, "generations": 2500, "local_search_intensity": 25},
    "medium": {"pop_size_per_island": 120, "generations": 4000, "local_search_intensity": 30},
    "hard": {"pop_size_per_island": 150, "generations": 6000, "local_search_intensity": 35},
}
PROBLEMS = {
    "easy": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "hard": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Worker Functions (for the multiprocessing Pool) ---
def init_worker(n_val, adj_bits_val):
    solver_cython.init_worker_cython(n_val, adj_bits_val)
    random.seed(); np.random.seed()

def eval_wrapper(solution_np: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    size, width = solver_cython.evaluate_solution_cy(solution_np)
    return solution_np, (int(size), int(width))

def local_search_worker(args: Tuple[np.ndarray, int]) -> np.ndarray:
    solution, intensity = args
    n = len(solution) - 1
    best_sol = solution
    _, best_score = eval_wrapper(best_sol)
    
    for _ in range(intensity):
        cand_sol = best_sol.copy()
        r = random.random()
        perm = cand_sol[:-1]

        if r < 0.4: # Inversion
            a, b = sorted(random.sample(range(n), 2))
            perm[a:b+1] = perm[a:b+1][::-1]
        elif r < 0.8: # Block Move
            block_size = random.randint(2, max(3, n // 50))
            if n > block_size:
                start = random.randint(0, n - block_size)
                block = perm[start:start+block_size]
                perm_deleted = np.delete(perm, np.s_[start:start+block_size])
                insert_pos = random.randint(0, len(perm_deleted))
                cand_sol[:-1] = np.insert(perm_deleted, insert_pos, block)
        else: # Torso Shift
            t = cand_sol[-1]
            shift = max(1, n // 20)
            new_t = t + random.randint(-shift, shift)
            cand_sol[-1] = max(0, min(n - 1, int(new_t)))

        _, cand_score = eval_wrapper(cand_sol)
        if (cand_score[0] > best_score[0]) or (cand_score[0] == best_score[0] and cand_score[1] < best_score[1]):
            best_sol = cand_sol
            best_score = cand_score
            
    return best_sol

# --- Core Algorithm Components ---
def load_graph(problem_id: str) -> Tuple[int, np.ndarray]:
    url = PROBLEMS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    edges, max_node = [], 0
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            edges.append((u, v)); max_node = max(max_node, u, v)
    n = max_node + 1
    adj = [set() for _ in range(n)]
    for u,v in edges: adj[u].add(v); adj[v].add(u)
    adj_bits = np.zeros(n, dtype=np.uint64)
    for u in range(n):
        for v in adj[u]: adj_bits[u] |= (np.uint64(1) << np.uint64(v))
    print(f"✅ Loaded graph with {n} nodes.")
    return n, adj_bits

def pmx_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int64)
    child[a:b+1] = p1[a:b+1]
    for i in range(a, b + 1):
        val = p2[i]
        if val not in child:
            pos = i
            while True:
                mapped = p1[pos]
                try: pos = np.where(p2 == mapped)[0][0]
                except IndexError: break
                if child[pos] == -1: child[pos] = val; break
    for i in range(n):
        if child[i] == -1: child[i] = p2[i]
    return child

def dominates(p_score, q_score) -> bool:
    return (p_score[0] >= q_score[0] and p_score[1] < q_score[1]) or \
           (p_score[0] > q_score[0] and p_score[1] <= q_score[1])

def crowding_selection(population: List[Dict], pop_size: int) -> List[Dict]:
    if len(population) <= pop_size: return population
    for p in population: p['dominates_set'], p['dominated_by_count'] = [], 0
    fronts = [[]]
    for p in population:
        for q in population:
            if p is q: continue
            if dominates(p['score'], q['score']): p['dominates_set'].append(q)
            elif dominates(q['score'], p['score']): p['dominated_by_count'] += 1
        if p['dominated_by_count'] == 0: fronts[0].append(p)
    new_population = []
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in p['dominates_set']:
                q['dominated_by_count'] -= 1
                if q['dominated_by_count'] == 0: next_front.append(q)
        if len(new_population) + len(fronts[i]) > pop_size:
            for p in fronts[i]: p['distance'] = 0.0
            for i_obj in range(2):
                fronts[i].sort(key=lambda p: p['score'][i_obj])
                if len(fronts[i]) > 1:
                    f_min, f_max = fronts[i][0]['score'][i_obj], fronts[i][-1]['score'][i_obj]
                    fronts[i][0]['distance'] = fronts[i][-1]['distance'] = float('inf')
                    if f_max > f_min:
                        for j in range(1, len(fronts[i]) - 1):
                            fronts[i][j]['distance'] += (fronts[i][j+1]['score'][i_obj] - fronts[i][j-1]['score'][i_obj]) / (f_max - f_min)
            fronts[i].sort(key=lambda p: p['distance'], reverse=True)
            new_population.extend(fronts[i][:pop_size - len(new_population)])
            break
        new_population.extend(fronts[i])
        fronts.append(next_front)
        i += 1
    return new_population

# --- Main Memetic Algorithm ---
def memetic_algorithm(n: int, adj_bits_np: np.ndarray, config: Dict, problem_id: str):
    num_islands, pop_size = config['num_islands'], config['pop_size_per_island']
    print(f"🧬 Initializing {num_islands} island populations of size {pop_size}...")
    islands, base_perm = [], np.arange(n, dtype=np.int64)
    for _ in range(num_islands):
        island_pop = [{'solution': np.append(np.random.permutation(base_perm), np.random.randint(0, n // 2))} for _ in range(pop_size)]
        islands.append(island_pop)

    best_score, stagnation = (-1, 999), 0
    base_mutation = config['mutation_rate']

    with multiprocessing.Pool(initializer=init_worker, initargs=(n, adj_bits_np)) as pool:
        for i in range(num_islands):
            results = pool.map(eval_wrapper, [p['solution'] for p in islands[i]])
            for sol, score in results:
                for p in islands[i]:
                    if np.array_equal(p['solution'], sol): p['score'] = score; break
        
        pbar = tqdm(range(config['generations']), desc="🚀 Evolving")
        for gen in pbar:
            mutation_rate = base_mutation * (config['mutation_boost_factor'] if stagnation >= config['stagnation_limit'] else 1.0)
            
            for i in range(num_islands):
                pop = islands[i]
                mating_pool = crowding_selection(pop, len(pop))
                offspring = []
                while len(offspring) < len(pop):
                    p1, p2 = random.sample(mating_pool, 2)
                    perm = pmx_crossover(p1['solution'][:-1], p2['solution'][:-1]) if random.random() < config['crossover_rate'] else p1['solution'][:-1].copy()
                    if random.random() < mutation_rate:
                        if random.random() < 0.5:
                            a, b = sorted(random.sample(range(n), 2)); perm[a:b+1] = perm[a:b+1][::-1]
                        else:
                            a, b = np.random.choice(n, 2, replace=False); perm[a], perm[b] = perm[b], perm[a]
                    t = int((p1['solution'][-1] + p2['solution'][-1]) / 2)
                    offspring.append(np.append(perm, t))

                improved = pool.map(local_search_worker, [(sol, config['local_search_intensity']) for sol in offspring])
                results = pool.map(eval_wrapper, improved)
                offspring_pop = [{'solution': sol, 'score': score} for sol, score in results]
                islands[i] = crowding_selection(pop + offspring_pop, len(pop))
                
                sorted_pop = sorted(islands[i], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                elite_args = [(p['solution'], config['local_search_intensity'] * config['elite_ls_multiplier']) for p in sorted_pop[:config['elite_count']]]
                if elite_args:
                    intensified = pool.map(local_search_worker, elite_args)
                    results = pool.map(eval_wrapper, intensified)
                    islands[i].extend([{'solution': sol, 'score': score} for sol, score in results])
                    islands[i] = crowding_selection(islands[i], len(pop))

            if stagnation > config['restart_stagnation_trigger']:
                pbar.write(f"⚠️ Stagnation limit reached. Shaking up population...")
                for i in range(num_islands):
                    num_replace = int(len(islands[i]) * config['restart_fraction'])
                    islands[i].sort(key=lambda p: (p['score'][0], -p['score'][1]))
                    for j in range(num_replace):
                        islands[i][j]['solution'] = np.append(np.random.permutation(base_perm), np.random.randint(0, n // 2))
                for i in range(num_islands):
                    num_replace = int(len(islands[i]) * config['restart_fraction'])
                    results = pool.map(eval_wrapper, [islands[i][j]['solution'] for j in range(num_replace)])
                    for k, (_, score) in enumerate(results): islands[i][k]['score'] = score
                stagnation = 0

            if gen > 0 and gen % config['migration_interval'] == 0:
                all_individuals = sorted([ind for island in islands for ind in island], key=lambda p: (p['score'][0], -p['score'][1]), reverse=True)
                migrants = [p['solution'] for p in all_individuals[:config['migration_size']]]
                if migrants:
                    for island in islands:
                        island.sort(key=lambda p: (p['score'][0], -p['score'][1]));
                        for j in range(len(migrants)):
                            if j < len(island): island[j]['solution'] = random.choice(migrants)

            found_better = False
            for island in islands:
                # Ensure island is not empty and solutions have scores
                valid_scores = [p['score'] for p in island if 'score' in p and p['score'] is not None]
                if not valid_scores: continue
                
                current_best_score = max(valid_scores, key=lambda s: (s[0], -s[1]))
                if (current_best_score[0] > best_score[0]) or (current_best_score[0] == best_score[0] and current_best_score[1] < best_score[1]):
                    best_score, found_better = current_best_score, True
            
            if found_better:
                stagnation = 0
                pbar.write(f"✨ Gen {gen+1}: New best score {best_score}")
            else:
                stagnation += 1
            pbar.set_postfix({"best_score": best_score, "stagn": stagnation})
    
    # Final result collection
    final_population = [p for island in islands for p in island]
    final_front = crowding_selection(final_population, 20)
    
    final_solutions = [p['solution'] for p in final_front]
    final_scores = [p['score'] for p in final_front]
    
    print(f"\n🏆 Final best individual solution (size, width): {best_score}")
    ref_point = [n, 501]; final_fitnesses_pygmo = [(s[1], -s[0]) for s in final_scores]
    hv = pg.hypervolume(final_fitnesses_pygmo)
    final_hv = -hv.compute(ref_point)
    print(f"📈 Final Hypervolume: {final_hv:,.2f}")

    with open(f"submission_{problem_id}.json", "w") as f:
        json.dump({"decisionVector": [s.tolist() for s in final_solutions], "problem": problem_id, "challenge": "spoc-3-torso-decompositions"}, f, indent=4)
    print(f"📄 Created final submission file.")

# --- Entry Point ---
if __name__ == "__main__":
    multiprocessing.freeze_support()
    problem_id = input("🔍 Select problem (easy/medium/hard): ").strip().lower() or "easy"
    if problem_id not in PROBLEMS: sys.exit("❌ Invalid problem ID.")
    
    config = CONFIG['general'].copy(); config.update(CONFIG[problem_id])
    
    start_time = time.time()
    n, adj_bits = load_graph(problem_id)
    memetic_algorithm(n, adj_bits, config, problem_id)
    print(f"\n⏱️  Total Optimization Time: {time.time() - start_time:.2f} seconds")

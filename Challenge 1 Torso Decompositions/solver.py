import argparse, ctypes, json, math, os, time, sys
import numpy as np
import torch
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eigh
import urllib.request
from tqdm import tqdm
import pygmo as pg

# --- Auto-compile CUDA module ---
def compile_cython_module(): # Note: The function name is a holdover, it compiles CUDA
    module_name, pyx_file = "evaluator", "evaluator.cu"
    so_file = "evaluator.so"
    
    if not os.path.exists(so_file) or os.path.getmtime(pyx_file) > os.path.getmtime(so_file):
        print(f"🚀 Building/updating CUDA module '{so_file}'...")
        try:
            # Use setup_cuda.py script to compile
            subprocess.check_call([sys.executable, "setup_cuda.py"])
        except Exception as e:
            print(f"❌ Failed to build CUDA module: {e}"); sys.exit(1)

if os.path.exists("setup_cuda.py") and os.path.exists("evaluator.cu"):
    compile_cython_module()
else:
    if not os.path.exists("evaluator.so"):
        print("❌ evaluator.so not found and setup_cuda.py or evaluator.cu is missing. Cannot proceed.")
        sys.exit(1)

libeval = ctypes.CDLL('./evaluator.so')
evaluate = libeval.evaluate_on_gpu
evaluate.argtypes = [
    ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_uint16),
    ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t, ctypes.c_size_t,
]

# --- CORRECTED Graph & Problem Info ---
GRAPH_SIZES = {
    "small-graph": 1357,
    "medium-graph": 4941,
    "large-graph": 15102,
}
PROBLEM_URLS = {
    "small-graph": "https://api.optimize.esa.int/data/spoc3/torso/easy.gr",
    "medium-graph": "https://api.optimize.esa.int/data/spoc3/torso/medium.gr",
    "large-graph": "https://api.optimize.esa.int/data/spoc3/torso/hard.gr",
}

# --- Helper Functions ---
def load_graph(problem_id, n_nodes):
    url = PROBLEM_URLS[problem_id]
    print(f"📥 Loading graph data for '{problem_id}'...")
    adj = np.zeros((n_nodes, n_nodes), dtype=bool)
    with urllib.request.urlopen(url) as f:
        for line in f:
            if line.startswith(b'#'): continue
            u, v = map(int, line.strip().split())
            adj[u, v] = adj[v, u] = True
    print(f"✅ Loaded graph with {n_nodes} nodes.")
    return adj

def get_node_features(adj, num_eigenvectors):
    print("🔬 Calculating node features (Laplacian eigenvectors)...")
    L = laplacian(adj.astype(np.float32), normed=True)
    eig_vals, eig_vecs = eigh(L)
    features = np.real(eig_vecs[:, 1:num_eigenvectors+1]).astype(np.float32)
    return features.T

def calculate_hvi(ts, widths, n_nodes):
    points = np.vstack([widths, -(n_nodes - ts)]).T
    ref_point = [n_nodes, 0]
    try:
        hv = pg.hypervolume(points)
        return -hv.compute(ref_point)
    except Exception as e:
        print(f"Could not compute hypervolume: {e}")
        return -1.0

def create_submission(elites, node_features, selected_ts, problem_id):
    logits = elites[selected_ts] @ node_features
    perms = logits.argsort(axis=1)
    return {
        "decisionVector": [perm.tolist() + [t.item()] for perm, t in zip(perms, selected_ts)],
        "problem": problem_id, "challenge": "spoc-3-torso-decompositions",
    }

# --- Main Neuroevolutionary Algorithm ---
def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated Neuroevolutionary Solver for SPOC3")
    parser.add_argument("--graph", choices=GRAPH_SIZES.keys(), default="small-graph")
    parser.add_argument("--eigenvectors", type=int, default=16)
    parser.add_argument("--mutation_stdev", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_generations", type=int, default=10000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 Using device: {device}")
    
    B, N = args.batch_size, GRAPH_SIZES[args.graph]
    E = args.eigenvectors
    
    adj = load_graph(args.graph, N)
    node_features = torch.from_numpy(get_node_features(adj, E)).to(device)

    population = torch.randn((B, E), dtype=torch.float32, device=device) * 0.1
    elites = torch.zeros((N, E), dtype=torch.float32, device=device)
    elite_fitnesses = torch.full((N,), 501, dtype=torch.int32, device=device)
    
    perms = torch.empty((B, N), dtype=torch.uint16, device=device)
    degrees = torch.empty((B, N), dtype=torch.uint16, device=device)
    fitnesses = torch.empty((B, N), dtype=torch.int32, device=device)
    
    adjs_batch_flat = np.tile(adj, (B, 1, 1)).flatten().astype(np.bool_)

    pbar = tqdm(range(args.max_generations), desc="🚀 Evolving on GPU")
    for generation in pbar:
        logits = population @ node_features
        perms_cpu = logits.argsort(axis=1).to(torch.uint16)
        
        degrees_cpu = torch.empty((B, N), dtype=torch.uint16)
        
        # We need a fresh adjacency matrix for each batch evaluation
        adj_for_eval = torch.from_numpy(np.tile(adj, (B, 1, 1))).to(device)

        evaluate(
            ctypes.cast(adj_for_eval.data_ptr(), ctypes.POINTER(ctypes.c_bool)),
            ctypes.cast(perms_cpu.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            ctypes.cast(degrees_cpu.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
            B, N
        )
        
        # Calculate final fitness on CPU from GPU results
        ts_tensor = torch.arange(N, device='cpu').unsqueeze(0)
        widths = torch.max(torch.where(torch.arange(N, device='cpu') >= ts_tensor.T, degrees_cpu, 0), axis=1).values
        fitnesses_cpu = widths.T # Shape (N, B)

        best_widths_in_batch, best_indices_in_batch = fitnesses_cpu.min(axis=1)
        
        is_better = best_widths_in_batch.cpu() < elite_fitnesses.cpu()
        elite_fitnesses[is_better] = best_widths_in_batch[is_better].to(device)
        elite_indices_to_update = best_indices_in_batch[is_better]
        elites[is_better] = population[elite_indices_to_update]

        parent_indices = torch.randint(0, N, (B,), device=device)
        population[:] = elites[parent_indices]
        population += torch.randn_like(population) * args.mutation_stdev

        if generation % 10 == 0:
            hvi = calculate_hvi(np.arange(N), elite_fitnesses.cpu().numpy(), N)
            pbar.set_postfix({"best_width_t0": elite_fitnesses[0].item(), "hvi": f"{hvi:,.0f}"})

        if generation > 0 and generation % args.checkpoint_every == 0:
            ts_to_submit = pg.non_dominated_front_2d(np.vstack([np.arange(N), elite_fitnesses.cpu().numpy()]).T)
            submission = create_submission(elites.cpu(), node_features.cpu(), torch.from_numpy(ts_to_submit), args.graph)
            
            # Create directories if they don't exist
            os.makedirs(f"submissions/{args.graph}", exist_ok=True)
            with open(f"submissions/{args.graph}/gen_{generation}.json", "w") as f:
                json.dump(submission, f)
            print(f"\n📄 Saved checkpoint submission for generation {generation}")

if __name__ == "__main__":
    main()

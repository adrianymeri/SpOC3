#include <cuda_runtime.h>
#include <cstdint>

// This kernel calculates the out-degrees and final fitness values for a batch of permutations.
__global__ void _evaluate(
    bool *adjs,          // A batch of adjacency matrices [B x N x N]
    uint16_t *perms,     // A batch of permutations [B x N]
    uint16_t *degrees,   // Output: degree of each node at elimination [B x N]
    int *fitnesses,      // Output: fitness (max width) for each torso `t` [B x N]
    size_t B,            // Batch size
    size_t N)            // Number of nodes
{
    // Assign one GPU thread per solution in the batch
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    // Pointers for this thread's specific data
    bool* adj = &adjs[idx * N * N];
    uint16_t* perm = &perms[idx * N];
    uint16_t* degree_out = &degrees[idx * N];
    int* fitness_out = &fitnesses[idx * N];
    
    // --- Main Evaluation Logic ---
    for (int step = 0; step < N; ++step) {
        uint16_t u = perm[step];
        uint16_t current_degree = 0;

        // Find adjacent nodes that appear later in the permutation
        for (int v_idx = step + 1; v_idx < N; ++v_idx) {
            uint16_t v = perm[v_idx];
            if (adj[u * N + v]) {
                current_degree++;
            }
        }
        degree_out[u] = current_degree; // Store the out-degree for node u

        // Simulate the fill-in process: connect neighbors of u
        for (int v1_idx = step + 1; v1_idx < N; ++v1_idx) {
            for (int v2_idx = v1_idx + 1; v2_idx < N; ++v2_idx) {
                uint16_t v1 = perm[v1_idx];
                uint16_t v2 = perm[v2_idx];
                // If v1 and v2 are both neighbors of u, they become connected
                if (adj[u * N + v1] && adj[u * N + v2]) {
                    adj[v1 * N + v2] = true;
                    adj[v2 * N + v1] = true;
                }
            }
        }
    }
    
    // --- Calculate final fitness scores from the raw degrees ---
    // For each possible torso start `t`, find the max out-degree in the torso
    for (int t = 0; t < N; ++t) {
        int max_degree_in_torso = 0;
        // The torso consists of nodes from perm[t] to perm[N-1]
        for (int i = t; i < N; ++i) {
            uint16_t node_in_torso = perm[i];
            if (degree_out[node_in_torso] > max_degree_in_torso) {
                max_degree_in_torso = degree_out[node_in_torso];
            }
        }
        // If width > 500, penalize it as 501, as per problem spec
        fitness_out[t] = (max_degree_in_torso > 500) ? 501 : max_degree_in_torso;
    }
}


// C-style wrapper function that Python can call via ctypes
extern "C" {
    void evaluate(
        bool *adjs,
        uint16_t *perms,
        uint16_t *degrees,
        int *fitnesses,
        size_t B,
        size_t N)
    {
        int threads_per_block = 256;
        int blocks_per_grid = (B + threads_per_block - 1) / threads_per_block;
        _evaluate<<<blocks_per_grid, threads_per_block>>>(adjs, perms, degrees, fitnesses, B, N);
        cudaDeviceSynchronize(); // Wait for GPU to finish
    }
}

#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel to calculate the out-degree for each node in each permutation
__global__ void _evaluate_kernel(
    const bool *adj_flat,       // Flattened adjacency matrix [N*N]
    const uint16_t *perms_flat, // Flattened permutations [B*N]
    uint16_t *degrees_out_flat, // Flattened output degrees [B*N]
    size_t B,                   // Batch size (number of solutions)
    size_t N)                   // Number of nodes
{
    // Assign one GPU thread to each solution in the batch
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    // Create pointers for this thread's data
    const bool* adj = &adj_flat[0]; // All threads share the single adjacency matrix
    const uint16_t* perm = &perms_flat[idx * N];
    uint16_t* degree_out = &degrees_out_flat[idx * N];

    // --- Allocate memory on the GPU's fast scratchpad memory ---
    extern __shared__ uint64_t shared_mem[];
    uint64_t* adj_bits = shared_mem;
    uint64_t* temp = &shared_mem[N];
    uint64_t* suffix_mask = &shared_mem[2 * N];

    // --- Convert adjacency matrix to bitsets for this thread ---
    // This happens in parallel for all threads in a block
    for(int i = threadIdx.y; i < N; i += blockDim.y) {
        uint64_t bits = 0;
        for(int j = 0; j < N; ++j) {
            if(adj[i * N + j]) {
                bits |= (uint64_t)1 << j;
            }
        }
        adj_bits[i] = bits;
    }
    __syncthreads(); // Wait for all threads in the block to finish

    // Make a thread-local copy for modification
    for(int i = threadIdx.y; i < N; i += blockDim.y) {
        temp[i] = adj_bits[i];
    }
    __syncthreads();

    // --- Main Evaluation Logic (same as our Cython version) ---
    uint64_t curr_mask = 0;
    for (int i = N - 1; i >= 0; --i) {
        suffix_mask[i] = curr_mask;
        curr_mask |= (uint64_t)1 << perm[i];
    }

    for (int i = 0; i < N; ++i) {
        uint16_t u = perm[i];
        uint64_t succ = temp[u] & suffix_mask[i];
        
        // __popcll is a native GPU instruction for fast bit counting
        degree_out[i] = __popcll(succ);
        
        if (succ == 0) continue;

        uint64_t s = succ;
        while (s > 0) {
            uint64_t v_bit = s & -s;
            s ^= v_bit;
            // __ffsll is a native GPU instruction to find bit index
            int v = __ffsll(v_bit) - 1;
            if (v >= 0 && v < N) {
                temp[v] |= (succ ^ v_bit);
            }
        }
    }
}

// C-style wrapper function that Python can call via ctypes
extern "C" {
    void evaluate_on_gpu(
        const bool *adj_flat,
        const uint16_t *perms_flat,
        uint16_t *degrees_out_flat,
        size_t B,
        size_t N)
    {
        // Configure GPU launch parameters
        int threads_per_block_x = 256;
        int threads_per_block_y = 1; // We use y-dim for parallel data prep
        dim3 threads(threads_per_block_x, threads_per_block_y);
        dim3 blocks((B + threads_per_block_x - 1) / threads_per_block_x);
        
        // Allocate shared memory for adj_bits, temp copy, and suffix_mask
        size_t shared_mem_size = 3 * N * sizeof(uint64_t);

        // Launch the kernel
        _evaluate_kernel<<<blocks, threads, shared_mem_size>>>(adj_flat, perms_flat, degrees_out_flat, B, N);
        
        // Wait for GPU to finish
        cudaDeviceSynchronize();
    }
}

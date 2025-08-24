#include <cuda_runtime.h>
#include <cstdint>

__global__ void _evaluate_kernel(
    const bool *adj_flat,
    const uint16_t *perms_flat,
    uint16_t *degrees_out_flat,
    size_t B,
    size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    const bool* adj = &adj_flat[0];
    const uint16_t* perm = &perms_flat[idx * N];
    uint16_t* degree_out = &degrees_out_flat[idx * N];

    extern __shared__ uint64_t shared_mem[];
    uint64_t* adj_bits = shared_mem;
    uint64_t* temp = &shared_mem[N];
    uint64_t* suffix_mask = &shared_mem[2 * N];

    for(int i = threadIdx.y; i < N; i += blockDim.y) {
        uint64_t bits = 0;
        for(int j = 0; j < N; ++j) {
            if(adj[i * N + j]) {
                bits |= (uint64_t)1 << j;
            }
        }
        adj_bits[i] = bits;
    }
    __syncthreads();

    for(int i = threadIdx.y; i < N; i += blockDim.y) {
        temp[i] = adj_bits[i];
    }
    __syncthreads();

    uint64_t curr_mask = 0;
    for (int i = N - 1; i >= 0; --i) {
        suffix_mask[i] = curr_mask;
        curr_mask |= (uint64_t)1 << perm[i];
    }

    for (int i = 0; i < N; ++i) {
        uint16_t u = perm[i];
        uint64_t succ = temp[u] & suffix_mask[i];
        
        degree_out[i] = __popcll(succ);
        
        if (succ == 0) continue;

        uint64_t s = succ;
        while (s > 0) {
            uint64_t v_bit = s & -s;
            s ^= v_bit;
            int v = __ffsll(v_bit) - 1;
            if (v >= 0 && v < N) {
                // THE FIX IS HERE: Cast the pointer and value to the type CUDA expects.
                atomicOr((unsigned long long int*)&temp[v], (unsigned long long int)(succ ^ v_bit));
            }
        }
    }
}


extern "C" {
    void evaluate_on_gpu(
        const bool *adj_flat,
        const uint16_t *perms_flat,
        uint16_t *degrees_out_flat,
        size_t B,
        size_t N)
    {
        dim3 threads(256, 4);
        dim3 blocks((B + threads.x - 1) / threads.x);
        
        size_t shared_mem_size = 3 * N * sizeof(uint64_t);

        _evaluate_kernel<<<blocks, threads, shared_mem_size>>>(adj_flat, perms_flat, degrees_out_flat, B, N);
        
        cudaDeviceSynchronize();
    }
}

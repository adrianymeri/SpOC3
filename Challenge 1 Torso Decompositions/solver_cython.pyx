#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np

ctypedef np.int64_t INT64_t
ctypedef np.uint64_t UINT64_t

cdef int N
cdef UINT64_t[:] ADJ_BITS

def init_worker_cython(int n_val, np.ndarray[UINT64_t, ndim=1] adj_bits_val):
    global N, ADJ_BITS
    N = n_val
    ADJ_BITS = adj_bits_val

cdef int bitcount_cy(UINT64_t x):
    cdef int c = 0
    while x > 0:
        x &= x - 1
        c += 1
    return c

cpdef tuple evaluate_solution_cy(np.ndarray[INT64_t, ndim=1] solution):
    cdef int t = solution[-1]
    if t < 0: t = 0

    cdef INT64_t[:] perm = solution[:-1]
    cdef int size = N - t
    if size <= 0: return (501, -0)

    cdef UINT64_t[:] suffix_mask = np.zeros(N, dtype=np.uint64)
    cdef UINT64_t curr_mask = 0
    cdef int i, u, v, out_deg, max_width
    cdef UINT64_t succ, s, v_bit, temp_v_bit
    
    for i in range(N - 1, -1, -1):
        suffix_mask[i] = curr_mask
        curr_mask |= (1 << perm[i])

    cdef UINT64_t[:] temp = ADJ_BITS.copy()
    max_width = 0
    
    for i in range(N):
        u = perm[i]
        succ = temp[u] & suffix_mask[i]
        out_deg = bitcount_cy(succ)
        if out_deg > max_width: max_width = out_deg
        if max_width > 500: return (501, -size)
        if succ == 0: continue
        
        s = succ
        while s > 0:
            v_bit = s & -s
            s ^= v_bit
            v = 0
            temp_v_bit = v_bit
            while temp_v_bit > 1:
                temp_v_bit >>= 1
                v += 1
            temp[v] |= (succ ^ v_bit)
            
    # Return (width, -size) for pygmo-style minimization
    return (max_width, -size)

/* _fastwalk.c -- C kernel for the elimination-game walk (GBFC++).
 *
 * Bitset fill-in propagation over uint64 words. Compiled on demand by
 * tools/fastwalk.py (gcc -O3 -shared -fPIC); the Python IncEval is the
 * fallback when no compiler is available. Verified bit-for-bit against
 * core.evaluate by tests in tools/fastwalk.py (self-test on import).
 */
#include <stdint.h>
#include <string.h>

/* Build suffix masks sm[i] = OR of bit(perm[j]) for j > i, for i in
 * [start, n). sm is an (n x W) caller buffer. */
static void build_sm(const int32_t *perm, int n, int W, uint64_t *sm, int start)
{
    memset(sm + (size_t)(n - 1) * W, 0, (size_t)W * 8);
    for (int i = n - 2; i >= start; --i) {
        const uint64_t *prev = sm + (size_t)(i + 1) * W;
        uint64_t *cur = sm + (size_t)i * W;
        memcpy(cur, prev, (size_t)W * 8);
        int v = perm[i + 1];
        cur[v >> 6] |= 1ULL << (v & 63);
    }
}

/* Walk steps [start, n): tmp (n x W) is the fill adjacency at `start`
 * (modified in place); deg_out[i] receives each step's fill degree.
 * If Cstride > 0, snapshot tmp into ckpts[i/Cstride] at every step where
 * i % Cstride == 0 (ckpts is (nckpt x n x W)). */
void walk(const int32_t *perm, int n, int W,
          uint64_t *tmp, uint64_t *sm,
          int start, int32_t *deg_out,
          int Cstride, uint64_t *ckpts)
{
    build_sm(perm, n, W, sm, start);
    uint64_t s[128];                      /* supports n <= 8192 */
    for (int i = start; i < n; ++i) {
        if (Cstride > 0 && (i % Cstride) == 0)
            memcpy(ckpts + (size_t)(i / Cstride) * n * W, tmp,
                   (size_t)n * W * 8);
        int v = perm[i];
        const uint64_t *smv = sm + (size_t)i * W;
        uint64_t *tv = tmp + (size_t)v * W;
        int deg = 0;
        for (int w = 0; w < W; ++w) {
            s[w] = tv[w] & smv[w];
            deg += __builtin_popcountll(s[w]);
        }
        deg_out[i] = deg;
        if (deg < 2)
            continue;                      /* no fill needed */
        for (int w = 0; w < W; ++w) {
            uint64_t x = s[w];
            while (x) {
                int b = __builtin_ctzll(x);
                x &= x - 1;
                int u = (w << 6) | b;
                uint64_t *tu = tmp + (size_t)u * W;
                for (int k = 0; k < W; ++k)
                    tu[k] |= s[k];
                tu[u >> 6] &= ~(1ULL << (u & 63));
            }
        }
    }
}

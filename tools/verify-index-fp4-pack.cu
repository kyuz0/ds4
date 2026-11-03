// Offline verification: pack the captured FP4_E8M0-valued index cache into
// 65 B/key (8-bit power-of-two scale + 128 nibbles) and decode it back; the
// decoded F32 values must be bit-identical to the captured ones.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static __device__ __host__ inline float e2m1_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f; case 1: return 0.5f; case 2: return 1.0f; case 3: return 1.5f;
    case 4: return 2.0f; case 5: return 3.0f; case 6: return 4.0f; default: return 6.0f;
    }
}
static __device__ __host__ inline int e2m1_code(float x) {
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0; float bd = fabsf(ax - e2m1_value(0));
    for (int i = 1; i < 8; i++) {
        float d = fabsf(ax - e2m1_value(i));
        if (d < bd || (d == bd && ((i & 1) == 0) && ((best & 1) != 0))) { best = i; bd = d; }
    }
    return best;
}
/* Packed key: bytes 0..3 = E8M0 exponents of the four per-32-value power-of-two
 * scales (the engine's FP4_E8M0 block size is 32), bytes 4..67 = 128 nibbles
 * (bits 0..2 magnitude code, bit 3 sign). 68 B/key instead of 512 B. */
struct Packed { unsigned char b[68]; };  /* 4 x E8M0 scale + 64 nibble bytes */
static __device__ __host__ inline void pack_key(const float *v, Packed *p) {
    for (int blk = 0; blk < 4; blk++) {
        float amax = 7.052966104933725e-38f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(v[blk*32+i]));
        float scale = exp2f(ceilf(log2f(amax / 6.0f)));
        int e; frexp(scale, &e);            /* scale = 2^(e-1) */
        p->b[blk] = (unsigned char)(e - 1 + 127);
        for (int i = 0; i < 32; i++) {
            float x = fminf(6.0f, fmaxf(-6.0f, v[blk*32+i] / scale));
            unsigned vb; memcpy(&vb, &v[blk*32+i], 4);
            int code = e2m1_code(x) | ((vb >> 31) ? 8 : 0);  /* keep -0 */
            int idx = blk*32 + i;
            unsigned char *dst = &p->b[4 + idx/2];
            if (idx & 1) *dst = (unsigned char)((*dst & 0x0fu) | (code << 4));
            else         *dst = (unsigned char)((*dst & 0xf0u) | code);
        }
    }
}
static __device__ __host__ inline void unpack_key(const Packed *p, float *v) {
    for (int blk = 0; blk < 4; blk++) {
        float scale = ldexpf(1.0f, (int)p->b[blk] - 127);
        for (int i = 0; i < 32; i++) {
            int idx = blk*32 + i;
            int byte = p->b[4 + idx/2];
            int code = (idx & 1) ? (byte >> 4) : (byte & 0x0f);
            float m = e2m1_value(code & 7) * scale;
            v[idx] = (code & 8) ? -m : m;
        }
    }
}
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s KEYS OUT\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    std::vector<float> keys(127u * 128u);
    if (fread(keys.data(), 4, keys.size(), f) != keys.size()) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    size_t bad = 0, n = keys.size() / 128u;
    for (size_t k = 0; k < n; k++) {
        Packed p; float back[128];
        pack_key(&keys[k*128], &p);
        unpack_key(&p, back);
        for (int i = 0; i < 128; i++) {
            uint32_t a, b;
            memcpy(&a, &keys[k*128+i], 4); memcpy(&b, &back[i], 4);
            if (a != b) {
                if (!bad) fprintf(stderr, "key %zu val %d: %.9g != %.9g\n", k, i, keys[k*128+i], back[i]);
                bad++;
            }
        }
    }
    printf("keys=%zu packed_bytes=%zu f32_bytes=%zu bitwise_mismatches=%zu\n",
           n, n*sizeof(Packed), n*128*4, bad);
    return bad ? 1 : 0;
}

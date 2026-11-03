/* Packed key cache for the indexer: the values stored in index_cache are already
 * FP4 (E2M1) times a per-32 power-of-two scale, so they fit in 4 bits.
 * Layout per 128-value key: 4 scale bytes (E8M0 exponent) + 64 nibble bytes
 * (bits 0..2 magnitude code, bit 3 sign) = 68 B instead of 512 B.
 * Verified bitwise against every captured cache (tools/verify-index-fp4-pack.cu). */
#pragma once

#define DS4_INDEX_FP4_KEY_BYTES 68u
#define DS4_INDEX_FP4_HEAD_BYTES 4u

__device__ static inline float ds4_index_e2m1_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

/* Round-to-nearest E2M1 magnitude code (same rule as dsv4_e2m1fn_dequant_dev). */
__device__ static inline int ds4_index_e2m1_code(float x) {
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float bd = fabsf(ax - ds4_index_e2m1_value(0));
    for (int i = 1; i < 8; i++) {
        const float d = fabsf(ax - ds4_index_e2m1_value(i));
        if (d < bd || (d == bd && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            bd = d;
        }
    }
    return best;
}

/* One block per key, 128 threads. Scales are written by lanes 0..3, then each
 * lane below 64 owns exactly one nibble byte and packs both of its values, so no
 * two threads ever read-modify-write the same byte. */
__global__ static void ds4_index_fp4_pack_kernel(
        unsigned char *packed, const float *keys, uint32_t n_keys) {
    const uint32_t key = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (key >= n_keys) return;
    unsigned char *dst = packed + (uint64_t)key * DS4_INDEX_FP4_KEY_BYTES;
    const float *src = keys + (uint64_t)key * 128u;
    if (lane < 4u) {                       /* per-32 pow2_ceil(amax/6), E8M0 byte */
        float amax = 7.052966104933725e-38f;
        for (uint32_t i = 0; i < 32u; i++) amax = fmaxf(amax, fabsf(src[lane * 32u + i]));
        int e;
        frexpf(exp2f(ceilf(log2f(amax / 6.0f))), &e);
        dst[lane] = (unsigned char)(e - 1 + 127);
    }
    __syncthreads();
    if (lane < 64u) {
        unsigned char byte = 0u;
        for (uint32_t h = 0; h < 2u; h++) {
            const uint32_t idx = lane * 2u + h;
            const float v = src[idx];
            const float scale = ldexpf(1.0f, (int)dst[idx >> 5u] - 127);
            const float x = fminf(6.0f, fmaxf(-6.0f, v / scale));
            const int code = ds4_index_e2m1_code(x) | ((__float_as_uint(v) >> 31u) ? 8 : 0);
            byte = (unsigned char)(byte | ((unsigned)code << (h * 4u)));
        }
        dst[DS4_INDEX_FP4_HEAD_BYTES + lane] = byte;
    }
}

__device__ static inline float ds4_index_fp4_load(
        const unsigned char *packed, uint32_t key, uint32_t lane);

/* Diagnostic: count values whose packed decode differs bitwise from the F32 cache. */
__global__ static void ds4_index_fp4_verify_kernel(
        unsigned long long *bad, const unsigned char *packed, const float *keys, uint32_t n_keys) {
    const uint64_t total = (uint64_t)n_keys * 128u;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (uint64_t)gridDim.x * blockDim.x) {
        const float d = ds4_index_fp4_load(packed, (uint32_t)(i >> 7u), (uint32_t)(i & 127u));
        if (__float_as_uint(d) != __float_as_uint(keys[i])) atomicAdd(bad, 1ull);
    }
}

/* Decode value `lane` of key `key` into a float (same arithmetic as the F32 path). */
__device__ static inline float ds4_index_fp4_load(
        const unsigned char *packed, uint32_t key, uint32_t lane) {
    const unsigned char *dst = packed + (uint64_t)key * DS4_INDEX_FP4_KEY_BYTES;
    const uint32_t base = lane & ~31u;
    const float scale = ldexpf(1.0f, (int)dst[base >> 5u] - 127);
    const unsigned char *b = dst + DS4_INDEX_FP4_HEAD_BYTES + (lane >> 1u);
    const int byte = *b;
    const int code = (lane & 1u) ? (byte >> 4) : (byte & 0x0f);
    const float m = ds4_index_e2m1_value(code & 7) * scale;
    return (code & 8) ? -m : m;
}

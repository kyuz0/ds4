/* Fused V4.1 indexer scorer for gfx1151 (default; DS4_ROCM_V41_INDEX_WMMA=0 disables,
 * =3 uses the F32 keys, =2 runs the GEMM path with a sampled exactness check).
 *
 * Replaces the wide FP32 head GEMM + head reduction. Queries and keys are both
 * FP4(E2M1) x power-of-two values per 32-wide block, so after dividing each
 * block by pow2_ceil(amax/6) every element is a multiple of 0.5 no larger than
 * 6: exact in FP16, and every per-block partial dot is exact in the FP32 WMMA
 * accumulator. Blocks are rescaled by exact powers of two and the head
 * reduction keeps the reference order and non-contracted mul/add, so scores
 * equal the FP32 GEMM path whenever that path's 128-term dot is itself exact.
 * No per-head scratch is written. */
#pragma once

__device__ __forceinline__ int v41x_block_exp(float amax) {
    return ilogbf(v41_pow2_ceil(v41_mul(fmaxf(amax, 7.052966104933725e-38f), 1.0f / 6.0f)));
}

#if defined(__gfx1151__)
typedef int v41x_i4 __attribute__((ext_vector_type(4)));
typedef int v41x_i8 __attribute__((ext_vector_type(8)));
typedef _Float16 v41x_h16 __attribute__((ext_vector_type(16)));
typedef float v41x_f8 __attribute__((ext_vector_type(8)));
typedef int v41x_i2 __attribute__((ext_vector_type(2)));
/* int8 operands: own 8 values (2 words) + the partner half-wave's 8 values. */
__device__ __forceinline__ v41x_i4 v41x_distribute8(v41x_i2 own) {
    v41x_i4 out;
    out[0] = own[0]; out[1] = own[1];
    out[2] = __builtin_amdgcn_ds_swizzle(own[0], 0x401f);
    out[3] = __builtin_amdgcn_ds_swizzle(own[1], 0x401f);
    return out;
}
__device__ __forceinline__ int v41x_pack4(float a, float b, float c, float d) {
    return (int)(((uint32_t)(uint8_t)(int8_t)(int)a) | ((uint32_t)(uint8_t)(int8_t)(int)b << 8) |
                 ((uint32_t)(uint8_t)(int8_t)(int)c << 16) | ((uint32_t)(uint8_t)(int8_t)(int)d << 24));
}
union v41x_words { v41x_i8 i; v41x_h16 h; };
/* Same own-half + swap16 operand layout as the qualified halo scorer. */
__device__ __forceinline__ v41x_h16 v41x_distribute(v41x_i4 own) {
    v41x_words out;
#pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        out.i[i] = own[i];
        out.i[i + 4] = __builtin_amdgcn_ds_swizzle(own[i], 0x401f);
    }
    return out.h;
}
__device__ __forceinline__ int v41x_pack2(float a, float b) {
    return __builtin_bit_cast(int, __floats2half2_rn(a, b));
}
#endif

/* Exact sum of four integer block dots d_b * 2^e_b: align to the smallest
 * exponent in int64 (|d_b| < 2^12), convert exactly to double, round once. */
__device__ __forceinline__ float v41x_exact_sum4(int d0, int d1, int d2, int d3,
                                                 int e0, int e1, int e2, int e3) {
    const int lo = min(min(e0, e1), min(e2, e3)), hi = max(max(e0, e1), max(e2, e3));
    if (hi - lo <= 15) {    /* |d| < 2^13: four aligned terms stay below 2^31 */
        const int acc = (d0 << (e0 - lo)) + (d1 << (e1 - lo)) + (d2 << (e2 - lo)) + (d3 << (e3 - lo));
        return ldexpf((float)acc, lo);
    }
    if (hi - lo <= 38) {
        const long long acc = ((long long)d0 << (e0 - lo)) + ((long long)d1 << (e1 - lo)) +
                              ((long long)d2 << (e2 - lo)) + ((long long)d3 << (e3 - lo));
        return (float)ldexp((double)acc, lo);
    }
    return (float)(ldexp((double)d0, e0) + ldexp((double)d1, e1) +
                   ldexp((double)d2, e2) + ldexp((double)d3, e3));
}

/* One block: 128 keys (8 waves x 16) x 32 query rows (two WMMA row tiles).
 * PACKED reads the 68-byte FP4 cache (4 E8M0 bytes + 64 nibble bytes): twice a
 * code's magnitude is already the int8 operand, with exponent byte-127. */
template <bool PACKED>
__global__ __launch_bounds__(256) static void v41_indexer_wmma_kernel(float *out,
        const int8_t *qn, const int *qe, const float *w, const float *keys, const unsigned char *packed,
        unsigned width, unsigned rows, unsigned start, unsigned ratio) {
#if defined(__gfx1151__)
    const unsigned tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5;
    const unsigned tc = blockIdx.x * 128u, tt = blockIdx.y * 32u;
    const unsigned last = min(tt + 32u, rows);
    if (tc >= (start + last) / ratio) {
        for (unsigned i = tid; i < 4096u; i += 256u) {
            const unsigned t = tt + i / 128u, c = tc + i % 128u;
            if (t < rows && c < width) out[(uint64_t)t * width + c] = -INFINITY;
        }
        return;
    }
    const unsigned c = tc + wave * 16u + (lane & 15u), khalf = (lane >> 4) * 8u;
    const bool cv = c < width;
    v41x_i4 bk[8];
    int ek[4];
    if (PACKED) {
        const unsigned char *kp = packed + (uint64_t)(cv ? c : 0u) * 68u;
#pragma unroll
        for (unsigned b = 0; b < 4u; ++b) {
            ek[b] = (int)kp[b] - 127;
#pragma unroll
            for (unsigned kk = 0; kk < 2u; ++kk) {
                const unsigned d0 = b * 32u + kk * 16u + khalf;
                const uint32_t nib = cv ? *reinterpret_cast<const uint32_t *>(kp + 4u + d0 / 2u) : 0u;
                v41x_i2 o;
#pragma unroll
                for (unsigned i = 0; i < 2u; ++i) {
                    uint32_t word = 0;
#pragma unroll
                    for (unsigned j = 0; j < 4u; ++j) {
                        const uint32_t code = (nib >> (4u * (4u * i + j))) & 15u;
                        const int mag = (int)((0xC8643210u >> (4u * (code & 7u))) & 15u);
                        word |= (uint32_t)(uint8_t)(int8_t)((code & 8u) ? -mag : mag) << (8u * j);
                    }
                    o[i] = (int)word;
                }
                bk[2u * b + kk] = v41x_distribute8(o);
            }
        }
    } else {
#pragma unroll
        for (unsigned b = 0; b < 4u; ++b) {
            float v[16];
#pragma unroll
            for (unsigned j = 0; j < 16u; ++j) {
                const unsigned d = (2u * b + j / 8u) * 16u + khalf + (j & 7u);
                v[j] = cv ? keys[(uint64_t)c * 128u + d] : 0.0f;
            }
            float amax = 0.0f;
#pragma unroll
            for (unsigned j = 0; j < 16u; ++j) amax = fmaxf(amax, fabsf(v[j]));
            amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
            const int e = v41x_block_exp(amax);
            ek[b] = e;
            v41x_i2 o0, o1;
#pragma unroll
            for (unsigned i = 0; i < 2u; ++i) {
                o0[i] = v41x_pack4(ldexpf(v[4 * i], 1 - e), ldexpf(v[4 * i + 1], 1 - e),
                                   ldexpf(v[4 * i + 2], 1 - e), ldexpf(v[4 * i + 3], 1 - e));
                o1[i] = v41x_pack4(ldexpf(v[8 + 4 * i], 1 - e), ldexpf(v[8 + 4 * i + 1], 1 - e),
                                   ldexpf(v[8 + 4 * i + 2], 1 - e), ldexpf(v[8 + 4 * i + 3], 1 - e));
            }
            bk[2 * b] = v41x_distribute8(o0);
            bk[2 * b + 1] = v41x_distribute8(o1);
        }
    }
    v41x_f8 total0 = {}, total1 = {};
    /* Per-block exponents and head weights for all 32 rows x 32 heads. */
    __shared__ int eqs[32 * 32 * 4];
    __shared__ float whs[32 * 32];
    for (unsigned i = tid; i < 32u * 32u * 4u; i += 256u) {
        const unsigned t = tt + i / 128u;
        eqs[i] = t < rows ? qe[(uint64_t)tt * 128u + i] : 0;
    }
    for (unsigned i = tid; i < 32u * 32u; i += 256u) {
        const unsigned t = tt + i / 32u;
        whs[i] = t < rows ? w[(uint64_t)tt * 32u + i] : 0.0f;
    }
    __syncthreads();
    for (unsigned h = 0; h < 32u; ++h) {
#pragma unroll
        for (unsigned rt = 0; rt < 2u; ++rt) {
            const int8_t *qrow = qn + ((uint64_t)(tt + rt * 16u + (lane & 15u)) * 32u + h) * 128u;
            v41x_i8 d[4];
#pragma unroll
            for (unsigned b = 0; b < 4u; ++b) {
                d[b] = (v41x_i8){};
#pragma unroll
                for (unsigned kk = 0; kk < 2u; ++kk) {
                    const unsigned k = 2u * b + kk;
                    const v41x_i2 own = *reinterpret_cast<const v41x_i2 *>(qrow + k * 16u + khalf);
                    d[b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, v41x_distribute8(own),
                                                                       true, bk[k], d[b], false);
                }
            }
#pragma unroll
            for (unsigned i = 0; i < 8u; ++i) {
                const unsigned r = rt * 16u + 2u * i + (lane >> 4);
                const int *er = eqs + (r * 32u + h) * 4u;
                const float dot = v41x_exact_sum4(d[0][i], d[1][i], d[2][i], d[3][i],
                    er[0] + ek[0] - 2, er[1] + ek[1] - 2, er[2] + ek[2] - 2, er[3] + ek[3] - 2);
                float weighted;
                asm volatile("v_mul_f32 %0, %1, %2" : "=v"(weighted)
                    : "v"(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f)), "v"(whs[r * 32u + h]));
                if (rt == 0u) total0[i] = v41_add(total0[i], weighted);
                else total1[i] = v41_add(total1[i], weighted);
            }
        }
    }
#pragma unroll
    for (unsigned rt = 0; rt < 2u; ++rt)
#pragma unroll
    for (unsigned i = 0; i < 8u; ++i) {
        const unsigned t = tt + rt * 16u + 2u * i + (lane >> 4);
        if (t < rows && cv)
            out[(uint64_t)t * width + c] = c >= (start + t + 1u) / ratio ? -INFINITY :
                (rt == 0u ? total0[i] : total1[i]);
    }
#endif
}

/* Normalize queries once: qn = 2*q/pow2_ceil(amax/6) per 32-block (exact int8),
 * qe = block exponent. Rows up to the 32-row tile padding are zero. */
__global__ static void v41_indexer_wmma_qprep_kernel(int8_t *qn, int *qe, const float *q,
                                                     unsigned rows) {
    const unsigned blk = blockIdx.x, lane = threadIdx.x;     /* one 32-value block per wave */
    const unsigned t = blk / 128u;
    const float v = t < rows ? q[(uint64_t)blk * 32u + lane] : 0.0f;
    float amax = fabsf(v);
    for (int dl = 16; dl; dl >>= 1) amax = fmaxf(amax, __shfl_xor(amax, dl, 32));
    const int e = v41x_block_exp(amax);
    qn[(uint64_t)blk * 32u + lane] = (int8_t)(int)ldexpf(v, 1 - e);
    if (!lane) qe[blk] = e;
}

static int v41_indexer_wmma_launch(float *out, const float *q, const float *w,
        const float *keys, const unsigned char *packed, unsigned width, unsigned rows,
        unsigned start, unsigned ratio) {
    const unsigned rows32 = (rows + 31u) & ~31u;
    const uint64_t qn_bytes = (uint64_t)rows32 * 32u * 128u, qe_bytes = (uint64_t)rows32 * 32u * 4u * 4u;
    char *scratch = (char *)cuda_tmp_alloc(qn_bytes + qe_bytes, "V4.1 WMMA indexer queries");
    if (!scratch) return 0;
    int8_t *qn = (int8_t *)scratch;
    int *qe = (int *)(scratch + qn_bytes);
    v41_indexer_wmma_qprep_kernel<<<rows32 * 128u, 32u>>>(qn, qe, q, rows);
    const dim3 grid((width + 127u) / 128u, rows32 / 32u);
    if (packed) v41_indexer_wmma_kernel<true><<<grid, 256u>>>(out, qn, qe, w, keys, packed, width, rows, start, ratio);
    else v41_indexer_wmma_kernel<false><<<grid, 256u>>>(out, qn, qe, w, keys, packed, width, rows, start, ratio);
    return 1;
}

static bool v41_indexer_wmma_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_ROCM_V41_INDEX_WMMA");
        cached = (!v || !*v || v[0] == '1' || v[0] == '3') ? 1 : 0;
    }
    return cached == 1;
}

/* Diagnostic (DS4_ROCM_V41_INDEX_WMMA=2): compare GEMM scores (a) and fused
 * scores (b) against a reference whose per-head dots are computed in double
 * (exact for FP4 x pow2 inputs) with the same float head reduction. */
__global__ static void v41_indexer_wmma_check_kernel(unsigned long long *cnt,
        const float *a, const float *b, const float *q, const float *w,
        const float *keys, unsigned width, unsigned rows, unsigned start, unsigned ratio) {
    const unsigned k = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (k >= width || t >= rows) return;
    const uint64_t i = (uint64_t)t * width + k;
    if (k >= (start + t + 1u) / ratio) return;
    float ref = 0.0f;
    bool dot_inexact = false;
    for (unsigned h = 0; h < 32u; ++h) {
        double dd = 0.0;
        const float *qp = q + ((uint64_t)t * 32u + h) * 128u, *kp = keys + (uint64_t)k * 128u;
        for (unsigned d = 0; d < 128u; ++d) dd += (double)qp[d] * (double)kp[d];
        const float dot = (float)dd;
        if ((double)dot != dd) dot_inexact = true;
        ref = v41_add(ref, v41_mul(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f), w[t * 32u + h]));
    }
    if (t == 0u) {           /* once per key: off-grid key blocks */
        for (unsigned b = 0; b < 4u; ++b) {
            float amax = 0.f;
            for (unsigned d = 0; d < 32u; ++d) amax = fmaxf(amax, fabsf(keys[(uint64_t)k * 128u + b * 32u + d]));
            const int e = v41x_block_exp(amax);
            for (unsigned d = 0; d < 32u; ++d) {
                const float v = ldexpf(keys[(uint64_t)k * 128u + b * 32u + d], 1 - e);
                if (v != rintf(v) || fabsf(v) > 12.f) { atomicAdd(&cnt[5], 1ull); break; }
            }
        }
    }
    if (k == 0u) {
        for (unsigned h = 0; h < 32u; ++h) for (unsigned b = 0; b < 4u; ++b) {
            const float *qp = q + ((uint64_t)t * 32u + h) * 128u + b * 32u;
            float amax = 0.f;
            for (unsigned d = 0; d < 32u; ++d) amax = fmaxf(amax, fabsf(qp[d]));
            const int e = v41x_block_exp(amax);
            for (unsigned d = 0; d < 32u; ++d) {
                const float v = ldexpf(qp[d], 1 - e);
                if (v != rintf(v) || fabsf(v) > 12.f) { atomicAdd(&cnt[6], 1ull); break; }
            }
        }
    }
    atomicAdd(&cnt[0], 1ull);
    if (a[i] != b[i]) atomicAdd(&cnt[1], 1ull);
    if (a[i] != ref) atomicAdd(&cnt[2], 1ull);
    if (b[i] != ref) atomicAdd(&cnt[3], 1ull);
    if (dot_inexact) atomicAdd(&cnt[4], 1ull);
}

#pragma once
#include "decode.hpp"
#include "direct_feed.hpp"
template <int MTILES=8, int BM=16, int BN=16, int BK=16, bool MID_F16=false, bool OUT_F16=false, bool SLOT_MAJOR=false, bool COMPACT=false>
__global__ static void q2_direct_x_n2_kernel(
        float *down_out,
        half *down_out_h,
        const char *down_base,
        const float *mid,
        const half *mid_h,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *pairs,
        const uint32_t *hot_experts,
        uint32_t hot_count,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t n_expert,
        uint32_t n_tokens = 0u) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
    extern __shared__ unsigned char raw_sh[];
    half *shA = reinterpret_cast<half *>(raw_sh);
    half *shB0 = shA + MTILES * BM * BK;
    half *shB1 = shB0 + BK * BN;
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t RAW_ROWS = 2u * BN;
    uint32_t *shW = reinterpret_cast<uint32_t *>(shB1 + BK * BN);
    float *shC = reinterpret_cast<float *>(shW + RAW_ROWS * RAW_DWORDS);
    const uint32_t hot_idx = (uint32_t)blockIdx.z;
    if (hot_idx >= hot_count) return;
    const uint32_t descriptor = hot_experts[hot_idx];
    const uint32_t expert = COMPACT ? (descriptor & 255u) : descriptor;
    const uint32_t count = counts[expert];
    const uint32_t m_group0 = (COMPACT ? (descriptor >> 8u) : (uint32_t)blockIdx.y) * MTILES * BM;
    if (m_group0 >= count) return;
    const uint32_t n0 = (uint32_t)blockIdx.x * (2u * BN);
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t first = offsets[expert];
    __shared__ uint32_t shPair[MTILES * BM];
    for (uint32_t j = tid; j < MTILES * BM; j += blockDim.x) {
        const uint32_t bucket_row = m_group0 + j;
        shPair[j] = (bucket_row < count) ? pairs[first + bucket_row] : UINT32_MAX;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, BM, BN, BK, half, rocwmma::row_major>;
    using frag_b = rocwmma::fragment<rocwmma::matrix_b, BM, BN, BK, half, rocwmma::col_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, BM, BN, BK, float>;
    static_assert(MTILES==4 && BM==16 && BN==16 && BK==16);
    static_assert(MID_F16 && OUT_F16 && !SLOT_MAJOR && COMPACT);
    const uint32_t x_pair = shPair[wave * BM + (tid & 15u)];
    const uint64_t x_start = uint64_t(x_pair == UINT32_MAX ? 0u : x_pair) * expert_mid_dim + ((tid & 16u) >> 1u);
    const half *x_row = mid_h + x_start;
    frag_a a;
    frag_b b0;
    frag_b b1;
    frag_c acc0;
    frag_c acc1;
    if (wave < MTILES) {
        rocwmma::fill_fragment(acc0, 0.0f);
        rocwmma::fill_fragment(acc1, 0.0f);
    }

    const unsigned char *dew = (const unsigned char *)down_base + (uint64_t)expert * down_expert_bytes;
    for (uint32_t kb = 0; kb < expert_mid_dim; kb += 256u) {
        for (uint32_t j = tid; j < RAW_ROWS * RAW_DWORDS; j += blockDim.x) {
            const uint32_t row_local = j / RAW_DWORDS;
            const uint32_t word = j - row_local * RAW_DWORDS;
            const uint32_t row = n0 + row_local;
            uint32_t v = 0u;
            if (row < out_dim) {
                const unsigned char *blk = dew + (uint64_t)row * down_row_bytes +
                                           (uint64_t)(kb >> 8u) * 84u;
                v = *reinterpret_cast<const uint32_t *>(blk + word * sizeof(uint32_t));
            }
            shW[j] = v;
        }
        __syncthreads();

        for (uint32_t krel = 0; krel < 256u && kb + krel < expert_mid_dim; krel += BK) {
            const uint32_t k0 = kb + krel;
            halo_q2::q2_K_dequant_pair_tile_half_rowwise_staged<BN, BK>(
                    shB0, shB1, shW, krel, tid);
            __syncthreads();
            if (wave < MTILES) {
                q2_direct_x::load_eight(a, x_row, k0, x_pair != UINT32_MAX);
                rocwmma::load_matrix_sync(b0, shB0, BN);
                rocwmma::load_matrix_sync(b1, shB1, BN);
                rocwmma::mma_sync(acc0, a, b0, acc0);
                rocwmma::mma_sync(acc1, a, b1, acc1);
            }
            __syncthreads();
        }
    }

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc0, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row0 = n0 + nn;
            const uint32_t tok = pair / n_expert;
            const uint32_t slot = pair - tok * n_expert;
            if (row0 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out_h[dst] = __float2half(shC[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row0;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row0;
                    down_out[dst] = shC[j];
                }
            }
        }
    }
    __syncthreads();

    if (wave < MTILES) {
        rocwmma::store_matrix_sync(shC + wave * BM * BN, acc1, BN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (uint32_t j = tid; j < MTILES * BM * BN; j += blockDim.x) {
        const uint32_t mt = j / (BM * BN);
        const uint32_t rem = j - mt * BM * BN;
        const uint32_t mm = rem / BN;
        const uint32_t nn = rem - mm * BN;
        const uint32_t pair = shPair[mt * BM + mm];
        if (pair != UINT32_MAX) {
            const uint32_t row1 = n0 + BN + nn;
            const uint32_t tok = pair / n_expert;
            const uint32_t slot = pair - tok * n_expert;
            if (row1 < out_dim) {
                if (OUT_F16) {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out_h[dst] = __float2half(shC[j]);
                } else {
                    uint64_t dst = (uint64_t)pair * out_dim + row1;
                    if (SLOT_MAJOR) dst = ((uint64_t)slot * n_tokens + tok) * out_dim + row1;
                    down_out[dst] = shC[j];
                }
            }
        }
    }
#endif
}

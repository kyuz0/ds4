#pragma once
#include "support.cuh"
#include "direct_feed.hpp"

#include "kv_access.hpp"
template <int MODE, uint32_t HEADS, bool F32_VEC2 = false>
__global__ static void bdqk_half_fused(
        half *packed,
        const float *sinks,
        const float *q,
        const half *raw_kv,
        const half *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_ring_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim, P2Rope rp, float *audit_heads) {
    constexpr uint32_t KEYS = 80u;
    constexpr uint32_t DIMS = 64u;
    constexpr uint32_t TILE = 16u;
    constexpr uint32_t STRIDE = KEYS;
    constexpr uint32_t HEAD_TILES = HEADS / TILE;
    constexpr uint32_t KEY_TILES = KEYS / TILE;
    constexpr uint32_t WORKGROUP = HEADS * 16u;
    const uint32_t t = (uint32_t)blockIdx.x;
    const uint32_t head_base = (uint32_t)blockIdx.y * HEADS;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t wave = tid >> 5u;

    __shared__ half sh_qp[HEADS * STRIDE];
    __shared__ half sh_kv[KEYS * STRIDE];
    __shared__ float sh_matrix[HEADS * STRIDE];
    __shared__ float row_max[HEADS];
    __shared__ float row_sum[HEADS];
    __shared__ float old_scale[HEADS];
    __shared__ uint32_t indexed_raw_rows[256];
    __shared__ uint32_t indexed_comp_rows[DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP];
    __shared__ uint32_t indexed_raw_count;
    __shared__ uint32_t indexed_raw_first;
    __shared__ uint32_t indexed_comp_count;

    uint32_t raw_count;
    uint32_t raw_first;
    uint32_t comp_count = 0u;
    if constexpr (MODE == 1) {
        const uint32_t qpos = pos0 + t;
        const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
        uint32_t visible_comp = n_comp;
        if (ratio != 0u) {
            visible_comp = (qpos + 1u) / ratio;
            if (visible_comp > n_comp) visible_comp = n_comp;
        }
        if (tid == 0u) {
            indexed_raw_count = 0u;
            indexed_raw_first = 0u;
            indexed_comp_count = 0u;
            if (n_raw != 0u) {
                const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
                if (qpos >= first_raw_pos) {
                    uint32_t lo = first_raw_pos;
                    if (window != 0u && qpos + 1u > window) {
                        const uint32_t wlo = qpos + 1u - window;
                        if (wlo > lo) lo = wlo;
                    }
                    const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                    if (hi >= lo) {
                        indexed_raw_first = lo - first_raw_pos;
                        indexed_raw_count = hi - lo + 1u;
                        if (indexed_raw_count > 256u) indexed_raw_count = 256u;
                    }
                }
            }
            for (uint32_t i = 0u;
                 i < top_k && indexed_comp_count < DS4_ROCM_ATTENTION_INDEXED_TOPK_CAP;
                 i++) {
                const int32_t ci = topk[(uint64_t)t * top_k + i];
                if (ci >= 0 && (uint32_t)ci < visible_comp) {
                    indexed_comp_rows[indexed_comp_count++] = (uint32_t)ci;
                }
            }
        }
        __syncthreads();
        for (uint32_t r = tid; r < indexed_raw_count; r += blockDim.x) {
            indexed_raw_rows[r] = (raw_ring_start + indexed_raw_first + r) % raw_cap;
        }
        __syncthreads();
        raw_count = indexed_raw_count;
        raw_first = 0u;
        comp_count = indexed_comp_count;
    } else if constexpr (MODE == 2) {
        const uint32_t qpos = pos0 + t;
        const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
        if (tid == 0u) {
            indexed_raw_count = 0u;
            indexed_raw_first = 0u;
            if (n_raw != 0u) {
                const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
                if (qpos >= first_raw_pos) {
                    uint32_t lo = first_raw_pos;
                    if (window != 0u && qpos + 1u > window) {
                        const uint32_t wlo = qpos + 1u - window;
                        if (wlo > lo) lo = wlo;
                    }
                    const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                    if (hi >= lo) {
                        indexed_raw_first = lo - first_raw_pos;
                        indexed_raw_count = hi - lo + 1u;
                        if (indexed_raw_count > 256u) indexed_raw_count = 256u;
                    }
                }
            }
            indexed_comp_count = n_comp;
            if (ratio != 0u) {
                indexed_comp_count = (qpos + 1u) / ratio;
                if (indexed_comp_count > n_comp) indexed_comp_count = n_comp;
            }
        }
        __syncthreads();
        for (uint32_t r = tid; r < indexed_raw_count; r += blockDim.x) {
            indexed_raw_rows[r] = (raw_ring_start + indexed_raw_first + r) % raw_cap;
        }
        __syncthreads();
        raw_count = indexed_raw_count;
        raw_first = 0u;
        comp_count = indexed_comp_count;
    } else {
        raw_count = window != 0u && t + 1u > window ? window : t + 1u;
        raw_first = t + 1u - raw_count;
        if (n_comp != 0u && ratio != 0u) {
            comp_count = (t + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);

    float accum[32];
#pragma unroll
    for (uint32_t i = 0; i < 32u; i++) accum[i] = 0.0f;
    if (tid < HEADS) {
        const uint32_t head = head_base + tid;
        row_max[tid] = head < n_head ? sinks[head] : -INFINITY;
        row_sum[tid] = head < n_head ? 1.0f : 0.0f;
    }
    __syncthreads();

    using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major>;
    using frag_b_col = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major>;
    using frag_b_row = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::row_major>;
    using frag_c = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

    for (uint32_t kb = 0; kb < n_score; kb += KEYS) {
        frag_a qa;
        frag_b_col kk;
        frag_c score_acc;
        if (wave < HEAD_TILES * KEY_TILES) rocwmma::fill_fragment(score_acc, 0.0f);

        if (wave < HEAD_TILES * KEY_TILES) {
            const uint32_t head_tile = wave / KEY_TILES;
            const uint32_t key_tile = wave - head_tile * KEY_TILES;
            const uint32_t lane = tid & 31u;
            const uint32_t head = head_base + head_tile * 16u + bdqk::row_from_lane(lane);
            const bool query_valid = head < n_head;
            const float *query_row = query_valid ? q + ((uint64_t)t * n_head + head) * 512u : q;
            const uint32_t si = kb + key_tile * 16u + bdqk::row_from_lane(lane);
            const bool key_valid = si < n_score;
            const half *key_row = raw_kv;
            if (key_valid) {
                if (si < raw_count) {
                    const uint32_t row = MODE != 0 ? indexed_raw_rows[si] : raw_first + si;
                    key_row = raw_kv + (uint64_t)row * 512u;
                } else {
                    const uint32_t local = si - raw_count;
                    const uint32_t row = MODE == 1 ? indexed_comp_rows[local] : local;
                    key_row = comp_kv + (uint64_t)row * 512u;
                }
            }
            for (uint32_t d0 = 0; d0 < 512u; d0 += TILE) {
                const uint32_t k = d0 + bdqk::k_from_lane(lane);
                bdqk::query_eight(qa, query_row, k, query_valid);
                bdqk::key_half_eight(kk, key_row, k, key_valid);
                rocwmma::mma_sync(score_acc, qa, kk, score_acc);
            }
        }
        if (wave < HEAD_TILES * KEY_TILES) {
            const uint32_t head_tile = wave / KEY_TILES;
            const uint32_t key_tile = wave - head_tile * KEY_TILES;
            rocwmma::store_matrix_sync(
                                       sh_matrix + head_tile * TILE * STRIDE + key_tile * TILE,
                                       score_acc,
                                       STRIDE, rocwmma::mem_row_major);
        }
        __syncthreads();

        const uint32_t hl = tid >> 4u;
        const uint32_t lane16 = tid & 15u;
        float block_max = -INFINITY;
#pragma unroll
        for (uint32_t i = 0; i < KEY_TILES; i++) {
            const uint32_t kl = lane16 + i * 16u;
            if (kb + kl < n_score) {
                block_max = fmaxf(block_max, sh_matrix[hl * STRIDE + kl] * scale);
            }
        }
#pragma unroll
        for (uint32_t delta = 1u; delta < 16u; delta <<= 1u) {
            block_max = fmaxf(block_max,
                              __shfl_xor_sync(FULL_WARP_MASK, block_max, delta, 32));
        }
        const float prev_max = row_max[hl];
        const float prev_sum = row_sum[hl];
        const float next_max = fmaxf(prev_max, block_max);
        const float prev_scale = prev_sum == 0.0f ? 0.0f : expf(prev_max - next_max);
        float block_sum = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < KEY_TILES; i++) {
            const uint32_t kl = lane16 + i * 16u;
            float p = 0.0f;
            if (kb + kl < n_score) {
                p = expf(sh_matrix[hl * STRIDE + kl] * scale - next_max);
                block_sum += p;
            }
            sh_qp[hl * STRIDE + kl] = __float2half(p);
        }
#pragma unroll
        for (uint32_t delta = 1u; delta < 16u; delta <<= 1u) {
            block_sum += __shfl_xor_sync(FULL_WARP_MASK, block_sum, delta, 32);
        }
        if (lane16 == 0u) {
            row_max[hl] = next_max;
            row_sum[hl] = prev_sum * prev_scale + block_sum;
            old_scale[hl] = prev_scale;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t out_idx = tid + i * WORKGROUP;
            accum[i] *= old_scale[out_idx / 512u];
        }

        for (uint32_t dim0 = 0; dim0 < 512u; dim0 += DIMS) {
            if constexpr (F32_VEC2) {
                kvh_stage<DIMS>(sh_kv, raw_kv, comp_kv,
                    indexed_raw_rows, indexed_comp_rows, raw_count, n_score,
                    kb, dim0, tid, blockDim.x);
            } else {
                for (uint32_t idx = tid; idx < KEYS * DIMS; idx += blockDim.x) {
                    const uint32_t kl = idx / DIMS;
                    const uint32_t d = idx - kl * DIMS;
                    const uint32_t score_idx = kb + kl;
                    float v = 0.0f;
                    if (score_idx < n_score) {
                        if (score_idx < raw_count) {
                            const uint32_t row = MODE != 0
                                ? indexed_raw_rows[score_idx]
                                : raw_first + score_idx;
                            v = raw_kv[(uint64_t)row * 512u + dim0 + d];
                        } else {
                            const uint32_t comp_local = score_idx - raw_count;
                            const uint32_t row = MODE == 1
                                ? indexed_comp_rows[comp_local]
                                : comp_local;
                            v = comp_kv[(uint64_t)row * 512u + dim0 + d];
                        }
                    }
                    sh_kv[kl * STRIDE + d] = __float2half(v);
                }
            }
            __syncthreads();

            frag_c pv_acc;
            if (wave < HEAD_TILES * 4u) rocwmma::fill_fragment(pv_acc, 0.0f);
#pragma unroll
            for (uint32_t kc = 0; kc < KEY_TILES; kc++) {
                frag_a pp;
                frag_b_row vv;
                if (wave < HEAD_TILES * 4u) {
                    const uint32_t head_tile = wave >> 2u;
                    const uint32_t dim_tile = wave & 3u;
                    rocwmma::load_matrix_sync(pp,
                            sh_qp + head_tile * TILE * STRIDE + kc * TILE, STRIDE);
                    rocwmma::load_matrix_sync(vv,
                            sh_kv + kc * TILE * STRIDE + dim_tile * TILE, STRIDE);
                    rocwmma::mma_sync(pv_acc, pp, vv, pv_acc);
                }
            }
            if (wave < HEAD_TILES * 4u) {
                const uint32_t head_tile = wave >> 2u;
                const uint32_t dim_tile = wave & 3u;
                rocwmma::store_matrix_sync(
                                           sh_matrix + head_tile * TILE * STRIDE + dim_tile * TILE,
                                           pv_acc,
                                           STRIDE, rocwmma::mem_row_major);
            }
            __syncthreads();

#pragma unroll
            for (uint32_t i = 0; i < 32u; i++) {
                const uint32_t out_idx = tid + i * WORKGROUP;
                const uint32_t head_local = out_idx / 512u;
                const uint32_t dim = out_idx - head_local * 512u;
                if (dim >= dim0 && dim < dim0 + DIMS) {
                    accum[i] += sh_matrix[head_local * STRIDE + dim - dim0];
                }
            }
            __syncthreads();
        }
    }

    static_assert(MODE==1 && HEADS==32 && F32_VEC2, "Frozen fresh4K dispatch only");
    __shared__ float2 cs[32];
    if(tid<32)cs[tid]=p2_coeff(rp,t,tid);
    __syncthreads();
#pragma unroll
    for(uint32_t i=0;i<32;++i) {
        const uint32_t head=head_base+i, dim=tid;
        const float inv=row_sum[i]==0.0f?0.0f:1.0f/row_sum[i];
        float value=p2_fp32(accum[i]*inv);
        if(audit_heads)audit_heads[((uint64_t)t*64+head)*512+dim]=value;
        if(dim>=448) {
            // All lanes reconverge before the shuffle; adjacent even/odd
            // dimensions belong to the same head and wave.
            const float other=__shfl_xor(value,1,32);
            const float x0=(dim&1)?other:value, x1=(dim&1)?value:other;
            const float2 z=cs[(dim-448)/2];
            value=(dim&1)?p2_fp32(x0*z.y+x1*z.x):p2_fp32(x0*z.x-x1*z.y);
        }
        packed[((uint64_t)(head/8)*n_tokens+t)*4096+(head%8)*512+dim]=__float2half_rn(p2_fp32(value));
    }
}

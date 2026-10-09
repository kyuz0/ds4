// SPDX-License-Identifier: MIT
namespace halo_attention {
#include "halo/attention/api.hpp"
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
#include "halo/attention/half.cuh"
#include "halo/attention/compact.hpp"
__global__ static void convert(half *dst,const float *raw,const float *comp,
        uint32_t nr,uint32_t cap,uint32_t start,uint32_t nc) {
    public2048_convert_one(dst,raw,comp,(uint64_t)blockIdx.x*256+threadIdx.x,nr,cap,start,nc);
}
#else
// Unreachable registration stubs; admission requires gfx1151.
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
        uint32_t head_dim, P2Rope rp, float *audit_heads) {}
__global__ static void convert(half*,const float*,const float*,uint32_t,uint32_t,uint32_t,uint32_t) {}
#endif
}

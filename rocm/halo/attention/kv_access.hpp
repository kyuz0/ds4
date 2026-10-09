#pragma once
#include <hip/hip_fp16.h>
#include <stdint.h>
#define KVH_HD __host__ __device__ __forceinline__
// The exact native vec2 conversion, also executed by the CPU probe through
// HIP's host implementation. No floating-point arithmetic before conversion.
KVH_HD half2 kvh_convert_pair(const float *src, uint64_t pair) {
    const float2 f = *reinterpret_cast<const float2 *>(src + pair * 2u);
    return __floats2half2_rn(f.x, f.y);
}
KVH_HD void kvh_convert_one(half *dst,const float *raw,const float *comp,
                          uint64_t pair,uint64_t raw_pairs,uint64_t total_pairs) {
    if(pair >= total_pairs) return;
    const half2 v=pair<raw_pairs ? kvh_convert_pair(raw,pair)
                              : kvh_convert_pair(comp,pair-raw_pairs);
    *reinterpret_cast<half2 *>(dst+pair*2u)=v;
}
// This is the actual score AND value consumer staging path. Only the global
// source type changes; LDS coordinates, masked +0, and ownership stay native.
template<unsigned WIDTH> KVH_HD void kvh_stage(half *sh_kv,
 const half *raw_kv,const half *comp_kv,const uint32_t *indexed_raw_rows,
 const uint32_t *indexed_comp_rows,uint32_t raw_count,uint32_t n_score,
 uint32_t kb,uint32_t dim0,uint32_t tid,uint32_t block_size) {
    for(uint32_t idx=tid;idx<80u*(WIDTH/2u);idx+=block_size) {
        const uint32_t kl=idx/(WIDTH/2u);
        const uint32_t d=(idx-kl*(WIDTH/2u))*2u;
        const uint32_t score_idx=kb+kl;
        half2 v=__floats2half2_rn(0.0f,0.0f);
        if(score_idx<n_score) {
            const half *src;
            if(score_idx<raw_count)
                src=raw_kv+(uint64_t)indexed_raw_rows[score_idx]*512u;
            else
                src=comp_kv+(uint64_t)indexed_comp_rows[score_idx-raw_count]*512u;
            v=*reinterpret_cast<const half2 *>(src+dim0+d);
        }
        *reinterpret_cast<half2 *>(sh_kv+kl*80u+d)=v;
    }
}

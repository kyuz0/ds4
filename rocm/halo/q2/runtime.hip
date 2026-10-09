// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include "../../../ds4_rocm_memory.h"
#include "../../ds4_rocm_halo_q2.h"
#include "decode.hpp"
__global__ static void halo_q2_prepare(half *dst,const unsigned char *src) {
    __shared__ uint32_t raw[32*21];
    __shared__ half decoded[32*16];
    unsigned row0=blockIdx.x*32, kb=blockIdx.y*256,tid=threadIdx.x;
    for(unsigned j=tid;j<32*21;j+=128)
        raw[j]=reinterpret_cast<const uint32_t*>(src+(row0+j/21)*672+(kb/256)*84)[j%21];
    __syncthreads();
    for(unsigned kr=0;kr<256;kr+=16){
        q2_K_dequant_pair_tile_half_rowwise_staged<16,16>(decoded,decoded+256,raw,kr,tid);
        __syncthreads();
        for(unsigned j=tid;j<512;j+=128)dst[(row0+j/16)*2048+kb+kr+j%16]=decoded[j];
        __syncthreads();
    }
}
__global__ static void halo_q2_gather(half *dst,const half *src,
        const uint32_t *pairs,unsigned first,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count*1024u)reinterpret_cast<uint32_t*>(dst)[i]=
        reinterpret_cast<const uint32_t*>(src)[size_t(pairs[first+i/1024])*1024+i%1024];
}

extern "C" __global__ void halo_q2_prepared_store_kernel(float*,const half*,const half*,int,int,int,const unsigned*,unsigned,half*);
static void *halo_q2_arena;
int ds4_rocm_halo_q2_prepare() {
    if (halo_q2_arena) return 1;
    if (ds4_rocm_malloc(&halo_q2_arena,96u<<20)!=hipSuccess) {
        (void)hipGetLastError();return 0;
    }
    return 1;
}
void ds4_rocm_halo_q2_release() {
    if(halo_q2_arena)(void)hipFree(halo_q2_arena);
    halo_q2_arena=nullptr;
}
int ds4_rocm_halo_q2_prepared(half *down,const char *weights,const half *mid,
        const unsigned *offsets,const unsigned *pairs,const unsigned *counts,unsigned rows) {
    if(!halo_q2_arena)return 0;
    unsigned host_offsets[257];
    if(hipMemcpy(host_offsets,offsets,sizeof(host_offsets),hipMemcpyDeviceToHost)!=hipSuccess)return 0;
    // Validate every host descriptor before the first preparation launch.
    for(unsigned e=0;e<256;e++)if(counts[e]>=512u) {
        unsigned count=counts[e],first=host_offsets[e];
        if(count>rows || first>rows*6u || count>rows*6u-first ||
           host_offsets[e+1]!=first+count)return 0;
    }
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
    auto wh=(half*)halo_q2_arena;
    auto xh=(half*)((char*)halo_q2_arena+(16u<<20));
    for(unsigned e=0;e<256;e++)if(counts[e]>=512u) {
        unsigned count=counts[e],first=host_offsets[e];
        halo_q2_prepare<<<dim3(128,8),128>>>(wh,(const unsigned char*)weights+(uint64_t)e*2752512);
        halo_q2_gather<<<(count*1024u+255u)/256u,256>>>(xh,mid,pairs,first,count);
        if(hipGetLastError()!=hipSuccess)return 0;
        halo_q2_prepared_store_kernel<<<((count+255u)/256u)*32u,256>>>(
            nullptr,xh,wh,(int)count,4096,2048,pairs,first,down);
        if(hipGetLastError()!=hipSuccess)return 0;
    }
    return 1;
#else
    return 0;
#endif
}

// SPDX-License-Identifier: MIT
// Normal backend entry: frozen S9 producers, native MMQ quantizer, stream 0.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include "../../ds4_rocm_halo_moe.h"
#include "../../../ds4_rocm_memory.h"
#include "../../../cuda/mmq/ggml.h"
#include "fresh/iq2_feed.hpp"
void quantize_mmq_q8_1_cuda(const float*,const int32_t*,void*,ggml_type,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,hipStream_t);
namespace iq2_rf {
struct PairDesc {unsigned expert,first0,first1,zero;};
#define DESC(N) extern "C" __global__ void N(const unsigned*,const unsigned*,Desc*,unsigned*,PairDesc*,unsigned*)
DESC(halo_s9_public_descriptors);DESC(halo_s9_fresh_descriptors);
#undef DESC
#define WIDE(N) extern "C" __global__ void N(const unsigned char*,const unsigned char*,const Q8*,const unsigned*,const Desc*,const unsigned*,const float*,float*,uint16_t*,float,float*,float*)
WIDE(halo_s9_public_wide);WIDE(halo_s9_fresh_wide);
#undef WIDE
#define TWIN(N) extern "C" __global__ void N(const unsigned char*,const unsigned char*,const Q8*,const unsigned*,const PairDesc*,const unsigned*,const float*,float*,uint16_t*,float,float*,float*)
TWIN(halo_s9_public_twin);TWIN(halo_s9_fresh_twin);
#undef TWIN
}
namespace {
void *owned[5] = {};
constexpr uint64_t public_bytes[5]={9437184,14336,4,3072,4};
// The exact qualified 4K up lease includes original guards/alignment gaps.
constexpr uint64_t payload[5]={256,18875136,18889984,18890752,18894336};
constexpr uint64_t reservation=18894848;
bool range(ds4_halo_span a) {
    uintptr_t p=(uintptr_t)a.ptr;
    return p && a.bytes && a.bytes <= UINTPTR_MAX-p;
}
bool apart(ds4_halo_span a,ds4_halo_span b) {
    return range(a) && range(b) &&
        ((uintptr_t)a.ptr+a.bytes <= (uintptr_t)b.ptr ||
         (uintptr_t)b.ptr+b.bytes <= (uintptr_t)a.ptr);
}
}
void ds4_rocm_halo_s9_release() {
    for (auto &p:owned) {if(p)(void)hipFree(p);p=nullptr;}
}
int ds4_rocm_halo_s9_run(unsigned rows,const void *gw,const void *uw,
    const float *x,const unsigned *pairs,const unsigned *offsets,
    const float *weights,float *mid,uint16_t *mh,float clamp,
    void *up,uint64_t up_bytes,const ds4_halo_span *live,unsigned count) {
    if ((rows!=2048 && rows!=4096) || !live || !count) return 0;
    // The existing native tensor extents and sorted routing are checked by caller.
    for(unsigned i=0;i<count;i++) {
        if(!range(live[i]))return 0;
        for(unsigned j=0;j<i;j++)if(!apart(live[i],live[j]))return 0;
    }
    void *buffers[5] = {};
    if(rows==4096) {
        ds4_halo_span borrowed={up,up_bytes};
        if(!range(borrowed) || ((uintptr_t)up&255) || up_bytes<reservation)return 0;
        for(unsigned i=0;i<count;i++)if(!apart(borrowed,live[i]))return 0;
        for(unsigned i=0;i<5;i++)buffers[i]=(char*)up+payload[i];
    } else {
        for(unsigned i=0;i<5;i++) {
            if(!owned[i] && ds4_rocm_malloc(&owned[i],public_bytes[i]+512)!=hipSuccess) {
                ds4_rocm_halo_s9_release();(void)hipGetLastError();return 0;
            }
            buffers[i]=(char*)owned[i]+256;
            ds4_halo_span arena={buffers[i],public_bytes[i]};
            for(unsigned j=0;j<count;j++)if(!apart(arena,live[j]))return 0;
        }
    }
    using namespace iq2_rf;
    auto q=(Q8*)buffers[0];auto d=(Desc*)buffers[1];auto nt=(unsigned*)buffers[2];
    auto pd=(PairDesc*)buffers[3];auto np=(unsigned*)buffers[4];
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
    if(rows==2048)halo_s9_public_descriptors<<<1,1>>>(pairs,offsets,d,nt,pd,np);
    else halo_s9_fresh_descriptors<<<1,1>>>(pairs,offsets,d,nt,pd,np);
    if(hipGetLastError()!=hipSuccess)return -1;
    quantize_mmq_q8_1_cuda(x,nullptr,q,GGML_TYPE_IQ2_XXS,4096,4096,0,0,4096,rows,1,1,nullptr);
    if(hipGetLastError()!=hipSuccess)return -1;
    if(rows==2048) {
        halo_s9_public_wide<<<dim3(16,768),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,d,nt,weights,mid,mh,clamp,nullptr,nullptr);
        halo_s9_public_twin<<<dim3(32,192),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,pd,np,weights,mid,mh,clamp,nullptr,nullptr);
    } else {
        halo_s9_fresh_wide<<<dim3(16,768),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,d,nt,weights,mid,mh,clamp,nullptr,nullptr);
        halo_s9_fresh_twin<<<dim3(32,192),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,pd,np,weights,mid,mh,clamp,nullptr,nullptr);
    }
    return hipGetLastError()==hipSuccess?1:-1;
#else
    return 0;
#endif
}

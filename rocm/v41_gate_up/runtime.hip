// SPDX-License-Identifier: MIT
// Normal backend entry: co-computed gate/up producers, native MMQ quantizer, stream 0.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include "../ds4_rocm_v41_gate_up.h"
#include "../../ds4_rocm_memory.h"
#include "../../cuda/mmq/ggml.h"
#include "tiles/iq2_feed.hpp"
void quantize_mmq_q8_1_cuda(const float*,const int32_t*,void*,ggml_type,
    int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,hipStream_t);
namespace iq2_rf {
struct PairDesc {unsigned expert,first0,first1,zero;};
#define DESC(N) extern "C" __global__ void N(const unsigned*,const unsigned*,Desc*,unsigned*,PairDesc*,unsigned*)
DESC(ds4_v41_gate_up_descriptors);
#undef DESC
#define WIDE(N) extern "C" __global__ void N(const unsigned char*,const unsigned char*,const Q8*,const unsigned*,const Desc*,const unsigned*,const float*,float*,uint16_t*,float,float*,float*)
WIDE(ds4_v41_gate_up_wide);
#undef WIDE
#define TWIN(N) extern "C" __global__ void N(const unsigned char*,const unsigned char*,const Q8*,const unsigned*,const PairDesc*,const unsigned*,const float*,float*,uint16_t*,float,float*,float*)
TWIN(ds4_v41_gate_up_twin);
#undef TWIN
}
namespace {
void *owned[5] = {};
constexpr uint64_t bytes[5]={11796480,24576,4,3072,4};
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
void ds4_rocm_v41_gate_up_release() {
    for (auto &p:owned) {if(p)(void)hipFree(p);p=nullptr;}
}
int ds4_rocm_v41_gate_up(unsigned rows,const void *gw,const void *uw,
    const float *x,const unsigned *pairs,const unsigned *offsets,
    const float *weights,float *mid,uint16_t *mh,float clamp,
    void *up,uint64_t up_bytes,const ds4_halo_span *live,unsigned count) {
    if (rows!=2048 || !live || !count) return 0;
    // The existing native tensor extents and sorted routing are checked by caller.
    for(unsigned i=0;i<count;i++) {
        if(!range(live[i]))return 0;
        for(unsigned j=0;j<i;j++)if(!apart(live[i],live[j]))return 0;
    }
    void *buffers[5] = {};
    for(unsigned i=0;i<5;i++) {
        if(!owned[i] && ds4_rocm_malloc(&owned[i],bytes[i]+512)!=hipSuccess) {
            ds4_rocm_v41_gate_up_release();(void)hipGetLastError();return 0;
        }
        buffers[i]=(char*)owned[i]+256;
        ds4_halo_span arena={buffers[i],bytes[i]};
        for(unsigned j=0;j<count;j++)if(!apart(arena,live[j]))return 0;
    }
    using namespace iq2_rf;
    auto q=(Q8*)buffers[0];auto d=(Desc*)buffers[1];auto nt=(unsigned*)buffers[2];
    auto pd=(PairDesc*)buffers[3];auto np=(unsigned*)buffers[4];
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
    ds4_v41_gate_up_descriptors<<<1,1>>>(pairs,offsets,d,nt,pd,np);
    if(hipGetLastError()!=hipSuccess)return -1;
    quantize_mmq_q8_1_cuda(x,nullptr,q,GGML_TYPE_IQ2_XXS,5120,5120,0,0,5120,rows,1,1,nullptr);
    if(hipGetLastError()!=hipSuccess)return -1;
    ds4_v41_gate_up_wide<<<dim3(18,1536),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,d,nt,weights,mid,mh,clamp,nullptr,nullptr);
    if(hipGetLastError()!=hipSuccess)return -1;
    ds4_v41_gate_up_twin<<<dim3(36,192),256>>>((const unsigned char*)gw,(const unsigned char*)uw,q,pairs,pd,np,weights,mid,mh,clamp,nullptr,nullptr);
    return hipGetLastError()==hipSuccess?1:-1;
#else
    return 0;
#endif
}

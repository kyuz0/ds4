#define S9_NAME halo_s9_public_wide
// SPDX-License-Identifier: MIT
// S9: native-association G/U permutation and register epilogue; down unchanged.
#if defined(IQ2_RF_LOCAL_COMPAT)
__attribute__((device)) float max(float,float);
__attribute__((device)) double max(double,double);
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "iq2_feed.hpp"
#include "iq2_grid.hpp"
#include "../epilogue.hpp"
#ifndef S9_CAPTURE
#define S9_CAPTURE 0
#endif
#ifndef S9_NAME
#error S9_NAME required
#endif
#if defined(IQ2_RF_LOCAL_COMPAT)
// Local cross-build only: unavailable OCKL ID wrappers, not math emulation.
extern "C" __device__ __attribute__((const,always_inline)) inline size_t __ockl_get_group_id(unsigned d){
 return d==0?__builtin_amdgcn_workgroup_id_x():d==1?__builtin_amdgcn_workgroup_id_y():__builtin_amdgcn_workgroup_id_z();
}
extern "C" __device__ __attribute__((const,always_inline)) inline size_t __ockl_get_local_id(unsigned d){
 return d==0?__builtin_amdgcn_workitem_id_x():d==1?__builtin_amdgcn_workitem_id_y():__builtin_amdgcn_workitem_id_z();
}
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
namespace iq2_rf {
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
using F8=float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ float mul_rn(float a,float b){
 float r;asm("v_mul_f32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;
}
__device__ __forceinline__ float fma_rn(float a,float b,float c){
 float r;asm("v_fma_f32 %0, %1, %2, %3":"=v"(r):"v"(a),"v"(b),"v"(c));return r;
}
__device__ __forceinline__ float from_half(uint16_t bits){return __half2float(*reinterpret_cast<const __half*>(&bits));}
// One compressed K128 lookahead: 32 payload bytes plus the unchanged FP16 scale.
struct PackedPanel { unsigned grids[4], signs[4], scale; };
__device__ __forceinline__ PackedPanel load_compressed(const unsigned char* wbase,unsigned kp){
 const uint16_t* raw=(const uint16_t*)(wbase+(kp/2)*66);
 PackedPanel p; p.scale=raw[0];
 #pragma unroll
 for(unsigned kg=0;kg<4;kg++){
  const uint16_t* packet=raw+1+((kp&1)*4+kg)*4;
  p.grids[kg]=read32_b2(packet);p.signs[kg]=read32_b2(packet+2);
 }
 return p;
}
__device__ __forceinline__ void consume_compressed(PackedPanel& p){
 // No machine instruction. Prevent future unpack/scale reconstruction moving
 // across the consumption boundary; memory dependencies remain compiler-visible.
 asm volatile("" : "+v"(p.grids[0]),"+v"(p.grids[1]),"+v"(p.grids[2]),"+v"(p.grids[3]),
                    "+v"(p.signs[0]),"+v"(p.signs[1]),"+v"(p.signs[2]),"+v"(p.signs[3]),"+v"(p.scale) : : "memory");
}
// Two feature tiles share one gathered Q8 panel and one set of B fragments.
// Native M64 singleton/tail descriptors only. Qualified twin kernel unchanged.
struct WideShared {Q8 x[64];float dw[4][256];unsigned token[64];};
static_assert(sizeof(WideShared)==13568);
extern "C" __global__ __launch_bounds__(256) void S9_NAME(
 const unsigned char* gw,const unsigned char* uw,const Q8* q,
 const unsigned* pairs,const Desc* desc,const unsigned* nt,const float* weights,float* mid,uint16_t* mid_half,float clamp,float* gate_cap,float* up_cap){
 if(blockIdx.y>=*nt)return;
 __shared__ WideShared sh;
 unsigned tid=threadIdx.x,lane=tid&31,wave=tid>>5;
 unsigned leg=(lane&15)&1,row=lane&15;
 Desc ds=desc[blockIdx.y];unsigned n=ds.count,n0=blockIdx.x*128;
 if(tid<64)sh.token[tid]=tid<n?pairs[ds.first+tid]/6:0;
 for(unsigned j=n*36+tid;j<64*36;j+=256)((unsigned*)sh.x)[j]=0;
 __syncthreads();
 float acc[2][4][8]={};
 const unsigned char* wbase[2];PackedPanel pending[2];
 #pragma unroll
 for(unsigned t=0;t<2;t++){
  wbase[t]=(leg?uw:gw)+(size_t)ds.expert*2162688+(size_t)s9::feature(n0,wave,t,row>>1)*1056;
  pending[t]=load_compressed(wbase[t],0);
 }
 for(unsigned kp=0;kp<32;kp++){
  for(unsigned j=tid;j<n*36;j+=256){unsigned m=j/36,c=j%36;
   ((unsigned*)&sh.x[m])[c]=((const unsigned*)(q+(size_t)kp*2048+sh.token[m]))[c];
  }
  unsigned grids[2][4],signs[2][4];
  #pragma unroll
  for(unsigned t=0;t<2;t++){
   consume_compressed(pending[t]);float d=from_half(uint16_t(pending[t].scale));
   #pragma unroll
   for(unsigned kg=0;kg<4;kg++){
    grids[t][kg]=pending[t].grids[kg];signs[t][kg]=pending[t].signs[kg];
    if(lane<16)sh.dw[kg][t*128+wave*16+row]=mul_rn(mul_rn(d,float((signs[t][kg]>>27)|1)),0.125f);
   }
  }
  __syncthreads();
  #pragma unroll
  for(unsigned kg=0;kg<4;kg++){
   I4 a0[2],a1[2];
   #pragma unroll
   for(unsigned t=0;t<2;t++){
    unsigned h=(lane>>4)*2;
    Words v0=decode8(iq2_rf_grid[(grids[t][kg]>>(8*h))&255],signs[t][kg]>>(7*h));
    Words v1=decode8(iq2_rf_grid[(grids[t][kg]>>(8*(h+1)))&255],signs[t][kg]>>(7*(h+1)));
    I4 own={int(v0.a),int(v0.b),int(v1.a),int(v1.b)},other;
    #pragma unroll
    for(unsigned r=0;r<4;r++)other[r]=__shfl_xor(own[r],16,32);
    a0[t]=(lane&16)?other:own;a1[t]=(lane&16)?own:other;
   }
   // Both current weight tiles are decoded before either lookahead starts.
   __builtin_amdgcn_sched_barrier(0);
   if(kg==3 && kp+1<32){
    #pragma unroll
    for(unsigned t=0;t<2;t++)pending[t]=load_compressed(wbase[t],kp+1);
   }
   #pragma unroll
   for(unsigned mt=0;mt<4;mt++)if(mt*16<n){
    unsigned m=mt*16+row;
    I4 b0=*(const I4*)&sh.x[m].q[kg*8],b1=*(const I4*)&sh.x[m].q[kg*8+4];
    float dx=sh.x[m].d[kg];
    #pragma unroll
    for(unsigned t=0;t<2;t++){
     I8 z={};
     z=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a0[t],true,b0,z,true);
     z=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a1[t],true,b1,z,true);
     #pragma unroll
     for(unsigned l=0;l<8;l++){
      float dw=sh.dw[kg][t*128+wave*16+2*l+(lane>>4)];
      if(mt==0 && l>=2)
       acc[t][mt][l]=fma_rn(mul_rn(float(z[l]),dx),dw,acc[t][mt][l]);
      else acc[t][mt][l]=fma_rn(float(z[l]),mul_rn(dw,dx),acc[t][mt][l]);
     }
     __builtin_amdgcn_sched_barrier(0); // bound I32 lifetime, as qualified twin
    }
   }
  }
  __syncthreads();
 }
 // no CTA handoff: gate/up are opposite physical halves of one wave.
 #pragma unroll
 for(unsigned mt=0;mt<4;mt++)if(mt*16<n){
  unsigned m=mt*16+row;
  unsigned p=m<n?pairs[ds.first+m]:0;
  float rw=m<n?weights[p]:0.0f;
  #pragma unroll
  for(unsigned t=0;t<2;t++){
   #if S9_CAPTURE
   if(m<n){
    #pragma unroll
    for(unsigned c=0;c<8;c++){
     unsigned f=s9::feature(n0,wave,t,c);
     ((lane&16)?up_cap:gate_cap)[(size_t)p*2048+f]=acc[t][mt][c];
    }
   }
   #endif
   #pragma unroll
   for(unsigned c=0;c<4;c++){
    float lo=acc[t][mt][c],hi=acc[t][mt][c+4];
    float send=(lane&16)?lo:hi,local=(lane&16)?hi:lo;
    float remote=__shfl_xor(send,16,32);
    float g=(lane&16)?remote:local,u=(lane&16)?local:remote;
    if(m<n){
     float value=s9::epilogue(g,u,rw,clamp);
     size_t at=(size_t)p*2048+s9::feature(n0,wave,t,c+((lane>>4)*4));
     mid[at]=value;mid_half[at]=s9::half_bits(value);
    }
   }
  }
 }
}
} // namespace iq2_rf

#else
// Unreachable registration stub: host admission rejects non-gfx1151.
namespace iq2_rf {
extern "C" __global__ __launch_bounds__(256) void S9_NAME(
 const unsigned char* gw,const unsigned char* uw,const Q8* q,
 const unsigned* pairs,const Desc* desc,const unsigned* nt,const float* weights,float* mid,uint16_t* mid_half,float clamp,float* gate_cap,float* up_cap){}
}
#endif

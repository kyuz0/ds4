// SPDX-License-Identifier: MIT
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
namespace halo_output_a_uncached {
#include "halo/output_a_uncached/index.hpp"
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
// Match actual native ISA: round(dw_half * float(dot)), THEN fma(...,dx,acc).
// Do not rewrite as dot*(dw*dx), nor turn this into a half GEMM.
__device__ __forceinline__ float native_term(unsigned dw,int dot,float dx,float acc){
 float z=float(dot),p;
 asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]":"=v"(p):"v"(dw),"v"(z));
 asm("v_fmac_f32_e32 %0, %1, %2":"+v"(acc):"v"(p),"v"(dx));return acc;
}
__device__ __forceinline__ float add_native(float a,float b){float r;asm("v_add_f32_e32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;}
__device__ __forceinline__ void local_barrier(){
 __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
 __builtin_amdgcn_s_barrier();
 __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
}
// T2048, eight groups of K4096 -> rank1024. Original Q8_0 W, xq, xscale.
// CTA covers16 token x16 rank outputs; eight waves own four native lane slots.
__global__ __launch_bounds__(256)
static void halo_output_a_q8_slots(float* low,const unsigned char* w,const int8_t* xq,const float* xs){
 const unsigned lane=threadIdx.x&31,wave=threadIdx.x>>5,row=lane&15,half=lane>>4;
 const unsigned g=blockIdx.z,n0=blockIdx.x*16,t0=blockIdx.y*16;
 __shared__ float partial[32][256];
 // Four separate native lane accumulators; no reassociation across slots.
 float acc[4][8]={};
 #pragma unroll 1
 for(unsigned step=0;step<4;step++){
  #pragma unroll
  for(unsigned q=0;q<4;q++){
   const unsigned b=oa::block(wave,q,step);
   const auto* wp=w+size_t(g*1024+n0+row)*4352+b*34;
   const size_t xr=oa::xrow(t0+row,g)*128+b;
   // Each physical half-wave loads its own16 code bytes; exchange once.
   I4 aw,bx;__builtin_memcpy(&aw,wp+2+half*16,16);
   __builtin_memcpy(&bx,xq+xr*32+half*16,16);
   I4 ao,bo;
   #pragma unroll
   for(unsigned j=0;j<4;j++){ao[j]=__shfl_xor(aw[j],16,32);bo[j]=__shfl_xor(bx[j],16,32);}
   I4 a0=half?ao:aw,a1=half?aw:ao,b0=half?bo:bx,b1=half?bx:bo;
   I8 z={};
   z=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a0,true,b0,z,true);
   z=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a1,true,b1,z,true);
   unsigned dw;unsigned short h;__builtin_memcpy(&h,wp,2);dw=h;
   const float dx=xs[xr];
   #pragma unroll
   for(unsigned j=0;j<8;j++){
    unsigned scale=__shfl(dw,2*j+half,32);
    acc[q][j]=native_term(scale,z[j],dx,acc[q][j]);
   }
   __builtin_amdgcn_sched_barrier(0);
  }
 }
 #pragma unroll
 for(unsigned q=0;q<4;q++){
  #pragma unroll
  for(unsigned j=0;j<8;j++)partial[oa::slot(wave,q)][oa::scratch(oa::element(lane,j))]=acc[q][j];
 }
 local_barrier(); // all32 native slots published, exactly once per CTA
 float r[32];
 #pragma unroll
 for(unsigned s=0;s<32;s++)r[s]=partial[s][oa::scratch(threadIdx.x)];
 // Same lane0 dependency tree as native shfl_down offsets16,8,4,2,1.
 #pragma unroll
 for(unsigned d=16;d>0;d>>=1){
  #pragma unroll
  for(unsigned s=0;s<d;s++)r[s]=add_native(r[s],r[s+d]);
 }
 low[oa::output(t0+threadIdx.x/16,g,n0+threadIdx.x%16)]=r[0];
}

}
#else
namespace halo_output_a_uncached {
// Unreachable registration stub; admission requires gfx1151.
__global__ __launch_bounds__(256)
static void halo_output_a_q8_slots(float*,const unsigned char*,const int8_t*,const float*) {}
}
#endif

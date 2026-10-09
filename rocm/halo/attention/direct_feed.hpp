#pragma once
#include <hip/hip_fp16.h>
#include <stdint.h>
#define DQK_HD __host__ __device__ __forceinline__
namespace bdqk {
// rocWMMA load representation on gfx11: lane%16 owns a row (A) or column
// (B); lane/16 owns its lower/upper eight K components. Keep rocWMMA's normal
// subsequent Swap16/to_wmma_input transform, rather than loading 16 halves.
template<class Frag> DQK_HD void query_eight(Frag&a,const float*row,unsigned k,bool valid){
#if defined(__HIP_DEVICE_COMPILE__)
 static_assert(Frag::num_elements==8);
#endif
 for(unsigned i=0;i<8;i++)a.x[i]=__float2half(valid?row[k+i]:0.0f);
}
template<class Frag> DQK_HD void key_half_eight(Frag&b,const half*row,unsigned k,bool valid){
#if defined(__HIP_DEVICE_COMPILE__)
 static_assert(Frag::num_elements==8);
#endif
 if(valid)__builtin_memcpy(&b.x[0],row+k,16);
 else for(unsigned i=0;i<8;i++)b.x[i]=__float2half(0.0f);
}
template<bool VEC2,class Frag> DQK_HD void key_float_eight(Frag&b,const float*row,unsigned k,bool valid){
#if defined(__HIP_DEVICE_COMPILE__)
 static_assert(Frag::num_elements==8);
#endif
 if constexpr(VEC2){
  for(unsigned i=0;i<4;i++){
   float2 f=valid?*reinterpret_cast<const float2*>(row+k+2*i):make_float2(0.0f,0.0f);
   half2 h=__floats2half2_rn(f.x,f.y);__builtin_memcpy(&b.x[2*i],&h,4);
  }
 }else for(unsigned i=0;i<8;i++)b.x[i]=__float2half(valid?row[k+i]:0.0f);
}
DQK_HD unsigned row_from_lane(unsigned lane){return lane&15u;}
DQK_HD unsigned k_from_lane(unsigned lane){return 8u*(lane>>4u);}
}

#pragma once
// Preserve the current gfx1151 fused-MMQ instruction/rounding contract.
// The donor used the older standalone SwiGLU multiply order.
namespace s9 {
__device__ __forceinline__ float bits(unsigned u){return __uint_as_float(u);}
__device__ __forceinline__ float add(float a,float b){float r;asm("v_add_f32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;}
__device__ __forceinline__ float mul(float a,float b){float r;asm("v_mul_f32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;}
__device__ __forceinline__ float mx(float a,float b){float r;asm("v_max_f32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;}
__device__ __forceinline__ float mn(float a,float b){float r;asm("v_min_f32 %0, %1, %2":"=v"(r):"v"(a),"v"(b));return r;}
__device__ __forceinline__ float maxmin(float a,float b,float c){float r;asm("v_maxmin_f32 %0, %1, %2, %3":"=v"(r):"v"(a),"v"(b),"v"(c));return r;}
// Expose TRANS32 dependencies to LLVM while requesting the exact native opcodes.
__device__ __forceinline__ float ex2(float a){return __builtin_amdgcn_exp2f(a);}
__device__ __forceinline__ float rcp(float a){return __builtin_amdgcn_rcpf(a);}
__device__ __forceinline__ uint16_t half_bits(float a){unsigned r;asm("v_cvt_f16_f32 %0, %1":"=v"(r):"v"(a));return uint16_t(r);}
__device__ __forceinline__ unsigned feature(unsigned n0,unsigned wave,unsigned tile,unsigned c){
 unsigned b=wave&1;return n0+tile*64+(wave>>1)*16+(c<2?2*b+c:2+6*b+c);
}
__device__ __forceinline__ float epilogue(float gate,float up,float weight,float clamp){
 float g=__builtin_amdgcn_classf(gate,0x1f8)?gate:0.0f;
 float u=__builtin_amdgcn_classf(up,0x1f8)?up:0.0f;
 float cp=mx(clamp,clamp),cm=mx(-clamp,-clamp);
 float gc=mn(mx(g,g),cp),uc=maxmin(mx(u,u),cm,cp);
 bool clipped=clamp>bits(0x358637bd);
 g=clipped?gc:g;u=clipped?uc:u;
 bool adjusted=g>bits(0x42aeac50);
 float shifted=add(bits(0xc2800000),g);
 float eg=adjusted?shifted:g;
 // Current fused MMQ writeback rounds weight*up before multiplying gate.
 float numerator=mul(mul(weight,u),g);
 float e=ex2(mul(bits(0xbfb8aa3b),eg));
 float ec=mul(bits(0x114b4ea4),e);
 e=adjusted?ec:e;
 return mul(numerator,rcp(add(1.0f,e)));
}
}

#pragma once
#include "api.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
__device__ __forceinline__ float p2_fp32(float x) {
    // The native global FP32 stores were observable rounding boundaries.
    // Prevent contraction through these boundaries into RoPE/half stores.
    asm volatile("" : "+v"(x));return x;
}
__device__ static float rope_yarn_ramp_dev(float low,float high,int i0) {
    float y=((float)(i0/2)-low)/fmaxf(0.001f,high-low);
    return 1.0f-fminf(1.0f,fmaxf(0.0f,y));
}
__device__ static float2 p2_coeff(P2Rope r,uint32_t t,uint32_t pair) {
    // Keep the native run-time operand in pow/division expressions; replacing
    // it by a compile-time64 could select a different fast-math evaluation.
    const uint32_t n_rot=r.n_rot;
    const uint32_t i=pair*2;
    const float freq_base=r.freq_base,freq_scale=r.freq_scale,ext_factor=r.ext_factor;
    const float attn_factor=r.attn_factor,beta_fast=r.beta_fast,beta_slow=r.beta_slow;
    const uint32_t n_ctx_orig=r.n_ctx_orig,pos0=r.pos0,pos_stride=r.pos_stride;
    const int inverse=r.inverse;
    float corr0=0.0f,corr1=0.0f;
    if(ext_factor!=0.0f) {
        float denom=2.0f*logf(freq_base);
        corr0=floorf((float)n_rot*logf((float)n_ctx_orig/(beta_fast*2.0f*(float)M_PI))/denom);
        corr1=ceilf((float)n_rot*logf((float)n_ctx_orig/(beta_slow*2.0f*(float)M_PI))/denom);
        corr0=fmaxf(0.0f,corr0);corr1=fminf((float)(n_rot-1),corr1);
    }
    const float theta_scale=powf(freq_base,-2.0f/(float)n_rot);
    float theta_extrap=(float)(pos0+t*pos_stride)*powf(theta_scale,(float)pair);
    float theta_interp=freq_scale*theta_extrap;
    float theta=theta_interp,mscale=attn_factor;
    if(ext_factor!=0.0f) {
        float ramp_mix=rope_yarn_ramp_dev(corr0,corr1,(int)i)*ext_factor;
        theta=theta_interp*(1.0f-ramp_mix)+theta_extrap*ramp_mix;
        mscale*=1.0f+0.1f*logf(1.0f/freq_scale);
    }
    float c=cosf(theta)*mscale,s=sinf(theta)*mscale;
    if(inverse)s=-s;
    return make_float2(c,s);
}
__device__ static void p2_store4(half *packed,uint32_t t,uint32_t h,uint32_t d,float4 v,const float2 *cs) {
    float z[4]={p2_fp32(v.x),p2_fp32(v.y),p2_fp32(v.z),p2_fp32(v.w)};
    if(d>=448) for(int j=0;j<4;j+=2) {
        const float2 a=cs[(d+j-448)/2];const float x0=z[j],x1=z[j+1];
        z[j]=p2_fp32(x0*a.x-x1*a.y);z[j+1]=p2_fp32(x0*a.y+x1*a.x);
    }
    half *out=packed+((uint64_t)(h/8)*4096+t)*4096+(h%8)*512+d;
    for(int j=0;j<4;++j)out[j]=__float2half_rn(p2_fp32(z[j]));
}

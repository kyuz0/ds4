/* Halo resident-key score kernel. Adapted from the frozen 2026-09-26
 * experiment; provenance and qualification limits: docs/HALO_EVIDENCE.md.
 * Keep the native FP16 WMMA K order and head-by-head FP32 accumulation.
 * Other GPU architectures retain the original indexer implementation. */
#include "ds4_rocm_halo_shapes.h"

#if defined(__gfx1151__)
// gfx1151-only prototype. Operand words duplicate the *frozen rocWMMA*
// own-half + swap16 layout, including the opposite K8 order in upper lanes.
using I4 = int __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
union Words { I8 i; H16 h; };
__device__ __forceinline__ H16 distribute(I4 own) {
    Words out;
    #pragma unroll
    for (unsigned i=0;i<4;++i) {
        out.i[i]=own[i];
        out.i[i+4]=__builtin_amdgcn_ds_swizzle(own[i],0x401f);
    }
    return out.h;
}

__device__ __forceinline__ I4 convert8(const float* p, bool valid) {
    I4 v={};
    if(valid) {
        // Both vectors are 16-byte aligned. Keep a bounded vector load,
        // not eight separately predicated scalar loads across key rows.
        float4 a=*reinterpret_cast<const float4*>(p);
        float4 b=*reinterpret_cast<const float4*>(p+4);
        v[0]=__builtin_bit_cast(int,__floats2half2_rn(a.x,a.y));
        v[1]=__builtin_bit_cast(int,__floats2half2_rn(a.z,a.w));
        v[2]=__builtin_bit_cast(int,__floats2half2_rn(b.x,b.y));
        v[3]=__builtin_bit_cast(int,__floats2half2_rn(b.z,b.w));
    }
    return v;
}

#endif

__global__ static void halo_indexer_scores_resident_keys_kernel(
        float* scores,const float* q,const float* weights,const float* keys,
        uint32_t nc,uint32_t nt,uint32_t pos0,uint32_t nh,uint32_t hd,
        uint32_t ratio,float scale,int causal) {
#if defined(__gfx1151__)
    const unsigned tid=threadIdx.x,lane=tid&31,wave=tid>>5;
    const unsigned tc=blockIdx.x*128,tt=blockIdx.y*16;
    if(tid>=256 || hd!=128 || nh!=64 || (causal && !ratio))return;
    if(causal) {
        unsigned last=min(tt+16,nt),visible=last>tt?min((pos0+last)/ratio,nc):0;
        if(tc>=visible) {
            for(unsigned i=tid;i<2048;i+=256) {
                unsigned t=tt+i/128,c=tc+i%128;
                if(t<nt && c<nc)scores[(uint64_t)t*nc+c]=-INFINITY;
            }
            return;
        }
    }
    // Query panel remains shared by all eight column waves. No global temp.
    __shared__ __half aq[16*128];
    __shared__ float wh[16];
    // Exactly the same 128 keys as the reference, loaded once per CTA,
    // converted once and distributed ONCE, then retained across all 64 heads.
    H16 bk[8];
    const unsigned c=tc+wave*16+(lane&15), khalf=(lane>>4)*8;
    #pragma unroll
    for(unsigned k=0;k<8;++k) {
        const float* kp=c<nc?keys+(uint64_t)c*128+k*16+khalf:keys;
        bk[k]=distribute(convert8(kp,c<nc));
    }
    F8 total={};
    for(unsigned h=0;h<64;++h) {
        for(unsigned i=tid;i<2048;i+=256) {
            unsigned t=tt+i/128,d=i%128;
            aq[i]=__float2half(t<nt?q[((uint64_t)t*64+h)*128+d]:0.0f);
        }
        if(tid<16) wh[tid]=tt+tid<nt?weights[(uint64_t)(tt+tid)*64+h]:0.0f;
        __syncthreads();
        F8 dot={};
        #pragma unroll
        for(unsigned k=0;k<8;++k) {
            I4 own=*reinterpret_cast<const I4*>(aq+(lane&15)*128+k*16+khalf);
            dot=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(distribute(own),bk[k],dot);
        }
        // Native output: query=2*i+half-wave, key=lane%16. Each score
        // retains head order 0..63; no inter-head partial sums or reassociation.
        #pragma unroll
        for(unsigned i=0;i<8;++i) {
            unsigned r=2*i+(lane>>4);
            if(tt+r<nt && c<nc)total[i]=fmaf(fmaxf(dot[i],0.0f),wh[r],total[i]);
        }
        // Protect aq/wh readers before the next head overwrites either.
        __syncthreads();
    }
    #pragma unroll
    for(unsigned i=0;i<8;++i) {
        unsigned t=tt+2*i+(lane>>4);
        if(t<nt && c<nc) {
            float out=total[i]*scale;
            if(causal && c>=(pos0+t+1)/ratio)out=-INFINITY;
            scores[(uint64_t)t*nc+c]=out;
        }
    }
#endif
}

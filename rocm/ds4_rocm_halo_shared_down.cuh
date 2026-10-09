// SPDX-License-Identifier: MIT
// Native physical FMA roles and fixed K32 schedule, qualified for 2K/4K.
namespace halo_shared_down {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
using U4 = unsigned __attribute__((ext_vector_type(4)));
using U2 = unsigned __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
    // Keep native per-barrier cache invalidation. This candidate amortizes
    // the whole panel schedule; it does not separately tune cache policy.
    asm volatile("buffer_gl0_inv" ::: "memory");
}

// Match native EE8C..F0A0 per-output physical operand roles and half selectors.
// a0/a1 each hold adjacent weight features, b adjacent token activations.
// Native first-token FMAs use X first; second-token FMAs use W first.
__device__ __forceinline__ void mix8(float& c0,float& c1,float& c2,float& c3,
                                    float& c4,float& c5,float& c6,float& c7,
                                    unsigned a0,unsigned a1,unsigned b) {
    asm volatile(
        "v_fma_mix_f32 %1, %10, %8, %1 op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %0, %10, %8, %0 op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %4, %8, %10, %4 op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %5, %8, %10, %5 op_sel:[1,1,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %3, %10, %9, %3 op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %2, %10, %9, %2 op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %6, %9, %10, %6 op_sel:[0,1,0] op_sel_hi:[1,1,0]\n\t"
        "v_fma_mix_f32 %7, %9, %10, %7 op_sel:[1,1,0] op_sel_hi:[1,1,0]"
        : "+&v"(c0),"+&v"(c1),"+&v"(c2),"+&v"(c3),
          "+&v"(c4),"+&v"(c5),"+&v"(c6),"+&v"(c7)
        : "v"(a0),"v"(a1),"v"(b));
}
__device__ __forceinline__ float native_scale(float value,float alpha) {
    float r; asm("v_mul_f32_e32 %0, %1, %2" : "=v"(r):"s"(alpha),"v"(value));return r;
}

// Same explicit parameter offsets as the installed 100-byte native payload.
// C, unused batch strides and opaque fields retained; no C read for beta=0.
__global__ static __launch_bounds__(256)
void shared_down_k32(float* D,const float* C,const unsigned short* A,const unsigned short* B,
                     float alpha,float beta,unsigned strideD1,unsigned strideD2,
                     unsigned strideC1,unsigned strideC2,unsigned strideA1,unsigned strideA2,
                     unsigned strideB1,unsigned strideB2,unsigned M,unsigned N,
                     unsigned batch,unsigned K,unsigned stagger_mask,unsigned reserved92,unsigned reserved96) {
    (void)C;(void)beta;(void)strideD2;(void)strideC1;(void)strideC2;
    (void)strideA2;(void)strideB2;(void)M;(void)N;(void)batch;(void)K;(void)reserved92;(void)reserved96;
    __shared__ __align__(16) unsigned short wa[32][64];
    __shared__ __align__(16) unsigned short xb[32][32];
    const unsigned tid=threadIdx.x;
    // Exact WGM8 full-band mapping of native prologue; no edge shapes admitted.
    const unsigned logical=blockIdx.x+64*(blockIdx.y&7);
    const unsigned tile_m=logical>>3, tile_n=(blockIdx.y&~7u)+(logical&7);
    const unsigned begin=(tile_m&stagger_mask)*128;
    const unsigned mp=2*(tid&15),np=2*(tid>>4);
    const auto* aligned_A=static_cast<const unsigned short*>(__builtin_assume_aligned(A,16));
    const auto* aligned_B=static_cast<const unsigned short*>(__builtin_assume_aligned(B,16));
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    for(unsigned panel=0;panel<64;++panel) {
        const unsigned k=(begin+32*panel)&2047;
        const unsigned ar=tid>>2, ak=8*(tid&3), br=tid>>3, bk=4*(tid&7);
        U4 aw;U2 bx;
        __builtin_memcpy(&aw,aligned_A+(tile_m*64+ar)*strideA1+k+ak,sizeof aw);
        __builtin_memcpy(&bx,aligned_B+(tile_n*32+br)*strideB1+k+bk,sizeof bx);
        lds_barrier(); // retire all readers before reusing the single buffer
        #pragma unroll
        for(unsigned j=0;j<4;++j) {
            wa[ak+2*j][ar]=static_cast<unsigned short>(aw[j]);
            wa[ak+2*j+1][ar]=static_cast<unsigned short>(aw[j]>>16);
        }
        #pragma unroll
        for(unsigned j=0;j<2;++j) {
            xb[bk+2*j][br]=static_cast<unsigned short>(bx[j]);
            xb[bk+2*j+1][br]=static_cast<unsigned short>(bx[j]>>16);
        }
        lds_barrier();
        #pragma unroll
        for(unsigned sub=0;sub<4;++sub) {
            // Retain native eight-K local prefetch depth, independently of
            // the larger global/LDS panel. No second LDS buffer.
            unsigned a0[8],a1[8],b[8];
            #pragma unroll
            for(unsigned j=0;j<8;++j) {
                __builtin_memcpy(&a0[j],&wa[8*sub+j][mp],sizeof(unsigned));
                __builtin_memcpy(&a1[j],&wa[8*sub+j][mp+32],sizeof(unsigned));
                __builtin_memcpy(&b[j],&xb[8*sub+j][np],sizeof(unsigned));
            }
            // Compiler scheduling boundary only, no extra hardware barrier.
            asm volatile("" ::: "memory");
            #pragma unroll
            for(unsigned j=0;j<8;++j)
                mix8(c0,c1,c2,c3,c4,c5,c6,c7,a0[j],a1[j],b[j]);
        }
    }
    const unsigned m=tile_m*64+mp,n=tile_n*32+np;
    D[n*strideD1+m]=native_scale(c0,alpha);
    D[n*strideD1+m+1]=native_scale(c1,alpha);
    D[n*strideD1+m+32]=native_scale(c2,alpha);
    D[n*strideD1+m+33]=native_scale(c3,alpha);
    D[(n+1)*strideD1+m]=native_scale(c4,alpha);
    D[(n+1)*strideD1+m+1]=native_scale(c5,alpha);
    D[(n+1)*strideD1+m+32]=native_scale(c6,alpha);
    D[(n+1)*strideD1+m+33]=native_scale(c7,alpha);
}
#else
__global__ static void shared_down_k32(float*,const float*,const unsigned short*,const unsigned short*,float,float,
 unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned) {}
#endif
}

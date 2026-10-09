#pragma once
// Physical CTA256(token) x128(rank), eight waves arranged4x2. The preceding donor
// barrier finishes every input-LDS reader. Reuse the same LDS in two phases.
namespace output_a_c {
__device__ __forceinline__ void local_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup","local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup","local");
}
template<class Fragments>
__device__ __forceinline__ void store_half(half *out,Fragments& c,half *lds,
                                         int r0,int t0,int tid) {
    const int wt=(tid/32)/2, wr=(tid/32)%2, lane=tid%32;
    // 128 token rows with136 half rank slots:34816B, fits55296B input LDS.
    using V=unsigned __attribute__((ext_vector_type(4)));
    auto phase=[&]<int P>() {
        if(wt/2==P) {
            auto rr=[&]<std::size_t R>() {
                auto tt=[&]<std::size_t T>() {
                    auto element=[&]<std::size_t E>() {
                        const float value=c[R][T].get()[E];
                        const int r=wr*64+int(T)*16+(lane%16);
                        const int t=(wt%2)*64+int(R)*16+2*int(E)+(lane/16);
                        lds[t*136+r]=__float2half_rn(value);
                    };
                    [&]<std::size_t... E>(std::index_sequence<E...>){(element.template operator()<E>(),...);}(std::make_index_sequence<8>{});
                };
                [&]<std::size_t... T>(std::index_sequence<T...>){(tt.template operator()<T>(),...);}(std::make_index_sequence<4>{});
            };
            [&]<std::size_t... R>(std::index_sequence<R...>){(rr.template operator()<R>(),...);}(std::make_index_sequence<4>{});
        }
        local_barrier();
        auto flush=[&]<std::size_t I>() {
            const int q=tid+int(I)*256, t=q/16,r=(q%16)*8;
            V bits;
            __builtin_memcpy(&bits,lds+t*136+r,16);
            // out already points at this batch's rank0. Full tiles only.
            __builtin_memcpy(__builtin_assume_aligned(out+std::size_t(t0+P*128+t)*8192+r0+r,16),&bits,16);
        };
        [&]<std::size_t... I>(std::index_sequence<I...>){(flush.template operator()<I>(),...);}(std::make_index_sequence<8>{});
        if constexpr(P==0)local_barrier();
    };
    phase.template operator()<0>();phase.template operator()<1>();
}
}

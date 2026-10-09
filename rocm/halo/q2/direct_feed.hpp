#pragma once
namespace q2_direct_x {
// Exactly the eight-half rocWMMA load fragment, before its unchanged Swap16.
// Every valid row is 4096B aligned relative to the allocation; k0 is multiple16.
template<class Frag>
__device__ __forceinline__ void load_eight(Frag& a,const half* row,unsigned k0,bool valid){
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1151__)
    static_assert(Frag::num_elements==8);
    using Words=unsigned __attribute__((ext_vector_type(4)));
    Words bits{};
    if(valid) __builtin_memcpy(&bits,__builtin_assume_aligned(row+k0,16),16);
    __builtin_memcpy(&a.x[0],&bits,16);
#else
    (void)a;(void)row;(void)k0;(void)valid;
#endif
}
}

// SPDX-License-Identifier: MIT
// Frozen grouped projection: preserve heads-first WMMA and half epilogue.
#include <type_traits>
#include <utility>
#include <cstddef>
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
#include "halo/output_a/candidate/kernel.hpp"
template<unsigned Rows>
__global__ static __launch_bounds__(256) void halo_output_a_kernel(
        half *out, const half *w, const half *heads) {
    using namespace rocm_wmma_gemm;
    output_a_k64_core::gemm_impl<float,half,m_layout::row_major,m_layout::row_major,
        m_layout::col_major,4,2,4,4,4,1,8,256,1>(out,heads,w,Rows,1024,4096,7);
}
#else
template<unsigned Rows>
__global__ static void halo_output_a_kernel(half*,const half*,const half*) {}
#endif

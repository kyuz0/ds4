#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <type_traits>
#include "prepared/kernel_lds_scope.hpp"
using namespace halo_q2_wmma_gemm;
// C is nullptr in production; the qualified prepared-store body is unchanged.
extern "C" __global__ __launch_bounds__(256)
void halo_q2_prepared_store_kernel(float* C,const half* X,const half* W,int M,int N,int K,
                       const unsigned* pairs,unsigned first,half* down) {
    q2_prepared_store::gemm_impl<float,half,m_layout::row_major,
        m_layout::row_major,m_layout::col_major,4,2,4,4,4,1,8,256,1>
        (C,X,W,M,N,K,pairs,first,down);
}

#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// Unreachable registration stub; admission requires gfx1151.
extern "C" __global__ __launch_bounds__(256)
void halo_q2_prepared_store_kernel(float*,const half*,const half*,int,int,int,
        const unsigned*,unsigned,half*) {}
#endif

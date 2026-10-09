#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "convert.hpp"
#include "kernel/kernel_lds_scope.hpp"
extern "C" __global__ void halo_shared_gu_prepare_w(_Float16*out,const unsigned char*w,uint64_t n){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n)out[i]=dense_bridge::weight_half(w,i);
}
extern "C" __global__ void halo_shared_gu_prepare_x(_Float16*out,const float*x,uint64_t n){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n)out[i]=dense_bridge::activation_half(x[i]);
}
// Two physical contracts, ONE geometry. Never swap native operands merely
// because dot products are mathematically commutative.
using namespace halo_shared_gu_wmma_gemm;
extern "C" __global__ __launch_bounds__(256) void halo_shared_gu_kernel(
 float*y,const half*w,const half*x,int m,int n,int k){
 rocm_wmma_lds_scope::gemm_impl<float,half,m_layout::col_major,
 m_layout::row_major,m_layout::col_major,4,2,4,4,4,1,8,256,1>(y,w,x,m,n,k);
}

#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// Unreachable registration stubs; admission requires gfx1151.
extern "C" __global__ void halo_shared_gu_prepare_w(_Float16*,const unsigned char*,uint64_t) {}
extern "C" __global__ void halo_shared_gu_prepare_x(_Float16*,const float*,uint64_t) {}
extern "C" __global__ __launch_bounds__(256)
void halo_shared_gu_kernel(float*,const half*,const half*,int,int,int) {}
#endif

// Normal runtime preflight validates the fixed external layout and dead-heads lease.
int ds4_rocm_halo_shared_gu(float *out,const unsigned char *w,const float *x,void *scratch) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
    auto wh=(_Float16*)scratch;auto xh=wh+8388608u;
    halo_shared_gu_prepare_w<<<32768,256>>>(wh,w,8388608u);
    halo_shared_gu_prepare_x<<<32768,256>>>(xh,x,8388608u);
    if(hipGetLastError()!=hipSuccess)return 0;
    halo_shared_gu_kernel<<<128,256>>>(out,(const half*)wh,(const half*)xh,2048,2048,4096);
    return hipGetLastError()==hipSuccess;
#else
    return 0;
#endif
}

#pragma once
namespace halo_hc {

// Preserve the native global-FP32 boundary explicitly. The unforced expression
// selects v_fma_mixlo_f16 under the pinned fast-math flags; we do not assume its
// equivalence or call that selection a compiler defect.
__device__ inline __half hch_native_half_product(float x,float scale) {
#if defined(__HIP_DEVICE_COMPILE__)
    float product;
    asm("v_mul_f32 %0, %1, %2" : "=v"(product) : "v"(x), "v"(scale));
#else
    volatile float product=x*scale;
#endif
    return __float2half(product);
}

__global__ static void hch_rms_half_kernel(__half *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    __half *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = hch_native_half_product(xr[i], scale);
    }
}

}

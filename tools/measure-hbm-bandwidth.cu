// Read-only baseline: streaming reduction over device memory.
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void readsum(const float *__restrict__ x, size_t n, float *out) {
    float acc = 0; size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) acc += x[i];
    if (acc == 1e30f) *out = acc;   /* keep it live without a reduction tree */
}
int main(void) {
    const size_t n = 128ull * 1024 * 1024;   /* 512 MB of floats */
    float *x = nullptr, *o = nullptr;
    if (hipMalloc(&x, n * 4) != hipSuccess || hipMalloc(&o, 4) != hipSuccess) return 1;
    hipMemset(x, 1, n * 4); hipDeviceSynchronize();
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    readsum<<<1024, 256>>>(x, n, o); hipDeviceSynchronize();
    const int iters = 20;
    hipEventRecord(s);
    for (int i = 0; i < iters; i++) readsum<<<1024, 256>>>(x, n, o);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e);
    printf("read %.0f MBx%d in %.1f ms -> %.1f GB/s read-only\n",
           n * 4.0 / 1e6, iters, ms, (double)n * 4 * iters / (ms / 1000.0) / 1e9);
    return 0;
}

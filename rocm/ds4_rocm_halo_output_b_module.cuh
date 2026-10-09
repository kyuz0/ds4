// SPDX-License-Identifier: MIT
#include "halo/output_b/code.inc"
static hipModule_t halo_output_b_module;
static hipFunction_t halo_output_b_function;
static int halo_output_b_unavailable;
static int halo_output_b_prepare() {
    if (halo_output_b_function) return 1;
    if (halo_output_b_unavailable || !halo_output_b_code_size) return 0;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    hipError_t e = hipModuleLoadData(&module, halo_output_b_code);
    if (e == hipSuccess) e = hipModuleGetFunction(&function, module, "output_b_direct");
    if (e != hipSuccess) {
        if (module) (void)hipModuleUnload(module);
        (void)hipGetLastError();
        halo_output_b_unavailable = 1;
        return 0;
    }
    halo_output_b_module = module;
    halo_output_b_function = function;
    return 1;
}
static void halo_output_b_release() {
    if (halo_output_b_module) (void)hipModuleUnload(halo_output_b_module);
    halo_output_b_module = nullptr;
    halo_output_b_function = nullptr;
    halo_output_b_unavailable = 0;
}
static int halo_output_b_launch(float *y, const half *w, const half *x, unsigned rows) {
    int m = 4096, n = (int)rows, k = 8192;
    void *args[] = {&y,&w,&x,&m,&n,&k};
    return hipModuleLaunchKernel(halo_output_b_function,rows/8u,1,1,256,1,1,
                                0,nullptr,args,nullptr) == hipSuccess;
}

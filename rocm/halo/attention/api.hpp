#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
struct P2Rope {
    uint32_t pos0,pos_stride,n_ctx_orig;
    int inverse;
    float freq_base,freq_scale,ext_factor,attn_factor,beta_fast,beta_slow;
    uint32_t n_rot;
};
static_assert(sizeof(P2Rope)==44 && alignof(P2Rope)==4,"Frozen kernel argument layout");

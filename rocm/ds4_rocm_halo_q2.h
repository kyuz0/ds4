// SPDX-License-Identifier: MIT
#pragma once
#include <hip/hip_fp16.h>
int ds4_rocm_halo_q2_prepare();
void ds4_rocm_halo_q2_release();
int ds4_rocm_halo_q2_prepared(half *down,const char *weights,const half *mid,
    const unsigned *offsets,const unsigned *pairs,const unsigned *counts,unsigned rows);

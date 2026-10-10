// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include "ds4_rocm_halo_moe.h"
// 0 = preflight fallback, 1 = completed enqueue, -1 = launch error (no replay).
int ds4_rocm_v41_gate_up(unsigned rows, const void *gw, const void *uw,
    const float *x, const unsigned *pairs, const unsigned *offsets,
    const float *weights, float *mid, uint16_t *mh, float clamp,
    void *up, uint64_t up_bytes, const ds4_halo_span *live, unsigned count);
void ds4_rocm_v41_gate_up_release();

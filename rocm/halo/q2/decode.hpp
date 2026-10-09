#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>
__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_pack_half2_bits(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

template <int BN, int BK>
__device__ __forceinline__ static void q2_K_dequant_pair_tile_half_rowwise_staged(
        half *shB0,
        half *shB1,
        const uint32_t *raw_rows,
        uint32_t k0,
        uint32_t tid) {
    const uint32_t g = (k0 & 255u) >> 4u;
    const uint32_t within = g & 7u;
    const uint32_t qbase = (g >> 3u) * 32u + (within & 1u) * 16u;
    const uint32_t shift = (within >> 1u) * 2u;
    constexpr uint32_t KG = 4u;
    constexpr uint32_t RAW_DWORDS = 84u / sizeof(uint32_t);
    constexpr uint32_t UNITS_PER_TILE = (uint32_t)(BN * (BK / KG));
    for (uint32_t j = tid; j < 2u * UNITS_PER_TILE; j += blockDim.x) {
        const uint32_t tile = j / UNITS_PER_TILE;
        const uint32_t rem = j - tile * UNITS_PER_TILE;
        const uint32_t nn = rem / (uint32_t)(BK / KG);
        const uint32_t kk0 = (rem - nn * (uint32_t)(BK / KG)) * KG;
        const uint32_t row = tile * (uint32_t)BN + nn;
        const unsigned char *blk =
                reinterpret_cast<const unsigned char *>(raw_rows + row * RAW_DWORDS);
        const uint32_t dm_bits = *reinterpret_cast<const uint32_t *>(blk + 80u);
        const float d = dev_f16_to_f32((uint16_t)dm_bits);
        const float dm = dev_f16_to_f32((uint16_t)(dm_bits >> 16u));
        const float s = (float)(blk[g] & 0x0fu);
        const float m = (float)(blk[g] >> 4u);
        const uint32_t qbits = *reinterpret_cast<const uint32_t *>(blk + 16u + qbase + kk0);
        const uint32_t q0 = (qbits >> shift) & 3u;
        const uint32_t q1 = (qbits >> (8u + shift)) & 3u;
        const uint32_t q2 = (qbits >> (16u + shift)) & 3u;
        const uint32_t q3 = (qbits >> (24u + shift)) & 3u;
        const float ds = d * s;
        const float dmm = dm * m;
        half *shB = tile == 0u ? shB0 : shB1;
        half *dst = shB + nn * (uint32_t)BK + kk0;
        *reinterpret_cast<uint32_t *>(dst) =
                dev_pack_half2_bits(ds * (float)q0 - dmm, ds * (float)q1 - dmm);
        *reinterpret_cast<uint32_t *>(dst + 2u) =
                dev_pack_half2_bits(ds * (float)q2 - dmm, ds * (float)q3 - dmm);
    }
}

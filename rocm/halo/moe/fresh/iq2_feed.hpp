// SPDX-License-Identifier: MIT
// Native IQ2_XXS signed-grid representation; adapted from pinned DS4/ggml.
#pragma once
#include <stdint.h>
#ifdef __HIPCC__
#define IQ2_RF_HD __host__ __device__ __forceinline__
#else
#define IQ2_RF_HD inline
#endif
namespace iq2_rf {
struct alignas(16) Q8 { float d[4]; int q[32]; };
struct Desc { unsigned expert, first, count, rank64; };
constexpr unsigned kTokens=4096,kPairs=24576,kK=4096,kN=2048,kExperts=256,kDescCap=896;
static_assert(sizeof(Q8)==144 && sizeof(Desc)==16);
struct Words { uint32_t a,b; };
IQ2_RF_HD uint32_t read32_b2(const uint16_t* p) { return uint32_t(p[0])|(uint32_t(p[1])<<16); }
IQ2_RF_HD unsigned parity7(unsigned c) {
    c^=c>>4;c^=c>>2;c^=c>>1;return c&1;
}
IQ2_RF_HD unsigned sign8(unsigned code) {code&=127;return code|(parity7(code)<<7);}
// Exact packed helpers from DS4's qualified ds4_rocm.h, renamed locally.
IQ2_RF_HD uint32_t cmpne4(uint32_t a,uint32_t b){
    uint32_t d=a^b;d|=d>>1;d|=d>>2;d|=d>>4;return (d&0x01010101u)*255u;
}
IQ2_RF_HD uint32_t sub4(uint32_t a,uint32_t b){
    return ((a|0x80808080u)-(b&0x7f7f7f7fu))^((a^~b)&0x80808080u);
}
IQ2_RF_HD Words decode8(uint64_t grid,unsigned code) {
    unsigned s=sign8(code);
    unsigned signs=s*0x01010101u;
    unsigned m0=cmpne4(signs&0x08040201u,0),m1=cmpne4(signs&0x80402010u,0);
    return {sub4(uint32_t(grid)^m0,m0),sub4(uint32_t(grid>>32)^m1,m1)};
}
}

/* Frozen K16-panel Q8 kernels, 2026-09-24. See docs/HALO_EVIDENCE.md.
 * FP16 conversion, WMMA operand roles, K order and FP32 stores are retained.
 * Only gfx1151 device code contains these intrinsics. */
namespace halo_q8 {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
namespace d_q8 {
using H16 = _Float16 __attribute__((ext_vector_type(16)));
static_assert(sizeof(H16) == 32 && alignof(H16) == 32);
constexpr unsigned panel_halves = 128 * 32;
__device__ __forceinline__ constexpr unsigned offset(unsigned row, unsigned part) {
    return (part >> 4) * (128 * 16) + row * 16;
}
__device__ __forceinline__ void store(_Float16* panel, unsigned row, unsigned part, H16 v) {
    *reinterpret_cast<H16*>(panel + offset(row, part)) = v;
}
__device__ __forceinline__ H16 load(const _Float16* panel, unsigned row, unsigned part) {
    return *reinterpret_cast<const H16*>(panel + offset(row, part));
}
}
using H16 = d_q8::H16;
using F8 = float __attribute__((ext_vector_type(8)));
using I16 = int8_t __attribute__((ext_vector_type(16)));

__device__ __forceinline__ void d_local_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__global__ void d_convert_x(const float *x, _Float16 *h, uint64_t count) {
    uint64_t j = 2 * (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j < count) {
        float2 v = *(const float2 *)(x + j);
        *(half2 *)(h + j) = __floats2half2_rn(v.x, v.y);
    }
}

__launch_bounds__(256, 1)
__global__ void d_cooperative_k16(float *out, const unsigned char *w,
                             const _Float16 *x, uint32_t tokens,
                             uint32_t k, uint32_t features, uint64_t row_bytes) {
    __shared__ __align__(32) _Float16 lx[128 * 32];
    __shared__ __align__(32) _Float16 lw[128 * 32];
    const uint32_t tid = threadIdx.x, lane = tid & 31, lane16 = lane & 15;
    const uint32_t wave = tid >> 5;
    const uint32_t bt = blockIdx.y * 128, bn = blockIdx.x * 128;
    const uint32_t wt = (wave >> 2) * 64, wn = (wave & 3) * 32;
    const uint32_t row = tid >> 1, part = (tid & 1) * 16;
    F8 acc[4][2] = {};
    for (uint32_t bi = 0; bi < k / 32; ++bi) {
        H16 xv = {};
        if (bt + row < tokens)
            xv = *(const H16 *)(x + uint64_t(bt + row) * k + bi * 32 + part);
        d_q8::store(lx, row, part, xv);
        H16 wv = {};
        if (bn + row < features) {
            const unsigned char *bp = w + uint64_t(bn + row) * row_bytes + bi * 34;
            _Float16 scale;
            I16 q;
            __builtin_memcpy(&scale, bp, 2);
            __builtin_memcpy(&q, bp + 2 + part, 16);
#pragma unroll
            for (uint32_t i = 0; i < 16; ++i)
                wv[i] = scale * (_Float16)(float)(int)q[i];
        }
        d_q8::store(lw, row, part, wv);
        d_local_barrier();
#pragma unroll
        for (uint32_t kh = 0; kh < 2; ++kh) {
            H16 b[2];
#pragma unroll
            for (uint32_t ns = 0; ns < 2; ++ns)
                b[ns] = d_q8::load(lw, wn + ns * 16 + lane16, kh * 16);
#pragma unroll
            for (uint32_t ts = 0; ts < 4; ++ts) {
                H16 a = d_q8::load(lx, wt + ts * 16 + lane16, kh * 16);
#pragma unroll
                for (uint32_t ns = 0; ns < 2; ++ns)
                    acc[ts][ns] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[ns], acc[ts][ns]);
            }
        }
        d_local_barrier();
    }
#pragma unroll
    for (uint32_t ts = 0; ts < 4; ++ts) {
#pragma unroll
        for (uint32_t ns = 0; ns < 2; ++ns) {
            uint32_t n = bn + wn + ns * 16 + lane16;
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                uint32_t t = bt + wt + ts * 16 + 2 * i + (lane >> 4);
                if (t < tokens && n < features) out[uint64_t(t) * features + n] = acc[ts][ns][i];
            }
        }
    }
}

#else
__global__ void d_convert_x(const float*, _Float16*, uint64_t) {}
__global__ void d_cooperative_k16(float*, const unsigned char*, const _Float16*,
        uint32_t, uint32_t, uint32_t, uint64_t) {}
#endif
}

namespace halo_qa {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx1151__)
namespace qa_lds {
using H16=_Float16 __attribute__((ext_vector_type(16)));
static_assert(sizeof(H16)==32&&alignof(H16)==32);
__device__ __forceinline__ constexpr unsigned offset(unsigned logical_half) {
    const unsigned row=logical_half>>5, k=logical_half&31;
    return (k>>4)*1024+row*16+(k&15);
}
__device__ __forceinline__ void store_pair(_Float16* panel,unsigned logical_half,uint32_t bits) {
    __builtin_memcpy(panel+offset(logical_half),&bits,4);
}
__device__ __forceinline__ H16 load(const _Float16* panel,unsigned row,unsigned half) {
    return *reinterpret_cast<const H16*>(panel+half*1024+row*16);
}
}
typedef _Float16 __attribute__((ext_vector_type(16))) ds4_q8_half16_t;
typedef float    __attribute__((ext_vector_type(8)))  ds4_q8_float8_t;

/* Configurable wave-row Q8_0 batched GEMM experiment for gfx1151.
 * This is the hipfire/llama.cpp-style MMQ shape adapted to DS4's existing
 * F32 activation buffers: each block stages a 64-token x 32-K activation tile
 * into LDS as f16, while each wave owns 16 output rows and computes four
 * 16-token WMMA columns.  It is opt-in from host code because it only wins once
 * the token batch is large enough to amortize the bigger tile. */
template <uint32_t M_TILE, uint32_t WARPS>
__launch_bounds__(WARPS * 32u, 1)
__global__ static void native_qa_k16_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    constexpr uint32_t N_TILE = 64u;
    constexpr uint32_t K_TILE = 32u;
    constexpr uint32_t M_PER_WARP = M_TILE / WARPS;
    constexpr uint32_t N_TILES_PER_WARP = N_TILE / 16u;

    const uint32_t block_m = (uint32_t)blockIdx.x * M_TILE;
    const uint32_t block_n = (uint32_t)blockIdx.y * N_TILE;
    if (block_m >= out_dim || block_n >= n_tokens) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t warp_m = block_m + warp_id * M_PER_WARP;
    const uint32_t my_row = warp_m + lane16;
    const uint32_t safe_row = my_row < out_dim ? my_row : (out_dim - 1u);
    const unsigned char *row_base = w + (uint64_t)safe_row * row_bytes;
    const uint32_t n_blocks = in_dim >> 5u;

    ds4_q8_float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    ds4_q8_float8_t acc1 = acc0;
    ds4_q8_float8_t acc2 = acc0;
    ds4_q8_float8_t acc3 = acc0;

    __shared__ __align__(32) _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid * 2u; j < N_TILE * K_TILE; j += blockDim.x * 2u) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            half2 xv = __floats2half2_rn(0.0f, 0.0f);
            if (tok < n_tokens) {
                const float2 f = *(const float2 *)(x + (uint64_t)tok * in_dim + bi * 32u + kk);
                xv = __floats2half2_rn(f.x, f.y);
            }
            uint32_t xv_bits;
            __builtin_memcpy(&xv_bits, &xv, 4);
            qa_lds::store_pair(lds_x, j, xv_bits);
        }
        __syncthreads();

        const unsigned char *bp = row_base + (uint64_t)bi * 34u;
        _Float16 sc;
        {
            uint16_t s_bits;
            __builtin_memcpy(&s_bits, bp, 2);
            __builtin_memcpy(&sc, &s_bits, 2);
        }

        const int8_t *w0 = (const int8_t *)(bp + 2u);
        const int8_t *w1 = (const int8_t *)(bp + 18u);
        ds4_q8_half16_t a0;
        ds4_q8_half16_t a1;
#pragma unroll
        for (uint32_t i = 0; i < 16u; i++) {
            a0[i] = sc * (_Float16)(float)(int)w0[i];
            a1[i] = sc * (_Float16)(float)(int)w1[i];
        }

#pragma unroll
        for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
            const uint32_t nt = ntile * 16u + lane16;
            const ds4_q8_half16_t b0 = qa_lds::load(lds_x, nt, 0);
            const ds4_q8_half16_t b1 = qa_lds::load(lds_x, nt, 1);
            if (ntile == 0u) {
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc0);
                acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc0);
            } else if (ntile == 1u) {
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc1);
                acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc1);
            } else if (ntile == 2u) {
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc2);
                acc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc2);
            } else {
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc3);
                acc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc3);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t ntile = 0; ntile < N_TILES_PER_WARP; ntile++) {
        const uint32_t tok = block_n + ntile * 16u + lane16;
        if (tok >= n_tokens) continue;
        ds4_q8_float8_t acc = ntile == 0u ? acc0 : (ntile == 1u ? acc1 : (ntile == 2u ? acc2 : acc3));
#pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t row = warp_m + 2u * j + (lane >> 4u);
            if (row < out_dim) out[(uint64_t)tok * out_dim + row] = acc[j];
        }
    }
}


#else
template <uint32_t M_TILE, uint32_t WARPS>
__global__ void native_qa_k16_kernel(float*, const unsigned char*, const float*,
        uint32_t, uint32_t, uint32_t, uint64_t) {}
#endif
}

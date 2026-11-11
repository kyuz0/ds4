/* DeepSeek V4.1 baseline primitives. Float-addressable storage preserves the released BF16/FP8/FP4 graph boundaries. */
#ifdef __HIP_PLATFORM_AMD__

/* AMDGPU ignores float_control(precise), and HIP's math wrappers select native trig while preprocessing -ffast-math. Use explicit OCML operations and division instructions so the existing translation unit keeps its tuned arithmetic. */
__device__ static float v41_add(float x, float y) { return __ocml_add_rte_f32(x, y); }
__device__ static float v41_sub(float x, float y) { return __ocml_sub_rte_f32(x, y); }
__device__ static float v41_mul(float x, float y) { return __ocml_mul_rte_f32(x, y); }
__device__ static float v41_div(float x, float y) {
    /* ROCm 10 declares but does not define __ocml_div_rte_f32. A local reciprocal(off) pragma still leaves afn, which AMDGPU lowers to an approximate reciprocal. Encode the backend's full F32 division refinement, including temporary denorm preservation, as one indivisible block. */
    float q, d, n, r, e;
    uint32_t mode;
#define DS4_V41_DIV_ASM(VCC) asm volatile( \
        "v_div_scale_f32 %1, " VCC ", %7, %7, %6\n\t" \
        "v_div_scale_f32 %2, " VCC ", %6, %7, %6\n\t" \
        "v_rcp_f32 %3, %1\n\t" \
        "s_getreg_b32 %5, hwreg(HW_REG_MODE, 4, 2)\n\t" \
        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 3\n\t" \
        "v_fma_f32 %4, -%1, %3, 1.0\n\t" \
        "v_fma_f32 %3, %4, %3, %3\n\t" \
        "v_mul_f32 %0, %2, %3\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "v_fma_f32 %0, %4, %3, %0\n\t" \
        "v_fma_f32 %4, -%1, %0, %2\n\t" \
        "s_setreg_b32 hwreg(HW_REG_MODE, 4, 2), %5\n\t" \
        "v_div_fmas_f32 %0, %4, %3, %0\n\t" \
        "v_div_fixup_f32 %0, %0, %7, %6\n\t" \
        : "=&v"(q), "=&v"(d), "=&v"(n), "=&v"(r), "=&v"(e), "=&s"(mode) \
        : "v"(x), "v"(y) : "vcc", "memory")
    /* The HIP compiler need not define a wave-size macro. Its target builtin folds before assembly, selecting the valid VCC operand for the actual wave mode. */
#if defined(__AMDGCN__)
    if (__builtin_amdgcn_wavefrontsize() == 64) DS4_V41_DIV_ASM("vcc");
    else
#endif
        DS4_V41_DIV_ASM("vcc_lo");
#undef DS4_V41_DIV_ASM
    return q;
}

__device__ static float v41_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return __uint_as_float(bits & 0xffff0000u);
}

__device__ static float v41_pow2_ceil(float x) {
    const uint32_t bits = __float_as_uint(x);
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

__device__ static float v41_sum32(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x = v41_add(x, __shfl_down(x, delta, 32));
    return __shfl(x, 0, 32);
}

__global__ static void v41_bf16_kernel(float *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = v41_bf16(x[i]);
}

__global__ static void v41_quantize_kernel(float *x, uint32_t format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    const uint32_t lane = threadIdx.x;
    const uint64_t i = (uint64_t)blockIdx.x * block + lane;
    const float value = lane < block ? v41_bf16(x[i]) : 0.0f;
    float amax = fabsf(value);
    for (int delta = 16; delta; delta >>= 1)
        amax = fmaxf(amax, __shfl_down(amax, delta, 32));
    amax = __shfl(amax, 0, 32);
    float result;
    if (format == DS4_V41_FP8_E8M0) {
        const float scale = v41_pow2_ceil(v41_mul(fmaxf(amax, 1.0e-4f), 1.0f / 448.0f));
        result = v41_mul(dsv4_e4m3fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    } else {
        const float scale = format == DS4_V41_FP4_E4M3 ?
            dsv4_e4m3fn_dequant_dev(v41_div(fmaxf(amax, 0.01171875f), 6.0f)) :
            v41_pow2_ceil(v41_mul(fmaxf(amax, 7.052966104933725e-38f), 1.0f / 6.0f));
        result = v41_mul(dsv4_e2m1fn_dequant_dev(v41_div(fabsf(value), scale)), scale);
    }
    if (lane < block) {
        /* Preserve signed zero even though the surrounding translation unit permits -fno-signed-zeros. */
        const uint32_t bits = (__float_as_uint(v41_bf16(result)) & 0x7fffffffu) | (__float_as_uint(value) & 0x80000000u);
        x[i] = __uint_as_float(bits);
    }
}

extern "C" int ds4_gpu_dsv41_quantize(ds4_gpu_tensor *x, uint32_t width, uint32_t rows,
                                      ds4_v41_activation_format format) {
    const uint32_t block = format == DS4_V41_FP4_E4M3 ? 16u : 32u;
    if (!width || !rows || format < DS4_V41_BF16 || format > DS4_V41_FP4_E4M3 ||
        (format != DS4_V41_BF16 && width % block) ||
        !cuda_tensor_has_elems2(x, width, rows, sizeof(float))) return 0;
    if (format == DS4_V41_BF16) {
        const uint64_t count = (uint64_t)width * rows;
        if ((count + 255u) / 256u > UINT32_MAX) return 0;
        v41_bf16_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>((float *)x->ptr, count);
    } else {
        const uint64_t blocks = (uint64_t)(width / block) * rows;
        if (blocks > UINT32_MAX) return 0;
        v41_quantize_kernel<<<(unsigned)blocks, 32>>>((float *)x->ptr, format);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 activation quantization");
}

/* One launch for [bf16(x) in place] -> rms_norm(x) * w -> bf16(out): the same
 * per-thread element order and reduction as rms_norm_batched_kernel, with the
 * v41_bf16 roundings of the separate v41_bf16_kernel passes applied in place. */
template <bool ROUND_IN>
__global__ __launch_bounds__(256) static void v41_norm_bf16_kernel(float *out, float *x,
        const float *w, uint32_t n, float eps) {
    float sum = 0.0f;
    uint32_t i = threadIdx.x;
    for (; i + 7u * 256u < n; i += 8u * 256u) {
        float v[8];
#pragma unroll
        for (uint32_t k = 0; k < 8u; ++k) {
            v[k] = x[i + k * 256u];
            if (ROUND_IN) { v[k] = v41_bf16(v[k]); x[i + k * 256u] = v[k]; }
        }
#pragma unroll
        for (uint32_t k = 0; k < 8u; ++k) sum += v[k] * v[k];
    }
    for (; i < n; i += 256u) {
        float v = x[i];
        if (ROUND_IN) { v = v41_bf16(v); x[i] = v; }
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (i = threadIdx.x; i < n; i += 256u) out[i] = v41_bf16(x[i] * scale * w[i]);
}

extern "C" int ds4_gpu_dsv41_norm_bf16(ds4_gpu_tensor *out, ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n, float eps, int round_in) {
    uint64_t weight_bytes = 0;
    if (!model_map || !n || !cuda_u64_mul_checked(n, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(x, n)) return 0;
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         weight_bytes, "rms_weight");
    if (!w) return 0;
    if (round_in) v41_norm_bf16_kernel<true><<<1, 256>>>((float *)out->ptr, (float *)x->ptr, w, n, eps);
    else v41_norm_bf16_kernel<false><<<1, 256>>>((float *)out->ptr, (float *)x->ptr, w, n, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 fused norm/bf16");
}

/* One launch for: hc split/Sinkhorn (thread 0, written to split_out), the HC
 * weighted sum x[d] = sum_h residual[h][d] * w[h], bf16(x) in place, and
 * rms_norm(x) * nw -> bf16. Per element and per reduction the arithmetic and
 * order equal hc_split_sinkhorn_kernel + hc_weighted_sum_kernel +
 * v41_norm_bf16_kernel<true> (256 norm partials over i = t + 256k, same tree). */
__global__ __launch_bounds__(1024) static void v41_hc_tail_kernel(
        float *split_out, const float *mix, const float *hscale, const float *hbase,
        uint32_t sinkhorn_iters, float hc_eps, const float *w, const float *residual,
        float *x, float *norm_out, const float *nw, float eps) {
    constexpr uint32_t N = 5120u;
    __shared__ float xs[N];
    __shared__ float partial[256];
    const uint32_t tid = threadIdx.x;
    if (tid == 0u) hc4_split_one(split_out, mix, hscale, hbase, sinkhorn_iters, hc_eps);
    for (uint32_t d = tid; d < N; d += 1024u) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4u; h++) acc += residual[(uint64_t)h * N + d] * w[h];
        const float v = v41_bf16(acc);
        x[d] = v;
        xs[d] = v;
    }
    __syncthreads();
    if (tid < 256u) {
        float sum = 0.0f;
        for (uint32_t i = tid; i < N; i += 256u) {
            const float v = xs[i];
            sum += v * v;
        }
        partial[tid] = sum;
    }
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)N + eps);
    for (uint32_t i = tid; i < N; i += 1024u) norm_out[i] = v41_bf16(xs[i] * scale * nw[i]);
}

/* The weighted-sum and norm part of v41_hc_tail_kernel for a batch of rows:
 * x[d] = bf16(sum_h residual[h][d] * w[h]), norm_out = bf16(rms_norm(x) * nw),
 * one block per row. Same per-element arithmetic and reduction tree as
 * hc_weighted_sum_kernel + v41_bf16_kernel + rms_norm_batched_kernel<true> +
 * v41_bf16_kernel. */
__global__ __launch_bounds__(1024) static void v41_hc_wsum_norm_rows_kernel(
        const float *w, uint32_t w_stride, const float *residual, float *x,
        float *norm_out, const float *nw, float eps) {
    constexpr uint32_t N = 5120u;
    __shared__ float xs[N];
    __shared__ float partial[256];
    const uint32_t tid = threadIdx.x;
    const uint64_t row = blockIdx.x;
    const float *wr = w + row * w_stride;
    const float *res = residual + row * 4u * N;
    float *xr = x + row * N;
    float *no = norm_out + row * N;
    for (uint32_t d = tid; d < N; d += 1024u) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4u; h++) acc += res[(uint64_t)h * N + d] * wr[h];
        const float v = v41_bf16(acc);
        xr[d] = v;
        xs[d] = v;
    }
    __syncthreads();
    if (tid < 256u) {
        float sum = 0.0f;
        for (uint32_t i = tid; i < N; i += 256u) {
            const float v = xs[i];
            sum += v * v;
        }
        partial[tid] = sum;
    }
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)N + eps);
    for (uint32_t i = tid; i < N; i += 1024u) no[i] = v41_bf16(xs[i] * scale * nw[i]);
}

extern "C" int ds4_gpu_dsv41_hc_wsum_norm_rows(ds4_gpu_tensor *x, ds4_gpu_tensor *norm_out,
        const ds4_gpu_tensor *residual, const ds4_gpu_tensor *w, uint32_t w_stride,
        const void *model_map, uint64_t model_size, uint64_t norm_weight_offset,
        uint32_t rows, float eps) {
    if (!x || !norm_out || !residual || !w || !model_map || !rows || w_stride < 4u ||
        !cuda_tensor_has_elems2(x, 5120u, rows, 4u) ||
        !cuda_tensor_has_elems2(norm_out, 5120u, rows, 4u) ||
        !cuda_tensor_has_elems2(residual, 4u * 5120u, rows, 4u) ||
        !cuda_tensor_has_elems2(w, w_stride, rows, 4u) ||
        !cuda_model_range_fits(model_size, norm_weight_offset, 5120u * 4u)) return 0;
    const float *nw = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                                                          5120u * 4u, "rms_weight");
    if (!nw) return 0;
    v41_hc_wsum_norm_rows_kernel<<<rows, 1024>>>((const float *)w->ptr, w_stride,
        (const float *)residual->ptr, (float *)x->ptr, (float *)norm_out->ptr, nw, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 batched HC weighted sum + norm");
}

extern "C" int ds4_gpu_dsv41_hc_tail(ds4_gpu_tensor *split_out, const ds4_gpu_tensor *mix,
        const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset,
        uint32_t sinkhorn_iters, float hc_eps, const ds4_gpu_tensor *w,
        const ds4_gpu_tensor *residual, ds4_gpu_tensor *x, ds4_gpu_tensor *norm_out,
        uint64_t norm_weight_offset, float eps) {
    if (!split_out || !mix || !w || !residual || !x || !norm_out || !model_map ||
        !cuda_tensor_has_bytes(mix, 24u * 4u) || !cuda_tensor_has_bytes(split_out, 24u * 4u) ||
        !cuda_tensor_has_bytes(w, 4u * 4u) || !cuda_tensor_has_f32(residual, 4u * 5120u) ||
        !cuda_tensor_has_f32(x, 5120u) || !cuda_tensor_has_f32(norm_out, 5120u) ||
        !cuda_model_range_fits(model_size, scale_offset, 3u * 4u) ||
        !cuda_model_range_fits(model_size, base_offset, 24u * 4u) ||
        !cuda_model_range_fits(model_size, norm_weight_offset, 5120u * 4u)) return 0;
    const float *hs = (const float *)cuda_model_range_ptr(model_map, scale_offset, 12u, "hc_scale");
    const float *hb = (const float *)cuda_model_range_ptr(model_map, base_offset, 96u, "hc_base");
    const float *nw = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset, 5120u * 4u, "rms_weight");
    if (!hs || !hb || !nw) return 0;
    v41_hc_tail_kernel<<<1, 1024>>>((float *)split_out->ptr, (const float *)mix->ptr, hs, hb,
        sinkhorn_iters, hc_eps, (const float *)w->ptr, (const float *)residual->ptr,
        (float *)x->ptr, (float *)norm_out->ptr, nw, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 fused HC tail");
}

struct v41_rope_args {
    uint32_t width, heads, start, stride, inverse;
    float frequencies[32];
};

__global__ static void v41_rope_kernel(float *x, v41_rope_args args) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x / args.heads;
    const float theta = v41_mul((float)(args.start + row * args.stride), args.frequencies[lane]);
    const float c = __ocml_cos_f32(theta), s = args.inverse ? -__ocml_sin_f32(theta) : __ocml_sin_f32(theta);
    const uint64_t i = (uint64_t)blockIdx.x * args.width + args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = v41_bf16(v41_sub(v41_mul(re, c), v41_mul(im, s)));
    x[i + 1u] = v41_bf16(v41_add(v41_mul(re, s), v41_mul(im, c)));
}

static float v41_rope_frequencies[2][32];
static pthread_once_t v41_rope_once = PTHREAD_ONCE_INIT;

/* Keep the reference's host pow/reciprocal and YaRN operation order; a frequency ULP grows into a phase error at long positions. */
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#endif
static void v41_init_rope_frequencies(void) {
    for (int kind = 0; kind < 2; kind++) {
        const float base = kind ? 160000.0f : 10000.0f;
        const float low = (float)floor(64.0 * log(65536.0 / (32.0 * 2.0 * M_PI)) / (2.0 * log(base)));
        const float high = (float)ceil(64.0 * log(65536.0 / (2.0 * M_PI)) / (2.0 * log(base)));
        for (int i = 0; i < 32; i++) {
            const float denominator = powf(base, (float)i / 32.0f);
            float f = 1.0f / denominator;
            if (kind) {
                const float ramp = fminf(1.0f, fmaxf(0.0f, (i - low) / (high - low)));
                const float smooth = 1.0f - ramp;
                const float interpolate = (f / 16.0f) * (1.0f - smooth);
                const float extrapolate = f * smooth;
                f = interpolate + extrapolate;
            }
            v41_rope_frequencies[kind][i] = f;
        }
    }
}
#ifndef __HIP_DEVICE_COMPILE__
#pragma float_control(pop)
#endif

extern "C" int ds4_gpu_dsv41_rope_stride(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                         uint32_t rows, uint32_t start, uint32_t stride,
                                         bool compressed, bool inverse) {
    uint64_t elems = 0;
    if (width < 64u || !heads || !rows || rows > 1048576u || !stride ||
        (uint64_t)start + (uint64_t)(rows - 1u) * stride >= 1048576u ||
        (uint64_t)heads * rows > UINT32_MAX ||
        !cuda_u64_mul3_checked(width, heads, rows, &elems) || !cuda_tensor_has_f32(x, elems)) return 0;
    if (pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, stride, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    v41_rope_kernel<<<heads * rows, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 unit-magnitude RoPE");
}

/* bf16(whole head) + RoPE in one launch: the non-rotated dims are rounded in
 * place and the rotated pair is rounded before rotation, exactly as the separate
 * v41_bf16_kernel pass followed by v41_rope_kernel. */
__global__ static void v41_bf16_rope_kernel(float *x, v41_rope_args args) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x / args.heads;
    const uint64_t base = (uint64_t)blockIdx.x * args.width;
    for (uint32_t j = lane; j < args.width - 64u; j += 32u) x[base + j] = v41_bf16(x[base + j]);
    const float theta = v41_mul((float)(args.start + row * args.stride), args.frequencies[lane]);
    const float c = __ocml_cos_f32(theta), s = args.inverse ? -__ocml_sin_f32(theta) : __ocml_sin_f32(theta);
    const uint64_t i = base + args.width - 64u + 2u * lane;
    const float re = v41_bf16(x[i]), im = v41_bf16(x[i + 1u]);
    x[i] = v41_bf16(v41_sub(v41_mul(re, c), v41_mul(im, s)));
    x[i + 1u] = v41_bf16(v41_add(v41_mul(re, s), v41_mul(im, c)));
}

extern "C" int ds4_gpu_dsv41_bf16_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                       uint32_t start, bool compressed, bool inverse) {
    uint64_t elems = 0;
    if (width < 64u || !heads || start >= 1048576u ||
        !cuda_u64_mul3_checked(width, heads, 1u, &elems) || !cuda_tensor_has_f32(x, elems)) return 0;
    if (pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, 1u, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    v41_bf16_rope_kernel<<<heads, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 bf16 + RoPE");
}

__global__ static void v41_add_bf16_kernel(float *out, const float *a, const float *b, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = v41_bf16(a[i] + b[i]);   /* add_kernel then v41_bf16_kernel */
}

extern "C" int ds4_gpu_dsv41_add_bf16(ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
                                      const ds4_gpu_tensor *b, uint32_t n) {
    if (!n || !cuda_tensor_has_f32(out, n) || !cuda_tensor_has_f32(a, n) || !cuda_tensor_has_f32(b, n)) return 0;
    v41_add_bf16_kernel<<<(n + 255u) / 256u, 256u>>>((float *)out->ptr, (const float *)a->ptr,
                                                    (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "V4.1 add + bf16");
}

/* The same fused bf16 + RoPE kernel for consecutive rows starting at `start`. */
extern "C" int ds4_gpu_dsv41_bf16_rope_rows(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                            uint32_t rows, uint32_t start, bool compressed,
                                            bool inverse) {
    uint64_t elems = 0;
    if (width < 64u || !heads || !rows || rows > 1048576u ||
        (uint64_t)start + (uint64_t)(rows - 1u) >= 1048576u ||
        (uint64_t)heads * rows > UINT32_MAX ||
        !cuda_u64_mul3_checked(width, heads, rows, &elems) || !cuda_tensor_has_f32(x, elems)) return 0;
    if (pthread_once(&v41_rope_once, v41_init_rope_frequencies)) return 0;
    v41_rope_args args = {width, heads, start, 1u, inverse, {0}};
    memcpy(args.frequencies, v41_rope_frequencies[compressed ? 1 : 0], sizeof(args.frequencies));
    v41_bf16_rope_kernel<<<heads * rows, 32>>>((float *)x->ptr, args);
    return cuda_ok(cudaGetLastError(), "V4.1 batched bf16 + RoPE");
}

extern "C" int ds4_gpu_dsv41_rope(ds4_gpu_tensor *x, uint32_t width, uint32_t heads,
                                  uint32_t rows, uint32_t start, bool compressed, bool inverse) {
    return ds4_gpu_dsv41_rope_stride(x, width, heads, rows, start, 1, compressed, inverse);
}

__global__ static void v41_engram_kernel(float *residual, const float *kv, const float *qw,
                                        const float *kw, const uint8_t *mask, uint32_t width, float eps) {
    const uint32_t token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (mask && !mask[token]) return;
    const uint64_t offset = ((uint64_t)token * 4u + head) * width;
    const uint64_t key = ((uint64_t)token * 5u + head) * width;
    const uint64_t value = ((uint64_t)token * 5u + 4u) * width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint32_t i = lane; i < width; i += 32u) {
        const float h = residual[offset + i], k = v41_bf16(kv[key + i]);
        const uint64_t wi = (uint64_t)head * width + i;
        h2 = v41_add(h2, v41_mul(h, h));
        k2 = v41_add(k2, v41_mul(k, k));
        dot = v41_add(dot, v41_mul(v41_mul(h, v41_mul(qw[wi], kw[wi])), k));
    }
    h2 = v41_sum32(h2);
    k2 = v41_sum32(k2);
    dot = v41_mul(v41_sum32(dot), __ocml_rsqrt_f32(v41_add(v41_div(h2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32(v41_add(v41_div(k2, (float)width), eps)));
    dot = v41_mul(dot, __ocml_rsqrt_f32((float)width));
    const float gate = v41_div(1.0f, v41_add(1.0f,
        __ocml_exp_f32(-copysignf(__ocml_sqrt_f32(fmaxf(fabsf(dot), 1.0e-6f)), dot))));
    for (uint32_t i = lane; i < width; i += 32u)
        residual[offset + i] = v41_bf16(v41_add(residual[offset + i], v41_mul(gate, v41_bf16(kv[value + i]))));
}

extern "C" int ds4_gpu_dsv41_engram_add(ds4_gpu_tensor *residual, const ds4_gpu_tensor *kv,
                                        const ds4_gpu_tensor *q_weight, const ds4_gpu_tensor *k_weight,
                                        const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows, float eps) {
    const uint64_t count = (uint64_t)width * rows;
    if (!width || !rows || !isfinite(eps) || eps <= 0 || count > UINT64_MAX / 5u ||
        !cuda_tensor_has_f32(residual, count * 4u) || !cuda_tensor_has_f32(kv, count * 5u) ||
        !cuda_tensor_has_f32(q_weight, (uint64_t)width * 4u) ||
        !cuda_tensor_has_f32(k_weight, (uint64_t)width * 4u) ||
        (mask && !cuda_tensor_has_bytes(mask, rows))) return 0;
    v41_engram_kernel<<<dim3(rows, 4), 32>>>((float *)residual->ptr, (const float *)kv->ptr,
        (const float *)q_weight->ptr, (const float *)k_weight->ptr,
        mask ? (const uint8_t *)mask->ptr : NULL, width, eps);
    return cuda_ok(cudaGetLastError(), "V4.1 Engram gate");
}

__global__ static void v41_pool_kernel(float *out, const float *kv, const float *scores,
                                      const float *previous_kv, const float *previous_scores,
                                      uint32_t width, uint32_t tail) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const int64_t a = (int64_t)blockIdx.y * 2 - tail;
    const uint64_t b = (uint64_t)(a + 1) * width + col;
    const float ka = a < 0 ? previous_kv[col] : kv[(uint64_t)a * width + col];
    const float sa = a < 0 ? previous_scores[col] : scores[(uint64_t)a * width + col];
    const float sb = scores[b], peak = fmaxf(sa, sb);
    const float ea = __ocml_exp_f32(v41_sub(sa, peak)), eb = __ocml_exp_f32(v41_sub(sb, peak));
    out[(uint64_t)blockIdx.y * width + col] = v41_bf16(v41_div(
        v41_add(v41_mul(ka, ea), v41_mul(kv[b], eb)), v41_add(ea, eb)));
}

extern "C" int ds4_gpu_dsv41_pool2(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv,
                                   const ds4_gpu_tensor *scores, ds4_gpu_tensor *previous_kv,
                                   ds4_gpu_tensor *previous_scores, uint32_t width, uint32_t rows, uint32_t start) {
    const uint64_t count = (uint64_t)width * rows;
    const uint32_t pairs = (uint32_t)(((uint64_t)rows + (start & 1u)) / 2u);
    if (!width || !rows || rows > UINT32_MAX - start ||
        !cuda_tensor_has_f32(kv, count) || !cuda_tensor_has_f32(scores, count) ||
        !cuda_tensor_has_f32(previous_kv, width) || !cuda_tensor_has_f32(previous_scores, width) ||
        (pairs && !cuda_tensor_has_f32(out, (uint64_t)width * pairs))) return 0;
    if (pairs) {
        v41_pool_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), pairs), 256>>>(
            (float *)out->ptr, (const float *)kv->ptr, (const float *)scores->ptr,
            (const float *)previous_kv->ptr, (const float *)previous_scores->ptr, width, start & 1u);
        if (!cuda_ok(cudaGetLastError(), "V4.1 KV pair pooling")) return 0;
    }
    /* Retain the last even input even at even frontiers, so snapshots are independent of chunk partitioning. */
    const uint32_t last_even = (start + rows - 1u) & ~1u;
    if (last_even >= start) {
        const uint64_t bytes = (uint64_t)width * sizeof(float), offset = (last_even - start) * bytes;
        if (!ds4_gpu_tensor_copy(previous_kv, 0, kv, offset, bytes) ||
            !ds4_gpu_tensor_copy(previous_scores, 0, scores, offset, bytes)) return 0;
    }
    return 1;
}

template <bool FILTER>
__global__ static void v41_candidates_kernel(float *out, const float *scores, const float *mask,
                                            uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    const uint32_t blocks = (width + 7u) / 8u;
    const uint32_t visible = min(width, (start + row + 1u) / ratio);
    if (FILTER) {
        if (col >= width) return;
        const uint64_t i = (uint64_t)row * width + col;
        out[i] = col < visible && mask[(uint64_t)row * blocks + col / 8u] == 0.0f ? scores[i] : -INFINITY;
    } else {
        if (col >= blocks) return;
        float best = -INFINITY;
        for (uint32_t i = col * 8u; i < min(visible, (col + 1u) * 8u); i++)
            best = fmaxf(best, scores[(uint64_t)row * width + i]);
        if (visible && col == (visible - 1u) / 8u) best = INFINITY;
        out[(uint64_t)row * blocks + col] = best;
    }
}

static int v41_candidates(ds4_gpu_tensor *out, const ds4_gpu_tensor *scores,
                           const ds4_gpu_tensor *mask, uint32_t width, uint32_t rows,
                           uint32_t start, uint32_t ratio) {
    if (!width || width > UINT32_MAX - 7u || !rows || !ratio || rows > UINT32_MAX - start) return 0;
    const uint32_t blocks = (width + 7u) / 8u, out_width = mask ? width : blocks;
    if (!cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, out_width, rows, 4u) ||
        (mask && !cuda_tensor_has_elems2(mask, blocks, rows, 4u))) return 0;
    const dim3 grid((unsigned)(((uint64_t)out_width + 255u) / 256u), rows);
    if (mask) v41_candidates_kernel<true><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        (const float *)mask->ptr, width, start, ratio);
    else v41_candidates_kernel<false><<<grid, 256>>>((float *)out->ptr, (const float *)scores->ptr,
        NULL, width, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 candidate selection");
}

extern "C" int ds4_gpu_dsv41_candidate_blocks(ds4_gpu_tensor *blocks, const ds4_gpu_tensor *scores,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return v41_candidates(blocks, scores, NULL, width, rows, start, ratio);
}

extern "C" int ds4_gpu_dsv41_candidate_filter(ds4_gpu_tensor *scores, const ds4_gpu_tensor *block_mask,
                                              uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    return block_mask && v41_candidates(scores, scores, block_mask, width, rows, start, ratio);
}

__global__ static void v41_carry_bf16_kernel(uint16_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    const uint64_t p = (uint64_t)blockIdx.y * words * 2u + col;
    const uint64_t f = (uint64_t)blockIdx.y * width + col;
    if (pack) packed[p] = (uint16_t)(__float_as_uint(plain[f]) >> 16u);
    else plain[f] = __uint_as_float((uint32_t)packed[p] << 16u);
}

__global__ static void v41_carry_mask_kernel(uint32_t *packed, float *plain, uint32_t width,
                                           uint32_t words, bool pack) {
    const uint32_t word = blockIdx.x * blockDim.x + threadIdx.x;
    if (word >= words) return;
    const uint64_t p = (uint64_t)blockIdx.y * words + word;
    uint32_t bits = pack ? 0u : packed[p];
    for (uint32_t bit = 0; bit < 32u && (uint64_t)word * 32u + bit < width; bit++) {
        const uint64_t f = (uint64_t)blockIdx.y * width + (uint64_t)word * 32u + bit;
        if (pack) bits |= plain[f] == 0.0f ? 1u << bit : 0u;
        else plain[f] = bits & (1u << bit) ? 0.0f : -INFINITY;
    }
    if (pack) packed[p] = bits;
}

/* Same bits as v41_carry_mask_kernel, one lane per float: coalesced reads and
 * writes; pack uses a wave32 ballot, unpack broadcasts the word to its 32 lanes. */
__global__ static void v41_carry_mask_lane_kernel(uint32_t *packed, float *plain, uint32_t width,
                                                uint32_t words, bool pack) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t word = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (word >= words) return;
    const uint64_t p = (uint64_t)blockIdx.y * words + word;
    const uint32_t col = word * 32u + lane;
    const bool valid = col < width;
    const uint64_t f = (uint64_t)blockIdx.y * width + col;
    if (pack) {
        const bool zero = valid && plain[f] == 0.0f;
        const uint32_t bits = (uint32_t)__ballot(zero);
        if (lane == 0u) packed[p] = bits;
    } else {
        const uint32_t bits = packed[p];
        if (valid) plain[f] = bits & (1u << lane) ? 0.0f : -INFINITY;
    }
}

extern "C" int ds4_gpu_dsv41_carry_copy(ds4_gpu_tensor *packed, uint32_t row_offset,
                                        ds4_gpu_tensor *plain, uint32_t width, uint32_t rows,
                                        uint32_t format, bool pack) {
    if (!width || !rows || rows > UINT32_MAX - row_offset ||
        format > DS4_V41_CARRY_MASK || packed == plain) return 0;
    const uint32_t words = format == DS4_V41_CARRY_BF16 ?
        (uint32_t)(((uint64_t)width + 1u) / 2u) : (uint32_t)(((uint64_t)width + 31u) / 32u);
    if (!cuda_tensor_has_elems2(packed, (uint64_t)row_offset + rows, words, 4u) ||
        !cuda_tensor_has_elems2(plain, rows, width, 4u)) return 0;
    uint32_t *p = (uint32_t *)packed->ptr + (uint64_t)row_offset * words;
    if (format == DS4_V41_CARRY_BF16)
        v41_carry_bf16_kernel<<<dim3((unsigned)(((uint64_t)width + 255u) / 256u), rows), 256>>>(
            (uint16_t *)p, (float *)plain->ptr, width, words, pack);
    else if (!getenv("DS4_ROCM_V41_CARRY_MASK_SCALAR"))
        v41_carry_mask_lane_kernel<<<dim3((unsigned)(((uint64_t)words + 7u) / 8u), rows), 256>>>(
            p, (float *)plain->ptr, width, words, pack);
    else v41_carry_mask_kernel<<<dim3((unsigned)(((uint64_t)words + 255u) / 256u), rows), 256>>>(
        p, (float *)plain->ptr, width, words, pack);
    return cuda_ok(cudaGetLastError(), "V4.1 compact prefill carry");
}

__global__ static void v41_gather_kernel(float *out, const float *source, const int32_t *ids,
                                        uint32_t source_rows) {
    const uint32_t row = blockIdx.x, col = threadIdx.x;
    const int32_t id = ids[row];
    /* Graph IDs come from top-k. Keep malformed IDs from turning a validation failure into an out-of-bounds load. */
    if ((uint32_t)id >= source_rows) {
        out[(uint64_t)row * 512u + col] = NAN;
        out[(uint64_t)row * 512u + col + 256u] = NAN;
        return;
    }
    out[(uint64_t)row * 512u + col] = source[(uint64_t)id * 512u + col];
    out[(uint64_t)row * 512u + col + 256u] = source[(uint64_t)id * 512u + col + 256u];
}

extern "C" int ds4_gpu_dsv41_gather_kv(ds4_gpu_tensor *out, const ds4_gpu_tensor *source,
                                       const ds4_gpu_tensor *ids, uint32_t source_rows, uint32_t selected_rows) {
    if (!source_rows || !selected_rows || selected_rows > 512u || selected_rows > source_rows ||
        !cuda_tensor_has_elems2(source, source_rows, 512u, 4u) ||
        !cuda_tensor_has_elems2(out, selected_rows, 512u, 4u) || !cuda_tensor_has_f32(ids, selected_rows)) return 0;
    v41_gather_kernel<<<selected_rows, 256>>>((float *)out->ptr, (const float *)source->ptr,
        (const int32_t *)ids->ptr, source_rows);
    return cuda_ok(cudaGetLastError(), "V4.1 sparse KV gather");
}

/* One wave per score: retain the scalar F32 reduction tree and scale boundary. */
__global__ static void v41_indexer_scalar_warp8_kernel(
        float *out, const float *q, const float *w, const float *k, uint32_t width, uint32_t start,
        uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    const float *p = k + (uint64_t) key * 128 + lane;
    float k0 = p[0], k1 = p[32], k2 = p[64], k3 = p[96], total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(a, c), v41_add(b, d));
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        total = v41_add(total, v41_mul(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f), w[row * 32 + h]));
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}

/* Packed-key twin of v41_indexer_scalar_warp8_kernel: identical wave layout,
 * identical per-head reduction tree and scale/weight boundary, only the key
 * fetch changes (4 bits + per-32 scale instead of F32). */
__global__ static void v41_indexer_scalar_warp8_fp4_kernel(
        float *out, const float *q, const float *w, const unsigned char *pk, uint32_t width,
        uint32_t start, uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    float k0 = ds4_index_fp4_load(pk, key, lane), k1 = ds4_index_fp4_load(pk, key, lane + 32u);
    float k2 = ds4_index_fp4_load(pk, key, lane + 64u), k3 = ds4_index_fp4_load(pk, key, lane + 96u);
    float total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(a, c), v41_add(b, d));
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        total = v41_add(total, v41_mul(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f), w[row * 32 + h]));
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}

enum {
    V41_IX_DISP_WIDE_GEMM = 0,
    V41_IX_DISP_WARP8_BATCH,
    V41_IX_DISP_SCALAR_ONE,
    V41_IX_DISP_SCALAR_BATCH,
    V41_IX_DISP_COUNT
};
static void v41_ix_dispatch_note(int path, uint32_t source_rows, uint32_t rows,
        uint32_t start, uint32_t ratio);
extern "C" int ds4_gpu_dsv41_indexer_scores_one(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *keys, uint32_t source_rows) {
    if (!ds4_rocm_is_gfx1151())
        return ds4_gpu_glm_indexer_score_one_tensor(scores, q, weights,
            keys, source_rows, 32u, 128u, 1.0f / 64.0f, false);
    if (!source_rows || source_rows > INT32_MAX ||
        !cuda_tensor_has_f32(scores, source_rows) ||
        !cuda_tensor_has_f32(q, 32u * 128u) ||
        !cuda_tensor_has_f32(weights, 32u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u)) return 0;
    v41_ix_dispatch_note(V41_IX_DISP_SCALAR_ONE, source_rows, 1u, source_rows - 1u, 1u);
    v41_indexer_scalar_warp8_kernel<<<(source_rows + 7u) / 8u, 256>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const float *)keys->ptr, source_rows, source_rows - 1u, 1u);
    return cuda_ok(cudaGetLastError(), "V4.1 scalar warp indexer");
}

__global__ static void v41_indexer_kernel(float *scores, const float *q, const float *weights,
                                         const float *keys, uint32_t width, uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x, token = blockIdx.y, lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    if (key >= (start + token + 1u) / ratio) {
        if (!threadIdx.x) scores[(uint64_t)token * width + key] = -INFINITY;
        return;
    }
    __shared__ float head_values[4];
    float total = 0.0f;
    for (uint32_t head0 = 0; head0 < 32u; head0 += 4u) {
        const uint32_t head = head0 + wave;
        const float *query = q + ((uint64_t)token * 32u + head) * 128u;
        const float *kv = keys + (uint64_t)key * 128u;
        float dot = 0.0f;
        for (uint32_t col = lane; col < 128u; col += 32u) dot = v41_add(dot, v41_mul(query[col], kv[col]));
        dot = v41_sum32(dot);
        if (!lane) head_values[wave] = v41_mul(fmaxf(v41_mul(dot, 1.0f / 64.0f), 0.0f), weights[(uint64_t)token * 32u + head]);
        __syncthreads();
        if (!threadIdx.x) for (uint32_t h = 0; h < 4u; h++) total = v41_add(total, head_values[h]);
        __syncthreads();
    }
    if (!threadIdx.x) scores[(uint64_t)token * width + key] = total;
}

__global__ static void v41_indexer_batch_warp8_kernel(
        float *out, const float *q, const float *w, const float *k, uint32_t width, uint32_t start,
        uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    const float *p = k + (uint64_t) key * 128 + lane;
    float k0 = p[0], k1 = p[32], k2 = p[64], k3 = p[96], total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(v41_add(v41_add(0.0f, a), b), c), d);
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        float weighted;
        // OCML wrappers alone do not prevent backend contraction across this boundary.
        // The control stores this product in LDS before adding it to the head total.
        asm volatile("v_mul_f32 %0, %1, %2" : "=v"(weighted)
            : "v"(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f)), "v"(w[row * 32 + h]));
        total = v41_add(total, weighted);
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}
__global__ static void v41_indexer_head_queries_kernel(float *qh,const float *q,unsigned rows){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*4096u)return;
 unsigned d=i%128,h=(i/128)%32,t=i/4096;qh[(h*rows+t)*128+d]=q[i];
}
__global__ static void v41_indexer_reduce_heads_kernel(float *out,const float *dots,const float *w,unsigned width,unsigned rows,unsigned h0,unsigned heads,unsigned start,unsigned ratio){
 unsigned k=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(k>=width)return;
 size_t i=(size_t)row*width+k;
 if(k>=(start+row+1)/ratio){out[i]=-INFINITY;return;}
 float v=h0?out[i]:0.f;
 for(unsigned h=0;h<heads;h++)v=v41_add(v,v41_mul(fmaxf(v41_mul(dots[((size_t)h*rows+row)*width+k],1.f/64),0.f),w[row*32+h0+h]));
 out[i]=v;
}
#include "ds4_rocm_v41_ix_wmma.cuh"
/* Keep the query tile wide for GEMM; partition heads to bound scratch instead
 * of shrinking N to eight columns at long context. All operands remain F32. */
static int v41_indexer_head_gemm(float *out, const float *q, const float *weights,
        const float *keys, uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if (v41_indexer_wmma_enabled()) {
        if (!v41_indexer_wmma_launch(out, q, weights, keys, NULL, width, rows, start, ratio)) return 0;
        return cuda_ok(cudaGetLastError(), "V4.1 fused WMMA indexer");
    }
    static int check = -1;
    if (check < 0) { const char *v = getenv("DS4_ROCM_V41_INDEX_WMMA"); check = v && v[0] == '2'; }
    static unsigned check_calls;
    const bool do_check = check && (check_calls++ % 97u) == 0u;
    const uint64_t per_head = (uint64_t)rows * width * sizeof(float);
    const uint32_t heads = (uint32_t)min(UINT64_C(32), (UINT64_C(64) << 20) / per_head);
    const uint64_t q_bytes = (uint64_t)rows * 32u * 128u * sizeof(float);
    char *scratch = (char *)cuda_tmp_alloc(q_bytes + heads * per_head, "V4.1 indexer head GEMM");
    if (!scratch) return 0;
    float *qh = (float *)scratch, *dots = (float *)(scratch + q_bytes);
    v41_indexer_head_queries_kernel<<<(rows * 4096u + 255u) / 256u, 256u>>>(qh, q, rows);
    if (!cuda_ok(cudaGetLastError(), "V4.1 indexer query preparation")) return 0;
    const float one = 1.0f, zero = 0.0f;
    for (uint32_t h0 = 0; h0 < 32u; h0 += heads) {
        const uint32_t count = min(heads, 32u - h0);
        if (!cublas_ok(cublasSgemmStridedBatched(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                width, rows, 128, &one, keys, 128, 0,
                qh + (uint64_t)h0 * rows * 128u, 128, (long long)rows * 128,
                &zero, dots, width, (long long)rows * width, count), "V4.1 indexer head GEMM")) return 0;
        v41_indexer_reduce_heads_kernel<<<dim3((width + 255u) / 256u, rows), 256u>>>(
            out, dots, weights, width, rows, h0, count, start, ratio);
        if (!cuda_ok(cudaGetLastError(), "V4.1 indexer head reduction")) return 0;
    }
    if (do_check) {
        float *alt = NULL; unsigned long long *cnt = NULL, h[7];
        if (cudaMalloc(&alt, (size_t)rows * width * 4u) == cudaSuccess &&
            cudaMalloc(&cnt, sizeof h) == cudaSuccess && cudaMemset(cnt, 0, sizeof h) == cudaSuccess) {
            (void)v41_indexer_wmma_launch(alt, q, weights, keys, NULL, width, rows, start, ratio);
            v41_indexer_wmma_check_kernel<<<dim3((width + 255u) / 256u, rows), 256u>>>(
                cnt, out, alt, q, weights, keys, width, rows, start, ratio);
            if (cudaMemcpy(h, cnt, sizeof h, cudaMemcpyDeviceToHost) == cudaSuccess)
                fprintf(stderr, "ds4: ixcheck width=%u start=%u ratio=%u visible=%llu gemm!=wmma=%llu gemm!=ref=%llu wmma!=ref=%llu inexact_dot=%llu offgrid_key_blocks=%llu offgrid_q_blocks=%llu\n",
                        width, start, ratio, h[0], h[1], h[2], h[3], h[4], h[5], h[6]);
        }
        cudaFree(alt); cudaFree(cnt);
    }
    return 1;
}

static unsigned long long v41_ix_dispatch_counts[V41_IX_DISP_COUNT];
static unsigned char v41_ix_dispatch_seen[V41_IX_DISP_COUNT];
static unsigned int v41_ix_dispatch_last_source[V41_IX_DISP_COUNT];
static int v41_ix_dispatch_log_enabled(void) {
    static int cached = -1;
    if (cached < 0) cached = getenv("DS4_ROCM_V41_INDEXER_DISPATCH_LOG") != NULL ? 1 : 0;
    return cached;
}
static void v41_ix_dispatch_report(void) {
    if (!v41_ix_dispatch_log_enabled()) return;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
        "indexer dispatch counts wide_gemm=%llu warp8_batch=%llu scalar_one=%llu scalar_batch=%llu\n",
        v41_ix_dispatch_counts[V41_IX_DISP_WIDE_GEMM],
        v41_ix_dispatch_counts[V41_IX_DISP_WARP8_BATCH],
        v41_ix_dispatch_counts[V41_IX_DISP_SCALAR_ONE],
        v41_ix_dispatch_counts[V41_IX_DISP_SCALAR_BATCH]);
}
static void v41_ix_dispatch_note(int path, uint32_t source_rows, uint32_t rows,
        uint32_t start, uint32_t ratio) {
    if (!v41_ix_dispatch_log_enabled()) return;
    static int registered = 0;
    if (!registered) { registered = 1; atexit(v41_ix_dispatch_report); }
    const unsigned long long count = ++v41_ix_dispatch_counts[path];
    if (v41_ix_dispatch_seen[path] && v41_ix_dispatch_last_source[path] == source_rows) return;
    static const char *const names[] = {"wide_head_gemm", "warp8_batch", "scalar_one", "scalar_batch"};
    v41_ix_dispatch_seen[path] = 1u;
    v41_ix_dispatch_last_source[path] = source_rows;
    fprintf(stderr, DS4_GPU_LOG_PREFIX
        "indexer dispatch=%s count=%llu source_rows=%u rows=%u start=%u ratio=%u\n",
        names[path], count, source_rows, rows, start, ratio);
}
#ifndef DS4_ROCM_V41_INDEXER_HEAD_WIDE_DEFAULT
#define DS4_ROCM_V41_INDEXER_HEAD_WIDE_DEFAULT 1
#endif
static bool v41_indexer_wide_head_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        /* Default-on after paired 110K append A/B and isolated dispatch
         * qualification; DS4_ROCM_V41_INDEXER_HEAD_WIDE=0 restores the 64K cap. */
        const char *wide = getenv("DS4_ROCM_V41_INDEXER_HEAD_WIDE");
        cached = wide ? (!strcmp(wide, "1") ? 1 : !strcmp(wide, "0") ? 0 : DS4_ROCM_V41_INDEXER_HEAD_WIDE_DEFAULT) : DS4_ROCM_V41_INDEXER_HEAD_WIDE_DEFAULT;
        if (cached) fprintf(stderr, DS4_GPU_LOG_PREFIX "V4.1 indexer wide head active\n");
    }
    return cached;
}


/* Packed-key twin of v41_indexer_batch_warp8_kernel: same wave layout, same
 * ((((0+a)+b)+c)+d) head reduction and the same asm v_mul_f32 weight boundary. */
__global__ static void v41_indexer_batch_warp8_fp4_kernel(
        float *out, const float *q, const float *w, const unsigned char *pk, uint32_t width,
        uint32_t start, uint32_t ratio) {
    uint32_t key = blockIdx.x * 8u + threadIdx.x / 32, lane = threadIdx.x & 31, row = blockIdx.y;
    if (key >= width) return;
    if (key >= (start + row + 1) / ratio) {
        if (!lane) out[(uint64_t) row * width + key] = -INFINITY;
        return;
    }
    float k0 = ds4_index_fp4_load(pk, key, lane), k1 = ds4_index_fp4_load(pk, key, lane + 32u);
    float k2 = ds4_index_fp4_load(pk, key, lane + 64u), k3 = ds4_index_fp4_load(pk, key, lane + 96u);
    float total = 0;
    for (uint32_t h = 0; h < 32; h++) {
        const float *x = q + ((uint64_t) row * 32 + h) * 128 + lane;
        float a = v41_mul(x[0], k0), b = v41_mul(x[32], k1), c = v41_mul(x[64], k2), d = v41_mul(x[96], k3);
        float dot = v41_add(v41_add(v41_add(v41_add(0.0f, a), b), c), d);
        for (int s = 16; s; s >>= 1) dot = v41_add(dot, __shfl_down(dot, s, 32));
        float weighted;
        asm volatile("v_mul_f32 %0, %1, %2" : "=v"(weighted)
            : "v"(fmaxf(v41_mul(dot, 1.f / 64.f), 0.f)), "v"(w[row * 32 + h]));
        total = v41_add(total, weighted);
    }
    if (!lane) out[(uint64_t) row * width + key] = total;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_batch(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                  const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                  uint32_t source_rows, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || !rows || rows > UINT32_MAX - start ||
        (start + rows) / ratio > source_rows || source_rows > INT32_MAX || rows > INT32_MAX ||
        !cuda_tensor_has_elems2(scores, source_rows, rows, 4u) ||
        !cuda_tensor_has_elems2(q, rows, 32u * 128u, 4u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u) ||
        !cuda_tensor_has_elems2(weights, rows, 32u, 4u)) return 0;
    if (ds4_rocm_is_gfx1151() && !g_quality_mode && rows >= 16u && v41_indexer_wmma_enabled()) {
        v41_ix_dispatch_note(V41_IX_DISP_WIDE_GEMM, source_rows, rows, start, ratio);
        if (!v41_indexer_wmma_launch((float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)keys->ptr, NULL, source_rows, rows, start, ratio)) return 0;
        return cuda_ok(cudaGetLastError(), "V4.1 fused WMMA indexer");
    }
    if (ds4_rocm_is_gfx1151()) {
        /* Small query tiles favor wave scoring. GEMM uses a bounded head tile
         * only for the wide, nearly fully visible batches qualified here. */
        const uint32_t head_cap = v41_indexer_wide_head_enabled() ? 131072u : 65536u;
        if (!g_quality_mode && g_cublas_ready && rows >= 31u && rows <= 32u &&
            source_rows >= 16384u && source_rows <= head_cap &&
            (start + 1u) / ratio >= source_rows - source_rows / 8u) {
            v41_ix_dispatch_note(V41_IX_DISP_WIDE_GEMM, source_rows, rows, start, ratio);
            return v41_indexer_head_gemm((float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr, (const float *)keys->ptr, source_rows, rows, start, ratio);
        }
        v41_ix_dispatch_note(V41_IX_DISP_WARP8_BATCH, source_rows, rows, start, ratio);
        v41_indexer_batch_warp8_kernel<<<dim3((source_rows + 7u) / 8u, rows), 256u>>>(
            (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)keys->ptr, source_rows, start, ratio);
    } else {
        v41_ix_dispatch_note(V41_IX_DISP_SCALAR_BATCH, source_rows, rows, start, ratio);
        v41_indexer_kernel<<<dim3(source_rows, rows), 128>>>((float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr, (const float *)keys->ptr, source_rows, start, ratio);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 causal FP4 index scores");
}

template <uint32_t SORT_N>
__global__ static void v41_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void v41_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void v41_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t score_stride, uint32_t start, uint32_t ratio,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    const uint32_t n_comp = (start + t + 1u) / ratio;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * score_stride;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

/* Preserve each row's original sort network, including score/index tie order.
 * Group rows with equal chunk counts; only launches and physical strides change. */
static int v41_indexer_topk_causal_batch(uint32_t *selected, const float *scores,
        uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    const uint32_t max_chunks = ((start + rows) / ratio + 4095u) / 4096u;
    const uint32_t scratch_sets = max_chunks + (max_chunks > 8u ? (max_chunks + 7u) / 8u : 0u);
    uint32_t *scratch = (uint32_t *)cuda_tmp_alloc((uint64_t)rows * scratch_sets * 512u * sizeof(uint32_t),
                                                  "V4.1 causal top-k rows");
    if (!scratch) return 0;
    for (uint32_t row = 0; row < rows;) {
        const uint32_t chunks = ((start + row + 1u) / ratio + 4095u) / 4096u;
        uint32_t end = row + 1u;
        while (end < rows && ((start + end + 1u) / ratio + 4095u) / 4096u == chunks) ++end;
        const uint32_t count = end - row, begin = start + row;
        const float *input = scores + (uint64_t)row * width;
        uint32_t *cur = scratch, sets = chunks, stride = chunks * 512u;
        v41_topk_chunk_pow2_kernel<4096><<<dim3(count, chunks), 1024>>>(cur, input,
            width, begin, ratio, count, 512u, stride);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k chunks")) return 0;
        while (sets > 8u) {
            const uint32_t next_sets = (sets + 7u) / 8u, next_stride = next_sets * 512u;
            uint32_t *next = cur + (uint64_t)count * stride;
            v41_topk_tree_merge_pow2_kernel<4096><<<dim3(count, next_sets), 1024>>>(next,
                cur, input, width, begin, ratio, count, 512u, sets, 8u, stride, next_stride);
            if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k tree")) return 0;
            cur = next; sets = next_sets; stride = next_stride;
        }
        v41_topk_merge_pow2_kernel<4096><<<count, 1024>>>(selected + (uint64_t)row * 512u,
            cur, input, width, begin, ratio, count, 512u, sets * 512u, stride);
        if (!cuda_ok(cudaGetLastError(), "V4.1 causal top-k final")) return 0;
        row = end;
    }
    return 1;
}

extern "C" int ds4_gpu_dsv41_indexer_topk_batch(ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
                                               uint32_t width, uint32_t rows, uint32_t start, uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !rows || rows > UINT32_MAX - start ||
        width > INT32_MAX || rows > INT32_MAX || (start + rows) / ratio > width ||
        !cuda_tensor_has_elems2(scores, width, rows, 4u) ||
        !cuda_tensor_has_elems2(selected, 512u, rows, 4u)) return 0;
    /* The batched causal top-k used to start above 8192 visible rows; below
     * that every row launched its own sort. It is bit-identical from 512
     * visible rows (frontier logits at 4K/12K/24K, 2026-11-10) and cuts the
     * top-k of a 24K prefill from 2.9 to 0.85 s.
     * DS4_ROCM_V41_TOPK_BATCH_MIN=8192 restores the old crossover. */
    static int batch_min = -1;
    if (batch_min < 0) {
        const char *v = getenv("DS4_ROCM_V41_TOPK_BATCH_MIN");
        batch_min = v && v[0] ? atoi(v) : 512;
        if (batch_min < 512) batch_min = 512;
    }
    if (ds4_rocm_is_gfx1151() && rows >= 2u && rows <= 32u && width <= 65536u &&
        (start + 1u) / ratio > (uint32_t)batch_min)
        return v41_indexer_topk_causal_batch((uint32_t *)selected->ptr, (const float *)scores->ptr,
                                             width, rows, start, ratio);
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t visible = (start + row + 1u) / ratio;
        if (!visible) continue;
        const uint32_t top = visible < 512u ? visible : 512u;
        ds4_gpu_tensor in = {(float *)scores->ptr + (uint64_t)row * width, (uint64_t)visible * 4u, 0};
        ds4_gpu_tensor out = {(uint32_t *)selected->ptr + (uint64_t)row * 512u, 512u * 4u, 0};
        if (visible > 1u && visible < 512u && ds4_rocm_is_gfx1151()) {
            /* Preserve the scalar order and untouched tail slots, while
             * avoiding its serial insertion sort for short prefill rows. */
            indexer_topk_1024_kernel<<<1u, 1024u>>>((uint32_t *)out.ptr,
                (const float *)in.ptr, visible, 1u, top);
            if (!cuda_ok(cudaGetLastError(), "V4.1 short causal top-k")) return 0;
        } else if (!ds4_gpu_indexer_topk_tensor(&out, &in, visible, 1u, top)) return 0;
    }
    return 1;
}

/* Metal tensor packing is an optional acceleration. The graph selects the complete F32 baseline above when this capability is absent. */
extern "C" int ds4_gpu_dsv41_tensor_ops_available(void) { return 0; }

extern "C" uint64_t ds4_gpu_dsv41_indexer_packed_bytes(uint32_t source_rows, uint32_t rows) {
    const uint64_t tiles = ((uint64_t)source_rows + 63u) / 64u;
    const uint64_t flags = (((uint64_t)rows + tiles) * 4u + 255u) & ~UINT64_C(255);
    return flags + (uint64_t)rows * 32u * 128u * 2u + tiles * 64u * 128u * 2u;
}

extern "C" int ds4_gpu_dsv41_indexer_pack(ds4_gpu_tensor *packed, const ds4_gpu_tensor *q,
                                         const ds4_gpu_tensor *keys, uint32_t source_rows, uint32_t rows) {
    (void)packed; (void)q; (void)keys; (void)source_rows; (void)rows;
    return 0;
}

extern "C" int ds4_gpu_dsv41_indexer_scores_packed(ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
                                                   const ds4_gpu_tensor *weights, const ds4_gpu_tensor *keys,
                                                   const ds4_gpu_tensor *packed, uint32_t source_rows,
                                                   uint32_t rows, uint32_t start, uint32_t ratio,
                                                   uint32_t packed_rows, uint32_t offset) {
    (void)scores; (void)q; (void)weights; (void)keys; (void)packed;
    (void)source_rows; (void)rows; (void)start; (void)ratio; (void)packed_rows; (void)offset;
    return 0;
}

/* Each wave still owns one output row. Lane l sums the same contiguous
 * ceil(K/32) inputs in ascending order. Tokens have separate accumulators;
 * only the exactly decoded F16 weight is reused. Lane zero still adds the
 * 32 partials in order. No activation conversion or split-K reduction. */
template<unsigned TT, unsigned WAVES>
__global__ static void f16_ordered_token_reuse(
        float *out, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint64_t row = (uint64_t)blockIdx.x * WAVES + wave;
    const uint64_t token = (uint64_t)blockIdx.y * TT;
    __shared__ float partial[WAVES][TT][32];
    float sum[TT] = {};
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)lane * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    if (row < out_dim) {
        const __half *wr = w + row * in_dim;
        for (uint64_t i = k0; i < k1; i++) {
            const float weight = __half2float(wr[i]);
#pragma unroll
            for (unsigned t = 0; t < TT; t++) {
                if (token + t < n_tok)
                    sum[t] += weight * x[(token + t) * in_dim + i];
            }
        }
    }
#pragma unroll
    for (unsigned t = 0; t < TT; t++) partial[wave][t][lane] = sum[t];
    __syncthreads();
    if (row < out_dim && lane == 0u) {
#pragma unroll
        for (unsigned t = 0; t < TT; t++) {
            if (token + t < n_tok) {
                float total = 0.0f;
                for (uint32_t i = 0; i < 32u; i++) total += partial[wave][t][i];
                out[(token + t) * out_dim + row] = total;
            }
        }
    }
}

/* Batch the V4.1 F32-input F16 projections without changing their arithmetic.
 * Each wave still owns one output row; every token keeps the production
 * shared-X kernel's strided FMA sequence and warp reduction. The only change
 * is reusing each decoded F16 weight across TT independent token accumulators. */
template<unsigned TT, unsigned WAVES, unsigned SEG>
__global__ static void v41_f16_sharedx_token_reuse(
        float *out, const __half *weight, const float *input,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    extern __shared__ float tile[];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t row = blockIdx.x * WAVES + wave;
    const uint32_t token0 = blockIdx.y * TT;
    float acc[TT] = {};
    const __half *wr = row < out_dim ? weight + (uint64_t)row * in_dim : NULL;

    for (uint32_t base = 0; base < in_dim; base += SEG) {
        const uint32_t remaining = in_dim - base;
        const uint32_t valid = remaining < SEG ? remaining : SEG;
        for (uint32_t j = tid; j < TT * SEG; j += WAVES * 32u) {
            const uint32_t token = j / SEG;
            const uint32_t offset = j - token * SEG;
            tile[j] = token0 + token < rows && offset < valid ?
                input[(uint64_t)(token0 + token) * in_dim + base + offset] : 0.0f;
        }
        __syncthreads();
        if (row < out_dim) {
            for (uint32_t offset = lane; offset < valid; offset += 32u) {
                const float w = __half2float(wr[base + offset]);
#pragma unroll
                for (uint32_t token = 0; token < TT; token++) {
                    if (token0 + token < rows)
                        acc[token] += w * tile[token * SEG + offset];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (uint32_t token = 0; token < TT; token++) {
        const float total = warp_sum_f32(acc[token]);
        if (row < out_dim && token0 + token < rows && lane == 0u)
            out[(uint64_t)(token0 + token) * out_dim + row] = total;
    }
}

template<unsigned TT, unsigned SEG>
static void v41_launch_f16_sharedx_token_reuse(
        float *out, const __half *weight, const float *input,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    const uint32_t waves = 16u;
    const dim3 grid((out_dim + waves - 1u) / waves,
                    (rows + TT - 1u) / TT, 1u);
    v41_f16_sharedx_token_reuse<TT, 16u, SEG>
        <<<grid, waves * 32u, TT * SEG * sizeof(float)>>>(
            out, weight, input, in_dim, out_dim, rows);
}

__global__ static void v41_hc_half_to_float(float *out, const __half *weight, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(weight[i]);
}
static hipError_t v41_hc_widen(float *out, const uint16_t *weight, uint64_t count) {
    v41_hc_half_to_float<<<(count + 255u) / 256u, 256u>>>(out, (const __half *)weight, count);
    return hipGetLastError();
}
static bool v41_hc_disjoint(const void *a, uint64_t na, const void *b, uint64_t nb) {
    const uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return na <= UINTPTR_MAX - pa && nb <= UINTPTR_MAX - pb &&
        (pa + na <= pb || pb + nb <= pa);
}
#include "ds4_rocm_hc_sgemm.cuh"

extern "C" void ds4_gpu_dsv41_hc_plan_free(ds4_gpu_dsv41_hc_plan *plan) {
    v41_hc_plan_destroy(plan);
}
extern "C" int ds4_gpu_dsv41_hc_project(ds4_gpu_dsv41_hc_plan **plan,
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t rows, const ds4_gpu_tensor *input,
        ds4_gpu_tensor *full_heads_scratch) {
    if (!rows || rows > 2048u || !ds4_rocm_is_gfx1151() ||
        g_quality_mode || cuda_runtime_config()->graph_dump || !g_rocblas_ready ||
        g_rocblas_f16_solution_set != DS4_ROCBLAS_F16_SOLUTIONS_5_6_8D1AE90E) return 0;
    const uint64_t weight_bytes = UINT64_C(20480) * 24u * 2u;
    const uint64_t in_bytes = UINT64_C(20480) * rows * 4u;
    const uint64_t out_bytes = UINT64_C(24) * rows * 4u;
    if (!plan || !model_map || !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_bytes(input, in_bytes) || !cuda_tensor_has_bytes(out, out_bytes)) return -1;
    if (!cuda_tensor_has_bytes(full_heads_scratch, v41_hc_scratch_bytes)) return 0;
    if ((uintptr_t)full_heads_scratch->ptr % 256u ||
        !v41_hc_disjoint(out->ptr, out_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(out->ptr, out_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes) ||
        !v41_hc_disjoint(input->ptr, in_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    const uint16_t *weight = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "V4.1 HC F16");
    if (!weight ||
        !v41_hc_disjoint(weight, weight_bytes, out->ptr, out_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, input->ptr, in_bytes) ||
        !v41_hc_disjoint(weight, weight_bytes, full_heads_scratch->ptr, v41_hc_scratch_bytes)) return -1;
    return v41_hc_plan_run(plan, (float *)out->ptr, weight, (const float *)input->ptr,
                           full_heads_scratch->ptr, rows);
}

/* Default Engram lane l consumes l+32*i, i=0..191, then the existing
 * shuffle16/8/4/2/1. Each token retains an independent accumulation chain.
 * Only F32 input reuse across the eight output waves changes. */
template<unsigned TT,unsigned WAVES=8,unsigned SEG=256>
__global__ static void engram_lds_token_reuse(float *out,const __half *w,const float *x) {
    constexpr unsigned K=6144,N=25600;
    static_assert(K % SEG == 0, "Engram LDS segment must divide K");
    const unsigned tid=threadIdx.x,lane=tid&31u,wave=tid>>5u;
    const unsigned row=blockIdx.x*WAVES+wave,token=blockIdx.y*TT;
    __shared__ float tile[TT][SEG];
    float acc[TT]={};
    for(unsigned base=0;base<K;base+=SEG) {
        for(unsigned j=tid;j<TT*SEG;j+=WAVES*32u) {
            const unsigned t=j/SEG,k=j%SEG;
            tile[t][k]=x[(uint64_t)(token+t)*K+base+k];
        }
        __syncthreads();
        /* Match the scalar production kernel's eight explicit FMAs per
         * 256-wide group. Keeping that loop shape preserves bitwise parity
         * under the production fast-math flags. */
        for(unsigned group=0;group<SEG;group+=256u) {
            const unsigned k=group+lane;
#pragma unroll
            for(unsigned t=0;t<TT;t++) {
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k])*tile[t][k];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+32u])*tile[t][k+32u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+64u])*tile[t][k+64u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+96u])*tile[t][k+96u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+128u])*tile[t][k+128u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+160u])*tile[t][k+160u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+192u])*tile[t][k+192u];
                acc[t]+=__half2float(w[(uint64_t)row*K+base+k+224u])*tile[t][k+224u];
            }
        }
        // Every reader finishes before any thread overwrites the next segment.
        __syncthreads();
    }
#pragma unroll
    for(unsigned t=0;t<TT;t++) {
        const float total=warp_sum_f32(acc[t]);
        if(lane==0)out[(uint64_t)(token+t)*N+row]=total;
    }
}

/* Engram table scales can exceed F16 range. Only use the matrix path when
 * every activation survives conversion exactly; retain F32 input otherwise. */
__global__ static void v41_engram_pack_checked_kernel(__half *out,
        const float *input, uint32_t count, uint32_t *loss) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = input[i];
        const __half packed = __float2half_rn(value);
        out[i] = packed;
        if (__half2float(packed) != value) atomicOr(loss, 1u);
    }
}

/* Return zero when the installed library cannot provide the qualified plan.
 * The existing plan cache owns descriptor cleanup across model changes. */
static cuda_hipblaslt_gemm_plan *v41_engram_lt_plan(uint32_t rows) {
    if (!g_hipblaslt_ready) return NULL;
    int version = 0;
    char revision[128] = {0};
    if (hipblasLtGetVersion(g_hipblaslt, &version) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtGetGitRevision(g_hipblaslt, revision) != HIPBLAS_STATUS_SUCCESS ||
        version != 100401 || strcmp(revision, "8d1ae90e") != 0) return NULL;
    return hipblaslt_gemm_plan_get(25600u, rows, 6144u,
        "V4.1 Engram F16/F32", HIP_R_32F, 2539);
}

/* Tiny HC projection: prefetch aligned pairs without changing the scalar
 * 640-element lane accumulation or the ordered sum of 32 lane partials. */
__global__ static void v41_hc_ordered_prefetch_kernel(float *out,
        const __half *weights, const float *input) {
    constexpr unsigned width = 20480u, outputs = 24u, chunk = 640u;
    const unsigned lane = threadIdx.x, row = blockIdx.x, token = blockIdx.y;
    const __half *w = weights + (uint64_t)row * width;
    const float *x = input + (uint64_t)token * width;
    __shared__ float partial[32];
    float sum = 0.0f;
    for (unsigned base = lane * chunk; base < (lane + 1u) * chunk; base += 16u) {
        float ww[16], xx[16];
#pragma unroll
        for (unsigned u = 0; u < 16u; u += 2u) {
            const float2 wp = __half22float2(*reinterpret_cast<const __half2 *>(w + base + u));
            const float2 xp = *reinterpret_cast<const float2 *>(x + base + u);
            ww[u] = wp.x; ww[u + 1u] = wp.y;
            xx[u] = xp.x; xx[u + 1u] = xp.y;
        }
#pragma unroll
        for (unsigned u = 0; u < 16u; u++) sum = __fmaf_rn(ww[u], xx[u], sum);
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0u) {
        float total = 0.0f;
        for (unsigned i = 0; i < 32u; i++) total += partial[i];
        out[(uint64_t)token * outputs + row] = total;
    }
}

extern "C" int ds4_gpu_dsv41_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                             uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width, outputs, sizeof(uint16_t), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    if (width == 6144u && outputs == 25600u && rows >= 2u && rows <= 6u &&
        ds4_rocm_is_gfx1151() && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        switch (rows) {
        case 2u: engram_lds_token_reuse<2,2,768><<<12800u, 64u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 3u: engram_lds_token_reuse<3,4,1024><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 4u: engram_lds_token_reuse<4,8,1024><<<3200u, 256u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 5u: engram_lds_token_reuse<5,4,1024><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        case 6u: engram_lds_token_reuse<6,4,512><<<6400u, 128u>>>(
            (float *)out->ptr, w, (const float *)in->ptr); break;
        }
        return cuda_ok(cudaGetLastError(), "V4.1 exact tiny-row Engram projection");
    }
    if (width == 6144u && outputs == 25600u && rows >= 32u && rows <= 2048u &&
        ds4_rocm_is_gfx1151() && !g_quality_mode && !cuda_runtime_config()->graph_dump) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 exact Engram F16");
        if (!w) return 0;
        cuda_hipblaslt_gemm_plan *plan = v41_engram_lt_plan(rows);
        if (plan) {
            const uint32_t count = 6144u * rows;
            const uint64_t packed_bytes = (uint64_t)count * sizeof(__half);
            __half *packed = (__half *)cuda_tmp_alloc(packed_bytes + sizeof(uint32_t),
                "V4.1 Engram checked F16 activations");
            if (!packed) return 0;
            uint32_t *loss = (uint32_t *)((char *)packed + packed_bytes);
            if (!cuda_ok(hipMemsetAsync(loss, 0, sizeof(*loss), 0),
                         "V4.1 Engram conversion flag reset")) return 0;
            v41_engram_pack_checked_kernel<<<(count + 255u) / 256u, 256u>>>(
                packed, (const float *)in->ptr, count, loss);
            if (!cuda_ok(cudaGetLastError(), "V4.1 Engram checked conversion")) return 0;
            uint32_t converted_loss = 0;
            if (!cuda_ok(cudaMemcpy(&converted_loss, loss, sizeof(converted_loss),
                                   cudaMemcpyDeviceToHost), "V4.1 Engram conversion check")) return 0;
            if (!converted_loss) {
                const float alpha = 1.0f, beta = 0.0f;
                if (!hipblaslt_ok(hipblasLtMatmul(g_hipblaslt, plan->desc, &alpha,
                        w, plan->a_desc, packed, plan->b_desc, &beta,
                        out->ptr, plan->c_desc, out->ptr, plan->d_desc,
                        &plan->algo, NULL, 0, 0), "V4.1 Engram F16/F32")) return 0;
                return cuda_ok(cudaGetLastError(), "V4.1 Engram matrix projection");
            }
        }
        if (rows == 2048u) {
            engram_lds_token_reuse<16><<<dim3(3200u, 128u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr);
            return cuda_ok(cudaGetLastError(), "V4.1 exact Engram F32-input projection");
        }
        /* Partial batches retain the existing per-row F32-input fallback. */
    }
    if (width == 20480u && outputs == 24u && rows >= 8u && rows <= 2048u &&
        ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "f16");
        if (!w) return 0;
        /* Keep four-token tiles for smaller batches; the measured 384-row
         * workload and full 2048-row tiles favor eight-token reuse. */
        if (rows < 384u) {
            f16_ordered_token_reuse<4,8><<<dim3(3u, (rows + 3u) / 4u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        } else {
            f16_ordered_token_reuse<8,8><<<dim3(3u, (rows + 7u) / 8u), 256u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        }
        return cuda_ok(cudaGetLastError(), "V4.1 F32-input HC projection");
    }
    if (rows >= 32u && ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 F32-input F16 projection");
        if (!w) return 0;
        if (width == 5120u && outputs == 512u) {
            v41_launch_f16_sharedx_token_reuse<4u, 128u>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            return cuda_ok(cudaGetLastError(), "V4.1 F16 512 token-reuse projection");
        }
        if (width == 5120u && outputs == 32u) {
            v41_launch_f16_sharedx_token_reuse<4u, 512u>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            return cuda_ok(cudaGetLastError(), "V4.1 F16 32 token-reuse projection");
        }
        /* Indexer query-B (1280 -> 4096): same token-reuse kernel instead of
         * one launch per row. DS4_ROCM_V41_INDEX_QB_BATCH=0 restores the loop. */
        static int index_qb_batch = -1;
        if (index_qb_batch < 0) {
            const char *v = getenv("DS4_ROCM_V41_INDEX_QB_BATCH");
            index_qb_batch = !(v && v[0] == '0');
        }
        if (index_qb_batch && width == 1280u && outputs == 4096u) {
            static int qb_tt = -1;
            if (qb_tt < 0) { const char *v = getenv("DS4_ROCM_V41_INDEX_QB_TT"); qb_tt = v && v[0] ? atoi(v) : 16; }
            if (qb_tt == 16)
                v41_launch_f16_sharedx_token_reuse<16u, 128u>(
                    (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            else if (qb_tt == 8)
                v41_launch_f16_sharedx_token_reuse<8u, 128u>(
                    (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            else
            v41_launch_f16_sharedx_token_reuse<4u, 128u>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
            return cuda_ok(cudaGetLastError(), "V4.1 F16 indexer query-B token-reuse projection");
        }
    }
    if (width == 20480u && outputs == 24u && rows <= 6u &&
        ds4_rocm_is_gfx1151()) {
        const __half *w = (const __half *)cuda_model_range_ptr(
            model_map, weight_offset, weight_bytes, "V4.1 tiny HC F16");
        if (!w) return 0;
        /* Pair loads require aligned weights and F32 inputs. Tensor views
         * can be less aligned; retain the scalar kernel for those views. */
        if (((uintptr_t)w % alignof(__half2)) == 0u &&
            ((uintptr_t)in->ptr % alignof(float2)) == 0u) {
            v41_hc_ordered_prefetch_kernel<<<dim3(outputs, rows), 32u>>>(
                (float *)out->ptr, w, (const float *)in->ptr);
        } else {
            matmul_f16_ordered_chunks_kernel<<<dim3(outputs, rows), 32u>>>(
                (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
        }
        return cuda_ok(cudaGetLastError(), "V4.1 tiny HC F16 projection");
    }
    /* The general batched F16 API casts inputs to F16. Row views preserve decode arithmetic and retain F32 activations. */
    for (uint32_t row = 0; row < rows; row++) {
        ds4_gpu_tensor x = {(float *)in->ptr + (uint64_t)row * width, (uint64_t)width * 4u, 0};
        ds4_gpu_tensor y = {(float *)out->ptr + (uint64_t)row * outputs, (uint64_t)outputs * 4u, 0};
        if (!ds4_gpu_matmul_f16_tensor(&y, model_map, model_size, weight_offset, width, outputs, &x, 1u)) return 0;
    }
    return 1;
}

/* Keep the scalar 256-lane reduction tree. Reuse weights across tiny rows
 * and replace its final five block barriers with the same warp additions. */
template <uint32_t TOKENS>
__global__ static void v41_router_f32_rows_kernel(float *out, const float *weights,
        const float *input, uint32_t outputs, uint32_t rows) {
    constexpr uint32_t width = 5120u;
    const uint32_t lane = threadIdx.x, row = blockIdx.x;
    float sum[TOKENS] = {};
    for (uint32_t base = 0; base < 20u; base += 4u) {
        float w[4], x[TOKENS][4];
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
            const uint32_t col = lane + (base + u) * 256u;
            w[u] = weights[(uint64_t)row * width + col];
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                x[t][u] = t < rows ? input[(uint64_t)t * width + col] : 0.0f;
        }
#pragma unroll
        for (uint32_t u = 0; u < 4u; u++) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                sum[t] = __fmaf_rn(w[u], x[t][u], sum[t]);
        }
    }
    __shared__ float partial[TOKENS][256];
#pragma unroll
    for (uint32_t t = 0; t < TOKENS; t++) partial[t][lane] = sum[t];
    __syncthreads();
    for (uint32_t stride = 128u; stride >= 32u; stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t t = 0; t < TOKENS; t++)
                partial[t][lane] += partial[t][lane + stride];
        }
        __syncthreads();
    }
    if (lane < 32u) {
#pragma unroll
        for (uint32_t t = 0; t < TOKENS; t++) {
            float value = partial[t][lane];
            for (uint32_t stride = 16u; stride; stride >>= 1u)
                value += __shfl_down(value, stride, 32);
            if (lane == 0u && t < rows) out[(uint64_t)t * outputs + row] = value;
        }
    }
}

extern "C" int ds4_gpu_dsv41_f32_projection_rows(ds4_gpu_tensor *out,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t width, uint32_t outputs, uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || !outputs || !rows || rows > 6u || !model_map ||
        !cuda_u64_mul3_checked(width, outputs, sizeof(float), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    const float *w = (const float *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "V4.1 tiny F32 projection");
    if (!w) return 0;
    if (ds4_rocm_is_gfx1151() && width == 5120u &&
        (outputs == 128u || outputs == 384u)) {
#define V41_ROUTER_ROWS(T) v41_router_f32_rows_kernel<T><<<outputs, 256u>>>( \
        (float *)out->ptr, w, (const float *)in->ptr, outputs, rows)
        if (rows == 1u) { V41_ROUTER_ROWS(1u); }
        else if (rows == 2u) { V41_ROUTER_ROWS(2u); }
        else if (rows == 3u) { V41_ROUTER_ROWS(3u); }
        else { V41_ROUTER_ROWS(6u); }
#undef V41_ROUTER_ROWS
        return cuda_ok(cudaGetLastError(), "V4.1 tiny F32 router projection");
    }
    /* Preserve the scalar F32 kernel, including its 256-lane reduction tree. */
    matmul_f32_kernel<<<dim3(outputs, rows), 256u>>>(
        (float *)out->ptr, w, (const float *)in->ptr, width, outputs, rows);
    return cuda_ok(cudaGetLastError(), "V4.1 tiny F32 projection");
}

extern "C" int ds4_gpu_hc_rms_scale_project_f16_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scale_scratch,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t in_dim, uint32_t out_dim, const ds4_gpu_tensor *x, uint32_t n_rows, float eps) {
    if (!in_dim || !out_dim || !n_rows || !isfinite(eps) || eps <= 0.0f) return 0;
    return ds4_gpu_rms_norm_plain_rows_tensor(scale_scratch, x, in_dim, n_rows, eps) &&
        ds4_gpu_dsv41_projection_rows(out, model_map, model_size, weight_offset, in_dim, out_dim, n_rows, scale_scratch);
}

/* Four input values per lane, four Q8 blocks per wave. Scalar V4.1
 * projections retain F32 activations; the block-wise reduction differs from
 * the original lane accumulation and is qualified independently. */
/* Opt-in: one launch for two Q8 single-row projections that share the same
 * input row and row_bytes (decode q_a + kv). Rows below rows_a read w_a, the
 * rest read w_b; per-row lane mapping and accumulation order are identical to
 * v41_q8_f32_blocks4_kernel, so the values are bit-identical. */
__global__ static void v41_q8_f32_blocks4_pair_kernel(
        float *out_a, float *out_b,
        const unsigned char *w_a, const unsigned char *w_b,
        const float *input, uint32_t width, uint32_t rows_a, uint32_t rows_b,
        uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row_idx = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t total = rows_a + rows_b;
    if (row_idx >= total) return;
    const bool second = row_idx >= rows_a;
    const uint64_t row = second ? (uint64_t)(row_idx - rows_a) : (uint64_t)row_idx;
    const unsigned char *w = second ? w_b : w_a;
    const uint32_t blocks = width / 32u;
    const uint32_t j = (lane & 7u) * 4u;
    float acc = 0.0f;
#pragma unroll 4
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *p = w + row * row_bytes + (uint64_t)b * 34u;
        const float d = q8_0_scale_scalar(p);
        float value = 0.0f;
        value += (float)((const int8_t *)(p + 2u))[j + 0] * input[b * 32u + j + 0];
        value += (float)((const int8_t *)(p + 2u))[j + 1] * input[b * 32u + j + 1];
        value += (float)((const int8_t *)(p + 2u))[j + 2] * input[b * 32u + j + 2];
        value += (float)((const int8_t *)(p + 2u))[j + 3] * input[b * 32u + j + 3];
        acc += d * value;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) (second ? out_b : out_a)[row] = acc;
}

__device__ static uint32_t g_v41_q8b4_pf;
static int g_v41_q8b4_pf_set;
static void v41_q8b4_prefetch_publish(void) {
    if (g_v41_q8b4_pf_set) return;
    g_v41_q8b4_pf_set = 1;
    /* Promoted: the unrolled blocks4 loop is on by default; =0 restores the
     * plain loop. */
    const char *pf_env = getenv("DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH");
    const uint32_t on = (pf_env != NULL && pf_env[0] == '0' && pf_env[1] == '\0') ? 0u : 1u;
    (void)hipMemcpyToSymbol(HIP_SYMBOL(g_v41_q8b4_pf), &on, sizeof(on), 0,
                            hipMemcpyHostToDevice);
    if (on) fprintf(stderr, "ds4: V4.1 Q8 blocks4 unrolled loop active\n");
}

template <bool ROUND = false>
__global__ static void v41_q8_f32_blocks4_kernel(float *out,
        const unsigned char *weights, const float *input,
        uint32_t width, uint32_t outputs, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t token = blockIdx.y;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    input += (uint64_t)token * width;
    out += (uint64_t)token * outputs;
    const uint32_t blocks = width / 32u;
    const uint32_t j = (lane & 7u) * 4u;
    float acc = 0.0f;
    if (g_v41_q8b4_pf != 0u) {
        /* Opt-in unrolled block loop: four iterations' loads are issued up
         * front so the 40-step dependent chain is hidden behind four memory
         * round trips instead of one. Same lane->block mapping and the same
         * per-block accumulation order, so the result is bit-identical. */
#pragma unroll 4
        for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
            const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
            const float d = q8_0_scale_scalar(p);
            float value = 0.0f;
            value += (float)((const int8_t *)(p + 2u))[j + 0] * input[b * 32u + j + 0];
            value += (float)((const int8_t *)(p + 2u))[j + 1] * input[b * 32u + j + 1];
            value += (float)((const int8_t *)(p + 2u))[j + 2] * input[b * 32u + j + 2];
            value += (float)((const int8_t *)(p + 2u))[j + 3] * input[b * 32u + j + 3];
            acc += d * value;
        }
    } else {
        for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
            const unsigned char *p = weights + row * row_bytes + (uint64_t)b * 34u;
            const float d = q8_0_scale_scalar(p);
            float value = 0.0f;
#pragma unroll
            for (uint32_t k = 0; k < 4u; k++)
                value += (float)((const int8_t *)(p + 2u))[j + k] * input[b * 32u + j + k];
            acc += d * value;
        }
    }
    acc = warp_sum_f32(acc);
    /* ROUND applies the same v41_bf16 as the separate v41_bf16_kernel pass. */
    if (lane == 0u) out[row] = ROUND ? v41_bf16(acc) : acc;
}

/* V4.1 shared-expert gate/up pair. Each warp owns one mid row and retains
 * the scalar blocks4 lane ownership, per-lane dot boundary and wave32
 * reduction tree. Rounding gate and up before SwiGLU preserves the two
 * standalone BF16 boundaries; the final rounding is the old mid BF16 pass. */
__global__ static void v41_q8_f32_pair_swiglu_bf16_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const unsigned char *gate_weights, const unsigned char *up_weights,
        const float *input, uint32_t width, uint32_t outputs,
        uint64_t row_bytes, float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= outputs) return;
    const uint32_t blocks = width / 32u;
    float acc_g = 0.0f, acc_u = 0.0f;
    for (uint32_t b = lane / 8u; b < blocks; b += 4u) {
        const unsigned char *pg = gate_weights + row * row_bytes + (uint64_t)b * 34u;
        const unsigned char *pu = up_weights + row * row_bytes + (uint64_t)b * 34u;
        const float dg = q8_0_scale_scalar(pg);
        const float du = q8_0_scale_scalar(pu);
        const uint32_t j = (lane & 7u) * 4u;
        float value_g = 0.0f, value_u = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 4u; k++) {
            const float x = input[b * 32u + j + k];
            value_g += (float)((const int8_t *)(pg + 2u))[j + k] * x;
            value_u += (float)((const int8_t *)(pu + 2u))[j + k] * x;
        }
        acc_g += dg * value_g;
        acc_u += du * value_u;
    }
    float g = warp_sum_f32(acc_g);
    float u = warp_sum_f32(acc_u);
    if (lane == 0u) {
        g = v41_bf16(g);
        u = v41_bf16(u);
        gate_out[row] = g;
        up_out[row] = u;
        if (clamp > 1.0e-6f) {
            g = fminf(g, clamp);
            u = fminf(fmaxf(u, -clamp), clamp);
        }
        const float s = g / (1.0f + expf(-g));
        mid_out[row] = v41_bf16(s * u * 1.0f);
    }
}

/* DSpark verifies two to six consecutive rows.  Reuse each packed Q8 block
 * across all rows while preserving the scalar blocks4 lane ownership,
 * per-lane dot boundary and wave32 reduction tree.  GROUPS supports the
 * interleaved eight-group attention-A layout; dense projections use one. */
template <uint32_t ROWS, uint32_t GROUPS = 1u>
__global__ __launch_bounds__(32) static void v41_q8_f32_tiny_rows_kernel(
        float *out, const unsigned char *weights, const float *input,
        uint32_t width, uint32_t rows_per_group, uint64_t row_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x;
    const uint32_t group = row / rows_per_group;
    const uint32_t outputs = rows_per_group * GROUPS;
    const uint32_t blocks = width / 32u;
    float acc[ROWS] = {};
    for (uint32_t base = lane / 8u; base < blocks; base += 8u) {
#pragma unroll
        for (uint32_t u = 0; u < 2u; u++) {
            const uint32_t b = base + 4u * u;
            if (b >= blocks) continue;
            const unsigned char *p = weights + (uint64_t)row * row_bytes +
                                     (uint64_t)b * 34u;
            const float scale = q8_0_scale_scalar(p);
            const uint32_t j = (lane & 7u) * 4u;
#pragma unroll
            for (uint32_t t = 0; t < ROWS; t++) {
                const float *x = input + ((uint64_t)t * GROUPS + group) * width +
                                 (uint64_t)b * 32u + j;
                float value = 0.0f;
#pragma unroll
                for (uint32_t k = 0; k < 4u; k++)
                    value += (float)((const int8_t *)(p + 2u))[j + k] * x[k];
                acc[t] += scale * value;
            }
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < ROWS; t++) {
        const float total = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * outputs + row] = total;
    }
}

template <uint32_t GROUPS>
static int v41_q8_f32_tiny_rows_launch(float *out, const unsigned char *weights,
        const float *input, uint32_t width, uint32_t rows_per_group,
        uint32_t rows, uint64_t row_bytes) {
    const uint32_t outputs = rows_per_group * GROUPS;
    switch (rows) {
        case 2u: v41_q8_f32_tiny_rows_kernel<2u, GROUPS><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 3u: v41_q8_f32_tiny_rows_kernel<3u, GROUPS><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 4u: v41_q8_f32_tiny_rows_kernel<4u, GROUPS><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 5u: v41_q8_f32_tiny_rows_kernel<5u, GROUPS><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        case 6u: v41_q8_f32_tiny_rows_kernel<6u, GROUPS><<<outputs, 32u>>>(
            out, weights, input, width, rows_per_group, row_bytes); break;
        default: return 0;
    }
    return cuda_ok(cudaGetLastError(), "V4.1 tiny-row Q8 projection");
}

/* Opt-in shape log for the Q8 blocks4 decode projections: which shapes are
 * launched, how often, to find pairs that can share one launch. */
static void v41_q8b4_log_shape(const char *site, uint32_t width, uint32_t outputs,
                               uint64_t row_bytes, uint32_t rows) {
    if (getenv("DS4_ROCM_V41_Q8_SHAPE_LOG") == NULL) return;
    static char seen[64][160];
    static int n_seen;
    char key[160];
    snprintf(key, sizeof(key), "%-12s width=%-5u outputs=%-5u row_bytes=%-6llu rows=%u",
             site, width, outputs, (unsigned long long)row_bytes, rows);
    for (int i = 0; i < n_seen; ++i) if (strcmp(seen[i], key) == 0) return;
    if (n_seen < 64) { snprintf(seen[n_seen++], sizeof(seen[0]), "%s", key); }
    fprintf(stderr, "ds4: q8b4 %s\n", key);
}

extern "C" int ds4_gpu_v41_leaf_note(const char *label, unsigned long long bytes);

/* Decode-scoped leaf timing for the fused attention projections. Only single-token
 * calls are accumulated, so prefill rows can never enter the totals. */
static int ds4_leaf_profile_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("DS4_V41_LEAF_PROFILE");
        cached = (v && *v && *v != '0') ? 1 : 0;
    }
    return cached;
}

extern "C" int ds4_gpu_dsv41_q8_projection_pair_rows(
        ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
        const void *model_map, uint64_t model_size,
        uint64_t offset_a, uint32_t outputs_a,
        uint64_t offset_b, uint32_t outputs_b,
        uint32_t width, const ds4_gpu_tensor *in) {
    const int leaf = ds4_leaf_profile_on() && in && in->bytes == (uint64_t)width * 4ull;
    if (leaf) (void)ds4_gpu_v41_leaf_begin();
    /* Promoted: q_a+kv fusion is on by default; =0 restores two launches. */
    {
        const char *e = getenv("DS4_ROCM_V41_Q8_FUSE_QAKV");
        if (e != NULL && e[0] == '0' && e[1] == '\0') return 0;
    }
    if (!out_a || !out_b || !model_map || !in || !width || width % 32u ||
        !outputs_a || !outputs_b || !ds4_rocm_is_gfx1151()) return 0;
    if (!cuda_tensor_has_elems2(in, width, 1u, 4u) ||
        !cuda_tensor_has_elems2(out_a, outputs_a, 1u, 4u) ||
        !cuda_tensor_has_elems2(out_b, outputs_b, 1u, 4u)) return 0;
    const uint64_t row_bytes = (uint64_t)(width / 32u) * 34u;
    uint64_t bytes_a = 0, bytes_b = 0;
    if (!cuda_u64_mul_checked(row_bytes, outputs_a, &bytes_a) ||
        !cuda_u64_mul_checked(row_bytes, outputs_b, &bytes_b) ||
        !cuda_model_range_fits(model_size, offset_a, bytes_a) ||
        !cuda_model_range_fits(model_size, offset_b, bytes_b)) return 0;
    const unsigned char *wa = (const unsigned char *)cuda_model_range_ptr(
        model_map, offset_a, bytes_a, "V4.1 Q8 pair A");
    const unsigned char *wb = (const unsigned char *)cuda_model_range_ptr(
        model_map, offset_b, bytes_b, "V4.1 Q8 pair B");
    if (!wa || !wb) return 0;
    const uint32_t total = outputs_a + outputs_b;
    v41_q8_f32_blocks4_pair_kernel<<<(total + 7u) / 8u, 256u>>>(
        (float *)out_a->ptr, (float *)out_b->ptr, wa, wb,
        (const float *)in->ptr, width, outputs_a, outputs_b, row_bytes);
    if (leaf) {
        const uint64_t per_out = (uint64_t)width * 34ull / 32ull;   /* q8_0 bytes per output row */
        (void)ds4_gpu_v41_leaf_end("attn_q_a_kv_pair",
            (unsigned long long)(((uint64_t)outputs_a + outputs_b) * width * 34ull / 32ull));
        (void)ds4_gpu_v41_leaf_note("attn_q_a", (unsigned long long)(outputs_a * per_out));
        (void)ds4_gpu_v41_leaf_note("attn_kv",  (unsigned long long)(outputs_b * per_out));
    }
    return cuda_ok(cudaGetLastError(), "V4.1 Q8 fused pair projection");
}

/* Batched sparse attention on the kernel from rocm/gufo_f16/ds41_v41.hip.cpp.
 * It reads the compressed KV rows in their packed FP4 form (256 code bytes +
 * 32 scale bytes per row instead of 2 KiB of F32), which is what bounds this
 * step: every token gathers its 512 selected rows twice. The rows in the F32
 * cache are already FP4 values (DS4_V41_FP4_E4M3, blocks of 16), so they are
 * packed on the fly for each call; DS4_ROCM_V41_ATTN_FP4_CHECK=1 verifies
 * that packing reproduces them exactly and reports mismatches.
 * `ring` holds the window rows before pos0 at slot pos % window, `chunk` this
 * batch's rows. Returns 1 done, 0 not applicable.
 * DS4_ROCM_V41_ATTN_FP4=0 disables. */
typedef struct ds41_fp4_rows { const uint8_t *codes; const uint8_t *scales; } ds41_fp4_rows;
extern "C" int ds41_v41_fp4_store(const float *x, uint8_t *codes, uint8_t *scales, uint32_t n,
                                  uint32_t block, int e4m3_scale);
extern "C" int ds41_v41_fp4_fake_quant(float *x, uint32_t n, uint32_t block, int e4m3_scale);
extern "C" int ds41_v41_attention(float *q, const float *ring, const float *chunk, uint32_t window,
        uint32_t floor, ds41_fp4_rows comp, const uint32_t *idx, uint32_t k, uint32_t ratio,
        const float *sinks, float *out, uint16_t *out_half, float *partial, uint32_t pos0,
        uint32_t n, uint32_t heads, uint32_t head_dim, float scale, const float *inv_freq,
        uint32_t rope_dim, float *rope_scratch);
__global__ static void v41_count_diff_kernel(const float *a, const float *b, uint64_t n, uint32_t *count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(a[i]) != __float_as_uint(b[i])) atomicAdd(count, 1u);
}
extern "C" int ds4_gpu_dsv41_attention_fp4_batch(ds4_gpu_tensor *heads, const void *model_map,
        uint64_t model_size, uint64_t sinks_offset, ds4_gpu_tensor *q, const ds4_gpu_tensor *ring,
        const ds4_gpu_tensor *chunk, const ds4_gpu_tensor *comp_kv, const ds4_gpu_tensor *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_comp, uint32_t top_k, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_ROCM_V41_ATTN_FP4");
        on = !(v && v[0] == '0') && ds4_rocm_is_gfx1151();
    }
    if (!on || g_quality_mode || n_tokens < 8u || n_tokens > 2048u || !ratio || top_k != 512u ||
        n_comp < top_k || window != 128u || n_head != 64u || head_dim != 512u ||
        !heads || !q || !ring || !chunk || !comp_kv || !topk || !model_map ||
        (uint64_t)n_comp * head_dim > UINT32_MAX ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        ring->bytes < (uint64_t)window * head_dim * sizeof(float) ||
        chunk->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t) ||
        !cuda_model_range_fits(model_size, sinks_offset, (uint64_t)n_head * sizeof(float))) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(model_map, sinks_offset,
        (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    static uint8_t *codes = NULL, *scales = NULL;
    static uint32_t cap_rows = 0;
    if (n_comp > cap_rows) {
        if (codes) (void)cudaFree(codes);
        if (scales) (void)cudaFree(scales);
        codes = scales = NULL; cap_rows = 0;
        const uint32_t want = n_comp < 16384u ? 16384u : n_comp + n_comp / 4u;
        if (!cuda_ok(cudaMalloc(&codes, (size_t)want * (head_dim / 2u)), "V4.1 FP4 KV codes") ||
            !cuda_ok(cudaMalloc(&scales, (size_t)want * (head_dim / 16u)), "V4.1 FP4 KV scales")) return 0;
        cap_rows = want;
    }
    const uint32_t n_values = n_comp * head_dim;
    static int check = -1;
    if (check < 0) check = getenv("DS4_ROCM_V41_ATTN_FP4_CHECK") != NULL;
    if (check) {
        float *tmp = NULL; uint32_t *cnt = NULL; uint32_t diff = 0;
        if (cuda_ok(cudaMalloc(&tmp, (size_t)n_values * sizeof(float)), "V4.1 FP4 check") &&
            cuda_ok(cudaMalloc(&cnt, sizeof(uint32_t)), "V4.1 FP4 check count")) {
            (void)cudaMemcpy(cnt, &diff, sizeof(diff), cudaMemcpyHostToDevice);
            (void)cudaMemcpy(tmp, comp_kv->ptr, (size_t)n_values * sizeof(float), cudaMemcpyDeviceToDevice);
            (void)ds41_v41_fp4_fake_quant(tmp, n_values, 16u, 1);
            v41_count_diff_kernel<<<(unsigned)((n_values + 255u) / 256u), 256u>>>(tmp,
                (const float *)comp_kv->ptr, n_values, cnt);
            (void)cudaMemcpy(&diff, cnt, sizeof(diff), cudaMemcpyDeviceToHost);
            static uint64_t calls = 0, bad = 0;
            calls++; bad += diff;
            if (diff || (calls % 64u) == 1u)
                fprintf(stderr, "ds4: V4.1 FP4 KV check call=%llu rows=%u mismatched values=%u (total %llu)\n",
                        (unsigned long long)calls, n_comp, diff, (unsigned long long)bad);
        }
        if (tmp) (void)cudaFree(tmp);
        if (cnt) (void)cudaFree(cnt);
    }
    if (ds41_v41_fp4_store((const float *)comp_kv->ptr, codes, scales, n_values, 16u, 1)) return 0;
    const ds41_fp4_rows comp = {codes, scales};
    if (ds41_v41_attention((float *)q->ptr, (const float *)ring->ptr, (const float *)chunk->ptr,
            window, 0u, comp, (const uint32_t *)topk->ptr, top_k, ratio, sinks,
            (float *)heads->ptr, NULL, NULL, pos0, n_tokens, n_head, head_dim,
            1.0f / sqrtf((float)head_dim), NULL, 0u, NULL)) return 0;
    static int logged = 0;
    if (!logged) { logged = 1; fprintf(stderr, "ds4: ROCm V4.1 FP4-row attention active (rows=%u)\n", n_tokens); }
    return 1;
}

/* Router projection (F32 384 x 5120) on the BF16 matrix cores
 * (rocm/gufo_f16/ds41_v41.hip.cpp). The router weights are BF16 values stored
 * as F32, so their BF16 copy is exact; it is made once per layer and kept.
 * The kernel splits each F32 activation into three BF16 terms, so every
 * product is exact and only the summation order differs from SGEMM.
 * Returns 1 done, 0 not applicable (caller uses its usual path).
 * DS4_ROCM_V41_ROUTER_BF16=0 disables. */
extern "C" int ds41_v41_matmul_bf16(const uint16_t *w, const float *x, float *out, uint32_t rows,
                                    uint32_t k, uint32_t n);
__global__ static void v41_f32_to_bf16_exact_kernel(uint16_t *out, const float *in, uint64_t n,
                                                   uint32_t *inexact) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t bits = __float_as_uint(in[i]);
    out[i] = (uint16_t)(bits >> 16);
    if (bits & 0xffffu) atomicOr(inexact, 1u);
}
extern "C" int ds4_gpu_dsv41_router_bf16_rows(ds4_gpu_tensor *out, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint32_t width, uint32_t outputs,
        uint32_t rows, const ds4_gpu_tensor *in) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_ROCM_V41_ROUTER_BF16");
        on = !(v && v[0] == '0') && ds4_rocm_is_gfx1151();
    }
    if (!on || g_quality_mode || rows < 16u || rows > 2048u || outputs != 384u || width != 5120u ||
        !model_map || !cuda_tensor_has_elems2(in, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out, outputs, rows, 4u) ||
        ((uintptr_t)in->ptr % 16u) != 0u) return 0;
    const uint64_t n = (uint64_t)outputs * width;
    if (!cuda_model_range_fits(model_size, weight_offset, n * sizeof(float))) return 0;
    struct entry { const void *map; uint64_t offset; uint16_t *w; int state; };
    static entry cache[64];
    entry *e = NULL;
    for (uint32_t i = 0; i < 64u; i++) {
        if (cache[i].state && cache[i].map == model_map && cache[i].offset == weight_offset) { e = &cache[i]; break; }
        if (!cache[i].state) {
            const float *w32 = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                n * sizeof(float), "V4.1 router F32");
            if (!w32) return 0;
            uint16_t *w16 = NULL; uint32_t *flag = NULL;
            if (!cuda_ok(cudaMalloc(&w16, n * sizeof(uint16_t)), "V4.1 router BF16 weights") ||
                !cuda_ok(cudaMalloc(&flag, sizeof(uint32_t)), "V4.1 router BF16 check")) return 0;
            uint32_t inexact = 0;
            if (!cuda_ok(cudaMemcpy(flag, &inexact, sizeof(inexact), cudaMemcpyHostToDevice), "V4.1 router BF16 check reset")) return 0;
            v41_f32_to_bf16_exact_kernel<<<(unsigned)((n + 255u) / 256u), 256u>>>(w16, w32, n, flag);
            if (!cuda_ok(cudaGetLastError(), "V4.1 router BF16 convert") ||
                !cuda_ok(cudaMemcpy(&inexact, flag, sizeof(inexact), cudaMemcpyDeviceToHost), "V4.1 router BF16 check read")) return 0;
            (void)cudaFree(flag);
            cache[i].map = model_map; cache[i].offset = weight_offset;
            if (inexact) {
                /* Not BF16 values: keep SGEMM for this matrix. */
                (void)cudaFree(w16);
                cache[i].w = NULL; cache[i].state = 2;
                fprintf(stderr, "ds4: V4.1 router weights at %llu are not exact BF16; keeping SGEMM\n",
                        (unsigned long long)weight_offset);
            } else {
                cache[i].w = w16; cache[i].state = 1;
            }
            e = &cache[i];
            break;
        }
    }
    if (!e || e->state != 1 || ((uintptr_t)e->w % 16u) != 0u) return 0;
    if (ds41_v41_matmul_bf16(e->w, (const float *)in->ptr, (float *)out->ptr, outputs, width, rows)) return 0;
    static int logged = 0;
    if (!logged) { logged = 1; fprintf(stderr, "ds4: ROCm V4.1 router on BF16 matrix cores active (rows=%u)\n", rows); }
    return 1;
}

/* Dense Q8_0 projections on the FP16 matrix cores (rocm/gufo_f16, see its
 * NOTICE.md). Same operand rounding as the WMMA row-tile kernels (FP16
 * activations and decoded weights, FP32 accumulation); only the summation
 * order differs. DS4_ROCM_V41_DENSE_F16=0 restores the row-tile kernels. */
extern "C" int ds41_v41_dense_f16(int type, const void *w, const void *x, void *out, uint32_t m,
                                  uint32_t n, uint32_t k, uint32_t groups, uint32_t io);
static int v41_dense_f16_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_ROCM_V41_DENSE_F16");
        on = !(v && v[0] == '0') && ds4_rocm_is_gfx1151();
    }
    return on;
}
static uint32_t v41_dense_f16_min_rows(void) {
    static int rows = -1;
    if (rows < 0) {
        const char *v = getenv("DS4_ROCM_V41_DENSE_F16_MIN_ROWS");
        rows = v && v[0] ? atoi(v) : 32;
        if (rows < 2) rows = 2;
    }
    return (uint32_t)rows;
}
static void v41_dense_f16_log(const char *what, uint32_t rows) {
    static int logged = 0;
    if (!logged) {
        logged = 1;
        fprintf(stderr, "ds4: ROCm V4.1 dense FP16 projections active (%s rows=%u)\n", what, rows);
    }
}

/* Two Q8_0 matrices over the same input rows in one launch (the same results
 * as two ds41_v41_dense_f16 calls): shared-expert gate + up, and attention
 * query-A + KV. Outputs are not rounded; the caller applies its BF16 passes.
 * Returns 1 done, 0 not applicable. DS4_ROCM_V41_DENSE_PAIR=0 disables. */
extern "C" int ds41_v41_dense_f16_pair(int type, const void *w1, void *out1, uint32_t m1,
                                       const void *w2, void *out2, uint32_t m2, const void *x,
                                       uint32_t n, uint32_t k, uint32_t io);
extern "C" int ds4_gpu_dsv41_q8_projection_pair_batch(ds4_gpu_tensor *out1, ds4_gpu_tensor *out2,
        const void *model_map, uint64_t model_size, uint64_t offset1, uint64_t offset2,
        uint32_t width, uint32_t outputs1, uint32_t outputs2, uint32_t rows,
        const ds4_gpu_tensor *in) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_ROCM_V41_DENSE_PAIR");
        on = !(v && v[0] == '0');
    }
    uint64_t bytes1 = 0, bytes2 = 0;
    if (!on || g_quality_mode || !v41_dense_f16_enabled() || rows < v41_dense_f16_min_rows() ||
        rows > 2048u || !width || width % 256u || !outputs1 || outputs1 % 256u ||
        !outputs2 || outputs2 % 256u || !model_map ||
        !cuda_u64_mul3_checked(width / 32u, outputs1, 34u, &bytes1) ||
        !cuda_u64_mul3_checked(width / 32u, outputs2, 34u, &bytes2) ||
        !cuda_model_range_fits(model_size, offset1, bytes1) ||
        !cuda_model_range_fits(model_size, offset2, bytes2) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) ||
        !cuda_tensor_has_elems2(out1, outputs1, rows, 4u) ||
        !cuda_tensor_has_elems2(out2, outputs2, rows, 4u)) return 0;
    const void *w1 = cuda_model_range_ptr(model_map, offset1, bytes1, "V4.1 Q8 pair A");
    const void *w2 = cuda_model_range_ptr(model_map, offset2, bytes2, "V4.1 Q8 pair B");
    if (!w1 || !w2) return 0;
    if (ds41_v41_dense_f16_pair(8, w1, out1->ptr, outputs1, w2, out2->ptr, outputs2, in->ptr,
                                rows, width, 0u)) return 0;
    return 1;
}

extern "C" int ds4_gpu_dsv41_q8_projection_rows(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                                uint64_t weight_offset, uint32_t width, uint32_t outputs,
                                                uint32_t rows, const ds4_gpu_tensor *in) {
    uint64_t weight_bytes = 0;
    if (!width || width % 32u || !outputs || !rows || rows > 8192u || !model_map ||
        !cuda_u64_mul3_checked(width / 32u, outputs, 34u, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, rows, 4u) || !cuda_tensor_has_elems2(out, outputs, rows, 4u)) return 0;
    const unsigned char *weights = (const unsigned char *)cuda_model_range_ptr(
        model_map, weight_offset, weight_bytes, "V4.1 Q8 projection");
    if (!weights) return 0;
    /* Opt-in: round to bf16 inside the rowtile epilogue and report it with a
     * return value of 2 so the caller can drop its separate rounding pass. */
    const bool want_bf16 = ds4_q8_bf16_fused() && g_q8_round_request != 0;
    int rounded_in_kernel = 0;
    if (!g_quality_mode && rows >= v41_dense_f16_min_rows() && rows <= 2048u &&
        width % 256u == 0u && outputs % 256u == 0u && v41_dense_f16_enabled()) {
        if (ds41_v41_dense_f16(8, weights, in->ptr, out->ptr, outputs, rows, width, 1u, 0u)) return 0;
        v41_dense_f16_log("q8 projection", rows);
        return 1;
    }
    if (rows == 1u && ds4_rocm_is_gfx1151()) {
        v41_q8b4_prefetch_publish();
        v41_q8b4_log_shape("rows1", width, outputs, (uint64_t)(width / 32u) * 34u, rows);
        static int decode_round = -1;
        if (decode_round < 0) {
            const char *v = getenv("DS4_ROCM_V41_Q8_DECODE_ROUND");
            decode_round = (v && v[0] == '0') ? 0 : 1;
        }
        if (decode_round && g_q8_round_request != 0) {
            v41_q8_f32_blocks4_kernel<true><<<(outputs + 7u) / 8u, 256u>>>(
                (float *)out->ptr, weights, (const float *)in->ptr,
                width, outputs, (uint64_t)(width / 32u) * 34u);
            rounded_in_kernel = 1;
        } else
        v41_q8_f32_blocks4_kernel<<<(outputs + 7u) / 8u, 256u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            width, outputs, (uint64_t)(width / 32u) * 34u);
    } else if (rows <= 6u && ds4_rocm_is_gfx1151()) {
        return v41_q8_f32_tiny_rows_launch<1u>((float *)out->ptr, weights,
            (const float *)in->ptr, width, outputs, rows,
            (uint64_t)(width / 32u) * 34u);
    } else if (!g_quality_mode && width == 1280u && (outputs == 32768u || outputs == 16384u) && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Query-B, including contiguous two-rank weight slices.
         * This numerical path rounds activations and decoded Q8 weights to
         * F16 before F32 accumulation; quality mode retains the F32 path. */
        if (want_bf16) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 16u, true><<<dim3(outputs / 256u, (rows + 63u) / 64u), 512u>>>(
                (float *)out->ptr, weights, (const float *)in->ptr,
                rows, width, outputs, UINT64_C(40) * 34u);
            rounded_in_kernel = 1;
        } else {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 16u><<<dim3(outputs / 256u, (rows + 63u) / 64u), 512u>>>(
            (float *)out->ptr, weights, (const float *)in->ptr,
            rows, width, outputs, UINT64_C(40) * 34u);
        }
    } else if (!g_quality_mode && rows >= 32u && rows <= 2048u && rows != 128u && ds4_rocm_is_gfx1151() &&
               ((width == 5120u && (outputs == 512u || outputs == 1280u || outputs == 2304u)) ||
                (width == 2304u && outputs == 5120u))) {
        /* Query-A, KV and shared-expert projections on prefill/replay tiles.
         * The shared-F32 path remains faster for the bounded 128-row tile.
         * A two-host, full-oracle sweep selected the 256-row/pad-8 tile for
         * the bulk of this topology.  The 32-row input-limited shape favors
         * 128 rows/pad-8, while Query-A crosses back to the 128-row tile at
         * the final 1984-row grid step. */
        if (width == 5120u && rows <= 32u) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u, 8u>
                <<<dim3((outputs + 127u) / 128u, (rows + 63u) / 64u), 256u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        } else if (width == 5120u && outputs == 512u && rows >= 1984u) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u>
                <<<dim3((outputs + 127u) / 128u, (rows + 63u) / 64u), 256u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        } else {
            if (want_bf16) {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 8u, true><<<dim3((outputs + 255u) / 256u, (rows + 63u) / 64u), 512u>>>(
                (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
            rounded_in_kernel = 1;
        } else {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<256u, 16u, 8u>
                <<<dim3((outputs + 255u) / 256u, (rows + 63u) / 64u), 512u>>>(
                    (float *)out->ptr, weights, (const float *)in->ptr,
                    rows, width, outputs, (uint64_t)(width / 32u) * 34u);
        }
        }
    } else if (width == 1280u && outputs == 32768u && rows >= 32u && rows <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Reuse sixteen query-B activation rows with the same F32 lane
         * accumulation and wave reduction; keep the existing block tile. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 16u, 8u);
    } else if (rows >= 32u && ds4_rocm_is_gfx1151()) {
        /* Reuse eight F32 activation rows without changing each lane's block
         * accumulation or wave reduction. No F16 cast or expanded weights. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, weights, (const float *)in->ptr,
            width / 32u, outputs, rows, (uint64_t)(width / 32u) * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3((unsigned)(((uint64_t)outputs + 7u) / 8u), rows), 256>>>(
            (float *)out->ptr, weights, (const float *)in->ptr, width, outputs, rows, width / 32u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 F32-input Q8 projection")) return 0;
    return rounded_in_kernel ? 2 : 1;
}

extern "C" int ds4_gpu_dsv41_shared_gate_up_swiglu_bf16_rows(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint32_t width, uint32_t outputs, const ds4_gpu_tensor *in,
        float clamp) {
    uint64_t weight_bytes = 0;
    if (!gate || !up || !mid || !model_map || !in || !width || width % 32u ||
        !outputs || !ds4_rocm_is_gfx1151() ||
        !cuda_u64_mul3_checked(width / 32u, outputs, 34u, &weight_bytes) ||
        !cuda_model_range_fits(model_size, gate_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, weight_bytes) ||
        !cuda_tensor_has_elems2(in, width, 1u, 4u) ||
        !cuda_tensor_has_elems2(gate, outputs, 1u, 4u) ||
        !cuda_tensor_has_elems2(up, outputs, 1u, 4u) ||
        !cuda_tensor_has_elems2(mid, outputs, 1u, 4u)) return 0;
    const unsigned char *gw = (const unsigned char *)cuda_model_range_ptr(
        model_map, gate_offset, weight_bytes, "V4.1 shared gate Q8");
    const unsigned char *uw = (const unsigned char *)cuda_model_range_ptr(
        model_map, up_offset, weight_bytes, "V4.1 shared up Q8");
    if (!gw || !uw) return 0;
    v41_q8_f32_pair_swiglu_bf16_kernel<<<(outputs + 7u) / 8u, 256u>>>(
        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
        gw, uw, (const float *)in->ptr, width, outputs,
        (uint64_t)(width / 32u) * 34u, clamp);
    return cuda_ok(cudaGetLastError(), "V4.1 shared gate/up pair");
}

/* V4.1 grouped output-A: retain physical token strides while using the
 * existing F16-operand/F32-accumulator WMMA body on bulk prefill rows. */
template <uint32_t M_TILE, uint32_t WARPS, uint32_t GROUPS = 8u>
__launch_bounds__(WARPS * 32u, 1)
__global__ static void v41_grouped_q8_f32_wmma_rowtile_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint32_t n_tokens,
        uint32_t in_dim,
        uint32_t out_dim,
        uint64_t row_bytes) {
    const uint32_t group = (uint32_t)blockIdx.z;
    w += (uint64_t)group * out_dim * row_bytes;
    x += (uint64_t)group * in_dim;
    out += (uint64_t)group * out_dim;
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

    __shared__ _Float16 lds_x[N_TILE * K_TILE];

    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        for (uint32_t j = tid * 2u; j < N_TILE * K_TILE; j += blockDim.x * 2u) {
            const uint32_t nt = j >> 5u;
            const uint32_t kk = j & 31u;
            const uint32_t tok = block_n + nt;
            half2 xv = __floats2half2_rn(0.0f, 0.0f);
            if (tok < n_tokens) {
                const float2 f = *(const float2 *)(x + (uint64_t)tok * (GROUPS * 4096u) + bi * 32u + kk);
                xv = __floats2half2_rn(f.x, f.y);
            }
            *(half2 *)(lds_x + j) = xv;
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
            const _Float16 *xb = lds_x + nt * K_TILE;
            const ds4_q8_half16_t b0 = *(const ds4_q8_half16_t *)(xb);
            const ds4_q8_half16_t b1 = *(const ds4_q8_half16_t *)(xb + 16u);
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
            if (row < out_dim) out[(uint64_t)tok * (GROUPS * 1024u) + row] = acc[j];
        }
    }
}

/* Scalar grouped projection with packed four-weight loads and F32 activations. */
template <bool ROUND = false>
__global__ static void v41_grouped_q8_f32_blocks4_kernel(
        float *out, const unsigned char *w, const float *x, int K, int M, int G) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= M * G) return;
    const uint32_t token = blockIdx.y;
    const float *in = x + ((size_t)token * G + row / M) * K;
    out += (size_t)token * M * G;
    float acc = 0;
    for (int b = lane / 8; b < K / 32; b += 4) {
        const unsigned char *p = w + ((size_t) row * (K / 32) + b) * 34;
        float d = __half2float(* (const __half *) p), v = 0;
        int j = (lane & 7) * 4;
#pragma unroll
        for (int k = 0; k < 4; k++) v += (float) ((const int8_t *) (p + 2)) [j + k] * in[b * 32 + j + k];
        acc += d * v;
    }
    acc = warp_sum_f32(acc);
    if (!lane) out[row] = ROUND ? v41_bf16(acc) : acc;   /* ROUND = the following bf16 pass */
}

static bool v41_attn_low_m256_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        /* Default-on after two paired 2K/8K/32K A/B cycles; opt out for old tile. */
        cached = getenv("DS4_ROCM_V41_DISABLE_ATTN_LOW_M256") != NULL ? 0 : 1;
    }
    return cached;
}

static bool v41_attn_out_pad8_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        /* Default-on after the 2K/8K/32K paired A/B; opt out for the old tile. */
        cached = getenv("DS4_ROCM_V41_DISABLE_ATTN_OUT_PAD8") != NULL ? 0 : 1;
    }
    return cached;
}

extern "C" int ds4_gpu_dsv41_attention_output_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    const int leaf = ds4_leaf_profile_on() && n_tokens == 1u;
    if (leaf) (void)ds4_gpu_v41_leaf_begin();
    const uint64_t a_bytes = UINT64_C(8192) * 128u * 34u, b_bytes = UINT64_C(5120) * 256u * 34u;
    if (!model_map || !n_tokens || !cuda_model_range_fits(model_size, out_a_offset, a_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, b_bytes) ||
        !cuda_tensor_has_elems2(heads, n_tokens, 32768u, 4u) ||
        !cuda_tensor_has_elems2(low, n_tokens, 8192u, 4u) || !cuda_tensor_has_elems2(out, n_tokens, 5120u, 4u)) return 0;
    const unsigned char *a = (const unsigned char *)cuda_model_range_ptr(model_map, out_a_offset, a_bytes, "V4.1 attn_out_a");
    const unsigned char *b = (const unsigned char *)cuda_model_range_ptr(model_map, out_b_offset, b_bytes, "V4.1 attn_out_b");
    if (!a || !b) return 0;
    if (n_tokens >= 2u && n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        if (!v41_q8_f32_tiny_rows_launch<8u>((float *)low->ptr, a,
                (const float *)heads->ptr, 4096u, 1024u, n_tokens,
                UINT64_C(128) * 34u) ||
            !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16) ||
            !v41_q8_f32_tiny_rows_launch<1u>((float *)out->ptr, b,
                (const float *)low->ptr, 8192u, 5120u, n_tokens,
                UINT64_C(256) * 34u)) return 0;
        return 1;
    }
    if (!g_quality_mode && n_tokens >= v41_dense_f16_min_rows() && n_tokens <= 2048u &&
        v41_dense_f16_enabled()) {
        /* Eight groups of 4096 -> 1024, the BF16 low boundary, then 8192 -> 5120. */
        if (ds41_v41_dense_f16(8, a, heads->ptr, low->ptr, 1024u, n_tokens, 4096u, 8u, 0u) ||
            !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16) ||
            ds41_v41_dense_f16(8, b, low->ptr, out->ptr, 5120u, n_tokens, 8192u, 1u, 0u)) return 0;
        v41_dense_f16_log("attention output", n_tokens);
        return 1;
    }
    bool low_rounded = false;
    if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
        static int fuse = -1;
        if (fuse < 0) { const char *v = getenv("DS4_ROCM_V41_NORM_FUSE"); fuse = !(v && v[0] == '0'); }
        if (fuse) {
            v41_grouped_q8_f32_blocks4_kernel<true><<<1024u, 256u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr, 4096, 1024, 8);
            low_rounded = true;
        } else
        v41_grouped_q8_f32_blocks4_kernel<<<1024u, 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr, 4096, 1024, 8);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Canonical eight groups of 4096 -> 1024, with physical F32 token
         * strides32768/8192. Keep the explicit BF16 low boundary below. */
        if (v41_attn_low_m256_enabled()) {
            static bool low_m256_logged = false;
            if (!low_m256_logged) {
                low_m256_logged = true;
                fprintf(stderr, "ds4: V4.1 attention low-A m256 tile active\n");
            }
            v41_grouped_q8_f32_wmma_rowtile_kernel<256u, 16u><<<dim3(4u, (n_tokens + 63u) / 64u, 8u), 512u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr,
                n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
        } else {
            v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u><<<dim3(8u, (n_tokens + 63u) / 64u, 8u), 256u>>>(
                (float *)low->ptr, a, (const float *)heads->ptr,
                n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
        }
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_grouped_q8_a_sharedx((float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 8u, 128u, 1024u, 128u * 34u, 8u, 8u, 8u);
    } else {
        grouped_q8_0_a_f32_batch_warp8_kernel<<<dim3(1024u, n_tokens), 256>>>((float *)low->ptr, a,
            (const float *)heads->ptr, 4096u, 1024u, 8u, n_tokens, 128u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 attention low projection") ||
        (!low_rounded && !ds4_gpu_dsv41_quantize(low, 8192u, n_tokens, DS4_V41_BF16))) return 0;
    if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
        /* The shared input is the same BF16-rounded output-A row above. */
        v41_q8b4_prefetch_publish();
        v41_q8b4_prefetch_publish();
    v41_q8b4_log_shape("attn_out_up", 8192u, 5120u, UINT64_C(256) * 34u, n_tokens);
    v41_q8_f32_blocks4_kernel<<<640u, 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            8192u, 5120u, UINT64_C(256) * 34u);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the BF16 boundary above and use the existing generic bulk
         * matrix path: F16-rounded operands with F32 accumulation. */
        if (v41_attn_out_pad8_enabled()) {
            static bool pad8_logged = false;
            if (!pad8_logged) {
                pad8_logged = true;
                fprintf(stderr, "ds4: V4.1 attention output-B pad8 tile active\n");
            }
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
                (float *)out->ptr, b, (const float *)low->ptr,
                n_tokens, 8192u, 5120u, UINT64_C(256) * 34u);
        } else {
            matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
                (float *)out->ptr, b, (const float *)low->ptr,
                n_tokens, 8192u, 5120u, UINT64_C(256) * 34u);
        }
    } else if (n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        /* Keep the existing BF16-rounded low rows and reuse sixteen tokens. */
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 16u, 8u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            256u, 5120u, n_tokens, 256u * 34u, 8u, 8u, 8u);
    } else {
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3(640u, n_tokens), 256>>>((float *)out->ptr, b,
            (const float *)low->ptr, 8192u, 5120u, n_tokens, 256u);
    }
    if (leaf) {
        (void)ds4_gpu_v41_leaf_end("attn_output_pair", (unsigned long long)(a_bytes + b_bytes));
        (void)ds4_gpu_v41_leaf_note("attn_output_a", (unsigned long long)a_bytes);
        (void)ds4_gpu_v41_leaf_note("attn_output_b", (unsigned long long)b_bytes);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 attention output projection");
}

extern "C" int ds4_gpu_dsv41_attention_output_tp_batch(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        const ds4_gpu_tensor *heads, uint32_t n_tokens, uint32_t tp_rank) {
    /* Heads and low rows are packed for this rank. Output-B keeps its
     * original 8192-column physical row stride while consuming 4096 columns. */
    const uint64_t a_bytes = UINT64_C(4096) * 128u * 34u;
    const uint64_t b_bytes = UINT64_C(5120) * 256u * 34u;
    if (tp_rank > 1u || !model_map || !n_tokens ||
        !cuda_model_range_fits(model_size, out_a_offset, 2u * a_bytes) ||
        !cuda_model_range_fits(model_size, out_b_offset, b_bytes) ||
        !cuda_tensor_has_elems2(heads, n_tokens, 16384u, 4u) ||
        !cuda_tensor_has_elems2(low, n_tokens, 4096u, 4u) ||
        !cuda_tensor_has_elems2(out, n_tokens, 5120u, 4u)) return 0;
    const unsigned char *a = (const unsigned char *)cuda_model_range_ptr(model_map,
        out_a_offset + tp_rank * a_bytes, a_bytes, "V4.1 TP attn_out_a");
    const unsigned char *b = (const unsigned char *)cuda_model_range_ptr(model_map,
        out_b_offset, b_bytes, "V4.1 TP attn_out_b");
    if (!a || !b) return 0;
    b += (uint64_t)tp_rank * 128u * 34u;
    if (n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        v41_grouped_q8_f32_blocks4_kernel<<<dim3(512u, n_tokens), 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr, 4096, 1024, 4);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        v41_grouped_q8_f32_wmma_rowtile_kernel<128u, 8u, 4u><<<dim3(8u, (n_tokens + 63u) / 64u, 4u), 256u>>>(
            (float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 4096u, 1024u, UINT64_C(128) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_grouped_q8_a_sharedx((float *)low->ptr, a, (const float *)heads->ptr,
            n_tokens, 4u, 128u, 1024u, 128u * 34u, 8u, 8u, 8u);
    } else {
        grouped_q8_0_a_f32_batch_warp8_kernel<<<dim3(512u, n_tokens), 256>>>(
            (float *)low->ptr, a, (const float *)heads->ptr,
            4096u, 1024u, 4u, n_tokens, 128u);
    }
    if (!cuda_ok(cudaGetLastError(), "V4.1 TP attention low projection") ||
        !ds4_gpu_dsv41_quantize(low, 4096u, n_tokens, DS4_V41_BF16)) return 0;
    if (n_tokens <= 6u && ds4_rocm_is_gfx1151()) {
        v41_q8b4_prefetch_publish();
        v41_q8b4_prefetch_publish();
    v41_q8b4_log_shape("attn_out_bt", 8192u, 5120u, UINT64_C(256) * 34u, n_tokens);
    v41_q8_f32_blocks4_kernel<<<dim3(640u, n_tokens), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            4096u, 5120u, UINT64_C(256) * 34u);
    } else if (!g_quality_mode && n_tokens >= 32u && n_tokens <= 2048u && ds4_rocm_is_gfx1151()) {
        matmul_q8_0_f32_batch_wmma_rowtile_kernel<128u, 8u><<<dim3(40u, (n_tokens + 63u) / 64u), 256u>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            n_tokens, 4096u, 5120u, UINT64_C(256) * 34u);
    } else if (n_tokens >= 32u && ds4_rocm_is_gfx1151()) {
        cuda_launch_q8_batch_sharedx((float *)out->ptr, b, (const float *)low->ptr,
            128u, 5120u, n_tokens, 256u * 34u, 8u, n_tokens <= 2048u ? 16u : 8u, 8u);
    } else {
        /* This scalar kernel accepts separate input length and weight stride;
         * its column guard excludes the unowned half of each physical row. */
        matmul_q8_0_f32_batch_warp8_kernel<<<dim3(640u, n_tokens), 256>>>(
            (float *)out->ptr, b, (const float *)low->ptr,
            4096u, 5120u, n_tokens, 256u);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 TP attention output projection");
}

/* Staged correctness reference. Validate global IDs before any pointer-table
 * lookup; a null table entry suppresses every unowned gate/up/down load.
 * The ordinary routed-MoE dispatch is untouched. */
extern "C" int ds4_gpu_dsv41_routed_moe_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *scratch,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t tp_rank) {
    constexpr uint32_t experts = 384u, owned = 192u, used = 6u;
    constexpr uint64_t gate_row = 1320u, down_row = 756u;
    constexpr uint64_t gate_expert = gate_row * 2304u, down_expert = down_row * 5120u;
    routed_moe_launch_plan plan;
    if (!g_deepseek41_model || tp_rank > 1u || !n_tokens || n_tokens > 2048u ||
        !routed_moe_build_plan(out, gate, up, mid, scratch, model_map, model_size,
            gate_offset, up_offset, down_offset, 16u, 10u, gate_expert, down_expert,
            5120u, 2304u, 5120u, selected, weights, experts, used, x, n_tokens, &plan)) return 0;
    const uint64_t pairs = (uint64_t)n_tokens * used;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (int32_t id : ids) if (id < 0 || (uint32_t)id >= experts) {
        fprintf(stderr, "ds4: V4.1 TP invalid global expert ID %d\n", id);
        return 0;
    }
    if (n_tokens >= 128u && !g_quality_mode && ds4_rocm_is_gfx1151()) {
        if (!ds4_gpu_dsv41_moe_tp_gate_up(gate, up, model_map, model_size,
                gate_offset, up_offset, selected, x, n_tokens, tp_rank)) return 0;
        const uint64_t count = pairs * 2304u;
        moe_swiglu_weighted_f32_kernel<<<(uint32_t)((count + 255u) / 256u), 256>>>(
            (float *)mid->ptr, (const float *)gate->ptr, (const float *)up->ptr,
            (const float *)weights->ptr, count, 2304u, 10.f);
        if (!cuda_ok(cudaGetLastError(), "V4.1 TP MMQ weighted activation") ||
            !ds4_gpu_synchronize()) return 0;
        /* Opt-in producer capture of the actual gate/up projections and the
         * input row the MMQ consumed, at the same sampled row as the mid/down
         * probe.  Settles whether the mid delta comes from the projection or
         * from a different input row. */
        const char *probe_row = getenv("DS4_ROCM_V41_DIAG_MOE_ROW");
        if (probe_row) {
            const uint32_t row = (uint32_t)strtoul(probe_row, nullptr, 10);
            if (row >= n_tokens) return 0;
            const uint64_t first_pair = (uint64_t)row * used;
            if (!v41_diag_moe_save("gate", (const float *)gate->ptr + first_pair * 2304u,
                                   (size_t)used * 2304u * sizeof(float)) ||
                !v41_diag_moe_save("up", (const float *)up->ptr + first_pair * 2304u,
                                   (size_t)used * 2304u * sizeof(float)) ||
                !v41_diag_moe_save("x", (const float *)x->ptr + (uint64_t)row * 5120u,
                                   5120u * sizeof(float))) return 0;
            fprintf(stderr, "ds4: V4.1 MoE projection probe row=%u tokens=%u\n", row, n_tokens);
        }
        return ds4_gpu_dsv41_moe_tp_down(out, scratch, mid, selected,
            model_map, model_size, down_offset, n_tokens, tp_rank);
    }
    const uint32_t first = tp_rank * owned;
    const char *g = cuda_model_range_ptr(model_map, gate_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 TP owned gate");
    const char *u = cuda_model_range_ptr(model_map, up_offset + first * gate_expert,
                                        owned * gate_expert, "V4.1 TP owned up");
    const char *d = cuda_model_range_ptr(model_map, down_offset + first * down_expert,
                                        owned * down_expert, "V4.1 TP owned down");
    if (!g || !u || !d) return 0;
    const char *tables[3][experts] = {};
    for (uint32_t i = 0; i < owned; ++i) {
        tables[0][first+i] = g + i * gate_expert;
        tables[1][first+i] = u + i * gate_expert;
        tables[2][first+i] = d + i * down_expert;
    }
    ds4_gpu_tensor *table = ds4_gpu_tensor_alloc(sizeof(tables));
    if (!table) return 0;
    const uint64_t mids = pairs * 2304u;
    int ok = ds4_gpu_tensor_write(table, 0, tables, sizeof(tables)) &&
        ds4_gpu_tensor_fill_f32(gate, 0.f, mids) &&
        ds4_gpu_tensor_fill_f32(up, 0.f, mids) &&
        ds4_gpu_tensor_fill_f32(mid, 0.f, mids);
    cuda_block_q8_K *xq = (cuda_block_q8_K *)scratch->ptr;
    if (ok) {
        q8_K_quantize_kernel<<<dim3(20u, n_tokens), 256u>>>(
            xq, (const float *)x->ptr, 5120u, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP input quantization");
    }
    const char *const *slots = (const char *const *)table->ptr;
    if (ok) {
        if (n_tokens == 1u && ds4_rocm_is_gfx1151()) {
            ds4_v41_launch_gate_up_wave_ptrs(
                ds4_rocm_v41_gate_waves(),
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                slots, slots + experts, xq, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, 0, gate_row, 20u, 2304u, used,
                1u, 0x3fu, 10.f);
        } else {
            moe_gate_up_mid_qwarp32_ptrs_kernel<<<dim3(18u, (uint32_t)pairs), 256u>>>(
                (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                slots, slots + experts, xq, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, gate_row, 20u, 2304u, used, 0x3fu, 10.f);
        }
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP owned gate/up");
    }
    if (ok) {
        if (n_tokens == 1u && !g_quality_mode && ds4_rocm_is_gfx1151()) {
            // Gate/up has consumed xq; reuse its scratch for the six mid rows.
            cuda_block_q8_K *midq = (cuda_block_q8_K *)scratch->ptr;
            q8_K_quantize_kernel<<<dim3(9u, used), 256u>>>(
                midq, (const float *)mid->ptr, 2304u, used);
            ok = cuda_ok(cudaGetLastError(), "V4.1 TP mid quantization");
            if (ok) {
                ds4_v41_launch_down_wave_ptrs(
                    ds4_rocm_v41_down_waves(),
                    (float *)out->ptr, slots + 2u * experts, midq,
                    (const int32_t *)selected->ptr, 0, down_row, 9u, 5120u, used);
                ok = cuda_ok(cudaGetLastError(), "V4.1 TP quantized wave down");
            }
        } else {
            moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<dim3(640u, n_tokens), 256u>>>(
                (float *)out->ptr, slots + 2u * experts, (const float *)mid->ptr,
                (const int32_t *)selected->ptr, n_tokens, 2304u, 5120u, down_row, used);
            ok = cuda_ok(cudaGetLastError(), "V4.1 TP owned down");
        }
    }
    /* The reference deliberately drains before releasing its pointer table.
     * Persistent tables and a queued service follow ownership qualification. */
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(table);
    return ok;
}

/* Staged ownership adapter for the unchanged qualified Q2 down dispatcher.
 * Pair order is stable within each local expert, as in its GPU sort. */
extern "C" int ds4_gpu_dsv41_moe_tp_down(
        ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *mid, const ds4_gpu_tensor *selected,
        const void *model_map, uint64_t model_size, uint64_t down_offset,
        uint32_t n_tokens, uint32_t tp_rank) {
    constexpr uint64_t expert_bytes = UINT64_C(5120) * 756u;
    const uint64_t pairs = (uint64_t)n_tokens * 6u, mids = pairs * 2304u;
    if (!g_deepseek41_model || !ds4_rocm_is_gfx1151() || tp_rank > 1u ||
        n_tokens < 2u || n_tokens > 2048u || !out || !scratch || !mid || !selected ||
        !model_map || out->bytes < (uint64_t)n_tokens * 5120u * sizeof(float) ||
        scratch->bytes < pairs * 5120u * sizeof(float) ||
        mid->bytes < mids * sizeof(float) || selected->bytes < pairs * sizeof(int32_t) ||
        down_offset > model_size || 384u * expert_bytes > model_size - down_offset) return 0;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (uint32_t row = 0; row < n_tokens; ++row) {
        for (uint32_t j = 0; j < 6u; ++j) {
            const int32_t id = ids[(size_t)row * 6u + j];
            if (id < 0 || id >= 384) return 0;
            for (uint32_t k = 0; k < j; ++k)
                if (id == ids[(size_t)row * 6u + k]) return 0;
        }
    }
    const uint32_t first = tp_rank * 192u;
    const char *d = cuda_model_range_ptr(model_map, down_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP owned bulk down");
    if (!d) return 0;
    /* Counts, offsets, spare hot-list storage, then original six-slot pair IDs. */
    constexpr uint32_t offsets_at = 192u, hot_at = 385u, pairs_at = 578u;
    std::vector<uint32_t> metadata(pairs_at + (size_t)pairs, 0u);
    uint32_t pos = 0;
    for (uint32_t e = 0; e < 192u; ++e) {
        metadata[offsets_at + e] = pos;
        for (uint32_t p = 0; p < pairs; ++p)
            if ((uint32_t)ids[p] == first + e) {
                metadata[pairs_at + pos++] = p;
                ++metadata[e];
            }
    }
    metadata[offsets_at + 192u] = pos;
    ds4_gpu_tensor *meta = ds4_gpu_tensor_alloc(metadata.size() * sizeof(uint32_t));
    ds4_gpu_tensor *mid_h = ds4_gpu_tensor_alloc(mids * sizeof(half));
    int ok = meta && mid_h;
    if (ok) ok = ds4_gpu_tensor_write(meta, 0, metadata.data(), metadata.size() * sizeof(uint32_t)) &&
        cuda_ok(cudaMemset(scratch->ptr, 0, pairs * 5120u * sizeof(half)),
                "V4.1 TP clear unowned down slots");
    if (ok) {
        f32_to_f16_kernel<<<(uint32_t)((mids + 255u) / 256u), 256>>>(
            (half *)mid_h->ptr, (const float *)mid->ptr, mids);
        ok = cuda_ok(cudaGetLastError(), "V4.1 TP down F16 mid");
    }
    if (ok) {
        uint32_t *m = (uint32_t *)meta->ptr;
        ok = routed_moe_q2_float_down_launch(out, scratch, mid,
            (const half *)mid_h->ptr, !g_quality_mode, d, m, m + offsets_at,
            m + pairs_at, m + hot_at, n_tokens, 192u, 6u, 2304u, 5120u,
            expert_bytes, 756u);
    }
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(mid_h);
    ds4_gpu_tensor_free(meta);
    return ok;
}

/* Isolated bulk operator: remap owned IDs to a contiguous192-expert table.
 * INT_MAX is a nonmatching sentinel in mm_ids_helper, so no unowned expert
 * contributes an assignment. Cleared output rows remain zero for those slots. */
extern "C" int ds4_gpu_dsv41_moe_tp_gate_up(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t tp_rank) {
    constexpr uint64_t expert_bytes = UINT64_C(2304) * 1320u;
    const uint64_t pairs = (uint64_t)n_tokens * 6u;
    const uint64_t output_bytes = pairs * 2304u * sizeof(float);
    if (!g_deepseek41_model || !ds4_rocm_is_gfx1151() || tp_rank > 1u ||
        n_tokens < 128u || n_tokens > 2048u || !gate || !up || !selected || !x ||
        !model_map || gate->bytes < output_bytes || up->bytes < output_bytes ||
        selected->bytes < pairs * sizeof(int32_t) ||
        x->bytes < (uint64_t)n_tokens * 5120u * sizeof(float) ||
        gate_offset > model_size || 384u * expert_bytes > model_size - gate_offset ||
        up_offset > model_size || 384u * expert_bytes > model_size - up_offset) return 0;
    std::vector<int32_t> ids((size_t)pairs);
    if (!ds4_gpu_tensor_read(selected, 0, ids.data(), pairs * sizeof(int32_t))) return 0;
    for (uint32_t row = 0; row < n_tokens; ++row) {
        for (uint32_t j = 0; j < 6u; ++j) {
            const int32_t id = ids[(size_t)row * 6u + j];
            if (id < 0 || id >= 384) return 0;
            for (uint32_t k = 0; k < j; ++k)
                if (id == ids[(size_t)row * 6u + k]) return 0;
        }
    }
    const uint32_t first = tp_rank * 192u;
    for (int32_t &id : ids)
        id = (uint32_t)id >= first && (uint32_t)id < first + 192u ? id - first : INT_MAX;
    const char *g = cuda_model_range_ptr(model_map, gate_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP MMQ owned gate");
    const char *u = cuda_model_range_ptr(model_map, up_offset + first * expert_bytes,
                                         192u * expert_bytes, "V4.1 TP MMQ owned up");
    if (!g || !u) return 0;
    ds4_gpu_tensor *local_ids = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    if (!local_ids) return 0;
    int ok = ds4_gpu_tensor_write(local_ids, 0, ids.data(), pairs * sizeof(int32_t)) &&
        ds4_gpu_tensor_fill_f32(gate, 0.f, pairs * 2304u) &&
        ds4_gpu_tensor_fill_f32(up, 0.f, pairs * 2304u) && ds4_mmq_init(0) == 0;
    if (ok) ok = ds4_mmq_iq2_xxs_moe_pair(g, u, (const float *)x->ptr,
        (const int32_t *)local_ids->ptr, (float *)gate->ptr, (float *)up->ptr,
        2304, 5120, (int)n_tokens, 192, 6, (cudaStream_t)0) == 0;
    if (!ds4_gpu_synchronize()) ok = 0;
    ds4_gpu_tensor_free(local_ids);
    return ok;
}

/* Average the HC copies entering each DSpark target layer directly into the
 * interleaved main-feature buffer. */
__global__ static void v41_hc_mean_kernel(const float *stream, float *out,
                                          uint32_t rows, uint32_t dim,
                                          uint32_t hc, uint32_t out_stride,
                                          uint32_t out_off) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = (uint32_t)(i / dim), col = (uint32_t)(i % dim);
    if (row >= rows) return;
    float acc = 0.0f;
    for (uint32_t c = 0; c < hc; c++)
        acc += stream[((uint64_t)row * hc + c) * dim + col];
    out[(uint64_t)row * out_stride + out_off + col] = acc / (float)hc;
}

extern "C" int ds4_gpu_dsv41_hc_mean(
        uint32_t rows, uint32_t dim, uint32_t hc,
        const ds4_gpu_tensor *stream, ds4_gpu_tensor *out,
        uint32_t out_stride, uint32_t out_off) {
    const uint64_t in_count = (uint64_t)rows * hc * dim;
    const uint64_t out_count = (uint64_t)rows * out_stride;
    if (!rows || !dim || !hc || out_stride < dim ||
        out_off > out_stride - dim || !stream || !out ||
        !stream->ptr || !out->ptr || in_count > stream->bytes / sizeof(float) ||
        out_count > out->bytes / sizeof(float)) return 0;
    v41_hc_mean_kernel<<<(unsigned)(((uint64_t)rows * dim + 255u) / 256u), 256>>>(
        (const float *)stream->ptr, (float *)out->ptr,
        rows, dim, hc, out_stride, out_off);
    return cuda_ok(cudaGetLastError(), "V4.1 stream mean");
}

__device__ static float v41_markov_value(const void *p, uint64_t i, int f16) {
    return f16 ? __half2float(((const half *)p)[i]) : ((const float *)p)[i];
}

__global__ static void v41_markov_transpose_f16_kernel(
        half *out, const half *in, uint32_t rows, uint32_t cols) {
    __shared__ half tile[32][33];
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t in_col = blockIdx.x * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t in_row = blockIdx.y * 32u + ty + j;
        if (in_col < cols && in_row < rows)
            tile[ty + j][tx] = in[(uint64_t)in_row * cols + in_col];
    }
    __syncthreads();
    const uint32_t out_col = blockIdx.y * 32u + tx;
#pragma unroll
    for (uint32_t j = 0; j < 32u; j += 8u) {
        const uint32_t out_row = blockIdx.x * 32u + ty + j;
        if (out_row < cols && out_col < rows)
            out[(uint64_t)out_row * rows + out_col] = tile[tx][ty + j];
    }
}

extern "C" int ds4_gpu_dsv41_markov_prepare_head(
        ds4_gpu_tensor *transposed, const void *model_map,
        uint64_t model_size, uint64_t head_offset,
        uint32_t vocab, uint32_t rank) {
    const uint64_t bytes = (uint64_t)vocab * rank * sizeof(half);
    if (!transposed || !transposed->ptr || transposed->bytes < bytes ||
        !model_map || !vocab || !rank ||
        !cuda_model_range_fits(model_size, head_offset, bytes)) return 0;
    const half *head = (const half *)cuda_model_range_ptr(
        model_map, head_offset, bytes, "V4.1 Markov head transpose");
    if (!head) return 0;
    v41_markov_transpose_f16_kernel<<<
        dim3((rank + 31u) / 32u, (vocab + 31u) / 32u), dim3(32u, 8u)>>>(
            (half *)transposed->ptr, head, vocab, rank);
    return cuda_ok(cudaGetLastError(), "V4.1 Markov head transpose");
}

__global__ static void v41_markov_part_kernel(
        const float *logits, const void *embed, const void *head,
        const half *head_t,
        const int *tokens, float *part_val, int *part_idx,
        uint32_t vocab, uint32_t rank, uint32_t step,
        uint32_t n_parts, int f16) {
    extern __shared__ float embedded[];
    const int prev = tokens[step];
    for (uint32_t r = threadIdx.x; r < rank; r += blockDim.x)
        embedded[r] = v41_markov_value(embed, (uint64_t)prev * rank + r, f16);
    __syncthreads();
    const float *lg = logits + (uint64_t)step * vocab;
    float best = -INFINITY;
    int best_id = -1;
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < vocab; v += n_parts * blockDim.x) {
        float score = lg[v];
        for (uint32_t r = 0; r < rank; r++)
            score = fmaf(head_t ? __half2float(head_t[(uint64_t)r * vocab + v]) :
                         v41_markov_value(head, (uint64_t)v * rank + r, f16),
                         embedded[r], score);
        if (score > best || (score == best && (best_id < 0 || (int)v < best_id))) {
            best = score;
            best_id = (int)v;
        }
    }
    __shared__ float values[256];
    __shared__ int ids[256];
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t k = 1; k < blockDim.x; k++)
            if (ids[k] >= 0 &&
                (values[k] > best ||
                 (values[k] == best && (best_id < 0 || ids[k] < best_id)))) {
                best = values[k];
                best_id = ids[k];
            }
        part_val[blockIdx.x] = best;
        part_idx[blockIdx.x] = best_id;
    }
}

__global__ static void v41_markov_final_kernel(
        const float *part_val, const int *part_idx, int *tokens,
        const float *x, const void *embed, const float *conf_proj,
        float *conf, uint32_t rank, uint32_t dim, uint32_t step,
        uint32_t n_parts, int f16) {
    __shared__ float values[256];
    __shared__ int ids[256];
    float best = -INFINITY;
    int best_id = -1;
    for (uint32_t p = threadIdx.x; p < n_parts; p += blockDim.x)
        if (part_idx[p] >= 0 &&
            (part_val[p] > best ||
             (part_val[p] == best && (best_id < 0 || part_idx[p] < best_id)))) {
            best = part_val[p];
            best_id = part_idx[p];
        }
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (uint32_t k = 1; k < blockDim.x; k++)
            if (ids[k] >= 0 &&
                (values[k] > best ||
                 (values[k] == best && (best_id < 0 || ids[k] < best_id)))) {
                best = values[k];
                best_id = ids[k];
            }
        tokens[step + 1u] = best_id < 0 ? 0 : best_id;
    }
    const int prev = tokens[step];
    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x)
        sum = fmaf(conf_proj[d], x[(uint64_t)step * dim + d], sum);
    for (uint32_t r = threadIdx.x; r < rank; r += blockDim.x)
        sum = fmaf(conf_proj[dim + r],
            v41_markov_value(embed, (uint64_t)prev * rank + r, f16), sum);
    values[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (uint32_t k = 0; k < blockDim.x; k++) total += values[k];
        conf[step] = total;
    }
}

extern "C" int ds4_gpu_dsv41_markov_chain(
        uint32_t block, uint32_t vocab, uint32_t rank, uint32_t dim,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t embed_offset, uint64_t head_offset, int f16,
        const ds4_gpu_tensor *transposed_head,
        const ds4_gpu_tensor *conf_proj, ds4_gpu_tensor *tokens,
        ds4_gpu_tensor *conf, ds4_gpu_tensor *parts, uint32_t n_parts) {
    const uint64_t weight_bytes = (uint64_t)vocab * rank * (f16 ? 2u : 4u);
    if (!block || !vocab || !rank || !dim || !n_parts || n_parts > 4096u ||
        !model_map || !logits || !x || !conf_proj || !tokens || !conf || !parts ||
        !logits->ptr || !x->ptr || !conf_proj->ptr || !tokens->ptr ||
        !conf->ptr || !parts->ptr ||
        (transposed_head && (!transposed_head->ptr || !f16 ||
            transposed_head->bytes < (uint64_t)vocab * rank * sizeof(half))) ||
        !cuda_model_range_fits(model_size, embed_offset, weight_bytes) ||
        !cuda_model_range_fits(model_size, head_offset, weight_bytes) ||
        (uint64_t)block * vocab > logits->bytes / sizeof(float) ||
        (uint64_t)block * dim > x->bytes / sizeof(float) ||
        (uint64_t)dim + rank > conf_proj->bytes / sizeof(float) ||
        (uint64_t)block + 1u > tokens->bytes / sizeof(int32_t) ||
        block > conf->bytes / sizeof(float) ||
        (uint64_t)n_parts * 2u > parts->bytes / sizeof(float)) return 0;
    const void *embed = cuda_model_range_ptr(model_map, embed_offset,
        weight_bytes, "V4.1 Markov embed");
    const void *head = cuda_model_range_ptr(model_map, head_offset,
        weight_bytes, "V4.1 Markov head");
    if (!embed || !head) return 0;
    float *part_val = (float *)parts->ptr;
    int *part_idx = (int *)(part_val + n_parts);
    const half *head_t = transposed_head ? (const half *)transposed_head->ptr : NULL;
    const uint32_t launch_parts = head_t && n_parts > (vocab + 255u) / 256u ?
        (vocab + 255u) / 256u : n_parts;
    for (uint32_t step = 0; step < block; step++) {
        v41_markov_part_kernel<<<launch_parts, 256, rank * sizeof(float)>>>(
            (const float *)logits->ptr, embed, head, head_t,
            (const int *)tokens->ptr, part_val, part_idx, vocab, rank, step,
            launch_parts, f16);
        v41_markov_final_kernel<<<1, 256>>>(part_val, part_idx,
            (int *)tokens->ptr, (const float *)x->ptr, embed,
            (const float *)conf_proj->ptr, (float *)conf->ptr,
            rank, dim, step, launch_parts, f16);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 Markov chain");
}

#endif

/* Pack F32 indexer keys (already FP4_E8M0-valued) into the 68-byte packed cache.
 * Purely additive: the F32 cache stays authoritative for every current reader. */
extern "C" int ds4_gpu_dsv41_index_fp4_pack(ds4_gpu_tensor *packed,
                                           const ds4_gpu_tensor *keys,
                                           uint32_t n_keys, uint32_t dst_first,
                                           uint32_t src_first) {
    if (!packed || !keys || n_keys == 0u) return 0;
    const uint64_t need = ((uint64_t)(dst_first + n_keys) * DS4_INDEX_FP4_KEY_BYTES + 15u) & ~(uint64_t)15u;
    if (packed->bytes < need ||
        keys->bytes < (uint64_t)(src_first + n_keys) * 128u * sizeof(float)) return 0;
    const float *src = (const float *)keys->ptr + (uint64_t)src_first * 128u;
    unsigned char *dst = (unsigned char *)packed->ptr +
        (uint64_t)dst_first * DS4_INDEX_FP4_KEY_BYTES;
    ds4_index_fp4_pack_kernel<<<n_keys, 128>>>(dst, src, n_keys);
    return cuda_ok(cudaGetLastError(), "V4.1 index FP4 pack");
}

extern "C" int ds4_gpu_dsv41_indexer_scores_one_fp4(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *packed_keys, uint32_t source_rows) {
    if (!source_rows || source_rows > INT32_MAX ||
        !cuda_tensor_has_f32(scores, source_rows) ||
        !cuda_tensor_has_f32(q, 32u * 128u) ||
        !cuda_tensor_has_f32(weights, 32u) ||
        packed_keys->bytes < (uint64_t)source_rows * DS4_INDEX_FP4_KEY_BYTES) return 0;
    v41_ix_dispatch_note(V41_IX_DISP_SCALAR_ONE, source_rows, 1u, source_rows - 1u, 1u);
    static int dcheck = -1;
    if (dcheck < 0) dcheck = getenv("DS4_ROCM_V41_INDEX_DECODE_CHECK") != NULL;
    static int decode_wmma = -1;
    if (decode_wmma < 0) {
        const char *v = getenv("DS4_ROCM_V41_INDEX_DECODE_WMMA");
        decode_wmma = (v && v[0] == '0') ? 0 : 1;
    }
    if (decode_wmma && v41_indexer_wmma_enabled() && ds4_rocm_is_gfx1151() && !dcheck) {
        if (!v41_indexer_decode_wmma_launch((float *)scores->ptr, (const float *)q->ptr,
                (const float *)weights->ptr, (const unsigned char *)packed_keys->ptr, source_rows)) return 0;
        return cuda_ok(cudaGetLastError(), "V4.1 WMMA decode indexer");
    }
    v41_indexer_scalar_warp8_fp4_kernel<<<(source_rows + 7u) / 8u, 256>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const unsigned char *)packed_keys->ptr, source_rows, source_rows - 1u, 1u);
    if (dcheck) {
        float *alt = NULL;
        if (cudaMalloc(&alt, (size_t)source_rows * 4u) == cudaSuccess &&
            v41_indexer_decode_wmma_launch(alt, (const float *)q->ptr, (const float *)weights->ptr,
                (const unsigned char *)packed_keys->ptr, source_rows)) {
            float *ha = (float *)malloc((size_t)source_rows * 4u), *hb = (float *)malloc((size_t)source_rows * 4u);
            if (ha && hb && cudaMemcpy(ha, scores->ptr, (size_t)source_rows * 4u, cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(hb, alt, (size_t)source_rows * 4u, cudaMemcpyDeviceToHost) == cudaSuccess) {
                unsigned bad = 0;
                for (uint32_t i = 0; i < source_rows; i++) bad += memcmp(&ha[i], &hb[i], 4) != 0;
                fprintf(stderr, "ds4: ixdecode check width=%u mismatches=%u\n", source_rows, bad);
            }
            free(ha); free(hb);
        }
        cudaFree(alt);
    }
    return cuda_ok(cudaGetLastError(), "V4.1 scalar warp indexer (fp4 keys)");
}

/* Diagnostic for gates: verify every packed key decodes back to the F32 cache bits. */
extern "C" int ds4_gpu_dsv41_index_fp4_verify(const ds4_gpu_tensor *packed,
                                             const ds4_gpu_tensor *keys, uint32_t n_keys) {
    if (!packed || !keys || n_keys == 0u) return 0;
    if (packed->bytes < (uint64_t)n_keys * DS4_INDEX_FP4_KEY_BYTES ||
        keys->bytes < (uint64_t)n_keys * 128u * sizeof(float)) return 0;
    ds4_gpu_tensor *host = ds4_gpu_tensor_alloc(sizeof(unsigned long long));
    if (!host) return 0;
    unsigned long long zero = 0;
    int ok = ds4_gpu_tensor_write(host, 0, &zero, sizeof zero);
    if (ok) {
        ds4_index_fp4_verify_kernel<<<64, 256>>>((unsigned long long *)host->ptr,
            (const unsigned char *)packed->ptr, (const float *)keys->ptr, n_keys);
        ok = cuda_ok(cudaGetLastError(), "V4.1 index FP4 verify");
    }
    unsigned long long bad = ~0ull;
    if (ok) ok = ds4_gpu_tensor_read(host, 0, &bad, sizeof bad);
    ds4_gpu_tensor_free(host);
    if (!ok) return -1;
    fprintf(stderr, "ds4: index fp4 verify keys=%u mismatches=%llu\n", n_keys, bad);
    return (int)(bad > 0x7fffffffull ? 0x7fffffff : bad);
}

/* Packed-key batch scorer. Mirrors the F32 dispatch exactly, including the wide
 * GEMM branch, which keeps using the F32 keys because it is a different reduction. */
extern "C" int ds4_gpu_dsv41_indexer_scores_batch_fp4(ds4_gpu_tensor *scores,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *packed,
        const ds4_gpu_tensor *keys, uint32_t source_rows, uint32_t rows, uint32_t start,
        uint32_t ratio) {
    if ((ratio != 1u && ratio != 2u) || !source_rows || !rows || rows > UINT32_MAX - start ||
        (start + rows) / ratio > source_rows || source_rows > INT32_MAX || rows > INT32_MAX ||
        !cuda_tensor_has_elems2(scores, source_rows, rows, 4u) ||
        !cuda_tensor_has_elems2(q, rows, 32u * 128u, 4u) ||
        !cuda_tensor_has_elems2(keys, source_rows, 128u, 4u) ||
        !cuda_tensor_has_elems2(weights, rows, 32u, 4u) ||
        packed->bytes < (uint64_t)source_rows * DS4_INDEX_FP4_KEY_BYTES) return 0;
    if (!g_quality_mode && rows >= 16u && v41_indexer_wmma_enabled() && ds4_rocm_is_gfx1151()) {
        v41_ix_dispatch_note(V41_IX_DISP_WIDE_GEMM, source_rows, rows, start, ratio);
        if (!v41_indexer_wmma_launch((float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
            (const float *)keys->ptr, (const unsigned char *)packed->ptr, source_rows, rows, start, ratio)) return 0;
        return cuda_ok(cudaGetLastError(), "V4.1 fused WMMA indexer (packed keys)");
    }
    if (!ds4_rocm_is_gfx1151()) return ds4_gpu_dsv41_indexer_scores_batch(
        scores, q, weights, keys, source_rows, rows, start, ratio);
    const uint32_t head_cap = v41_indexer_wide_head_enabled() ? 131072u : 65536u;
    if (!g_quality_mode && g_cublas_ready && rows >= 31u && rows <= 32u &&
        source_rows >= 16384u && source_rows <= head_cap &&
        (start + 1u) / ratio >= source_rows - source_rows / 8u) {
        v41_ix_dispatch_note(V41_IX_DISP_WIDE_GEMM, source_rows, rows, start, ratio);
        return v41_indexer_head_gemm((float *)scores->ptr, (const float *)q->ptr,
            (const float *)weights->ptr, (const float *)keys->ptr, source_rows, rows, start, ratio);
    }
    v41_ix_dispatch_note(V41_IX_DISP_WARP8_BATCH, source_rows, rows, start, ratio);
    v41_indexer_batch_warp8_fp4_kernel<<<dim3((source_rows + 7u) / 8u, rows), 256u>>>(
        (float *)scores->ptr, (const float *)q->ptr, (const float *)weights->ptr,
        (const unsigned char *)packed->ptr, source_rows, start, ratio);
    return cuda_ok(cudaGetLastError(), "V4.1 causal FP4 index scores (packed keys)");
}

/* Per-leaf GPU timing for the attention projections: hipEvents around the launch, so the
 * number is device time rather than the host-inflated stage marks. Gated by the caller. */
enum { DS4_LEAF_PROFILE_SLOTS = 32 };
static struct { char label[48]; double ms; unsigned long long calls, bytes; } g_leaf_prof[DS4_LEAF_PROFILE_SLOTS];
static int g_leaf_prof_n;
static cudaEvent_t g_leaf_ev[2];
static int g_leaf_ev_ready;

static void ds4_leaf_profile_print(void) {
    double tot = 0;
    for (int i = 0; i < g_leaf_prof_n; i++) tot += g_leaf_prof[i].ms;
    fprintf(stderr, "ds4: leaf profile (per-token total %.2f ms across %d leaves)\n", tot, g_leaf_prof_n);
    for (int i = 0; i < g_leaf_prof_n; i++) {
        const double gbs = g_leaf_prof[i].ms > 0 ?
            g_leaf_prof[i].bytes / 1e9 / (g_leaf_prof[i].ms / 1000.0) : 0;
        fprintf(stderr, "ds4:   %-34s %9.2f ms %6llu calls %9.1f MB total %6.1f GB/s\n",
                g_leaf_prof[i].label, g_leaf_prof[i].ms, g_leaf_prof[i].calls,
                g_leaf_prof[i].bytes / 1e6, gbs);
    }
}


extern "C" int ds4_gpu_v41_leaf_begin(void) {
    if (!g_leaf_ev_ready) {
        if (cudaEventCreate(&g_leaf_ev[0]) != cudaSuccess ||
            cudaEventCreate(&g_leaf_ev[1]) != cudaSuccess) return 0;
        g_leaf_ev_ready = 1;
        atexit(ds4_leaf_profile_print);
    }
    return cudaEventRecord(g_leaf_ev[0]) == cudaSuccess;
}

/* Group per-layer tensors (blk.17.attn_q_b.weight) under the leaf name so the
 * table holds leaves, not layers. */
static const char *ds4_leaf_canon(const char *label) {
    if (strncmp(label, "blk.", 4) != 0) return label;
    const char *dot = strchr(label + 4, '.');
    return dot ? dot + 1 : label;
}


/* Call and byte accounting for a leaf that shares a fused kernel: no independent
 * event timing exists, so only counts and bytes are recorded. */
/* Host-side span (no GPU events): used for enqueue/drain time, which is CPU blocking. */
extern "C" int ds4_gpu_v41_leaf_host_note(const char *label, double ms) {
    if (!g_leaf_ev_ready) { if (!ds4_gpu_v41_leaf_begin()) return 0; }
    for (int i = 0; i < g_leaf_prof_n; i++) {
        if (strncmp(g_leaf_prof[i].label, label, sizeof(g_leaf_prof[i].label)) == 0) {
            g_leaf_prof[i].ms += ms; g_leaf_prof[i].calls += 1;
            return 1;
        }
    }
    if (g_leaf_prof_n < DS4_LEAF_PROFILE_SLOTS) {
        snprintf(g_leaf_prof[g_leaf_prof_n].label, sizeof(g_leaf_prof[0].label), "%s", label);
        g_leaf_prof[g_leaf_prof_n].ms = ms;
        g_leaf_prof[g_leaf_prof_n].calls = 1;
        g_leaf_prof_n += 1;
    }
    return 1;
}

extern "C" int ds4_gpu_v41_leaf_note(const char *label, unsigned long long bytes) {
    if (!g_leaf_ev_ready) {
        if (!ds4_gpu_v41_leaf_begin()) return 0;
    }
    for (int i = 0; i < g_leaf_prof_n; i++) {
        if (strncmp(g_leaf_prof[i].label, label, sizeof(g_leaf_prof[i].label)) == 0) {
            g_leaf_prof[i].calls += 1;
            g_leaf_prof[i].bytes += bytes;
            return 1;
        }
    }
    if (g_leaf_prof_n < DS4_LEAF_PROFILE_SLOTS) {
        snprintf(g_leaf_prof[g_leaf_prof_n].label, sizeof(g_leaf_prof[0].label), "%s", label);
        g_leaf_prof[g_leaf_prof_n].ms = 0.0;
        g_leaf_prof[g_leaf_prof_n].calls = 1;
        g_leaf_prof[g_leaf_prof_n].bytes = bytes;
        g_leaf_prof_n += 1;
    }
    return 1;
}

extern "C" int ds4_gpu_v41_leaf_end(const char *label_in, unsigned long long bytes) {
    const char *label = ds4_leaf_canon(label_in);
    if (!g_leaf_ev_ready) return 0;
    if (cudaEventRecord(g_leaf_ev[1]) != cudaSuccess ||
        cudaEventSynchronize(g_leaf_ev[1]) != cudaSuccess) return 0;
    float ms = 0;
    if (cudaEventElapsedTime(&ms, g_leaf_ev[0], g_leaf_ev[1]) != cudaSuccess) return 0;
    for (int i = 0; i < g_leaf_prof_n; i++) {
        if (strncmp(g_leaf_prof[i].label, label, sizeof(g_leaf_prof[i].label)) == 0) {
            g_leaf_prof[i].ms += ms; g_leaf_prof[i].calls += 1;
            g_leaf_prof[i].bytes += bytes;
            return 1;
        }
    }
    if (g_leaf_prof_n < DS4_LEAF_PROFILE_SLOTS) {
        snprintf(g_leaf_prof[g_leaf_prof_n].label, sizeof(g_leaf_prof[0].label), "%s", label);
        g_leaf_prof[g_leaf_prof_n].ms = ms;
        g_leaf_prof[g_leaf_prof_n].calls = 1;
        g_leaf_prof[g_leaf_prof_n].bytes = bytes;
        g_leaf_prof_n += 1;
    }
    return 1;
}

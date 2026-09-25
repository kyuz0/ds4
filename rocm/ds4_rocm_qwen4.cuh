/* Qwen3.8 Flash Next on gfx1151. Scalar operators follow the independent
 * Qwen CPU contracts and the CUDA implementation. Expert prefill uses native
 * AMD wave32 WMMA with the GGUF IQ2_XXS/Q2_K/Q4_K/MXFP4 readers below.
 * Included by ds4_rocm.cu to share the established ROCm allocation lifetime. */

#include "../ds4_qwen4_vision.h"
#include <hip/hip_bfloat16.h>

namespace qwen4_rocm {

__device__ __forceinline__ float qwen_f16_to_f32(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}


__device__ __forceinline__ float sum(float x);
__device__ __forceinline__ float sigmoid(float x);
template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid = NULL, const uint8_t *signs = NULL);
__device__ float injection(const float *inj, unsigned hc, unsigned stream);

struct rope_args { float freq[32], scale; unsigned nrot; };
static float rope_freq[32];
static float rope_scale = 1;
static bool rope_set;

static rope_args rope(unsigned nrot, float base) {
    rope_args r = {};
    r.nrot = nrot;
    r.scale = rope_set ? rope_scale : 1;
    for (unsigned i = 0; i < nrot / 2; i++)
        r.freq[i] = rope_set ? rope_freq[i] : powf(base, -2.0f * i / nrot);
    return r;
}

__device__ void apply_rope(float *row, const uint32_t *pos, rope_args r) {
    const unsigned i = threadIdx.x;
    if (i < r.nrot / 2) {
        const float theta = (float)pos[i % 3] * r.freq[i];
        const float c = cosf(theta) * r.scale, s = sinf(theta) * r.scale;
        const float a = row[i], b = row[i + r.nrot / 2];
        row[i] = a * c - b * s;
        row[i + r.nrot / 2] = a * s + b * c;
    }
    __syncwarp();
}

__global__ void attn_prep(float *qout, float *gate, __half *kc, __half *vc,
        float *iqout, float *ikc, const float *qg, const float *kp, const float *vp,
        const float *iq, const float *ik, const uint32_t *pos3,
        const float *gq, const float *gk, const float *giq,
        unsigned H, unsigned Hkv, unsigned D, unsigned Hi, unsigned Di,
        unsigned pos0, float eps, rope_args rp) {
    const unsigned slot = blockIdx.x, t = blockIdx.y, pos = pos0 + t, lane = threadIdx.x;
    __shared__ float row[256];
    if (slot == H + Hkv + Hi) {
        for (unsigned i = lane; i < Di; i += 32) ikc[(uint64_t)pos * Di + i] = ik[(uint64_t)t * Di + i];
        return;
    }
    const bool isq = slot < H, isk = !isq && slot < H + Hkv;
    const unsigned h = isq ? slot : isk ? slot - H : slot - H - Hkv;
    const unsigned dim = isq || isk ? D : Di;
    const float *src = isq ? qg + ((uint64_t)t * H + h) * 2 * D :
                      isk ? kp + ((uint64_t)t * Hkv + h) * D : iq + ((uint64_t)t * Hi + h) * Di;
    const float *gamma = isq ? gq : isk ? gk : giq;
    const unsigned npt = dim / 32;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) { const float v = src[lane * npt + i]; ss += v * v; }
    const float inv = rsqrtf(sum(ss) / dim + eps);
    for (unsigned i = lane; i < dim; i += 32) row[i] = src[i] * inv * gamma[i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)pos * 4, rp);
    for (unsigned i = lane; i < dim; i += 32) {
        if (isq) {
            qout[((uint64_t)t * H + h) * D + i] = row[i];
            gate[((uint64_t)t * H + h) * D + i] = src[D + i];
        } else if (isk) {
            kc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(row[i]);
            vc[((uint64_t)pos * Hkv + h) * D + i] = __float2half_rn(vp[((uint64_t)t * Hkv + h) * D + i]);
        } else iqout[((uint64_t)t * Hi + h) * Di + i] = row[i];
    }
}

__global__ void block_key(__half *out, const float *ik, const uint32_t *pos3,
        const float *gamma, unsigned block0, unsigned ratio, unsigned D, float eps, rope_args rp) {
    const unsigned b = block0 + blockIdx.x, lane = threadIdx.x, npt = D / 32;
    __shared__ float row[128];
    float ss = 0, v[4];
    for (unsigned i = 0; i < npt; i++) {
        float a = 0;
        for (unsigned t = 0; t < ratio; t++) a += ik[((uint64_t)b * ratio + t) * D + lane * npt + i];
        v[i] = a / ratio;
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) row[lane * npt + i] = v[i] * inv * gamma[lane * npt + i];
    __syncwarp();
    apply_rope(row, pos3 + (uint64_t)b * ratio * 4, rp);
    for (unsigned i = lane; i < D; i += 32) out[(uint64_t)b * D + i] = __float2half_rn(row[i]);
}

__global__ void idx_score(float *out, const float *q, const __half *key,
        unsigned N, unsigned H, unsigned D, unsigned pos0, unsigned ratio) {
    const unsigned b = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, lane = threadIdx.x & 31;
    if (b >= N) return;
    float score = 0;
    if (b >= (pos0 + t + 1) / ratio) score = -3e38f;
    else for (unsigned h = 0; h < H; h++) {
        float a = 0;
        for (unsigned i = lane; i < D; i += 32)
            a += q[((uint64_t)t * H + h) * D + i] * __half2float(key[(uint64_t)b * D + i]);
        score += fmaxf(sum(a), 0);
    }
    if (!lane) out[(uint64_t)t * N + b] = score;
}

__global__ void tile_max(unsigned *out, const float *scores, unsigned N, unsigned tiles) {
    const unsigned tile = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (tile >= tiles) return;
    unsigned v = 0;
    for (unsigned i = tile * 8; i < min(N, tile * 8 + 8); i++)
        v = max(v, __float_as_uint(fmaxf(scores[(uint64_t)t * N + i], 0)));
    out[(uint64_t)t * tiles + tile] = v;
}

/* Exact radix threshold and stable gather, with the same greater-than then
 * equal-score ordering as Metal. No context-dependent candidate truncation. */
__global__ void idx_select(int *out, const float *score, unsigned N, unsigned K) {
    const unsigned tid = threadIdx.x, t = blockIdx.x;
    __shared__ unsigned hist[256], gt[256], eq[256], threshold, need;
    const float *row = score + (uint64_t)t * N;
    if (!tid) { threshold = 0; need = K; }
    __syncthreads();
    for (unsigned pass = 0; pass < 4; pass++) {
        const unsigned shift = 24 - 8 * pass, mask = pass ? (0xffffffffu << (shift + 8)) : 0;
        hist[tid] = 0;
        __syncthreads();
        for (unsigned i = tid; i < N; i += 256) {
            const unsigned key = __float_as_uint(fmaxf(row[i], 0));
            if ((key & mask) == threshold) atomicAdd(hist + ((key >> shift) & 255), 1u);
        }
        __syncthreads();
        if (!tid) for (int d = 255; d >= 0; d--) {
            if (hist[d] >= need) { threshold |= (unsigned)d << shift; break; }
            need -= hist[d];
        }
        __syncthreads();
    }
    const unsigned chunk = (N + 255) / 256, begin = min(N, tid * chunk), end = min(N, begin + chunk);
    unsigned ng = 0, ne = 0;
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        ng += key > threshold; ne += key == threshold;
    }
    gt[tid] = ng; eq[tid] = ne;
    __syncthreads();
    if (!tid) {
        unsigned pg = 0, pe = 0;
        for (unsigned i = 0; i < 256; i++) {
            const unsigned g = gt[i], e = eq[i];
            gt[i] = pg; eq[i] = pe; pg += g; pe += e;
        }
    }
    __syncthreads();
    unsigned g = gt[tid], e = eq[tid];
    for (unsigned i = begin; i < end; i++) {
        const unsigned key = __float_as_uint(fmaxf(row[i], 0));
        if (key > threshold) out[(uint64_t)t * K + g++] = i;
        else if (key == threshold) { if (e < need) out[(uint64_t)t * K + K - need + e] = i; e++; }
    }
}

__global__ void idx_expand(int *out, unsigned *count, const int *blocks,
        unsigned K, unsigned ratio, unsigned pos0, unsigned stride) {
    const unsigned t = blockIdx.x, pos = pos0 + t, tail = (pos + 1) / ratio * ratio;
    for (unsigned i = threadIdx.x; i < K * ratio; i += blockDim.x)
        out[(uint64_t)t * stride + i] = blocks[(uint64_t)t * K + i / ratio] * ratio + i % ratio;
    for (unsigned i = tail + threadIdx.x; i <= pos; i += blockDim.x)
        out[(uint64_t)t * stride + K * ratio + i - tail] = i;
    if (!threadIdx.x) count[t] = K * ratio + pos + 1 - tail;
}

template<unsigned D>
__global__ void attention(float *out, float *partial, const float *q, const float *gate,
        const __half *kc, const __half *vc, const int *sel, const unsigned *counts,
        unsigned H, unsigned Hkv, unsigned pos0, unsigned stride, bool sparse,
        unsigned splits, unsigned per, float scale) {
    const unsigned h = blockIdx.x * 4 + threadIdx.x / 32, t = blockIdx.y, split = blockIdx.z;
    if (h >= H) return;
    const unsigned lane = threadIdx.x & 31, kh = h / (H / Hkv), n = sparse ? counts[t] : pos0 + t + 1;
    float qv[D / 32], acc[D / 32] = {}, m = -3e38f, denom = 0;
    for (unsigned i = 0; i < D / 32; i++) qv[i] = q[((uint64_t)t * H + h) * D + lane + 32 * i] * scale;
    for (unsigned j = split * per; j < min(n, (split + 1) * per); j++) {
        const unsigned p = sparse ? (unsigned)sel[(uint64_t)t * stride + j] : j;
        if (p > pos0 + t) continue;
        float score = 0;
        for (unsigned i = 0; i < D / 32; i++) score += qv[i] * __half2float(kc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        score = sum(score);
        const float nm = fmaxf(m, score), correction = expf(m - nm), w = expf(score - nm);
        denom = denom * correction + w;
        for (unsigned i = 0; i < D / 32; i++) acc[i] = acc[i] * correction + w * __half2float(vc[((uint64_t)p * Hkv + kh) * D + lane + 32 * i]);
        m = nm;
    }
    if (splits == 1) {
        for (unsigned i = 0; i < D / 32; i++) {
            const uint64_t p = ((uint64_t)t * H + h) * D + lane + 32 * i;
            out[p] = (denom > 0 ? acc[i] / denom : 0) * sigmoid(gate[p]);
        }
    } else {
        float *dst = partial + (((uint64_t)t * H + h) * splits + split) * (D + 2);
        if (!lane) { dst[0] = m; dst[1] = denom; }
        for (unsigned i = 0; i < D / 32; i++) dst[2 + lane + 32 * i] = acc[i];
    }
}

__global__ void attn_merge(float *out, const float *partial, const float *gate,
        unsigned H, unsigned D, unsigned splits) {
    const unsigned h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    const float *p = partial + ((uint64_t)t * H + h) * splits * (D + 2);
    float m = -3e38f, denom = 0, acc = 0;
    for (unsigned s = 0; s < splits; s++) m = fmaxf(m, p[s * (D + 2)]);
    for (unsigned s = 0; s < splits; s++) {
        const float *row = p + s * (D + 2);
        const float w = row[1] > 0 ? expf(row[0] - m) : 0;
        denom += row[1] * w; acc += row[2 + d] * w;
    }
    const uint64_t i = ((uint64_t)t * H + h) * D + d;
    out[i] = (denom > 0 ? acc / denom : 0) * sigmoid(gate[i]);
}

static bool tensor(const ds4_gpu_tensor *t, uint64_t bytes) {
    return t && t->ptr && bytes <= t->bytes;
}

static uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
    case 0: return n * 4;
    case 1: case 30: return n * 2;
    case 2: return n % 32 ? 0 : n / 32 * 18;
    case 8: return n % 32 ? 0 : n / 32 * 34;
    case 39: return n % 32 ? 0 : n / 32 * 17;
    case 10: return n % 256 ? 0 : n / 256 * 84;
    case 12: return n % 256 ? 0 : n / 256 * 144;
    case 16: return n % 256 ? 0 : n / 256 * 66;
    default: return 0;
    }
}

static const char *weight(const void *map, uint64_t size, uint64_t off, uint64_t bytes) {
    if (!map || !bytes || off > size || bytes > size - off) return NULL;
    // Adjacent tensors may share pages. Use cached device ranges or staged
    // copies; overlapping partial host registrations can invalidate HIP copies.
    return cuda_model_range_ptr(map, off, bytes, "Qwen weights", false);
}

static int launched(void) { return cuda_ok(cudaGetLastError(), "Qwen kernel"); }

#if defined(__gfx1151__)
template<int MASK>
__device__ __forceinline__ float qwen_dpp_xor(float x) {
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(x), 0x160 | MASK, 0xf, 0xf, true));
}
#endif

__device__ __forceinline__ float sum(float x) {
#if defined(__gfx1151__)
    x += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(x), __float_as_int(x),
        0x76543210, 0xFEDCBA98, true, false));
    x += qwen_dpp_xor<8>(x);
    x += qwen_dpp_xor<4>(x);
    x += qwen_dpp_xor<2>(x);
    x += qwen_dpp_xor<1>(x);
#else
    for (int d = 16; d; d >>= 1) x += __shfl_xor(x, d, 32);
#endif
    return x;
}

__device__ __forceinline__ float block_sum(float x, float *shared) {
    x = sum(x);
    if (!(threadIdx.x & 31)) shared[threadIdx.x / 32] = x;
    __syncthreads();
    float result = 0;
    for (unsigned i = 0; i < blockDim.x / 32; i++) result += shared[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }
__device__ __forceinline__ float softplus(float x) {
    return x > 20 ? x : x < -20 ? expf(x) : log1pf(expf(x));
}

/* These readers preserve the GGUF values, including padded Q2_K down rows.
 * Templates remove unused formats from each matrix kernel. */
template<unsigned TYPE>
__device__ __forceinline__ float value(const char *row, unsigned i,
        const uint64_t *grid_table = NULL, const uint8_t *sign_table = NULL) {
    if (TYPE == 0) return ((const float *)row)[i];
    if (TYPE == 1) return __half2float(((const __half *)row)[i]);
    if (TYPE == 30) return (float)((const hip_bfloat16 *)row)[i];
    if (TYPE == 8) {
        const char *b = row + (i / 32) * 34;
        return __half2float(*(const __half *)b) * (float)((const int8_t *)b)[2 + i % 32];
    }
    if (TYPE == 2) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 18;
        return __half2float(*(const __half *)b) *
            (float)((int)((b[2 + i % 16] >> (4 * (i % 32 / 16))) & 15) - 8);
    }
    if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row + (i / 32) * 17;
        const unsigned q = (b[1 + i % 16] >> (4 * (i % 32 / 16))) & 15;
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0] << 23);
        return (q & 8 ? -levels[q & 7] : levels[q & 7]) * scale;
    }
    if (TYPE == 10) {
        const cuda_block_q2_K *b = (const cuda_block_q2_K *)row + i / 256;
        const unsigned j = i % 256, sc = b->scales[j / 16];
        const unsigned q = (b->qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
        return qwen_f16_to_f32(b->d) * (sc & 15) * q - qwen_f16_to_f32(b->dmin) * (sc >> 4);
    }
    if (TYPE == 12) {
        const cuda_block_q4_K *b = (const cuda_block_q4_K *)row + i / 256;
        const unsigned j = i % 256, group = j / 32;
        unsigned sc, mn;
        if (group < 4) { sc = b->scales[group] & 63; mn = b->scales[group + 4] & 63; }
        else {
            sc = (b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4);
            mn = (b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4);
        }
        const unsigned q = (b->qs[j / 64 * 32 + j % 32] >> (4 * (group & 1))) & 15;
        return qwen_f16_to_f32(b->d) * sc * q - qwen_f16_to_f32(b->dmin) * mn;
    }
    if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row + i / 256;
        const unsigned j = i % 256, group = j / 32, sub = j % 32 / 8;
        const uint16_t *p = b->qs + group * 4;
        const unsigned grid_ids = (unsigned)p[0] | ((unsigned)p[1] << 16);
        const unsigned signs_scale = (unsigned)p[2] | ((unsigned)p[3] << 16);
        const unsigned gi = (grid_ids >> (8 * sub)) & 255, si = (signs_scale >> (7 * sub)) & 127;
        const uint64_t grid = grid_table ? grid_table[gi] : cuda_iq2xxs_grid[gi];
        const unsigned signs = sign_table ? sign_table[si] : cuda_ksigns_iq2xs[si];
        float v = (float)((grid >> (8 * (j & 7))) & 255);
        if (signs & (1u << (j & 7))) v = -v;
        return qwen_f16_to_f32(b->d) * (.5f + (signs_scale >> 28)) * .25f * v;
    }
    return 0;
}

__device__ __forceinline__ float scalar(const char *row, unsigned i, unsigned type) {
    switch (type) {
    case 0: return value<0>(row, i);
    case 1: return value<1>(row, i);
    case 2: return value<2>(row, i);
    case 8: return value<8>(row, i);
    case 30: return value<30>(row, i);
    case 39: return value<39>(row, i);
    default: return 0;
    }
}

template<unsigned TYPE>
__device__ __forceinline__ float4 value4(const char *row, unsigned i,
        const uint64_t *grid_table, const uint8_t *sign_table) {
    float4 out;
    float *v = (float *)&out;
    if (TYPE == 1) {
        const uint2 bits = *(const uint2 *)(row+i*2);
        const float2 a = __half22float2(*(__half2 *)&bits.x), b = __half22float2(*(__half2 *)&bits.y);
        out = make_float4(a.x,a.y,b.x,b.y);
    } else if (TYPE == 0) {
        out = *(const float4 *)(row+i*4);
    } else if (TYPE == 8) {
        const char *b = row+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        out = make_float4((float)(int8_t)q[0]*scale,(float)(int8_t)(q[0]>>8)*scale,
                         (float)(int8_t)q[1]*scale,(float)(int8_t)(q[1]>>8)*scale);
    } else if (TYPE == 39) {
        const uint8_t *b = (const uint8_t *)row+(i/32)*17;
        const float scale = b[0] == 0 ? 0x1p-127f : __uint_as_float((unsigned)b[0]<<23);
        unsigned packed;
        memcpy(&packed,b+1+i%16,4);
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            const unsigned q = (packed>>(k*8+(i%32/16)*4))&15, mag = q&7;
            const float level = mag < 2 ? .5f*mag : __uint_as_float(((mag/2+126)<<23)|((mag&1)<<22));
            v[k] = (q&8 ? -level : level)*scale;
        }
    } else if (TYPE == 16) {
        const cuda_block_iq2_xxs *b = (const cuda_block_iq2_xxs *)row+i/256;
        const unsigned j = i%256, sub = (j%32)/8;
        const uint16_t *p = b->qs+(j/32)*4;
        const unsigned ids = (unsigned)p[0]|((unsigned)p[1]<<16);
        const unsigned ss = (unsigned)p[2]|((unsigned)p[3]<<16);
        const uint64_t grid = grid_table[(ids>>(8*sub))&255];
        const unsigned signs = sign_table[(ss>>(7*sub))&127];
        const float scale = qwen_f16_to_f32(b->d)*(.5f+(ss>>28))*.25f;
        /* Apply four signs before conversion. IQ2's nonzero magnitudes keep
         * the packed negations from carrying into neighboring bytes. */
        const unsigned offset = j&7;
        const unsigned bits = (((signs>>offset)&15)*0x00204081u)&0x01010101u;
        const unsigned packed = (((unsigned)(grid>>(8*offset)))^(bits*255u))+bits;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) {
            v[k] = scale*(float)(int8_t)(packed>>(8*k));
        }
    } else if (TYPE == 10 || TYPE == 12) {
        const unsigned j = i%256;
        unsigned sc, mn, qs, shift;
        float d, dm;
        if (TYPE == 10) {
            const cuda_block_q2_K *b = (const cuda_block_q2_K *)row+i/256;
            const unsigned s = b->scales[j/16];
            sc = s&15; mn = s>>4;
            d = qwen_f16_to_f32(b->d); dm = qwen_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/128)*32+j%32);
            shift = 2*((j%128)/32);
        } else {
            const cuda_block_q4_K *b = (const cuda_block_q4_K *)row+i/256;
            const unsigned group = j/32;
            if (group < 4) { sc = b->scales[group]&63; mn = b->scales[group+4]&63; }
            else {
                sc = (b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
                mn = (b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
            }
            d = qwen_f16_to_f32(b->d); dm = qwen_f16_to_f32(b->dmin);
            qs = *(const unsigned *)(b->qs+(j/64)*32+j%32);
            shift = 4*(group&1);
        }
        const float scale = d*sc, offset = dm*mn;
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = scale*((qs>>(8*k+shift))&(TYPE == 10 ? 3 : 15))-offset;
    } else {
        #pragma unroll
        for (unsigned k = 0; k < 4; k++) v[k] = value<TYPE>(row,i+k,grid_table,sign_table);
    }
    return out;
}

static uint64_t expert_row_bytes(unsigned type, unsigned K) {
    if (type == 10 && K % 32) return 0;
    return row_bytes(type, type == 10 ? ((uint64_t)K + 255) / 256 * 256 : K);
}

__global__ void router(int *selected, float *weights, const float *logits,
        const float *x, const char *gate, float *shared_gate,
        unsigned NE, unsigned NS, unsigned K, unsigned type) {
    const unsigned t = blockIdx.x, tid = threadIdx.x;
    __shared__ float p[512], maxima[256], red[32];
    __shared__ unsigned ids[256];
    float mx = -FLT_MAX;
    for (unsigned e = tid; e < NE; e += 256) mx = fmaxf(mx, logits[(uint64_t)t * NE + e]);
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    mx = maxima[0];
    float ps = 0;
    for (unsigned e = tid; e < NE; e += 256) { p[e] = expf(logits[(uint64_t)t * NE + e] - mx); ps += p[e]; }
    const float total = block_sum(ps, red);
    for (unsigned e = tid; e < NE; e += 256) p[e] /= total;
    if (K) {
        float v = 0;
        for (unsigned i = tid; i < K; i += 256) v += scalar(gate, i, type) * x[(uint64_t)t * K + i];
        v = block_sum(v, red);
        if (!tid) shared_gate[t] = v;
    }
    __syncthreads();
    for (unsigned s = 0; s < NS; s++) {
        float best = -1;
        unsigned id = UINT_MAX;
        for (unsigned e = tid; e < NE; e += 256) if (p[e] > best) { best = p[e]; id = e; }
        maxima[tid] = best; ids[tid] = id;
        __syncthreads();
        for (unsigned stride = 128; stride; stride /= 2) {
            if (tid < stride && (maxima[tid + stride] > maxima[tid] ||
                (maxima[tid + stride] == maxima[tid] && ids[tid + stride] < ids[tid]))) {
                maxima[tid] = maxima[tid + stride]; ids[tid] = ids[tid + stride];
            }
            __syncthreads();
        }
        if (!tid) {
            selected[(uint64_t)t * NS + s] = ids[0];
            weights[(uint64_t)t * NS + s] = maxima[0];
            p[ids[0]] = -1;
        }
        __syncthreads();
    }
    if (tid < NS) {
        float denom = 0;
        for (unsigned s = 0; s < NS; s++) denom += weights[(uint64_t)t * NS + s];
        red[tid] = denom;
    }
    __syncthreads();
    if (tid < NS) weights[(uint64_t)t * NS + tid] /= red[tid];
}

template<unsigned TYPE, bool DOWN>
__global__ void moe_mv(float *out, const float *x, const int *selected,
        const char *w0, const char *w1, const char *sh0, const char *sh1,
        unsigned shared_type, unsigned NE, unsigned NS, unsigned K, unsigned M,
        uint64_t rb, uint64_t srb) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, slot = blockIdx.y, t = blockIdx.z;
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE == 16) {
        for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) grid_table[i] = cuda_iq2xxs_grid[i];
        for (unsigned i = threadIdx.x; i < 128; i += blockDim.x) sign_table[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    if (row >= M) return;
    const bool shared = slot == NS;
    const unsigned stride = NS + (shared_type != UINT_MAX);
    const uint64_t pair = (uint64_t)t * stride + slot;
    const float *xt = x + (DOWN ? pair : t) * K;
    float a = 0, b = 0;
    if (shared) {
        for (unsigned i = threadIdx.x & 31; i < K; i += 32) {
            a += scalar(sh0 + row * srb, i, shared_type) * xt[i];
            if (!DOWN) b += scalar(sh1 + row * srb, i, shared_type) * xt[i];
        }
        a = sum(a); b = sum(b);
    } else {
        const int e = selected[(uint64_t)t * NS + slot];
        if (e >= 0 && (unsigned)e < NE) {
            const uint64_t off = ((uint64_t)e * M + row) * rb;
            if ((TYPE == 16 || TYPE == 10 || TYPE == 12 || TYPE == 39) && !((uintptr_t)xt&15)) {
                for (unsigned i = (threadIdx.x&31)*4; i < K; i += 128) {
                    const float4 xv = *(const float4 *)(xt+i);
                    const float4 av = value4<TYPE>(w0+off,i,grid_table,sign_table);
                    a += av.x*xv.x; a += av.y*xv.y; a += av.z*xv.z; a += av.w*xv.w;
                    if (!DOWN) {
                        const float4 bv = value4<TYPE>(w1+off,i,grid_table,sign_table);
                        b += bv.x*xv.x; b += bv.y*xv.y; b += bv.z*xv.z; b += bv.w*xv.w;
                    }
                }
                a = sum(a); b = sum(b);
            } else {
                a = dot<TYPE>(w0+off,xt,K,grid_table,sign_table);
                if (!DOWN) b = dot<TYPE>(w1+off,xt,K,grid_table,sign_table);
            }
        }
    }
    if (!(threadIdx.x & 31)) out[pair * M + row] = DOWN ? a : silu(a) * b;
}

template<unsigned TYPE,bool DOWN,unsigned OR,unsigned STEPS>
__global__ void moe_mv_reuse(float*out,const float*x,const int*selected,const char*w0,const char*w1,const char*sh0,const char*sh1,unsigned st,unsigned NE,unsigned NS,unsigned K,unsigned M,uint64_t rb,uint64_t srb,unsigned T){
 constexpr unsigned WAVES=1; constexpr bool TABLE=false, REUSE=true;
 const unsigned lane=threadIdx.x&31,row0=(blockIdx.x*WAVES+threadIdx.x/32)*OR;
 const unsigned slot=blockIdx.y%(NS+1),token=blockIdx.y/(NS+1),stride=NS+1;
 const bool shared=slot==NS;
 int e=shared?-1:selected[token*NS+slot];
 if constexpr(REUSE){
  if(shared&&token)return;
  if(!shared&&e>=0&&(unsigned)e<NE)for(unsigned t=0;t<token;t++)for(unsigned s=0;s<NS;s++)if(selected[t*NS+s]==e)return;
 }
 __shared__ uint64_t grid_table[TYPE==16&&TABLE?256:1];__shared__ uint8_t sign_table[TYPE==16&&TABLE?128:1];
 if constexpr(TYPE==16&&TABLE){for(unsigned i=threadIdx.x;i<256;i+=blockDim.x)grid_table[i]=cuda_iq2xxs_grid[i];for(unsigned i=threadIdx.x;i<128;i+=blockDim.x)sign_table[i]=cuda_ksigns_iq2xs[i];__syncthreads();}
 if(row0>=M)return;
 int pairs[3]={-1,-1,-1};pairs[token]=token*stride+slot;
 if constexpr(REUSE){for(unsigned t=token+1;t<T;t++){
  if(shared)pairs[t]=t*stride+NS;
  else if(e>=0&&(unsigned)e<NE)for(unsigned s=0;s<NS;s++)if(selected[t*NS+s]==e){pairs[t]=t*stride+s;break;}
 }}
 float a[OR][3]={},b[OR][3]={};
 if(shared){
  for(unsigned i=lane;i<K;i+=32){
   #pragma unroll
   for(unsigned j=0;j<OR;j++)if(row0+j<M){const float av=scalar(sh0+(uint64_t)(row0+j)*srb,i,st);float bv=0;if constexpr(!DOWN)bv=scalar(sh1+(uint64_t)(row0+j)*srb,i,st);
    #pragma unroll
    for(unsigned t=0;t<3;t++)if(pairs[t]>=0){float v=x[(uint64_t)(DOWN?pairs[t]:t)*K+i];a[j][t]=__fmaf_rn(av,v,a[j][t]);if constexpr(!DOWN)b[j][t]=__fmaf_rn(bv,v,b[j][t]);}
   }
  }
 }else if(e>=0&&(unsigned)e<NE){
  for(unsigned base=lane*4;base<K;base+=128*STEPS){
   float4 av[STEPS][OR],bv[STEPS][OR],xv[STEPS][3];
   #pragma unroll
   for(unsigned s=0;s<STEPS;s++)if(base+s*128<K){unsigned i=base+s*128;
    #pragma unroll
    for(unsigned t=0;t<3;t++)if(pairs[t]>=0)xv[s][t]=*(const float4*)(x+(uint64_t)(DOWN?pairs[t]:t)*K+i);
    #pragma unroll
    for(unsigned j=0;j<OR;j++)if(row0+j<M){uint64_t off=((uint64_t)e*M+row0+j)*rb;
     av[s][j]=value4<TYPE>(w0+off,i,TABLE?grid_table:cuda_iq2xxs_grid,TABLE?sign_table:cuda_ksigns_iq2xs);
     if constexpr(!DOWN)bv[s][j]=value4<TYPE>(w1+off,i,TABLE?grid_table:cuda_iq2xxs_grid,TABLE?sign_table:cuda_ksigns_iq2xs);
    }
   }
   #pragma unroll
   for(unsigned s=0;s<STEPS;s++)if(base+s*128<K){
    #pragma unroll
    for(unsigned j=0;j<OR;j++)if(row0+j<M){
     #pragma unroll
     for(unsigned t=0;t<3;t++)if(pairs[t]>=0){const float4 u=av[s][j],v=xv[s][t];a[j][t]=__fmaf_rn(u.x,v.x,a[j][t]);a[j][t]=__fmaf_rn(u.y,v.y,a[j][t]);a[j][t]=__fmaf_rn(u.z,v.z,a[j][t]);a[j][t]=__fmaf_rn(u.w,v.w,a[j][t]);
      if constexpr(!DOWN){const float4 u=bv[s][j];b[j][t]=__fmaf_rn(u.x,v.x,b[j][t]);b[j][t]=__fmaf_rn(u.y,v.y,b[j][t]);b[j][t]=__fmaf_rn(u.z,v.z,b[j][t]);b[j][t]=__fmaf_rn(u.w,v.w,b[j][t]);}
     }
    }
   }
  }
 }
 #pragma unroll
 for(unsigned j=0;j<OR;j++)if(row0+j<M){
  #pragma unroll
  for(unsigned t=0;t<3;t++)if(pairs[t]>=0){float u=sum(a[j][t]),v=sum(b[j][t]);if(!lane)out[(uint64_t)pairs[t]*M+row0+j]=DOWN?u:silu(u)*v;}
 }
}

static int moe_mv_dispatch(float *out, const float *x, const int *sel,
        const char *w0, const char *w1, const char *s0, const char *s1,
        unsigned type, unsigned st, unsigned NE, unsigned T, unsigned NS, unsigned K, unsigned M, bool down) {
    const uint64_t rb = expert_row_bytes(type, K), srb = row_bytes(st, K);
    if (ds4_rocm_is_gfx1151() && T >= 2 && T <= 3 && NS == 10 && NE == 512 && st == 8 && !((uintptr_t)x&15)) {
        if (!down && K == 2560 && M == 640 && (type == 16 || type == 12)) {
            const dim3 grouped(M,(NS+1)*T);
            if (type == 16) moe_mv_reuse<16,false,1,2><<<grouped,32,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,T);
            else moe_mv_reuse<12,false,1,2><<<grouped,32,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,T);
            return launched();
        }
        if (down && K == 640 && M == 2560 && (type == 10 || type == 39)) {
            const dim3 grouped((M+1)/2,(NS+1)*T);
            if (type == 10) moe_mv_reuse<10,true,2,2><<<grouped,32,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,T);
            else moe_mv_reuse<39,true,2,1><<<grouped,32,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb,T);
            return launched();
        }
    }
    const dim3 grid((M + 3) / 4, NS + (st != UINT_MAX), T);
#define QWEN_MOE(TYPE) case TYPE: \
    if (down) moe_mv<TYPE, true><<<grid,128,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb); \
    else moe_mv<TYPE, false><<<grid,128,0,0>>>(out,x,sel,w0,w1,s0,s1,st,NE,NS,K,M,rb,srb); break
    switch (type) {
    QWEN_MOE(0); QWEN_MOE(1); QWEN_MOE(2); QWEN_MOE(8); QWEN_MOE(10);
    QWEN_MOE(12); QWEN_MOE(16); QWEN_MOE(30); QWEN_MOE(39);
    default: return 0;
    }
#undef QWEN_MOE
    return launched();
}

__global__ void moe_reduce(float *out, float *R, const float *inj, const float *part,
        const float *weights, const float *gate, const float *shared,
        unsigned NS, unsigned stride, unsigned D, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    __shared__ float inject[4];
    if (threadIdx.x < hc) inject[threadIdx.x] = injection(inj + (uint64_t)t * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= D) return;
    float v = 0;
    for (unsigned s = 0; s < NS; s++) v += weights[(uint64_t)t * NS + s] * part[((uint64_t)t * stride + s) * D + d];
    if (gate) v += sigmoid(gate[t]) * (shared ? shared[(uint64_t)t * D + d] : part[((uint64_t)t * stride + NS) * D + d]);
    out[(uint64_t)t * D + d] = v;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)t * hc + s) * D + d] += inject[s] * v;
}

__global__ void expert_lists(int *lists, int *counts, const int *selected,
        unsigned pairs, unsigned NE, unsigned cap) {
    __shared__ unsigned counters[512];
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counters[e] = 0;
    __syncthreads();
    for (unsigned p = threadIdx.x; p < pairs; p += blockDim.x) {
        const unsigned e = (unsigned)selected[p];
        if (e < NE) {
            const unsigned i = atomicAdd(counters + e, 1u);
            if (i < cap) lists[(uint64_t)e * cap + i] = p;
        }
    }
    __syncthreads();
    for (unsigned e = threadIdx.x; e < NE; e += blockDim.x) counts[e] = min(counters[e], cap);
}

/* Tiles share dequantized weights across 16 tokens without requantizing the
 * activations. Expert counts determine the work; no expert list is truncated. */
template<unsigned TYPE, bool DOWN, bool EXPERT>
__global__ void matrix(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned T, unsigned NS, unsigned NO,
        unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    const unsigned r = threadIdx.x % 16, c = threadIdx.x / 16, e = blockIdx.y;
    const unsigned row = blockIdx.x * 16 + r;
    const unsigned count = EXPERT ? (unsigned)counts[e] : T;
    __shared__ float a[16][33], b[16][33], v[16][33];
    for (unsigned t0 = blockIdx.z * 16; t0 < count; t0 += gridDim.z * 16) {
        const unsigned item = t0 + c;
        const int pair = EXPERT && item < count ? lists[(uint64_t)e * cap + item] : (int)item;
        const unsigned tok = EXPERT ? pair / NS : item, slot = EXPERT ? pair % NS : 0;
        float acc = 0, up = 0;
        for (unsigned k0 = 0; k0 < K; k0 += 32) {
            for (unsigned j = threadIdx.x; j < 16 * 32; j += 256) {
                const unsigned rr = j / 32, kk = j % 32, global_row = blockIdx.x * 16 + rr;
                const unsigned logical = k0 + kk;
                const char *wr = w0 + ((uint64_t)(EXPERT ? e : 0) * M + global_row) * rb;
                a[rr][kk] = global_row < M && logical < K ? value<TYPE>(wr, logical) : 0;
                if (EXPERT && !DOWN) {
                    const char *ur = w1 + ((uint64_t)e * M + global_row) * rb;
                    b[rr][kk] = global_row < M && logical < K ? value<TYPE>(ur, logical) : 0;
                }
                const unsigned ti = t0 + rr;
                const int pp = EXPERT && ti < count ? lists[(uint64_t)e * cap + ti] : (int)ti;
                const unsigned tt = EXPERT ? pp / NS : ti, ss = EXPERT ? pp % NS : 0;
                const uint64_t xx = DOWN && EXPERT ? (uint64_t)tt * NO + ss : tt;
                v[rr][kk] = ti < count && logical < K ? x[xx * K + logical] : 0;
            }
            __syncthreads();
            #pragma unroll
            for (unsigned k = 0; k < 32; k++) {
                acc += a[r][k] * v[c][k];
                if (EXPERT && !DOWN) up += b[r][k] * v[c][k];
            }
            __syncthreads();
        }
        if (row < M && item < count) {
            const uint64_t op = EXPERT ? (uint64_t)tok * NO + slot : item;
            out[op * M + row] = EXPERT && !DOWN ? silu(acc) * up : acc;
        }
    }
}

// Build independent tile lists for experts with similar routed-token counts.
__global__ void expert_tiles_range(unsigned *prefix, const int *counts, unsigned NE,
        unsigned nt, unsigned lo, unsigned hi) {
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        const unsigned count = (unsigned)counts[e];
        if (count > lo && count <= hi) n += (count + nt - 1) / nt;
        prefix[e + 1] = n;
    }
}

__global__ void expert_tiles(unsigned *prefix, const int *counts, unsigned NE, unsigned nt) {
    unsigned n = 0;
    prefix[0] = 0;
    for (unsigned e = 0; e < NE; e++) {
        n += ((unsigned)counts[e]+nt-1)/nt;
        prefix[e+1] = n;
    }
}

/* A/B WMMA operands repeat across the two wave halves. Output element
 * i belongs to row 2*i+lane/16 and column lane%16. Keep the same rounded
 * operands as the production Qwen half-tile oracle, with FP32 accumulation. */
typedef _Float16 __attribute__((ext_vector_type(16))) half16;
typedef float __attribute__((ext_vector_type(8))) float8;

template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64, unsigned WAVES = 8>
__launch_bounds__(WAVES * 32)
__global__ void matrix_half_tile(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    static_assert(NR % 16 == 0 && WAVES % (NR / 16) == 0 &&
                  NT % (16 * (WAVES / (NR / 16))) == 0, "complete WMMA tiles");
    const unsigned tid=threadIdx.x, wave=tid/32, lane=tid%32, lane16=lane%16;
    const unsigned nr=(M+NR-1)/NR, job=blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e=0, end=NE;
    while (e<end) { const unsigned mid=(e+end)/2; if (tiles[mid+1]<=job) e=mid+1; else end=mid; }
    const unsigned count=counts[e], t0=(job-tiles[e])*NT, r0=(blockIdx.x%nr)*NR;
    const unsigned wr=wave%(NR/16), wc=wave/(NR/16);
    constexpr unsigned NC=NT/(WAVES/(NR/16));
    // Pad LDS rows for the Q2 and Q4 expert WMMA tiles.
    constexpr unsigned LD = 72;
    __shared__ _Float16 a[NR][LD], u[DOWN ? 1 : NR][LD], b[NT][LD];
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE==16) {
        for (unsigned i=tid;i<256;i+=WAVES*32) grid_table[i]=cuda_iq2xxs_grid[i];
        for (unsigned i=tid;i<128;i+=WAVES*32) sign_table[i]=cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    float8 acc[NC/16]={}, up[NC/16]={};
    for (unsigned k0=0;k0<K;k0+=64) {
        for (unsigned i=tid*4;i<NR*64;i+=WAVES*32*4) {
            const unsigned row=r0+i/64, k=k0+i%64;
            const uint64_t off=((uint64_t)e*M+row)*rb;
            float4 av = {}, uv = {};
            if (row<M && k+3<K) {
                av=value4<TYPE>(w0+off,k,grid_table,sign_table);
                if (!DOWN) uv=value4<TYPE>(w1+off,k,grid_table,sign_table);
            } else if (row<M) {
                #pragma unroll
                for (unsigned j=0;j<4;j++) if (k+j<K) {
                    ((float *)&av)[j]=value<TYPE>(w0+off,k+j,grid_table,sign_table);
                    if (!DOWN) ((float *)&uv)[j]=value<TYPE>(w1+off,k+j,grid_table,sign_table);
                }
            }
            #pragma unroll
            for (unsigned j=0;j<4;j++) {
                a[i/64][i%64+j]=(_Float16)((float *)&av)[j];
                if (!DOWN) u[i/64][i%64+j]=(_Float16)((float *)&uv)[j];
            }
        }
        for (unsigned i=tid;i<NT*64;i+=WAVES*32) {
            const unsigned item=t0+i/64, k=k0+i%64;
            const unsigned pair=item<count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row=DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            b[i/64][i%64]=(_Float16)(item<count && k<K ? x[row*K+k] : 0);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k=0;k<64;k+=16) {
            half16 av, uv;
            #pragma unroll
            for (unsigned j=0;j<16;j++) {
                av[j]=a[wr*16+lane16][k+j];
                if (!DOWN) uv[j]=u[wr*16+lane16][k+j];
            }
            #pragma unroll
            for (unsigned c=0;c<NC/16;c++) {
                half16 bv;
                #pragma unroll
                for (unsigned j=0;j<16;j++) bv[j]=b[wc*NC+c*16+lane16][k+j];
                acc[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,acc[c]);
                if (!DOWN) up[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(uv,bv,up[c]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned c=0;c<NC/16;c++) {
        const unsigned item=t0+wc*NC+c*16+lane16;
        if (item<count) {
            const unsigned pair=lists[(uint64_t)e*cap+item];
            #pragma unroll
            for (unsigned j=0;j<8;j++) {
                const unsigned row=r0+wr*16+2*j+lane/16;
                if (row<M) out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row]=DOWN ? acc[c][j] : silu(acc[c][j])*up[c][j];
            }
        }
    }
}

/* Stage the next gate block in registers while consuming the LDS tile. */
template<unsigned TYPE, bool DOWN, unsigned NT, unsigned NR = 64>
__launch_bounds__(256)
__global__ void matrix_half_tile_prefetch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, const unsigned *tiles, unsigned NE,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, uint64_t rb) {
    const unsigned tid=threadIdx.x, wave=tid/32, lane=tid%32, lane16=lane%16;
    const unsigned nr=(M+NR-1)/NR, job=blockIdx.x/nr;
    if (job >= tiles[NE]) return;
    unsigned e=0, end=NE;
    while (e<end) { const unsigned mid=(e+end)/2; if (tiles[mid+1]<=job) e=mid+1; else end=mid; }
    const unsigned count=counts[e], t0=(job-tiles[e])*NT, r0=(blockIdx.x%nr)*NR;
    const unsigned wr=wave%(NR/16), wc=wave/(NR/16);
    constexpr unsigned NC=NT/(8/(NR/16));
    // Pad LDS rows for the Q2 and Q4 expert WMMA tiles.
    constexpr unsigned LD = 72;
    __shared__ _Float16 a[NR][LD], u[DOWN ? 1 : NR][LD], b[NT][LD];
    __shared__ uint64_t grid_table[TYPE == 16 ? 256 : 1];
    __shared__ uint8_t sign_table[TYPE == 16 ? 128 : 1];
    if (TYPE==16) {
        for (unsigned i=tid;i<256;i+=256) grid_table[i]=cuda_iq2xxs_grid[i];
        for (unsigned i=tid;i<128;i+=256) sign_table[i]=cuda_ksigns_iq2xs[i];
        __syncthreads();
    }
    float8 acc[NC/16]={}, up[NC/16]={};
    constexpr unsigned NW = NR*64/(256*4), NX = NT*64/256;
    half16 next_a, next_u;
    half16 next_b[NX/16];
    static_assert(NW == 4 && NX%16 == 0, "prefetch tile shape");
    // Bootstrap iteration stores the first block. Later iterations prefetch
    // the next block into registers while WMMA consumes the shared tile.
    for (unsigned step=0;step<=K;step+=64) {
        const unsigned k0=step;
        if (step < K) {
        #pragma unroll
        for (unsigned slot=0;slot<NW;slot++) {
            const unsigned i=tid*4+slot*256*4;
            const unsigned row=r0+i/64, k=k0+i%64;
            const uint64_t off=((uint64_t)e*M+row)*rb;
            float4 av = {}, uv = {};
            if (row<M && k+3<K) {
                av=value4<TYPE>(w0+off,k,grid_table,sign_table);
                if (!DOWN) uv=value4<TYPE>(w1+off,k,grid_table,sign_table);
            } else if (row<M) {
                #pragma unroll
                for (unsigned j=0;j<4;j++) if (k+j<K) {
                    ((float *)&av)[j]=value<TYPE>(w0+off,k+j,grid_table,sign_table);
                    if (!DOWN) ((float *)&uv)[j]=value<TYPE>(w1+off,k+j,grid_table,sign_table);
                }
            }
            #pragma unroll
            for (unsigned j=0;j<4;j++) {
                next_a[slot*4+j]=(_Float16)((float *)&av)[j];
                if (!DOWN) next_u[slot*4+j]=(_Float16)((float *)&uv)[j];
            }
        }
        #pragma unroll
        for (unsigned slot=0;slot<NX;slot++) {
            const unsigned i=tid+slot*256;
            const unsigned item=t0+i/64, k=k0+i%64;
            const unsigned pair=item<count ? lists[(uint64_t)e*cap+item] : 0;
            const uint64_t row=DOWN ? (uint64_t)(pair/NS)*NO+pair%NS : pair/NS;
            next_b[slot/16][slot%16]=(_Float16)(item<count && k<K ? x[row*K+k] : 0);
        }
        }
        if (step) {
        #pragma unroll
        for (unsigned k=0;k<64;k+=16) {
            half16 av, uv;
            #pragma unroll
            for (unsigned j=0;j<16;j++) {
                av[j]=a[wr*16+lane16][k+j];
                if (!DOWN) uv[j]=u[wr*16+lane16][k+j];
            }
            #pragma unroll
            for (unsigned c=0;c<NC/16;c++) {
                half16 bv;
                #pragma unroll
                for (unsigned j=0;j<16;j++) bv[j]=b[wc*NC+c*16+lane16][k+j];
                acc[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,acc[c]);
                if (!DOWN) up[c]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(uv,bv,up[c]);
            }
        }
        }
        __syncthreads();
        if (step < K) {
            #pragma unroll
            for (unsigned slot=0;slot<NW;slot++) {
                const unsigned i=tid*4+slot*256*4;
                #pragma unroll
                for (unsigned j=0;j<4;j++) {
                    a[i/64][i%64+j]=next_a[slot*4+j];
                    if (!DOWN) u[i/64][i%64+j]=next_u[slot*4+j];
                }
            }
            #pragma unroll
            for (unsigned slot=0;slot<NX;slot++) {
                const unsigned i=tid+slot*256;
                b[i/64][i%64]=next_b[slot/16][slot%16];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned c=0;c<NC/16;c++) {
        const unsigned item=t0+wc*NC+c*16+lane16;
        if (item<count) {
            const unsigned pair=lists[(uint64_t)e*cap+item];
            #pragma unroll
            for (unsigned j=0;j<8;j++) {
                const unsigned row=r0+wr*16+2*j+lane/16;
                if (row<M) out[((uint64_t)(pair/NS)*NO+pair%NS)*M+row]=DOWN ? acc[c][j] : silu(acc[c][j])*up[c][j];
            }
        }
    }
}

static int matrix_dispatch(float *out, const float *x, const char *w0, const char *w1,
        const int *lists, const int *counts, unsigned type, unsigned NE, unsigned T,
        unsigned NS, unsigned NO, unsigned K, unsigned M, unsigned cap, bool down) {
    const dim3 grid((M + 15) / 16, lists ? NE : 1, std::min(8u, (T + 15) / 16));
    const uint64_t rb = expert_row_bytes(type, K);
        if (ds4_rocm_is_gfx1151() && lists && !g_quality_mode && (type == 16 || type == 10 || type == 12 || type == 39)) {
            // Keep bulk Q2 prefetch and wide down tiles; group the remaining
            // matrix work by actual expert occupancy instead of padding every
            // expert to the same tile width.
            const bool grouped = T > 256 &&
                (T < 2048 || (type == 12 && !down && T < 7168));
            const unsigned passes = grouped ? (down ? 3 : 4) : 1;
            for (unsigned pass = 0; pass < passes; pass++) {
                const unsigned nt = grouped ? 16u << pass :
                    T >= 2048 ? (down ? 32 : 128) : T <= 256 ? 16 : 32;
                const unsigned lo = !grouped || pass == 0 ? 0 :
                    pass == 1 ? 16 : pass == 2 ? 64 : 128;
                const unsigned hi = !grouped || pass + 1 == passes ? UINT_MAX :
                    pass == 0 ? 16 : pass == 1 ? 64 : 128;
                unsigned *tiles = (unsigned *)cuda_tmp_alloc(((uint64_t)NE+1)*4,"Qwen expert tiles");
                if (!tiles) return 0;
                if (grouped) expert_tiles_range<<<1,1,0,0>>>(tiles,counts,NE,nt,lo,hi);
                else expert_tiles<<<1,1,0,0>>>(tiles,counts,NE,nt);
                if (!launched()) return 0;
                /* Bulk down tiles reuse activations across 128 output rows. */
                const unsigned nr = down && (T >= 2048 || (nt == 64 && type != 10 && type != 16)) ? 128 : 64;
                const uint64_t jobs = std::min(((uint64_t)T*NS+nt-1)/nt+NE,
                    (uint64_t)NE * (((uint64_t)hi+nt-1)/nt));
                const uint64_t blocks = jobs*((M+nr-1)/nr);
                if (blocks > INT_MAX) return 0;
#define QWEN_HALF(TYPE, DOWN) \
                if (DOWN && T >= 2048) matrix_half_tile<TYPE,DOWN,32,128><<<blocks,256,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
                else if (nt == 128 && (TYPE == 16 || (TYPE == 12 && T >= 7168)) && !DOWN && !(K%64)) matrix_half_tile_prefetch<TYPE,DOWN,128,64><<<blocks,256,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
                else if (nt == 128) matrix_half_tile<TYPE,DOWN,128,64><<<blocks,256,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
                else if (nt == 64) matrix_half_tile<TYPE,DOWN,64,((!DOWN || TYPE == 10 || TYPE == 16) ? 64 : 128)><<<blocks,256,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
                else if (nt == 16) matrix_half_tile<TYPE,DOWN,16,64,4><<<blocks,128,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb); \
                else matrix_half_tile<TYPE,DOWN,32><<<blocks,256,0,0>>>(out,x,w0,w1,lists,counts,tiles,NE,NS,NO,K,M,cap,rb)
#define QWEN_HALF_TYPE(TYPE) case TYPE: if (down) { QWEN_HALF(TYPE,true); } else { QWEN_HALF(TYPE,false); } break
                switch (type) { QWEN_HALF_TYPE(16); QWEN_HALF_TYPE(10); QWEN_HALF_TYPE(12); QWEN_HALF_TYPE(39); }
#undef QWEN_HALF_TYPE
#undef QWEN_HALF
                if (!launched()) return 0;
            }
            return launched();
        }
#define QWEN_MM(TYPE) case TYPE: \
    if (!lists) matrix<TYPE,false,false><<<grid,256,0,0>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else if (down) matrix<TYPE,true,true><<<grid,256,0,0>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); \
    else matrix<TYPE,false,true><<<grid,256,0,0>>>(out,x,w0,w1,lists,counts,T,NS,NO,K,M,cap,rb); break
    switch (type) {
    QWEN_MM(0); QWEN_MM(1); QWEN_MM(2); QWEN_MM(8); QWEN_MM(10);
    QWEN_MM(12); QWEN_MM(16); QWEN_MM(30); QWEN_MM(39);
    default: return 0;
    }
#undef QWEN_MM
    return launched();
}

template<unsigned TYPE>
__device__ __forceinline__ float dot(const char *row, const float *x, unsigned n,
        const uint64_t *grid, const uint8_t *signs) {
    float acc = 0;
    if (TYPE == 1 && !(n%4) && !((uintptr_t)x&15) && !((uintptr_t)row&7)) {
        for (unsigned i = (threadIdx.x&31)*4; i < n; i += 128) {
            const float4 w = value4<TYPE>(row,i,grid,signs), v = *(const float4 *)(x+i);
            acc += w.x*v.x; acc += w.y*v.y; acc += w.z*v.z; acc += w.w*v.w;
        }
    } else {
        for (unsigned i = threadIdx.x & 31; i < n; i += 32) acc += value<TYPE>(row, i, grid, signs) * x[i];
    }
    return sum(acc);
}

template<unsigned TYPE>
__global__ void matvec(float *out, const char *w, const float *x,
                       unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    if (row >= M) return;
    const float v = dot<TYPE>(w + row * stride, x + (uint64_t)tok * K, K);
    if (!(threadIdx.x & 31)) out[(uint64_t)tok * M + row] = v;
}

template<unsigned TYPE, unsigned ROWS>
__global__ void matvec_rows(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    float a[ROWS] = {};
    const char *wr = w+(uint64_t)row*stride;
    if (TYPE == 1 && !(K%4) && !((uintptr_t)x&15) && !((uintptr_t)wr&7)) {
        for (unsigned i = lane*4; i < K; i += 128) {
            const float4 v = value4<TYPE>(wr,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                const float4 xv = *(const float4 *)(x+(uint64_t)t*K+i);
                a[t] += v.x*xv.x; a[t] += v.y*xv.y; a[t] += v.z*xv.z; a[t] += v.w*xv.w;
            }
        }
    } else {
        for (unsigned i = lane; i < K; i += 32) {
            const float v = value<TYPE>(wr,i);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) a[t] += v*x[(uint64_t)t*K+i];
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(a[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS>
__global__ void matvec_q8(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned row = blockIdx.x*4+threadIdx.x/32, lane = threadIdx.x&31;
    if (row >= M) return;
    const char *wr = w+(uint64_t)row*stride;
    float acc[ROWS] = {};
    for (unsigned i = lane*4; i < K; i += 128) {
        const char *b = wr+(i/32)*34;
        const float scale = __half2float(*(const __half *)b);
        const uint16_t *q = (const uint16_t *)(b+2+i%32);
        const unsigned q01 = q[0], q23 = q[1];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float4 v = *(const float4 *)(x+(uint64_t)t*K+i);
            float part = (float)(int8_t)q01*v.x;
            part += (float)(int8_t)(q01>>8)*v.y;
            part += (float)(int8_t)q23*v.z;
            part += (float)(int8_t)(q23>>8)*v.w;
            acc[t] += part*scale;
        }
    }
    #pragma unroll
    for (unsigned t = 0; t < ROWS; t++) if (t < T) {
        const float v = sum(acc[t]);
        if (!lane) out[(uint64_t)t*M+row] = v;
    }
}

template<unsigned ROWS, unsigned OUTROWS, unsigned WAVES>
__global__ void matvec_f16_input_first(float *out, const char *w, const float *x,
        unsigned T, unsigned K, unsigned M, uint64_t stride) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned row0 = (blockIdx.x*WAVES + threadIdx.x/32)*OUTROWS;
    if (row0 >= M) return;
    float acc[OUTROWS][ROWS] = {};
    for (unsigned i = lane*4; i < K; i += 128) {
        float4 input[ROWS];
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) input[t] = *(const float4 *)(x+(uint64_t)t*K+i);
        #pragma unroll
        for (unsigned j = 0; j < OUTROWS; j++) if (row0+j < M) {
            const float4 v = value4<1>(w+(uint64_t)(row0+j)*stride,i,NULL,NULL);
            #pragma unroll
            for (unsigned t = 0; t < ROWS; t++) if (t < T) {
                acc[j][t] += v.x*input[t].x; acc[j][t] += v.y*input[t].y;
                acc[j][t] += v.z*input[t].z; acc[j][t] += v.w*input[t].w;
            }
        }
    }
    #pragma unroll
    for (unsigned j = 0; j < OUTROWS; j++) if (row0+j < M) {
        #pragma unroll
        for (unsigned t = 0; t < ROWS; t++) if (t < T) {
            const float v = sum(acc[j][t]);
            if (!lane) out[(uint64_t)t*M+row0+j] = v;
        }
    }
}

template<unsigned TYPE,unsigned ROWS,unsigned WAVES,bool LDS=false,unsigned OUTROWS=1,unsigned STEPS=2>
__global__ void matvec_prefetch(float* out,const char* w,const float* x,unsigned T,unsigned K,unsigned M,uint64_t stride){
 extern __shared__ float sx[];
 if constexpr(LDS){for(unsigned i=threadIdx.x;i<T*K;i+=blockDim.x)sx[i]=x[i];__syncthreads();x=sx;}
 const unsigned lane=threadIdx.x&31,row0=(blockIdx.x*WAVES+threadIdx.x/32)*OUTROWS;
 if(row0>=M)return;
 float acc[OUTROWS][ROWS]={};
 for(unsigned base=lane*4;base<K;base+=128*STEPS){
  float4 xv[STEPS][ROWS]; float4 wv[STEPS][OUTROWS]; float scales[STEPS][OUTROWS];
  #pragma unroll
  for(unsigned s=0;s<STEPS;s++){
   const unsigned i=base+128*s;
   if(i<K){
    #pragma unroll
    for(unsigned t=0;t<ROWS;t++)if(t<T)xv[s][t]=*(const float4*)(x+(uint64_t)t*K+i);
    #pragma unroll
    for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M){
     const char* wr=w+(uint64_t)(row0+j)*stride;
if constexpr(TYPE == 8) {
     const char*b=wr+(i/32)*34;scales[s][j]=__half2float(*(const __half*)b);
     const uint16_t*q=(const uint16_t*)(b+2+i%32);unsigned a=q[0],c=q[1];
     wv[s][j]=make_float4(__fmul_rn((float)(int8_t)a,scales[s][j]),__fmul_rn((float)(int8_t)(a>>8),scales[s][j]),__fmul_rn((float)(int8_t)c,scales[s][j]),__fmul_rn((float)(int8_t)(c>>8),scales[s][j]));
} else {
     wv[s][j]=value4<1>(wr,i,NULL,NULL);
}
    }
   }
  }
  #pragma unroll
  for(unsigned s=0;s<STEPS;s++)if(base+128*s<K){
   #pragma unroll
   for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M){
    #pragma unroll
    for(unsigned t=0;t<ROWS;t++)if(t<T){
     const float4 v=wv[s][j],a=xv[s][t];
if constexpr(TYPE == 8) {
     acc[j][t]=__fmaf_rn(v.x,a.x,acc[j][t]);acc[j][t]=__fmaf_rn(v.y,a.y,acc[j][t]);acc[j][t]=__fmaf_rn(v.z,a.z,acc[j][t]);acc[j][t]=__fmaf_rn(v.w,a.w,acc[j][t]);
} else {
     acc[j][t]+=v.x*a.x;acc[j][t]+=v.y*a.y;acc[j][t]+=v.z*a.z;acc[j][t]+=v.w*a.w;
}
    }
   }
  }
 }
 #pragma unroll
 for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M){
  #pragma unroll
  for(unsigned t=0;t<ROWS;t++)if(t<T){float v=sum(acc[j][t]);if(!lane)out[(uint64_t)t*M+row0+j]=v;}
 }
}

template<unsigned ROWS,unsigned WAVES,unsigned OUTROWS=1,unsigned STEPS=4,bool LDS=false>
__global__ void matvec_f32_prefetch(float*out,const char*w,const float*x,unsigned T,unsigned K,unsigned M,uint64_t stride){
 extern __shared__ float sx[];
 if constexpr(LDS){for(unsigned i=threadIdx.x;i<T*K;i+=blockDim.x)sx[i]=x[i];__syncthreads();x=sx;}
 const unsigned lane=threadIdx.x&31,row0=(blockIdx.x*WAVES+threadIdx.x/32)*OUTROWS;
 if(row0>=M)return;
 float acc[OUTROWS][ROWS]={};
 for(unsigned base=lane;base<K;base+=32*STEPS){
  float xv[STEPS][ROWS],wv[STEPS][OUTROWS];
  #pragma unroll
  for(unsigned s=0;s<STEPS;s++)if(base+32*s<K){
   #pragma unroll
   for(unsigned t=0;t<ROWS;t++)if(t<T)xv[s][t]=x[(uint64_t)t*K+base+32*s];
   #pragma unroll
   for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M)wv[s][j]=value<0>(w+(uint64_t)(row0+j)*stride,base+32*s);
  }
  #pragma unroll
  for(unsigned s=0;s<STEPS;s++)if(base+32*s<K){
   #pragma unroll
   for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M){
    #pragma unroll
    for(unsigned t=0;t<ROWS;t++)if(t<T)acc[j][t]=__fmaf_rn(wv[s][j],xv[s][t],acc[j][t]);
   }
  }
 }
 #pragma unroll
 for(unsigned j=0;j<OUTROWS;j++)if(row0+j<M){
  #pragma unroll
  for(unsigned t=0;t<ROWS;t++)if(t<T){float a=sum(acc[j][t]);if(!lane)out[(uint64_t)t*M+row0+j]=a;}
 }
}

static int matvec_dispatch(float *out, const char *w, const float *x,
                           unsigned type, unsigned T, unsigned K, unsigned M) {
    const dim3 grid((M + 3) / 4, T);
    const uint64_t stride = row_bytes(type, K);
    if (!stride) return 0;
    // Preserve the FP32 accumulation tree while overlapping four scalar loads.
    if (ds4_rocm_is_gfx1151() && type == 0 && T >= 1 && T <= 3 && K == 2560 && (M == 48 || M == 512) && !((uintptr_t)x&15) && !((uintptr_t)w&3)) {
#define QWEN_F32_PREFETCH(ROWS) \
        if (M == 48) matvec_f32_prefetch<ROWS,4><<<(M+3)/4,128,0,0>>>(out,w,x,T,K,M,stride); \
        else matvec_f32_prefetch<ROWS,2><<<(M+1)/2,64,0,0>>>(out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_F32_PREFETCH(1); }
        else if (T == 2) { QWEN_F32_PREFETCH(2); }
        else { QWEN_F32_PREFETCH(4); }
#undef QWEN_F32_PREFETCH
        return launched();
    }
    // Ordered prefetch shares each dequantized Q8 weight across MTP rows.
    if (ds4_rocm_is_gfx1151() && T <= 3 && !((uintptr_t)x&15) && !((uintptr_t)w&7)) {
        // The wide vocabulary projection amortizes its weight read across verification rows.
        if (type == 8 && T > 1 && K == 2560 && M == 248320) {
            if (T == 2) matvec_prefetch<8,2,1,false,1,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
            else matvec_prefetch<8,4,1,false,2,4><<<(M+1)/2,32,0,0>>>(out,w,x,T,K,M,stride);
            return launched();
        }
        if (type == 8 && T > 1 && ((K == 2560 && (M == 128 || M == 512 || M == 6144 || M == 10240 || M == 12288)) || ((K == 6144 || K == 5120) && M == 2560))) {
            if (T == 2) matvec_prefetch<8,2,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
            else matvec_prefetch<8,4,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
            return launched();
        }
        if (type == 1 && T == 1 && M == 320 && K == 10240) {
            matvec_prefetch<1,1,8,true><<<40,256,40960,0>>>(out,w,x,T,K,M,stride);
            return launched();
        }
        if (type == 1 && M == 4 && K == 10240) {
            if (T == 1) matvec_prefetch<1,1,2><<<2,64,0,0>>>(out,w,x,T,K,M,stride);
            else if (T == 2) matvec_prefetch<1,2,2><<<2,64,0,0>>>(out,w,x,T,K,M,stride);
            else matvec_prefetch<1,4,2><<<2,64,0,0>>>(out,w,x,T,K,M,stride);
            return launched();
        }
    }
    if (type == 1 && T > 1 && T <= 8 && M == 320 && K == 10240 &&
        !((uintptr_t)x&15) && !((uintptr_t)w&7) && ds4_rocm_is_gfx1151()) {
        if (T == 2) matvec_f16_input_first<2,1,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
        else if (T <= 4) matvec_f16_input_first<4,1,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
        else matvec_f16_input_first<8,1,1><<<M,32,0,0>>>(out,w,x,T,K,M,stride);
        return launched();
    }
    if (type == 8 && T <= 8 && !((uintptr_t)x&15)) {
#define QWEN_Q8_ROWS(N) matvec_q8<N><<<(M+3)/4,128,0,0>>>(out,w,x,T,K,M,stride)
        if (T == 1) { QWEN_Q8_ROWS(1); }
        else if (T == 2) { QWEN_Q8_ROWS(2); }
        else if (T <= 4) { QWEN_Q8_ROWS(4); }
        else { QWEN_Q8_ROWS(8); }
#undef QWEN_Q8_ROWS
        return launched();
    }
#define QWEN_MV(TYPE) case TYPE: \
    if (T == 2) matvec_rows<TYPE,2><<<(M+3)/4,128,0,0>>>(out,w,x,T,K,M,stride); \
    else if (T > 2 && T <= 4) matvec_rows<TYPE,4><<<(M+3)/4,128,0,0>>>(out,w,x,T,K,M,stride); \
    else if (T > 4 && T <= 8) matvec_rows<TYPE,8><<<(M+3)/4,128,0,0>>>(out,w,x,T,K,M,stride); \
    else matvec<TYPE><<<grid,128,0,0>>>(out,w,x,K,M,stride); break
    switch (type) {
    QWEN_MV(0); QWEN_MV(1); QWEN_MV(2); QWEN_MV(8); QWEN_MV(10);
    QWEN_MV(12); QWEN_MV(16); QWEN_MV(30); QWEN_MV(39);
    default: return 0;
    }
#undef QWEN_MV
    return launched();
}

template<unsigned TYPE>
__global__ void unpack(float *out, const char *w, unsigned K, unsigned M, uint64_t rb) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (uint64_t)K*M) out[i] = value<TYPE>(w+(i/K)*rb,i%K);
}

/* Power-of-two row scaling protects half range before operand rounding. */
template<unsigned TYPE>
__global__ void pack_half_rows(float *scales, __half *hi,
        const char *x, unsigned K, uint64_t rb) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    __shared__ float maxima[256], inv;
    float mx = 0;
    for (unsigned k = tid; k < K; k += 256) mx = fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid] = mx;
    __syncthreads();
    for (unsigned d = 128; d; d /= 2) {
        if (tid < d) maxima[tid] = fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e = maxima[0] > 0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        inv = ldexpf(1,-e); scales[row] = ldexpf(1,e);
    }
    __syncthreads();
    for (unsigned k = tid; k < K; k += 256) {
        const float v = value<TYPE>(x+(uint64_t)row*rb,k)*inv;
        const __half h = __float2half_rn(v);
        hi[(uint64_t)row*K+k] = h;
    }
}

__global__ void dense_rescale(float *out, const float *scales,
        unsigned M, unsigned N) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*N) out[i] *= scales[i/M];
}

/* F16/Q8 products use FP32 accumulation and output, including strided
 * output slices. This gfx1151 library build's first heuristic is a slow VALU
 * kernel; use the measured zero-workspace WMMA solution after support checks.
 * An unknown library or unsupported shape keeps the ordinary BLAS fallback. */
static int dense_half_solution(unsigned M, unsigned T, unsigned K, unsigned stride) {
    if (T != 32) return 2539;
    if ((M == 1344 && K == 6144 && stride == 2560) ||
        (M == 2880 && K == 2560 && stride == 6144) ||
        (M == 3264 && K == 2560 &&
         (stride == 6144 || stride == 10240 || stride == 12288)) ||
        (M == 10240 && K == 320 && stride == 10240)) return 2538;
    if ((M == 128 && K == 2560 && stride == 128) ||
        (M == 320 && K == 10240 && stride == 320) ||
        (M == 448 && K == 2560 && stride == 10240) ||
        (M == 512 && K == 2560 && stride == 512) ||
        (M == 640 && K == 2560 && stride == 640) ||
        (M == 960 && K == 5120 && stride == 2560) ||
        (M == 1216 && K == 6144 && stride == 2560) ||
        (M == 1600 && K == 5120 && stride == 2560) ||
        (M == 2496 && K == 2560 && stride == 12288) ||
        (M == 2560 && ((K == 640 && stride == 2560) ||
                       (K == 2560 && stride == 2560)))) return 2537;
    return 2539;
}

static int dense_half_product(float *out, const __half *w, const __half *x,
        unsigned M, unsigned T, unsigned K, unsigned stride,
        float alpha, float beta, const char *label) {
    if (ds4_rocm_is_gfx1151() && g_hipblaslt_ready && T >= 32) {
        int version = 0;
        char revision[128] = {};
        if (hipblasLtGetVersion(g_hipblaslt, &version) == HIPBLAS_STATUS_SUCCESS &&
            hipblasLtGetGitRevision(g_hipblaslt, revision) == HIPBLAS_STATUS_SUCCESS &&
            version == 100401 && !strcmp(revision, "8d1ae90e")) {
            // Long WMMA accumulation failed the independent FP32 projection
            // bound. Sum shorter K segments; retain the original row strides.
            const unsigned segment = K > 4096 ? 1024 : K;
            const unsigned tail = K % segment ? K % segment : segment;
            const int solution = dense_half_solution(M,T,K,stride);
            if (hipblaslt_gemm_plan_get(M,T,segment,label,HIP_R_32F,solution,stride,K) &&
                hipblaslt_gemm_plan_get(M,T,tail,label,HIP_R_32F,solution,stride,K)) {
                // Preflight every layout before writing any output: fallback
                // must not consume a partially accumulated product.
                for (unsigned k = 0; k < K; k += segment) {
                    const unsigned n = std::min(segment,K-k);
                    cuda_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(
                        M,T,n,label,HIP_R_32F,solution,stride,K);
                    const float add = k ? 1.0f : beta;
                    if (!p || !hipblaslt_ok(hipblasLtMatmul(g_hipblaslt,p->desc,
                        &alpha,w+k,p->a_desc,x+k,p->b_desc,&add,
                        out,p->c_desc,out,p->d_desc,&p->algo,NULL,0,0),label)) return 0;
                }
                return 1;
            }
        }
    }
    return cublas_ok(cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        M, T, K, &alpha, w, CUDA_R_16F, K, x, CUDA_R_16F, K,
        &beta, out, CUDA_R_32F, stride, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT), label);
}

static int dense_f16_blas(float *out, const float *x, const __half *w,
        unsigned T, unsigned K, unsigned M) {
    const uint64_t xn = (uint64_t)T*K;
    const uint64_t packed_bytes = (xn*sizeof(__half)+3)&~UINT64_C(3);
    __half *hi = (__half *)cuda_tmp_alloc(packed_bytes+(uint64_t)T*sizeof(float), "Qwen F16 projection scratch");
    if (!hi) return 0;
    float *scales = (float *)((char *)hi+packed_bytes);
    pack_half_rows<0><<<T,256,0,0>>>(scales,hi,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1;
    // One half-operand product with FP32 accumulation/output.
    // This intentionally rounds activations; quality mode uses FP32 operands.
    if (!dense_half_product(out,w,hi,M,T,K,M,one,zero,"Qwen F16 projection")) return 0;
    dense_rescale<<<((uint64_t)M*T+255)/256,256,0,0>>>(out,scales,M,T);
    return launched();
}

__global__ void dense_rescale2(float *out, const float *xs, const float *ws,
        unsigned M, unsigned T, unsigned stride) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if (i < (uint64_t)M*T) {
        const uint64_t dst = (i/M)*stride+i%M;
        out[dst] = out[dst]*xs[i/M]*ws[i%M];
    }
}

/* Read Q8 blocks directly for gfx1151 bulk projections.  Shape-specific tiles
 * keep the WMMA work resident across the Qwen dense projection dimensions. */
template<unsigned WAVES, unsigned NT, unsigned KT, unsigned PAD, unsigned VEC>
__launch_bounds__(WAVES*32u,1)
__global__ void qwen_q8_direct(float *out,
        const unsigned char *w, const __half *x, const float *xs, const float *ws,
        unsigned T, unsigned K, unsigned M, unsigned begin, unsigned length) {
    static_assert(KT%32==0 && NT%16==0);
    constexpr unsigned NF=NT/16, MT=WAVES*16;
    const unsigned block_m=blockIdx.x*MT, block_t=blockIdx.y*NT;
    const unsigned tid=threadIdx.x, wave=tid>>5, lane=tid&31, lane16=lane&15;
    const unsigned row0=block_m+wave*16, my_row=row0+lane16;
    const unsigned safe_row=my_row<M ? my_row : M-1;
    const uint64_t rb=(uint64_t)(K/32)*34;
    const unsigned char *wr=w+(uint64_t)safe_row*rb;
    const float winv=1.0f/ws[safe_row];
    __shared__ _Float16 sx[NT][KT+PAD];
    float8 acc[NF]={};
    for (unsigned k0=0;k0<length;k0+=KT) {
        for (unsigned j=tid*VEC;j<NT*KT;j+=WAVES*32*VEC) {
            const unsigned tok=j/KT, kk=j%KT;
            #pragma unroll
            for (unsigned z=0;z<VEC;z++) {
                const unsigned gt=block_t+tok, gk=begin+k0+kk+z;
                if (kk+z<KT) sx[tok][kk+z]=(gt<T && gk<begin+length) ? (_Float16)x[(uint64_t)gt*K+gk] : (_Float16)0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned kb=0;kb<KT/32;kb++) {
            if (k0+kb*32>=length) break;
            const unsigned char *bp=wr+(uint64_t)((begin+k0)/32+kb)*34;
            uint16_t sb; __builtin_memcpy(&sb,bp,2);
            const float sc=qwen_f16_to_f32(sb)*winv;
            const int8_t *q=(const int8_t *)(bp+2);
            half16 a0,a1;
            #pragma unroll
            for (unsigned i=0;i<16;i++) {
                a0[i]=(_Float16)(sc*(float)(int)q[i]);
                a1[i]=(_Float16)(sc*(float)(int)q[16+i]);
            }
            #pragma unroll
            for (unsigned f=0;f<NF;f++) {
                const unsigned tok=f*16+lane16;
                const _Float16 *xp=&sx[tok][kb*32];
                const half16 b0=*(const half16 *)xp, b1=*(const half16 *)(xp+16);
                acc[f]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0,b0,acc[f]);
                acc[f]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1,b1,acc[f]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned f=0;f<NF;f++) {
        const unsigned tok=block_t+f*16+lane16;
        if (tok>=T) continue;
        #pragma unroll
        for (unsigned j=0;j<8;j++) {
            const unsigned row=row0+2*j+(lane>>4);
            if (row<M) {
                const uint64_t dst=(uint64_t)tok*M+row;
                const float v=acc[f][j]*xs[tok]*ws[row];
                out[dst]=begin ? out[dst]+v : v;
            }
        }
    }
}

static int launch_qwen_q8_direct(float *out, const unsigned char *w,
        const __half *x, const float *xs, const float *ws, unsigned T,
        unsigned K, unsigned M, unsigned begin, unsigned length) {
    if (T<=64) {
        if (M<=2560) {
            qwen_q8_direct<8,16,64,8,4><<<dim3((M+127)/128,(T+15)/16),256,0,0>>>(
                out,w,x,xs,ws,T,K,M,begin,length);
        } else if (M==6144 || M==10240 || M==12288) {
            qwen_q8_direct<24,32,256,8,4><<<dim3((M+383)/384,(T+31)/32),768,0,0>>>(
                out,w,x,xs,ws,T,K,M,begin,length);
        } else {
            qwen_q8_direct<16,64,128,8,4><<<dim3((M+255)/256,(T+63)/64),512,0,0>>>(
                out,w,x,xs,ws,T,K,M,begin,length);
        }
    } else if (M<=640 && T<=128) {
        qwen_q8_direct<8,16,64,8,4><<<dim3((M+127)/128,(T+15)/16),256,0,0>>>(
            out,w,x,xs,ws,T,K,M,begin,length);
    } else if ((M==640 && T>=768 && T<1536) ||
            (M==2560 && !(T>=256 && T<768))) {
        qwen_q8_direct<16,64,256,8,8><<<dim3((M+255)/256,(T+63)/64),512,0,0>>>(
            out,w,x,xs,ws,T,K,M,begin,length);
    } else if ((M==640 && T>=1536) || (M==6144 && T>=256) ||
            ((M==10240 || M==12288) && T>=128 && T<1536)) {
        qwen_q8_direct<16,128,128,8,8><<<dim3((M+255)/256,(T+127)/128),512,0,0>>>(
            out,w,x,xs,ws,T,K,M,begin,length);
    } else if ((M==10240 || M==12288) && T>=1536) {
        qwen_q8_direct<24,128,128,8,8><<<dim3((M+383)/384,(T+127)/128),768,0,0>>>(
            out,w,x,xs,ws,T,K,M,begin,length);
    } else {
        qwen_q8_direct<16,64,128,8,4><<<dim3((M+255)/256,(T+63)/64),512,0,0>>>(
            out,w,x,xs,ws,T,K,M,begin,length);
    }
    return launched();
}

template<unsigned TYPE>
__global__ void qwen_dense_row_scales(float *scales, const char *x,
        unsigned K, uint64_t rb) {
    const unsigned row=blockIdx.x, tid=threadIdx.x;
    __shared__ float maxima[256];
    float mx=0;
    for (unsigned k=tid;k<K;k+=256) mx=fmaxf(mx,fabsf(value<TYPE>(x+(uint64_t)row*rb,k)));
    maxima[tid]=mx;
    __syncthreads();
    for (unsigned d=128;d;d/=2) {
        if (tid<d) maxima[tid]=fmaxf(maxima[tid],maxima[tid+d]);
        __syncthreads();
    }
    if (!tid) {
        const int e=maxima[0]>0 ? max(-120,min(120,(int)((__float_as_uint(maxima[0])>>23)&255)-127)) : 0;
        scales[row]=ldexpf(1,e);
    }
}

static int dense_q8_blas(float *out, const float *x, const char *w,
        unsigned T, unsigned K, unsigned M) {
    if (T>=32 && !(K%128) && ds4_rocm_is_gfx1151()) {
        const uint64_t xn=(uint64_t)T*K;
        const uint64_t xbytes=(xn*sizeof(__half)+15)&~UINT64_C(15);
        __half *xh=(__half *)cuda_tmp_alloc(xbytes+((uint64_t)T+M)*sizeof(float),
            "Qwen direct Q8 projection scratch");
        if (!xh) return 0;
        float *xs=(float *)((char *)xh+xbytes), *ws=xs+T;
        pack_half_rows<0><<<T,256,0,0>>>(xs,xh,(const char *)x,K,(uint64_t)K*4);
        qwen_dense_row_scales<8><<<M,256,0,0>>>(ws,w,K,row_bytes(8,K));
        if (!launched()) return 0;
        const unsigned segment=K>4096 ? 1024 : K;
        for (unsigned begin=0;begin<K;begin+=segment) {
            const unsigned length=std::min(segment,K-begin);
            if (!launch_qwen_q8_direct(out,(const unsigned char *)w,xh,xs,ws,
                    T,K,M,begin,length)) return 0;
        }
        return 1;
    }
    // Retain the measured row tiling with a 16 MiB packed-weight budget.
    const unsigned bound = (unsigned)std::max(UINT64_C(1),(UINT64_C(16)<<20)/((uint64_t)K*sizeof(__half)));
    const unsigned tile = std::min(M,bound >= 64 ? bound/64*64 : bound);
    const uint64_t xn = (uint64_t)T*K, wn = (uint64_t)tile*K;
    const uint64_t xbytes = (xn*sizeof(__half)+15)&~UINT64_C(15);
    const uint64_t wbytes = (wn*sizeof(__half)+3)&~UINT64_C(3);
    __half *xh = (__half *)cuda_tmp_alloc(xbytes+wbytes+((uint64_t)T+tile)*sizeof(float),"Qwen Q8 projection scratch");
    if (!xh) return 0;
    __half *wh = (__half *)((char *)xh+xbytes);
    float *xs = (float *)((char *)wh+wbytes), *ws = xs+T;
    pack_half_rows<0><<<T,256,0,0>>>(xs,xh,(const char *)x,K,(uint64_t)K*4);
    if (!launched()) return 0;
    const float zero = 0, one = 1;
    const uint64_t rb = row_bytes(8,K);
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        pack_half_rows<8><<<n,256,0,0>>>(ws,wh,w+(uint64_t)r*rb,K,rb);
        if (!launched()) return 0;
        // Round both scaled Q8 weights and activations to F16.
        if (!dense_half_product(out+r,wh,xh,n,T,K,M,one,zero,"Qwen Q8 projection")) return 0;
        dense_rescale2<<<((uint64_t)n*T+255)/256,256,0,0>>>(out+r,xs,ws,n,T,M);
        if (!launched()) return 0;
    }
    return 1;
}

/* Bound the temporary weight expansion instead of retaining an FP32 copy
 * of each projection. hipBLAS uses FP32 math, including FP32 activations. */
static int dense_blas(float *out, const float *x, const char *w,
        unsigned type, unsigned T, unsigned K, unsigned M) {
    const unsigned tile = std::min(M, (unsigned)std::max(UINT64_C(1), (UINT64_C(64)*1024*1024)/((uint64_t)K*4)));
    float *scratch = type ? (float *)cuda_tmp_alloc((uint64_t)tile*K*4, "Qwen dense FP32 tile") : NULL;
    if (type && !scratch) return 0;
    const uint64_t rb = row_bytes(type,K);
    const float alpha = 1, beta = 0;
    for (unsigned r = 0; r < M; r += tile) {
        const unsigned n = std::min(tile,M-r);
        const float *wf = type ? scratch : (const float *)w+(uint64_t)r*K;
        if (type) {
#define QWEN_UNPACK(TYPE) case TYPE: unpack<TYPE><<<((uint64_t)n*K+255)/256,256,0,0>>>(scratch,w+(uint64_t)r*rb,K,n,rb); break
            switch (type) {
            QWEN_UNPACK(1); QWEN_UNPACK(2); QWEN_UNPACK(8); QWEN_UNPACK(10);
            QWEN_UNPACK(12); QWEN_UNPACK(16); QWEN_UNPACK(30); QWEN_UNPACK(39);
            default: return 0;
            }
#undef QWEN_UNPACK
            if (!launched()) return 0;
        }
        const int fp32_solution = T == 32 && K == 2560 && n == 48 ? 2842 :
                                  T == 32 && K == 2560 && n == 512 ? 3019 :
                                  T >= 512 && K == 2560 && (n == 48 || n == 512) ? 2921 : -1;
        if (!type && !g_quality_mode && fp32_solution >= 0 &&
                ds4_rocm_is_gfx1151() && g_hipblaslt_ready) {
            int version = 0;
            char revision[128] = {};
            if (hipblasLtGetVersion(g_hipblaslt,&version) == HIPBLAS_STATUS_SUCCESS &&
                    hipblasLtGetGitRevision(g_hipblaslt,revision) == HIPBLAS_STATUS_SUCCESS &&
                    version == 100401 && !strcmp(revision,"8d1ae90e")) {
                cuda_hipblaslt_gemm_plan *p = hipblaslt_gemm_plan_get(
                    n,T,K,"Qwen FP32 projection",HIP_R_32F,fp32_solution,M,K,HIP_R_32F);
                if (p) {
                    if (!hipblaslt_ok(hipblasLtMatmul(g_hipblaslt,p->desc,
                        &alpha,wf,p->a_desc,x,p->b_desc,&beta,out+r,p->c_desc,
                        out+r,p->d_desc,&p->algo,NULL,0,0),"Qwen FP32 projection")) return 0;
                    continue;
                }
            }
        }
        if (!cublas_ok(cublasGemmEx(g_cublas,CUBLAS_OP_T,CUBLAS_OP_N,
                n,T,K,&alpha,wf,CUDA_R_32F,K,x,CUDA_R_32F,K,&beta,out+r,CUDA_R_32F,M,
                HIPBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT), "Qwen FP32 projection")) return 0;
    }
    return 1;
}

__global__ void mtp_stage(float *cat, const float *e, const float *R,
        const float *ge, const float *gh, unsigned E, unsigned hc, float eps) {
    const unsigned row = blockIdx.x, tid = threadIdx.x;
    const bool emb = row == 0;
    const unsigned n = emb ? E : E * hc;
    const float *rs = emb ? e : R;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < n; i += blockDim.x) ss += rs[i] * rs[i];
    const float inv = rsqrtf(block_sum(ss, red) / n + eps);
    const float *src = emb ? e : R + (uint64_t)(row - 1) * E;
    const float *g = emb ? ge : gh + (uint64_t)(row - 1) * E;
    float *o = cat + (uint64_t)row * 2 * E;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        o[(emb ? 0 : E) + i] = src[i] * inv * g[i];
        o[(emb ? E : 0) + i] = 0;
    }
}

__global__ void mtp_combine(float *out, const float *proj, unsigned E, unsigned hc) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < E * hc) out[i] = proj[i % E] + proj[E + i];
}

struct max_pair { float value; int index; };
template<bool FINISH>
__global__ void argmax(int *out, max_pair *scratch, const float *logits, unsigned n) {
    const unsigned tid = threadIdx.x;
    __shared__ max_pair best[256];
    max_pair v = {-1e30f, 0};
    for (unsigned i = (FINISH ? 0 : blockIdx.x * 4096) + tid;
         i < (FINISH ? n : min(n, (blockIdx.x + 1) * 4096)); i += 256) {
        const max_pair p = FINISH ? scratch[i] : max_pair{logits[i], (int)i};
        if (p.value > v.value || (p.value == v.value && p.index < v.index)) v = p;
    }
    best[tid] = v;
    __syncthreads();
    for (unsigned s = 128; s; s /= 2) {
        if (tid < s) {
            const max_pair p = best[tid + s], q = best[tid];
            if (p.value > q.value || (p.value == q.value && p.index < q.index)) best[tid] = p;
        }
        __syncthreads();
    }
    if (!tid) { if (FINISH) *out = best[0].index; else scratch[blockIdx.x] = best[0]; }
}

__global__ void vis_patch(float *x, const float *a, const float *b, const float *bias,
        const float *pos, unsigned N, unsigned E) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (uint64_t)N * E) x[i] = a[i] + b[i] + bias[i % E] + pos[i];
}

__global__ void vis_norm(float *out, const float *x, const float *w, const float *b, unsigned E, float eps) {
    const unsigned tid = threadIdx.x;
    const uint64_t base = (uint64_t)blockIdx.x * E;
    __shared__ float red[32];
    float v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) v += x[base + i];
    const float mean = block_sum(v, red) / E;
    v = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { const float d = x[base + i] - mean; v += d*d; }
    const float inv = rsqrtf(block_sum(v, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

__global__ void vis_qkv(float *q, float *k, float *v, const float *qkv, const float *bias,
        unsigned H, unsigned D, unsigned grid_w) {
    const unsigned row = blockIdx.x, tid = threadIdx.x, E = H*D, hd = D/2, qd = D/4;
    const unsigned blk = row/4, within = row%4, w2 = grid_w/2;
    const float hp = (blk/w2)*2 + within/2, wp = (blk%w2)*2 + within%2;
    const float *src = qkv + (uint64_t)row*3*E;
    const uint64_t off = (uint64_t)row*E;
    for (unsigned idx = tid; idx < 2*H*hd; idx += blockDim.x) {
        const unsigned part = idx/(H*hd), rem = idx%(H*hd), h = rem/hd, i = rem%hd;
        const unsigned base = part*E + h*D;
        const float x0 = src[base+i] + bias[base+i], x1 = src[base+i+hd] + bias[base+i+hd];
        const unsigned j = i < qd ? i : i-qd;
        const float theta = (i < qd ? hp : wp) * powf(10000.0f, -2.0f*j/hd);
        float s, c; sincosf(theta, &s, &c);
        float *dst = part ? k : q;
        dst[off+h*D+i] = x0*c - x1*s;
        dst[off+h*D+i+hd] = x0*s + x1*c;
    }
    for (unsigned i = tid; i < E; i += blockDim.x) v[off+i] = src[2*E+i] + bias[2*E+i];
}

static constexpr unsigned VIS_ATTN_WAVES = 32;
static constexpr unsigned VIS_ATTN_KEY_TILE = 48;

__global__ void vis_attention(float *out, const float *q, const float *k, const float *v,
        unsigned N, unsigned H, unsigned D) {
    extern __shared__ float kv[];
    const unsigned row = blockIdx.x*VIS_ATTN_WAVES + threadIdx.x/32, h = blockIdx.y;
    const unsigned lane = threadIdx.x & 31, E = H*D;
    const bool active = row < N;
    const uint64_t qb = (uint64_t)row*E + h*D;
    float query[3] = {}, acc[3] = {};
    for (unsigned j = 0; active && j < 3; j++) if (lane+j*32 < D) query[j] = q[qb+lane+j*32];
    float mx = -INFINITY, denom = 0;
    for (unsigned p0 = 0; p0 < N; p0 += VIS_ATTN_KEY_TILE) {
        const unsigned count = min(VIS_ATTN_KEY_TILE, N-p0), elems = count*D;
        for (unsigned i = threadIdx.x; i < 2*elems; i += blockDim.x) {
            const bool is_v = i >= elems;
            const unsigned j = is_v ? i-elems : i;
            const uint64_t src = (uint64_t)(p0+j/D)*E + h*D + j%D;
            kv[i] = is_v ? v[src] : k[src];
        }
        __syncthreads();
        if (active) for (unsigned p = 0; p < count; p++) {
            const unsigned kb = p*D;
            float score = 0;
            for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D) score += query[j]*kv[kb+lane+j*32];
            score = sum(score) * rsqrtf((float)D);
            const float nm = fmaxf(mx, score), old = expf(mx-nm), prob = expf(score-nm);
            denom = denom*old + prob;
            for (unsigned j = 0; j < 3; j++) if (lane+j*32 < D)
                acc[j] = acc[j]*old + prob*kv[elems+kb+lane+j*32];
            mx = nm;
        }
        __syncthreads();
    }
    for (unsigned j = 0; active && j < 3; j++) if (lane+j*32 < D) out[qb+lane+j*32] = acc[j]/denom;
}

__global__ void vis_add(float *x, const float *add, const float *bias, unsigned N, unsigned E, unsigned mode) {
    const uint64_t i = (uint64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (uint64_t)N*E) return;
    if (add) { x[i] += add[i] + bias[i%E]; return; }
    float t = x[i] + bias[i%E];
    if (!mode) t *= .5f * (1 + tanhf(fminf(30, fmaxf(-30, .7978845608f*(t+.044715f*t*t*t)))));
    else if (mode == 1) t *= .5f * (1 + erff(t*.70710678f));
    x[i] = t;
}

__global__ void hc_norm(float *xn, float *inj, const float *R, const float *gamma,
                        const char *wi, unsigned type, unsigned E, unsigned hc,
                        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x / 8, chunk = blockIdx.x % 8, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    float acc[4] = {};
    const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
    for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
        const float v = R[base + i] * inv * gamma[stream * E + i];
        xn[base + i] = v;
        for (unsigned j = 0; j < ni; j++) acc[j] += scalar(wi, j * dim + stream * E + i, type) * v;
    }
    for (unsigned j = 0; j < ni; j++) {
        const float v = block_sum(acc[j], red);
        if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
    }
}

__global__ void hc_norm_one(float *xn, float *inj, const float *R, const float *gamma,
                        const char *wi, unsigned type, unsigned E, unsigned hc,
                        unsigned ni, float eps) {
    const unsigned stream = blockIdx.x, tok = blockIdx.y;
    const unsigned tid = threadIdx.x, dim = E * hc;
    const uint64_t base = ((uint64_t)tok * hc + stream) * E;
    __shared__ float red[32];
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) ss += R[base + i] * R[base + i];
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned chunk=0; chunk<8; ++chunk) {
        float acc[4] = {};
        const unsigned per = (E + 7) / 8, end = min(E, (chunk + 1) * per);
        for (unsigned i = chunk * per + tid; i < end; i += blockDim.x) {
            const float v = R[base + i] * inv * gamma[stream * E + i];
            xn[base + i] = v;
            for (unsigned j = 0; j < ni; j++) acc[j] += scalar(wi, j * dim + stream * E + i, type) * v;
        }
        for (unsigned j = 0; j < ni; j++) {
            const float v = block_sum(acc[j], red);
            if (!tid) inj[((uint64_t)tok * hc * 8 + stream * 8 + chunk) * ni + j] = v;
        }
    }
}

template<unsigned TYPE>
__global__ void hc_mix(float *out, const float *xn, const float *lo,
                       const char *up, unsigned E, unsigned hc, unsigned rank, uint64_t rb) {
    const unsigned d = blockIdx.x * 4 + threadIdx.x / 32, tok = blockIdx.y;
    __shared__ __align__(16) float activated[512];
    if (rank <= 512) {
        for (unsigned r = threadIdx.x; r < rank; r += blockDim.x)
            activated[r] = silu(lo[(uint64_t)tok*rank+r]/hc);
        __syncthreads();
    }
    if (d >= E) return;
    const unsigned lane = threadIdx.x & 31, stream = lane / 8, l = lane % 8;
    float a = 0;
    if (stream < hc) {
        const char *row = up+((uint64_t)stream*E+d)*rb;
        if (!(rank%4) && rank <= 512 &&
            !(TYPE == 0 ? (uintptr_t)row&15 : TYPE == 1 ? (uintptr_t)row&7 : 0)) {
            for (unsigned r = l*4; r < rank; r += 32) {
                const float4 w = value4<TYPE>(row,r,NULL,NULL), v = *(const float4 *)(activated+r);
                a += w.x*v.x; a += w.y*v.y; a += w.z*v.z; a += w.w*v.w;
            }
        } else {
            for (unsigned r = l; r < rank; r += 8)
                a += value<TYPE>(row,r) *
                    (rank <= 512 ? activated[r] : silu(lo[(uint64_t)tok * rank + r] / hc));
        }
    }
    for (unsigned off = 1; off <= 4; off *= 2) a += __shfl_xor( a, off, 32);
    float v = stream < hc ? sigmoid(a) * xn[((uint64_t)tok * hc + stream) * E + d] : 0;
    v += __shfl_xor( v, 8, 32);
    v += __shfl_xor( v, 16, 32);
    if (!lane) out[(uint64_t)tok * E + d] = v / hc;
}

__device__ float injection(const float *inj, unsigned hc, unsigned stream) {
    float v = 0;
    for (unsigned i = 0; i < hc * 8; i++) v += inj[i * hc + stream];
    return 2 * sigmoid(v / hc);
}

__global__ void hc_combine(float *R, const float *out, const float *inj, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    __shared__ float weights[4];
    if (threadIdx.x < hc) weights[threadIdx.x] = injection(inj + (uint64_t)tok * hc * hc * 8, hc, threadIdx.x);
    __syncthreads();
    if (d >= E) return;
    for (unsigned s = 0; s < hc; s++) R[((uint64_t)tok * hc + s) * E + d] += weights[s] * out[(uint64_t)tok * E + d];
}

__global__ void hc_lo(float *dst, const float *src, uint64_t n, unsigned hc) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = silu(src[i] / hc);
}

__global__ void hc_rows(float *dst, const float *up, const float *xn, unsigned E, unsigned hc) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= E) return;
    float v = 0;
    for (unsigned s = 0; s < hc; s++) {
        const uint64_t i = ((uint64_t)t * hc + s) * E + d;
        v += sigmoid(up[i]) * xn[i];
    }
    dst[(uint64_t)t * E + d] = v / hc;
}

} // namespace qwen4_rocm

namespace qwen4_rocm {

__global__ void conv(float *x, float *history, const float *w, unsigned T,
                     unsigned C, unsigned K, bool activate,
                     float *snap, unsigned snap_t, float *snap2, unsigned snap2_t) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float win[3], taps[4];
    for (unsigned i = 0; i < K - 1; i++) win[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = w[c * K + i];
    for (unsigned t = 0; t < T; t++) {
        const uint64_t pos = (uint64_t)t * C + c;
        const float raw = x[pos];
        float v = taps[K - 1] * raw;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * win[i];
        for (unsigned i = 0; i + 2 < K; i++) win[i] = win[i + 1];
        win[K - 2] = raw;
        x[pos] = activate ? silu(v) : v;
        if (snap && t == snap_t) for (unsigned i = 0; i < K - 1; i++) snap[(uint64_t)i * C + c] = win[i];
        if (snap2 && t == snap2_t) for (unsigned i = 0; i < K - 1; i++) snap2[(uint64_t)i * C + c] = win[i];
    }
    for (unsigned i = 0; i < K - 1; i++) history[(uint64_t)i * C + c] = win[i];
}

__global__ void gdn_prep(float *qkv, float *a, float *b, const float *A, const float *bias,
                         unsigned Hk, unsigned Hv, unsigned D) {
    const unsigned h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const unsigned C = (2 * Hk + Hv) * D, npt = D / 32;
    float *q = qkv + (uint64_t)t * C + h * D + lane * npt, *k = q + Hk * D;
    float qs = 0, ks = 0;
    for (unsigned i = 0; i < npt; i++) { qs += q[i] * q[i]; ks += k[i] * k[i]; }
    qs = rsqrtf(sum(qs) + 1e-6f) * rsqrtf((float)D);
    ks = rsqrtf(sum(ks) + 1e-6f);
    for (unsigned i = 0; i < npt; i++) { q[i] *= qs; k[i] *= ks; }
    if (!h) for (unsigned j = lane; j < Hv; j += 32) {
        const uint64_t p = (uint64_t)t * Hv + j;
        a[p] = expf(A[j] * softplus(a[p] + bias[j]));
        b[p] = sigmoid(b[p]);
    }
}

/* One warp owns a state row across the whole chunk. In particular, MTP
 * snapshots contain the state after the requested token, not the final row. */
template<unsigned ROWS, unsigned D>
__global__ void gdn_scan(float *out, float *state, const float *qkv,
                         const float *a, const float *b, unsigned T, unsigned Hk,
                         unsigned Hv, float *snap, unsigned st,
                         float *snap2, unsigned st2) {
    const unsigned dv = (blockIdx.x * 4 + threadIdx.x / 32)*ROWS, h = blockIdx.y;
    if (dv >= D) return;
    const unsigned npt = D / 32, k0 = (threadIdx.x & 31) * npt, kh = h % Hk;
    const unsigned C = (2 * Hk + Hv) * D;
    const uint64_t idx = ((uint64_t)h * D + dv) * D + k0;
    float s[ROWS][4];
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) s[r][i] = state[idx+(uint64_t)r*D+i];
    for (unsigned t = 0; t < T; t++) {
        const float *q = qkv + (uint64_t)t * C + kh * D + k0, *k = q + Hk * D;
        const float decay = a[(uint64_t)t * Hv + h], beta = b[(uint64_t)t * Hv + h];
        #pragma unroll
        for (unsigned r = 0; r < ROWS; r++) {
            const float v = qkv[(uint64_t)t*C+2*Hk*D+h*D+dv+r];
            float u = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] *= decay; u += s[r][i]*k[i]; }
            const float delta = (v-sum(u))*beta;
            float o = 0;
            #pragma unroll
            for (unsigned i = 0; i < npt; i++) { s[r][i] += k[i]*delta; o += s[r][i]*q[i]; }
            o = sum(o);
            if (!(threadIdx.x&31)) out[((uint64_t)t*Hv+h)*D+dv+r] = o;
            if (snap && t == st) for (unsigned i = 0; i < npt; i++) snap[idx+(uint64_t)r*D+i] = s[r][i];
            if (snap2 && t == st2) for (unsigned i = 0; i < npt; i++) snap2[idx+(uint64_t)r*D+i] = s[r][i];
        }
    }
    #pragma unroll
    for (unsigned r = 0; r < ROWS; r++)
        #pragma unroll
        for (unsigned i = 0; i < npt; i++) state[idx+(uint64_t)r*D+i] = s[r][i];
}

__global__ void gdn_out(float *o, const float *z, const float *w, unsigned H, unsigned D, float eps) {
    const unsigned h = blockIdx.x, t = blockIdx.y, npt = D / 32, k0 = threadIdx.x * npt;
    const uint64_t idx = ((uint64_t)t * H + h) * D + k0;
    float ss = 0;
    for (unsigned i = 0; i < npt; i++) ss += o[idx + i] * o[idx + i];
    const float r = rsqrtf(sum(ss) / D + eps);
    for (unsigned i = 0; i < npt; i++) o[idx + i] = o[idx + i] * r * w[k0 + i] * sigmoid(z[idx + i]);
}

__global__ void ngram_gate(float *gated, float *normed, const float *R, const float *key,
                           const float *val, const float *gk, const float *gq, const float *gc,
                           unsigned E, unsigned hc, float eps) {
    const unsigned s = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const uint64_t base = ((uint64_t)t * hc + s) * E;
    __shared__ float red[32];
    float sk = 0, sq = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) { sk += key[base + i] * key[base + i]; sq += R[base + i] * R[base + i]; }
    const float ik = rsqrtf(block_sum(sk, red) / E + eps);
    const float iq = rsqrtf(block_sum(sq, red) / E + eps);
    float dot = 0;
    for (unsigned i = tid; i < E; i += blockDim.x)
        dot += (key[base + i] * ik * gk[s * E + i]) * (R[base + i] * iq * gq[s * E + i]);
    const float a = block_sum(dot, red) * rsqrtf((float)E);
    const float mag = sqrtf(fmaxf(fabsf(a), 1e-6f));
    const float gate = sigmoid(a > 0 ? mag : a < 0 ? -mag : 0);
    float ss = 0;
    for (unsigned i = tid; i < E; i += blockDim.x) {
        const float v = gate * val[(uint64_t)t * E + i];
        gated[base + i] = v;
        ss += v * v;
    }
    const float inv = rsqrtf(block_sum(ss, red) / E + eps);
    for (unsigned i = tid; i < E; i += blockDim.x) normed[base + i] = gated[base + i] * inv * gc[s * E + i];
}

__global__ void ngram_conv(float *R, const float *gated, const float *normed, float *history,
                           const char *w, unsigned type, unsigned T, unsigned C, unsigned K,
                           unsigned dilation, float *snap, unsigned st, float *snap2, unsigned st2) {
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const unsigned H = (K - 1) * dilation;
    float hist[9], taps[4];
    for (unsigned i = 0; i < H; i++) hist[i] = history[(uint64_t)i * C + c];
    for (unsigned i = 0; i < K; i++) taps[i] = scalar(w, c * K + i, type);
    for (unsigned t = 0; t < T; t++) {
        const uint64_t p = (uint64_t)t * C + c;
        const float cur = normed[p];
        float v = taps[K - 1] * cur;
        for (unsigned i = 0; i < K - 1; i++) v += taps[i] * hist[i * dilation];
        for (unsigned i = 0; i + 1 < H; i++) hist[i] = hist[i + 1];
        hist[H - 1] = cur;
        R[p] += gated[p] + silu(v);
        if (snap && t == st) for (unsigned i = 0; i < H; i++) snap[(uint64_t)i * C + c] = hist[i];
        if (snap2 && t == st2) for (unsigned i = 0; i < H; i++) snap2[(uint64_t)i * C + c] = hist[i];
    }
    for (unsigned i = 0; i < H; i++) history[(uint64_t)i * C + c] = hist[i];
}

} // namespace qwen4_rocm

extern "C" int ds4_gpu_qwen4_conv_stream_tensor(ds4_gpu_tensor *x, ds4_gpu_tensor *history,
        const void *map, uint64_t size, uint64_t offset, uint32_t T, uint32_t C, uint32_t K, bool activate) {
    using namespace qwen4_rocm;
    if (!T || !C || K < 2 || K > 4 || !tensor(x, (uint64_t)T * C * 4) || !tensor(history, (uint64_t)(K - 1) * C * 4)) return 0;
    const char *w = weight(map, size, offset, (uint64_t)C * K * 4);
    if (!w) return 0;
    conv<<<(C + 255) / 256, 256, 0, 0>>>((float *)x->ptr, (float *)history->ptr,
        (const float *)w, T, C, K, activate, NULL, UINT_MAX, NULL, UINT_MAX);
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_prep_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *a, ds4_gpu_tensor *b,
        const void *map, uint64_t size, uint64_t ao, uint64_t bo, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D) {
    using namespace qwen4_rocm;
    if (!T || !Hk || !Hv || D < 32 || D > 128 || D % 32 ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4)) return 0;
    const char *A = weight(map, size, ao, (uint64_t)Hv * 4), *bias = weight(map, size, bo, (uint64_t)Hv * 4);
    if (!A || !bias) return 0;
    gdn_prep<<<dim3(Hk, T), 32, 0, 0>>>((float *)qkv->ptr,
        (float *)a->ptr, (float *)b->ptr, (const float *)A, (const float *)bias, Hk, Hv, D);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_verify_rows_exact(bool on) { (void)on; }

extern "C" int ds4_gpu_qwen4_gdn_scan_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *qkv, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b,
        uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_rocm;
    const uint64_t bytes = (uint64_t)Hv * D * D * 4;
    if (!T || !Hk || !Hv || Hv % Hk || D < 32 || D > 128 || D % 32 ||
        !tensor(out, (uint64_t)T * Hv * D * 4) || !tensor(state, bytes) ||
        !tensor(qkv, (uint64_t)T * (2 * Hk + Hv) * D * 4) ||
        !tensor(a, (uint64_t)T * Hv * 4) || !tensor(b, (uint64_t)T * Hv * 4) ||
        (snap && !tensor(snap, bytes)) || (snap2 && !tensor(snap2, bytes))) return 0;
#define QWEN_GDN(ROWS, DIM) gdn_scan<ROWS,DIM><<<dim3((DIM+4*ROWS-1)/(4*ROWS),Hv),128,0,0>>>((float *)out->ptr, \
        (float *)state->ptr,(const float *)qkv->ptr,(const float *)a->ptr,(const float *)b->ptr, \
        T,Hk,Hv,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2)
    // Use eight state rows per wave for gfx1151 prefill.
    if (D == 128 && T > 8 && ds4_rocm_is_gfx1151()) {
        QWEN_GDN(8,128);
        return launched();
    }
#define QWEN_GDN_DIM(DIM) case DIM: if (T > 8) { QWEN_GDN(4,DIM); } else { QWEN_GDN(1,DIM); } break
    switch (D) { QWEN_GDN_DIM(32); QWEN_GDN_DIM(64); QWEN_GDN_DIM(96); QWEN_GDN_DIM(128); }
#undef QWEN_GDN_DIM
#undef QWEN_GDN
    return launched();
}

extern "C" int ds4_gpu_qwen4_gdn_out_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *z,
        const void *map, uint64_t size, uint64_t off, uint32_t T, uint32_t H, uint32_t D, float eps) {
    using namespace qwen4_rocm;
    const uint64_t n = (uint64_t)T * H * D;
    if (!n || D < 32 || D > 128 || D % 32 || !tensor(out, n * 4) || !tensor(z, n * 4)) return 0;
    const char *w = weight(map, size, off, (uint64_t)D * 4);
    if (!w) return 0;
    gdn_out<<<dim3(H, T), 32, 0, 0>>>((float *)out->ptr, (const float *)z->ptr, (const float *)w, H, D, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_gate_tensor(ds4_gpu_tensor *gated, ds4_gpu_tensor *normed,
        const ds4_gpu_tensor *R, const ds4_gpu_tensor *key, const ds4_gpu_tensor *val,
        const void *map, uint64_t size, uint64_t ko, uint64_t qo, uint64_t co,
        uint32_t T, uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_rocm;
    const uint64_t bytes = (uint64_t)T * E * hc * 4, wb = (uint64_t)E * hc * 4;
    if (!T || !E || !hc || hc > 4 || !tensor(gated, bytes) || !tensor(normed, bytes) ||
        !tensor(R, bytes) || !tensor(key, bytes) || !tensor(val, (uint64_t)T * E * 4)) return 0;
    const char *gk = weight(map, size, ko, wb), *gq = weight(map, size, qo, wb), *gc = weight(map, size, co, wb);
    if (!gk || !gq || !gc) return 0;
    ngram_gate<<<dim3(hc, T), 128, 0, 0>>>((float *)gated->ptr, (float *)normed->ptr,
        (const float *)R->ptr, (const float *)key->ptr, (const float *)val->ptr,
        (const float *)gk, (const float *)gq, (const float *)gc, E, hc, eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_ple_conv_tensor(ds4_gpu_tensor *R, const ds4_gpu_tensor *gated,
        const ds4_gpu_tensor *normed, ds4_gpu_tensor *history, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t T, uint32_t C, uint32_t K, uint32_t dilation,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_rocm;
    const uint64_t n = (uint64_t)T * C * 4, hb = (uint64_t)(K - 1) * dilation * C * 4;
    if (!T || !C || K < 2 || K > 4 || !dilation || dilation > 3 ||
        (type != 0 && type != 1) || !tensor(R, n) || !tensor(gated, n) || !tensor(normed, n) ||
        !tensor(history, hb) || (snap && !tensor(snap, hb)) || (snap2 && !tensor(snap2, hb))) return 0;
    const char *w = weight(map, size, off, row_bytes(type, (uint64_t)C * K));
    if (!w) return 0;
    ngram_conv<<<(C + 255) / 256, 256, 0, 0>>>((float *)R->ptr,
        (const float *)gated->ptr, (const float *)normed->ptr, (float *)history->ptr, w, type, T, C, K,
        dilation, snap ? (float *)snap->ptr : NULL, st, snap2 ? (float *)snap2->ptr : NULL, st2);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_norm_tensor(ds4_gpu_tensor *xn, ds4_gpu_tensor *inj,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t go, uint64_t io,
        uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    using namespace qwen4_rocm;
    const uint64_t n = (uint64_t)T * E * hc;
    if (!T || !E || !hc || hc > 8 || ni > 4 ||
        !tensor(xn, n * 4) || !tensor(R, n * 4) ||
        (ni && !tensor(inj, (uint64_t)T * hc * 8 * ni * 4))) return 0;
    const char *gamma = weight(map, size, go, (uint64_t)E * hc * 4);
    const char *wi = ni ? weight(map, size, io, row_bytes(type, (uint64_t)E * hc) * ni) : gamma;
    if (!gamma || !wi || (type != 0 && type != 1 && type != 8)) return 0;
    // Keep the same reduction schedule for decode, append and bulk prefill.
    // Reuse one RMS reduction while preserving all eight injection partials.
    if (T >= 128 && E == 2560 && hc == 4 && (ni == 0 || (ni == 4 && type == 1)) && ds4_rocm_is_gfx1151()) {
        hc_norm_one<<<dim3(hc, T), 128, 0, 0>>>((float *)xn->ptr,
            ni ? (float *)inj->ptr : NULL, (const float *)R->ptr, (const float *)gamma, wi, type, E, hc, ni, eps);
    } else {
        hc_norm<<<dim3(hc * 8, T), 128, 0, 0>>>((float *)xn->ptr,
            ni ? (float *)inj->ptr : NULL, (const float *)R->ptr, (const float *)gamma, wi, type, E, hc, ni, eps);
    }
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_gate_mix_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *xn, const ds4_gpu_tensor *lo, const void *map, uint64_t size,
        uint64_t offset, uint32_t type, uint32_t T, uint32_t E, uint32_t hc, uint32_t rank) {
    using namespace qwen4_rocm;
    if (!T || !E || !hc || hc > 4 || !rank || !tensor(out, (uint64_t)T * E * 4) ||
        !tensor(xn, (uint64_t)T * E * hc * 4) || !tensor(lo, (uint64_t)T * rank * 4)) return 0;
    const char *w = weight(map, size, offset, row_bytes(type, rank) * E * hc);
    if (!w || (type != 0 && type != 1 && type != 8)) return 0;
#define QWEN_HC(TYPE) hc_mix<TYPE><<<dim3((E + 3) / 4, T),128,0,0>>>((float *)out->ptr, \
        (const float *)xn->ptr,(const float *)lo->ptr,w,E,hc,rank,row_bytes(TYPE,rank))
    if (type == 0) { QWEN_HC(0); }
    else if (type == 1) { QWEN_HC(1); }
    else { QWEN_HC(8); }
#undef QWEN_HC
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_combine_tensor(ds4_gpu_tensor *R,
        const ds4_gpu_tensor *out, const ds4_gpu_tensor *inj, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_rocm;
    if (!T || !E || !hc || hc > 4 || !tensor(R, (uint64_t)T * E * hc * 4) ||
        !tensor(out, (uint64_t)T * E * 4) || !tensor(inj, (uint64_t)T * hc * hc * 8 * 4)) return 0;
    hc_combine<<<dim3((E + 255) / 256, T), 256, 0, 0>>>(
        (float *)R->ptr, (const float *)out->ptr, (const float *)inj->ptr, E, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_lo_act_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src, uint32_t T, uint32_t hc, uint32_t rank) {
    using namespace qwen4_rocm;
    const uint64_t n = (uint64_t)T * rank;
    if (!n || !hc || !tensor(dst, n * 4) || !tensor(src, n * 4)) return 0;
    hc_lo<<<(n + 255) / 256, 256, 0, 0>>>((float *)dst->ptr, (const float *)src->ptr, n, hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_hc_mix_rows_tensor(ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *up, const ds4_gpu_tensor *xn, uint32_t T, uint32_t E, uint32_t hc) {
    using namespace qwen4_rocm;
    if (!T || !E || !hc || hc > 4 || !tensor(dst, (uint64_t)T * E * 4) ||
        !tensor(up, (uint64_t)T * hc * E * 4) || !tensor(xn, (uint64_t)T * hc * E * 4)) return 0;
    hc_rows<<<dim3((E + 255) / 256, T), 256, 0, 0>>>(
        (float *)dst->ptr, (const float *)up->ptr, (const float *)xn->ptr, E, hc);
    return launched();
}

extern "C" void ds4_gpu_qwen4_set_rope(const float *freq, uint32_t n, float scale) {
    qwen4_rocm::rope_set = freq != NULL;
    qwen4_rocm::rope_scale = freq ? scale : 1;
    memset(qwen4_rocm::rope_freq, 0, sizeof(qwen4_rocm::rope_freq));
    if (freq) memcpy(qwen4_rocm::rope_freq, freq, std::min(n, 32u) * sizeof(float));
}

extern "C" int ds4_gpu_qwen4_attn_prep_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *kc, ds4_gpu_tensor *vc, ds4_gpu_tensor *iqout, ds4_gpu_tensor *ikc,
        const ds4_gpu_tensor *qg, const ds4_gpu_tensor *kp, const ds4_gpu_tensor *vp,
        const ds4_gpu_tensor *iq, const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3,
        const void *map, uint64_t size, uint64_t qo, uint64_t ko, uint64_t io,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t nrot,
        uint32_t Hi, uint32_t Di, uint32_t pos0, uint32_t cap, float base, float eps) {
    using namespace qwen4_rocm;
    const uint64_t qb = (uint64_t)T * H * D * 4, kb = (uint64_t)T * Hkv * D * 4, ib = (uint64_t)T * Hi * Di * 4;
    if (!T || !H || !Hkv || H % Hkv || !Hi || D < 32 || D > 256 || D % 32 ||
        Di < 32 || Di > 128 || Di % 32 || nrot > 64 || nrot > D || nrot > Di || nrot % 2 ||
        (uint64_t)pos0 + T > cap || !tensor(q, qb) || !tensor(gate, qb) || !tensor(qg, qb * 2) ||
        !tensor(kp, kb) || !tensor(vp, kb) || !tensor(iq, ib) || !tensor(iqout, ib) ||
        !tensor(ik, (uint64_t)T * Di * 4) || !tensor(ikc, (uint64_t)cap * Di * 4) ||
        !tensor(kc, (uint64_t)cap * Hkv * D * 2) || !tensor(vc, (uint64_t)cap * Hkv * D * 2) ||
        !tensor(pos3, (uint64_t)cap * 16)) return 0;
    const char *gq = weight(map, size, qo, D * 4), *gk = weight(map, size, ko, D * 4), *giq = weight(map, size, io, Di * 4);
    if (!gq || !gk || !giq) return 0;
    attn_prep<<<dim3(H + Hkv + Hi + 1, T), 32, 0, 0>>>((float *)q->ptr,
        (float *)gate->ptr, (__half *)kc->ptr, (__half *)vc->ptr, (float *)iqout->ptr, (float *)ikc->ptr,
        (const float *)qg->ptr, (const float *)kp->ptr, (const float *)vp->ptr, (const float *)iq->ptr,
        (const float *)ik->ptr, (const uint32_t *)pos3->ptr, (const float *)gq, (const float *)gk, (const float *)giq,
        H, Hkv, D, Hi, Di, pos0, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_block_key_tensor(ds4_gpu_tensor *out,
        const ds4_gpu_tensor *ik, const ds4_gpu_tensor *pos3, const void *map, uint64_t size,
        uint64_t off, uint32_t b0, uint32_t N, uint32_t ratio, uint32_t D, uint32_t nrot,
        float base, float eps) {
    using namespace qwen4_rocm;
    const uint64_t rows = (uint64_t)b0 + N;
    if (!N || !ratio || D < 32 || D > 128 || D % 32 || nrot > D || nrot > 64 || nrot % 2 ||
        !tensor(out, rows * D * 2) || !tensor(ik, rows * ratio * D * 4) || !tensor(pos3, rows * ratio * 16)) return 0;
    const char *w = weight(map, size, off, D * 4);
    if (!w) return 0;
    block_key<<<N, 32, 0, 0>>>((__half *)out->ptr, (const float *)ik->ptr,
        (const uint32_t *)pos3->ptr, (const float *)w, b0, ratio, D, eps, rope(nrot, base));
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_score_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *tiles,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *key, uint32_t T, uint32_t N,
        uint32_t H, uint32_t D, uint32_t pos0, uint32_t ratio) {
    using namespace qwen4_rocm;
    const unsigned nt = (N + 7) / 8;
    if (!T || !N || !H || !D || !ratio || !tensor(out, (uint64_t)T * N * 4) ||
        !tensor(q, (uint64_t)T * H * D * 4) || !tensor(key, (uint64_t)N * D * 2) ||
        (tiles && !tensor(tiles, (uint64_t)T * nt * 4))) return 0;
    idx_score<<<dim3((N + 3) / 4, T), 128, 0, 0>>>((float *)out->ptr,
        (const float *)q->ptr, (const __half *)key->ptr, N, H, D, pos0, ratio);
    if (!launched()) return 0;
    if (tiles) tile_max<<<dim3((nt + 255) / 256, T), 256, 0, 0>>>(
        (unsigned *)tiles->ptr, (const float *)out->ptr, N, nt);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_select_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *score,
        const ds4_gpu_tensor *tiles, uint32_t N, uint32_t T, uint32_t K) {
    using namespace qwen4_rocm;
    (void)tiles;
    if (!T || !K || K > N || !tensor(out, (uint64_t)T * K * 4) || !tensor(score, (uint64_t)T * N * 4)) return 0;
    idx_select<<<T, 256, 0, 0>>>((int *)out->ptr, (const float *)score->ptr, N, K);
    return launched();
}

extern "C" int ds4_gpu_qwen4_idx_expand_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *count,
        const ds4_gpu_tensor *blocks, uint32_t T, uint32_t K, uint32_t ratio, uint32_t pos0, uint32_t stride) {
    using namespace qwen4_rocm;
    if (!T || !K || !ratio || (uint64_t)K * ratio + ratio - 1 > stride ||
        !tensor(out, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4) || !tensor(blocks, (uint64_t)T * K * 4)) return 0;
    idx_expand<<<T, 256, 0, 0>>>((int *)out->ptr, (unsigned *)count->ptr,
        (const int *)blocks->ptr, K, ratio, pos0, stride);
    return launched();
}

extern "C" uint64_t ds4_gpu_qwen4_attn_part_floats(uint32_t T, uint32_t H, uint32_t D) {
    return (uint64_t)T * H * 64 * (D + 2);
}

extern "C" int ds4_gpu_qwen4_dense_mm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *map, uint64_t size, uint64_t off, uint32_t type, uint32_t T, uint32_t K, uint32_t M) {
    using namespace qwen4_rocm;
    if (!T || !K || !M || !tensor(x, (uint64_t)T*K*4) || !tensor(out, (uint64_t)T*M*4)) return 0;
    const uint64_t rb = row_bytes(type,K);
    if (!rb) return 0;
    const char *w = weight(map, size, off, rb*M);
    if (!w) return 0;
    if (T <= 8) return matvec_dispatch((float *)out->ptr, w, (const float *)x->ptr, type, T, K, M);
    if (type == 1 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode)
        return dense_f16_blas((float *)out->ptr,(const float *)x->ptr,(const __half *)w,T,K,M);
    if (type == 8 && T >= 32 && T <= INT_MAX && K <= INT_MAX && M <= INT_MAX &&
        g_cublas_ready && !g_quality_mode)
        return dense_q8_blas((float *)out->ptr,(const float *)x->ptr,w,T,K,M);
    if (g_cublas_ready && T >= 32 && K <= INT_MAX && M <= INT_MAX && T <= INT_MAX)
        return dense_blas((float *)out->ptr,(const float *)x->ptr,w,type,T,K,M);
    return matrix_dispatch((float *)out->ptr, (const float *)x->ptr, w, NULL, NULL, NULL,
                           type, 1, T, 1, 1, K, M, 0, false);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *map,
        uint64_t size, uint64_t off, uint64_t K, uint64_t M, const ds4_gpu_tensor *x, uint64_t T) {
    if (K > UINT_MAX || M > UINT_MAX || T > UINT_MAX) return 0;
    return ds4_gpu_qwen4_dense_mm_tensor(out, x, map, size, off, 8, T, K, M);
}

extern "C" int ds4_gpu_qwen4_matmul_q8_0_weights_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *w,
        uint32_t K, uint32_t M, const ds4_gpu_tensor *x) {
    using namespace qwen4_rocm;
    const uint64_t rb = row_bytes(8, K);
    if (!K || !M || !rb || !tensor(w, rb*M) || !tensor(x, (uint64_t)K*4) || !tensor(out, (uint64_t)M*4)) return 0;
    return matvec_dispatch((float *)out->ptr, (const char *)w->ptr, (const float *)x->ptr, 8, 1, K, M);
}

extern "C" int ds4_gpu_qwen4_multi_gemv_tensor(const ds4_gpu_tensor *x, uint32_t T,
        uint32_t K, uint32_t N, ds4_gpu_tensor *const *outs, const void *map, uint64_t size,
        const uint64_t *offsets, const uint32_t *types, const uint32_t *rows) {
    if (!N || N > 4 || !outs || !offsets || !types || !rows) return 0;
    for (unsigned i = 0; i < N; i++)
        if (!ds4_gpu_qwen4_dense_mm_tensor(outs[i], x, map, size, offsets[i], types[i], T, K, rows[i])) return 0;
    return 1;
}

extern "C" int ds4_gpu_qwen4_q8_pair_tensor(ds4_gpu_tensor *o0, ds4_gpu_tensor *o1,
        const void *map, uint64_t size, uint64_t w0, uint64_t w1, uint64_t K,
        uint64_t M0, uint64_t M1, const ds4_gpu_tensor *x, uint64_t T) {
    if ((M0 & 1) || (M1 & 1)) return 0;
    return ds4_gpu_qwen4_matmul_q8_0_tensor(o0, map, size, w0, K, M0, x, T) &&
           ds4_gpu_qwen4_matmul_q8_0_tensor(o1, map, size, w1, K, M1, x, T);
}

extern "C" int ds4_gpu_qwen4_router_topk_tensor(ds4_gpu_tensor *sel, ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t K, ds4_gpu_tensor *sg, uint32_t T, uint32_t NE, uint32_t NS) {
    using namespace qwen4_rocm;
    if (!T || !NE || NE > 512 || !NS || NS > NE || NS > 32 ||
        !tensor(sel, (uint64_t)T*NS*4) || !tensor(weights, (uint64_t)T*NS*4) || !tensor(logits, (uint64_t)T*NE*4)) return 0;
    const char *wg = K ? weight(map, size, off, row_bytes(type, K)) : NULL;
    if (K && (!wg || !tensor(x, (uint64_t)T*K*4) || !tensor(sg, (uint64_t)T*4))) return 0;
    router<<<T,256,0,0>>>((int *)sel->ptr, (float *)weights->ptr, (const float *)logits->ptr,
        K ? (const float *)x->ptr : NULL, wg, K ? (float *)sg->ptr : NULL, NE, NS, K, type);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_mid_tensor(ds4_gpu_tensor *mid, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t go, uint64_t uo,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t sgo, uint64_t suo, uint32_t st) {
    using namespace qwen4_rocm;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(mid, (uint64_t)T*NO*M*4) ||
        !tensor(x, (uint64_t)T*K*4) || !tensor(sel, (uint64_t)T*NS*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type, K)*M*NE, sb = row_bytes(st, K)*M;
    const char *g = weight(map,size,go,bytes), *u = weight(map,size,uo,bytes);
    const char *sg = st != UINT_MAX ? weight(map,size,sgo,sb) : NULL;
    const char *su = st != UINT_MAX ? weight(map,size,suo,sb) : NULL;
    if (!g || !u || (st != UINT_MAX && (!sg || !su))) return 0;
    return moe_mv_dispatch((float *)mid->ptr,(const float *)x->ptr,(const int *)sel->ptr,
                           g,u,sg,su,type,st,NE,T,NS,K,M,false);
}

extern "C" int ds4_gpu_qwen4_moe_down_tensor(ds4_gpu_tensor *part, const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *sel, const void *map, uint64_t size, uint64_t off,
        uint32_t type, uint32_t NE, uint32_t T, uint32_t NS, uint32_t K, uint32_t M,
        uint64_t so, uint32_t st) {
    using namespace qwen4_rocm;
    const unsigned NO = NS + (st != UINT_MAX);
    if (!T || !NE || !NS || !K || !M || !tensor(part,(uint64_t)T*NO*M*4) ||
        !tensor(mid,(uint64_t)T*NO*K*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    const char *w = weight(map,size,off,expert_row_bytes(type,K)*M*NE);
    const char *sw = st != UINT_MAX ? weight(map,size,so,row_bytes(st,K)*M) : NULL;
    if (!w || (st != UINT_MAX && !sw)) return 0;
    return moe_mv_dispatch((float *)part->ptr,(const float *)mid->ptr,(const int *)sel->ptr,
                           w,NULL,sw,NULL,type,st,NE,T,NS,K,M,true);
}

extern "C" int ds4_gpu_qwen4_moe_reduce_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *part,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *sg, const ds4_gpu_tensor *sh,
        ds4_gpu_tensor *R, const ds4_gpu_tensor *inj, uint32_t T, uint32_t NS, uint32_t stride,
        uint32_t D, uint32_t hc) {
    using namespace qwen4_rocm;
    if (!T || !NS || !D || stride < NS + (sg && !sh) || hc > 4 ||
        !tensor(out,(uint64_t)T*D*4) || !tensor(part,(uint64_t)T*stride*D*4) ||
        !tensor(weights,(uint64_t)T*NS*4) || (sg && !tensor(sg,(uint64_t)T*4)) ||
        (sh && !tensor(sh,(uint64_t)T*D*4)) ||
        (hc && (!tensor(R,(uint64_t)T*hc*D*4) || !tensor(inj,(uint64_t)T*hc*hc*8*4)))) return 0;
    moe_reduce<<<dim3((D+255)/256,T),256,0,0>>>((float *)out->ptr,
        hc ? (float *)R->ptr : NULL, hc ? (const float *)inj->ptr : NULL, (const float *)part->ptr,
        (const float *)weights->ptr, sg ? (const float *)sg->ptr : NULL, sh ? (const float *)sh->ptr : NULL,
        NS,stride,D,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_moe_build_lists_tensor(ds4_gpu_tensor *lists, ds4_gpu_tensor *counts,
        const ds4_gpu_tensor *sel, uint32_t T, uint32_t NS, uint32_t NE, uint32_t cap) {
    using namespace qwen4_rocm;
    if (!T || !NS || NS > NE || !NE || NE > 512 || cap < T || (uint64_t)T*NS > INT_MAX ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4) || !tensor(sel,(uint64_t)T*NS*4)) return 0;
    expert_lists<<<1,256,0,0>>>((int *)lists->ptr,(int *)counts->ptr,(const int *)sel->ptr,T*NS,NE,cap);
    return launched();
}

static int qwen4_moe_mm(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t o0, uint64_t o1, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap, bool down) {
    using namespace qwen4_rocm;
    if (!T || !NE || !NS || NS > NE || NO < NS || !K || !M || cap < T ||
        !tensor(out,(uint64_t)T*NO*M*4) || !tensor(x,(uint64_t)T*(down ? NO : 1)*K*4) ||
        !tensor(lists,(uint64_t)NE*cap*4) || !tensor(counts,(uint64_t)NE*4)) return 0;
    const uint64_t bytes = expert_row_bytes(type,K)*M*NE;
    const char *w0 = weight(map,size,o0,bytes), *w1 = down ? NULL : weight(map,size,o1,bytes);
    if (!w0 || (!down && !w1)) return 0;
    return matrix_dispatch((float *)out->ptr,(const float *)x->ptr,w0,w1,(const int *)lists->ptr,
        (const int *)counts->ptr,type,NE,T,NS,NO,K,M,cap,down);
}

extern "C" int ds4_gpu_qwen4_moe_mm_mid_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t go, uint64_t uo, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,go,uo,type,NE,T,NS,NO,K,M,cap,false);
}

extern "C" int ds4_gpu_qwen4_moe_mm_down_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *lists, const ds4_gpu_tensor *counts, const void *map, uint64_t size,
        uint64_t off, uint32_t type, uint32_t NE, uint32_t T, uint32_t NS,
        uint32_t NO, uint32_t K, uint32_t M, uint32_t cap) {
    return qwen4_moe_mm(out,x,lists,counts,map,size,off,0,type,NE,T,NS,NO,K,M,cap,true);
}

extern "C" int ds4_gpu_qwen4_gdn_front_tensor(ds4_gpu_tensor *qkv, ds4_gpu_tensor *state,
        const ds4_gpu_tensor *mixed, ds4_gpu_tensor *ga, ds4_gpu_tensor *gb,
        const void *map, uint64_t size, uint64_t co, uint64_t ao, uint64_t bo, uint64_t so, uint64_t dto,
        uint32_t type, uint32_t T, uint32_t Hk, uint32_t Hv, uint32_t D, uint32_t CK, uint32_t K,
        ds4_gpu_tensor *snap, uint32_t st, ds4_gpu_tensor *snap2, uint32_t st2) {
    using namespace qwen4_rocm;
    const uint64_t C = ((uint64_t)2*Hk+Hv)*D, hb = (CK-1)*C*4;
    if (!T || !Hk || !Hv || !D || !K || C > UINT_MAX || CK < 2 || CK > 4 ||
        !tensor(qkv,(uint64_t)T*C*4) || !tensor(state,hb) ||
        (snap && (st >= T || !tensor(snap,hb))) || (snap2 && (st2 >= T || !tensor(snap2,hb)))) return 0;
    const char *w = weight(map,size,co,C*CK*4);
    if (!w || !ds4_gpu_qwen4_dense_mm_tensor(ga,mixed,map,size,ao,type,T,K,Hv) ||
              !ds4_gpu_qwen4_dense_mm_tensor(gb,mixed,map,size,bo,type,T,K,Hv)) return 0;
    conv<<<(C+255)/256,256,0,0>>>((float *)qkv->ptr,(float *)state->ptr,(const float *)w,
        T,C,CK,true,snap ? (float *)snap->ptr : NULL,st,snap2 ? (float *)snap2->ptr : NULL,st2);
    return launched() && ds4_gpu_qwen4_gdn_prep_tensor(qkv,ga,gb,map,size,so,dto,T,Hk,Hv,D);
}

extern "C" int ds4_gpu_qwen4_decode_fusions_enabled(void) { return 1; }

extern "C" int ds4_gpu_qwen4_hc_combine_norm_tensor(ds4_gpu_tensor *next, const ds4_gpu_tensor *blk,
        const ds4_gpu_tensor *oldinj, ds4_gpu_tensor *xn, ds4_gpu_tensor *inj, const ds4_gpu_tensor *R,
        const void *map, uint64_t size, uint64_t go, uint64_t io, uint32_t type,
        uint32_t T, uint32_t E, uint32_t hc, uint32_t ni, float eps) {
    const uint64_t bytes = (uint64_t)T*E*hc*4;
    if (!qwen4_rocm::tensor(next,bytes) || !qwen4_rocm::tensor(R,bytes) || next->ptr == R->ptr ||
        !inj || !oldinj || inj->ptr == oldinj->ptr) return 0;
    if (!cuda_ok(cudaMemcpyAsync(next->ptr,R->ptr,bytes,cudaMemcpyDeviceToDevice,0),"Qwen HC copy")) return 0;
    return ds4_gpu_qwen4_hc_combine_tensor(next,blk,oldinj,T,E,hc) &&
           ds4_gpu_qwen4_hc_norm_tensor(xn,inj,next,map,size,go,io,type,T,E,hc,ni,eps);
}

extern "C" int ds4_gpu_qwen4_mtp_stage_tensor(ds4_gpu_tensor *cat, const ds4_gpu_tensor *e,
        const ds4_gpu_tensor *R, const void *map, uint64_t size, uint64_t eo, uint64_t ho,
        uint32_t E, uint32_t hc, float eps) {
    using namespace qwen4_rocm;
    if (!E || !hc || hc > 4 || !tensor(cat,(uint64_t)(hc+1)*2*E*4) ||
        !tensor(e,(uint64_t)E*4) || !tensor(R,(uint64_t)hc*E*4)) return 0;
    const char *ge = weight(map,size,eo,(uint64_t)E*4), *gh = weight(map,size,ho,(uint64_t)hc*E*4);
    if (!ge || !gh) return 0;
    mtp_stage<<<hc+1,256,0,0>>>((float *)cat->ptr,(const float *)e->ptr,
        (const float *)R->ptr,(const float *)ge,(const float *)gh,E,hc,eps);
    return launched();
}

extern "C" int ds4_gpu_qwen4_mtp_combine_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *proj,
        uint32_t E, uint32_t hc) {
    using namespace qwen4_rocm;
    if (!E || !hc || hc > 4 || !tensor(out,(uint64_t)hc*E*4) || !tensor(proj,(uint64_t)(hc+1)*E*4)) return 0;
    mtp_combine<<<((uint64_t)E*hc+255)/256,256,0,0>>>((float *)out->ptr,(const float *)proj->ptr,E,hc);
    return launched();
}

extern "C" int ds4_gpu_qwen4_argmax_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *scratch,
        const ds4_gpu_tensor *logits, uint32_t N) {
    using namespace qwen4_rocm;
    const uint64_t blocks = ((uint64_t)N+4095)/4096;
    if (!N || !tensor(out,4) || !tensor(logits,(uint64_t)N*4) || !tensor(scratch,blocks*sizeof(max_pair))) return 0;
    argmax<false><<<blocks,256,0,0>>>((int *)out->ptr,(max_pair *)scratch->ptr,(const float *)logits->ptr,N);
    argmax<true><<<1,256,0,0>>>((int *)out->ptr,(max_pair *)scratch->ptr,NULL,blocks);
    return launched();
}

extern "C" int ds4_gpu_qwen4_vision_encode(float *out, const float *patches, const float *pos_embed,
        uint32_t N, uint32_t grid_w, const void *map, uint64_t size, const ds4_qwen4_vision_weights *w) {
    using namespace qwen4_rocm;
    if (!out || !patches || !pos_embed || !w || !map || !N || N > 65535 || ds4_gpu_commands_active()) return 0;
    const unsigned E = w->n_embd, FF = w->n_ff, H = w->n_head, D = H ? E/H : 0;
    const unsigned P = w->n_patch, M = w->n_merge, O = w->n_out;
    if (M != 2 || !grid_w || grid_w%M || N%grid_w || (N/grid_w)%M ||
        (D != 64 && D != 72) || H*D != E || !FF || !O || !P || P > 32 || E%32) return 0;
    const unsigned IP = 3*P*P, ME = E*M*M, merged = N/(M*M);
    std::vector<ds4_gpu_tensor *> buffers;
    auto alloc = [&](uint64_t n) {
        ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n*4);
        buffers.push_back(t);
        return t;
    };
    ds4_gpu_tensor *patch = alloc((uint64_t)N*IP), *a0 = alloc((uint64_t)N*E), *a1 = alloc((uint64_t)N*E);
    ds4_gpu_tensor *pos = alloc((uint64_t)N*E), *x = alloc((uint64_t)N*E), *tmp = alloc((uint64_t)N*E);
    ds4_gpu_tensor *qkv = alloc((uint64_t)N*3*E), *q = alloc((uint64_t)N*E), *k = alloc((uint64_t)N*E);
    ds4_gpu_tensor *v = alloc((uint64_t)N*E), *attn = alloc((uint64_t)N*E), *ffn = alloc((uint64_t)N*FF);
    ds4_gpu_tensor *m0 = alloc((uint64_t)merged*ME), *res = alloc((uint64_t)merged*O);
    bool ok = true, active = false;
    for (const auto b : buffers) if (!b) ok = false;
    auto ptr = [](ds4_gpu_tensor *t) { return (float *)t->ptr; };
    auto wf = [&](uint64_t off, unsigned n) { return (const float *)weight(map,size,off,(uint64_t)n*4); };
    auto mm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t off, unsigned type,
                  unsigned rows, unsigned K, unsigned M) {
        return ds4_gpu_qwen4_dense_mm_tensor(dst,src,map,size,off,type,rows,K,M);
    };
    auto norm = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t wo, uint64_t bo) {
        const float *wgt = wf(wo,E), *bias = wf(bo,E);
        if (!wgt || !bias) return 0;
        vis_norm<<<N,256,0,0>>>(ptr(dst),(const float *)src->ptr,wgt,bias,E,w->eps);
        return launched();
    };
    auto add = [&](ds4_gpu_tensor *dst, const ds4_gpu_tensor *src, uint64_t bo, unsigned rows,
                   unsigned width, unsigned mode) {
        const float *bias = wf(bo,width);
        if (!bias) return 0;
        vis_add<<<((uint64_t)rows*width+255)/256,256,0,0>>>(ptr(dst),
            src ? (const float *)src->ptr : NULL,bias,rows,width,mode);
        return launched();
    };
    do {
        if (!ok) break;
        ok = ds4_gpu_tensor_write(patch,0,patches,(uint64_t)N*IP*4) &&
             ds4_gpu_tensor_write(pos,0,pos_embed,(uint64_t)N*E*4);
        if (!ok || !(active = ds4_gpu_begin_commands())) { ok = false; break; }
        ok = mm(a0,patch,w->patch_w0,w->patch_type,N,IP,E) && mm(a1,patch,w->patch_w1,w->patch_type,N,IP,E);
        const float *pb = wf(w->patch_b,E);
        if (!ok || !pb) { ok = false; break; }
        vis_patch<<<((uint64_t)N*E+255)/256,256,0,0>>>(ptr(x),ptr(a0),ptr(a1),pb,ptr(pos),N,E);
        ok = launched();
        for (unsigned l = 0; ok && l < DS4_QWEN4_VISION_LAYERS; l++) {
            const auto &lw = w->layer[l];
            ok = norm(tmp,x,lw.ln1_w,lw.ln1_b) && mm(qkv,tmp,lw.qkv_w,lw.qkv_type,N,E,3*E);
            const float *qb = wf(lw.qkv_b,3*E);
            if (!ok || !qb) { ok = false; break; }
            vis_qkv<<<N,256,0,0>>>(ptr(q),ptr(k),ptr(v),ptr(qkv),qb,H,D,grid_w);
            vis_attention<<<dim3((N + VIS_ATTN_WAVES - 1) / VIS_ATTN_WAVES, H),
                32*VIS_ATTN_WAVES, 2*VIS_ATTN_KEY_TILE*D*sizeof(float), 0>>>(
                    ptr(attn),ptr(q),ptr(k),ptr(v),N,H,D);
            ok = launched() && mm(tmp,attn,lw.out_w,lw.out_type,N,E,E) && add(x,tmp,lw.out_b,N,E,2) &&
                 norm(tmp,x,lw.ln2_w,lw.ln2_b) && mm(ffn,tmp,lw.up_w,lw.up_type,N,E,FF) &&
                 add(ffn,NULL,lw.up_b,N,FF,0) && mm(tmp,ffn,lw.down_w,lw.down_type,N,FF,E) &&
                 add(x,tmp,lw.down_b,N,E,2);
        }
        ok = ok && norm(tmp,x,w->post_ln_w,w->post_ln_b) && mm(m0,tmp,w->mm0_w,w->mm0_type,merged,ME,ME) &&
             add(m0,NULL,w->mm0_b,merged,ME,1) && mm(res,m0,w->mm2_w,w->mm2_type,merged,ME,O) &&
             add(res,NULL,w->mm2_b,merged,O,2);
    } while (false);
    if (active && !ds4_gpu_end_commands()) ok = false;
    if (ok) ok = ds4_gpu_tensor_read(res,0,out,(uint64_t)merged*O*4);
    for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) ds4_gpu_tensor_free(*it);
    return ok;
}

#include "ds4_rocm_qwen4_attention_wmma.cuh"

extern "C" int ds4_gpu_qwen4_attn_decode_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *gate, const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc,
        const ds4_gpu_tensor *sel, const ds4_gpu_tensor *count, ds4_gpu_tensor *partial,
        uint32_t T, uint32_t H, uint32_t Hkv, uint32_t D, uint32_t pos0, bool sparse, uint32_t stride, float scale) {
    using namespace qwen4_rocm;
    const uint64_t n = (uint64_t)T * H * D * 4, cb = ((uint64_t)pos0 + T) * Hkv * D * 2;
    if (!T || !H || !Hkv || H % Hkv || (D != 32 && D != 128 && D != 256) ||
        !tensor(out, n) || !tensor(q, n) || !tensor(gate, n) || !tensor(kc, cb) || !tensor(vc, cb) ||
        (sparse && (!stride || !tensor(sel, (uint64_t)T * stride * 4) || !tensor(count, (uint64_t)T * 4)))) return 0;
    const unsigned keys = sparse ? stride : pos0 + T;
    const unsigned splits = partial ? std::min(64u, (keys + 31) / 32) : 1;
    if (partial && !tensor(partial, (uint64_t)T * H * splits * (D + 2) * 4)) return 0;
    // Preserve the quality-mode and narrow-token paths. The mask scratch is
    // consumed on the same stream before the existing temporary arena is reused.
    if (!g_quality_mode && ds4_rocm_is_gfx1151() && T >= 512 &&
        H == 24 && Hkv == 2 && D == 256 && scale == 0.0625f &&
        (uint64_t)pos0 + T <= 65536 && (!sparse || stride == 2052)) {
        using namespace qwen4_wmma_attention;
        const float *qp=(const float *)q->ptr, *gp=(const float *)gate->ptr;
        const __half *kp=(const __half *)kc->ptr, *vp=(const __half *)vc->ptr;
        float *dst=(float *)out->ptr;
        if (sparse) {
            const unsigned words=(pos0+T+127)/128;
            unsigned *mask=(unsigned *)cuda_tmp_alloc((uint64_t)T*words*sizeof(unsigned), "Qwen attention mask");
            if (!mask) return 0;
            selected_to_mask<<<T,256,0,0>>>(mask,(const int *)sel->ptr,(const unsigned *)count->ptr,stride,words,pos0);
            if (!launched()) return 0;
            WmmaCausalAttentionKernel<4,16,true,false><<<dim3((T+3)/4,2),256,0,0>>>(qp,gp,kp,vp,mask,words,dst,pos0,T);
        } else {
            WmmaCausalAttentionKernel<16,16,false,true><<<dim3((T+15)/16,12),256,0,0>>>(qp,gp,kp,vp,NULL,0,dst,pos0,T);
        }
        return launched();
    }
    const dim3 grid((H + 3) / 4, T, splits);
#define QWEN_ATTN(DIM) attention<DIM><<<grid, 128, 0, 0>>>((float *)out->ptr, \
        partial ? (float *)partial->ptr : NULL, (const float *)q->ptr, (const float *)gate->ptr, \
        (const __half *)kc->ptr, (const __half *)vc->ptr, sparse ? (const int *)sel->ptr : NULL, \
        sparse ? (const unsigned *)count->ptr : NULL, H, Hkv, pos0, stride, sparse, splits, (keys + splits - 1) / splits, scale)
    if (D == 32) { QWEN_ATTN(32); }
    else if (D == 128) { QWEN_ATTN(128); }
    else { QWEN_ATTN(256); }
#undef QWEN_ATTN
    if (!launched()) return 0;
    if (splits > 1) attn_merge<<<dim3(H, T), D, 0, 0>>>((float *)out->ptr,
        (const float *)partial->ptr, (const float *)gate->ptr, H, D, splits);
    return launched();
}

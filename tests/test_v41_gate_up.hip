// Compare Halo gate/up with the current native MMQ operator at production shapes.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cuda/mmq/ds4_mmq.h"
#include "rocm/ds4_rocm_v41_gate_up.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); \
        exit(1); \
    } \
} while (0)
#define HIP(call) CHECK((call) == hipSuccess)

struct Buffer {
    unsigned char *base, *p;
    size_t n;

    explicit Buffer(size_t bytes) : n(bytes) {
        HIP(hipMalloc(&base, n + 512));
        p = base + 256; // Preserve the production scratch alignment.
        HIP(hipMemset(base, 0xa5, n + 512));
    }
    ~Buffer() { (void)hipFree(base); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    void check() const {
        unsigned char guards[512];
        HIP(hipMemcpy(guards, base, 256, hipMemcpyDeviceToHost));
        HIP(hipMemcpy(guards + 256, p + n, 256, hipMemcpyDeviceToHost));
        for (unsigned char byte : guards) CHECK(byte == 0xa5);
    }
};

// IQ2_XXS has a half scale followed by 64 packed bytes per 256-value block.
// Use finite scales and deterministic, nonuniform packed weights.
__global__ void fill_iq2(unsigned char *p, size_t n, unsigned salt) {
    size_t i = blockIdx.x * 256ull + threadIdx.x;
    if (i >= n) return;
    unsigned byte = i % 66;
    p[i] = byte == 0 ? 0 : byte == 1 ? 0x20 :
        (unsigned char)((i * 1664525ull + salt * 1013904223ull) >> 19);
}

int main(int argc, char **argv) {
    CHECK(argc == 3);
    unsigned rows = atoi(argv[1]), pattern = atoi(argv[2]);
    CHECK(rows == 2048 && pattern < 3);
    constexpr size_t K = 5120, M = 2304, E = 384, T = 6;
    constexpr size_t W = E * M * 1320;
    const size_t N = rows * T * M;
    Buffer gw(W), uw(W), x(rows * K * 4), ids(rows * T * 4);
    Buffer weights(rows * T * 4), pairs(rows * T * 4), offsets((E+1) * 4);
    Buffer gate(N * 4), scratch(N * 4), native(N * 4), nativeh(N * 2);
    Buffer fast(N * 4), fasth(N * 2);
    std::vector<Buffer *> buffers = {
        &gw, &uw, &x, &ids, &weights, &pairs, &offsets,
        &gate, &scratch, &native, &nativeh, &fast, &fasth
    };
    fill_iq2<<<(W + 255) / 256, 256>>>(gw.p, W, 17);
    fill_iq2<<<(W + 255) / 256, 256>>>(uw.p, W, 29);
    HIP(hipDeviceSynchronize());

    std::vector<float> hx(rows * K), hw(rows * T);
    std::vector<unsigned> hi(rows * T), hp(rows * T), ho(E+1, 0);
    for (size_t i = 0; i < hx.size(); i++)
        hx[i] = std::sin(float(i % 104729) * .013f) * .25f;
    for (unsigned r = 0; r < rows; r++) {
        for (unsigned j = 0; j < T; j++) {
            unsigned i = r * T + j;
            // Balanced, concentrated, and mixed expert occupancy.
            hi[i] = pattern == 0 ? (r * 7 + j * 31) % E :
                    pattern == 1 ? j :
                    r % 5 == 0 ? j : (r * 13 + j * 29) % E;
            hw[i] = (.1f + float((i * 17) % 101) / 101.f) / 6.f;
            ho[hi[i] + 1]++;
        }
    }
    for (unsigned e = 1; e <= E; e++) ho[e] += ho[e - 1];
    auto cursor = ho;
    for (unsigned i = 0; i < hi.size(); i++) hp[cursor[hi[i]]++] = i;
    HIP(hipMemcpy(x.p, hx.data(), x.n, hipMemcpyHostToDevice));
    HIP(hipMemcpy(ids.p, hi.data(), ids.n, hipMemcpyHostToDevice));
    HIP(hipMemcpy(weights.p, hw.data(), weights.n, hipMemcpyHostToDevice));
    HIP(hipMemcpy(pairs.p, hp.data(), pairs.n, hipMemcpyHostToDevice));
    HIP(hipMemcpy(offsets.p, ho.data(), offsets.n, hipMemcpyHostToDevice));

    CHECK(ds4_mmq_init(0) == 0);
    // Match the engine's native 2048-row MMQ domains, including for 4K input.
    for (unsigned r = 0; r < rows; r += 2048) {
        size_t o = size_t(r) * T * M;
        CHECK(ds4_mmq_iq2_xxs_moe_pair_swiglu(
            gw.p, uw.p, (float *)x.p + r * K, (int32_t *)ids.p + r * T,
            (float *)weights.p + r * T, (float *)gate.p + o,
            (float *)scratch.p + o, (float *)native.p + o,
            (uint16_t *)nativeh.p + o, M, K, 2048, E, T, 32.f, nullptr) == 0);
        HIP(hipDeviceSynchronize());
    }
    for (auto b : buffers) b->check();

    ds4_halo_span live[] = {
        {gw.p, gw.n}, {uw.p, uw.n}, {x.p, x.n}, {pairs.p, pairs.n},
        {offsets.p, offsets.n}, {weights.p, weights.n},
        {fast.p, fast.n}, {fasth.p, fasth.n}
    };
    CHECK(ds4_rocm_v41_gate_up(
        rows, gw.p, uw.p, (float *)x.p, (unsigned *)pairs.p,
        (unsigned *)offsets.p, (float *)weights.p, (float *)fast.p,
        (uint16_t *)fasth.p, 32.f, scratch.p, scratch.n, live, 8) == 1);
    HIP(hipDeviceSynchronize());
    for (auto b : buffers) b->check();

    std::vector<float> a(N), b(N);
    HIP(hipMemcpy(a.data(), native.p, native.n, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(b.data(), fast.p, fast.n, hipMemcpyDeviceToHost));
    size_t different = 0;
    double maxabs = 0;
    for (size_t i = 0; i < N; i++) {
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        different += memcmp(&a[i], &b[i], 4) != 0;
        maxabs = std::max(maxabs, double(std::abs(a[i] - b[i])));
    }
    std::vector<uint16_t> ah(N), bh(N);
    HIP(hipMemcpy(ah.data(), nativeh.p, nativeh.n, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(bh.data(), fasth.p, fasth.n, hipMemcpyDeviceToHost));
    size_t half_different = 0;
    for (size_t i = 0; i < N; i++) half_different += ah[i] != bh[i];
    printf("{\"rows\":%u,\"pattern\":%u,\"elements\":%zu,"
           "\"f32_differences\":%zu,\"f16_differences\":%zu,"
           "\"max_abs\":%.9g,\"canaries\":true}\n",
           rows, pattern, N, different, half_different, maxabs);
    ds4_rocm_v41_gate_up_release();
    CHECK(different == 0 && half_different == 0);
    return 0;
}

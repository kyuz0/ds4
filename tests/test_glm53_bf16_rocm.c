#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "GLM BF16 test failed: %s\n", what);
        exit(1);
    }
}

static uint32_t random_bits(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static float bf16_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void check_shape(uint32_t in_dim, uint32_t out_dim) {
    const uint64_t bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    uint16_t *weights = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANON, -1, 0);
    require_ok(weights != MAP_FAILED, "weight allocation");
    uint32_t state = 19417;
    for (uint64_t i = 0; i < bytes / sizeof(uint16_t); i++) {
        float v = (float)((int)(random_bits(&state) % 2001u) - 1000) / 32768.0f;
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        weights[i] = (uint16_t)(bits >> 16);
    }
    enum { ROWS = 3, GUARD = 32 };
    float *input = malloc((size_t)ROWS * in_dim * sizeof(float));
    float *actual = malloc(((size_t)ROWS * out_dim + 2u * GUARD) * sizeof(float));
    float *scalar = malloc((size_t)out_dim * sizeof(float));
    require_ok(input && actual && scalar, "host allocation");
    for (uint64_t i = 0; i < (uint64_t)ROWS * in_dim; i++) {
        input[i] = (float)((int)(random_bits(&state) % 2001u) - 1000) / 1000.0f;
    }
    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(weights, bytes), "weight registration");
    ds4_gpu_set_glm_model(true);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)ROWS * in_dim * sizeof(float));
    ds4_gpu_tensor *storage = ds4_gpu_tensor_alloc(((uint64_t)ROWS * out_dim + 2u * GUARD) * sizeof(float));
    ds4_gpu_tensor *one = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    require_ok(x && storage && one, "GPU allocation");
    require_ok(ds4_gpu_tensor_write(x, 0, input, (uint64_t)ROWS * in_dim * sizeof(float)), "input upload");
    for (uint32_t rows = 1; rows <= ROWS; rows++) {
        const uint64_t count = (uint64_t)rows * out_dim;
        const uint64_t total = count + 2u * GUARD;
        for (uint64_t i = 0; i < total; i++) actual[i] = 1234567.0f;
        require_ok(ds4_gpu_tensor_write(storage, 0, actual, total * sizeof(float)), "canary upload");
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(storage, GUARD * sizeof(float), count * sizeof(float));
        require_ok(out != NULL, "output view");
        require_ok(ds4_gpu_glm53_matmul_bf16(out, weights, bytes, 0, in_dim, out_dim, x, rows), "BF16 dispatch");
        require_ok(ds4_gpu_tensor_read(storage, 0, actual, total * sizeof(float)), "output synchronization");
        for (uint32_t i = 0; i < GUARD; i++) {
            require_ok(actual[i] == 1234567.0f && actual[GUARD + count + i] == 1234567.0f, "output canaries");
        }
        for (uint32_t row = 0; row < rows; row++) {
            ds4_gpu_tensor *xr = ds4_gpu_tensor_view(x, (uint64_t)row * in_dim * sizeof(float), (uint64_t)in_dim * sizeof(float));
            require_ok(xr != NULL, "input view");
            require_ok(ds4_gpu_glm53_matmul_bf16(one, weights, bytes, 0, in_dim, out_dim, xr, 1), "scalar control");
            require_ok(ds4_gpu_tensor_read(one, 0, scalar, (uint64_t)out_dim * sizeof(float)), "scalar synchronization");
            require_ok(memcmp(scalar, actual + GUARD + (uint64_t)row * out_dim, (size_t)out_dim * sizeof(float)) == 0, "bitwise scalar agreement");
            for (uint32_t col = 0; col < out_dim; col++) {
                double expected = 0;
                for (uint32_t k = 0; k < in_dim; k++) {
                    expected += (double)bf16_float(weights[(uint64_t)col * in_dim + k]) * input[(uint64_t)row * in_dim + k];
                }
                const float value = actual[GUARD + (uint64_t)row * out_dim + col];
                require_ok(isfinite(value) && fabs(value - expected) <= 5e-5 + 5e-5 * fabs(expected), "independent FP64 reference");
            }
            ds4_gpu_tensor_free(xr);
        }
        ds4_gpu_tensor_free(out);
    }
    ds4_gpu_tensor_free(one);
    ds4_gpu_tensor_free(storage);
    ds4_gpu_tensor_free(x);
    ds4_gpu_set_glm_model(false);
    ds4_gpu_cleanup();
    free(scalar);
    free(actual);
    free(input);
    munmap(weights, bytes);
    printf("GLM BF16 %ux%u, rows 1/2/3: PASS\n", in_dim, out_dim);
}

int main(void) {
    check_shape(4096, 4096);
    check_shape(4098, 4097);
    check_shape(8192, 4096);
    check_shape(4096, 8193);
    return 0;
}

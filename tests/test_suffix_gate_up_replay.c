/* Gate/up batch-shape replay: identical captured layer-20 normalized input,
 * identical routing ids and weights, only n_tokens (and therefore the MMQ
 * dispatch shape) varies. Shows whether the mid delta between the suffix arm
 * and the full-body arm is a batch-shape rounding effect or a real input bug. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return -1; \
} } while (0)

enum { U = 6u, H = 2304u, X = 5120u, OWNED = 192u };

static int join(char *dst, size_t cap, const char *dir, const char *rel) {
    const int n = snprintf(dst, cap, "%s/%s", dir, rel);
    CHECK(n > 0 && (size_t)n < cap);
    return 1;
}

static int read_exact(const char *path, void *buf, size_t bytes) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    CHECK(fd >= 0);
    CHECK(fstat(fd, &st) == 0 && (size_t)st.st_size == bytes);
    size_t got = 0;
    while (got < bytes) {
        const ssize_t n = read(fd, (unsigned char *)buf + got, bytes - got);
        if (n < 0 && errno == EINTR) continue;
        CHECK(n > 0);
        got += (size_t)n;
    }
    CHECK(close(fd) == 0);
    return 1;
}

static int write_file(const char *path, const void *buf, size_t bytes) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    CHECK(fd >= 0);
    size_t put = 0;
    while (put < bytes) {
        const ssize_t n = write(fd, (const unsigned char *)buf + put, bytes - put);
        if (n < 0 && errno == EINTR) continue;
        CHECK(n > 0);
        put += (size_t)n;
    }
    CHECK(close(fd) == 0);
    return 1;
}

/* Fused MMQ gate/up + SwiGLU write-back, the path the production launcher
 * selects for layer-20 MoE ("tuned MMQ IQ2 gate/up with fused SwiGLU"). */
extern int ds4_mmq_init(int device);
extern int ds4_mmq_iq2_xxs_moe_pair_swiglu(
        const void *W_gate, const void *W_up, const float *X, const int32_t *ids,
        const float *router_weights, float *gate, float *discard, float *mid_f32,
        void *mid_f16, int M, int K, int n_tokens, int n_experts, int n_expert_used,
        float clamp, void *stream);

static int32_t g_ids[U];
static float g_x[X], g_w[U], g_mid_c[U * H], g_mid_r[U * H];

/* Production epilogue: moe_swiglu_weighted_f32_kernel with clamp 10. */
static double mid_of(double g, double u, double w) {
    const double g2 = g > 10.0 ? 10.0 : g;
    const double u2 = u > 10.0 ? 10.0 : (u < -10.0 ? -10.0 : u);
    return (g2 / (1.0 + exp(-g2))) * u2 * w;
}

typedef struct Run {
    uint32_t n;
    uint32_t owned[U];
    float *mid, *gate, *up;
} Run;

static const void *g_model;
static uint64_t g_model_size, g_gate_off, g_up_off;

static int run_shape(Run *r, uint32_t n, const char *out_dir) {
    char path[512], rel[256];
    const uint64_t pairs = (uint64_t)n * U;
    memset(r, 0, sizeof(*r));
    r->n = n;
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * X * sizeof(float));
    int32_t *ids = malloc((size_t)pairs * sizeof(int32_t));
    float *xb = malloc((size_t)n * X * sizeof(float));
    CHECK(gate && up && sel && x && ids && xb);
    for (uint32_t row = 0; row < n; ++row) {
        const float scale = row == 0u ? 1.0f : 1.0f + 1.0e-3f * (float)row;
        for (uint32_t k = 0; k < X; ++k) xb[(size_t)row * X + k] = g_x[k] * scale;
        for (uint32_t j = 0; j < U; ++j)
            ids[(size_t)row * U + j] = row == 0u ? g_ids[j]
                                                 : (int32_t)((row * U + j) % 384u);
    }
    CHECK(ds4_gpu_tensor_write(sel, 0, ids, pairs * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(x, 0, xb, (uint64_t)n * X * sizeof(float)));
    CHECK(ds4_gpu_dsv41_moe_tp_gate_up(gate, up, g_model, g_model_size,
                                       g_gate_off, g_up_off, sel, x, n, 0));
    CHECK(ds4_gpu_synchronize());
    r->mid = malloc((size_t)U * H * sizeof(float));
    r->gate = malloc((size_t)U * H * sizeof(float));
    r->up = malloc((size_t)U * H * sizeof(float));
    CHECK(r->mid && r->gate && r->up);
    for (uint32_t j = 0; j < U; ++j) {
        CHECK(ds4_gpu_tensor_read(gate, (uint64_t)j * H * sizeof(float),
                                  r->gate + (size_t)j * H, (size_t)H * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(up, (uint64_t)j * H * sizeof(float),
                                  r->up + (size_t)j * H, (size_t)H * sizeof(float)));
        r->owned[j] = g_ids[j] >= 0 && g_ids[j] < OWNED;
        for (uint32_t k = 0; k < H; ++k)
            r->mid[(size_t)j * H + k] = r->owned[j]
                ? (float)mid_of(r->gate[(size_t)j * H + k], r->up[(size_t)j * H + k], g_w[j])
                : 0.0f;
    }
    snprintf(rel, sizeof(rel), "gate_up_n%u.raw", n);
    CHECK(join(path, sizeof(path), out_dir, rel));
    CHECK(write_file(path, r->mid, (size_t)U * H * sizeof(float)));
    free(ids);
    free(xb);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(sel);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    return 1;
}

static void cmp_mid(const char *label, const Run *a, const float *b, const char *bname) {
    for (uint32_t j = 0; j < U; ++j) {
        if (!a->owned[j]) continue;
        uint32_t diffs = 0, big = 0;
        double mx = 0.0, scale = 0.0;
        for (uint32_t k = 0; k < H; ++k) {
            const double x = a->mid[(size_t)j * H + k];
            const double y = b[(size_t)j * H + k];
            if (x != y) {
                ++diffs;
                const double d = fabs(x - y);
                if (d > mx) mx = d;
                if (d > 1e-6) ++big;
            }
            if (fabs(y) > scale) scale = fabs(y);
        }
        printf("MID %s slot=%u expert=%d vs=%s diffs=%u big=%u max_abs=%.6g scale=%.4g\n",
               label, j, g_ids[j], bname, diffs, big, mx, scale);
    }
}

/* Same input/ids/weights, only the batch shape varies, through the fused path.
 * All pointers handed to the MMQ entry must be device addresses. */
static int fused_shapes(const char *out_dir) {
    static const uint32_t shapes[] = {366u, 512u, 1024u, 2048u};
    enum { NS = 4 };
    static float mids[NS][U * H];
    const uint64_t expert_bytes = (uint64_t)H * 1320u;
    /* The MMQ entry wants device weights; stage just the six experts the
     * sampled row selects and remap ids into that table (same bytes as
     * production, which uses a per-rank owned table). */
    ds4_gpu_tensor *gate_w = ds4_gpu_tensor_alloc((uint64_t)U * expert_bytes);
    ds4_gpu_tensor *up_w = ds4_gpu_tensor_alloc((uint64_t)U * expert_bytes);
    CHECK(gate_w && up_w);
    int32_t local_of[384];
    for (int i = 0; i < 384; ++i) local_of[i] = -1;
    for (uint32_t j = 0; j < U; ++j) {
        CHECK(g_ids[j] >= 0 && g_ids[j] < 384);
        local_of[g_ids[j]] = (int32_t)j;
        CHECK(ds4_gpu_tensor_write(gate_w, (uint64_t)j * expert_bytes,
                                   (const char *)g_model + g_gate_off + (uint64_t)g_ids[j] * expert_bytes,
                                   expert_bytes));
        CHECK(ds4_gpu_tensor_write(up_w, (uint64_t)j * expert_bytes,
                                   (const char *)g_model + g_up_off + (uint64_t)g_ids[j] * expert_bytes,
                                   expert_bytes));
    }
    for (int s = 0; s < NS; ++s) {
        const uint32_t n = shapes[s];
        const uint64_t pairs = (uint64_t)n * U;
        float *x = malloc((size_t)n * X * sizeof(float));
        int32_t *ids = malloc((size_t)pairs * sizeof(int32_t));
        float *w = malloc((size_t)n * U * sizeof(float));
        CHECK(x && ids && w);
        for (uint32_t row = 0; row < n; ++row) {
            const float sc = row == 0u ? 1.0f : 1.0f + 1.0e-3f * (float)row;
            for (uint32_t k = 0; k < X; ++k) x[(size_t)row * X + k] = g_x[k] * sc;
            for (uint32_t j = 0; j < U; ++j) {
                ids[(size_t)row * U + j] = row == 0u ? (int32_t)j
                                                     : (int32_t)((row * U + j) % U);
                w[(size_t)row * U + j] = row == 0u ? g_w[j] : 0.01f;
            }
        }
        ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)n * X * sizeof(float));
        ds4_gpu_tensor *idt = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
        ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc((uint64_t)n * U * sizeof(float));
        ds4_gpu_tensor *gatet = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
        ds4_gpu_tensor *midt = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
        ds4_gpu_tensor *midh = ds4_gpu_tensor_alloc(pairs * H * sizeof(uint16_t));
        CHECK(xt && idt && wt && gatet && midt && midh);
        CHECK(ds4_gpu_tensor_write(xt, 0, x, (uint64_t)n * X * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(idt, 0, ids, pairs * sizeof(int32_t)));
        CHECK(ds4_gpu_tensor_write(wt, 0, w, (uint64_t)n * U * sizeof(float)));
        CHECK(ds4_mmq_init(0) == 0);
        const int rc = ds4_mmq_iq2_xxs_moe_pair_swiglu(
            ds4_gpu_tensor_contents(gate_w), ds4_gpu_tensor_contents(up_w),
            (const float *)ds4_gpu_tensor_contents(xt),
            (const int32_t *)ds4_gpu_tensor_contents(idt),
            (const float *)ds4_gpu_tensor_contents(wt),
            (float *)ds4_gpu_tensor_contents(gatet), NULL,
            (float *)ds4_gpu_tensor_contents(midt),
            ds4_gpu_tensor_contents(midh), (int)H, (int)X, (int)n, (int)U, (int)U,
            10.0f, NULL);
        CHECK(rc == 0);
        CHECK(ds4_gpu_synchronize());
        float *row0 = malloc((size_t)U * H * sizeof(float));
        CHECK(row0 && ds4_gpu_tensor_read(midt, 0, row0, (size_t)U * H * sizeof(float)));
        memcpy(mids[s], row0, sizeof(mids[s]));
        free(row0);
        char rel[64], path[512];
        snprintf(rel, sizeof(rel), "fused_mid_n%u.raw", n);
        CHECK(join(path, sizeof(path), out_dir, rel));
        CHECK(write_file(path, mids[s], sizeof(mids[s])));
        ds4_gpu_tensor_free(midh); ds4_gpu_tensor_free(midt);
        ds4_gpu_tensor_free(gatet); ds4_gpu_tensor_free(wt);
        ds4_gpu_tensor_free(idt); ds4_gpu_tensor_free(xt);
        free(w); free(ids); free(x);
    }
    ds4_gpu_tensor_free(up_w);
    ds4_gpu_tensor_free(gate_w);
    for (int s = 0; s < NS; ++s)
        for (uint32_t j = 0; j < U; ++j) {
            if (g_ids[j] < 0 || g_ids[j] >= OWNED) continue;
            uint32_t diffs = 0; double mx = 0.0, scale = 0.0;
            for (uint32_t k = 0; k < H; ++k) {
                const double a = mids[s][(size_t)j * H + k];
                const double b = mids[NS - 1][(size_t)j * H + k];
                if (fabs(b) > scale) scale = fabs(b);
                if (a != b) { ++diffs; if (fabs(a - b) > mx) mx = fabs(a - b); }
            }
            printf("FUSED n%u_vs_n2048 slot=%u expert=%d diffs=%u max_abs=%.6g scale=%.4g\n",
                   shapes[s], j, g_ids[j], diffs, mx, scale);
        }
    for (int s = 0; s < NS; ++s)
        for (uint32_t j = 0; j < U; ++j) {
            if (g_ids[j] < 0 || g_ids[j] >= OWNED) continue;
            uint32_t dc = 0, dr = 0; double mc = 0.0, mr = 0.0;
            for (uint32_t k = 0; k < H; ++k) {
                const double v = mids[s][(size_t)j * H + k];
                if (v != g_mid_c[(size_t)j * H + k]) {
                    ++dc;
                    if (fabs(v - g_mid_c[(size_t)j * H + k]) > mc) mc = fabs(v - g_mid_c[(size_t)j * H + k]);
                }
                if (v != g_mid_r[(size_t)j * H + k]) {
                    ++dr;
                    if (fabs(v - g_mid_r[(size_t)j * H + k]) > mr) mr = fabs(v - g_mid_r[(size_t)j * H + k]);
                }
            }
            printf("FUSED_CAPTURE n%u slot=%u expert=%d vs_cand diffs=%u max=%.6g vs_ref diffs=%u max=%.6g\n",
                   shapes[s], j, g_ids[j], dc, mc, dr, mr);
        }
    return 1;
}

/* ---- production-faithful frame replay: 384-expert table, real counts ---- */
static uint32_t g_counts_c[384], g_counts_r[384];

static int read_counts(const char *path, uint32_t *c) {
    return read_exact(path, c, 384 * sizeof(uint32_t));
}

static int build_ids_for(uint32_t n, const uint32_t *counts, uint32_t target_row, int32_t *ids) {
    uint32_t rem[384];
    memcpy(rem, counts, sizeof(rem));
    uint64_t sum = 0;
    for (int e = 0; e < 384; ++e) sum += rem[e];
    CHECK(sum == (uint64_t)n * U && target_row < n);
    for (uint32_t j = 0; j < U; ++j) {
        CHECK(rem[g_ids[j]] > 0);
        rem[g_ids[j]]--;
        ids[(size_t)target_row * U + j] = g_ids[j];
    }
    for (uint32_t row = 0; row < n; ++row) {
        if (row == target_row) continue;
        int32_t picked[U];
        for (uint32_t j = 0; j < U; ++j) {
            int best = -1;
            uint32_t best_count = 0;
            for (int e = 0; e < 384; ++e) {
                int used = 0;
                for (uint32_t k = 0; k < j; ++k) used |= picked[k] == e;
                if (!used && rem[e] > best_count) { best = e; best_count = rem[e]; }
            }
            CHECK(best >= 0 && best_count > 0);
            picked[j] = best;
            rem[best]--;
            ids[(size_t)row * U + j] = best;
        }
    }
    for (int e = 0; e < 384; ++e) CHECK(rem[e] == 0);
    return 1;
}

typedef struct Frame {
    uint32_t n, row;
    float gate[U * H], mid[U * H];
    uint32_t rank[U];
} Frame;

static int replay_frame(ds4_gpu_tensor *gate_w, ds4_gpu_tensor *up_w,
                        uint32_t n, const uint32_t *counts, uint32_t target_row,
                        Frame *fr) {
    const uint64_t pairs = (uint64_t)n * U;
    memset(fr, 0, sizeof(*fr));
    fr->n = n;
    fr->row = target_row;
    float *x = malloc((size_t)n * X * sizeof(float));
    int32_t *ids = malloc((size_t)pairs * sizeof(int32_t));
    float *w = malloc((size_t)n * U * sizeof(float));
    CHECK(x && ids && w);
    for (uint32_t row = 0; row < n; ++row) {
        const float sc = row == target_row ? 1.0f
                                           : 1.0f + 1.0e-3f * (float)(row % 97u);
        for (uint32_t k = 0; k < X; ++k) x[(size_t)row * X + k] = g_x[k] * sc;
        for (uint32_t j = 0; j < U; ++j) w[(size_t)row * U + j] = 0.5f / U;
    }
    for (uint32_t j = 0; j < U; ++j) w[(size_t)target_row * U + j] = g_w[j];
    CHECK(build_ids_for(n, counts, target_row, ids));
    for (uint32_t j = 0; j < U; ++j) {
        uint32_t r = 0;
        for (uint32_t row = 0; row < target_row; ++row)
            for (uint32_t k = 0; k < U; ++k) r += ids[(size_t)row * U + k] == g_ids[j];
        fr->rank[j] = r;
    }
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc((uint64_t)n * X * sizeof(float));
    ds4_gpu_tensor *idt = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc((uint64_t)n * U * sizeof(float));
    ds4_gpu_tensor *gatet = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
    ds4_gpu_tensor *midt = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
    ds4_gpu_tensor *midh = ds4_gpu_tensor_alloc(pairs * H * sizeof(uint16_t));
    CHECK(xt && idt && wt && gatet && midt && midh);
    CHECK(ds4_gpu_tensor_write(xt, 0, x, (uint64_t)n * X * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(idt, 0, ids, pairs * sizeof(int32_t)));
    CHECK(ds4_gpu_tensor_write(wt, 0, w, (uint64_t)n * U * sizeof(float)));
    CHECK(ds4_mmq_init(0) == 0);
    CHECK(ds4_mmq_iq2_xxs_moe_pair_swiglu(
              ds4_gpu_tensor_contents(gate_w), ds4_gpu_tensor_contents(up_w),
              (const float *)ds4_gpu_tensor_contents(xt),
              (const int32_t *)ds4_gpu_tensor_contents(idt),
              (const float *)ds4_gpu_tensor_contents(wt),
              (float *)ds4_gpu_tensor_contents(gatet), NULL,
              (float *)ds4_gpu_tensor_contents(midt),
              ds4_gpu_tensor_contents(midh), (int)H, (int)X, (int)n, 384, (int)U,
              10.0f, NULL) == 0);
    CHECK(ds4_gpu_synchronize());
    const uint64_t off = (uint64_t)target_row * U * H * sizeof(float);
    CHECK(ds4_gpu_tensor_read(gatet, off, fr->gate, sizeof(fr->gate)));
    CHECK(ds4_gpu_tensor_read(midt, off, fr->mid, sizeof(fr->mid)));
    ds4_gpu_tensor_free(midh); ds4_gpu_tensor_free(midt);
    ds4_gpu_tensor_free(gatet); ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(idt); ds4_gpu_tensor_free(xt);
    free(w); free(ids); free(x);
    return 1;
}

static void frame_cmp(const char *what, const char *label, const Frame *a, const Frame *b) {
    for (uint32_t j = 0; j < U; ++j) {
        if (g_ids[j] < 0 || g_ids[j] >= OWNED) continue;
        uint32_t diffs = 0; double mx = 0.0, scale = 0.0;
        const float *pa = what[0] == 'g' ? a->gate : a->mid;
        const float *pb = what[0] == 'g' ? b->gate : b->mid;
        for (uint32_t k = 0; k < H; ++k) {
            const double y = pb[(size_t)j * H + k];
            if (fabs(y) > scale) scale = fabs(y);
            if (pa[(size_t)j * H + k] != y) {
                ++diffs;
                if (fabs((double)pa[(size_t)j * H + k] - y) > mx) mx = fabs((double)pa[(size_t)j * H + k] - y);
            }
        }
        printf("%s %s slot=%u expert=%d diffs=%u max_abs=%.6g scale=%.4g rankA=%u rankB=%u\n",
               what, label, j, g_ids[j], diffs, mx, scale, a->rank[j], b->rank[j]);
    }
}

static int production_frames(const char *run_dir, const char *out_dir) {
    const uint64_t expert_bytes = (uint64_t)H * 1320u;
    char path[512];
    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/moe_counts_u32.raw") &&
          read_counts(path, g_counts_c));
    CHECK(join(path, sizeof(path), run_dir, "reference/diag/moe_counts_u32.raw") &&
          read_counts(path, g_counts_r));
    ds4_gpu_tensor *gate_w = ds4_gpu_tensor_alloc((uint64_t)384 * expert_bytes);
    ds4_gpu_tensor *up_w = ds4_gpu_tensor_alloc((uint64_t)384 * expert_bytes);
    CHECK(gate_w && up_w);
    for (int e = 0; e < 384; ++e) {
        CHECK(ds4_gpu_tensor_write(gate_w, (uint64_t)e * expert_bytes,
                                   (const char *)g_model + g_gate_off + (uint64_t)e * expert_bytes,
                                   expert_bytes));
        CHECK(ds4_gpu_tensor_write(up_w, (uint64_t)e * expert_bytes,
                                   (const char *)g_model + g_up_off + (uint64_t)e * expert_bytes,
                                   expert_bytes));
    }
    static Frame A, B, C, D;
    CHECK(replay_frame(gate_w, up_w, 366u, g_counts_c, 239u, &A));    /* production candidate */
    CHECK(replay_frame(gate_w, up_w, 2048u, g_counts_r, 1921u, &B));   /* production reference */
    CHECK(replay_frame(gate_w, up_w, 366u, g_counts_c, 239u, &C));     /* repeat after other shape */
    CHECK(replay_frame(gate_w, up_w, 2048u, g_counts_r, 1921u, &D));   /* repeat */
    printf("FRAMES: cand n=366 row=239, ref n=2048 row=1921, then both repeated\n");
    frame_cmp("gate", "A_cand_vs_B_ref", &A, &B);
    frame_cmp("mid", "A_cand_vs_B_ref", &A, &B);
    frame_cmp("gate", "A_cand_vs_C_cand_repeat", &A, &C);
    frame_cmp("mid", "A_cand_vs_C_cand_repeat", &A, &C);
    frame_cmp("gate", "B_ref_vs_D_ref_repeat", &B, &D);
    frame_cmp("mid", "B_ref_vs_D_ref_repeat", &B, &D);
    char rel[64];
    snprintf(rel, sizeof(rel), "frame_gate_cand.raw");
    CHECK(join(path, sizeof(path), out_dir, rel));
    CHECK(write_file(path, A.gate, sizeof(A.gate)));
    snprintf(rel, sizeof(rel), "frame_gate_ref.raw");
    CHECK(join(path, sizeof(path), out_dir, rel));
    CHECK(write_file(path, B.gate, sizeof(B.gate)));
    ds4_gpu_tensor_free(up_w);
    ds4_gpu_tensor_free(gate_w);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL RUN_DIR OUT_DIR GATE_OFFSET UP_OFFSET\n", argv[0]);
        return 2;
    }
    const char *run_dir = argv[2], *out_dir = argv[3];
    g_gate_off = strtoull(argv[4], NULL, 10);
    g_up_off = strtoull(argv[5], NULL, 10);
    char path[512];
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    CHECK(fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0);
    g_model_size = (uint64_t)st.st_size;
    g_model = mmap(NULL, (size_t)g_model_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g_model == MAP_FAILED) fprintf(stderr, "mmap: %s\n", strerror(errno));
    CHECK(g_model != MAP_FAILED);
    CHECK(close(fd) == 0);

    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/shared_routed_ffn_20_selected_8065_8066.raw") &&
          read_exact(path, g_ids, sizeof(g_ids)));
    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/shared_routed_ffn_20_route_weights_8065_8066.raw") &&
          read_exact(path, g_w, sizeof(g_w)));
    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/hc_ffn_norm_20_norm_8065_8066.raw") &&
          read_exact(path, g_x, sizeof(g_x)));
    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/moe_mid_f32.raw") &&
          read_exact(path, g_mid_c, sizeof(g_mid_c)));
    CHECK(join(path, sizeof(path), run_dir, "reference/diag/moe_mid_f32.raw") &&
          read_exact(path, g_mid_r, sizeof(g_mid_r)));

    CHECK(production_frames(run_dir, out_dir));
    return 0;
    CHECK(ds4_gpu_init());
    ds4_gpu_set_deepseek41_model(true);
    CHECK(ds4_gpu_set_model_map(g_model, g_model_size));

    static const uint32_t shapes[] = {128u, 200u, 366u, 512u, 1024u, 2048u};
    enum { NS = (int)(sizeof(shapes) / sizeof(shapes[0])) };
    Run runs[NS];
    for (int i = 0; i < NS; ++i) CHECK(run_shape(&runs[i], shapes[i], out_dir));
    const Run *big = &runs[NS - 1];
    for (int i = 0; i < NS; ++i) {
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "n%u", shapes[i]);
        printf("SHAPE %s owned_slots=%u%u%u%u%u%u\n", lbl,
               runs[i].owned[0], runs[i].owned[1], runs[i].owned[2],
               runs[i].owned[3], runs[i].owned[4], runs[i].owned[5]);
        cmp_mid(lbl, &runs[i], big->mid, "n2048");
    }
    for (int i = 0; i < NS; ++i) {
        char lbl[32];
        snprintf(lbl, sizeof(lbl), "n%u", shapes[i]);
        cmp_mid(lbl, &runs[i], g_mid_c, "capture_cand");
        cmp_mid(lbl, &runs[i], g_mid_r, "capture_ref");
    }
    /* gate/up themselves, not just the weighted mid */
    for (uint32_t j = 0; j < U; ++j) {
        if (!big->owned[j]) continue;
        uint32_t dg = 0, du = 0;
        double mg = 0.0, mu = 0.0;
        for (uint32_t k = 0; k < H; ++k) {
            const double a = runs[2].gate[(size_t)j * H + k];
            const double b = big->gate[(size_t)j * H + k];
            if (a != b) { ++dg; if (fabs(a - b) > mg) mg = fabs(a - b); }
            const double c = runs[2].up[(size_t)j * H + k];
            const double d = big->up[(size_t)j * H + k];
            if (c != d) { ++du; if (fabs(c - d) > mu) mu = fabs(c - d); }
        }
        printf("PROJ slot=%u expert=%d n366_vs_n2048 gate_diffs=%u gate_max=%.6g up_diffs=%u up_max=%.6g\n",
               j, g_ids[j], dg, mg, du, mu);
    }
    puts("GATE_UP REPLAY COMPLETE rc=0");
    for (int i = 0; i < NS; ++i) {
        free(runs[i].mid);
        free(runs[i].gate);
        free(runs[i].up);
    }
    ds4_gpu_cleanup();
    CHECK(munmap((void *)g_model, (size_t)g_model_size) == 0);
    return 0;
}

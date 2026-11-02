/* Focused V4.1 suffix down replay: identical captured mid activations and
 * identical layer-20 owned Q2 expert weights through the actual down kernels.
 * First repeat identical shape/input; then vary only placement/path. */
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

enum { E = 384u, U = 6u, H = 2304u, M = 5120u, OWNED = 192u };
enum { HOT = 8u, TILE = 64u, MTILES = 4u, BM = 16u, BN = 16u };
static int32_t target_ids[U];

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

static float f16_to_f32(uint16_t bits) {
    _Float16 h;
    memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

typedef struct Arm {
    uint32_t n, row;
    uint32_t counts[E], offsets[E];
    uint32_t pair[U], pair_owned[U];
    uint32_t hot_count, hot_max, tile_count;
    uint32_t tile_offsets[E + 1];
    uint32_t owned_hot_count, owned_hot_max, owned_tile_count;
    uint32_t owned_offsets[OWNED + 1], owned_tile_offsets[OWNED + 1];
    uint32_t *ids;
    uint16_t *mid, *down_cap;
    int const_mid, fill_all;
} Arm;

static int build_ids(Arm *a) {
    uint32_t rem[E];
    uint8_t used[E];
    memcpy(rem, a->counts, sizeof(rem));
    memset(a->ids, 0, (size_t)a->n * U * sizeof(uint32_t));
    for (uint32_t j = 0; j < U; ++j) {
        const int32_t e = target_ids[j];
        CHECK(e >= 0 && (uint32_t)e < E && rem[e] > 0);
        for (uint32_t k = 0; k < j; ++k) CHECK(target_ids[k] != e);
        rem[e]--;
        a->ids[(size_t)a->row * U + j] = (uint32_t)e;
    }
    for (uint32_t row = 0; row < a->n; ++row) {
        if (row == a->row) continue;
        memset(used, 0, sizeof(used));
        for (uint32_t j = 0; j < U; ++j) {
            uint32_t best = E, best_count = 0;
            for (uint32_t e = 0; e < E; ++e) {
                if (!used[e] && rem[e] > best_count) {
                    best = e;
                    best_count = rem[e];
                }
            }
            CHECK(best < E && best_count > 0);
            used[best] = 1;
            rem[best]--;
            a->ids[(size_t)row * U + j] = best;
        }
    }
    for (uint32_t e = 0; e < E; ++e) CHECK(rem[e] == 0);
    return 1;
}

static int derive_arm(Arm *a);

static int load_arm(Arm *a, const char *dir, uint32_t n, uint32_t row) {
    char path[512];
    memset(a, 0, sizeof(*a));
    a->n = n;
    a->row = row;
    CHECK(n >= 2 && n <= 2048 && row < n);
    CHECK(join(path, sizeof(path), dir, "diag/moe_counts_u32.raw") &&
          read_exact(path, a->counts, sizeof(a->counts)));
    CHECK(join(path, sizeof(path), dir, "diag/moe_mid_f16.raw"));
    a->mid = malloc((size_t)U * H * sizeof(uint16_t));
    CHECK(a->mid && read_exact(path, a->mid, (size_t)U * H * sizeof(uint16_t)));
    CHECK(join(path, sizeof(path), dir, "diag/moe_down.raw"));
    a->down_cap = malloc((size_t)U * M * sizeof(uint16_t));
    CHECK(a->down_cap && read_exact(path, a->down_cap, (size_t)U * M * sizeof(uint16_t)));
    a->ids = calloc((size_t)n * U, sizeof(uint32_t));
    CHECK(a->ids);
    CHECK(build_ids(a));
    return derive_arm(a);
}

/* Same captured mid/down, but a synthetic id map of `nrows` rows, every row
 * carrying the target experts. The target pair's rank inside its expert bucket
 * is therefore exactly `target_row`, which isolates bucket position. */
static int load_synth(Arm *a, const char *dir, uint32_t nrows, uint32_t target_row,
                      int const_mid, int fill_all) {
    char path[512];
    memset(a, 0, sizeof(*a));
    a->n = nrows;
    a->row = target_row;
    a->const_mid = const_mid;
    a->fill_all = fill_all;
    CHECK(nrows >= 2 && nrows <= 2048 && target_row < nrows);
    CHECK(join(path, sizeof(path), dir, "diag/moe_mid_f16.raw"));
    a->mid = malloc((size_t)U * H * sizeof(uint16_t));
    CHECK(a->mid && read_exact(path, a->mid, (size_t)U * H * sizeof(uint16_t)));
    CHECK(join(path, sizeof(path), dir, "diag/moe_down.raw"));
    a->down_cap = malloc((size_t)U * M * sizeof(uint16_t));
    CHECK(a->down_cap && read_exact(path, a->down_cap, (size_t)U * M * sizeof(uint16_t)));
    a->ids = calloc((size_t)nrows * U, sizeof(uint32_t));
    CHECK(a->ids);
    for (uint32_t r = 0; r < nrows; ++r)
        for (uint32_t j = 0; j < U; ++j)
            a->ids[(size_t)r * U + j] = (uint32_t)target_ids[j];
    memset(a->counts, 0, sizeof(a->counts));
    for (size_t i = 0; i < (size_t)nrows * U; ++i) ++a->counts[a->ids[i]];
    return derive_arm(a);
}

static int derive_arm(Arm *a) {
    const uint32_t row = a->row;
    uint64_t sum = 0;
    for (uint32_t e = 0; e < E; ++e) sum += a->counts[e];
    CHECK(sum == (uint64_t)a->n * U);

    uint32_t off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        a->offsets[e] = off;
        off += a->counts[e];
    }
    CHECK(off == (uint32_t)sum);
    for (uint32_t j = 0; j < U; ++j) {
        const uint32_t e = (uint32_t)target_ids[j];
        uint32_t rank = 0;
        for (uint32_t r = 0; r < row; ++r)
            for (uint32_t k = 0; k < U; ++k) rank += a->ids[(size_t)r * U + k] == e;
        a->pair[j] = a->offsets[e] + rank;
    }

    for (uint32_t e = 0; e < E; ++e) {
        if (a->counts[e] >= HOT) {
            ++a->hot_count;
            if (a->counts[e] > a->hot_max) a->hot_max = a->counts[e];
        }
    }
    off = 0;
    for (uint32_t e = 0; e < E; ++e) {
        a->tile_offsets[e] = off;
        if (a->counts[e] >= HOT) off += (a->counts[e] + TILE - 1u) / TILE;
    }
    a->tile_offsets[E] = off;
    a->tile_count = off;

    off = 0;
    for (uint32_t e = 0; e < OWNED; ++e) {
        a->owned_offsets[e] = off;
        off += a->counts[e];
        if (a->counts[e] >= HOT) {
            ++a->owned_hot_count;
            if (a->counts[e] > a->owned_hot_max) a->owned_hot_max = a->counts[e];
        }
    }
    a->owned_offsets[OWNED] = off;
    off = 0;
    for (uint32_t e = 0; e < OWNED; ++e) {
        a->owned_tile_offsets[e] = off;
        if (a->counts[e] >= HOT) off += (a->counts[e] + TILE - 1u) / TILE;
    }
    a->owned_tile_offsets[OWNED] = off;
    a->owned_tile_count = off;

    for (uint32_t j = 0; j < U; ++j) {
        const uint32_t e = (uint32_t)target_ids[j];
        a->pair_owned[j] = UINT32_MAX;
        if (e >= OWNED) continue;
        uint32_t rank = 0;
        for (uint32_t r = 0; r < row; ++r)
            for (uint32_t k = 0; k < U; ++k) rank += a->ids[(size_t)r * U + k] == e;
        a->pair_owned[j] = a->owned_offsets[e] + rank;
    }
    return 1;
}

typedef struct CaseResult {
    const char *name;
    const Arm *arm;
    int force_compact;
    uint16_t *out;
} CaseResult;

static double max_abs_value(const uint16_t *a);

static const void *g_model;
static uint64_t g_model_size, g_down_offset;

static int run_case(const Arm *a, const char *name, int force_compact,
                    const char *out_dir, int quiet, CaseResult *res) {
    char path[512], rel[256];
    const uint64_t pairs = (uint64_t)a->n * U;
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(pairs * H * sizeof(float));
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(pairs * sizeof(int32_t));
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc(pairs * M * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)a->n * M * sizeof(float));
    float *mid_buf = calloc((size_t)pairs * H, sizeof(float));
    int32_t *ids_buf = malloc((size_t)pairs * sizeof(int32_t));
    CHECK(mid && selected && scratch && out && mid_buf && ids_buf);
    memcpy(ids_buf, a->ids, (size_t)pairs * sizeof(int32_t));
    for (uint32_t row = 0; row < a->n; ++row) {
        if (row != a->row && !a->fill_all) continue;
        for (uint32_t j = 0; j < U; ++j)
            for (uint32_t k = 0; k < H; ++k)
                mid_buf[((size_t)row * U + j) * H + k] = a->const_mid == 2
                    ? (k == 5u ? 1.0f : 0.0f)
                    : (a->const_mid ? 1.0f : f16_to_f32(a->mid[(size_t)j * H + k]));
    }
    CHECK(ds4_gpu_tensor_write(mid, 0, mid_buf, pairs * H * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(selected, 0, ids_buf, pairs * sizeof(int32_t)));
    CHECK(unsetenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_COMPACT") == 0);
    CHECK(unsetenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_HOTLIST") == 0);
    CHECK(setenv(force_compact ? "DS4_ROCM_V41_DIAG_DOWN_FORCE_COMPACT"
                              : "DS4_ROCM_V41_DIAG_DOWN_FORCE_HOTLIST", "1", 1) == 0);
    CHECK(ds4_gpu_dsv41_moe_tp_down(out, scratch, mid, selected,
                                    g_model, g_model_size, g_down_offset, a->n, 0));
    CHECK(ds4_gpu_synchronize());
    res->out = malloc((size_t)U * M * sizeof(uint16_t));
    CHECK(res->out);
    for (uint32_t j = 0; j < U; ++j) {
        CHECK(ds4_gpu_tensor_read(scratch, (uint64_t)(a->row * U + j) * M * sizeof(uint16_t),
                                  res->out + (size_t)j * M, (size_t)M * sizeof(uint16_t)));
    }
    res->name = name;
    res->arm = a;
    res->force_compact = force_compact;

    /* With a constant mid in every row, all pairs of one expert must produce
     * the same row, so rank differences show up inside a single launch. */
    if (a->fill_all) {
        uint16_t *alt = malloc((size_t)M * sizeof(uint16_t));
        CHECK(alt);
        for (uint32_t j = 0; j < U; ++j) {
            CHECK(ds4_gpu_tensor_read(scratch, (uint64_t)j * M * sizeof(uint16_t),
                                      alt, (size_t)M * sizeof(uint16_t)));
            uint32_t diffs = 0;
            double max_abs = 0.0;
            for (uint32_t r = 0; r < M; ++r) {
                if (alt[r] == res->out[(size_t)j * M + r]) continue;
                ++diffs;
                const double d = fabs((double)f16_to_f32(alt[r]) -
                                      (double)f16_to_f32(res->out[(size_t)j * M + r]));
                if (d > max_abs) max_abs = d;
            }
            printf("INTRA %s slot=%u expert=%d pair0_vs_pair%u diffs=%u max_abs=%.9g\n",
                   name, j, target_ids[j], a->row, diffs, max_abs);
        }
        free(alt);
    }

    snprintf(rel, sizeof(rel), "case_%s_target_f16.raw", name);
    CHECK(join(path, sizeof(path), out_dir, rel));
    CHECK(write_file(path, res->out, (size_t)U * M * sizeof(uint16_t)));

    uint64_t counts_sum = 0, owned_sum = 0;
    for (uint32_t e = 0; e < E; ++e) counts_sum += a->counts[e];
    for (uint32_t e = 0; e < OWNED; ++e) owned_sum += a->counts[e];
    printf("CASE %s path=%s n=%u row=%u counts_sum=%llu owned_sum=%llu "
           "hot=%u max=%u tiles=%u owned_hot=%u owned_max=%u owned_tiles=%u\n",
           name, force_compact ? "compact" : "hotlist", a->n, a->row,
           (unsigned long long)counts_sum, (unsigned long long)owned_sum,
           a->hot_count, a->hot_max, a->tile_count,
           a->owned_hot_count, a->owned_hot_max, a->owned_tile_count);
    printf("  grid compact x=160 y=1 z=%u block=128 shmem=9856\n", a->owned_tile_count);
    printf("  grid hotlist x=160 y=%u z=%u block=128 shmem=9856\n",
           (a->owned_hot_max + TILE - 1u) / TILE, a->owned_hot_count);
    for (uint32_t j = 0; j < U; ++j)
        if (!quiet)
            printf("  mag slot=%u expert=%d replayed_maxabs=%.9g captured_maxabs=%.9g\n",
                   j, target_ids[j], max_abs_value(res->out + (size_t)j * M),
                   max_abs_value(a->down_cap + (size_t)j * M));
    for (uint32_t j = 0; j < U; ++j) {
        if (quiet) break;
        const uint32_t e = (uint32_t)target_ids[j];
        const uint32_t pr = a->pair_owned[j];
        uint32_t rank_local = UINT32_MAX, tile_local = UINT32_MAX, tile_global = UINT32_MAX;
        uint32_t mt = UINT32_MAX, mm = UINT32_MAX;
        if (pr != UINT32_MAX && a->counts[e] >= HOT) {
            rank_local = pr - a->owned_offsets[e];
            tile_local = rank_local / TILE;
            tile_global = a->owned_tile_offsets[e] + tile_local;
            mt = (rank_local % TILE) / BM;
            mm = (rank_local % TILE) % BM;
        }
        printf("  target slot=%u expert=%d owned=%d pair_global=%u pair_owned=%u "
               "rank_local=%u counts=%u tile_local=%u tile_global=%u mt=%u mm=%u\n",
               j, target_ids[j], e < OWNED, a->pair[j], pr, rank_local,
               a->counts[e], tile_local, tile_global, mt, mm);
    }
    fflush(stdout);

    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(mid);
    free(ids_buf);
    free(mid_buf);
    CHECK(unsetenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_COMPACT") == 0);
    CHECK(unsetenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_HOTLIST") == 0);
    return 1;
}

static uint32_t g_bad_outputs;

static void compare_outputs(const char *label, const uint16_t *a, const uint16_t *b) {
    for (uint32_t j = 1; j <= 2; ++j) {
        const int exact = memcmp(a + (size_t)j * M, b + (size_t)j * M,
                                 (size_t)M * sizeof(uint16_t)) == 0;
        double max_abs = 0.0;
        uint32_t diffs = 0, nonfinite = 0;
        for (uint32_t r = 0; r < M; ++r) {
            const float x = f16_to_f32(a[(size_t)j * M + r]);
            const float y = f16_to_f32(b[(size_t)j * M + r]);
            if (!isfinite(x) || !isfinite(y)) {
                ++nonfinite;
                continue;
            }
            if (x != y) ++diffs;
            const double d = fabs((double)x - (double)y);
            if (d > max_abs) max_abs = d;
        }
        if (nonfinite) ++g_bad_outputs;
        printf("CMP %s slot=%u expert=%d exact=%d diffs=%u nonfinite=%u max_abs=%.9g\n",
               label, j, target_ids[j], exact, diffs, nonfinite, max_abs);
    }
}

static double max_abs_value(const uint16_t *a) {
    double m = 0.0;
    for (uint32_t r = 0; r < M; ++r) {
        const double v = fabs((double)f16_to_f32(a[r]));
        if (v > m) m = v;
    }
    return m;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        fprintf(stderr, "usage: %s MODEL RUN_DIR N_CAND ROW_CAND N_REF ROW_REF DOWN_OFFSET OUT_DIR\n",
                argv[0]);
        return 2;
    }
    const uint32_t n_c = (uint32_t)strtoul(argv[3], NULL, 10);
    const uint32_t row_c = (uint32_t)strtoul(argv[4], NULL, 10);
    const uint32_t n_r = (uint32_t)strtoul(argv[5], NULL, 10);
    const uint32_t row_r = (uint32_t)strtoul(argv[6], NULL, 10);
    g_down_offset = strtoull(argv[7], NULL, 10);
    const char *run_dir = argv[2], *out_dir = argv[8];
    char path[512], cand_dir[512], ref_dir[512];

    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    CHECK(fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0);
    g_model_size = (uint64_t)st.st_size;
    /* Down replay only reads weights. A writable private map of 365 GB
     * exceeds the host commit heuristic; read-only maps are accepted. */
    g_model = mmap(NULL, (size_t)g_model_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (g_model == MAP_FAILED) fprintf(stderr, "mmap read-only model: %s\n", strerror(errno));
    CHECK(g_model != MAP_FAILED);
    CHECK(close(fd) == 0);

    CHECK(join(path, sizeof(path), run_dir, "candidate/diag/shared_routed_ffn_20_selected_8065_8066.raw"));
    int32_t cand_sel[U], ref_sel[U];
    CHECK(read_exact(path, cand_sel, sizeof(cand_sel)));
    CHECK(join(path, sizeof(path), run_dir, "reference/diag/shared_routed_ffn_20_selected_8065_8066.raw"));
    CHECK(read_exact(path, ref_sel, sizeof(ref_sel)));
    CHECK(memcmp(cand_sel, ref_sel, sizeof(cand_sel)) == 0);
    memcpy(target_ids, cand_sel, sizeof(target_ids));

    CHECK(ds4_gpu_init());
    ds4_gpu_set_deepseek41_model(true);
    CHECK(ds4_gpu_set_model_map(g_model, g_model_size));
    CHECK(join(cand_dir, sizeof(cand_dir), run_dir, "candidate"));
    CHECK(join(ref_dir, sizeof(ref_dir), run_dir, "reference"));
    Arm cand, ref;
    CHECK(load_arm(&cand, cand_dir, n_c, row_c));
    CHECK(load_arm(&ref, ref_dir, n_r, row_r));
    enum { NSYN = 77 };
    Arm syn[NSYN];
    uint32_t spec_nrows[NSYN], spec_row[NSYN];
    int spec_const[NSYN];
    for (int i = 0; i < 64; ++i) {          /* rank sweep inside one 64-row tile */
        spec_nrows[i] = 64u;
        spec_row[i] = (uint32_t)i;
        spec_const[i] = 0;
    }
    spec_nrows[64] = 64u; spec_row[64] = 63u; spec_const[64] = 0;  /* repeat */
    spec_nrows[65] = 192u; spec_row[65] = 127u; spec_const[65] = 0;
    /* Constant mid: same shape, but any 1-ULP input difference would vanish. */
    for (int i = 0; i < 8; ++i) {
        static const uint32_t ranks[8] = {0u, 1u, 2u, 3u, 40u, 41u, 62u, 63u};
        spec_nrows[66 + i] = 64u;
        spec_row[66 + i] = ranks[i];
        spec_const[66 + i] = 1;
    }
    /* Constant mid in every row: same expert, all 64 pairs identical. */
    spec_nrows[74] = 64u; spec_row[74] = 63u; spec_const[74] = 1;
    spec_nrows[75] = 64u; spec_row[75] = 63u; spec_const[75] = 1;
    spec_nrows[76] = 64u; spec_row[76] = 63u; spec_const[76] = 2;  /* single k */
    for (int i = 0; i < NSYN; ++i)
        CHECK(load_synth(&syn[i], cand_dir, spec_nrows[i], spec_row[i], spec_const[i],
                         i >= 74));
    const uint64_t expert_bytes = UINT64_C(5120) * 756;
    CHECK(g_down_offset <= g_model_size &&
          (uint64_t)E * expert_bytes <= g_model_size - g_down_offset);

    CaseResult cases[5 + NSYN];
    memset(cases, 0, sizeof(cases));
    CHECK(run_case(&cand, "cand_compact", 1, out_dir, 0, &cases[0]));
    CHECK(run_case(&cand, "cand_hotlist", 0, out_dir, 0, &cases[1]));
    CHECK(run_case(&ref, "ref_compact", 1, out_dir, 0, &cases[2]));
    CHECK(run_case(&ref, "ref_hotlist", 0, out_dir, 0, &cases[3]));
    CHECK(run_case(&cand, "cand_compact_repeat", 1, out_dir, 1, &cases[4]));
    for (int i = 0; i < NSYN; ++i) {
        char nm[64];
        snprintf(nm, sizeof(nm), "syn_n%u_r%u%s%s", spec_nrows[i], spec_row[i],
                 spec_const[i] == 2 ? "_k5" : (spec_const[i] ? "_ones" : ""),
                 i == 75 ? "_hotlist" : "");
        CHECK(run_case(&syn[i], nm, i != 75, out_dir, i >= 74, &cases[5 + i]));
    }

    compare_outputs("cand_repeat_vs_cand_compact", cases[4].out, cases[0].out);
    compare_outputs("cand_compact_vs_ref_compact", cases[0].out, cases[2].out);
    printf("RANK SWEEP (nrows=64, diff vs rank 0):\n");
    for (int i = 1; i < NSYN; ++i) {
        char nm[96];
        snprintf(nm, sizeof(nm), "rank%u%s", spec_row[i], spec_const[i] ? "_ones" : "");
        compare_outputs(nm, cases[5 + i].out,
                        cases[5 + (spec_const[i] ? 66 : 0)].out);
    }
    compare_outputs("syn_n64_r0_vs_cand_compact", cases[5].out, cases[0].out);

    compare_outputs("cand_hotlist_vs_cand_compact", cases[1].out, cases[0].out);
    compare_outputs("ref_hotlist_vs_ref_compact", cases[3].out, cases[2].out);
    compare_outputs("cand_hotlist_vs_ref_compact", cases[1].out, cases[2].out);
    compare_outputs("cand_compact_vs_ref_compact", cases[0].out, cases[2].out);
    compare_outputs("cand_hotlist_vs_ref_hotlist", cases[1].out, cases[3].out);
    compare_outputs("cand_compact_vs_ref_hotlist", cases[0].out, cases[3].out);
    compare_outputs("cand_hotlist_vs_capture_cand", cases[1].out, cand.down_cap);
    compare_outputs("cand_compact_vs_capture_cand", cases[0].out, cand.down_cap);
    compare_outputs("ref_compact_vs_capture_ref", cases[2].out, ref.down_cap);
    compare_outputs("ref_hotlist_vs_capture_ref", cases[3].out, ref.down_cap);
    fflush(stdout);
    if (g_bad_outputs) {
        fprintf(stderr, "replay produced non-finite outputs; weights or layout are wrong\n");
        return 1;
    }
    puts("REPLAY COMPLETE rc=0");

    for (uint32_t i = 0; i < 5u + NSYN; ++i) free(cases[i].out);
    free(cand.ids);
    free(cand.mid);
    free(cand.down_cap);
    free(ref.ids);
    free(ref.mid);
    free(ref.down_cap);
    for (int i = 0; i < NSYN; ++i) {
        free(syn[i].ids);
        free(syn[i].mid);
        free(syn[i].down_cap);
    }
    ds4_gpu_cleanup();
    CHECK(munmap((void *)g_model, (size_t)g_model_size) == 0);
    return 0;
}

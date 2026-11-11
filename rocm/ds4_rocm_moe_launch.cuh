static int routed_moe_u64_add_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static int routed_moe_align256_checked(uint64_t v, uint64_t *out) {
    if (!out || v > UINT64_MAX - 255ull) return 0;
    *out = (v + 255ull) & ~255ull;
    return 1;
}

/* Exact worst-case number of nonempty expert tiles. Give every live expert
 * its first tile, then charge complete tile_m groups beyond those first
 * assignments. The former ceil(routes/tile_m)+expert_count bound launched
 * hundreds of empty blocks for a 12--36 route speculative verifier. */
static uint32_t routed_moe_expert_tile_capacity(
        uint32_t pair_count,
        uint32_t tile_m,
        uint32_t bucket_count) {
    const uint32_t first = pair_count < bucket_count ? pair_count : bucket_count;
    return first + (pair_count - first) / tile_m;
}

enum {
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START = 0,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
    DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
    DS4_ROCM_MOE_DECODE_PROFILE_PREP_START,
    DS4_ROCM_MOE_DECODE_PROFILE_PREP_WAIT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_START,
    DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_END,
    DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT
};

typedef struct {
    uint64_t calls;
    uint64_t split_calls;
    uint64_t q8_gateup_calls;
    uint64_t q8_down_calls;
    double finish_missing_ms;
    double split_finish_missing_ms;
    double split_down_wait_ms;
    double split_prep_wait_ms;
    double split_xquant_ms;
    double gate_resident_ms;
    double gate_missing_ms;
    double gate_full_ms;
    double mid_quant_ms;
    double down_ms;
    double profile_collect_ms;
} ds4_rocm_moe_decode_profile_stats;

typedef struct {
    int split_prep_wait;
    int split_xquant;
    int gate_resident;
    int gate_missing;
    int gate_full;
    int mid_quant;
    int down;
} ds4_rocm_moe_decode_profile_record;

static ds4_rocm_moe_decode_profile_stats g_moe_decode_profile_stats;
static cudaEvent_t g_moe_decode_profile_events[DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT];

/* Lightweight selected-split phase counters.  The heavy decode profile drains
 * the stream; these counters only touch host-side integers and are suitable
 * for paired throughput runs. */
typedef struct {
    uint64_t calls;
    uint64_t resident_only;
    uint64_t mixed;
    uint64_t resident_experts;
    uint64_t missing_experts;
    uint32_t max_missing;
} ds4_rocm_v41_split_counts;
static ds4_rocm_v41_split_counts g_v41_split_counts;
static int g_v41_split_counts_enabled = -1;
static int g_v41_split_counts_registered = 0;

static void ds4_rocm_v41_split_counts_print(void) {
    const ds4_rocm_v41_split_counts *c = &g_v41_split_counts;
    if (c->calls == 0u) return;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "V4.1 split counts calls=%llu resident_only=%llu "
            "mixed=%llu resident_experts=%llu missing_experts=%llu max_missing=%u\n",
            (unsigned long long)c->calls,
            (unsigned long long)c->resident_only,
            (unsigned long long)c->mixed,
            (unsigned long long)c->resident_experts,
            (unsigned long long)c->missing_experts,
            c->max_missing);
}

static int ds4_rocm_v41_split_counts_on(void) {
    if (g_v41_split_counts_enabled < 0) {
        const char *env = getenv("DS4_ROCM_V41_SPLIT_COUNTS");
        g_v41_split_counts_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        if (g_v41_split_counts_enabled && !g_v41_split_counts_registered) {
            atexit(ds4_rocm_v41_split_counts_print);
            g_v41_split_counts_registered = 1;
        }
    }
    return g_v41_split_counts_enabled;
}

static void ds4_rocm_v41_split_counts_note(uint32_t resident_mask,
                                            uint32_t missing_mask) {
    if (!ds4_rocm_v41_split_counts_on()) return;
    const uint32_t resident = (uint32_t)__builtin_popcount(resident_mask);
    const uint32_t missing = (uint32_t)__builtin_popcount(missing_mask);
    g_v41_split_counts.calls++;
    g_v41_split_counts.resident_experts += resident;
    g_v41_split_counts.missing_experts += missing;
    if (missing == 0u) g_v41_split_counts.resident_only++;
    else g_v41_split_counts.mixed++;
    if (missing > g_v41_split_counts.max_missing) {
        g_v41_split_counts.max_missing = missing;
    }
}
static int g_moe_decode_profile_enabled = -1;
static int g_moe_decode_profile_registered = 0;
static int g_moe_decode_profile_events_ready = 0;

static void routed_moe_decode_profile_print(void) {
    const ds4_rocm_moe_decode_profile_stats *p =
        &g_moe_decode_profile_stats;
    if (p->calls == 0u) return;
    const double calls = (double)p->calls;
    fprintf(stderr,
            DS4_GPU_LOG_PREFIX "decode MoE profile calls=%llu "
            "split=%llu q8_gateup=%llu q8_down=%llu "
            "finish_missing=%.3f ms split_finish_missing=%.3f ms "
            "split_down_wait=%.3f ms split_prep_wait=%.3f ms split_xquant=%.3f ms "
            "gate_resident=%.3f ms "
            "gate_missing=%.3f ms gate_full=%.3f ms "
            "mid_quant=%.3f ms down=%.3f ms "
            "profile_collect=%.3f ms avg_sum=%.3f ms\n",
            (unsigned long long)p->calls,
            (unsigned long long)p->split_calls,
            (unsigned long long)p->q8_gateup_calls,
            (unsigned long long)p->q8_down_calls,
            p->finish_missing_ms,
            p->split_finish_missing_ms,
            p->split_down_wait_ms,
            p->split_prep_wait_ms,
            p->split_xquant_ms,
            p->gate_resident_ms,
            p->gate_missing_ms,
            p->gate_full_ms,
            p->mid_quant_ms,
            p->down_ms,
            p->profile_collect_ms,
            (p->finish_missing_ms + p->split_finish_missing_ms +
             p->split_down_wait_ms + p->split_prep_wait_ms + p->split_xquant_ms +
             p->gate_resident_ms + p->gate_missing_ms +
             p->gate_full_ms + p->mid_quant_ms + p->down_ms) / calls);
}

static int routed_moe_decode_profile_enabled(void) {
    if (g_moe_decode_profile_enabled < 0) {
        const char *env = getenv("DS4_ROCM_MOE_DECODE_PROFILE");
        g_moe_decode_profile_enabled =
            (env != NULL && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
        if (g_moe_decode_profile_enabled && !g_moe_decode_profile_registered) {
            atexit(routed_moe_decode_profile_print);
            g_moe_decode_profile_registered = 1;
        }
    }
    return g_moe_decode_profile_enabled;
}

static int routed_moe_decode_profile_ensure_events(void) {
    if (g_moe_decode_profile_events_ready) return 1;
    for (uint32_t i = 0; i < DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT; i++) {
        cudaError_t err = cudaEventCreate(&g_moe_decode_profile_events[i]);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "decode MoE profile event create failed: %s\n",
                    cudaGetErrorString(err));
            for (uint32_t j = 0; j < i; j++) {
                (void)cudaEventDestroy(g_moe_decode_profile_events[j]);
                g_moe_decode_profile_events[j] = NULL;
            }
            return 0;
        }
    }
    g_moe_decode_profile_events_ready = 1;
    return 1;
}

static int routed_moe_decode_profile_record_event(uint32_t ev, const char *what) {
    if (ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    return cuda_ok(cudaEventRecord(g_moe_decode_profile_events[ev], 0), what);
}

static int routed_moe_decode_profile_add_event_ms(uint32_t start_ev,
                                                  uint32_t end_ev,
                                                  double *accum,
                                                  const char *what) {
    if (!accum ||
        start_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        end_ev >= DS4_ROCM_MOE_DECODE_PROFILE_EVENT_COUNT ||
        !g_moe_decode_profile_events_ready) {
        return 0;
    }
    if (!cuda_ok(cudaEventSynchronize(g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    float ms = 0.0f;
    if (!cuda_ok(cudaEventElapsedTime(&ms,
                                      g_moe_decode_profile_events[start_ev],
                                      g_moe_decode_profile_events[end_ev]),
                 what)) {
        return 0;
    }
    *accum += (double)ms;
    return 1;
}

static int routed_moe_decode_profile_collect(
        const ds4_rocm_moe_decode_profile_record *rec) {
    if (!rec) return 1;
    if (rec->split_prep_wait &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_PREP_START,
                DS4_ROCM_MOE_DECODE_PROFILE_PREP_WAIT_END,
                &g_moe_decode_profile_stats.split_prep_wait_ms,
                "decode MoE profile split prep wait")) {
        return 0;
    }
    if (rec->split_xquant &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_END,
                &g_moe_decode_profile_stats.split_xquant_ms,
                "decode MoE profile split x quant")) {
        return 0;
    }
    if (rec->gate_resident &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                &g_moe_decode_profile_stats.gate_resident_ms,
                "decode MoE profile resident gate/up")) {
        return 0;
    }
    if (rec->gate_missing &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                &g_moe_decode_profile_stats.gate_missing_ms,
                "decode MoE profile missing gate/up")) {
        return 0;
    }
    if (rec->gate_full &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                &g_moe_decode_profile_stats.gate_full_ms,
                "decode MoE profile gate/up")) {
        return 0;
    }
    if (rec->mid_quant &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                &g_moe_decode_profile_stats.mid_quant_ms,
                "decode MoE profile mid quant")) {
        return 0;
    }
    if (rec->down &&
        !routed_moe_decode_profile_add_event_ms(
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                &g_moe_decode_profile_stats.down_ms,
                "decode MoE profile down")) {
        return 0;
    }
    return 1;
}

/* Mixed IQ2_XXS-gate/Q2_K-down models already compute routed mid activations
 * as float.  Reuse the newer Q2_K expert-batch/WMMA down kernels instead of
 * re-quantizing mid to Q8_K and taking the older qwarp down path.  This keeps
 * the CyberNeurova all-Q2 path untouched while giving the standard IQ2 mix the
 * same fast Q2 down projection used by q2k_path. */
/* Opt-in producer capture; graph supplies a local row only for layer20/token8065. */
static bool v41_diag_moe_save(const char *name, const void *device, size_t bytes) {
    const char *dir = getenv("DS4_METAL_V41_DIAG_DUMP_DIR");
    if (!dir || !device || !bytes) return false;
    char path[1024];
    const int n = snprintf(path, sizeof(path), "%s/moe_%s.raw", dir, name);
    if (n <= 0 || (size_t)n >= sizeof(path)) return false;
    void *host = malloc(bytes);
    if (!host) return false;
    bool ok = cuda_ok(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "MoE diagnostic read");
    FILE *fp = ok ? fopen(path, "wb") : nullptr;
    if (!fp) ok = false;
    else {
        if (fwrite(host, 1, bytes, fp) != bytes) ok = false;
        if (fclose(fp)) ok = false;
    }
    free(host);
    return ok;
}

static int routed_moe_q2_float_down_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *down,
        const ds4_gpu_tensor *mid,
        const half *mid_h_hot,
        int hot_mid_f16,
        const char *down_w,
        const uint32_t *counts,
        const uint32_t *offsets,
        const uint32_t *sorted_pairs,
        uint32_t *hot_experts_dev,
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_expert,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes, uint32_t hot_capacity = DS4_ROCM_MAX_N_EXPERT) {
    if (!out || !down || !mid || !down_w || !counts || !offsets || !sorted_pairs ||
        n_tokens == 0u || n_total_expert == 0u || n_total_expert > DS4_ROCM_MAX_N_EXPERT ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        (expert_mid_dim % CUDA_QK_K) != 0u || expert_mid_dim == 0u || out_dim == 0u ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }

    uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (n_tokens >= 8u &&
        !cuda_ok(cudaMemcpy(h_counts, counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                 "routed_moe iq2/q2 float-down counts copy")) {
        return 0;
    }

    const uint32_t down_tile = n_tokens <= 8u ? 2u : 4u;
    const uint32_t down_rpb = 32u;
    const uint32_t down_threads = down_rpb * 32u;
    const size_t down_shmem = (size_t)down_tile * 256u * sizeof(float);
    const int use_f16_down = (out_dim & 1u) == 0u;
    half *down_h = use_f16_down ? (half *)down->ptr : NULL;

    const bool diag_force_compact = getenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_COMPACT") != NULL;
    const bool diag_force_hotlist = getenv("DS4_ROCM_V41_DIAG_DOWN_FORCE_HOTLIST") != NULL;

    uint32_t hot_count = 0u;
    uint32_t hot_max = 0u;
    const uint32_t hot_threshold = 8u;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    const int use_wmma_hot = n_tokens >= hot_threshold && hot_experts_dev &&
        !g_quality_mode &&
        (expert_mid_dim % 16u) == 0u && (out_dim % 16u) == 0u;
#else
    const int use_wmma_hot = 0;
#endif
    uint32_t h_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
    if (use_wmma_hot) {
        for (uint32_t e = 0; e < n_total_expert; e++) {
            const uint32_t c = h_counts[e];
            if (c >= hot_threshold) {
                h_hot[hot_count++] = e;
                if (c > hot_max) hot_max = c;
            }
        }
    }

    const uint32_t original_hot_count = hot_count;
    bool halo_compact = use_wmma_hot && g_halo_prefill_model && halo_prefill_scope(n_tokens) &&
        !g_glm_model && !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
        n_total_expert==256u && n_expert==6u && expert_mid_dim==2048u && out_dim==4096u &&
        down_expert_bytes==2752512u && down_row_bytes==672u && hot_mid_f16 && mid_h_hot &&
        use_f16_down && (((uintptr_t)mid_h_hot&15u)==0) && (((uintptr_t)down_w&3u)==0);
    bool v41_compact = use_wmma_hot &&
        g_deepseek41_model && !g_glm_model && !g_quality_mode && !g_ssd_streaming_mode &&
        ds4_rocm_is_gfx1151() && (n_tokens==2048u || diag_force_compact) &&
        n_total_expert==384u && n_expert==6u && expert_mid_dim==2304u && out_dim==5120u &&
        down_expert_bytes==3870720u && down_row_bytes==756u && hot_mid_f16 && mid_h_hot &&
        use_f16_down && (((uintptr_t)mid_h_hot&15u)==0) && (((uintptr_t)down_w&3u)==0);
    if(diag_force_hotlist) v41_compact = false;
    uint32_t h_tiles[1024], tile_count=0, prepared_count=0;
    if(v41_compact) {
        for(uint32_t e=0;e<384u;e++)if(h_counts[e]>=hot_threshold) {
            for(uint32_t j=0;j<h_counts[e];j+=64u) {
                if(tile_count==1024u){v41_compact=false;break;}
                h_tiles[tile_count++]=e|((j/64u)<<9u);
            }
            if(!v41_compact)break;
        }
        if(tile_count>hot_capacity)v41_compact=false;
    }
    if(halo_compact) {
        for(uint32_t e=0;e<256u;e++)if(h_counts[e]>=hot_threshold) {
            if(h_counts[e]>=512u)++prepared_count;
            else for(uint32_t j=0;j<h_counts[e];j+=64u) {
                if(tile_count==640u){halo_compact=false;break;}
                h_tiles[tile_count++]=e|((j/64u)<<8u);
            }
            if(!halo_compact)break;
        }
        if(tile_count>hot_capacity || (prepared_count && !ds4_rocm_halo_q2_prepare()))halo_compact=false;
        if(halo_compact) {
            hot_count=0;hot_max=0;
            for(uint32_t e=0;e<256u;e++)if(h_counts[e]>=hot_threshold && h_counts[e]<512u) {
                h_hot[hot_count++]=e;if(h_counts[e]>hot_max)hot_max=h_counts[e];
            }
        }
    }
    const uint32_t scalar_max = original_hot_count != 0u ? hot_threshold : 0u;
    const bool compact_active = n_tokens <= 8u && hot_experts_dev != NULL;
    uint32_t *active_count = compact_active ? hot_experts_dev : NULL;
    uint32_t *active_experts = compact_active ? hot_experts_dev + 1u : NULL;
    if (compact_active) {
        moe_build_active_experts_kernel<<<1u, 1u>>>(
                active_count, active_experts, counts, n_total_expert);
        if (!cuda_ok(cudaGetLastError(),
                     "routed_moe iq2/q2 active expert list launch")) {
            return 0;
        }
    }
    const dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb,
                         compact_active ? n_tokens * n_expert : n_total_expert,
                         1u);
    if (use_f16_down) {
        if (down_tile == 2u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<2,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else if (down_tile == 4u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else if (down_tile == 8u) {
            moe_down_q2K_expert_batch_sharedmid_kernel<8,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<16,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                    active_count, active_experts);
        }
    } else if (down_tile == 2u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<2><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else if (down_tile == 4u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else if (down_tile == 8u) {
        moe_down_q2K_expert_batch_sharedmid_kernel<8><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    } else {
        moe_down_q2K_expert_batch_sharedmid_kernel<16><<<down_grid, down_threads, down_shmem>>>(
                (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                down_expert_bytes, down_row_bytes, n_expert, n_tokens,
                active_count, active_experts);
    }
    if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down scalar launch")) return 0;
    if (halo_compact && prepared_count && !ds4_rocm_halo_q2_prepared(down_h,down_w,
            mid_h_hot,offsets,sorted_pairs,h_counts,n_tokens)) return 0;
    if (hot_count != 0u &&
        !cuda_ok(cudaMemcpy(hot_experts_dev, (halo_compact || v41_compact) ? h_tiles : h_hot,
                           ((halo_compact || v41_compact) ? tile_count : hot_count) * sizeof(uint32_t), cudaMemcpyHostToDevice),
                 "routed_moe iq2/q2 float-down hot copy")) {
        return 0;
    }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    if (use_wmma_hot && hot_count != 0u) {
        if (v41_compact && tile_count) {
            const dim3 grid(160u,1u,tile_count);
#define V41_Q2_ARGS nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,hot_experts_dev,tile_count,2304u,5120u,3870720u,756u,6u,0u
            halo_q2::q2_wstage64_n2_kernel<4,16,16,16,true,true,false,true,9><<<grid,128,9856>>>(V41_Q2_ARGS);
#undef V41_Q2_ARGS
            if(!cuda_ok(cudaGetLastError(), "V4.1 Q2 prefill launch"))return 0;
            static bool logged=false;
            if(!logged){logged=true;fprintf(stderr,"ds4: ROCm V4.1 Q2 reused panels rows=%u tiles=%u\n",n_tokens,tile_count);}
        } else if (halo_compact && tile_count) {
            const dim3 grid(128u,1u,tile_count);
            if(n_tokens==2048u) {
                halo_q2::q2_wstage64_n2_kernel<4,16,16,16,true,true,false,true><<<grid,128,9856>>>(
                    nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,
                    hot_experts_dev,tile_count,2048u,4096u,2752512u,672u,6u,0u);
            } else {
                halo_q2::q2_direct_x_n2_kernel<4,16,16,16,true,true,false,true><<<grid,128,9856>>>(
                    nullptr,down_h,down_w,nullptr,mid_h_hot,counts,offsets,sorted_pairs,
                    hot_experts_dev,tile_count,2048u,4096u,2752512u,672u,6u,0u);
            }
            if(!cuda_ok(cudaGetLastError(), "Halo Q2 direct-X compact launch"))return 0;
        } else
        {

        constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
        const int no_n2 = 0;
        /* Opt-in: 8 row-tiles per block for the hotlist WMMA down path (the
         * 14% kernel). Per-row math is unchanged, so values stay identical. */
        const uint32_t wmma_mtiles =
            getenv("DS4_ROCM_V41_DOWN_MT8") != NULL ? 8u : 4u;
        if (!no_n2) {
            if (wmma_mtiles == 4u) {
                constexpr uint32_t mt = 4u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    /* Opt-in: hold the next weight chunk in registers while the
                     * current one dequantises (bit-identical values). */
                    static const bool down_prefetch =
                        getenv("DS4_ROCM_V41_DOWN_PREFETCH") != NULL;
                    if (down_prefetch) {
                        moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true,false,true><<<grid, block, shmem_n2>>>(
                                NULL, down_h, down_w, NULL, mid_h_hot,
                                counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                                expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                    } else {
                        moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, down_h, down_w, NULL, mid_h_hot,
                                counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                                expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                    }
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else if (wmma_mtiles == 16u) {
                constexpr uint32_t mt = 16u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<16,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            } else {
                constexpr uint32_t mt = 8u;
                const dim3 block(32u * mt, 1u, 1u);
                const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                                (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
                const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                        (mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
                if (use_f16_down && hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (use_f16_down) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                            NULL, down_h, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else if (hot_mid_f16 && mid_h_hot) {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,false><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, NULL, mid_h_hot,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                } else {
                    moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                            (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                            counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                            expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
                }
            }
        } else if (wmma_mtiles == 16u) {
            constexpr uint32_t mt = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<16,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else if (wmma_mtiles == 4u) {
            constexpr uint32_t mt = 4u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<4,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        } else {
            constexpr uint32_t mt = 8u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid(out_dim / bn, (hot_max + mt * bm - 1u) / (mt * bm), hot_count);
            const size_t shmem = (mt * bm * bk + bk * bn) * sizeof(half) +
                                 (mt * bm * bn) * sizeof(float);
            moe_down_q2K_hotlist_wmma_kernel<8,16,16,16><<<grid, block, shmem>>>(
                    (float *)down->ptr, down_w, (const float *)mid->ptr,
                    counts, offsets, sorted_pairs, hot_experts_dev, hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes);
        }
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down wmma launch")) return 0;
    }
#endif

    const char *probe_row = getenv("DS4_ROCM_V41_DIAG_MOE_ROW");
    if (probe_row && g_deepseek41_model) {
        const uint32_t row = (uint32_t)strtoul(probe_row, nullptr, 10);
        if (row >= n_tokens) return 0;
        const uint64_t first_pair = (uint64_t)row * n_expert;
        if (!v41_diag_moe_save("counts_u32", counts, n_total_expert * sizeof(uint32_t)) ||
            !v41_diag_moe_save("down", (const char *)down->ptr +
                first_pair * out_dim * (use_f16_down ? sizeof(half) : sizeof(float)),
                (size_t)n_expert * out_dim * (use_f16_down ? sizeof(half) : sizeof(float)))) return 0;
        if (hot_mid_f16 && mid_h_hot &&
            !v41_diag_moe_save("mid_f16", mid_h_hot + first_pair * expert_mid_dim,
                (size_t)n_expert * expert_mid_dim * sizeof(half))) return 0;
        if (!v41_diag_moe_save("mid_f32", (const float *)mid->ptr + first_pair * expert_mid_dim,
                (size_t)n_expert * expert_mid_dim * sizeof(float))) return 0;
        fprintf(stderr, "ds4: MoE diagnostic row=%u tokens=%u hot=%u threshold=%u mid_f16=%d down_f16=%d\n",
                row, n_tokens, hot_count, hot_threshold, hot_mid_f16, use_f16_down);
    }
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    if (use_f16_down && (out_dim & 1u) == 0u) {
        const uint64_t n2 = (uint64_t)n_tokens * (out_dim >> 1u);
        moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else if (use_f16_down) {
        moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, down_h, out_dim, n_expert, n_tokens);
    } else {
        moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
    }
    return cuda_ok(cudaGetLastError(), "routed_moe iq2/q2 float-down sum launch");
}

typedef struct {
    int q4k_path;
    int iq2_path;
    int iq2_iq2_path;
    int q2k_path;
    int mxfp4_path;
    uint64_t gate_bytes;
    uint64_t down_bytes;
} routed_moe_launch_plan;

static int routed_moe_build_plan(
        const ds4_gpu_tensor *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        const ds4_gpu_tensor *mid,
        const ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        routed_moe_launch_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0u ||
        n_expert == 0u || n_expert > DS4_ROCM_N_EXPERT_USED ||
        expert_in_dim == 0u || expert_mid_dim == 0u || out_dim == 0u ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        !cuda_tensor_has_elems2(x, n_tokens, expert_in_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(selected, n_tokens, n_expert, sizeof(int32_t)) ||
        !cuda_tensor_has_elems2(weights, n_tokens, n_expert, sizeof(float)) ||
        !cuda_tensor_has_elems3(gate, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(up, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(mid, n_tokens, n_expert, expert_mid_dim, sizeof(float)) ||
        !cuda_tensor_has_elems3(down, n_tokens, n_expert, out_dim, sizeof(float)) ||
        !cuda_tensor_has_elems2(out, n_tokens, out_dim, sizeof(float))) {
        return 0;
    }
    plan->q4k_path = (gate_type == 12u && down_type == 12u);
    plan->iq2_path = (gate_type == 16u && down_type == 10u);
    plan->iq2_iq2_path = (gate_type == 16u && down_type == 16u);
    plan->q2k_path = (gate_type == 10u && down_type == 10u);
    plan->mxfp4_path = (gate_type == 39u && down_type == 39u);
    if (!plan->q4k_path && !plan->iq2_path &&
        !plan->iq2_iq2_path && !plan->q2k_path &&
        !plan->mxfp4_path) return 0;
    if (!cuda_u64_mul_checked(n_total_expert, gate_expert_bytes, &plan->gate_bytes) ||
        !cuda_u64_mul_checked(n_total_expert, down_expert_bytes, &plan->down_bytes) ||
        !cuda_model_range_fits(model_size, gate_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, up_offset, plan->gate_bytes) ||
        !cuda_model_range_fits(model_size, down_offset, plan->down_bytes)) {
        return 0;
    }
    return 1;
}

static int routed_moe_full_table_is_cached(
        const void *model_map,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_bytes,
        uint64_t down_bytes) {
    return cuda_model_range_is_cached(model_map, gate_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, up_offset, gate_bytes) &&
           cuda_model_range_is_cached(model_map, down_offset, down_bytes);
}

extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
        const void *W_gate, const void *W_up, const void *W_down,
        const float *X, const int32_t *ids, const float *router_weights,
        float *gate_f32, float *up_f32, float *mid_f32, float *down_f32,
        int expert_mid_dim, int expert_in_dim, int out_dim, int n_tokens,
        int n_experts, int n_expert_used, float clamp, int preserve_reduction,
        cudaStream_t stream);

/* Declarations for the opt-in in-process aligned-SoA path.  Both headers that
 * declare these are C++/std::vector based and are not included in this TU. */
extern "C" int ds4_rocm_repack_layer(const char *model_path, int layer,
                                    const void **out_gate, const void **out_up,
                                    const void **out_down);
extern "C" int ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
        const void *W_gate, const void *W_up, const void *W_down,
        const float *X, const int32_t *ids, const float *router_weights,
        float *gate_f32, float *up_f32, float *mid_f32, float *down_f32,
        int expert_mid_dim, int expert_in_dim, int out_dim, int n_tokens,
        int n_experts, int n_expert_used, float clamp, int preserve_reduction,
        cudaStream_t stream);

/* Routed FP16 expert kernels (rocm/gufo_f16, see its NOTICE.md). */
extern "C" int ds41_v41_routed_f16(int type, const void *w, const void *w_up, uint64_t expert_bytes,
        const uint16_t *x, uint32_t x_div, const int32_t *rows, const int32_t *bounds,
        const int32_t *tiles, uint32_t n_tiles, const float *route_w, float limit, void *y,
        int y_half, uint32_t m, uint32_t k, const void *const *w_table,
        const void *const *w_up_table);
extern "C" int ds41_v41_to_half(const float *x, uint16_t *y, uint64_t n);
extern "C" int ds41_v41_routed_plan_device(const int32_t *ids, uint32_t slots, uint32_t n_experts,
        int32_t *rows, int32_t *bounds, int32_t *tiles);

/* Default (DS4_ROCM_V41_ROUTED_F16=0 restores MMQ): V4.1 IQ2_XXS gate/up +
 * Q2_K down for a whole staged layer on the FP16 matrix cores. Activations are
 * rounded to FP16 instead of MMQ's 8-bit blocks, so results differ from the MMQ
 * path at rounding level. Measured 2026-11-10: per-layer error against exact
 * arithmetic 0.06 % (MMQ 0.4-0.7 %); official scorer NLL 0.6385 -> 0.6462
 * (default) and 0.6414 -> 0.6441 (continued 100), per-case signs flipping
 * between the two; 24K prefill 335 -> 394 tok/s, 48-256 token appends -12..-24 %.
 * Returns 1 = done, 0 = not applicable (caller continues), -1 = launch error. */
static int v41_routed_f16_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("DS4_ROCM_V41_ROUTED_F16");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* Smallest batch that takes the FP16 path (DS4_ROCM_V41_ROUTED_F16_MIN_ROWS). */
static uint32_t v41_routed_f16_min_rows(void) {
    static int rows = -1;
    if (rows < 0) {
        const char *v = getenv("DS4_ROCM_V41_ROUTED_F16_MIN_ROWS");
        rows = v && v[0] ? atoi(v) : 48;
        if (rows < 2) rows = 2;
    }
    return (uint32_t)rows;
}

static int v41_routed_f16_try(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *down,
        const char *gate_w, const char *up_w, const char *down_w,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t n_tokens,
        const void *const *gate_table, const void *const *up_table,
        const void *const *down_table) {
    static int32_t *plan_rows = NULL, *plan_bounds = NULL, *plan_tiles = NULL;
    static uint32_t plan_slots = 0;
    const uint32_t slots = n_tokens * n_expert;
    const uint32_t tile_cap = slots / 64u + n_total_expert;
    const uint64_t xh_bytes = (uint64_t)n_tokens * expert_in_dim * sizeof(uint16_t);
    const uint64_t mid_bytes = (uint64_t)slots * expert_mid_dim * sizeof(uint16_t);
    const uint64_t down_bytes = (uint64_t)slots * out_dim * sizeof(float);
    if (!gate_w || !up_w || !down_w || !gate || !up || !down || !out || !x ||
        up->bytes < xh_bytes || gate->bytes < mid_bytes || down->bytes < down_bytes ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float) ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)slots * sizeof(int32_t) ||
        weights->bytes < (uint64_t)slots * sizeof(float)) return 0;
    if (slots > plan_slots) {
        if (plan_rows) (void)cudaFree(plan_rows);
        if (plan_bounds) (void)cudaFree(plan_bounds);
        if (plan_tiles) (void)cudaFree(plan_tiles);
        plan_rows = plan_bounds = plan_tiles = NULL;
        plan_slots = 0;
        const uint32_t cap = slots < 2048u * 6u ? 2048u * 6u : slots;
        if (!cuda_ok(cudaMalloc(&plan_rows, (size_t)cap * sizeof(int32_t)), "routed f16 plan rows") ||
            !cuda_ok(cudaMalloc(&plan_bounds, 513u * sizeof(int32_t)), "routed f16 plan bounds") ||
            !cuda_ok(cudaMalloc(&plan_tiles, (size_t)(cap / 64u + 512u) * sizeof(int32_t)),
                     "routed f16 plan tiles")) return -1;
        plan_slots = cap;
    }
    uint16_t *xh = (uint16_t *)up->ptr;     /* FP16 activations */
    uint16_t *mid_h = (uint16_t *)gate->ptr; /* FP16 weighted SwiGLU per slot */
    if (ds41_v41_to_half((const float *)x->ptr, xh, (uint64_t)n_tokens * expert_in_dim) ||
        ds41_v41_routed_plan_device((const int32_t *)selected->ptr, slots, n_total_expert,
                                    plan_rows, plan_bounds, plan_tiles) ||
        ds41_v41_routed_f16(16, gate_w, up_w, gate_expert_bytes, xh, n_expert,
                            plan_rows, plan_bounds, plan_tiles, tile_cap,
                            (const float *)weights->ptr, clamp, mid_h, 1,
                            expert_mid_dim, expert_in_dim, gate_table, up_table) ||
        ds41_v41_routed_f16(10, down_w, NULL, down_expert_bytes, mid_h, 1u,
                            plan_rows, plan_bounds, plan_tiles, tile_cap,
                            NULL, 0.0f, down->ptr, 0, out_dim, expert_mid_dim, down_table, NULL)) {
        fprintf(stderr, "ds4: V4.1 routed FP16 launch failed\n");
        return -1;
    }
    const uint64_t n = (uint64_t)n_tokens * out_dim;
    moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr,
                                             out_dim, n_expert, n_tokens);
    if (!cuda_ok(cudaGetLastError(), "routed f16 sum launch")) return -1;
    static int logged = 0;
    if (!logged || (getenv("DS4_ROCM_V41_ROUTED_F16_LOG") && logged < 64)) {
        logged++;
        fprintf(stderr, "ds4: ROCm V4.1 routed FP16 gate/up + down active (rows=%u)\n", n_tokens);
    }
    return 1;
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t layer_index,
        uint32_t n_tokens,
        bool force_resident) {
    /* Opt-in layer-scoped aligned-SoA path: map pre-built artifacts and run the
     * fused SoA pipeline with the caller's original global expert ids, bypassing
     * the staged/selected machinery for this layer. DS4_V41_IQ2SOA_DIR must hold
     * l<layer>_gate.soa, l<layer>_up.soa and l<layer>_down.soa. */
    {
        static const long soa_layer = getenv("DS4_V41_IQ2SOA_LAYER") ?
            strtol(getenv("DS4_V41_IQ2SOA_LAYER"), nullptr, 10) : -1;
        if (soa_layer >= 0 && (long)layer_index == soa_layer &&
            n_tokens >= 8u && n_tokens <= 2048u && n_expert == 6u &&
            n_total_expert == 384u && expert_mid_dim == 2304u &&
            expert_in_dim == 5120u && out_dim == 5120u && selected && weights && x &&
            gate && up && mid && down) {
            static const void *soa_dev[3];
            static int soa_state;
            if (soa_state == 0)
                soa_state = ds4_rocm_repack_layer(getenv("DS4_V41_IQ2SOA_MODEL"),
                                                  (int)soa_layer,
                                                  &soa_dev[0], &soa_dev[1], &soa_dev[2]) ? 1 : -1;
            const void *maps[3] = {soa_dev[0], soa_dev[1], soa_dev[2]};
            const int maps_ready = soa_state;
            if (maps_ready == 1) {
                const int rc = ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(
                    maps[0], maps[1], maps[2],
                    (const float *)x->ptr, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                    (float *)down->ptr,
                    (int)expert_mid_dim, (int)expert_in_dim, (int)out_dim,
                    (int)n_tokens, (int)n_total_expert, (int)n_expert,
                    clamp, /*preserve_reduction=*/1, (cudaStream_t)0);
                if (rc == 0 && cuda_ok(cudaGetLastError(), "V4.1 SoA fused MoE"))
                    return 1;
            }
        }
    }
    routed_moe_launch_plan plan;
    if (!routed_moe_build_plan(out, gate, up, mid, down, model_map, model_size,
                               gate_offset, up_offset, down_offset, gate_type, down_type,
                               gate_expert_bytes, down_expert_bytes, expert_in_dim,
                               expert_mid_dim, out_dim, selected, weights, n_total_expert, n_expert, x,
                               n_tokens, &plan)) {
        return 0;
    }
    const int q4k_path = plan.q4k_path;
    const int iq2_path = plan.iq2_path;
    const int iq2_iq2_path = plan.iq2_iq2_path;
    const int iq2_gate_path = iq2_path || iq2_iq2_path;
    const int q2k_path = plan.q2k_path;
    const int mxfp4_path = plan.mxfp4_path;
    const uint64_t gate_bytes = plan.gate_bytes;
    const uint64_t down_bytes = plan.down_bytes;
    uint64_t pair_count64 = 0;
    if (!cuda_u64_mul_checked(n_tokens, n_expert, &pair_count64) ||
        pair_count64 > UINT32_MAX) {
        return 0;
    }
    const uint32_t pair_count = (uint32_t)pair_count64;
    const ds4_gpu_tensor *selected_exec = selected;
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    const char **gate_slot_ptrs = NULL;
    const char **up_slot_ptrs = NULL;
    const char **down_slot_ptrs = NULL;
    const char **resident_gate_slot_ptrs = NULL;
    const char **resident_up_slot_ptrs = NULL;
    const char **missing_gate_slot_ptrs = NULL;
    const char **missing_up_slot_ptrs = NULL;
    const uint8_t *stream_batch_pair_missing = NULL;
    uint32_t stream_resident_mask = 0;
    uint32_t stream_missing_mask = 0;
    uint32_t stream_batch_unique = 0;
    uint32_t stream_batch_resident_count = 0;
    uint32_t stream_batch_missing_count = 0;
    const int stream_full_layer =
        (n_tokens > 1u || force_resident) &&
        cuda_stream_layer_expert_cache_apply(model_map,
                                             layer_index,
                                             n_total_expert,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             &gate_w,
                                             &up_w,
                                             &down_w);
    const int full_table_cached =
        !stream_full_layer &&
        routed_moe_full_table_is_cached(model_map,
                                        gate_offset,
                                        up_offset,
                                        down_offset,
                                        gate_bytes,
                                        down_bytes);
    const int batch_stream_split_selected =
        !stream_full_layer &&
        !full_table_cached &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_apply_split(model_map,
                                               layer_index,
                                               n_total_expert,
                                               n_expert,
                                               n_tokens,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               &selected_exec,
                                               &resident_gate_slot_ptrs,
                                               &resident_up_slot_ptrs,
                                               &missing_gate_slot_ptrs,
                                               &missing_up_slot_ptrs,
                                               &down_slot_ptrs,
                                               &stream_batch_pair_missing,
                                               &stream_batch_resident_count,
                                               &stream_batch_missing_count,
                                               &stream_batch_unique);
    const int batch_stream_selected =
        !stream_full_layer &&
        !full_table_cached &&
        !batch_stream_split_selected &&
        n_tokens > 1u &&
        (iq2_gate_path || q2k_path) &&
        n_expert <= DS4_ROCM_N_EXPERT_USED &&
        cuda_stream_batch_selected_prepare(model_map,
                                           model_size,
                                           layer_index,
                                           selected,
                                           n_tokens,
                                           n_total_expert,
                                           n_expert,
                                           gate_offset,
                                           up_offset,
                                           down_offset,
                                           gate_expert_bytes,
                                           down_expert_bytes,
                                           &selected_exec,
                                           &gate_slot_ptrs,
                                           &up_slot_ptrs,
                                           &down_slot_ptrs,
                                           &stream_batch_unique);
    /* One-token resident/missing split. The split path waits for the
     * pointer-table upload before launching the resident gate/up kernel, so
     * resident compute can overlap the pending missing-expert reads. It was
     * validated bit-exact against the compact path across 2048..32768. */
    int split_selected = 0;
    if (!stream_full_layer && n_tokens == 1u) {
        split_selected =
            cuda_stream_selected_apply_split(model_map,
                                             layer_index,
                                             n_total_expert,
                                             n_expert,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             &selected_exec,
                                             &gate_w,
                                             &up_w,
                                             &down_w,
                                             &gate_slot_ptrs,
                                             &up_slot_ptrs,
                                             &down_slot_ptrs,
                                             &stream_resident_mask,
                                             &stream_missing_mask);
    }
    const int compact_selected =
        split_selected ||
        (!stream_full_layer &&
        n_tokens == 1u &&
        cuda_stream_selected_apply(model_map,
                                   layer_index,
                                   n_total_expert,
                                   n_expert,
                                   gate_expert_bytes,
                                   down_expert_bytes,
                                   &selected_exec,
                                   &gate_w,
                                   &up_w,
                                   &down_w));
    if (!compact_selected && !batch_stream_selected && !batch_stream_split_selected) {
        if (g_ssd_streaming_mode &&
            n_total_expert > n_expert &&
            !stream_full_layer &&
            !full_table_cached) {
            fprintf(stderr,
                    DS4_GPU_LOG_PREFIX "SSD streaming routed MoE missing compact selected experts "
                    "(layer=%u tokens=%u total_experts=%u selected=%u); full expert table is not mapped\n",
                    layer_index,
                    n_tokens,
                    n_total_expert,
                    n_expert);
            return 0;
        }
        if (!stream_full_layer) {
            gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
            up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
            down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
        }
    }
    if (batch_stream_selected || batch_stream_split_selected) {
        if (!down_slot_ptrs ||
            stream_batch_unique == 0) {
            return 0;
        }
        if (batch_stream_selected && (!gate_slot_ptrs || !up_slot_ptrs)) return 0;
        if (batch_stream_split_selected &&
            (!resident_gate_slot_ptrs || !resident_up_slot_ptrs ||
             !missing_gate_slot_ptrs || !missing_up_slot_ptrs ||
             !stream_batch_pair_missing)) {
            return 0;
        }
        if (!cuda_stream_batch_selected_wait_upload_ready()) return 0;
    } else if (!gate_w || !up_w || !down_w) {
        return 0;
    }
    const int split_prep_profile =
        split_selected && n_tokens == 1u && routed_moe_decode_profile_enabled();
    if (split_prep_profile) {
        if (!routed_moe_decode_profile_ensure_events() ||
            !routed_moe_decode_profile_record_event(
                DS4_ROCM_MOE_DECODE_PROFILE_PREP_START,
                "IQ2 decode MoE profile split prep start")) return 0;
    }
    if (compact_selected && !cuda_stream_selected_wait_upload_ready()) return 0;
    if (split_prep_profile &&
        !routed_moe_decode_profile_record_event(
            DS4_ROCM_MOE_DECODE_PROFILE_PREP_WAIT_END,
            "IQ2 decode MoE profile split prep wait end")) return 0;

    /* The short last chunk of a selective sweep shares its layer slot with the
     * larger chunks before it and is served here at any size. */
    const bool slot_pointer_mode = stream_full_layer && layer_index < 40u &&
        (g_stream_layer_expert_cache[layer_index & 1u].pointer_mode ||
         g_stream_layer_expert_cache[layer_index & 1u].f16_any_rows) &&
        g_stream_layer_expert_cache[layer_index & 1u].layer == layer_index;
    if (stream_full_layer && v41_routed_f16_enabled() && g_deepseek41_model &&
        !g_glm_model && !g_quality_mode && ds4_rocm_is_gfx1151() &&
        (n_tokens >= v41_routed_f16_min_rows() || (slot_pointer_mode && n_tokens >= 1u)) &&
        n_total_expert == 384u && n_expert == 6u &&
        expert_in_dim == 5120u && expert_mid_dim == 2304u && out_dim == 5120u &&
        gate_type == 16u && down_type == 10u &&
        gate_expert_bytes == 3041280u && down_expert_bytes == 3870720u) {
        /* A pointer-mode slot holds only the experts read from disk; routed
         * residents are read in place through these tables. */
        const void *const *gate_table = NULL, *const *up_table = NULL, *const *down_table = NULL;
        int pointer_mode = 0;
        if (!cuda_stream_v41_layer_pointer_tables(model_map, layer_index, gate_offset, up_offset,
                down_offset, gate_expert_bytes, down_expert_bytes,
                &gate_table, &up_table, &down_table, &pointer_mode)) return 0;
        const int f16_rc = v41_routed_f16_try(out, gate, up, down, gate_w, up_w, down_w,
            gate_expert_bytes, down_expert_bytes, expert_in_dim, expert_mid_dim, out_dim,
            selected, weights, n_total_expert, n_expert, clamp, x, n_tokens,
            gate_table, up_table, down_table);
        if (f16_rc != 0) return f16_rc > 0;
        if (pointer_mode) {
            fprintf(stderr, "ds4: V4.1 pointer-mode layer %u cannot fall back to MMQ\n", layer_index);
            return 0;
        }
    } else if (stream_full_layer && layer_index < 40u &&
               g_stream_layer_expert_cache[layer_index & 1u].pointer_mode &&
               g_stream_layer_expert_cache[layer_index & 1u].layer == layer_index) {
        /* No other kernel reads experts through pointer tables. */
        fprintf(stderr, "ds4: V4.1 pointer-mode layer %u reached a non-FP16 routed path (rows=%u)\n",
                layer_index, n_tokens);
        return 0;
    }
    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    uint64_t xq_count = 0;
    uint64_t midq_count = 0;
    uint64_t xq_bytes = 0;
    uint64_t midq_bytes = 0;
    if (!cuda_u64_mul_checked(n_tokens, xq_blocks, &xq_count) ||
        !cuda_u64_mul_checked(pair_count64, midq_blocks, &midq_count) ||
        !cuda_u64_mul_checked(xq_count, sizeof(cuda_block_q8_K), &xq_bytes) ||
        !cuda_u64_mul_checked(midq_count, sizeof(cuda_block_q8_K), &midq_bytes)) {
        return 0;
    }
    if (!q2k_path && down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        /* Correctness rollback for the optimized resident IQ2 prefill path. */
        const uint32_t disable_resident_iq2_sorted =
            iq2_gate_path && getenv("DS4_ROCM_DISABLE_RESIDENT_IQ2_SORTED") != NULL;
        const uint32_t use_mxfp4_tiny_batch =
            mxfp4_path && n_tokens <= 4u;
        const uint32_t use_sorted_pairs =
            n_tokens > 1u &&
            !use_mxfp4_tiny_batch &&
            (!q4k_path || n_tokens >= 32u) &&
            !disable_resident_iq2_sorted;
        const uint32_t use_expert_tiles = use_sorted_pairs;
        const uint32_t expert_tile_m = n_tokens <= 8u ? 4u : 8u;
        const uint32_t write_gate_up = 0u;
        const bool use_v41_wave_gate = g_deepseek41_model && iq2_gate_path &&
            n_tokens == 1u && n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && ds4_rocm_is_gfx1151();
        const uint32_t use_p2_sorted = 0u;
        const uint32_t use_atomic_down =
            !mxfp4_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_gate_row2048 =
            !q4k_path && !mxfp4_path && use_expert_tiles && n_tokens >= 128u;
        const uint32_t use_down_tile16 =
            !q4k_path && !mxfp4_path && use_atomic_down && n_tokens >= 128u;
        const uint32_t use_decode_lut_gate =
            !mxfp4_path && n_tokens == 1u && xq_blocks <= 16u;
        const uint32_t gate_row_span = 1024u;
        const uint32_t down_row_span = 1024u;
        const uint32_t use_down_row2048 =
            !q4k_path && !mxfp4_path && use_atomic_down && use_down_tile16;
        const uint32_t use_direct_down_sum6 =
            (n_tokens == 1u || use_mxfp4_tiny_batch) &&
            n_expert <= DS4_ROCM_N_EXPERT_USED;
        const uint32_t use_mxfp4_ldsB =
            mxfp4_path && use_expert_tiles && n_tokens >= 128u &&
            16u * gate_row_bytes <= 60u * 1024u &&
            (expert_mid_dim % 8u) == 0u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_LDSB") != NULL;
        const uint32_t use_mxfp4_tile32 =
            mxfp4_path && use_expert_tiles && n_tokens >= 32u &&
            (expert_mid_dim % 32u) == 0u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_TILE32") != NULL;
        /* Occupancy variants (bit-exact; see kernel comments): tile4 cuts
         * the staged-activation LDS to ~29 KiB (4 blocks/WGP), row64 runs
         * 512 threads per block (2 blocks/WGP = 32 resident warps).  Both
         * target the tile8 kernel's latency-bound profile.  Default off. */
        const uint32_t use_mxfp4_tile4 =
            mxfp4_path && use_expert_tiles && !use_mxfp4_tile32 && !use_mxfp4_ldsB &&
            n_tokens >= 4u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_TILE4") != NULL;
        const uint32_t use_mxfp4_row64 =
            mxfp4_path && use_expert_tiles && !use_mxfp4_tile32 && !use_mxfp4_ldsB &&
            !use_mxfp4_tile4 && n_tokens >= 8u &&
            getenv("DS4_ROCM_ENABLE_MXFP4_ROW64") != NULL;
        /* V4.1 Q2 uses canonical IQ2 gate/up rows and Q2_K down rows.
         * Keep other expert layouts on their existing admissions. */
        const uint32_t v41_mmq_topology =
            g_deepseek41_model && n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && n_tokens <= 2048u &&
            gate_row_bytes == 1320u && gate_expert_bytes == 3041280u &&
            down_row_bytes == 756u && down_expert_bytes == 3870720u &&
            ds4_rocm_is_gfx1151();
        const uint32_t use_rocm_mmq_gateup =
            ok && iq2_path && n_tokens >= 128u && !g_quality_mode &&
            (n_total_expert <= 256u || v41_mmq_topology) &&
            !batch_stream_selected && !batch_stream_split_selected &&
            !split_selected && !compact_selected && gate_w && up_w &&
            (stream_full_layer || full_table_cached) &&
            ds4_rocm_gfx1151_flag("DS4_ROCM_MMQ_IQ2");
        uint32_t down_row_groups = 1u;
        {
            const char *rge = getenv("DS4_ROCM_MXFP4_DOWN_RGROUP");
            if (rge && rge[0]) {
                const long rv = strtol(rge, NULL, 10);
                if (rv >= 1 && rv <= 8) down_row_groups = (uint32_t)rv;
            }
        }
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t *tile128_total = NULL;
        uint32_t *tile128_experts = NULL;
        uint32_t *tile128_starts = NULL;
        uint32_t *tile32_total = NULL;
        uint32_t *tile32_experts = NULL;
        uint32_t *tile32_starts = NULL;
        uint32_t *tile4_total = NULL;
        uint32_t *tile4_experts = NULL;
        uint32_t *tile4_starts = NULL;
        uint32_t *iq2_gate_hot_dev = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        uint32_t tile128_capacity = 0;
        uint32_t tile32_capacity = 0;
        uint32_t tile4_capacity = 0;
        if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
            fprintf(stderr,
                    "ds4: moe path n=%u mxfp4=%d stream_full=%d full_cached=%d "
                    "batch_stream=%d split=%d compact=%d sorted=%u tiles=%u\n",
                    n_tokens, mxfp4_path, stream_full_layer, full_table_cached,
                    batch_stream_selected, batch_stream_split_selected,
                    compact_selected, use_sorted_pairs, use_expert_tiles);
        }
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        int split_xquant_profile = 0;
        if (ok && split_prep_profile) {
            if (!routed_moe_decode_profile_record_event(
                    DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_START,
                    "IQ2 decode MoE profile split x quant start")) return 0;
            split_xquant_profile = 1;
        }
        if (!use_rocm_mmq_gateup) {
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        }
        if (split_xquant_profile && ok &&
            !routed_moe_decode_profile_record_event(
                DS4_ROCM_MOE_DECODE_PROFILE_XQUANT_END,
                "IQ2 decode MoE profile split x quant end")) return 0;
        if (ok && (batch_stream_selected || batch_stream_split_selected)) {
            dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
            if (batch_stream_split_selected) {
                const bool unique_split = n_tokens <= 8u && stream_batch_unique != 0u &&
                    !getenv("DS4_ROCM_DISABLE_MOE_UNIQUE_BATCH");
                const dim3 ugrid((expert_mid_dim + 127u) / 128u, stream_batch_unique, 1);
                if (stream_batch_resident_count != 0u && unique_split) {
                    moe_gate_up_mid_unique_ptrs_kernel<8u><<<ugrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            resident_gate_slot_ptrs, resident_up_slot_ptrs, xq,
                            (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                            gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            (uint32_t)pair_count, clamp, stream_batch_pair_missing, 0u);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch resident unique gate/up launch");
                } else if (stream_batch_resident_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            resident_gate_slot_ptrs,
                            resident_up_slot_ptrs,
                            stream_batch_pair_missing,
                            0u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch resident gate/up launch");
                }
                if (!ok) {
                    (void)cuda_stream_batch_selected_finish_pending_missing();
                } else {
                    ok = cuda_stream_batch_selected_finish_pending_missing();
                }
                if (ok && stream_batch_missing_count != 0u && unique_split) {
                    moe_gate_up_mid_unique_ptrs_kernel<8u><<<ugrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            missing_gate_slot_ptrs, missing_up_slot_ptrs, xq,
                            (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                            gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            (uint32_t)pair_count, clamp, stream_batch_pair_missing, 1u);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch missing unique gate/up launch");
                } else if (ok && stream_batch_missing_count != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_split_kernel<<<qgrid, 256>>>(
                            (float *)gate->ptr,
                            (float *)up->ptr,
                            (float *)mid->ptr,
                            missing_gate_slot_ptrs,
                            missing_up_slot_ptrs,
                            stream_batch_pair_missing,
                            1u,
                            xq,
                            (const int32_t *)selected_exec->ptr,
                            (const float *)weights->ptr,
                            gate_row_bytes,
                            xq_blocks,
                            expert_mid_dim,
                            n_expert,
                            0xffffffffu,
                            clamp);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch missing gate/up launch");
                }
            } else if (n_tokens <= 8u && stream_batch_unique != 0u &&
                       !getenv("DS4_ROCM_DISABLE_MOE_UNIQUE_BATCH")) {
                /* Small batches (DSpark verify): read each unique expert once. */
                moe_gate_up_mid_unique_ptrs_kernel<8u><<<dim3((expert_mid_dim + 127u) / 128u,
                                                             stream_batch_unique, 1), 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        (uint32_t)pair_count, clamp);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe streaming batch unique gate/up launch");
            } else {
                moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe streaming batch gate/up launch");
            }
            if (ok && !iq2_path) {
                dim3 midq_grid(midq_blocks, pair_count, 1);
                q8_K_quantize_kernel<<<midq_grid, 256>>>(
                        midq,
                        (const float *)mid->ptr,
                        expert_mid_dim,
                        pair_count);
                ok = cuda_ok(cudaGetLastError(), "routed_moe streaming batch mid quantize launch");
            }
            if (ok) {
                dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
                if (iq2_iq2_path) {
                    moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch iq2 down launch");
                } else if (iq2_path) {
                    const ds4_rocm_runtime_config *runtime_cfg =
                        cuda_runtime_config();
                    uint32_t rows_per_block = runtime_cfg->moe_decode_down_rpb;
                    if (rows_per_block == 0u) rows_per_block = 1u;
                    const uint32_t threads = rows_per_block * 32u;
                    dim3 float_grid(
                        (out_dim + rows_per_block - 1u) / rows_per_block,
                        n_tokens,
                        1);
                    moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<
                            float_grid, threads>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_row_bytes,
                        n_expert);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch fp32 q2 down launch");
                } else {
                    moe_down_sum6_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert,
                            n_tokens);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe streaming batch down launch");
                }
            }
            if (ok) ok = cuda_stream_batch_selected_mark_inflight();
            return ok;
        }
        if (ok && use_sorted_pairs) {
            const uint32_t bucket_count = n_total_expert;
            const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = routed_moe_expert_tile_capacity(
                pair_count, expert_tile_m, bucket_count);
            tile16_capacity = use_down_tile16 ? routed_moe_expert_tile_capacity(
                pair_count, 16u, bucket_count) : 0u;
            tile128_capacity = use_mxfp4_ldsB ? routed_moe_expert_tile_capacity(
                pair_count, 128u, bucket_count) : 0u;
            tile32_capacity = use_mxfp4_tile32 ? routed_moe_expert_tile_capacity(
                pair_count, 32u, bucket_count) : 0u;
            tile4_capacity = use_mxfp4_tile4 ? routed_moe_expert_tile_capacity(
                pair_count, 4u, bucket_count) : 0u;
            const uint64_t tile_offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t iq2_gate_hot_off = tile16_starts_off + tile16_starts_bytes;
            const uint64_t iq2_gate_hot_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
            const uint64_t tile128_offsets_off = iq2_gate_hot_off + iq2_gate_hot_bytes;
            const uint64_t tile128_offsets_bytes = use_mxfp4_ldsB ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile128_total_off = tile128_offsets_off + tile128_offsets_bytes;
            const uint64_t tile128_total_bytes = use_mxfp4_ldsB ? sizeof(uint32_t) : 0u;
            const uint64_t tile128_experts_off = tile128_total_off + tile128_total_bytes;
            const uint64_t tile128_experts_bytes = (uint64_t)tile128_capacity * sizeof(uint32_t);
            const uint64_t tile128_starts_off = tile128_experts_off + tile128_experts_bytes;
            const uint64_t tile128_starts_bytes = (uint64_t)tile128_capacity * sizeof(uint32_t);
            const uint64_t tile32_offsets_off = tile128_starts_off + tile128_starts_bytes;
            const uint64_t tile32_offsets_bytes = use_mxfp4_tile32 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile32_total_off = tile32_offsets_off + tile32_offsets_bytes;
            const uint64_t tile32_total_bytes = use_mxfp4_tile32 ? sizeof(uint32_t) : 0u;
            const uint64_t tile32_experts_off = tile32_total_off + tile32_total_bytes;
            const uint64_t tile32_experts_bytes = (uint64_t)tile32_capacity * sizeof(uint32_t);
            const uint64_t tile32_starts_off = tile32_experts_off + tile32_experts_bytes;
            const uint64_t tile32_starts_bytes = (uint64_t)tile32_capacity * sizeof(uint32_t);
            const uint64_t tile4_offsets_off = tile32_starts_off + tile32_starts_bytes;
            const uint64_t tile4_offsets_bytes = use_mxfp4_tile4 ? (uint64_t)(bucket_count + 1u) * sizeof(uint32_t) : 0u;
            const uint64_t tile4_total_off = tile4_offsets_off + tile4_offsets_bytes;
            const uint64_t tile4_total_bytes = use_mxfp4_tile4 ? sizeof(uint32_t) : 0u;
            const uint64_t tile4_experts_off = tile4_total_off + tile4_total_bytes;
            const uint64_t tile4_experts_bytes = (uint64_t)tile4_capacity * sizeof(uint32_t);
            const uint64_t tile4_starts_off = tile4_experts_off + tile4_experts_bytes;
            const uint64_t tile4_starts_bytes = (uint64_t)tile4_capacity * sizeof(uint32_t);
            const uint64_t scratch_bytes = tile4_starts_off + tile4_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                iq2_gate_hot_dev = (uint32_t *)(scratch + iq2_gate_hot_off);
                uint32_t *tile128_offsets = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_offsets_off) : NULL;
                tile128_total = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_total_off) : NULL;
                tile128_experts = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_experts_off) : NULL;
                tile128_starts = use_mxfp4_ldsB ? (uint32_t *)(scratch + tile128_starts_off) : NULL;
                uint32_t *tile32_offsets = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_offsets_off) : NULL;
                tile32_total = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_total_off) : NULL;
                tile32_experts = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_experts_off) : NULL;
                tile32_starts = use_mxfp4_tile32 ? (uint32_t *)(scratch + tile32_starts_off) : NULL;
                uint32_t *tile4_offsets = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_offsets_off) : NULL;
                tile4_total = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_total_off) : NULL;
                tile4_experts = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_experts_off) : NULL;
                tile4_starts = use_mxfp4_tile4 ? (uint32_t *)(scratch + tile4_starts_off) : NULL;
                const uint32_t use_narrow_sorted_builder =
                    g_dspark_verify_mode && iq2_gate_path && use_expert_tiles &&
                    pair_count <= 48u && !use_down_tile16 &&
                    !use_mxfp4_ldsB && !use_mxfp4_tile32 && !use_mxfp4_tile4;
                if (use_narrow_sorted_builder) {
                    moe_build_narrow_sorted_tiles_kernel<<<1u, 256u>>>(
                        counts, offsets, sorted_pairs, tile_offsets, tile_total,
                        tile_experts, tile_starts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count, expert_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(),
                                 "routed_moe narrow sorted tiles launch");
                } else {
                    ok = cuda_ok(cudaMemset(counts, 0, counts_bytes),
                                 "routed_moe sorted counts clear");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok && !use_narrow_sorted_builder) {
                    moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                        sorted_pairs,
                        offsets,
                        (const int32_t *)selected_exec->ptr,
                        pair_count,
                        bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles && !use_narrow_sorted_builder) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_mxfp4_ldsB) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile128_offsets, tile128_total, counts, 128u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile128 offsets launch");
                }
                if (ok && use_mxfp4_ldsB) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile128_experts, tile128_starts, tile128_offsets, counts, 128u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile128 build launch");
                }
                if (ok && use_mxfp4_tile32) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile32_offsets, tile32_total, counts, 32u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile32 offsets launch");
                }
                if (ok && use_mxfp4_tile32) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile32_experts, tile32_starts, tile32_offsets, counts, 32u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile32 build launch");
                }
                if (ok && use_mxfp4_tile4) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, 0 /* default stream */>>>(tile4_offsets, tile4_total, counts, 4u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile4 offsets launch");
                }
                if (ok && use_mxfp4_tile4) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256, 0, 0 /* default stream */>>>(
                        tile4_experts, tile4_starts, tile4_offsets, counts, 4u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile4 build launch");
                }
                if (ok && use_expert_tiles && !use_narrow_sorted_builder) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<(bucket_count + 255u) / 256u, 256>>>(
                            tile16_experts, tile16_starts, tile16_offsets, counts, 16u, bucket_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        uint32_t iq2_gate_hot_count = 0u;
        uint32_t iq2_gate_hot_max = 0u;
        const uint32_t iq2_gate_hot_threshold = 8u;
        const uint32_t iq2_down_hot_threshold = 8u;
        uint32_t h_iq2_gate_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t use_iq2_gate_wmma =
            ok && !use_rocm_mmq_gateup && iq2_gate_path &&
            n_tokens >= iq2_gate_hot_threshold && n_expert == 6u && !write_gate_up &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts && iq2_gate_hot_dev && use_expert_tiles &&
            (expert_in_dim % 16u) == 0u && (expert_mid_dim % 16u) == 0u &&
            !g_quality_mode;
        if (use_iq2_gate_wmma) {
            uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
            if (!cuda_ok(cudaMemcpy(h_counts, sorted_counts, n_total_expert * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe iq2 gate wmma counts copy")) {
                ok = 0;
            } else {
                for (uint32_t e = 0; e < n_total_expert; e++) {
                    const uint32_t c = h_counts[e];
                    if (c >= iq2_gate_hot_threshold) {
                        h_iq2_gate_hot[iq2_gate_hot_count++] = e;
                        if (c > iq2_gate_hot_max) iq2_gate_hot_max = c;
                    }
                }
                if (iq2_gate_hot_count != 0u &&
                    !cuda_ok(cudaMemcpy(iq2_gate_hot_dev, h_iq2_gate_hot,
                                        iq2_gate_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                             "routed_moe iq2 gate hot copy")) {
                    ok = 0;
                }
            }
        }
        const uint32_t iq2_gate_scalar_max = iq2_gate_hot_count != 0u ? iq2_gate_hot_threshold : 0u;
        const int use_iq2_hot_f16_mid =
            ((use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
              iq2_gate_hot_threshold == iq2_down_hot_threshold) ||
             use_rocm_mmq_gateup) &&
            (out_dim & 1u) == 0u && !g_quality_mode;
        half *iq2_hot_mid_h = use_iq2_hot_f16_mid ? (half *)gate->ptr : NULL;
        half *mmq_epilogue_mid_h = NULL;
        if (use_rocm_mmq_gateup && use_iq2_hot_f16_mid) {
            uint64_t down_h_bytes = 0;
            uint64_t mid_h_bytes = 0;
            if (cuda_u64_mul3_checked(pair_count64, out_dim, sizeof(half),
                                      &down_h_bytes) &&
                cuda_u64_mul3_checked(pair_count64, expert_mid_dim, sizeof(half),
                                      &mid_h_bytes) &&
                down_h_bytes <= down->bytes &&
                mid_h_bytes <= down->bytes - down_h_bytes) {
                mmq_epilogue_mid_h =
                    (half *)((char *)down->ptr + down_h_bytes);
                iq2_hot_mid_h = mmq_epilogue_mid_h;
            }
        }
        const int use_iq2_x_f16 = use_iq2_gate_wmma && iq2_gate_hot_count != 0u &&
            up->bytes >= (uint64_t)n_tokens * expert_in_dim * sizeof(half);
        half *iq2_x_h = use_iq2_x_f16 ? (half *)up->ptr : NULL;
        if (ok && use_iq2_x_f16) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(iq2_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 gate x f16 launch");
        }
        int mmq_gateup_done = 0;
        if (ok && g_halo_prefill_model && halo_prefill_scope(n_tokens) && use_rocm_mmq_gateup && !g_glm_model &&
            !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
            (n_tokens == 2048u || n_tokens == 4096u) && expert_in_dim == 4096u &&
            expert_mid_dim == 2048u && out_dim == 4096u && n_total_expert == 256u &&
            n_expert == 6u && gate_type == 16u && down_type == 10u &&
            gate_expert_bytes == 2162688u && gate_row_bytes == 1056u &&
            down_expert_bytes == 2752512u && down_row_bytes == 672u &&
            use_expert_tiles && sorted_pairs && sorted_offsets && sorted_counts &&
            use_iq2_hot_f16_mid && !write_gate_up) {
            const ds4_halo_span live[] = {
                {gate->ptr,gate->bytes},{mid->ptr,mid->bytes},{down->ptr,down->bytes},
                {x->ptr,x->bytes},{out->ptr,out->bytes},{selected_exec->ptr,selected_exec->bytes},
                {weights->ptr,weights->bytes},
                {sorted_pairs,pair_count64*sizeof(uint32_t)},
                {sorted_offsets,(n_total_expert+1u)*sizeof(uint32_t)},
                {sorted_counts,n_total_expert*sizeof(uint32_t)},
                {gate_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {up_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {down_w,(uint64_t)n_total_expert*down_expert_bytes}
            };
            const int selected = ds4_rocm_halo_s9_run(n_tokens,gate_w,up_w,
                (const float*)x->ptr,sorted_pairs,sorted_offsets,(const float*)weights->ptr,
                (float*)mid->ptr,(uint16_t*)iq2_hot_mid_h,clamp,up->ptr,up->bytes,
                live,sizeof(live)/sizeof(live[0]));
            if (selected < 0) return 0;
            mmq_gateup_done = selected;
        }
        if (ok && g_deepseek41_model && use_rocm_mmq_gateup && !g_glm_model &&
            !g_quality_mode && !g_ssd_streaming_mode && ds4_rocm_is_gfx1151() &&
            n_tokens == 2048u && expert_in_dim == 5120u &&
            expert_mid_dim == 2304u && out_dim == 5120u && n_total_expert == 384u &&
            n_expert == 6u && gate_type == 16u && down_type == 10u &&
            gate_expert_bytes == 3041280u && gate_row_bytes == 1320u &&
            down_expert_bytes == 3870720u && down_row_bytes == 756u &&
            use_expert_tiles && sorted_pairs && sorted_offsets && sorted_counts &&
            use_iq2_hot_f16_mid && !write_gate_up) {
            const ds4_halo_span live[] = {
                {gate->ptr,gate->bytes},{mid->ptr,mid->bytes},{down->ptr,down->bytes},
                {x->ptr,x->bytes},{out->ptr,out->bytes},{selected_exec->ptr,selected_exec->bytes},
                {weights->ptr,weights->bytes},
                {sorted_pairs,pair_count64*sizeof(uint32_t)},
                {sorted_offsets,(n_total_expert+1u)*sizeof(uint32_t)},
                {sorted_counts,n_total_expert*sizeof(uint32_t)},
                {gate_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {up_w,(uint64_t)n_total_expert*gate_expert_bytes},
                {down_w,(uint64_t)n_total_expert*down_expert_bytes}
            };
            const int selected = ds4_rocm_v41_gate_up(n_tokens,gate_w,up_w,
                (const float*)x->ptr,sorted_pairs,sorted_offsets,(const float*)weights->ptr,
                (float*)mid->ptr,(uint16_t*)iq2_hot_mid_h,clamp,up->ptr,up->bytes,
                live,sizeof(live)/sizeof(live[0]));
            if (selected < 0) return 0;
            mmq_gateup_done = selected;
            static int logged_v41_gate_up = 0;
            if (selected && !logged_v41_gate_up) {
                logged_v41_gate_up = 1;
                fprintf(stderr, "ds4: ROCm V4.1 co-computed IQ2 gate/up rows=%u\n", n_tokens);
            }
        }
        if (ok && use_rocm_mmq_gateup && !mmq_gateup_done) {
            int mmq_rc = ds4_mmq_init(0) == 0 ? 0 : -1;
            const int use_fused_swiglu = mmq_epilogue_mid_h != NULL;
            /* The gfx1151 MMQ pair is stable through 2048 token rows. Tile
             * larger prefills instead of letting its flattened assignment
             * grid corrupt the tail of a 4096-row batch. */
            const uint32_t mmq_token_cap = 2048u;
            for (uint32_t token0 = 0; mmq_rc == 0 && token0 < n_tokens; ) {
                const uint32_t tile_tokens =
                    n_tokens - token0 < mmq_token_cap ?
                    n_tokens - token0 : mmq_token_cap;
                const uint64_t x_offset =
                    (uint64_t)token0 * expert_in_dim;
                const uint64_t pair_offset =
                    (uint64_t)token0 * n_expert;
                const uint64_t out_offset =
                    pair_offset * expert_mid_dim;
                if (use_fused_swiglu) {
                    mmq_rc = ds4_mmq_iq2_xxs_moe_pair_swiglu(
                        gate_w, up_w,
                        (const float *)x->ptr + x_offset,
                        (const int32_t *)selected_exec->ptr + pair_offset,
                        (const float *)weights->ptr + pair_offset,
                        (float *)gate->ptr + out_offset,
                        (float *)up->ptr + out_offset,
                        (float *)mid->ptr + out_offset,
                        mmq_epilogue_mid_h + out_offset,
                        (int)expert_mid_dim, (int)expert_in_dim,
                        (int)tile_tokens, (int)n_total_expert, (int)n_expert,
                        clamp, (cudaStream_t)0);
                } else {
                    mmq_rc = ds4_mmq_iq2_xxs_moe_pair(
                        gate_w, up_w,
                        (const float *)x->ptr + x_offset,
                        (const int32_t *)selected_exec->ptr + pair_offset,
                        (float *)gate->ptr + out_offset,
                        (float *)up->ptr + out_offset,
                        (int)expert_mid_dim, (int)expert_in_dim,
                        (int)tile_tokens, (int)n_total_expert, (int)n_expert,
                        (cudaStream_t)0);
                }
                token0 += tile_tokens;
            }
            if (mmq_rc == 0) {
                const uint64_t mid_count = pair_count64 * expert_mid_dim;
                if (use_fused_swiglu) {
                    mmq_gateup_done = 1;
                } else {
                    moe_swiglu_weighted_f32_kernel<<<
                        (uint32_t)((mid_count + 255u) / 256u), 256>>>(
                        (float *)mid->ptr, (const float *)gate->ptr,
                        (const float *)up->ptr, (const float *)weights->ptr,
                        mid_count, expert_mid_dim, clamp);
                    mmq_gateup_done = cuda_ok(
                        cudaGetLastError(), "routed_moe MMQ gate/up epilogue launch");
                    if (mmq_gateup_done && use_iq2_hot_f16_mid) {
                        f32_to_f16_kernel<<<
                            (uint32_t)((mid_count + 255u) / 256u), 256>>>(
                            iq2_hot_mid_h, (const float *)mid->ptr, mid_count);
                        mmq_gateup_done = cuda_ok(
                            cudaGetLastError(), "routed_moe MMQ mid f16 launch");
                    }
                }
                static int logged_mmq_gateup = 0;
                if (mmq_gateup_done && !logged_mmq_gateup) {
                    logged_mmq_gateup = 1;
                    fprintf(stderr,
                        "ds4: ROCm routed MoE using tuned MMQ IQ2 gate/up%s\n",
                        use_fused_swiglu ? " with fused SwiGLU write-back" : "");
                }
            } else if (!v41_mmq_topology) {
                (void)cudaGetLastError();
                static int logged_mmq_fallback = 0;
                if (!logged_mmq_fallback) {
                    logged_mmq_fallback = 1;
                    fprintf(stderr, "ds4: ROCm MMQ IQ2 gate/up returned %d; falling back\n", mmq_rc);
                }
            }
            /* Opt-in producer probe in the path production actually uses
             * (fused MMQ gate/up + SwiGLU write-back). Drains first so the
             * snapshot cannot be confused with a partially published buffer. */
            if (ok && mmq_gateup_done && gate && mid && x && weights) {
                const char *probe_row = getenv("DS4_ROCM_V41_DIAG_MOE_ROW");
                if (probe_row) {
                    const uint32_t row = (uint32_t)strtoul(probe_row, nullptr, 10);
                    if (row >= n_tokens) return 0;
                    const uint64_t first_pair = (uint64_t)row * n_expert;
                    const size_t mid_elems = (size_t)n_expert * expert_mid_dim;
                    if (!cuda_ok(cudaDeviceSynchronize(), "MoE producer probe drain") ||
                        !v41_diag_moe_save("x", (const float *)x->ptr +
                                           (uint64_t)row * expert_in_dim,
                                           (size_t)expert_in_dim * sizeof(float)) ||
                        !v41_diag_moe_save("gate", (const float *)gate->ptr +
                                           first_pair * expert_mid_dim,
                                           mid_elems * sizeof(float)) ||
                        !v41_diag_moe_save("mid_sync", (const float *)mid->ptr +
                                           first_pair * expert_mid_dim,
                                           mid_elems * sizeof(float)) ||
                        !v41_diag_moe_save("weights", (const float *)weights->ptr + first_pair,
                                           (size_t)n_expert * sizeof(float))) return 0;
                    fprintf(stderr, "ds4: MoE fused producer probe row=%u tokens=%u\n",
                            row, n_tokens);
                }
            }
            if (v41_mmq_topology && !mmq_gateup_done) {
                /* This route skipped Q8_K x preparation. A legacy fallback
                 * would consume unwritten xq or stale half mid storage. */
                fprintf(stderr, "ds4: V4.1 MMQ gate/up or epilogue failed (%d)\n", mmq_rc);
                if (!cuda_ok(cudaStreamSynchronize((cudaStream_t)0),
                             "V4.1 MMQ failure drain")) abort();
                return 0;
            }
        }
        int split_gateup_done = 0;
        const int iq2_decode_profile =
            n_tokens == 1u && routed_moe_decode_profile_enabled();
        ds4_rocm_moe_decode_profile_record iq2_profile_rec = {0};
        iq2_profile_rec.split_prep_wait = split_prep_profile;
        iq2_profile_rec.split_xquant = split_xquant_profile;
        if (iq2_decode_profile) {
            if (!routed_moe_decode_profile_ensure_events()) return 0;
            g_moe_decode_profile_stats.calls++;
            if (split_selected) g_moe_decode_profile_stats.split_calls++;
        }
        if (ok && split_selected) {
            ds4_rocm_v41_split_counts_note(stream_resident_mask,
                                           stream_missing_mask);
            const int allow_resident_only =
                getenv("DS4_ROCM_V41_DISABLE_ALL_RESIDENT_SPLIT") == NULL;
            const int split_supported =
                iq2_gate_path &&
                n_tokens == 1u &&
                n_expert <= DS4_ROCM_N_EXPERT_USED &&
                !q4k_path &&
                !sorted_pairs &&
                stream_resident_mask != 0 &&
                (stream_missing_mask != 0 || allow_resident_only);
            if (split_supported) {
                if (stream_missing_mask == 0u && allow_resident_only) {
                    static int logged_all_resident = 0;
                    if (!logged_all_resident) {
                        logged_all_resident = 1;
                        fprintf(stderr,
                                "ds4: V4.1 all-resident split active "
                                "(resident_mask=0x%x)\n",
                                stream_resident_mask);
                    }
                }
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (iq2_decode_profile &&
                    !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                        "IQ2 decode MoE profile resident gate/up start")) {
                    return 0;
                }
                if (use_v41_wave_gate) {
                    ds4_v41_launch_gate_up_wave_ptrs(
                        ds4_rocm_v41_gate_waves(),
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, stream_resident_mask, clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_resident_mask,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp);
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe split resident gate/up launch");
                if (iq2_decode_profile && ok) {
                    iq2_profile_rec.gate_resident = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                            "IQ2 decode MoE profile resident gate/up end")) {
                        return 0;
                    }
                }
                double split_finish_t0 = 0.0;
                if (iq2_decode_profile) split_finish_t0 = cuda_wall_sec();
                if (!ok) {
                    (void)cuda_stream_selected_finish_pending_missing(0);
                } else if (getenv("DS4_ROCM_V41_ENABLE_PARTIAL_GATEUP") == NULL) {
                    ok = cuda_stream_selected_finish_pending_missing(0);
                } else {
                    ok = cuda_stream_selected_finish_pending_gateup();
                    static int logged_partial_gateup = 0;
                    if (ok && !logged_partial_gateup) {
                        logged_partial_gateup = 1;
                        fprintf(stderr,
                                "ds4: V4.1 split partial gate/up wait active\n");
                    }
                }
                if (iq2_decode_profile) {
                    g_moe_decode_profile_stats.split_finish_missing_ms +=
                        (cuda_wall_sec() - split_finish_t0) * 1000.0;
                }
                if (ok && stream_missing_mask != 0u && iq2_decode_profile &&
                    !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                        "IQ2 decode MoE profile missing gate/up start")) {
                    return 0;
                }
                if (ok && stream_missing_mask != 0u && use_v41_wave_gate) {
                    ds4_v41_launch_gate_up_wave_ptrs(
                        ds4_rocm_v41_gate_waves(),
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_slot_ptrs, up_slot_ptrs, xq,
                        (const int32_t *)selected_exec->ptr, (const float *)weights->ptr,
                        0, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, stream_missing_mask, clamp);
                } else if (ok && stream_missing_mask != 0u && use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        stream_missing_mask,
                        clamp);
                } else if (ok && stream_missing_mask != 0u) {
                    moe_gate_up_mid_qwarp32_ptrs_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp);
                }
                if (ok) ok = cuda_ok(cudaGetLastError(), "routed_moe split missing gate/up launch");
                if (!ok) {
                    (void)cuda_stream_selected_finish_pending_missing(0);
                }
                if (iq2_decode_profile && ok && stream_missing_mask != 0u) {
                    iq2_profile_rec.gate_missing = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                            "IQ2 decode MoE profile missing gate/up end")) {
                        return 0;
                    }
                }
                split_gateup_done = ok;
                if (split_gateup_done &&
                    getenv("DS4_ROCM_V41_SPLIT_DEBUG") != NULL) {
                    static int logged_v41_split = 0;
                    if (!logged_v41_split) {
                        logged_v41_split = 1;
                        fprintf(stderr,
                                "ds4: ROCm V4.1 split resident/missing decode active "
                                "(resident_mask=0x%x missing_mask=0x%x)\n",
                                stream_resident_mask, stream_missing_mask);
                    }
                }
            } else {
                ok = cuda_stream_selected_finish_pending_missing(
                        stream_resident_mask | stream_missing_mask);
                /* finish_pending_missing(all) compacts both masks into the
                 * selected cache and records a fresh upload event.  The wait
                 * above only covered the pointer table from load, so join the
                 * compact copy before the fallback kernels consume it. */
                if (ok) ok = cuda_stream_selected_wait_upload_ready();
            }
        }
        if (ok && !split_gateup_done && !mmq_gateup_done) {
            if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
                fprintf(stderr,
                        "ds4: moe gate/up launch check: sorted_pairs=%p offsets=%p counts=%p "
                        "tile_total=%p tile_experts=%p tile_starts=%p\n",
                        (void *)sorted_pairs, (void *)sorted_offsets,
                        (void *)sorted_counts, (void *)tile_total,
                        (void *)tile_experts, (void *)tile_starts);
            }
            dim3 mgrid((expert_mid_dim + 31u) / 32u, pair_count, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_gate_up_mid_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    }
                } else if (mxfp4_path) {
                    if (use_mxfp4_tile32 && tile32_total && tile32_experts && tile32_starts) {
                        dim3 t32grid((expert_mid_dim + 31u) / 32u, tile32_capacity, 1);
                        moe_gate_up_mid_mxfp4_expert_tile32_row32_kernel<<<t32grid, 256, 0, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile32_total, tile32_experts, tile32_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_ldsB && tile128_total && tile128_experts && tile128_starts) {
                        /* B-staged prefill path: 8 output rows x up to 128
                         * tokens per block; expert weights read ~once. */
                        const uint32_t ldsB_shmem = 16u * (uint32_t)gate_row_bytes;
                        if (ldsB_shmem > 48u * 1024u) {
                            static int ldsB_shmem_attr_set = 0;
                            if (!ldsB_shmem_attr_set) {
                                const cudaError_t attr_err = cudaFuncSetAttribute(
                                    (const void *)moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    16u * 3808u);
                                if (attr_err != cudaSuccess) {
                                    (void)cudaGetLastError();
                                    return 0;
                                }
                                ldsB_shmem_attr_set = 1;
                            }
                        }
                        dim3 bgrid(expert_mid_dim / 8u, tile128_capacity, 1);
                        moe_gate_up_mid_mxfp4_expert_row8_ldsB_kernel<<<bgrid, 256, ldsB_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile128_total, tile128_experts, tile128_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_tile4) {
                        const uint32_t tile4_shmem = xq_blocks <= 28u ?
                            4u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                        dim3 tgrid(tile4_capacity, (expert_mid_dim + 31u) / 32u, 1);
                        moe_gate_up_mid_mxfp4_expert_tileN_kernel<4u, 32u><<<tgrid, 256, tile4_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile4_total, tile4_experts, tile4_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else if (use_mxfp4_row64) {
                        const uint32_t row64_shmem = xq_blocks <= 28u ?
                            8u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                        if (row64_shmem > 48u * 1024u) {
                            static int row64_shmem_attr_set = 0;
                            if (!row64_shmem_attr_set) {
                                /* Hoisted: the compat macro cannot see a
                                 * comma inside a template-id argument. */
                                static const void *row64_fn =
                                    (const void *)&moe_gate_up_mid_mxfp4_expert_tileN_kernel<8u, 64u>;
                                const cudaError_t attr_err = cudaFuncSetAttribute(
                                    row64_fn,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    8u * 28u * 260u);
                                if (attr_err != cudaSuccess) {
                                    (void)cudaGetLastError();
                                    return 0;
                                }
                                row64_shmem_attr_set = 1;
                            }
                        }
                        dim3 tgrid(tile_capacity, (expert_mid_dim + 63u) / 64u, 1);
                        moe_gate_up_mid_mxfp4_expert_tileN_kernel<8u, 64u><<<tgrid, 512, row64_shmem, 0 /* default stream */>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            0u, write_gate_up, clamp);
                    } else {
                    dim3 tgrid(tile_capacity, (expert_mid_dim + 31u) / 32u, 1);
                    if (getenv("DS4_ROCM_MOE_PATH_DEBUG") != NULL) {
                        fprintf(stderr,
                                "ds4: moe mxfp4 gate/up tile8 launch grid=(%u,%u) mid_dim=%u tile_capacity=%u xq_blocks=%u\n",
                                tgrid.x, tgrid.y, expert_mid_dim, tile_capacity, xq_blocks);
                    }
                    /* LDS staging of 8 activation rows as aligned quant
                     * slices plus scales, sized to the actual xq_blocks.
                     * Raised to 28 blocks (7168-dim x at Q8_K=256): 56.9 KiB
                     * needs the >48 KiB dynamic-smem opt-in, set once. */
                    const uint32_t tile8_shmem = xq_blocks <= 28u ?
                        8u * xq_blocks * (256u + (uint32_t)sizeof(float)) : 0u;
                    if (tile8_shmem > 48u * 1024u) {
                        static int tile8_shmem_attr_set = 0;
                        if (!tile8_shmem_attr_set) {
                            const cudaError_t attr_err = cudaFuncSetAttribute(
                                (const void *)moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                8u * 28u * 260u);
                            if (attr_err != cudaSuccess) {
                                (void)cudaGetLastError();
                                return 0;
                            }
                            tile8_shmem_attr_set = 1;
                        }
                    }
                    moe_gate_up_mid_mxfp4_expert_tile8_row32_kernel<<<tgrid, 256, tile8_shmem, 0 /* default stream */>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        0u, write_gate_up, clamp);
                    }
                } else if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            iq2_gate_scalar_max, write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        iq2_gate_scalar_max, write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                if (q4k_path) {
                    moe_gate_up_mid_q4K_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, pair_count, 1);
                if (q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (mxfp4_path) {
                    dim3 mxgrid((expert_mid_dim + 7u) / 8u, pair_count, 1);
                    moe_gate_up_mid_decode_mxfp4_qwarp32_kernel<<<mxgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_v41_wave_gate) {
                    // V4.1 scalar: raw IQ2 and existing Q8_K activation.
                    // Four waves own four output rows; each lane handles32 values.
                    ds4_v41_launch_gate_up_wave_pairs(
                        ds4_rocm_v41_gate_waves(),
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr, gate_expert_bytes, gate_row_bytes,
                        xq_blocks, expert_mid_dim, n_expert, write_gate_up,
                        0xffffffffu, clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        0xffffffffu,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            if (ok && use_iq2_gate_wmma && iq2_gate_hot_count != 0u) {
                constexpr uint32_t bm = 16u, bn = 16u, bk = 16u;
                const uint32_t wmma_mtiles = 4u;
                if (wmma_mtiles == 4u) {
                    constexpr uint32_t mt = 4u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<4,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                } else {
                    constexpr uint32_t mt = 8u;
                    const dim3 block(32u * mt, 1u, 1u);
                    const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                                    (iq2_gate_hot_max + mt * bm - 1u) / (mt * bm),
                                    iq2_gate_hot_count);
                    const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                            (4u * mt * bm * bn) * sizeof(float);
                    if (use_iq2_hot_f16_mid && use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_hot_f16_mid) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,true><<<grid, block, shmem_n2>>>(
                                NULL, iq2_hot_mid_h, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else if (use_iq2_x_f16) {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16,false,true><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, iq2_x_h,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    } else {
                        moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel<8,16,16,16><<<grid, block, shmem_n2>>>(
                                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, NULL,
                                (const float *)weights->ptr, sorted_counts, sorted_offsets, sorted_pairs,
                                iq2_gate_hot_dev, iq2_gate_hot_count, expert_in_dim, expert_mid_dim,
                                gate_expert_bytes, gate_row_bytes, clamp);
                    }
                }
                ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 wmma hot gate/up launch");
            }
#endif
        }
        const uint32_t use_iq2_q2_float_down =
            ok && iq2_path && n_tokens > 1u &&
            n_expert <= DS4_ROCM_N_EXPERT_USED &&
            sorted_pairs && sorted_offsets && sorted_counts && tile_experts;
        const uint32_t use_v41_q8_down =
            ok && g_deepseek41_model && iq2_path && n_tokens == 1u && !g_quality_mode &&
            n_total_expert == 384u && n_expert == 6u &&
            expert_in_dim == 5120u && expert_mid_dim == 2304u &&
            out_dim == 5120u && ds4_rocm_is_gfx1151();
        const uint32_t use_iq2_q2_decode_float_down =
            ok && iq2_path && n_tokens == 1u && !use_v41_q8_down &&
            n_expert <= DS4_ROCM_N_EXPERT_USED;
        if (ok && !use_iq2_q2_float_down &&
            !use_iq2_q2_decode_float_down) {
            if (iq2_decode_profile &&
                !routed_moe_decode_profile_record_event(
                    DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                    "IQ2 decode MoE profile mid quant start")) {
                return 0;
            }
            dim3 midq_grid(midq_blocks, pair_count, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
            if (iq2_decode_profile && ok) {
                iq2_profile_rec.mid_quant = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                        "IQ2 decode MoE profile mid quant end")) {
                    return 0;
                }
            }
        }
        if (ok && split_gateup_done) {
            const double down_wait_t0 =
                iq2_decode_profile ? cuda_wall_sec() : 0.0;
            ok = cuda_stream_selected_finish_pending_missing(0);
            if (iq2_decode_profile) {
                g_moe_decode_profile_stats.split_down_wait_ms +=
                    (cuda_wall_sec() - down_wait_t0) * 1000.0;
            }
        }
        int direct_iq2_down_done = 0;
        if (ok && use_v41_q8_down) {
            if (iq2_decode_profile &&
                !routed_moe_decode_profile_record_event(
                    DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                    "IQ2 decode MoE profile down start")) {
                return 0;
            }
            if (split_gateup_done) {
                ds4_v41_launch_down_wave_ptrs(
                    ds4_rocm_v41_down_waves(),
                    (float *)out->ptr, down_slot_ptrs, midq,
                    (const int32_t *)selected_exec->ptr, 0,
                    down_row_bytes, midq_blocks, out_dim, n_expert);
            } else {
                ds4_v41_launch_down_wave(
                    ds4_rocm_v41_down_waves(),
                    (float *)out->ptr, down_w, midq,
                    (const int32_t *)selected_exec->ptr, down_expert_bytes,
                    down_row_bytes, midq_blocks, out_dim, n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "V4.1 quantized wave down launch");
            direct_iq2_down_done = ok;
            if (iq2_decode_profile && ok) {
                iq2_profile_rec.down = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                        "IQ2 decode MoE profile down end")) {
                    return 0;
                }
            }
        } else if (ok && use_iq2_q2_decode_float_down) {
            const ds4_rocm_runtime_config *runtime_cfg = cuda_runtime_config();
            uint32_t rows_per_block = runtime_cfg->moe_decode_down_rpb;
            if (rows_per_block == 0u) rows_per_block = 1u;
            const uint32_t threads = rows_per_block * 32u;
            const dim3 dgrid(
                (out_dim + rows_per_block - 1u) / rows_per_block,
                n_tokens,
                1);
            if (split_gateup_done) {
                moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<dgrid, threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            } else {
                moe_down_q2K_sum_rows_w32_kernel<<<dgrid, threads>>>(
                    (float *)out->ptr,
                    down_w,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_expert_bytes,
                    down_row_bytes,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(),
                         "routed_moe decode fp32 q2 down launch");
            direct_iq2_down_done = ok;
        } else if (ok && iq2_iq2_path) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens, 1);
            if (split_gateup_done) {
                moe_down_iq2_sum_qwarp32_ptrs_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            } else {
                moe_down_iq2_sum_qwarp32_batch_kernel<<<dgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert,
                        n_tokens);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe iq2 down launch");
            direct_iq2_down_done = ok;
        }
        int split_ptr_down_done = 0;
        if (ok && !direct_iq2_down_done && split_gateup_done) {
            moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    midq,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe split ptr down launch");
            split_ptr_down_done = ok;
        }
        if (ok) {
            if (direct_iq2_down_done) {
                /* The IQ2 direct-sum kernel writes final token rows. */
            } else if (split_ptr_down_done) {
                /* The split pointer-table path writes the final token row. */
            } else if (use_iq2_q2_float_down) {
                if (!g_quality_mode &&
                    g_dspark_verify_mode && n_tokens <= 6u &&
                    ds4_rocm_gfx1151_flag(
                        "DS4_ROCM_DSPARK_Q2_DOWN_MMVQ") &&
                    ds4_mmq_init(0) == 0 &&
                    ds4_mmq_q2_K_moe_down_sum6_vec(
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (float *)out->ptr,
                        (int)out_dim,
                        (int)expert_mid_dim,
                        (int)n_tokens,
                        (int)n_total_expert,
                        (int)n_expert,
                        (cudaStream_t)0) == 0) {
                    ok = 1;
                } else {
                    ok = routed_moe_q2_float_down_launch(
                            out, down, mid, iq2_hot_mid_h, use_iq2_hot_f16_mid, down_w,
                            sorted_counts, sorted_offsets, sorted_pairs, tile_experts,
                            n_tokens, n_total_expert, n_expert, expert_mid_dim, out_dim,
                            down_expert_bytes, down_row_bytes, tile_capacity);
                }
            } else {
            dim3 dgrid((out_dim + 31u) / 32u, pair_count, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else if (mxfp4_path) {
                    dim3 mxgrid((out_dim + 7u) / 8u, n_tokens, 1);
                    if (n_tokens == 1u) {
                        moe_down_mxfp4_sum6_qwarp32_kernel<false><<<mxgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                    } else {
                        moe_down_mxfp4_sum6_qwarp32_kernel<true><<<mxgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                    }
                } else {
                    moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (q4k_path) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    if (expert_tile_m == 8u) {
                        moe_down_q4K_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        moe_down_q4K_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (mxfp4_path) {
                    const uint32_t row_blocks = (out_dim + 31u) / 32u;
                    dim3 tgrid((row_blocks + down_row_groups - 1u) / down_row_groups, down_tile_capacity, 1);
                    moe_down_mxfp4_expert_tile8_row32_kernel<<<tgrid, 256, 0, 0 /* default stream */>>>(
                        (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts,
                        down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, 0u, down_row_groups);
                } else if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected_exec->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                if (q4k_path) {
                    moe_down_q4K_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        sorted_pairs,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else {
                if (q4k_path) {
                    moe_down_q4K_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                        (float *)down->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected_exec->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
            }
        }
        if (ok && !direct_iq2_down_done && !use_atomic_down &&
            !use_direct_down_sum6 && !use_iq2_q2_float_down) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        if (iq2_decode_profile) {
            const double collect_t0 = cuda_wall_sec();
            if (!routed_moe_decode_profile_collect(&iq2_profile_rec)) ok = 0;
            g_moe_decode_profile_stats.profile_collect_ms +=
                (cuda_wall_sec() - collect_t0) * 1000.0;
        }
        return ok;
    }

    const ds4_rocm_runtime_config *cfg = cuda_runtime_config();
    if (q2k_path && (batch_stream_selected || batch_stream_split_selected)) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                       pair_count,
                       1);
        if (batch_stream_split_selected) {
            if (stream_batch_resident_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        resident_gate_slot_ptrs,
                        resident_up_slot_ptrs,
                        stream_batch_pair_missing,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch resident gate/up launch");
            }
            if (!ok) {
                (void)cuda_stream_batch_selected_finish_pending_missing();
            } else {
                ok = cuda_stream_batch_selected_finish_pending_missing();
            }
            if (ok && stream_batch_missing_count != 0u) {
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        missing_gate_slot_ptrs,
                        missing_up_slot_ptrs,
                        stream_batch_pair_missing,
                        1u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        0xffffffffu,
                        clamp,
                        store_gate_up);
                ok = cuda_ok(cudaGetLastError(),
                             "routed_moe q2 streaming batch missing gate/up launch");
            }
        } else {
            moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_slot_ptrs,
                    up_slot_ptrs,
                    NULL,
                    0u,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    0xffffffffu,
                    clamp,
                    store_gate_up);
            ok = cuda_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch gate/up launch");
        }
        if (ok) {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block,
                           n_tokens,
                           1);
            moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                    (float *)out->ptr,
                    down_slot_ptrs,
                    (const float *)mid->ptr,
                    (const int32_t *)selected_exec->ptr,
                    n_tokens,
                    expert_mid_dim,
                    out_dim,
                    down_row_bytes,
                    n_expert);
            ok = cuda_ok(cudaGetLastError(),
                         "routed_moe q2 streaming batch down launch");
        }
        if (ok) ok = cuda_stream_batch_selected_mark_inflight();
        return ok;
    }

    if (q2k_path && n_tokens >= 32u && !cfg->graph_dump) {
        const uint32_t bucket_count = n_total_expert;
        const uint64_t counts_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t offsets_bytes = (uint64_t)(bucket_count + 1u) * sizeof(uint32_t);
        const uint64_t cursors_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        uint64_t sorted_bytes = 0;
        const uint64_t hot_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t hot_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_gate_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
        const uint64_t f16_low_down_bytes = (uint64_t)bucket_count * sizeof(uint32_t);
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        const int moe_wmma_hot = !g_quality_mode &&
                                 expert_in_dim % 16u == 0u &&
                                 expert_mid_dim % 16u == 0u &&
                                 out_dim % 16u == 0u;
#else
        const int moe_wmma_hot = 0;
#endif
        uint64_t f16_mid_bytes = 0;
        uint64_t f16_down_bytes = 0;
        uint64_t wmma_x_bytes = 0;
        if (!cuda_u64_mul_checked(pair_count64, sizeof(uint32_t), &sorted_bytes) ||
            (moe_wmma_hot && (
                !cuda_u64_mul3_checked(pair_count64, expert_mid_dim, sizeof(__half), &f16_mid_bytes) ||
                !cuda_u64_mul3_checked(pair_count64, out_dim, sizeof(__half), &f16_down_bytes) ||
                !cuda_u64_mul3_checked(n_tokens, expert_in_dim, sizeof(__half), &wmma_x_bytes)))) {
            return 0;
        }
        uint64_t wmma_list_base = 0;
        uint64_t base_scratch_end = 0;
        uint64_t tmp64 = 0;
        uint64_t f16_mid_off = 0;
        uint64_t f16_down_off = 0;
        uint64_t wmma_x_off = 0;
        uint64_t scratch_bytes = 0;
        if (!routed_moe_u64_add_checked(counts_bytes, offsets_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, cursors_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, sorted_bytes, &wmma_list_base) ||
            !routed_moe_u64_add_checked(wmma_list_base, hot_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, hot_down_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_gate_bytes, &base_scratch_end) ||
            !routed_moe_u64_add_checked(base_scratch_end, f16_low_down_bytes, &base_scratch_end) ||
            !routed_moe_align256_checked(base_scratch_end, &f16_mid_off) ||
            !routed_moe_u64_add_checked(f16_mid_off, f16_mid_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &f16_down_off) ||
            !routed_moe_u64_add_checked(f16_down_off, f16_down_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &wmma_x_off) ||
            !routed_moe_u64_add_checked(wmma_x_off, wmma_x_bytes, &tmp64) ||
            !routed_moe_align256_checked(tmp64, &scratch_bytes)) {
            return 0;
        }
        uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes, "routed_moe q2 expert batch buckets");
        if (!scratch) return 0;
        uint32_t *counts = (uint32_t *)scratch;
        uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
        uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
        uint32_t *sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
        uint32_t *wmma_gate_hot_dev = (uint32_t *)(scratch + wmma_list_base);
        uint32_t *wmma_down_hot_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes);
        uint32_t *wmma_gate_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes);
        uint32_t *wmma_down_f16_low_dev = (uint32_t *)(scratch + wmma_list_base + hot_gate_bytes + hot_down_bytes + f16_low_gate_bytes);
        __half *wmma_mid_h = moe_wmma_hot ? (__half *)(scratch + f16_mid_off) : NULL;
        __half *wmma_down_h = moe_wmma_hot ? (__half *)(scratch + f16_down_off) : NULL;
        __half *wmma_x_h = moe_wmma_hot ? (__half *)(scratch + wmma_x_off) : NULL;
        ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe q2 expert counts clear");
        if (ok) {
            moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                    counts,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert count launch");
        }
        if (ok) {
            moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts, bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert prefix launch");
        }
        if (ok) {
            moe_scatter_sorted_pairs_deterministic_kernel<<<bucket_count, 1u>>>(
                    sorted_pairs,
                    offsets,
                    (const int32_t *)selected_exec->ptr,
                    pair_count,
                    bucket_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert scatter launch");
        }
        if (ok && moe_wmma_hot) {
            const uint64_t xh_count = (uint64_t)n_tokens * expert_in_dim;
            f32_to_f16_kernel<<<(xh_count + 255u) / 256u, 256>>>(wmma_x_h, (const float *)x->ptr, xh_count);
            ok = cuda_ok(cudaGetLastError(), "routed_moe q2 wmma x f16 launch");
        }
        if (!ok) return 0;

        uint32_t wmma_f16_hot_count = 0u, wmma_f16_hot_max = 0u;
        uint32_t wmma_f16_low_count = 0u, wmma_f16_low_max = 0u;
        uint32_t h_counts[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_hot[DS4_ROCM_MAX_N_EXPERT] = {0};
        uint32_t h_f16_low[DS4_ROCM_MAX_N_EXPERT] = {0};
        const uint32_t wmma_hot_threshold = 8u;
        const uint32_t wmma_f16_low_threshold = 64u;
        if (moe_wmma_hot) {
            if (!cuda_ok(cudaMemcpy(h_counts, counts, bucket_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                         "routed_moe q2 wmma counts copy")) return 0;
            for (uint32_t e = 0; e < bucket_count; e++) {
                const uint32_t c = h_counts[e];
                if (c >= wmma_hot_threshold) {
                    if (c < wmma_f16_low_threshold) {
                        h_f16_low[wmma_f16_low_count++] = e;
                        if (c > wmma_f16_low_max) wmma_f16_low_max = c;
                    } else {
                        h_f16_hot[wmma_f16_hot_count++] = e;
                        if (c > wmma_f16_hot_max) wmma_f16_hot_max = c;
                    }
                }
            }
        }
        const uint32_t gate_rpb = 16u;
        const uint32_t down_rpb = 16u;
        const uint32_t gate_threads = gate_rpb * 32u;
        const uint32_t down_threads = down_rpb * 32u;
        const size_t gate_shmem = 4u * 256u * sizeof(float);
        const size_t down_shmem = 4u * 256u * sizeof(float);
        const uint32_t scalar_max = moe_wmma_hot && (wmma_f16_low_count != 0u || wmma_f16_hot_count != 0u)
            ? wmma_hot_threshold : 0u;
        dim3 gate_grid((expert_mid_dim + gate_rpb - 1u) / gate_rpb, bucket_count, 1);
        moe_gate_up_mid_q2K_expert_batch_sharedx_kernel<4><<<gate_grid, gate_threads, gate_shmem>>>(
                (float *)mid->ptr, NULL, gate_w, up_w, (const float *)x->ptr, (const float *)weights->ptr,
                counts, offsets, sorted_pairs, 1u, scalar_max, expert_in_dim, expert_mid_dim,
                gate_expert_bytes, gate_row_bytes, n_expert, clamp);
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert gate/up launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt4 * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_f16_low_dev, wmma_f16_low_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low gate/up launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((expert_mid_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 4u * bk * bn) * sizeof(half) +
                                    (4u * mt * bm * bn) * sizeof(float);
            if (!cuda_ok(cudaMemcpy(wmma_gate_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid hot copy")) return 0;
            moe_gate_up_mid_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_mid_h, gate_w, up_w, (const float *)x->ptr, wmma_x_h, (const float *)weights->ptr,
                    counts, offsets, sorted_pairs, wmma_gate_hot_dev, wmma_f16_hot_count,
                    expert_in_dim, expert_mid_dim, gate_expert_bytes, gate_row_bytes, n_expert, clamp);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid gate/up launch")) return 0;
        }
#endif
        dim3 down_grid((out_dim + down_rpb - 1u) / down_rpb, bucket_count, 1);
        if (moe_wmma_hot) {
            moe_down_q2K_expert_batch_sharedmid_kernel<4,false,true><<<down_grid, down_threads, down_shmem>>>(
                    NULL, wmma_down_h, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        } else {
            moe_down_q2K_expert_batch_sharedmid_kernel<4><<<down_grid, down_threads, down_shmem>>>(
                    (float *)down->ptr, NULL, down_w, (const float *)mid->ptr, NULL,
                    counts, offsets, sorted_pairs, 1u, scalar_max, expert_mid_dim, out_dim,
                    down_expert_bytes, down_row_bytes, n_expert);
        }
        if (!cuda_ok(cudaGetLastError(), "routed_moe q2 expert down launch")) return 0;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        if (moe_wmma_hot && wmma_f16_low_count != 0u) {
            constexpr uint32_t mt4 = 4u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt4, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_low_max + mt4 * bm - 1u) / (mt4 * bm),
                            wmma_f16_low_count);
            const size_t shmem_n2 = (mt4 * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt4 * bm * bn) * sizeof(float) + 2u * bn * 84u;
            if (!cuda_ok(cudaMemcpy(wmma_down_f16_low_dev, h_f16_low,
                                    wmma_f16_low_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-low down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<4,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_f16_low_dev, wmma_f16_low_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-low down launch")) return 0;
        }
        if (moe_wmma_hot && wmma_f16_hot_count != 0u) {
            constexpr uint32_t mt = 8u, bm = 16u, bn = 16u, bk = 16u;
            const dim3 block(32u * mt, 1u, 1u);
            const dim3 grid((out_dim + 2u * bn - 1u) / (2u * bn),
                            (wmma_f16_hot_max + mt * bm - 1u) / (mt * bm),
                            wmma_f16_hot_count);
            const size_t shmem_n2 = (mt * bm * bk + 2u * bk * bn) * sizeof(half) +
                                    (2u * mt * bm * bn) * sizeof(float) + 2u * bn * 84u;
            if (!cuda_ok(cudaMemcpy(wmma_down_hot_dev, h_f16_hot,
                                    wmma_f16_hot_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                         "routed_moe q2 wmma f16-mid down hot copy")) return 0;
            moe_down_q2K_hotlist_wmma_n2_kernel<8,16,16,16,true,true><<<grid, block, shmem_n2>>>(
                    NULL, wmma_down_h, down_w, NULL, wmma_mid_h,
                    counts, offsets, sorted_pairs, wmma_down_hot_dev, wmma_f16_hot_count,
                    expert_mid_dim, out_dim, down_expert_bytes, down_row_bytes, n_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q2 wmma f16-mid down launch")) return 0;
        }
#endif
        const uint64_t n = (uint64_t)n_tokens * out_dim;
        if (moe_wmma_hot) {
            if ((out_dim & 1u) == 0u) {
                const uint64_t n2 = n >> 1u;
                moe_sum_f16x2_kernel<<<(n2 + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            } else {
                moe_sum_f16_kernel<<<(n + 255u) / 256u, 256>>>(
                        (float *)out->ptr, wmma_down_h, out_dim, n_expert, n_tokens);
            }
        } else {
            moe_sum_kernel<<<(n + 255u) / 256u, 256>>>(
                    (float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe q2 expert sum launch");
        if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
        return ok;
    }

    if (q2k_path) {
        uint32_t gate_rows_per_block = cfg->moe_decode_gate_rpb;
        if (gate_rows_per_block == 0u) gate_rows_per_block = 1u;
        const uint32_t gate_threads = gate_rows_per_block * 32u;
        uint32_t down_rows_per_block = cfg->moe_decode_down_rpb;
        if (down_rows_per_block == 0u) down_rows_per_block = 1u;
        const uint32_t down_threads = down_rows_per_block * 32u;
        const int store_gate_up = (g_quality_mode || cfg->graph_dump) ? 1 : 0;
        const int q8k_gateup = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= xq_bytes;
        const int decode_profile =
            n_tokens == 1u && routed_moe_decode_profile_enabled();
        ds4_rocm_moe_decode_profile_record decode_profile_rec = {0};
        if (decode_profile) {
            if (!routed_moe_decode_profile_ensure_events()) return 0;
            g_moe_decode_profile_stats.calls++;
            if (split_selected) g_moe_decode_profile_stats.split_calls++;
        }
        int ok_gateup = 1;
        if (split_selected) {
            const uint32_t compact_mask = stream_resident_mask | stream_missing_mask;
            if (compact_mask == 0u) return 0;
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block,
                           pair_count,
                           1);
            if (stream_resident_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_START,
                            "Q2 decode MoE profile resident gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_resident_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = cuda_ok(cudaGetLastError(),
                                     "routed_moe q2 split resident gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_resident = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_RESIDENT_END,
                            "Q2 decode MoE profile resident gate/up end")) {
                        return 0;
                    }
                }
            }
            if (!ok_gateup) {
                (void)cuda_stream_selected_finish_pending_missing(0);
                return 0;
            }
            const double finish_missing_t0 =
                decode_profile ? cuda_wall_sec() : 0.0;
            ok_gateup = cuda_stream_selected_finish_pending_missing(0);
            if (decode_profile) {
                g_moe_decode_profile_stats.finish_missing_ms +=
                    (cuda_wall_sec() - finish_missing_t0) * 1000.0;
            }
            if (ok_gateup && stream_missing_mask != 0u) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_START,
                            "Q2 decode MoE profile missing gate/up start")) {
                    return 0;
                }
                moe_gate_up_mid_q2K_rows_w32_ptrs_kernel<<<gate_grid, gate_threads>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_slot_ptrs,
                        up_slot_ptrs,
                        NULL,
                        0u,
                        (const float *)x->ptr,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_row_bytes,
                        expert_in_dim,
                        expert_mid_dim,
                        n_expert,
                        stream_missing_mask,
                        clamp,
                        store_gate_up);
                ok_gateup = cuda_ok(cudaGetLastError(),
                                     "routed_moe q2 split missing gate/up launch");
                if (decode_profile && ok_gateup) {
                    decode_profile_rec.gate_missing = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_GATE_MISSING_END,
                            "Q2 decode MoE profile missing gate/up end")) {
                        return 0;
                    }
                }
            }
        } else if (q8k_gateup) {
            if (decode_profile) g_moe_decode_profile_stats.q8_gateup_calls++;
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            cuda_block_q8_K *xq_gate = (cuda_block_q8_K *)down->ptr;
            dim3 xq_grid(xq_blocks, n_tokens, 1);
            q8_K_quantize_kernel<<<xq_grid, 256>>>(xq_gate, (const float *)x->ptr, expert_in_dim, n_tokens);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate input quantize launch");
            if (ok_gateup) {
                dim3 gate_grid((expert_mid_dim + 255u) / 256u, pair_count, 1);
                moe_gate_up_mid_q2K_decode_q8_qwarp32_kernel<<<gate_grid, 256u>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq_gate,
                        (const int32_t *)selected_exec->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        (uint32_t)store_gate_up,
                        clamp);
                ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k gate/up launch");
            }
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else if (gate_rows_per_block == 1u) {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid(expert_mid_dim, pair_count, 1);
            moe_gate_up_mid_q2K_rows_rpb1_w32_kernel<<<gate_grid, 32u>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        } else {
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_START,
                        "Q2 decode MoE profile gate/up start")) {
                return 0;
            }
            dim3 gate_grid((expert_mid_dim + gate_rows_per_block - 1u) / gate_rows_per_block, pair_count, 1);
            moe_gate_up_mid_q2K_rows_w32_kernel<<<gate_grid, gate_threads>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    (const float *)x->ptr,
                    (const int32_t *)selected_exec->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    expert_in_dim,
                    expert_mid_dim,
                    n_expert,
                    clamp,
                    store_gate_up);
            ok_gateup = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows gate/up launch");
            if (decode_profile && ok_gateup) {
                decode_profile_rec.gate_full = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_GATE_FULL_END,
                        "Q2 decode MoE profile gate/up end")) {
                    return 0;
                }
            }
        }
        if (!ok_gateup) return 0;
        int ok_decode_moe = 1;
        const int q8k_down = !g_quality_mode && n_tokens == 1u &&
            down->bytes >= midq_bytes;
        if (decode_profile && q8k_down) {
            g_moe_decode_profile_stats.q8_down_calls++;
        }
        if (q8k_down) {
            cuda_block_q8_K *midq = (cuda_block_q8_K *)down->ptr;
            dim3 midq_grid(midq_blocks, pair_count, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_START,
                        "Q2 decode MoE profile mid quant start")) {
                return 0;
            }
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, pair_count);
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k mid quantize launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.mid_quant = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_MID_QUANT_END,
                        "Q2 decode MoE profile mid quant end")) {
                    return 0;
                }
            }
            if (ok_decode_moe) {
                if (decode_profile &&
                    !routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                            "Q2 decode MoE profile down start")) {
                    return 0;
                }
                if (split_selected) {
                    moe_down_sum6_qwarp32_ptrs_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_slot_ptrs,
                            midq,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                } else {
                    moe_down_sum6_qwarp32_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected_exec->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim,
                            n_expert);
                }
                ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip q8k down launch");
                if (decode_profile && ok_decode_moe) {
                    decode_profile_rec.down = 1;
                    if (!routed_moe_decode_profile_record_event(
                            DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                            "Q2 decode MoE profile down end")) {
                        return 0;
                    }
                }
            }
        } else {
            dim3 down_grid((out_dim + down_rows_per_block - 1u) / down_rows_per_block, n_tokens, 1);
            if (decode_profile &&
                !routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_START,
                        "Q2 decode MoE profile down start")) {
                return 0;
            }
            if (split_selected) {
                moe_down_q2K_sum_rows_w32_ptrs_batch_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_slot_ptrs,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_row_bytes,
                        n_expert);
            } else {
                moe_down_q2K_sum_rows_w32_kernel<<<down_grid, down_threads>>>(
                        (float *)out->ptr,
                        down_w,
                        (const float *)mid->ptr,
                        (const int32_t *)selected_exec->ptr,
                        n_tokens,
                        expert_mid_dim,
                        out_dim,
                        down_expert_bytes,
                        down_row_bytes,
                        n_expert);
            }
            ok_decode_moe = cuda_ok(cudaGetLastError(), "routed_moe q2 oldhip rows down launch");
            if (decode_profile && ok_decode_moe) {
                decode_profile_rec.down = 1;
                if (!routed_moe_decode_profile_record_event(
                        DS4_ROCM_MOE_DECODE_PROFILE_DOWN_END,
                        "Q2 decode MoE profile down end")) {
                    return 0;
                }
            }
        }
        if (ok_decode_moe && compact_selected) {
            ok_decode_moe = cuda_stream_selected_mark_inflight();
        }
        if (ok_decode_moe && decode_profile) {
            ok_decode_moe =
                routed_moe_decode_profile_collect(&decode_profile_rec);
        }
        return ok_decode_moe;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, pair_count, 1);
        if (q2k_path) {
            moe_gate_up_mid_q2K_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        } else {
            moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
                (float *)gate->ptr,
                (float *)up->ptr,
                (float *)mid->ptr,
                gate_w,
                up_w,
                (const float *)x->ptr,
                (const int32_t *)selected_exec->ptr,
                (const float *)weights->ptr,
                gate_expert_bytes,
                gate_row_bytes,
                expert_in_dim,
                expert_mid_dim,
                n_expert,
                clamp);
        }
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, pair_count, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected_exec->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    if (ok && compact_selected) ok = cuda_stream_selected_mark_inflight();
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in, uint32_t layer_index, bool force_resident) {
    if (add_in) {
        fprintf(stderr, "ds4: routed MoE addend fold is Metal-only\n");
        return 0;
    }
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, 1,
                             force_resident);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16, bool force_resident) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, n_tokens,
                             force_resident);
}

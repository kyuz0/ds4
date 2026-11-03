/* In-process aligned-SoA repack for one MoE layer, for the ROCm build.
 * Uses the shared ds4_repack builders (same layout the CUDA self-load path and
 * the weight server emit), so no weight bytes are invented here. */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../cuda/mmq/ds4_repack.h"

extern "C" int ds4_rocm_repack_layer(const char *model_path, int layer,
                                    const void **out_gate, const void **out_up,
                                    const void **out_down) {
    if (!model_path || layer < 0 || !out_gate || !out_up || !out_down) return 0;
    ds4_repack_file map;
    if (!ds4_repack_map_file("ds4", model_path, map)) return 0;
    std::vector<ds4_repack_tensor> all;
    if (!ds4_repack_collect_catalog("ds4", map, nullptr, &all)) {
        ds4_repack_unmap_file(map);
        return 0;
    }
    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
    std::vector<ds4_repack_tensor> records;
    for (const ds4_repack_tensor &t : all)
        if (strncmp(t.name.c_str(), prefix, strlen(prefix)) == 0) records.push_back(t);
    ds4_repack_unmap_file(map);
    if (records.empty()) return 0;

    ds4_repack_build_args args;
    args.log_prefix = "ds4";
    args.model_id = "base";
    args.path = model_path;
    args.records = &records;
    args.device = 0;
    args.copy_chunk_bytes = 256ull * 1048576ull;

    static std::vector<ds4_repack_artifact> artifacts;
    artifacts.clear();
    uint64_t part = 0;
    if (!ds4_repack_build_iq2_aligned(args, artifacts, &part)) return 0;
    if (!ds4_repack_build_q2k_aligned(args, artifacts, &part)) return 0;

    const void *g = nullptr, *u = nullptr, *d = nullptr;
    for (const ds4_repack_artifact &a : artifacts) {
        const char *n = a.t ? a.t->name.c_str() : "";
        if (a.kind == DS4_REPACK_IQ2_XXS_ALIGNED_MOE) {
            if (strstr(n, "ffn_gate_exps")) g = a.dev;
            else if (strstr(n, "ffn_up_exps")) u = a.dev;
        } else if (a.kind == DS4_REPACK_Q2_K_ALIGNED_MOE) {
            if (strstr(n, "ffn_down_exps")) d = a.dev;
        }
    }
    if (!g || !u || !d) return 0;
    *out_gate = g; *out_up = u; *out_down = d;
    fprintf(stderr, "ds4: V4.1 SoA repack layer %d resident (gate/up/down)\n", layer);
    return 1;
}

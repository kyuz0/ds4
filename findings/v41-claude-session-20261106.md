# V4.1 Strix Halo — session 2026-11-06 (ctx 128K, cache 92GB)

All runs: `ds4-bench --ctx-alloc 131072 --ssd-streaming --ssd-streaming-cache-experts 92GB`,
prompt `speed-bench/promessi_sposi.txt`, generated text compared with `--show-output`,
frontier logits gated with `ak gate logits`.

## 1. Fused int8-WMMA indexer scorer — `DS4_ROCM_V41_INDEX_WMMA=1` (prefill)

Kernel trace of a 64K prefill: in the last quarter (positions 48K–64K) the wide indexer
head GEMM (`cublasSgemmStridedBatched`, Tensile FP32 MT32x32x8) is 25.5 % of time and its
head reduction 4.8 %; both grow linearly with context. The GEMM writes a
32 heads × 32 rows × n_comp F32 scratch per call and re-reads it.

`rocm/ds4_rocm_v41_ix_wmma.cuh` replaces GEMM + reduction (and the warp8 batch scorer for
batches of >= 16 rows) with one kernel:
- queries and keys are FP4(E2M1) × pow2 per 32-block; dividing each block by
  pow2_ceil(amax/6) makes 2·x an integer in [-12, 12] → exact int8 operands;
- `v_wmma_i32_16x16x16_iu8` gives exact per-block integer dots; the four blocks are combined
  exactly in int64 (double fallback for exponent spread > 38) and rounded once, so each dot
  equals the correctly rounded exact dot — which is what the FP32 GEMM produces (checked:
  `gemm!=ref=0`, `inexact_dot=0` in every sampled call);
- head reduction keeps the reference order and the non-contracted mul/add;
- `=1` reads the 68-byte packed FP4 keys, `=3` the F32 keys. Block = 128 keys × 32 rows.

Pitfall found: an FP16-WMMA version was *not* exact. RDNA3.5 FP16 WMMA accumulation drops
low bits for mixed-sign sums (unit test: ±0.25 errors on 23 % of outputs, exact for
non-negative inputs). It passed the 32K gate by luck and failed at 64K (top-k moved,
text diverged). Integer WMMA has no such issue. Unit test: scratchpad `u.cu` (synthetic
FP4 data, spreads 0/3/8, packed and F32 keys: 0 mismatches).

Diagnostic: `DS4_ROCM_V41_INDEX_WMMA=2` runs GEMM path + fused kernel + a double
reference on every 97th wide call and prints `ixcheck` counters.

Results:
- F32-key, 16-row tiles, 64K, 2 ABBA pairs: prefill 171.50/170.66 → 188.81/188.68 tps
  (+10.3 %), logits bit-exact, text identical.
- Current version (queries pre-normalized once to int8 + exponents by a prep kernel; no
  per-head LDS staging/barriers; 128 keys × 32 rows per block; packed keys; int32 fast path
  for the exact combine), 120K frontier, all flags vs stock: prefill **142.72 → 182.46 tps
  (+28 %)**, logits bit-exact, 32 generated tokens identical. (Single runs, same session.)
- Previous version cost 6.0 ms per 32-row call at 120K (~5 TOPS): the per-score double
  combine and per-block re-normalization of all queries dominated.

## 2. Decode drain / shared-expert reordering — `DS4_ROCM_V41_STREAM_QUEUE=1`, `DS4_ROCM_V41_SHARED_FIRST=1`

- STREAM_QUEUE: with SSD streaming every decode layer ended in `cudaDeviceSynchronize`.
  The blocking selected-id readback already orders all earlier work before any expert-cache
  eviction/upload, so only the Engram (layer 13) and final drains are kept.
- SHARED_FIRST: the routing-independent shared expert is enqueued before the selected-id
  readback (now an async pinned copy + event), so the GPU computes it while the host waits.
- Bit-exact, identical text (14 runs). Gains are small: 8K +2.5 % (STREAM_QUEUE alone,
  4 ABBA), 32K +0.9 % (both, 3 ABBA, all pairs >= 0). The 11.6 ms "host_drain" was mostly
  overlapped GPU work, as AGENTS.md warns.

## 3. Decode budget (4K ctx trace, both flags on)

79.4 ms/token = 64.4 ms GPU busy + 15.0 ms idle (profiler-inflated). Decode reads
10.5 GB/token (`ak facts decode-bytes`), floor 47 ms at 223 GB/s. The big GEMVs already run
at 210–220 GB/s (Q8 blocks4 4.85 GB in 22.2 ms; MoE gate/up 1.46 GB in 6.9 ms; down 0.93 GB
in 4.2 ms). Remaining: ~2245 small inter-kernel gaps (~5 ms), 668 `v41_bf16` launches,
250 D2D copies per token, SSD misses (0.7 % of lookups at 8K, no evictions).

## 4. Packed FP4 key cache was not restored with session payloads

`index_cache_fp4` is derived state but is not in `ds41_state_spans`, so a payload restore
left it stale for `DS4_ROCM_V41_INDEX_FP4=1` decode. Fixed: the payload load rebuilds it
from the restored F32 caches (`ds4_gpu_dsv41_index_fp4_pack`).

## 5. Cache size

With ctx 131072 the planner accepts 96 GB (9588 experts vs 9156 at 92 GB) and refuses
100 GB (needs 123.01 GiB of 121.36). Not measured yet.

## 6. Ideas from other engines (survey of ds4-argodrive, lucebox, DeepInfer)

- Mailbox for router ids (argodrive +6.6 % on Metal): our SHARED_FIRST async pinned
  readback is the HIP analogue; measured gain here ~1 %. Never spin a GPU wave on host
  memory (lucebox: MES timeout / GPU reset with cwsr off).
- Decoder bounded replay / suffix for prefill (DeepInfer, lucebox): already in this tree as
  opt-in `DS4_METAL_V41_DECODER_SUFFIX=1`; not bit-exact (findings/codex-suffix-decision.md).
- Next-layer expert prediction: argodrive measured it net-negative (-1 % … -8 %); low value
  at our 0.7 % miss rate.
- hipGraph replay of decode segments: lucebox measured a captured graph 11 % slower than
  eager on gfx1151.

## 7. Decoder suffix (`DS4_METAL_V41_DECODER_SUFFIX=1`, existing, not bit-exact)

On top of the exact flags:

| case | prefill off → on | logits | 128-token greedy text |
|---|---|---|---|
| 24K prose | 202.5 → 311.4 | max_abs 0.80, argmax =, top-k set = | diverges at char 170/354 |
| 24K code (ds4.c) | 196.6 → 312.5 | max_abs 0.36, argmax =, top-k set = | diverges at char 276/494 |
| 24K docs (README+docs) | 188.8 → 307.5 | max_abs 0.40, argmax =, top-k set = | identical |
| 120K prose | 182.5 → 380.1 | max_abs 0.67, argmax = | 32 tokens identical |

Decode is unaffected beyond noise. An apparent ~12 % decode drop in one batch hit both arms
(machine slow phase; clocks float 1.0–2.3 GHz under `auto`): rerun all flags 11.39 vs stock
11.24 tps at 24K / 128 tokens.

## Defaults (promoted 2026-11-06 on user directive)

ROCm defaults now (each disabled with `=0`): `DS4_ROCM_V41_INDEX_WMMA`,
`DS4_ROCM_V41_INDEX_FP4`, `DS4_ROCM_V41_STREAM_QUEUE`, `DS4_ROCM_V41_SHARED_FIRST`.
Non-ROCm backends are unchanged. Default-binary check, 32K gen 64: logits bit-exact vs
stock, text identical, prefill 200.8 tps.

Decoder suffix, now also ROCm default (`DS4_METAL_V41_DECODER_SUFFIX=0` disables):
- Plain suffix cost -11 % decode: layers 20-39 prefill only a shrinking tail (layer 39: 1
  row), so the post-prefill expert-cache seed, which ranks experts from the last chunk's
  routes, left 778 slots empty; decode misses 557 vs 321 over 128 tokens.
- Fix `DS4_ROCM_V41_SUFFIX_MIN_ROWS` (default 4096): every suffix layer computes at least
  its last 4096 rows. Misses 284 (below the full decoder's 321), cache full.
- 24K prose gen 128, ABBA x2: prefill 199 → 308 tps (+54 %), decode 10.86 → 11.26 tps.
- Long-prefix perplexity (16384-token prefix, 512 scored tokens; new `ds4
  --perplexity-prefix N`), off → on: prose 4.0458 → 4.0224, code 1.1247 → 1.1250,
  docs 2.7488 → 2.7542. Mixed sign, rounding level.

## 8. Decode round 2 (all ROCm default, bit-exact)

Trace (8K, 48 tokens) before: 92.9 ms/token profiled, 27.8 ms GPU idle, 2370 kernels/token.
After: 82.0 ms, 17.1 ms idle, 2170 kernels/token.

| change (env, =0 disables) | what | measured |
|---|---|---|
| `DS4_ROCM_V41_ENGRAM_ASYNC` | 2×24 Engram disk rows per token read concurrently (16-thread pool), awaited at layer 1 instead of serially before the token | with the two below: 24K gen128 ABBA×3 11.39 → 11.74 tps (+3.1 %) |
| `DS4_ROCM_V41_Q8_DECODE_ROUND` | decode Q8 GEMV rounds to bf16 in its epilogue (same `v41_bf16`), ~80 fewer launches/token | (same A/B) |
| `DS4_ROCM_V41_PTRS_ZERO_COPY` | expert pointer table in coherent mapped host memory: no 3×H2D + event per layer (was a ~53 µs copy stall, 2.1 ms/token) | (same A/B) |
| `DS4_ROCM_NORM_BATCHED` | RMS norm loads batched 8 ahead, same accumulation order (unit test bit-identical; n=20480 16.6 → 4.7 µs) | ABBA×3 11.68 → 11.85 tps (+1.5 %) |

Not worth it / rejected: 96 GB cache (misses 827 → 765 per 512 tokens, decode unchanged);
pipelining `v41_hc_ordered_prefetch_kernel` (no gain, FMA-chain/occupancy bound);
n-gram speculative decode (expert-union bytes and misses make it ~break-even under SSD
streaming).

Remaining decode budget (24K, ~81 ms/token): ~53 ms bandwidth-bound GEMVs at 210–220 GB/s,
~10 ms small kernels, ~5 ms launch gaps, ~4 ms expert misses (1.6/token).

# V4.1 Strix Halo — session 2026-11-06 (ctx 128K, cache 92GB)

## Result: session-start binary vs final defaults (cdd3773)

| case | before | after |
|---|---|---|
| 24K prefill (ABBA×2) | 181.6 tps | 322.3 tps (+77 %) |
| 24K decode, gen 128 | 10.85 tps | 12.21 tps (+12.5 %) |
| 120K prefill (1 pair) | 137.9 tps | 370.8 tps (2.7×) |
| 120K decode, gen 64 | 11.11 tps | 11.40 tps (1 pair; ABBA×2 earlier: 11.90) |

All changes are ROCm defaults with `=0` escape hatches. Everything is bit-exact except the
decoder suffix (rounding-level; `DS4_METAL_V41_DECODER_SUFFIX=0` restores bit-identity
with the stock engine). Raw runs: findings/v41-decode2/final-before-after.txt.

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

| `DS4_ROCM_V41_NORM_FUSE` | one-launch [bf16(x)]→norm→bf16(out); hc_expand + bf16 store (+ 16-byte `pre` copy); q_a/kv rounding folded into the norms; grouped output_a GEMV rounds in its epilogue. ~490 fewer launches/token | ABBA×3 11.90 → 12.05 tps (+1.3 %) |

| `DS4_ROCM_V41_INDEX_DECODE_WMMA` | decode indexer scorer on int8 WMMA (32 heads as M, packed keys, exact combine, reference head order); was 5.4 ms/token at 120K. Check mode `DS4_ROCM_V41_INDEX_DECODE_CHECK=1`: 0 mismatches | 120K gen64 ABBA×2 11.25 → 11.90 tps (+5.8 %) |

Not worth it / rejected: 96 GB cache (misses 827 → 765 per 512 tokens, decode unchanged);
pipelining `v41_hc_ordered_prefetch_kernel` (no gain, FMA-chain/occupancy bound);
n-gram speculative decode (expert-union bytes and misses make it ~break-even under SSD
streaming).

Remaining decode budget (24K, ~81 ms/token): ~53 ms bandwidth-bound GEMVs at 210–220 GB/s,
~10 ms small kernels, ~5 ms launch gaps, ~4 ms expert misses (1.6/token).

## 9. Cold prefill is SSD-bound: cap layer-load read concurrency

`DS4_ROCM_STREAM_READ_PROFILE=1`: a cold 24K prefill reads 145 GiB (every layer's 384
experts; the cache is empty in a fresh process, so layer reuse has 0 hits) and the main
thread waits 37 s of 79 s. The NVMe (KIOXIA EXCERIA BASIC, DRAM-less) does 4.9 GB/s with
8 MB sequential reads but the 16-worker pool got ~2 GB/s.

Workers sweep, cold 8K prefill (tps): 1: 214, 2: 228, 3: 229, 4: 231, 8: 220, 16: 202, 18: 197.
Four workers globally cost decode 2.6 % (misses lose parallelism), so
`DS4_ROCM_STREAM_LAYER_READ_WORKERS` (default 4) caps only large batches (prefill layer
loads, > 18 jobs); decode miss batches keep all 16 workers.
8K gen 256 ABBA×3: prefill 205.9 → 227.5 tps (+10.5 %), decode 12.43 → 12.35 tps (−0.6 %,
noise-level). A long-running server reuses cached experts during prefill and reads less.

Rejected: spin-waiting the decode readback (`DS4_ROCM_ASYNC_READ_SPIN=1`, neutral at 24K
and 120K).

## 10. DSpark + SSD streaming (opt-in, in progress)

Drafter: `kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF` @22a073e, `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf`,
SHA-256 7a2217ca…d4b5 (matches docs/STRIX_HALO.md), in /home/dani/models/deepseek-v4.1/.
Needs an 86 GB expert cache (92 GB exceeds the planner with the 7.4 GiB drafter).

Enable: `DS4_ROCM_V41_DSPARK_SSD=1 DS4_ROCM_V41_DSPARK_SSD_SYNC_VERIFY=1 ds4-bench ... --mtp-model DRAFTER --dspark`
(opens the three SSD gates: engine open, --mtp-model check, ds41_draft_init).

Code prompt (ds4.c), 8K, 128 greedy tokens: acceptance 66 %, ~2.9-3.0 draft tokens accepted per
cycle; generated text identical to plain decode in every arm.

| arm | decode tps |
|---|---|
| plain decode | 11.84 |
| DSpark scalar verify | 7.05 (verifies rows one by one) |
| DSpark batched verify, async selective stage | 3.39 (stage path re-reads experts) |
| batched verify, synchronous selected prepare | 11.45 |
| + unique-expert gate/up kernel (decode each IQ2 block once, integer dot per row) | **12.70** |

Verify cycle (6 rows) ~290 ms: gate/up ~52, down ~48 (per-(row,expert), rounding-level fix possible),
SSD-miss stalls ~52 (staged reads + H2D copies, ~23 experts/layer touched), dense 6-row GEMVs ~24,
attention ~17. Target for ~20 tok/s on code: ~160 ms per cycle.

### 10b. The SSD is the DSpark bottleneck (and ~12 % of plain decode)

Random O_DIRECT reads at random offsets of the model file (tools/ssd-random-read.c), KIOXIA
EXCERIA BASIC (DRAM-less): 512 KB-3 MB requests, 4-32 threads -> 1.03-1.61 GB/s, never more
(4.9 GB/s only for sequential reads). A missed expert (~9.3 MB) therefore costs ~6 ms whatever
the software does. Decode-only accounting (DSpark, code, 128 tokens): 537 misses, main-thread
read wait 4.0 s = ~40 % of decode. Plain decode: ~1.6 misses/token ~ 10 ms of ~82 ms.
Neither read chunking for verify misses nor removing the read cap changed this.

DSpark draft width (DS4_DSPARK_FIXED_WIDTH, adaptive off), code 8K gen128:
2: 12.20 tps (441 misses), 3: 13.43 (422), 4: 12.89 (531), 5: ~12.5 (537). Plain: 11.84.
Best: width 3, +13 % on code with this drive. A DRAM-equipped NVMe that sustains
5+ GB/s on large random reads would cut miss cost ~3-4x for both plain decode and DSpark.

## 11. Agentic coding (Pi replay, 2026-11-07)

Replay: the rendered prompt of a real 15-turn Pi session (~40K tokens), sliced at the
real request boundaries (`--ctx-frontiers 1878,...,40304`, 120 generated tokens each,
92GB cache, 128K ctx).

- Tool-calling failures seen in Pi were a client config problem (`models.json` without
  `thinkingFormat: deepseek` / reasoning-content compat). The engine output was bit-identical to
  the session-start binary at 512-2048 tokens.
- Appends and prefill tails under 256 tokens run token-by-token (~80-140 ms/token): a
  222-token append took 30.7 s. Sending them through the layer sweep instead
  (`DS4_ROCM_V41_PREFILL_MIN=32`) cut that to ~10 s but failed the official scorer
  (NLL +17 %, 0/9 wins). Rejected (H009).
- **Token-by-token and sweep prefill are different computations, before this session too.**
  At 600 tokens (prose) the two paths give different frontier logits and different greedy
  text; the current build is bit-identical to the session-start build on *each* path
  (`ak gate logits` pass/pass), so this is pre-existing engine behaviour, not a
  regression. The official scorer prefers the token-by-token tail (0.6395 vs 0.7502).
  F32 quality kernels move the sweep logits only slightly (max |d| 1.44 vs 14), so it is
  not rounding. A CPU arbiter is unavailable (V4.1 needs Metal/CUDA/ROCm). Open question
  for upstream.

### 11b. Expert-cache policy for agent sessions (new defaults)

Same replay, two ABBA rounds, greedy text identical in every arm (cache policy only):

| arm | total prefill | decode tok/s | decode misses |
|---|---|---|---|
| append seed 100 % (old default) | 372.9 / 376.1 s | 10.66 / 10.73 | 10414 |
| append seed 75 % | 346.7 / 347.1 / 349.1 s | 10.58 / 10.58 / 10.61 | 10197 |
| 75 % + grow cache after prefill | 336.9 / 340.1 s | 11.01 / 11.03 | 8632 / 8683 |

Both are now default on ROCm (`DS4_ROCM_V41_APPEND_SEED_PCT`, default 75;
`DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL=0` disables the growth). Combined vs the old
defaults: agent prefill -9 %, decode +3.5 %, decode SSD misses -17 %.

### 11c. Protecting the decode working set during prefill: rejected (H010)

Opt-in experiment (patch kept out of tree): while a prompt is being prefilled, LRU eviction
first skipped experts that decode used within its last N tokens. Agent replay, 92GB, 128K:

| N decode tokens protected | decode tok/s | decode misses | total prefill |
|---|---|---|---|
| 0 (current default) | 11.06 | 8624 | 331.9 s |
| 256 | 10.70 | 9435 | 342.1 s |
| 1024 | 9.58 | 14691 | 350.4 s |

Decode right after an append needs the experts the *new* content activated (as H007 showed
for the post-prefill seed); keeping the previous decode set evicts exactly those. The slow
first ~50-150 decode tokens after a tool result are new-content misses, not lost old ones.

Also from the live session: the disk KV cache never checkpointed long chats (exact-multiple
rule vs one-pass V4.1 appends), fixed in 5b79d52 / 9fba89a; resumes now load from disk.

### 11d. DSpark on the agent replay: no gain (stays opt-in)

Width 3, SSD verify prefetch, 86GB cache (drafter memory) vs plain 92GB: decode 10.43 vs
11.06 tok/s (-6 %), total prefill 357.8 vs 331.9 s (+8 %), greedy text identical,
accept rate 64.6 % (1.65 drafts accepted per cycle). Agent turns are short decodes right
after new content, so the smaller cache and verify overhead outweigh the accepted drafts.

## 12. Batched rows counted the shared expert twice (bug from 41bb7cf, fixed)

Symptom: every batched path (layer-major prefill sweep, short batches, DSpark verify)
disagreed with token-by-token decode by far more than rounding, and scored worse on the
official scorer (continued-prefill 100: per-token tail 0.6395 vs sweep tail 0.7502).

Method: `DS4_V41_TRACE_POS=P DS4_V41_TRACE_DIR=D` dumps, for the token at position P,
per-layer vectors from the single-row step and the batched row step (residual, after-attn,
MoE input/selection/weights/shared/routed, Engram rows, q/kv/heads, attention block).
8192-token prefix (identical in both runs), then a 6-token append token-by-token vs one batch.
- Layer 0: attention half identical (0.0); MoE half rel. 0.28. Same experts and weights,
  shared expert equal; "routed" differed by rel 2.69.
- CPU reference of the layer-0 routed experts from the GGUF weights (IQ2_XXS gate/up,
  Q2_K down, clamp 10): the single-row output matches (rel 1.8e-2); the batched one is
  exactly REF + SHARED (least squares 1.000 / 1.000, residual 1.5e-3).
- Cause: `ds41_moe_batch` must fold the shared expert into the routed partial only on the
  TP rank that owns it. 41bb7cf rewrote `(!shared_owner || rank != owner || add)` as
  `if (!shared_owner || rank != owner) add` (inverted), and single-GPU callers add
  routed + shared again. Every batched row on one GPU counted the shared expert twice in
  all 40 layers. Single-row decode (ds41_moe) was correct, so generation looked fine.
- After the fix, layer outputs match to rounding (L0 2.6e-3, attention heads 2.7e-3 at L1);
  the remaining growth tracks MoE selection flips at near-tie gates (top-6 weights
  0.19-0.24), not further defects.

Official scorer, 20260911-long manifest, 92GB SSD streaming (`api_target_mae` = error vs the
API reference log-probs stored in the manifest's continuations/):

| arm | avg NLL | api_target_mae |
|---|---|---|
| default prefill, before fix | 0.6956 | 0.493 |
| default prefill, after fix | **0.6486** (-6.8 %, 5/9 wins) | **0.441** |
| cont.-prefill 100, token-by-token tail, after fix | 0.6471 | 0.448 |
| cont.-prefill 100, 6-row short prefill (mode 2), after fix | 0.6531 (+0.9 % vs per-token, 4/9) | 0.445 |

Batched and token-by-token paths now agree; short prefill mode 2 is in band and became the
ROCm default (`DS4_ROCM_V41_SHORT_PREFILL=0` restores token-by-token, `1` = 8-row generic).

Agent replay after the fix: total prefill 331.9 -> 326.5 s (short appends 222: 26.7 -> 19.4 s,
332: 10.2 -> 8.0 s, 92: 6.9 -> 5.4 s), decode 11.06 -> 8.86 tok/s. The decode change is a
workload change, not a regression: the replay continues raw slices of a recorded session,
and before the fix the model mostly echoed the tool output it was reading ("$1.class",
"| 1 | 2 | 3 ..."), which reuses few experts; after the fix it writes agent reasoning and
tool calls ("Need inspect src. Also read docs/development/CONTEXT.md ..."), with ~1.5x the
decode misses. With token-by-token tails the fixed build decodes the same (8.94 tok/s), and
the single-row decode code is unchanged by the fix. Decode baselines on this replay must
be re-taken after this commit.

### 12b. DSpark re-measured after the fix: still a loss

Agent replay, fixed build: DSpark width 3 (86GB) decode 7.48 vs plain 8.86 tok/s (-16 %),
prefill 372.6 vs 326.5 s, accept rate 58.1 % (1.33 drafts/cycle). With correct prefill the
model writes varied reasoning, which the drafter predicts worse. DSpark stays opt-in.

## 13. Medium appends in one layer sweep (ROCm default)

Profile of a 717-token append (DS4_METAL_GRAPH_PREFILL_PROFILE): 14.3 s = layer-load wait 8.3 s
(58 %), compute 5.2 s (36 %), post-prefill seed 0.75 s. 717 rows touch all 384 experts per layer,
so the uncached ~40 % must stream from the SSD each sweep. ds41_prefill_count split appends
over 2048 tokens into several sweeps (3072 = 2048 + 1024; 5171 = 4096 + 1075), each paying a
full layer-streaming pass. Now 2048 < n < 8192 runs as one sweep with the same 2048-row
partitions (`DS4_ROCM_V41_DISABLE_MEDIUM_SWEEP=1` restores the split).

8192-token prefix + appends: 3072 29.0 -> 18.0 s, 5171 35.8 -> 28.7 s (both bit-identical
frontier logits), 2134 22.1 -> 15.1 s (86-token remainder now in the sweep instead of 6-row
batches: KL 6e-4, same top-1; both tail paths passed the official scorer, section 12).
Agent replay: total prefill 326.5 -> 306.3 s (-6.2 %), decode 8.86 -> 8.95 tok/s.

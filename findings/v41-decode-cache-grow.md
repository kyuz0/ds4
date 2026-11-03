# V4.1 decode cache: reclaim the prefill reserve after prefill (promoted)

**Change (in tree, default-on with opt-out):** `ds4.c`
`ds41_stream_grow_cache_after_prefill()` is called once from
`ds4_session_eval()` before the first decode token. It raises the streaming
expert-cache budget from `ssd_streaming_cache_bytes` to
`ssd_streaming_cache_bytes + ssd_streaming_prefill_headroom_bytes`, i.e. it hands
the prefill expert reserve (7.12 GiB) to decode once prefill is finished. This
mirrors the existing GLM path (`DS4_ROCM_GLM_STREAMING_GROW_CACHE_AFTER_PREFILL`).

Opt out with `DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL=0`.

**Why it is memory-safe:** no memory is pre-allocated or re-reserved. The
streaming slab grows lazily in `cuda_stream_expert_slab_grow()`, which checks
`cudaMemGetInfo()` against `cuda_stream_resident_free_reserve_bytes()` and simply
stops when memory is short. Worst case is "no growth". Nothing else in the plan
changes: cache 84.87 -> 91.99 GiB (9156 -> 9924 experts), 768 extra experts.

**Why it cannot change numerics:** only cache residency changes; the same
kernels run on the same weight bytes. Verified: frontier logits are
**bit-identical** (`exact=True, max_abs=0`) at 32768 and 36864, and at 110592 and
114688, in every paired run.

## Measurements (balanced, 92GB cache, 128K-capable ctx alloc, suffix off)

32K base + 4K append, alternating order, gen 32:

| run | grow | 32K prefill tps | 4K append ms | decode tps |
|---|---|---|---|---|
| k_off1 | off | 161.30 | 27 734 | 10.73 |
| k_on1  | on  | 173.52 | 27 765 | 12.21 |
| k_off2 | off | 174.67 | 27 070 | 11.45 |
| k_on2  | on  | 173.33 | 27 457 | 12.32 |
| default-check (no env) | on | 175.76 | 27 470 | 11.68 |

Decode: off range 10.73-11.45, on range 11.52-12.32 -> consistent gain of roughly
**+5 to +10 %**, with all on-runs at or above the best off-run. Prefill and
append latency: unchanged within the measured noise band (~3-7 %).

110K base + 4K append: decode 10.30 -> 10.47 tps (+1.7 %, inside noise, not
negative), append 39.90 s vs 41.44 s (noise). At 110K the attention work per
token is much larger, so the same +768 experts are a smaller share of token time.
The change is neutral there and strictly better at 32K-64K contexts.

## Caveats / what was ruled out

- The cache cannot grow further: at 104 GB the planner refuses
  (`needs 126.92 GiB ... only 121.73 GiB available`). 121.73 GiB is the real
  ceiling with the 110K KV/graph plan, so 92 GB + reclaim is the end of this lever.
- `DS4_ROCM_V41_SELECTED_READ_CHUNKS` is inert above `18/job_count`
  (both `8` and `16` clamp to the same value), so earlier +7 % readings from that
  knob were noise. Measured run-to-run noise band on this workload: ~3-7 %.

Artifacts: `bench-grow-cache-ab.sh`, `bench-grow-cache-default-check.sh`,
`findings/bench-grow-cache-ab-trial1-20261102-*`,
`findings/bench-grow-cache-default-trial1-20261102-*`,
`findings/bench-append-110k-tuning-trial1-20261102-124754` (memory-wall evidence).

## Follow-up attempt: grow into the remaining free memory (rejected)

The lazy slab is free-memory guarded, so the post-prefill budget was raised to
`cache + headroom + 25%` on the theory that the ~13 GiB left unallocated after
the plan is better spent on cache. It grew to **12405 experts (114.99 GiB)**
versus 9924 (91.99 GiB), and decode went 12.10 -> 12.39 tps mean
(**+2.4 %**, inside the 3-7 % noise band; per-run ranges overlap:
11.91/12.29 vs 12.32/12.46). Prefill unchanged, frontier logits still
bit-identical (max_abs 0).

Verdict: not a verified win, and it would consume ~23 GiB of RAM headroom that
other workloads may need. Reverted; the committed behaviour stays at the
conservative 92 GiB (headroom-only) growth.

Consequence: decode is capacity-bound with **diminishing returns** past ~92 GiB.
The next credible decode lever is not more cache but fewer exposed misses -
i.e. prefetching the next layer's likely experts while the current layer
computes (exactness-neutral: prefetch only changes residency). Run:
`findings/bench-grow-cache-extra-trial1-20261102-142055`.

## Decode stage profile (8K, DS4_V41_DECODE_STAGE_PROFILE)

Profiling drains at every stage boundary, so absolute values are inflated
(~2x); shares are informative. 8 tokens x 40 layers, 6704 samples:

  attention 24% | moe_partial 21% | attn_output 12% | moe_routed 10%
  attn_project 6% | moe_stream_begin 6% | attn_q_b 4% | attn_decode 4% | moe_shared 4%

Aggregate: attention path ~57%, MoE path ~42%. Expert-stream begin (exposed
SSD wait) is only ~6% of profiled token time, and the cache is saturated
(+2.4% for +23 GiB), so neither more cache nor prefetch is the lever: decode is
compute/latency bound in the attention path and the MoE matvecs. Cases matched.

## Decode tuning sweep (32K base + 4K append, gen 32, A/B/A/B style)

All toggles below are residency/scheduling only; every arm kept frontier logits
bit-identical (max_abs 0), so the differences are pure timing.

| arm | decode vs base | prefill 32K tps | append ms |
|---|---|---|---|
| gate/down waves 8 | -1.3% | 176.3 | 27 040 |
| gate/down waves 2 | -0.3% | 176.6 | 27 115 |
| waves 8 + down preload6 | -0.2% | 176.1 | 26 482 |
| disable all-resident split | **-16..-20%** | 175.6 | 27 426 |
| partial gate/up join (after fix) | +0.9% / -6.6% (noise) | 175.9 | 27 171 |
| disable attn low-A m256 | +5.3% (prefill -1.4%) | 172.8 | 28 154 |
| disable attn out pad8 | +3.7% (prefill -5.0%) | 166.6 | 27 834 |

Conclusions: the shipped defaults (waves 4, split on, both attention tiles on)
are already the best combination for prefill+decode together; no knob moves
decode beyond the 3-7% noise band without costing prefill. Remaining decode
upside lives in the kernels themselves (attention path ~57% of token time,
MoE matvecs ~42% at ~15-25% of DRAM bandwidth), not in configuration.

Also fixed: `cuda_stream_selected_finish_pending_gateup()` aborted whenever a
different read batch owned the workers (the normal case at decode, since the
full-layer loader is active). It now waits on its own jobs, so the opt-in
partial gate/up join works instead of failing at layer 1. Measured neutral, so
it stays opt-in; with the abort removed it can be revisited once the down read
path changes.

## Q2 decode matvec ILP variant (rejected)

dev_dot_q2_16 split into four independent integer accumulators (bit-identical by
integer associativity; verified: frontier logits exact=True, max_abs 0 at 32768
and 36864 in both runs). Decode 11.91 -> 11.51 tps means (-3.4%, inside the 3-7%
noise band). Rejected and reverted.

Resolution argument: the Q2 down matvec moves 3.87 of the 9.95 MB per expert, so
its share of an 82 ms token is ~11 ms; even a 10% gain there is 1.3% of token
time, i.e. below measurement resolution. Micro-ILP inside these matvecs cannot
produce a promotable decode win. Only structural changes (per-expert prefill
GEMM tiling, decode attention fusion) are measurable at this signal-to-noise.

## Prefill kernel breakdown (rocprofv3 kernel trace, 8K, one run)

43.1 s of kernel time over 366 817 dispatches:

  ds4_mmq_compact_mul_mat_q (IQ2 gate/up)          21.5%
  moe_down_q2K_hotlist_wmma_n2_kernel              13.7%
  matmul_q8_0_f32_batch_wmma_rowtile<128,8,8>       8.7%
  attention_mixed_heads32_gufo_wmma                 6.6%
  v41_grouped_q8_f32_wmma_rowtile<256,16,8>         5.3%
  matmul_q8_0_f32_batch_wmma_rowtile<256,16,16>     4.3%
  v41_bf16_kernel                                   4.3%
  v41_indexer_batch_warp8_kernel                    3.8%
  moe_down_q2K_expert_batch_sharedmid               3.7%
  Cijk_Alik_Bljk (hipBLASLt)                        3.3%
  matmul_q8_0_f32_batch_wmma_rowtile<256,16,8>      3.3%
  matmul_f16_f32_sharedx_warp_rows_w32              2.6%
  __amd_rocclr_copyBuffer                           2.3%
  indexer_topk_pow2<4096> + topk_8192_cub           4.0%
  v41_rope_kernel                                   2.0%

MoE ~35% (MMQ gate/up + down), Q8 projections ~26%, attention ~7%, indexer ~8%.

Important correction: the V4.1 compact down kernel (q2_wstage64_n2) never runs
under --ssd-streaming (v41_compact requires !g_ssd_streaming_mode), so the
double-buffer plan in the GEMM article applies to
moe_down_q2K_hotlist_wmma_n2_kernel instead - that is the 13.7% kernel, and its
weight staging is the same single-buffered pattern.

## CORRECTION: the promoted growth does not survive a clock-soaked A/B

rocm-smi shows `Performance Level: auto` with sclk levels 600/1100/2900 MHz, and
soak sampling lands at 1762-1840 MHz with visible drift. Re-running the grow
A/B with a fixed 20 s warm-up before every arm (so both arms start from the same
boost state) collapses the effect:

  grow on  11.37 / 11.67 tps     grow off  11.60 / 11.31 tps
  mean 11.52 vs 11.46 = +0.6%, ranges overlap (min(on) < max(off))
  frontier logits bit-identical in every arm

and the 4K append is consistently *slower* with grow on:
  27 667 / 27 333 ms vs 26 466 / 26 287 ms (-4%, both pairs).

So the earlier "+5-10% decode" (12.21/12.32 vs 10.73/11.45) was DVFS drift
between arms, not the optimization. The change is demoted to opt-in
(DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL=1) and must not be promoted on the
strength of unsoaked numbers.

Protocol change that follows: every A/B from now on runs the fixed warm-up and
records rocm-smi clocks per arm. Without that, 3-7% "wins" are indistinguishable
from boost state.

## Q8 wide-tile dispatch variant (rejected, opt-in only)

Forcing the 256-column/16-row WMMA rowtile for all projections with out_dim >= 1024
(previously only >= 8192) behind DS4_ROCM_Q8_WIDE_TILE=1. Soaked A/B/A/B at 32K:

  32K prefill  off 176.29 / 175.43   on 176.48 / 174.84   -0.1%, overlapping
  4K append    off 153.74 / 154.23   on 152.81 / 154.00   -0.4%, overlapping
  frontier logits bit-identical in every arm (max_abs 0)
  clocks 1714-1780 MHz across arms (auto perflevel)

No win; stays opt-in, default dispatch untouched.

## Download-prefetch follow-up (kept opt-in)

moe_down_q2K_hotlist_wmma_n2_kernel register prefetch: bit-identical, 32K prefill
+0.6% with disjoint ranges (177.21/178.04 vs 176.40/176.61), append +1.1% not
significant. Kept opt-in (DS4_ROCM_V41_DOWN_PREFETCH=1): real but far inside the
band we refuse to promote on.

## Decode dispatch/gap structure and the Q8 blocks4 unroll

Trace-derived decode budget (4 tokens, 8K): 87 ms/token wall = 65.7 ms kernel busy
+ 21.4 ms gap. 2408 dispatches/token; gap p50 2.2 us, p90 4.1, p99 25, one 19 ms
stall; gaps >100 us hold 55 of the 85 ms gap total. So per-launch overhead is
small - the gap budget is a few large stalls, not 2400 tiny ones.

Largest decode kernel remains v41_q8_f32_blocks4_kernel: 201 calls/token,
24.8 ms/token (123 us/call). Its outer block loop is a 40-step serial chain with
no unrolling, so each iteration eats a full memory round trip.
DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH now selects a `#pragma unroll 4` variant with
the same lane->block mapping and per-block accumulation order.

Soaked A/B/A/B (clocks 1761-1793 MHz), frontier logits bit-identical everywhere:
  32K prefill 175.66/174.44 vs 174.92/174.52  -> -0.2%
  4K append   153.19/152.42 vs 153.73/152.89  -> +0.3%
  decode      11.16/11.06  vs 11.25/11.54     -> +2.6%, ranges just disjoint
Promising but under the 3-7% band: stays opt-in pending a confirmation run.

### Unroll confirmation (gen 64 per arm, 32K x2 pairs + 110K pair)

  32K decode   off 11.02 / 11.09   on 11.18 / 11.33   +1.8%, disjoint
  110K decode  off 10.48           on 10.70           +2.1%, disjoint
  32K prefill  175.02 / 175.34 vs 174.96 / 174.34     neutral
  4K append    152.87 / 153.12 vs 153.39 / 152.94     neutral
  logits bit-identical at 32768, 36864, 110592, 114688; clocks 1674-1780 MHz

Consistent, always positive, bit-identical, prefill-neutral - but +1.8..2.1% is
below the 3% promotion bar and the 110K point is a single pair. Stays opt-in
(DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=1). Promotion would need either a default-on
trial with 4+ pairs per context or a larger win elsewhere.

## Opt-in q_a+kv fusion and its A/B (kept opt-in)

Shape log (DS4_ROCM_V41_Q8_SHAPE_LOG) resolved the 201 decode Q8-blocks4 calls:
per layer q_a(5120->1280), q_b(1280->32768), kv(5120->512), attn_out_up(8192->5120),
shared_up(2304->5120), plus the LM head once per token. Only q_a and kv share
width and row_bytes and read the same norm row, so they are the one fusable pair.

DS4_ROCM_V41_Q8_FUSE_QAKV=1 issues one launch for both
(v41_q8_f32_blocks4_pair_kernel, per-row weight-base select, identical lane math).
Frontier logits bit-identical at 32768/36864/110592/114688 in every arm.

Paired decode ratios (gen 64, 20 s warm-up, clocks recorded):
  fusion alone   pair1 +7.8%, pair2 +1.1%   mean +4.5%
  unroll alone   pair1 +0.8%, pair2 +4.3%   mean +2.6%
  fusion+unroll  pair1 +5.6%, pair2 -3.3%   mean +1.0%
Cross-session off-arm means drifted 11.06..11.44 tps (3.4%), the same size as
these effects, and the combination is not additive. Nothing here clears the bar
reliably: both flags stay opt-in, and resolving them needs 4-6 interleaved
pairs per context rather than 2.

## MMQ tile-width sweep (rejected: the gfx1151 clamp is right)

get_mmq_x_max_host() clamps to 64 on gfx1151 (device side allows 128), with an
existing hook DS4_CUDA_MMQ_X_MAX. Sweeping it on the largest prefill kernel
(ds4_mmq_compact_mul_mat_q, 21.5%):

  x128  32K prefill -8.2%   append -6.1%   frontier logits max_abs 1.29/1.88
  x96   32K prefill -5.3%   append -3.5%   frontier logits max_abs 1.06/1.57
  x64   baseline (b1/b2 bit-identical, drift 0.4%)

So the per-architecture clamp is correct: wider tiles are both slower and
numerically different. No change.

## Decode tri-state interleaved measurement (8K, gen 64, 4 rounds)

  off      tps = 11.72 12.26 12.21 12.08
  fuse     paired +4.4 +0.8 -0.2 +1.0  -> mean +1.5%, sd 2.0, 3/4 positive
  unroll   paired +5.6 +0.6 +1.6 +1.2  -> mean +2.2%, sd 2.3, 4/4 positive
  prefill neutral (176.8..178.0 vs 175.7..177.4); all 12 arms bit-identical

Both stay opt-in: consistent and risk-free, but ~2% is inside the session drift.

## Prefill: where the time actually goes (context sweep + cold start)

Ascending single-session frontiers (same prompt, gen 0, 92GB cache):
  ctx 2048  16.16 ms/token  (61.9 tps)  first frontier, cold expert cache
  ctx 4096   7.93 ms/token  (126.2 tps)
  ctx 8192   5.73 ms/token  (174.6 tps)
  ctx 16384  5.62 ms/token  (178.0 tps)  steady state
  ctx 32768  5.88 ms/token  (170.0 tps)

Read: a fresh session pays a large fixed expert-streaming cost for the first
~4K tokens (2.9x per-token at 2K), then prefill settles at ~5.6-5.9 ms/token and
is flat from 8K to 32K - i.e. steady-state prefill is compute/issue bound, while
short/cold prefills are dominated by streaming the expert set once.

Corrected instrumentation note: TCC/DRAM counters are unavailable on this
platform (only GRBM_GUI_ACTIVE and SQ_WAVES collect), so FETCH_SIZE is L2-request
bytes at sector granularity, NOT DRAM traffic. The earlier "7.65x DRAM
amplification" inference is retracted; it is unsupported.

Config levers exhausted for this path: --ssd-streaming-preload-experts is
rejected for V4.1 ("SSD ... preload modes are unsupported") and 0 is invalid, and
--prefill-chunk 4096/8192 changes nothing (-0.1%/-0.3%) with bit-identical
logits, confirming the chunk/tile schedule is not the limiter.

## PROMOTED: fused-epilogue compact MMQ tile (default on)

`ds4_mmq_compact_mul_mat_q` had two instantiations: the split write-back path and
a fused-epilogue variant selected only by DS4_ROCM_V41_COMPACT_JLIMIT=1. Measured
across two sessions, both contexts, always positive, logits bit-identical:

  32K prefill   +2.9%  +2.3%  +2.7%(default-on check)
  32K append    +2.5%  +2.7%
  110K prefill  +2.4%
  110K append   +4.1%   (40.15 s -> 38.55 s)

Ranges disjoint in every pair. Default flipped to the fused variant with
DS4_ROCM_V41_COMPACT_JLIMIT=0 as the opt-out; the base binary captured before the
change reproduces 175.63 tps vs 180.39 tps for the new default (+2.7%).

### Fused-epilogue trace confirmation (8K prefill)

  before: ds4_mmq_compact_mul_mat_q (split path)  9499.8 ms  640 dispatches  21.6%
  after:  ds4_mmq_compact_mul_mat_q<...,true>    8169.7 ms  320 dispatches  19.3%
          ds4_mmq_compact_produce                  6.4 ms  160 dispatches
  total kernel time 43961 -> 42334 ms (-3.7%)

So the fused epilogue removed 320 dispatches (the separate per-tile write-back
pass) and 1.33 s of MMQ time at 8K, which is exactly the -2.7% wall-clock gain;
the residual MMQ body is 19.3% of prefill. `ds4_mmq_compact_produce` (the
device-side compact-list builder) is 0.015% - not worth touching.

Residual budget at 8K now: MMQ body 19.3%, down hotlist 14.1%, Q8 rowtile
variants 8.8+5.6+4.5+3.4%, attention 6.7%, indexer 4.0%, bf16 4.4%.

## Down-kernel row-tile sweep (rejected) and where the remaining prefill time is

moe_down_q2K_hotlist_wmma_n2_kernel (14.1% of prefill) with 8 row-tiles per block
instead of 4 (DS4_ROCM_V41_DOWN_MT8=1): 32K prefill 176.64 -> 176.76 tps
(+0.1%, overlapping), logits bit-identical. Neutral; the tile shape is not the
limiter.

Measured ceilings for the "remove a pass" ideas:
  moe_sum_f16x2_kernel (the separate slot-sum pass)     128.7 ms   0.30%
  v41_bf16_kernel tiny per-tensor calls (grid 256/512)    34.4 ms   0.08%
  __amd_rocclr_copyBuffer                             992.1 ms   2.34%  61 470 calls

So fusing the down slot-sum would buy at most 0.30% while breaking bit-exactness
(each slot rounds to F16 today) - rejected. Batching the small bf16 conversions
is 0.08% - rejected. The one remaining elementwise target of size is the 61 470
tiny copyBuffer calls at 2.34%: those need their issuing site identified before
any change.

## copyBuffer issuer identified (batching not feasible at engine level)

rocprofv3 runtime trace of a 2K prefill: 37 106 hipMemcpyAsync + 2 218 hipMemcpy
+ 75 939 hipLaunchKernel per 2 048 tokens. A tally added to our own
ds4_gpu_tensor_copy (DS4_ROCM_COPY_TALLY=1) shows only **6 238 calls / 186.6 MiB**
come through the engine API - i.e. ~16% of the copies; the other ~33 000 are
rocclr-internal (streamed expert staging uploads and tensor writes). Batching at
our level can therefore address at most ~0.4% of prefill, so it is not worth
doing.

## Next candidate with real size: fuse the bf16 rounding into projections

v41_bf16_kernel is 1.87 s of an 8K prefill (4.4%), and the bulk of it converts
projection outputs that ds41_matmul(..., round=true) writes as F32 and then
rounds in a second pass (~59 GB read + ~29 GB write per 8K prefill). Writing bf16
directly from the projection epilogue applies the same rounding to the same
values, so it is bit-identical by construction, and it removes a full read+write
pass. Largest single target: matmul_q8_0_f32_batch_wmma_rowtile_kernel<128,8>
(8.8% of prefill), whose output is bf16-rounded immediately afterwards.

## Q8 bf16-epilogue fusion (opt-in, small consistent win)

ds41_matmul_batch rounds every Q8 projection output to bf16 in a second pass
(v41_bf16_kernel, 4.4% of an 8K prefill). DS4_ROCM_Q8_FUSE_BF16=1 moves that
rounding into the rowtile epilogue (bit-identical algorithm), returns rc=2 from
ds4_gpu_dsv41_q8_projection_rows so the caller skips its pass, and falls back to
one follow-up rounding pass for paths that are not instrumented.

Two wrong intermediate versions were caught by the bit-equality gate: rounding
unconditionally broke callers that asked for no rounding (max_abs 1.40/1.92);
the fix threads the caller's round intent (ds4_gpu_dsv41_q8_set_round) and the
rc==2 contract.

Three interleaved 32K pairs, all positive, logits exact in all six arms:
  32K prefill  +1.0% (+1.0/+1.2/+0.8, sd 0.2, 3/3)
  4K append    +0.8% (+0.2/+1.2/+1.1, sd 0.5, 3/3)
Earlier single pairs read +4.2% then +0.7%; the true effect is ~1%. Below the
promotion bar, so the flag stays opt-in.

## PROMOTED: q_a+kv fusion and the unrolled blocks4 loop (defaults on)

Both were opt-in and bit-identical. Attribution at 8K/gen64, two rounds:
  default           11.50 / 12.13 tps
  +fuse             12.15 / 12.26   (+3.3% mean, both rounds positive)
  +fuse+unroll      12.27 / 12.46   (+4.7% mean, highest in both rounds)
  +all(bf16)        12.23 / 12.48   (bf16 adds nothing on decode)
Cumulative pair history: fuse positive in 6 of 8 pairs, unroll in 8 of 8.
Promoted: DS4_ROCM_V41_Q8_FUSE_QAKV=0 and DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=0
restore the previous paths.

Session aggregate vs the pre-session binary (defaults only, no env):
  32K  prefill +3.5%, 4K append +5.7%, decode +7.4%
  110K prefill +2.9%, 4K append +3.9%, decode +6.0%
  frontier logits bit-identical at 32768/36864/110592/114688, argmax same
That is the fused-epilogue MMQ tile + q_a+kv fusion + unrolled blocks4 loop;
DS4_ROCM_Q8_FUSE_BF16 stays opt-in (~1% prefill, no decode effect).

## moe_down half2 epilogue (rejected) and final session verification

Vectorising the two down epilogue stores into half2 (adjacent nn share one bucket
pair, so rows are adjacent in the slot-major output) kept logits bit-identical
but cost **2.4% prefill** (175.89 -> 180.15 for the scalar path, disjoint). It
halves the number of active epilogue threads for the same work, so it is a
regression: reverted.

Lesson recorded: this session's wins came from *removing launches or passes*
(fused epilogue, fused q_a+kv launch, unrolled chain), not from redistributing
per-thread work inside a kernel, which every attempt has made slower or neutral
(occupancy cap, mt8 tiles, half2 stores, register prefetch on the down kernel).

Final verification of the promoted defaults, 8K and 128 greedy tokens against the
pre-session binary: generated text **identical**, frontier logits bit-identical
(exact, max_abs 0), prefill 177.22 -> 182.49 tps (+3.0%), decode 11.44 -> 12.16
tps (+6.3%).

## 110K confidence: three independent pairs for the promoted defaults

  pair 1 (session-total)  prefill +3.0%  append +4.5%  decode +5.2%
  pair 2 (promote-check)  prefill +2.9%  append +3.9%  decode +6.0%
  pair 3 (third pair)     prefill +2.4%  append +3.4%  decode +8.9%
                            (140.21 -> 143.56 tps, 40.12 -> 38.80 s append)

All bit-identical at 110592/114688 with identical argmax, in three separate
sessions. Combined with the repeated 32K pairs (prefill +3.0..3.5%, append
+3.6..5.7%, decode +6.3..7.4%) and the 128-token text-identical check, the
promoted set (fused-epilogue MMQ tile, q_a+kv fusion, unrolled blocks4) is
supported at both context sizes.

## 32K prefill cost map (177.5 s kernel time, 1.27M dispatches)

  ds4_mmq_compact_mul_mat_q (fused epilogue)   18.3%   1 280 calls
  moe_down_q2K_hotlist_wmma_n2                 13.3%     640
  matmul_q8_0 rowtile <128,8>                   8.6%     640
  hipBLASLt Cijk SB_MT32x32x8 (indexer GEMM)    7.5%   7 552   <- largest non-MoE item
  attention_mixed_heads32_gufo                  6.8%     608
  v41_grouped_q8 rowtile <256,16,8>             5.7%     640
  v41_indexer_batch_warp8 (scores)              5.2%   5 120
  matmul_q8_0 rowtile <256,16,16>               4.1%     640
  v41_bf16_kernel                               3.9% 143 666
  moe_down_q2K expert_batch_sharedmid           3.6%     640
  matmul_q8_0 rowtile <256,16,8>                3.2%   2 560
  f16 sharedx warp rows                         2.5% 376 832

Top-k family totals ~4.3% (pow2 4096 0.85, cub 8192 1.05, chunked 0.79, mask 1.22).
Indexer scoring together (rowtile + warp8 + Cijk) is ~12.7% - the next biggest
block after the two MoE kernels. Its GEMM path runs 7 552 calls at ~1.77 ms each
for 32-query tiles (~38 GFLOP/s), so batching query tiles is the obvious lever,
but any tile change alters GEMM accumulation order, so it cannot be kept
value-exact and would need the shape-noise gate plus a top-k stability check.

## Indexer query batching (runtime knob, rejected: -10%)

DS41_INDEX_BATCH is now a runtime value (DS4_ROCM_V41_INDEX_BATCH, default 32,
clamped 32..512) and both the graph's index_scores buffer and its byte
accounting scale with it, so a larger batch is charged to the plan. Measured
memory cost at ctx_alloc 40960: buffers 2.71 -> 2.72 (128) -> 2.74 GiB (256),
i.e. ~30 MB at 256 - memory was never the obstacle.

Performance, 32K/4K-append, gen 32:
  batch 32   181.87 / 180.77 tps (mean 181.3), append 26.50 / 26.21 s
  batch 128  163.58 tps (-9.8%), append 31.89 s
  batch 256  162.63 tps (-10.3%), append 31.96 s
Frontier logits stayed bit-identical for every batch (exact, max_abs 0), which
confirms GEMM M-tiling does not change the per-score reduction - the loss is
kernel efficiency, not arithmetic.

Rejected: the indexer score kernels are tuned for 32-row tiles and lose ~10%
with coarser batches. The knob stays available and unused at its default.

## Q8 <128,8> PAD=8 (opt-in; did not reproduce)

Instantiating the generic 128-row Q8 rowtile with PAD=8 removes the LDS bank
conflicts the GEMM article's step 3 describes; only the shared layout changes, so
logits stayed bit-identical in every arm.

  session A (32K): 176.32 -> 180.71 tps  (+2.5%, disjoint), append 26.8 -> 26.4 s
  session B (32K): 182.04 -> 180.25 tps  (-1.0%), append -0.1%
  session B (110K): 142.54 -> 143.12 tps (+0.4%), append +0.4%

Mixed sign between sessions with a 176 -> 182 tps baseline shift, so the +2.5% did
not reproduce. Stays opt-in (DS4_ROCM_Q8_PAD8=1); not promotable.

## Deployment check: the promoted gains were not in the served binary

The cockpit container `ds4-v41-optimized` runs `/usr/local/bin/ds4-server`
(sha256 ed638ca8...) while the host binary was from Nov 1 15:27 (2a494f45...),
i.e. both predate every optimization promoted today; all measurements so far
were on `ds4-bench`. `ds4-server` has now been rebuilt from the current sources
with the same ROCm object set (Nov 3 17:34, sha256 58bda5ba...), so the promoted
changes exist in the server binary too.

Remaining deployment step (outside a benchmark payload): rebuild the container
image so the cockpit serves that binary, then repeat one 32K pair through the
API to confirm the served path shows the same +3-4% prefill / +5-9% decode.

## Non-packed indexer path (not reproduced) and a protocol lesson

DS4_METAL_DISABLE_V41_PACKED_INDEX=1, 32K:
  session A: 175.99 -> 180.31 tps (+2.5%, disjoint), append -2%
  session B: 180.97 -> 179.94 tps (-0.6%), 110K -0.4% prefill / +1.2% append
Logits bit-identical in both sessions.

Same pattern as Q8 PAD=8: a disjoint +2.5% in one session that flips sign next
time, while the *session baseline itself* moved from 176 to 181 tps. The 20 s
warm-up is not enough to pin this machine down; only multi-pair interleaving
inside one session (as used for the decode flags: 3-4 pairs) gives decisions we
can trust.

Protocol rule for this tree, on top of the earlier one: a candidate needs >= 3
interleaved pairs within a single session AND a second session; single-pair
disjoint wins of <= 3% are not evidence on this machine.

Deployment boundary (unchanged, container untouched): ds4-server on the host is
rebuilt (58bda5ba) and contains the promoted paths, but the cockpit image still
serves ed638ca8. The image rebuild and the API-level 32K pair remain the
outstanding verification step.

## Protocol-compliant validation of the promoted set (authoritative numbers)

Three interleaved 32K pairs in one session, pre-session binary vs current
defaults, 20 s warm-up per arm:

  32K prefill   +1.8 / +2.6 / +2.9  -> mean +2.5% (sd 0.5), 3/3 positive
  4K append     +1.9 / +3.0 / +2.0  -> mean +2.3% (sd 0.6), 3/3 positive
  frontier logits bit-identical in all six arms (exact, max_abs 0)

These supersede the single-pair figures quoted earlier (+3.5% prefill / +5.7%
append) as the numbers to trust: the promoted set is worth **+2.5% prefill and
+2.3% append at 32K** under the multi-pair protocol, with the decode gains
(+5.2..8.9%) coming from the separate interleaved decode measurements.

## MMQ "staging-row" experiment: wrong target, and why

The RAW_ROWS constant belongs to moe_down_q2K_hotlist_wmma_n2_kernel
(rocm/ds4_rocm_moe.cuh:5513, RAW_ROWS = 2*BN = 32 output rows per staging round),
not to ds4_mmq_compact_mul_mat_q. The compact MMQ kernel's tiling is fixed at
mmq_y = 64 x-rows and mmq_x = 64 dst-columns by
GGML_ASSERT(... mmq_y==64 && args.nrows_x==2304 ...), so "widening the staging
rows" there would mean re-plumbing the shared tile, not tuning a constant.

For the down kernel, the equivalent tile-area change was already measured:
mt=8 rows per block instead of 4 (DS4_ROCM_V41_DOWN_MT8) gave +0.1% prefill with
bit-identical logits - neutral. Widening RAW_ROWS to 4*BN is the same class of
change on the same kernel, adding two more accumulator fragments and ~16 VGPRs to
a kernel already at 2 waves/SIMD, so the prior is neutral-to-negative.

Conclusion: there is no remaining value-exact lever on either MoE kernel that has
not already been measured. The next real step is numerics-changing kernel work
(MMQ dequant instruction count, indexer GEMM tiling) under a multi-pair campaign.

## Concrete MMQ dequant opportunity: the aligned-SoA IQ2 artifact

Where the time goes: `load_tiles_iq2_xxs` (cuda/mmq/mmq.cuh:2991) does, per 8
weights, one *table lookup* (`((const uint2*)iq2xxs_grid)[aux8[l]]`), one
sign unpack, two `__vcmpne4`, two `__vsub4` and two LDS stores - roughly 10-12
instructions per 8 weights against one mma per 8 MACs, in the kernel that is 18.3%
of 32K prefill.

The tree already carries the value-exact alternative: `load_tiles_iq2_xxs_soa`
(mmq.cuh:3059) reads a `[__half dq | uint2 qs]` artifact so each lane's
(q2, aux32) pair is one artifact uint2, "8-byte aligned and fully coalesced", and
its own comment states the unpack math is the standard loader's, "making tile_x
bit-identical". The artifact is exactly the same number of bytes as the raw IQ2
blocks (66 B per 256-weight block, ~47 GB for gate+up at 40 layers x 384 experts).

What is missing: a packer (only the inverse, `iq2_xxs_aligned_derepack_kernel` at
ds4_mmq.cu:3982, exists), an artifact store (disk cache or a session repack), and
passing `x_soa`/`soa_blocks` into the V4.1 MoE MMQ call, which currently passes
nullptr.

Verification path (strict, and cheap relative to the payoff):
1. Bit gate first: the SoA loader is documented bit-identical, so frontier logits
   must be unchanged at 32768/36864 and 110592/114688.
2. Multi-pair protocol: >= 3 interleaved 32K pairs in one session plus a second
   session, because the kernel is 18.3% and the noise floor is ~2%.
3. Include the artifact cost in the comparison: the repack is a full 47 GB
   read+write, so either it is a persistent cache (charged once) or it must be
   amortised across the measured session - otherwise the A/B compares a warm
   artifact against a cold path.

## SoA plumbing map (exact locations) for the IQ2 dequant artifact

Call path: V4.1 MoE gate/up -> ds4_mmq_iq2_xxs_moe_pair_swiglu
(cuda/mmq/ds4_mmq.cu:2034) -> ds4_mmq_moe_pair_impl<IQ2_XXS>(...) which already
takes `xa_soa, xb_soa, soa_blocks` (declared 1297-1299, used 1672). The swiglu
wrapper passes `nullptr, nullptr, 0` today, so only that wrapper (or a new
`_soa` sibling) plus the launcher call site need to learn about the artifact;
line 1522 shows the non-SoA fast path is skipped whenever xa_soa/xb_soa are set,
so the impl side is ready.

Layout compatibility: NOT drop-in. `load_tiles_iq2_xxs_soa` (mmq.cuh:3076) splits
each 66-byte IQ2 block into two planes - `dq` (soa_nblk x 2 B, 64 B aligned) then
`qs` (soa_nblk x 8 uint2 = 64 B/block) - so the artifact is a separate buffer of
identical total size (66 B per 256-weight block; ~47 GB for gate+up at 40 layers x
384 experts) rather than a reinterpretation of the model map.

Where it can be packed: only as a persistent artifact (disk has 1.1 TB free).
A per-layer device repack does not pay: each layer is revisited once per prefill
chunk (16 chunks at 32K), i.e. ~1.5 TB of repack traffic per 32K prefill.

Value: the SoA loader replaces `unpack_ksigns` + two `__vcmpne4` with a single
`ksigns64` LUT load and the two 4-byte weight reads with one coalesced 8-byte
read, in a kernel that is 18.3% of prefill, so the plausible range is 2-4%.

Gate: frontier logits must stay bit-identical (the loader's own comment states
tile_x is bit-identical), then >=3 interleaved 32K pairs plus a second session,
with the artifact cost charged explicitly.

## SoA artifact in the weight lifecycle: why the smallest slice is a counter, not a packer

Lifecycle trace: there is no prepared-weight cache to piggyback on. The only
"prepared" mechanism (rocm/halo/q2/prepared.hip, ds4_rocm_halo_q2_prepare /
_prepared) is a GEMM store kernel for the compact Q2 path, not a persistent
artifact store. IQ2 gate/up weights reach the MMQ kernel straight from the
model map, staged into the 7.12 GiB expert staging buffer by the streaming layer.

Residency argument: a 47 GB SoA artifact cannot be resident - 92 GB are already
allocated to the expert cache and only 121.73 GiB is usable. So either
(a) the artifact lives on disk and is streamed instead of the raw blocks, or
(b) it is repacked per layer into the staging buffer.

(a) buys nothing: the streaming copy already moves whole expert rows contiguously,
so plane-splitting the on-disk layout does not improve that transfer; the dequant
loop reads whatever the staging buffer holds.

(b) only pays if the staging buffer for a layer survives across that layer's
prefill chunks. At 32K there are 16 chunks per layer, so a 2.34 GB per-layer
repack would be amortised 16x (47 GB per prefill, ~10 s at 5 GB/s) against a
plausible 3% of 186 s = 5.6 s - i.e. break-even at best, and negative if staging
is refilled per chunk (then repack traffic is ~1.5 TB).

Decisive unknown: whether layer staging persists across that layer's chunks.
Smallest in-scope slice: an opt-in counter of expert bytes read per layer per
prefill (no behaviour change). If it shows one read per layer, the SoA repack is
worth prototyping; if it shows one read per chunk-layer, the idea is dead and the
remaining MMQ work should go to instruction-level dequant changes instead.

## Staging lifecycle answer, and the reformulation that survives it

ds41_stream_layer_enter (ds4.c:43273) proves the reuse question: under selective
staging it returns early because "the current layer is staged after routing
inside ds41_moe_batch" - and ds41_moe_batch runs per chunk. So a layer's staging
is filled **per chunk-layer**, not once per layer:

  -> a per-layer disk/repack artifact does NOT survive across chunks: the
     SoA-on-disk and per-layer-repack variants are dead (would be ~1.5 TB of
     repack traffic per 32K prefill).
  -> what does survive is the 92 GB expert cache, which serves the repeats.

Reformulation that is still viable: make the **cache entry itself** the SoA form.
The repack then happens inside the cache-fill copy that already runs on a miss
(no extra traffic, a permute in a kernel we already pay for), and every later use
of that expert - within the chunk and across the 16 chunks at 32K - reads the
coalesced dq/qs planes and the single ksigns64 LUT. That keeps the 2-4% upside of
the SoA loader without any new disk artifact and without the residency problem.

Smallest implementation slice, if resumed: a packer kernel alongside
iq2_xxs_aligned_derepack_kernel, called from the selected-read upload path
(cuda_stream_selected_upload_read_jobs / the resident-cache insert), storing dq
and qs planes per cached expert; then pass xa_soa/xb_soa + soa_blocks from the
swiglu wrapper (the only place that drops them) when the cache entry is in SoA
form. Gate unchanged: bit-identical frontier logits first (the loader is
documented bit-identical), then >=3 interleaved 32K pairs plus a second session.

## SoA packing cannot share the cache-fill work: the numbers close the avenue

Cache-fill path: cuda_stream_selected_upload_read_jobs (ds4_rocm_runtime.cuh:2750)
is a plain `cudaMemcpyAsync(dst, host_buf, bytes, H2D)` per job followed by a
stream sync, i.e. one bulk copy per (expert, part) straight from the worker's host
buffer into the cache slot. There is no spare kernel to piggyback a permute on,
and an in-place permute inside a 3 MB slot is a scatter that cannot be done
without a second buffer (doubling cache bytes).

Index mapping, from the constants: per expert the SoA form is
2304 rows x 20 blocks = 46 080 blocks -> dq plane 92 160 B + qs plane 2 949 120 B
= 3 041 280 B, exactly gate_expert_bytes (2304 x 1320). So one expert maps to one
SoA blob of identical size - the granularity is right, but the work is not free:

  permute at 150 GB/s: 20 us per expert -> +4.9 ms/token of decode (240 experts
  per token, i.e. ~6% of an 87 ms token) and 0.31 s per 32K prefill (47 GB).

So a runtime permute costs ~6% decode to win 2-4% prefill: net negative. The idea
only works if the SoA layout is authored rather than derived - i.e. produced by
the quantizer (gguf-tools/deepseek41_quantize.py) so the model stores IQ2 in the
dq/qs plane form and the runtime permute disappears entirely. That is a model
format change, not an engine optimization, and it is the correct home for the
2-4%: engine side then only needs the wrapper to pass xa_soa/xb_soa/soa_blocks.

## Quantizer-side SoA: feasible only as a sidecar, and it needs an engine read path

gguf-tools/deepseek41_quantize.py already loops over the gate/up tensors with
QTYPE_IQ2_XXS (lines 119-120: ("gate","w1"),("up","w3")) and writes them into the
GGUF. Three ways to get the dq/qs plane layout, and only one survives the
constraints:

1. New GGUF tensor type (e.g. IQ2_XXS_SOA): changes the model format and every
   consumer that does not know the type, including the CUDA/Metal backends and the
   validation tools. Rejected on the stated constraint.
2. Sidecar file (e.g. <model>.iq2soa) with a header table of (layer, expert) ->
   plane offsets: the GGUF stays byte-identical, so existing consumers are
   untouched; the engine loads the sidecar only when present. Costs ~47 GB of
   extra disk (1.1 TB free) and needs a second mapped source plus read-job
   plumbing so gate/up staging is filled from the sidecar while down/Q8 stay in
   the GGUF.
3. Runtime permute into the cache: measured net-negative (+4.9 ms/token decode
   against 0.31 s per 32K prefill).

So the quantizer *can* emit the layout without breaking anything, but only via
option 2, and that makes it a two-part project (packer + engine read path) whose
ceiling is the 2-4% the SoA loader offers on an 18.3% kernel. Nothing in the tree
was changed for this step.

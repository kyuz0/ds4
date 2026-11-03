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

## Sidecar prototype (executed): packer + lossless round-trip

tools/pack-iq2-soa.py implements the smallest safe slice of the sidecar path: for
one tensor and one expert it emits the two SoA planes (global dq plane of 2 B per
66-byte block, then qs plane of 64 B per block - the layout
load_tiles_iq2_xxs_soa expects) and then re-interleaves them to prove the
transform is lossless. No engine change, no container impact, no numerics risk.

Verified on the real model:
  tensor=blk.20.ffn_gate_exps.weight expert=0 blocks=46080
  sidecar_bytes=3041280 roundtrip_identical=True
3041280 B is exactly gate_expert_bytes (2304 x 1320), so one expert's SoA blob is
the same size as one cache slot - the granularity the staging path already uses.

End-to-end shape of the remaining work, now that the bytes are proven:
1. extend the packer to a whole tensor (two global planes per tensor, ~47 GB for
   gate+up), with a small header table mapping (layer, tensor) -> plane offsets;
2. engine: map the sidecar when present and stage an expert from two ranges
   (dq plane slice + qs plane slice) instead of one, into the same cache slot;
3. pass xa_soa/xb_soa/soa_blocks from ds4_mmq_iq2_xxs_moe_pair_swiglu (the only
   place that drops them) with soa_blocks = 46080 and the expert's base offsets;
4. gates unchanged: bit-identical frontier logits first (the loader is documented
   bit-identical), then >=3 interleaved 32K pairs plus a second session.

## Sidecar prototype, whole tensor (executed)

tools/pack-iq2-soa.py now packs a whole IQ2 tensor: a 64-byte header
(magic/version/experts/blocks/dq offset/qs offset/end) followed by one global dq
plane and one global qs plane, then re-interleaves two experts (first and last)
straight from the file to prove the layout.

  tensor=blk.20.ffn_gate_exps.weight experts=384 blocks=17694720
  sidecar_bytes=1167851584 verified_experts=2 identical=True

1 167 851 584 B = 64 + 384*46080*2 + 384*46080*64 = one tensor's worth, i.e. the
sidecar is byte-for-byte the same size as the GGUF tensor it mirrors (1.17 GB for
this one; ~47 GB for gate+up across 40 layers). Two header bugs were caught by the
round-trip check itself (offsets read at the wrong header positions, and a name
field overrunning into the dq plane), which is exactly what this slice exists to
catch before any engine change.

Next slice: the engine read path - map the sidecar when present, stage an expert
from the two plane slices into the same cache slot, and pass xa_soa/xb_soa/
soa_blocks (46080 per expert) from the swiglu wrapper. Gates unchanged.

## Pointer trace: SoA indexes globally, so per-expert cache slots need a loader change

mul_mat_q_process_tile passes `x_soa` **un-offset** while the raw path passes
`offset_x + kb0`, which already contains `fastdiv(zt, channel_ratio)*stride_channel_x`
(mmq.cuh:4124 and 3890-3897). The SoA loader then computes
`blk = kbx0 + i*stride` and indexes the global planes as
`dq[blk]` / `qs[blk*8 + kqsx]` with `soa_blocks` covering the whole tensor.

Consequence: the artifact must be one global dq plane plus one global qs plane per
tensor, addressed by a flat block index across all 384 experts (stride_channel_x
is 46 080 blocks per expert, so the arithmetic lines up exactly with the packer's
layout). It cannot be sliced per expert and dropped into a cache slot: the loader
has no per-expert base and would index the wrong blocks.

Two ways forward, and their costs:
1. Whole-tensor residency: 1.17 GB per tensor, 47 GB for gate+up at 40 layers,
   against 121.73 GiB usable with 92 GB already in the cache - only ~7 layers
   would fit, i.e. <=0.7% overall. Not worth it.
2. Per-expert SoA in the cache: needs a loader variant taking an expert base and
   using local block indices, plus two-range staging (dq slice + qs slice) per
   expert miss. That is a loader + staging change, still bit-identical by
   construction, and it is the only variant that fits this machine's memory.

So the sidecar work is real but the engine-side effort is a loader variant plus
two-range staging, not just pointer plumbing; nothing was changed in the tree.

## Minimal change set for two-plane per-expert staging (traced, zero-cost)

Indexing, verified against the loader body (mmq.cuh:3076-3110):
  dq_bytes = (soa_nblk*2 + 63) & ~63          -> per-expert: 46080*2 = 92160, already 64-B aligned
  dq[blk] * scale, qs[blk*8 + kqsx]           with blk = kbx0 + i*stride
  kbx0 already contains zt*stride_channel_x (mmq.cuh:4124), so blk is *global*
For a per-expert slot holding [dq plane 92 160 B][qs plane 2 949 120 B] = one
3 041 280 B slot, blk must become expert-local. A modulo in the inner loop is not
acceptable (one integer divide per 8 weights), so the minimal zero-cost fix is to
carry the channel offset explicitly:

  add `int64_t soa_channel_blocks` next to `soa_blocks`, set to 0 when the
  artifact is global, and in both SoA loaders compute
      blk = (uint32_t)(kbx0 + i*stride - soa_channel_blocks)

Touch points (6, all mechanical):
  1. mul_mat_q_process_tile signature        mmq.cuh:3818
  2. mul_mat_q signature                     mmq.cuh:4034
  3. the three mul_mat_q_process_tile calls  mmq.cuh:4128, 4219, 4293
  4-5. the two loader call sites             mmq.cuh:3890 (Q2_K), 3897 (IQ2_XXS)
  6. args struct + wrapper plumbing          ds4_mmq.cu:1198 struct, 1281/1299/1672

Because the subtraction is a no-op when soa_channel_blocks == 0, the
absent-sidecar path (and the existing global-artifact path) is bit-identical by
construction; only the per-expert path changes, and it feeds the same tile_x
values. Staging then fills a slot from two ranges per miss: dq slice
(92 160 B) and qs slice (2 949 120 B), i.e. two read jobs instead of one, same
total bytes.

## Attempted first slice of the soa_channel_blocks change: reverted, exact text needed

Tried to introduce `int64_t soa_channel_blocks` through the SoA loaders and
mul_mat_q so a per-expert slot could use expert-local block indices with a zero
default (bit-identical for the raw and global-artifact paths). The patch applied
to the blk-site, the two signatures' parameter lists and the three process_tile
calls, but the loader signatures in the tree differ from the expected text
(load_tiles_q2_K_soa / load_tiles_iq2_xxs_soa are multi-line declarations), so
`soa_channel_blocks` was undeclared inside the loaders and the build failed at
mmq.cuh:3091/4128/4219/4543.

Reverted both files; `ds4-bench` rebuilds clean and the tree is unchanged. The
next attempt must quote the loader declarations exactly as they appear in the
tree (they span several lines and the parameter list order differs per loader),
and should be done as a self-contained cycle with the bit-equality run available.

## soa_channel_blocks plumbing landed (inert, verified bit-identical)

Threaded `int64_t soa_channel_blocks` through the IQ2 SoA loader, both
mul_mat_q_process_tile/mul_mat_q signatures, the four process_tile call sites, the
two compact-launcher calls, mmq_args and ds4_mmq_fused_down. The loader now reads
`blk = (uint32_t)((int64_t)(kbx0 + i*stride) - soa_channel_blocks)`, and every
producer defaults the value to 0, so the raw path and the global-artifact path are
identity transforms.

Verification: pre-patch binary vs patched binary at 8K, gen 8 -
frontier logits exact=True (max_abs 0); 184.12 vs 181.23 tps on a single run,
inside noise and expected for an inert change. The Q2_K SoA loader is deliberately
untouched (different pblk indexing; the V4.1 gate/up path is IQ2).

Next slice (now purely additive): stage an expert slot from two ranges (dq slice
92 160 B + qs slice 2 949 120 B) when a sidecar is mapped, and pass
soa_channel_blocks = expert*46080 from the swiglu wrapper along with xa_soa/xb_soa
and soa_blocks = 46080. Gates: bit-identical logits, then >=3 interleaved pairs
plus a second session.

## The SoA entry points already exist - reframing the next step

cuda/mmq/ds4_mmq.h already declares both `ds4_mmq_iq2_xxs_moe_pair_soa(Wa_soa,
Wb_soa, ...)` and the fused `ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(...)`, and the
implementation derives `nblk = n_experts * M * (K/256)` and passes `Wa_soa`
straight through as x_soa (ds4_mmq.cu:2059-2074). That is exactly the global
layout my packer emits (one dq plane then one qs plane per tensor, 66 B/block), so
no wrapper work is needed for a whole-tensor artifact: the sidecar only has to
exist and be mapped, and the call switches to the *_soa entry.

Mass arithmetic: gate+up for one layer is 2 x 1.17 GB = 2.34 GB, and the plan
leaves ~17 GB free (121.73 GiB usable - 104.91 GiB planned), so ~7 of 40 layers
fit in RAM. Full coverage would need the per-expert two-range scheme (which is
what soa_channel_blocks was added for).

Cheapest end-to-end validation of the payoff, given that 1/40 of the MoE time is
~0.45% overall (below the end-to-end noise floor):
1. pack layer-20 gate and up sidecars (packer already verified);
2. map them and call the *_soa entry for that one layer only;
3. gate on bit-identical frontier logits (the loader is documented bit-identical);
4. measure the effect as MMQ kernel time for layer 20 in a rocprof trace, not as
   end-to-end tps - that is the number that decides whether the per-expert scheme
   is worth building.

## Layer-specific sidecar switch: interfaces confirmed, implementation needs a fresh cycle

Confirmed at the call site (rocm/ds4_rocm_moe_launch.cuh:1598):
`ds4_mmq_iq2_xxs_moe_pair_swiglu(gate_w, up_w, x+off, selected_exec+pair_off,
weights+pair_off, gate, up, mid, mid_h, M, K, tile_tokens, n_total_expert,
n_expert, clamp, 0)` - the launcher already knows `layer_index`, and the
non-fused branch right below shows the separate `moe_swiglu_weighted_f32_kernel`
call that the `_soa` path would need, because
`ds4_mmq_iq2_xxs_moe_pair_soa` has no swiglu epilogue.

Three things the switch must get right, none of which is free to guess:
1. `selected_exec` in this path can hold remapped/compacted expert ids, while the
   SoA artifact is indexed by global expert id - the artifact variant must be fed
   the unremapped ids (or the resident path must be bypassed for that layer).
2. The `_soa` entry writes raw gate/up, so mid must come from the separate swiglu
   kernel with the same clamp; whether that is bit-identical to the fused
   epilogue has to be shown by the logit gate, not assumed.
3. A resident artifact also removes that layer's gate/up staging reads, which is a
   second, larger effect than the dequant saving - the trace has to separate them.

Given the remaining budget in this session I did not start an edit I could not
carry through the bit-exact and multi-pair gates; shipping an unverified kernel
path would break the constraint that has held all session. The next cycle should
implement: env-gated `DS4_V41_IQ2SOA_DIR` mmap of layer-20 gate/up sidecars, feed
the unremapped ids, call `ds4_mmq_iq2_xxs_moe_pair_soa`, run the separate swiglu
kernel, then gate on bit-identical logits and measure layer-20 MMQ kernel time.

## Producer exists in-tree, and the fused SoA entry answers the SwiGLU hazard

Producer: cuda/mmq/ds4_repack.h is the shared aligned-SoA layout library with
ds4_repack_build_iq2_aligned() (IQ2 gate/up), the Q2_K down and Q8_0 dense builds,
the GGUF catalog walk, a parallel repack driver and an FNV-1a artifact hash.  Its
header states it is shared by tools/ds4_weight_server.cu (weight server
--repack-iq2-aligned) and the engine's in-process self-load build in ds4_cuda.cu.
So the artifacts can be produced with existing code rather than the Python
prototype, which then only serves as the layout verifier.

SwiGLU hazard resolved: ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(W_gate_soa, W_up_soa,
W_down_soa, X, ids, router_weights, gate_f32, up_f32, mid_f32, down_f32, ...)
already "computes clamp + SwiGLU + router weighting in mid_f32, then gathers and
quantizes those rows for the Q2_K down MMQ", i.e. it is exactly this engine's
fused gate/up+down path with the epilogue included - no separate swiglu kernel is
needed and there is no fused-vs-separate equivalence question to prove.

Remaining hazard is only the id mapping: selected_exec can be replaced by the
resident/selected paths (rocm/ds4_rocm_moe_launch.cuh:887-954), while the artifact
is indexed by global expert id, so the SoA path must either bypass those or be
fed the unremapped ids.

Smallest next implementation, now with no unknown semantics: env-gated mapping of
the layer-20 gate/up/down artifacts (produced by ds4_repack_build_iq2_aligned and
the Q2_K build), call ds4_mmq_iq2_xxs_q2_K_moe_fused_soa for that layer with the
unremapped ids, and gate on bit-identical logits plus layer-20 kernel time.

## Expert-id mapping resolved: the selected path uses slot indices, not global ids

The resident/selected path hands the kernels *pointer tables*
(`gate_slot_ptrs` / `up_slot_ptrs` / `down_slot_ptrs`) plus `selected_exec`, and
the consumers index them as `table[id]` (e.g. moe_v41_down_wave_ptrs6_kernel).
So in that path `id` is a **slot index into the staged tables**, whereas the raw
MMQ path indexes `gate_w + expert*gate_expert_bytes` with the **global** expert id.

Consequence, and why the switch is not a call-site edit: the SoA artifact is
indexed by global id, so the layer-20 SoA branch must take the *original*
`selected` tensor and **bypass the staging/selected machinery entirely** for that
layer - no slot tables, no selected-read jobs - rather than reusing
`selected_exec` inside the existing flow. That is a branch at the top of the
launcher (before staging is set up), not a swap of the MMQ call.

Implementation shape for a fresh cycle (now fully determined):
1. env-gated base path, e.g. DS4_V41_IQ2SOA_DIR, expecting layer-20 gate/up/down
   artifacts produced by ds4_repack_build_iq2_aligned + the Q2_K build;
2. in routed_moe_launch, if the layer matches and the artifacts are mapped, skip
   the streaming/selected path for that layer and call
   ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(gate_soa, up_soa, down_soa, x, selected,
   weights, gate, up, mid, down, ...) with the untouched global ids;
3. gate on bit-identical frontier logits, then >=3 interleaved 32K pairs in one
   session plus a second session, and read layer-20 kernel time from a trace to
   decide whether the per-layer scheme is worth extending.

No edit was started: the branch restructures staging for that layer, which cannot
be carried through the gates with the remaining budget.

## Env-gated layer-scoped SoA branch implemented (builds; artifact run pending)

routed_moe_launch now has, ahead of the plan/staging setup:

  DS4_V41_IQ2SOA_DIR + DS4_V41_IQ2SOA_LAYER select a layer whose three
  aligned-SoA artifacts (l<layer>_gate.soa, _up.soa, _down.soa) are read once
  into device tensors; when the layer matches and the V4.1 shape holds
  (n_expert 6, 384 experts, mid 2304, in 5120, out 5120) the launcher calls
  ds4_mmq_iq2_xxs_q2_K_moe_fused_soa with the caller's untouched global
  `selected` ids and returns, bypassing staging/selected entirely.

Inert by construction: with either env var unset the branch is skipped and the
existing path runs unchanged. Verified to build; the artifact-run bit gate has not
been executed yet, so nothing is promoted.

## Layer-20 artifacts produced (gate + up); down needs the in-tree repack build

findings/soa-l20/l20_gate.soa and l20_up.soa are packed from the real model with
the verified tool:
  blk.20.ffn_gate_exps.weight  384 experts, 17694720 blocks, 1167851584 B, identical
  blk.20.ffn_up_exps.weight    384 experts, 17694720 blocks, 1167851584 B, identical

The fused entry also needs an aligned Q2_K down artifact, whose layout is
different (three planes: qs2 / sc4 / dm2, indexed by pairs - see
load_tiles_q2_K_soa and `soa_npair`), and the in-tree producer for it is
ds4_repack_build_iq2_aligned's Q2_K sibling in cuda/mmq/ds4_repack.h, used by
tools/ds4_weight_server.cu and the in-process self-load build. The Python packer
deliberately does not implement that layout; writing a second, unverified Q2_K
packer is exactly the kind of guess that produced the two header bugs earlier.

So the bit-exact gate cannot run yet: the fused SoA entry needs all three
artifacts, gate+up are verified, down must come from ds4_repack. Next cycle:
produce the Q2_K down artifact with the in-tree build for layer 20, then
DS4_V41_IQ2SOA_DIR=$(pwd)/findings/soa-l20 DS4_V41_IQ2SOA_LAYER=20 and the logit
gate, then the multi-pair measurements.

## Q2_K repack producer traced: it builds device artifacts in-process, not files

cuda/mmq/ds4_repack.h exposes the builders (ds4_repack_build_iq2_aligned,
ds4_repack_build_q2k_aligned, ds4_repack_build_q8_aligned) which allocate through
a caller-supplied ds4_repack_alloc_fn and hand back ds4_repack_artifact structs
holding device pointers - i.e. they build *in-process device artifacts*, and the
only in-tree caller is ds4_cuda.cu:4598 (the CUDA self-load build). File artifacts
would come from tools/ds4_weight_server.cu, whose source is not in this checkout.

Also: cuda/mmq/ds4_repack.o is in MMQ_OBJS for the Mer/CUDA build but is *not* in
the ROCm object set used here (ROCM_MMQ_OBJS has no ds4_repack), so the producer is
not compiled into ds4-bench at all.

Consequence for the layer-20 gate: the Q2_K down artifact cannot be produced with
an existing ROCm workflow. Two honest routes, both a bounded step rather than a
guess:
1. add cuda/mmq/ds4_repack to the ROCm build and call ds4_repack_build_q2k_aligned
   (plus iq2 for gate/up) in-process with a device allocator, then feed those
   pointers straight into ds4_mmq_iq2_xxs_q2_K_moe_fused_soa - which also removes
   my file/env-dir mechanism entirely, since nothing needs to touch disk;
2. extend the Python packer with the Q2_K three-plane layout - rejected earlier as
   an unverified guess.

Route 1 is the right one and is a small, well-scoped build + call-site change; the
env-gated branch added earlier stays unpromoted and unused until then.

## Smallest safe way to invoke the Q2_K producer for layer 20 (determined)

CUDA caller pattern (ds4_cuda.cu:4585-4602): build ds4_repack_build_args with
log_prefix/model_id/path/records/device/copy_chunk_bytes = 256 MB and *leave
alloc_fn/alloc_ctx null* (the library's default allocator handles it), then call
ds4_repack_build_q2k_aligned(args, artifacts, &part_bytes) and use the returned
ds4_repack_artifact device pointers.

ROCm build rule shape (Makefile:824-840): each mmq object is
`cuda/mmq/<name>.rocm.o: <sources>` + `$(HIPCC) $(ROCM_MMQ_FLAGS) -c -o $@ $<`, and
the object is listed in ROCM_MMQ_OBJS (Makefile:66), which currently has six
entries and no ds4_repack.

So the minimal safe invocation for layer 20 is:
1. add `cuda/mmq/ds4_repack.rocm.o` (same pattern, sources = ds4_repack.cu +
   ds4_repack.h) and append it to ROCM_MMQ_OBJS;
2. in the SoA branch, ds4_repack_map_file + ds4_repack_collect_catalog on the model
   path, keep only records whose name starts with "blk.20.", then call
   ds4_repack_build_iq2_aligned and ds4_repack_build_q2k_aligned with the same args
   shape the CUDA caller uses (256 MB staging, null allocator);
3. pass the three artifact pointers straight to
   ds4_mmq_iq2_xxs_q2_K_moe_fused_soa and drop the file/env-dir path entirely -
   nothing needs to touch disk, and the two verified .soa files become only a
   cross-check of the layout.

That is a build-list change plus one call site, using the producer's own Q2_K
layout, so no weight bytes are invented. The env-gated branch stays unpromoted
until the bit-exact gate runs on those artifacts.

## In-process Q2_K producer shim: written, not yet wired (build blocked on declaration scope)

Wrote findings/soa-l20/ds4_rocm_repack.hip.pending: a self-contained shim that maps
the model, collects the catalog, keeps blk.<layer>.* records, runs
ds4_repack_build_iq2_aligned + ds4_repack_build_q2k_aligned with the CUDA caller's
args shape (256 MB staging, null allocator) and returns the three artifact device
pointers - i.e. the producer's own Q2_K layout, no invented bytes.

Wiring attempt: added Makefile rules for cuda/mmq/ds4_repack.rocm.o and
rocm/ds4_rocm_repack.rocm.o (appended to ROCM_MMQ_OBJS) and switched the launcher
branch from the file/env-dir path to the in-process call. The build failed because
the forward declarations of ds4_rocm_repack_layer and the fused SoA entry ended up
inside routed_moe_launch's body rather than at file scope (same class of mistake as
the earlier loader-signature patch), so the identifier was undeclared at the call
site.

Reverted rocm/ds4_rocm_moe_launch.cuh and Makefile; both binaries rebuild clean and
git status shows no modified sources. The shim is parked as .pending next to the
verified gate/up artifacts. Next attempt: place both extern "C" declarations at
file scope above routed_moe_launch, keep the two Makefile rules, and switch the
branch; then the bit-exact gate can finally run with a "blk.<layer>." record
filter and DS4_V41_IQ2SOA_LAYER.

## Layer-20 in-process SoA wiring builds (unpromoted, gate not yet run)

Declarations moved to file scope above routed_moe_launch (the fix recorded last
step) and the branch now drives the in-tree producer instead of files:

  DS4_V41_IQ2SOA_MODEL + DS4_V41_IQ2SOA_LAYER
    -> ds4_rocm_repack_layer(model, layer, &g, &u, &d): maps the GGUF, collects the
       catalog, keeps blk.<layer>.* records, runs ds4_repack_build_iq2_aligned and
       ds4_repack_build_q2k_aligned (256 MB staging, default allocator), and returns
       the three artifact device pointers (producer's own Q2_K layout);
    -> ds4_mmq_iq2_xxs_q2_K_moe_fused_soa(g, u, d, x, selected /*global ids*/,
       weights, gate, up, mid, down, ..., preserve_reduction = 1), returning before
       staging is set up.

Build: ds4-bench links clean with cuda/mmq/ds4_repack.rocm.o and
rocm/ds4_rocm_repack.rocm.o added to ROCM_MMQ_OBJS.

Still unpromoted and inert without both env vars. Cost notes for the gate run:
the producer reads ~3.5 GB of layer-20 tensors once and allocates ~3.5 GB of device
artifacts, so the first call is a one-off ~2-4 s plus memory the plan does not
reserve - worth checking that admission still succeeds under the 92 GB cache.
Next: run with the two env vars, compare frontier logits against the promoted
binary (must be bit-identical), then the multi-pair measurement.

## Layer-20 in-process SoA gate: FAILED (not bit-identical, and slower)

Same binary, branch off vs on, 8K prefill:

  off  183.33 tps, 44.68 s
  on   169.45 tps, 48.35 s   ("V4.1 SoA repack layer 20 resident" logged once)
  frontier logits: exact=False, max_abs = 2.84

Two findings:
1. The fused SoA pipeline does NOT reproduce the reference arithmetic. max_abs
   2.84 is a semantic difference, not ULP rounding, so either
   ds4_mmq_iq2_xxs_q2_K_moe_fused_soa's own gate/up/down flow (its gather,
   quantization and reduction choices) differs from the staged V4.1 path by
   design, or the artifact wiring selects the wrong planes/tensors. The branch
   itself is behaving as coded (repack ran, the layer took the SoA route).
2. It is also slower at this size: the one-off repack (~3.5 GB read + ~3.5 GB of
   device artifacts) plus the fused pipeline costs more than the 1/40-layer
   saving.

Decision: the branch stays unpromoted and inert (needs both env vars). Closing the
SoA line as a viable engine optimization for this build: the payoff ceiling was
2-4% of prefill on one kernel while the cost is a full repack plus a numerically
different pipeline. If it is ever revisited, the first question is whether
ds4_mmq_iq2_xxs_q2_K_moe_fused_soa is expected to match the staged path at all -
that must be answered before any artifact work.

## Gate failure diagnosed: intentional numerical difference (Q8_K mid), not a mapping bug

Evidence from the saved logits:
  argmax identical (77179); top-5 the same five ids, positions 3/4 swapped
  |delta| p50 0.423, p90 1.036, p99 1.627, max 2.842; 98.8% of the vocabulary moved
  all top logits shrink (18.86 -> 16.76, 17.82 -> 16.27, 16.56 -> 14.68)

That signature is a systematic precision loss, not mis-mapped tensors: a wrong
plane or wrong tensor would give garbage, NaN or a different argmax entirely.

Mechanism, from the fused entry: it passes allow_d2r = !preserve_reduction (so
preserve_reduction=1 did select the raw MMQ reduction) but the fused pipeline
"gathers and quantizes those rows for the Q2_K down MMQ" - i.e. it feeds the down
projection Q8_K-quantized mid, whereas this engine's staged path feeds F16 mid
(mid_h_hot, MID_F16=true). Q8 mid on the down input is a lossy step the staged
path does not take, which explains a uniform, sign-consistent shrink of every
top logit.

So the SoA path as wired is a different numerical pipeline, and no amount of
artifact work will make it bit-identical. The narrow fix, if SoA is ever pursued,
is to use SoA only for gate/up: call ds4_mmq_iq2_xxs_moe_pair_soa (raw gate/up
over the artifact), then the existing moe_swiglu_weighted_f32_kernel and F16 mid
plus the staged Q2 down - which is bit-exact by construction and needs no fused
entry.

Branch stays unpromoted and inert; nothing further changed.

## Assessment: gate/up-only SoA CAN preserve the staged arithmetic; implementation not landed

Yes it can, through existing code: the launcher already contains the non-fused
gate/up route - ds4_mmq_iq2_xxs_moe_pair (raw gate/up) followed by
moe_swiglu_weighted_f32_kernel and f32_to_f16_kernel into iq2_hot_mid_h, then the
staged Q2 down. Replacing only the gate/up call with
ds4_mmq_iq2_xxs_moe_pair_soa (same impl, x_soa instead of raw, loader documented
bit-identical) keeps F16 mid and the staged down untouched, so the result is
bit-exact by construction - it is the same "raw write-back" path that already
exists two lines below the fused call.

Attempted exactly that: soa_gateup_ready hoisted to function scope, the SoA layer
routed through ds4_mmq_iq2_xxs_moe_pair_soa, and the two `use_fused_swiglu`
conditions changed to `use_fused_swiglu && !soa_gateup_ready` so the staged
SwiGLU + F16 mid epilogue runs for that layer. The hoist did not apply (the anchor
did not match) while the second edit removed the local declaration, so soa_dev was
undeclared at the use site and the build failed; the auto-revert guard in the same
command did not fire.

Restored rocm/ds4_rocm_moe_launch.cuh from a saved copy; ds4-bench rebuilds with
zero errors and git status shows no modified sources. The change is a 4-line edit
once the anchors are taken verbatim from the file; remaining gate is the same
bit-exact comparison, which should now pass, followed by the multi-pair run to see
whether the dequant saving survives the repack cost.

## Gate/up-only SoA also fails the bit gate (max_abs 15.2) - line closed for good

Implemented the narrow variant exactly as assessed: the SoA layer's gate/up comes
from ds4_mmq_iq2_xxs_moe_pair_soa (raw write-back), and the existing
moe_swiglu_weighted_f32_kernel + f32_to_f16_kernel + staged Q2 down stay in charge
(the epilogue conditions became use_fused_swiglu && !soa_gateup_ready). Build
clean.

Gate at 8K, same binary, branch off vs on:
  off 183.90 tps, on 175.40 tps, decode 12.24 -> 6.47 tps
  frontier logits exact=False, max_abs 15.21   (worse than the fused variant's 2.84)

So the SoA gate/up values themselves do not match this engine's gate/up, i.e. the
fault is in the artifact/loader contract rather than in the mid boundary. Likely
candidates: the artifact's expert stride (soa_blocks = 384*2304*20 is the whole
tensor, not per expert), or the ids the pair_soa entry receives versus the staged
table order. Diagnosing that needs the loader-side block math checked against the
producer's layout - work I cannot complete in this session, and not worth guessing
at.

Reverted rocm/ds4_rocm_moe_launch.cuh (branch returns to the previously committed
fused variant, which is itself unpromoted and inert). ds4-bench rebuilds with zero
errors and no sources are modified. The SoA line is closed: three variants
(fused, gate/up-only) all fail the bit gate by orders more than the 2-4% upside.

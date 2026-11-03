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

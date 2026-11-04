# V4.1 Strix Halo — session summary (state, evidence, next steps)

## Promoted (default-on, verified)
| change | commit | effect (32K, 3 interleaved pairs) |
|---|---|---|
| fused-epilogue compact MMQ tile | 37cc58c | prefill/append |
| q_a+kv projection fusion | 6cd2dc8 | decode |
| unrolled blocks4 loop | 6cd2dc8 | decode |
| **verified totals** | | **prefill +2.5% (sd 0.5), 4K append +2.3% (sd 0.6), decode +5–9%** |

Frontier logits are **bit-identical** at 32768/36864/110592/114688 in every
promoted measurement, with identical argmax and identical 128-token generation at
8K. 110K figures rest on three independent pairs (+2.4…3.0% prefill, +3.4…4.5%
append, +5.2…8.9% decode).

## Measurement protocol (mandatory here)
`rocm-smi` shows `Performance Level: auto` with 600/1100/2900 MHz and 1700–1850 MHz
observed. Two "disjoint +2.5%" wins (Q8 PAD=8, non-packed indexer) failed to
reproduce, so:
- >= 3 interleaved pairs **within one session**, plus a second session;
- fixed warm-up before every arm and per-arm clock samples;
- single-pair disjoint wins <= 3% are not evidence;
- bit-equality gate for anything that must not change arithmetic.

## Closed lines (do not retry)
- **Decoder suffix** (`findings/codex-suffix-decision.md`): row-count pruning makes
  the fused MMQ gate/up rounding row-block dependent (1–2 F32 ULP), intrinsic and
  unfixable without recomputing the rows you saved. Default-off.
- **Aligned-SoA IQ2 artifacts** (this session, three variants): on-disk sidecar,
  fused in-process, gate/up-only in-process — all fail the bit gate
  (max_abs 2.84 and 15.21) and are slower once the repack is charged. The failure
  is in the artifact/loader contract (whole-tensor `soa_blocks` vs per-expert use,
  or the ids handed to `ds4_mmq_iq2_xxs_moe_pair_soa`), not in the mid boundary.
  The producer (`cuda/mmq/ds4_repack` + `rocm/ds4_rocm_repack.hip`) is wired and
  builds but the branch is inert and unpromoted.
- Knob sweeps, all measured neutral or negative: gate/down waves 2/4/8, down
  mt=8, down preload6, attention low-A/pad8 toggles, MMQ x-max 96/128 (slower and
  numerically different), prefill chunk 4096/8192 (bit-identical but neutral),
  cache beyond the prefill reserve (+2.4%, inside noise), selected-read chunks
  (inert above 18/jobs), copy batching (only 16% of copies are ours), slot-sum
  fusion (0.30% ceiling), half2 epilogue stores (-2.4%), decode-cache growth after
  prefill (promoted then demoted: the +5–10% was DVFS drift).
- `--ssd-streaming-preload-experts` is rejected for V4.1; `DS41_INDEX_BATCH` is a
  graph-allocation unit, so larger indexer batches need memory-planning work and
  measured -10% anyway.

## Where the time goes (32K prefill, kernel trace)
MMQ compact 18.3%, down hotlist 13.3%, Q8 rowtile <128,8> 8.6%, hipBLASLt indexer
GEMM 7.5%, attention 6.8%, grouped Q8 5.7%, indexer warp8 5.2%, bf16 3.9%.
Decode: 87 ms/token = 65.7 busy + 21.4 gap, 2408 dispatches/token, gap p50 2.2 us
(two large stalls dominate, not launch count).

## Next steps with real upside (all need kernel work, not knobs)
1. MMQ IQ2 dequant instruction count (18.3% block): the SoA loader's `ksigns64` LUT
   is the shape of it, but it needs the artifact contract debugged first.
2. Indexer scoring ~12.7% (rowtile + warp8 + Cijk): Cijk runs 7552 calls at
   ~1.77 ms for 32-query tiles (~38 GFLOP/s); larger batches change GEMM
   accumulation order, so they need the shape-noise gate plus a top-k stability
   check. Row batching alone measured -10%.
3. Deployment: host `ds4-server` is rebuilt (58bda5ba) and contains the promoted
   paths, but the cockpit image still serves an older binary. Rebuilding the image
   and repeating one API-level 32K pair is the outstanding verification; the
   container was deliberately left untouched.

## Environment notes
- 128 GB machine, 121.73 GiB usable; a 104 GB cache request is refused by the
  planner (`needs 126.92 GiB`), so 92 GB + reclaim is the ceiling.
- `ds4-bench` and `ds4-server` both build from current sources; `cuda/mmq/
  ds4_repack.rocm.o` and `rocm/ds4_rocm_repack.rocm.o` are in `ROCM_MMQ_OBJS`.
- TCC/DRAM counters are unavailable on gfx1151 (only GRBM_GUI_ACTIVE and
  SQ_WAVES), so FETCH_SIZE measures L2 sector requests, not DRAM.

## 64K coverage added (two interleaved pairs, bit-exact)

Frontier 65536 plus a 4K append (69632), pre-session binary vs current defaults:

  64K prefill   +3.0% / +3.0%  (162.56/162.77 -> 167.37/167.64 tps)  mean +3.0%
  4K append     +5.3% / +1.8%                                         mean +3.6%
  frontier logits exact=True (max_abs 0) at both frontiers, both pairs

That closes the context-coverage gap: the promoted set is now measured at 32K
(+2.5% prefill, +2.3% append, 3 pairs), 64K (+3.0% / +3.6%, 2 pairs) and 110K
(+2.4..3.0% / +3.4..4.5%, 3 pairs), with decode +5-9%, every arm bit-identical.

## Final state check (after the 64K confirmation)

Both shipped binaries carry the promoted paths: `ds4-bench` (3dc11904, 19:33) and
`ds4-server` (58bda5ba, 17:34) each contain the fused-SwiGLU MMQ message, the Q8
blocks4 unrolled-loop message and the decode-cache-grow message, i.e. the same
code state the benchmarks measured. Only the three pre-existing dirty files
(EXPERIMENTS.md, ds4_bench.c, ds4_help.c) remain modified, untouched by this work.

Context coverage is now complete for the promoted set: 32K (+2.5% prefill, +2.3%
append, 3 pairs), 64K (+3.0%, +3.6%, 2 pairs), 110K (+2.4..3.0%, +3.4..4.5%,
3 pairs), decode +5-9%, all bit-identical.

Remaining work, in value order:
1. 110K decode multi-pair - the only figure still resting on single pairs per
   session (two pairs would need ~54 min, inside the supervisor limit but tight).
2. Deployment: rebuild the cockpit image from the host ds4-server and repeat one
   API-level 32K pair; the container has been deliberately left untouched.
3. Kernel work under the shape-noise gate: MMQ IQ2 dequant instruction count
   (18.3% of prefill) and indexer GEMM tiling (12.7%, needs a top-k stability
   check because larger batches measured -10%).

## 110K decode multi-pair (two interleaved pairs, bit-exact)

Frontier 110592, gen 32, pre-session binary vs current defaults, 15 s warm-up:

  pair1  prefill 140.39 -> 143.04 tps (+1.9%),  decode 10.360 -> 10.680 tps (+3.1%)
  pair2  prefill 139.68 -> 143.37 tps (+2.6%),  decode 10.620 -> 11.140 tps (+4.9%)
  frontier logits exact=True (max_abs 0) in both pairs

Together with the three earlier single pairs, 110K decode is positive in five of
five measurements (+3.1, +4.9, +5.2, +6.0, +8.9%) and prefill in five of five
(+1.9, +2.6, +2.4, +2.9, +3.0%), always bit-identical. This closes the last figure
that rested on single pairs per session.

## Generalisation gate: bit-identical on an unseen prompt

Every tuning measurement used speed-bench/promessi_sposi.txt, so the promoted set
was re-checked against a prompt built from repository sources (698 827 bytes of
README/AGENTS/Makefile/headers/kernels), never used for tuning:

  frontier logits exact=True, max_abs 0, argmax identical
  generated text identical (8 greedy tokens)

So the promoted changes are input-independent at the bit level, not just on the
tuning prompt. The timings from that run (prefill 164.74 -> 175.85 tps, decode
11.34 -> 7.62 tps) are single runs with only 8 generated tokens, so decode is
dominated by first-token cost and neither figure is evidence - the multi-pair
measurements at gen 32-64 remain the numbers of record.

## Release completeness: the interactive CLI was stale

`ds4` (the interactive CLI) was still the Nov 1 binary and did not contain the
promoted paths, while `ds4-bench` and `ds4-server` did. Rebuilt it from the current
sources with the same ROCm object set; all three now carry the fused-SwiGLU MMQ
and Q8 blocks4 unrolled-loop paths:

  ds4        unrolled 1, fused 1   (rebuilt 21:07)
  ds4-server unrolled 1, fused 1
  ds4-bench  unrolled 1, fused 1

So no shipped entry point remains on the old code. Nothing changed in the source
tree - this was a build-state gap, not a code gap.

## Rebuilt binaries smoke-checked

The freshly rebuilt entry points execute: `./ds4 --help` and
`./ds4-server --help` both print their usage and exit 0, so the CLI/server relink
(which uses a different object set from ds4-bench: ds4_cli.o + linenoise, and
ds4_server.o + ds4_kvstore.o + rax) is healthy, not just present in the file list.

With that, the release surface is complete and verified at the binary level:
ds4, ds4-server and ds4-bench all build from the current sources, all contain the
promoted paths, and all start correctly.

## Reproducibility caveat: the reference binary is a build artifact, not a commit

Every promoted A/B compared against `/tmp/ds4-bench-base`
(sha256 17995cba…, 2026-11-03 08:59). That binary was captured from the working
tree at the start of the jlimit work, i.e. it matches **no commit**: it postdates
41bb7cf (which already contains the session's earlier diagnostic changes) and
predates the promoted changes. After a reboot or /tmp cleanup the reference is
gone and the +2.5%/+3.0%/+2.4% figures cannot be re-derived with that payload.

To re-anchor future comparisons on versioned code instead, rebuild a reference
from a worktree at the pre-promotion commit and capture it under findings/:

  git worktree add /tmp/base-41bb7cf 41bb7cf
  cd /tmp/base-41bb7cf && make -j8 ds4-bench \
    CORE_OBJS="ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o \
      ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o ds4_engram.o \
      cuda/mmq/ds4_ggml_stubs.rocm.o cuda/mmq/ds4_mmq.rocm.o cuda/mmq/quantize.rocm.o \
      cuda/mmq/mmid.rocm.o cuda/mmq/mmvq.rocm.o cuda/mmq/d2r_stubs.rocm.o \
      rocm/v41_gate_up/tiles/descriptors.o rocm/v41_gate_up/tiles/twin.o \
      rocm/v41_gate_up/tiles/wide.o rocm/v41_gate_up/runtime.o \
      rocm/halo/shared_gu/kernel.o rocm/halo/q2/runtime.o rocm/halo/q2/prepared.o \
      rocm/halo/moe/runtime.o rocm/halo/moe/public/{wide,twin,descriptors}.o \
      rocm/halo/moe/fresh/{wide,twin,descriptors}.o" \
    CFLAGS="-O3 -ffast-math -g -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE \
      -fno-finite-math-only -fPIC -DDS4_ROCM_BUILD" \
    DS4_LINK="/usr/bin/hipcc -std=c++20 -O3 -ffast-math -g -fno-finite-math-only \
      -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument \
      --offload-arch=gfx1151" \
    DS4_LINK_LIBS="-lm -pthread -lhipblas -lhipblaslt -lrocblas"
  cp /tmp/base-41bb7cf/ds4-bench findings/ref-bin-41bb7cf

Note that 41bb7cf is *pre-session*, so a comparison against it also includes the
session's earlier diagnostic-only changes; those are inert (env-gated) and the
measured effect is therefore attributable to the promoted three.

Future payloads should take the reference binary as an argument rather than
hardcoding /tmp/ds4-bench-base.

## IQ2 dequant: ksigns64 LUT replaces the popcount sign trick (+1.4% prefill)

load_tiles_iq2_xxs built the two sign-mask words per 8 weights with
unpack_ksigns (popcount + multiply) followed by two __vcmpne4. The 128-entry
ksigns64 table that the aligned-SoA loader already uses encodes exactly those
masks, so one LUT load replaces all four operations.

Bit gate: frontier logits exact=True (max_abs 0) in all six arms - the values are
identical by construction, and the measurement confirms it.

Performance, three interleaved pairs at 32K against the pre-LUT binary:
  32K prefill   +2.1 / +1.0 / +1.2  -> mean +1.4%  (3/3 positive)
  4K append     +3.4 / +1.8 / +1.6  -> mean +2.3%  (3/3 positive)

This is on top of the already-promoted set, so the cumulative position at 32K is
roughly +3.9% prefill and +4.6% append versus the pre-session binary. One session
only so far; a second session is the remaining confirmation for the performance
claim (correctness is already established by bit-equality).

### ksigns64 LUT - second session confirms

Three interleaved pairs in a second session, pre-LUT binary vs LUT binary:
  32K prefill  +0.4 / +2.2 / +1.1  -> mean +1.2% (3/3 positive)
  4K append    +0.7 / +1.8 / +0.8  -> mean +1.1% (3/3 positive)
  frontier logits exact=True in all six arms

Both sessions agree (prefill +1.4% then +1.2%, append +2.3% then +1.1%), all pairs
positive, logits identical throughout. Confirmed: ~+1.3% prefill and ~+1.7%
append, so the cumulative position at 32K is ~+3.8% prefill and ~+4.0% append
versus the pre-session binary, plus +5-7% decode.

### Next candidate in the same class (decode side)

The V4.1 decode IQ2 path has its own copy of the popcount trick:
dev_unpack_iq2_signs(v) in rocm/ds4_rocm_moe.cuh:39 is byte-for-byte the same
computation (popc, xor, multiply by 0x01010101) followed by two __vcmpne4, used by
dev_iq2_dp4a_8 and dev_iq2_i8x8_lut - i.e. on the decode gate/up matvec. The same
argument applies: a 128-entry mask table (cuda_ksigns_iq2xs is already used in this
file for the other path) removes the popcount/multiply/compares from the decode
inner loop, and is value-identical, so the bit gate applies unchanged.

Note the ggml IQ2_XS loader at mmq.cuh:3163 has the same pattern but IQ2_XS is not
this model's quantisation (gate/up are IQ2_XXS), so it would buy nothing here.

### ksigns64 equivalence proven by exhaustion (all 128 indices)

Host check of the table against the arithmetic it replaces, for every 7-bit index:
  entry low  32 bits == __vcmpne4(unpack_ksigns(i) & 0x08040201, 0)
  entry high 32 bits == __vcmpne4(unpack_ksigns(i) & 0x80402010, 0)
  mismatches: 0 / 128

So the committed mmq.cuh LUT change is not merely empirically identical on our
prompts - it is identical for every possible sign index, which is why the bit gate
passed in six arms across two sessions.

Decode-side candidate needs one more fact before it can be written safely: the
V4.1 decode path feeds `cuda_ksigns_iq2xs[sign]` into dev_iq2_dp4a_8, which then
applies dev_unpack_iq2_signs to that value. That is, the table there is indexed by
a 7-bit pattern and its *entries* are then run through the popcount trick, so the
two formulations are not interchangeable without establishing what cuda_ksigns_iq2xs
actually holds. The equivalent swap is only safe once that is checked the same way
(exhaustive comparison over the table), which is the next step for that site.

### Decode-side equivalence established (all 128 indices), swap not yet written

Exact semantics of the decode composition, verified from the shipped tables:
`dev_iq2_dp4a_8(..., cuda_ksigns_iq2xs[idx], ...)` passes the *byte entry*
(ksigns_iq2xs is a 128-entry uint8 permutation, not the identity), and the callee
then applies dev_unpack_iq2_signs to that byte and derives the two masks with
__vcmpne4.

Exhaustive check over all 128 indices:
  vcmpne4(unpack_ksigns(ksigns_iq2xs[i]) & 0x08040201, 0) == ksigns64[i] low  word
  vcmpne4(unpack_ksigns(ksigns_iq2xs[i]) & 0x80402010, 0) == ksigns64[i] high word
  mismatches: 0 / 128

So the whole byte-table + popcount + two-compare chain is exactly `ksigns64[i]`,
and the decode swap is safe to write - but it requires changing the *callers* to
pass the 7-bit index rather than the permuted byte, at three sites
(dev_iq2_pair_16 lines 68-69 and the dp4a loop at line 981, plus the
dev_iq2_i8x8_lut users). That is a multi-site edit on the decode hot path and was
not started, because it cannot be completed and gated with the remaining budget:
an unverified decode path would be worse than the small gain it buys.

## Decode-side swap: blocked on table visibility, not on the maths

The equivalence is proven (0/128 mismatches), but the swap could not be written
this session because of a concrete missing fact: `rocm/ds4_rocm_moe.cuh` uses the
*byte* tables (`cuda_ksigns_iq2xs`, `cuda_iq2xxs_grid`), while the mask table
`ksigns64` lives in `cuda/mmq/ggml-common.h` - which is compiled into the
`cuda/mmq` TUs and is not obviously included by the ROCm TU that builds
`ds4_rocm.cu`. The definitions of the `cuda_`-prefixed tables were not located
within the search budget either, so whether `ksigns64` is visible there is
unverified.

Two ways to unblock, both mechanical:
1. include `cuda/mmq/ggml-common.h` in the ROCm TU (it guards itself with
   GGML_COMMON_IMPL and only defines static const tables), or
2. emit a 128-entry `uint2` mask table next to the existing byte table in
   `rocm/ds4_rocm_moe.cuh`, generated once from the byte table and verified against
   the 0/128 equivalence above.

Either way the call sites then pass the 7-bit index instead of the permuted byte
(dev_iq2_pair_16 lines 68-69, the dp4a loop at 981, dev_iq2_i8x8_lut users), after
which the normal bit gate and multi-pair run apply. Nothing was edited: an
unverified change to the decode hot path would be worse than the 1-2% it buys.

## Decode IQ2 sign masks via a local table (bit-exact; decode +3.3% in session 1)

ksigns64 is not visible in the ROCm TU (`cuda/mmq/ggml-common.h` is not on its
include path), so the 128-entry uint2 mask table was emitted beside the existing
byte table in rocm/ds4_rocm_moe.cuh, generated from ksigns_iq2xs with the proven
popcount trick and verified equal to ksigns64 for all 128 entries. dev_iq2_dp4a_8
now takes the 7-bit index and reads that table; dev_dot_iq2_pair_16 and
v41_iq2_pair32 pass the raw index instead of the permuted byte. dev_iq2_i8x8_lut
and its byte-table users were deliberately left unchanged.

Bit gate: frontier logits exact=True (max_abs 0) in all three pairs - which also
confirms that the dev_dot_iq2_pair_16 callers really do pass indices.

Performance, 8K gen 64, three interleaved pairs, pre-change binary vs new:
  decode   +8.5 / -0.2 / +1.7  -> mean +3.3% (2/3 positive, median +1.7%)
  prefill  +1.5 / -0.7 / -0.3  -> mean +0.2% (neutral, as expected)

So the decode effect is probably +1..3%: positive but noisier than the prefill LUT
(which was 3/3 in both sessions). A second session is required before treating the
decode figure as established; correctness is already settled by bit-equality.

### Decode sign-mask LUT - second session does NOT confirm the gain

  session 1  decode +8.5 / -0.2 / +1.7  -> mean +3.3% (2/3), median +1.7%
  session 2  decode +0.7 / +0.6 / -0.3  -> mean +0.3% (2/3), median +0.6%
             prefill -0.9 / -0.2 / +0.1 -> mean -0.3% (neutral)
  frontier logits exact=True in all six arms across both sessions

So the decode-side swap is bit-identical (proven by exhaustion and gated twice) but
its performance benefit is NOT established: the two sessions bracket +0.3..3.3%
with one negative pair each, i.e. it sits at the noise floor. Compare the prefill
LUT, which was 6/6 pairs positive across two sessions (+1.4% then +1.2%).

Decision: keep the change (bit-identical, prefill-neutral, so it cannot regress
correctness or prefill) but do not claim a decode gain from it - the promoted
decode figure stays +5-7%.

## Decode attention: the concrete target is the indexer key read volume

Inspection of the decode attention path (ds41_attention, and the indexer dispatch in
rocm/ds4_rocm_v41.cuh) narrows the 57%-of-token-time attention block to one
dominant, measurable cost: every decode token re-reads the whole indexer key cache.

g->index_cache[] is allocated as `cap * 128 * sizeof(float)` per level (ds4.c:40680),
i.e. F32, 512 B per compressed key, and ds4_gpu_dsv41_indexer_scores_batch walks all
n_comp keys for the single query row each token:

  ctx  32K  ratio 1 -> 16.8 MB/layer ->  671 MB/token over 40 layers
  ctx  32K  ratio 4 ->  4.2 MB/layer ->  168 MB/token
  ctx 110K  ratio 1 -> 56.6 MB/layer -> 2267 MB/token
  ctx 110K  ratio 4 -> 14.2 MB/layer ->  566 MB/token

At ~11 tps that is 2-7 GB/s of pure indexer-key traffic, i.e. a large share of the
attention block and of total decode bandwidth. The activations are already
quantised elsewhere (index_q is FP4 E8M0 before scoring), so the keys are the odd
one out.

Concrete candidate, correctness-preserving in structure: store index_cache in a
narrower type (F16 halves it, FP8_E4M3 quarters it) and have the indexer score
kernels read that, dequantising in-register. Same dot products, ~2-4x fewer bytes
per token, no extra passes. It does change the key values' precision, so it needs
the shape-noise gate rather than bit-equality, plus a top-k stability check: the
selected compressed indices must not move, otherwise routing changes downstream.

Measured prerequisite (cheap): confirm the read volume with rocprof FETCH_SIZE on
v41_indexer_batch_warp8_kernel at 32K decode, and check the score deltas that an
F16 key cache would introduce before committing to the change.

## The index cache is FP4-valued but stored as F32 (7.5x avoidable traffic)

Offline analysis of the captured layer-20 indexer cache
(attention_core_index_20_index_cache3_8065_8192.raw, 127 keys x 128 values):

  distinct values: 23, and they are exactly {0, +-0.0625, 0.125, 0.1875, 0.25,
  0.375, 0.5, 0.75, 1, 1.5, 2, 3} - i.e. FP4 (E2M1) magnitudes times a per-block
  power-of-two E8M0 scale, which is the engine's own DS4_V41_FP4_E8M0 format
  (index_q is already quantised that way before scoring)
  every value is an exact multiple of 1/16; max |v| = 3
  F16-rounding the keys reproduces every value exactly (0/16256 changed), and the
  F32-vs-F16 score delta over all 32 heads is 0 with 0/127 rank moves

So the keys carry about 4 bits of information per value while occupying 32. Packed
FP4_E8M0 (4 bits + one scale byte per 32 values) is ~7.5x smaller and reconstructs
the same values, hence the same dot products and the same selected indices - a
bit-exact change, not a shape-noise one.

Traffic effect at the current rates (40 layers, whole key set per token):
  32K  ratio 1: 671 MB -> 89 MB     110K ratio 1: 2267 MB -> 302 MB
  32K  ratio 4: 168 MB -> 22 MB     110K ratio 4:  566 MB ->  75 MB

Implementation path: store g->index_cache[] in the FP4_E8M0 layout the engine
already produces (ds4_gpu_dsv41_quantize with DS4_V41_FP4_E8M0), have
ds4_gpu_dsv41_indexer_scores_batch / v41_indexer_batch_warp8_kernel dequantise in
registers, and keep the same dot-product order so the scores are identical. Gate:
bit-identical frontier logits first (the values are unchanged by construction),
then the usual multi-pair run with decode as the metric.

Caveat to close first: the sample is one layer of one capture (127 keys). Confirm
the codebook holds across layers and positions - e.g. two more layers from the same
capture set or a fresh diag run - before implementing.

### FP4 codebook verification across captures: 10/10 clean

All available index_cache captures (five sessions x both arms, layer 20,
positions 8065..8192, 16 256 values each):

  values 16256, distinct 23, values outside the FP4(E2M1) x E8M0 codebook: 0,
  max |v| = 3 - identical in every capture

So the FP4-valued property is stable across sessions and across the two arms, not a
one-off. No exceptions to report. Coverage limit worth stating: every capture is
layer 20 (the diagnostic probe is layer-scoped) and the same level (index_cache3),
so layers 0..19/21..39 and levels 0..2 are inferred rather than sampled - but the
values are produced by one shared quantise path (the same one that already emits
DS4_V41_FP4_E8M0 for index_q), so the property belongs to the format rather than to
a layer.

Implementation is the next step and is a real change: repack g->index_cache[] into
the FP4_E8M0 layout, update the level publish path that writes it
(ds4.c:41045 copies 128 floats per key), and teach
ds4_gpu_dsv41_indexer_scores_batch / v41_indexer_batch_warp8_kernel to dequantise in
registers while keeping the dot-product order, then gate on bit-identical logits.

## index_cache FP4 repack: exact diff identified, not started

Inspection resolves the whole shape of the change:

Allocation (ds4.c:40679-40680)
  cap = ctx/(i<3?2:1)+1;  index_cache[i] = alloc(cap * 128 * sizeof(float))   -> F32, 512 B/key
Publish (ds4.c:41044-41046), per new compressed key:
  ds4_gpu_dsv41_quantize(g->index_k, 128, 1, DS4_V41_FP4_E8M0)   // FP4 quantise, still F32 storage
  ds4_gpu_tensor_copy(g->index_cache[owner], (n_comp-1)*128*4, g->index_k, 0, 128*4)
Score (rocm/ds4_rocm_v41.cuh: ds4_gpu_dsv41_indexer_scores_batch -> v41_indexer_batch_warp8_kernel)
  reads keys as `const float *`, 128 floats per key.

So the values are already quantised to FP4 and then copied as 32-bit floats: the
publish path writes 512 B where 65 B (one E8M0 scale byte + 64 B of nibbles) carry
the same information.

Minimal bit-exact change set (five pieces):
1. allocate index_cache[i] as cap * 65 bytes instead of cap * 512;
2. add a "pack FP4_E8M0" kernel that writes the packed key (same scale/value
   arithmetic the current in-place F32 quantiser uses, so the decoded values are
   bit-identical);
3. publish: run that kernel from g->index_k into the cache slot, replacing the
   quantise + 128*4 copy;
4. add a packed-key variant of the warp8 scoring kernel that loads the scale byte
   and nibbles and dequantises into registers, keeping the per-head dot order;
5. dispatch the packed variant wherever index_cache is scored.
Gate: frontier logits bit-identical (the decoded values are unchanged), then the
multi-pair run with decode as the metric.

Not started: this is a new quantiser kernel plus a new scoring kernel on the decode
hot path, and it cannot be written, built and gated in the remaining budget. A
half-finished variant here would risk the indexer's correctness, which propagates
into routing - clearly worse than deferring it.

## FP4 helpers inspected: the pack kernel is now fully specified

Helpers that already exist (rocm/ds4_rocm_norm_rope.cuh:332-357):
  dsv4_e2m1fn_value_dev(i)  - the 8 E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}
  dsv4_e2m1fn_dequant_dev(x) - round-to-nearest of those magnitudes with sign,
    implemented as a search that already determines the code index `best`
And the current key quantiser (rocm/ds4_rocm_v41.cuh:61-86, v41_quantize_kernel with
DS4_V41_FP4_E8M0) does, per 32-value block:
  scale = pow2_ceil(max(amax, tiny) / 6)          -> a power of two
  stored = dequant_e2m1(|v| / scale) * scale       -> F32, sign preserved

Two consequences that make the repack cheap and bit-exact:
1. the scale is always a power of two, so it needs one E8M0 byte per 32 values;
2. the stored value is exactly value(code) * scale, so packing the code (3 magnitude
   bits + 1 sign bit) and decoding with the same multiply reproduces the current F32
   bits exactly - no re-rounding and no tolerance.

So the pack kernel is: same per-block scale computation, code = the `best` index the
dequant search already finds, emit nibble + one scale byte; the decode in the scorer
is value(code) * 2^(exp-127). Storage 65 B/key instead of 512 B.

Still not implemented: it is a new kernel pair on the decode hot path and the
remaining budget cannot carry it through build, bit gate, multi-pair and a top-k
stability check. Everything needed is now written down: allocation cap*65, the pack
kernel semantics above, the publish replacement, a packed-key warp8 scoring variant
keeping the per-head dot order, and the dispatch.

## Correction: the index_cache repack touches six consumers, not one path

Tracing every reference to g->index_cache[] in ds4.c shows the change is wider than
the five-piece sketch recorded earlier:

  writers
    41045  compress path: quantise index_k to FP4_E8M0 then copy 128 floats
    41645  prefill reuse: copy a range of keys from another cache level
  readers
    41111  ds4_gpu_dsv41_indexer_scores_one      (decode, single row)
    41117  ds4_gpu_dsv41_indexer_scores_batch    (small batches)
    41694  ds4_gpu_dsv41_indexer_scores_batch    (prefill tiles)
    41692  ds4_gpu_dsv41_indexer_scores_packed   (packed-index path)
    41682  ds4_gpu_dsv41_indexer_pack            (builds index_packed FROM the cache)

So a dtype change must convert two writers, three score entry points and the
index-packed builder together - i.e. it is a coordinated refactor of the indexer's
storage layer, not a local edit. Landing it piecemeal would leave the indexer
reading one layout while another writer wrote the other, which is precisely the
failure mode that produces garbage selections rather than a measurable slowdown.

Not implemented, deliberately: with the gate unavailable in the remaining budget a
partial conversion is the one outcome that could silently corrupt routing. The
motivating measurement stands (keys carry ~4 bits of information in 32; 671 -> 89
MB/token at 32K ratio 1, 2267 -> 302 MB/token at 110K), and the recipe is fully
written down including this corrected scope.

## FP4 index-cache pack/decode transform: verified bitwise on all captures

tools/verify-index-fp4-pack.cu implements the pack/decode pair for the measured
layout - 4 x E8M0 scale bytes (block size 32, matching v41_quantize_kernel's
FP4_E8M0 path) + 64 nibble bytes holding 3 magnitude bits and 1 sign bit per value,
68 B per 128-value key - and checks it against every captured cache:

  all 10 captures, 127 keys x 128 values each:
  packed_bytes=8636 vs f32_bytes=65024  bitwise_mismatches = 0

Two real bugs were found and fixed by that check, both worth recording because they
are the traps for this work:
1. placing the four scale bytes at b[0..3] while nibbles started at b[1] made them
   overlap - 1701 mismatches, every one a value from block 0/1/2;
2. signed zero: testing `v < 0.0f` misses -0.0, so nine of the ten captures showed
   1002 mismatches that were all `-0 != 0`. The sign bit must be tested directly,
   which is exactly why v41_quantize_kernel re-applies
   `(bits & 0x7fffffff) | (value_bits & 0x80000000)` after quantising.

So the transform itself is now proven exact (not merely close), and the remaining
work is the plumbing across the two writers and five readers already mapped.

## FP4 index keys: correct, but the optimization cannot pay off (measured)

Implemented and gated: 68 B/key packed index cache (4 x E8M0 per-32 scale + 64 nibble
bytes), written by both publishers, read by the decode scorer and both prefill batch
scorers behind DS4_ROCM_V41_INDEX_FP4 (default-off).

Correctness: frontier logits bit-identical at 8192 and 114688 (0/129280 both), selected
indices byte-identical over 128 records of top-512, on-device verifier 0 mismatches.

Performance, six interleaved 32K gen-64 off/on pairs over two sessions:
  +3.0, +3.6, +1.1 (session 1, mean +2.5 %, all positive)
  +3.4, +1.6, -2.2 (session 2, mean +0.9 %, sign flips)
  combined mean +1.75 %, median +2.30 %, sd 2.18

Verdict: not reproducible. Session 1 looked like a real +2.5 %, session 2 refuted it,
so the effect is indistinguishable from DVFS/measurement noise. Do not promote.

Why the ceiling is this low, and why more implementation work cannot fix it: at 32K the
indexer reads 16384 keys x 512 B x 16 layers = 134 MB per token, but decode takes ~90 ms
per token, so that is only ~1.4 GB/s - far below HBM bandwidth, and mostly L2-resident.
Replacing 512 B with 68 B removes 116 MB/token, i.e. ~1.2 GB/s of demand on a device with
hundreds of GB/s. The original motivation (671 MB/token at 32K, 2267 MB/token at 110K)
was a real figure misinterpreted as a bottleneck: spread over a ~90 ms token it is a
bandwidth fraction, not a limit. Even a perfect dequant-in-GEMM conversion of the last
F32 reader (the wide-GEMM branch) would chase the same non-bottleneck.

Consequences:
- the F32 index cache must NOT be dropped on the strength of this work; with the packed
  copy added, write traffic is currently 13 % higher for no measured gain;
- the packed path stays opt-in and unshipped; it is kept only as a correct, gated
  building block in case the indexer ever becomes bandwidth-bound (e.g. much longer
  contexts on slower memory);
- no further effort goes into this line. The next speed work should target where the
  90 ms actually goes, not the index cache.

Instrumentation notes for whoever revisits this:
- frontier logits are captured at the end of prefill, so they cannot verify a decode-only
  change; selected-index equality was the load-bearing check there;
- rocm-smi --showclkfrq prints a "supported frequencies" table, not the live clock, and
  warns the GPU is in a low-power state when sampled outside a run - the per-arm clock
  capture in bench-fp4-index-decode-pairs.sh is therefore not yet trustworthy; prefill
  stability (183.03/183.28/183.07 tps) was used as the stability proxy instead.

## 32K decode apportionment (DS4_V41_DECODE_STAGE_PROFILE, 40 layers x 16 tokens)

  moe_partial       861.3 ms  26.2%     attention      600.6 ms  18.3%
  moe_routed        658.7 ms  20.1%     attn_output    251.8 ms   7.7%
  attn_project      191.6 ms   5.8%     attn_q_b       142.4 ms   4.3%
  moe_shared        124.6 ms   3.8%     before_attn    110.2 ms   3.4%
  attn_decode        64.4 ms   2.0%     attn_select     54.7 ms   1.7%
  after_attention    48.6 ms   1.5%     attn_q_a        47.3 ms   1.4%
  moe_route          46.7 ms   1.4%     moe_stream_begin 26.5 ms  0.8%
  attn_rope_window   13.4 ms   0.4%     attn_gather     12.3 ms   0.4%

Total instrumented 3285 ms for 16 tokens; the per-stage synchronize inflates wall
time (real decode ~90 ms/token), so read the shares, not the absolute ms.

CORRECTION (same log, re-parsed): the stage marks are nested, so the first reading of
this table double counted. moe_partial is the parent of moe_route/moe_stream_begin/
moe_shared/moe_routed (53.8 vs 53.5 ms per token across 40 layers - verified equal), and
attention is a parent of the attn_* marks. Real shares per token: MoE ~26 % (routed 20.1,
shared 3.8, route 1.4, stream 0.8), attention parent ~18 %, plus attn_q_a+q_b+project+rope
~12 % and attn_output 7.7 %. So MoE is the largest single bucket but NOT 52 %, and no
child may be added to its parent. bench-decode-apportion-32k.sh now reports parents and
children separately to make the mistake unrepeatable.

Two conclusions that redirect the work:
- MoE routed expert matmuls (20.1 %) are the largest single bucket; attention parent
  ~18 % with a further ~12 % in projections.
- attn_select, the indexer select that the whole packed-FP4 exercise optimised, is
  1.7 % of the token. Independent confirmation that the repack could never have paid
  off, from a different direction than the six-pair timing result.

Next step: moe_partial is a bucket (router matmul + select, stream begin, shared expert,
routed expert matmuls), not a single kernel, so instrument inside it before optimising
anything - gate/up vs down vs stream-wait - rather than guessing which of the 26 % is real.

## Rejected: partial gate/up waiting for streamed experts (DS4_ROCM_V41_ENABLE_PARTIAL_GATEUP)

The internal event profiler (DS4_ROCM_MOE_DECODE_PROFILE) finally located the real decode
stall, after my own moe_stream_wait probe misled me: draining right after the prefetch is
issued measures enqueue cost (0.6 ms), not I/O latency. The wait surfaces at point of use:

  gate_resident        111.3 ms  26% of routed MoE
  split_finish_missing 220.5 ms  51%   wait for streamed experts before computing
  down                  69.8 ms  16%
  xquant+mid_quant       9.3 ms   2%

So decode IS partly SSD-starved (~12-17 % of the token), and the previous "streaming is a
non-issue" conclusion was a sampling artifact.

The obvious fix already existed as an opt-in: wait only for gate/up, letting the down fetch
overlap gate/up math. Three interleaved 32K gen-64 pairs, same binary, banner-confirmed
(o1/o3 absent, p1/p3 present, so the arm really took the path):

  -3.3 %, +2.5 %, -1.9 %   mean -0.92 %, sign flips

Rejected. Mechanism: the stall is bounded by SSD delivery, not by scheduling. Overlapping
gate/up does not shorten it because the down weights are still needed afterwards, so total
I/O and the critical path are unchanged. Cutting this wait requires either fewer streamed
bytes, a higher resident hit rate, or overlap across layers (prefetch layer L+1 during layer
L's attention) - not a finer-grained wait inside one layer.

Keep the default (env unset = wait for everything); no code change.

Next measurement before the next attempt: miss rate and bytes streamed per token
(the logs show resident_mask=0x3f, i.e. 6 of 8 experts resident), because that decides
whether prefetch-across-layers can help at all or whether the fix must reduce bytes.

## Pre-attention route+fetch hoist: engages, correct, but the stall does not move

Implementation (opt-in DS4_ROCM_V41_PREFETCH_ROUTE, banner in the same patch): the router
and ds41_stream_selected_begin run after the layer norm and before attention, so the
per-layer expert delivery could overlap the attention stage; moe_partial skips its own
router/fetch exactly once via g->route_prefetched.

Verification, all six arms completed:
  banner: 0 in every off arm, 1 in every on arm (engagement proven, not assumed)
  generated text: IDENTICAL in all three pairs of both sessions (185 chars)
  frontier logits: exact in both sessions
  decode: +1.9/-0.2/+3.3 (session 1) and +1.7/-0.4/+0.9 (session 2)
          combined +1.20 % over 6 pairs, 4/6 positive, sd ~1.6

Verdict: not evidence. The mechanism check is what settles it - split_finish_missing went
from 2771 ms to 2733 ms (-1.4 %), i.e. the point-of-use wait did not shrink. If the fetch
had really overlapped attention (18 % of the token, ~15 ms), that stall should have
collapsed toward zero and decode should have gained ~15 %. It did not move at all.

Interpretation: issuing the fetch earlier does not make it overlap, because it is enqueued
into the same execution stream that attention then occupies; the reads are also
direct-returned, so there is no staging stage that could proceed independently. Hiding this
stall needs the fetch on an independent stream with the compute stream waiting on an event,
not a reordering of calls inside one layer.

Keep the flag opt-in and unshipped. Remaining per-layer-token budget for anyone continuing:
stall 352 us, gate_resident 175 us, down 109 us - the stall is still the largest single
item, and the fix has to be a real overlap primitive (separate stream + event), not more
reordering.

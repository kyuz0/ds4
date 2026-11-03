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

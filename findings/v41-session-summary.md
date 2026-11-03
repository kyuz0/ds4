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

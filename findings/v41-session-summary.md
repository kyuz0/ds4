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

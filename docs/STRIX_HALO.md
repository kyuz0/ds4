# AMD Strix Halo

[README](../README.md) | [Getting started](../README.md#start-here)

The reference system is a 128 GB Strix Halo with Radeon 8060S (`gfx1151`),
such as the Framework Desktop. The ROCm build uses the standard binary names
and selects the ROCm backend by default.

## Prerequisites

For a container setup, see the maintained
[ROCm toolbox](https://github.com/kyuz0/strix-halo-ds4-toolbox/blob/main/toolboxes/Dockerfile.rocm-10.0).
It can also be managed with
[AI Toolbox Cockpit](https://github.com/kyuz0/ai-toolbox-cockpit).

For a native Ubuntu build you need HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA,
and hipCUB development files. The Ubuntu 26.04 setup used these packages:

```sh
sudo apt-get update
sudo apt-get install -y hipcc rocminfo rocm-smi \
  libamdhip64-dev libhipblas-dev libhipblaslt-dev librocblas-dev \
  librocwmma-dev libhipcub-dev
sudo usermod -aG render,video "$USER"
```

Log out and back in after changing groups. `rocminfo` must report `gfx1151`
and be able to open `/dev/kfd` before DwarfStar can run.

Some packaged rocWMMA headers omit `rocwmma/internal/`. If compilation fails
there, install the complete headers matching your ROCm installation, or use
the container. Do not mix header versions as a general workaround.

## GPU-visible memory

Check the GPU-visible memory pool reported by `rocminfo`. Some 128 GB systems expose only about 62 GiB to the GPU. The tested 128 GB Fedora Linux Strix Halo system, running a recent kernel and ROCm 10.0, used these boot parameters:

```text
amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856
```

The GTT/TTM settings expose about 124 GiB to the GPU. An SSD expert-cache request such as `92GB` is fitted to that GPU-visible limit as well as available system RAM; a stock ~62 GiB pool can therefore yield a much smaller cache. `amd_iommu=off` was part of the tested setup, but is not required for GTT sizing and disables DMA isolation. Keep RAM available for the OS. See the [host configuration guide](https://strix-halo-toolboxes.com/#config) for Fedora, Ubuntu/Debian, and systemd-boot instructions.

## Build and run Flash

```sh
make strix-halo
./download_model.sh ds4f-q2
./ds4 --rocm
```

`make rocm` is an alias. Use the current 0731 Q2 download for a first run;
larger mixed and Q4 models have substantially higher memory requirements.
Flash's ROCm resident and pipeline paths should not be confused with the GLM
SSD-streaming path.

## DeepSeek V4.1 Flash

- ROCm 10.0 supports published V4.1 Flash Q2 text/vision, resident experts, SSD streaming and [two-machine TCP/RoCE](CLUSTERING_ROCM.md). Engram remains disk-backed.
- SSD measurements: 128 GB Framework Desktop, Ryzen AI Max+ 395, Radeon `gfx1151`; SK hynix PC711 1 TB (PCIe 3.0 ×4, ext4) holds the model. Linux `7.2.5-100.fc43.x86_64`, ROCm SDK `10.0.0-4` / HIP `7.15.26333`, TuneD **`accelerator-performance`**.

### SSD performance

Native `ds4-bench`, 76 GiB expert-cache request, 34,816 allocated context, 128 greedy output tokens per frontier, no DSpark or images. First row is fresh prefill; subsequent rows append to restored prefixes. One run per row, startup excluded; no cold-cache claim. **Tokens/s**:

| Context tokens | Appended tokens | Prefill | Decode |
|---:|---:|---:|---:|
| 2,048 | 2,048 | 47.32 | 10.18 |
| 4,096 | 2,048 | 86.48 | 10.27 |
| 8,192 | 4,096 | 154.58 | 10.46 |
| 16,384 | 8,192 | 271.06 | 10.34 |
| 32,768 | 16,384 | 333.84 | 10.34 |

- Full frontier logits and continuations match the corresponding controls.
- The tuned Engram matrix path requires hipBLASLt 100401, revision `8d1ae90e`; other library versions retain the fallback and may have different prefill performance.
- Cache admission depends on available RAM, context and sessions; images may need a smaller cache. The GPU-visible limit shares system RAM and is not a cache budget. Populated 256K is unqualified.
- Resident/SSD image and cache checks pass; see [quality results](../QA_BEFORE_RELEASES.md#deepseek-v41-flash-rocmgfx1151). Image inputs are excluded from text timing.

### Run text or vision

```bash
make strix-halo ROCM_ARCH=gfx1151
./download_model.sh ds41f-q2
./download_model.sh ds41f-vision
MODEL=gguf/DeepSeek-V4.1-Flash-Q2.gguf
VISION=gguf/DeepSeek-V4.1-Flash-Vision.gguf

# CLI, text
./ds4 --rocm -m "$MODEL" --ssd-streaming \
  --ssd-streaming-cache-experts 76GB --ctx 34816

# HTTP server, text and images
./ds4-server --rocm -m "$MODEL" --vision "$VISION" \
  --ssd-streaming --ssd-streaming-cache-experts 76GB --ctx 34816 \
  --batched-session 1 --host 127.0.0.1 --port 8080
```

For sufficient RAM to keep experts resident, omit both SSD options. Keep `--vision` for images and set `--ctx` to the required allocation. See [image requests](MODELS.md#vision).

### Reproduce SSD measurements

Prepare `bench-prompt.txt` as in the [cluster benchmark](CLUSTERING_ROCM.md#reproduce-the-table), then run from the engine build directory:

```bash
tuned-adm active    # Verify accelerator-performance during the workload
DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 ./ds4-bench --rocm -m "$MODEL" \
  --ssd-streaming --ssd-streaming-cache-experts 76GB \
  --prompt-file bench-prompt.txt \
  --ctx-start 2048 --ctx-max 32768 --step-mul 2 --ctx-alloc 34816 \
  --gen-tokens 128 --show-output --csv ssd.csv \
  --dump-frontier-logits-dir ssd-frontiers
```

Save CSV, full frontier files, printed output, revision/build flags, model filename/size, active power profile and memory/swap counters. Leave sufficient RAM for the 2 GiB snapshot cap.

### Resident/missing expert split

`DS4_ROCM_SELECTED_SPLIT=1` runs the cached experts of a decode step while the misses are still being read.
It is now deterministic: when a miss layer fell back to the compact table, the generic kernels could read that
table before the D2D compaction copies on the upload stream had finished. Together with bf16 rounding fused
into kernel epilogues (−765 launches/token), the SSD sweep above gives prefill / decode tokens/s:

| Context tokens | Appended tokens | Without | `DS4_ROCM_SELECTED_SPLIT=1` |
|---:|---:|---:|---:|
| 2,048 | 2,048 | 90.87 / 10.30 | 90.82 / 13.79 |
| 4,096 | 2,048 | 147.64 / 10.33 | 147.67 / 13.88 |
| 8,192 | 4,096 | 195.19 / 10.46 | 195.11 / 14.01 |
| 16,384 | 8,192 | 306.46 / 10.41 | 305.82 / 13.93 |
| 32,768 | 16,384 | 351.66 / 10.41 | 351.40 / 13.87 |

Same protocol as above, but with a 72 GB cache request (the largest admitted on the measuring machine) and a
WD SN850X (PCIe 4.0 ×4). Frontier logits are byte-identical between the two columns.

### DSpark speculative decoding

Build the support file from the official checkpoint's `mtp.*` tensors. These are shards 44-46 of 48 of
`deepseek-ai/DeepSeek-V4.1-Flash`, about 8 GB. The conversion uses the gguf-tools of
[antirez/ds4#1073](https://github.com/antirez/ds4/pull/1073); the wrapper skips the tokenizer metadata that only
the main GGUF needs:

```bash
git fetch https://github.com/antirez/ds4 pull/1073/head:pr-1073 && git worktree add ../ds4-pr1073 pr-1073
make -C ../ds4-pr1073/gguf-tools
cp tools/strix-halo/dspark41_config.json /path/to/shards/config.json
python3 tools/strix-halo/dspark41_support.py --gguf-tools ../ds4-pr1073/gguf-tools \
  --hf /path/to/shards --dspark-out gguf/DeepSeek-V4.1-Flash-DSpark-Q2.gguf
```

Then add the support file to the server command:

```bash
DS4_ROCM_SELECTED_SPLIT=1 ./ds4-server --rocm -m "$MODEL" --vision "$VISION" \
  --ssd-streaming --ssd-streaming-cache-experts 72GB --ctx 34816 --batched-session 1 \
  --dspark --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark-Q2.gguf
```

- **What stays resident.** The three draft stages (4.6 GB, 128 experts, top-3) stay resident; the trunk keeps
  streaming its experts.
- **How drafting runs.** Verify runs 2-6 rows through the same-session row batch, with per-row arithmetic
  identical to decode. Greedy requests verify the drafts up to the first one whose confidence is below 0.5
  (`--dspark-confidence` overrides it), with no proposal-rate adaptation (`DS4_DSPARK_V41_ADAPTIVE_DRAFTING=1`
  restores it). Sampling keeps 0.7 and the adaptation.
- **Batched sessions.** With `--batched-session 1` speculation runs directly on the one resident session.

Measured on a 12k-token prompt with greedy chat requests and `tools/strix-halo/bench_server.sh`. Values are
client-side decode tokens/s for the first and second identical request:

| Workload | Without DSpark | With DSpark | Draft acceptance | Output |
|---|---:|---:|---:|---|
| Rewrite a 143-line C function (164 tokens) | 10.79 / 10.80 | 18.96 / 19.50 | 97.0%, 4.6 tokens/cycle | identical |
| Italian prose analysis (256 tokens) | 11.49 / 11.62 | 13.06 / 13.17 | 82.0%, 1.5 tokens/cycle | near-tie differences |

- **Second requests without #1089.** Without live-session rewind the second request rebuilds its session, so
  both requests run on caches refilled after a prefill.
- **With #1089.** The draft ring must be clamped at the rewind frontier. The second request then reaches
  31.7 tok/s on the code rewrite (13.8 without DSpark) and 16.6 on the prose (13.4).
- **Near-tie differences on prose.** Verify rows use different attention and norm kernels than single-token
  decode, so a near-tie token can change.
- **Expert prefetch.** Next-layer expert prefetch (`DS4_V41_PREFETCH`, off) reaches 71% recall at top-6, but the
  extra router pass costs what the reads save.

The table above was measured with the earlier greedy defaults: threshold 0.7 and proposal-rate adaptation. With
`DS4_DSPARK_SPEC_LOG=1` the log prints each draft position's confidence and the accepted length. On 1,180 cycles of
Italian and English prose, the acceptance in each confidence band matched the head's probability in both
languages (Italian 70/79/86/97% and English 64/74/82/98% for the 0.6/0.7/0.8/0.9 bands). An extra verify row costs
about 18 ms against 73 ms for a decode step, so drafts pay from about 0.5.

Four Italian and four English requests (a 12k-token code review answered in prose, an essay, an explanation and an
email), a code-writing request and the code rewrite above, up to 384 tokens, expert cache 68GB, two rounds. Values are
client-side decode tokens/s, averaged per language:

| Greedy DSpark policy | Italian | English | Code writing | Code rewrite |
|---|---:|---:|---:|---:|
| Threshold 0.7, adaptation on | 14.36 / 14.27 | 14.32 / 14.53 | 19.86 / 19.87 | 19.9 |
| Threshold 0.5, adaptation off (now the default) | 15.41 / 15.48 | 15.54 / 15.44 | 19.97 / 19.92 | 19.9-20.0 |

The code rewrite output is byte-identical between the two; on prose a near-tie token can change.

## GLM 5.3 Flash

The reference Q2 setup uses SSD streaming to leave room for its graph and KV
state. Begin with automatic cache sizing and a small context:

```sh
./download_model.sh glm53-q2
./ds4 --rocm -m gguf/GLM-5.3-Flash-Q2.gguf \
  --ssd-streaming --ctx 4096
```

GLM 5.2 also supports ROCm streaming. Full-model GLM 5.2 inference requires it;
distributed layer slices can be resident. See [SSD streaming](SSD_STREAMING.md)
before adjusting the cache budget.

Both GLM 5.3 Flash and DeepSeek Flash Vision Experimental support images on
ROCm. Add the matching encoder with `--vision FILE`, as described in
[models and vision](MODELS.md#vision).

For a model-free routed-kernel check, use `make test-mxfp4-rocm`.
Full-model validation is described in [testing](TESTING.md).

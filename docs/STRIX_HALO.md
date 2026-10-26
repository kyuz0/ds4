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

DeepSeek V4 Flash 0731 IQ2_XXS/Q2_K with Q8 projections uses optimized
2K/4K resident-prefill kernels automatically on `gfx1151`. Dispatch checks
tensor layouts, context bounds and execution mode before selecting them;
other shapes use the existing kernels. See [Halo source credits](HALO_THIRD_PARTY.md).

## DeepSeek V4.1 Flash

- ROCm 10.0 supports published V4.1 Flash Q2 text/vision, resident experts, resident DSpark speculative decoding, SSD streaming and [two-machine TCP/RoCE](CLUSTERING_ROCM.md). Engram remains disk-backed.
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

### Short and medium SSD appends

Agent turns append small tool results and user messages to a warm session. With SSD streaming, ROCm used to handle an append in two ways: below 256 tokens it decoded the append one token at a time, at generation speed; from 256 tokens it ran layer sweeps of up to 2,048 rows, and decoded any remainder below 256 tokens one token at a time.

- **Short appends (2 to 255 tokens)** now run as batched steps of six rows on the DSpark verifier's gfx1151 tiny-batch kernels, as CUDA does for SSD streaming. `DS4_ROCM_V41_SHORT_PREFILL=0` restores token-major decode; another value from 2 to 8 sets the rows per step (eight rows use the generic kernels and measured no faster than token-major decode).
- **Medium appends (2,049 to 8,191 tokens)** now run as one layer sweep, tail included, instead of a 2,048-row sweep plus a token-major tail. `DS4_ROCM_V41_MEDIUM_SWEEP=0` restores the old split; `1` applies the CUDA rule (tail of at least 256 tokens).

Measured on a 128 GB Strix Halo with a Lexar NQ790 2 TB (PCIe 4.0 ×4, btrfs), `--ssd-streaming --ssd-streaming-cache-experts 84GB --ctx 262144`, Q2 `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix`, warm chat session of about 40K tokens, server `prompt done` time in seconds, mean of the same two appends before and after:

| Appended tokens | Before | After |
|---:|---:|---:|
| 29, 31 | 4.2 | 3.3 |
| 46, 56 | 6.1 | 4.4 |
| 79, 84 | 9.1 | 6.3 |
| 133, 153 | 16.2 | 10.3 |
| 220, 231 | 26.5 | 17.1 |
| 565, 592 | 16.8 | 17.3 |
| 1,197, 1,243 | 16.7 | 17.2 |
| 1,813, 2,075 | 20.2 | 19.9 |
| 3,105, 3,257 | 37.8 | 29.9 |

Appends of 565 to 1,813 tokens take unchanged paths; their spread is run-to-run noise.

- Medium sweep: frontier logits are bit-identical to the old split (`ds4-bench`, 3,105-token appends from 20,000 tokens, 4 frontiers).
- Short prefill: argmax identical at every frontier, KL 3e-8 to 2e-7 against token-major decode (231-token appends, 4 frontiers). For scale, routing the same 231-token appends through a layer sweep instead of token-major decode differs by KL 1e-8 to 1e-6. Two control runs are bit-identical.

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

### Resident DSpark speculative decoding

DSpark requires resident target experts and a separate support GGUF. The qualified support file is the released native-MXFP4 draft at the pinned revision below (7,966,294,016 bytes; SHA256 `7a2217ca6ef27cbce4ac934d8b6f59f2f6158ec288527bbe72b86dcbed95d4b5`):

```bash
DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
curl -L --fail -o "$DSPARK" \
  https://huggingface.co/kernelpool/DeepSeek-V4.1-Flash-MXFP4-GGUF/resolve/22a073e51781f43e6814885bb4b3cbe540f175ef/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf

./ds4-server --rocm -m "$MODEL" --mtp-model "$DSPARK" --dspark \
  --ctx 8192 --host 127.0.0.1 --port 8080
```

Use normal positive-temperature sampling with `--dspark`; `--mtp-exact-sampling` is a separate diagnostic mode. Set `DS4_DSPARK_STATS=1` to print proposal, acceptance and verifier counters at shutdown. The target contributes about 151.76 GiB of resident tensor spans and the support model about 7.42 GiB, before context and runtime buffers. DSpark is not available with SSD expert streaming.

### Decoder SWA Bounded Replay

For resident V4.1 inference, opt in to Decoder SWA Bounded Replay with `DS4_ENABLE_V41_DECODER_SWA_BOUNDED_REPLAY=1`. It is off by default, replays only the final 128 prompt tokens through decoder layers, and does not apply to SSD streaming:

```bash
DS4_ENABLE_V41_DECODER_SWA_BOUNDED_REPLAY=1 ./ds4 --rocm -m "$MODEL" --ctx 34816
```

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

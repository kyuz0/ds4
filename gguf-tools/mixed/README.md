# Mixed GGUF Splicing

This directory contains a small GGUF splicer for experiments where only some
routed-expert layers are upgraded from one quantization to another.

The tool does not dequantize or requantize weights. It reads two compatible
GGUF files, uses the first one as the base, and copies selected routed-expert
tensors from the second one. All other tensors remain byte-for-byte identical to
the base file.

## Last Six Layers Q4 Experiment

The best mixed Flash experiment so far keeps the fixed-imatrix 2-bit GGUF as the
base, then replaces only the routed experts in layers `37-42` with the same
tensors from the fixed-imatrix Q4 GGUF.

This produces a file around 91 GB instead of the full Q4 file around 153 GB. In
the local output-agreement checks, this last-six-layers file was statistically
closer to the full Q4 GGUF than the plain 2-bit GGUF. This is not the same as a
benchmark score, but it is a useful signal that the mixed file behaves more like
the stronger Q4 model while keeping most of the 2-bit size.

## Usage

```bash
python3 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py \
  --base /path/to/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf \
  --donor /path/to/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-fixed.gguf \
  --q4-layers 37-42 \
  --out /path/to/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf
```

Use `--dry-run` first to check the selected tensors and expected size delta.
Use `--force` only when you really want to overwrite the output file.

Layer ranges can be changed without editing the script:

```bash
--q4-layers 0-2,40-42
```

## GLM-5.3 Flash Q2/Q4

To build the 137.9 GB mixed model for a 192 GB Strix Halo system, use the
published Q4_K GGUF as the base and copy routed gate, up, and down experts in
blocks `3-28` from the published Q2 GGUF:

```bash
python3 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py \
  --base /path/to/GLM-5.3-Flash-Q4_K.gguf \
  --donor /path/to/GLM-5.3-Flash-Q2.gguf \
  --donor-layers 3-28 \
  --out /path/to/GLM-5.3-Flash-Mixed-L03-28.gguf
```

All other tensors, including the embedded MTP block, come from the Q4_K base.
The output is a GGUF model file; use the matching GLM-5.3 Flash vision encoder
GGUF for image inputs.

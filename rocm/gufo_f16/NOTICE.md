# Imported V4.1 kernel source notice

`ds41_routed_f16.hip.cpp`, `ds41_dense_f16.hip.cpp`, `ds41_v41.hip.cpp` and `ds41_v41.h` are copied unchanged from
[franzmoca/ds41-gufo-extension](https://github.com/franzmoca/ds41-gufo-extension)
at commit `706876f14f9ab743fced0fd49dc184ded191295a` (`kernels/rocm/`), MIT licensed
(`LICENSE.ds41-gufo-extension`; parts derive from Gufo, `LICENSE.gufo`).
`mmq/ggml-common.h` is that repository's vendored llama.cpp header (MIT, The ggml authors).

The kernel runs the batched routed-expert gate/up and down projections on the
FP16 matrix cores: activations are rounded to FP16 and each IQ2_XXS / Q2_K weight
block is decoded to FP16 with its scale folded in. It is not bit-identical to the
MMQ path (which quantizes activations to 8-bit blocks), so it is gated by the
official scorer, not the bit-exact rule.

Used by `routed_moe_launch` (`rocm/ds4_rocm_moe_launch.cuh`) by default for
staged layers of 48 rows or more; `DS4_ROCM_V41_ROUTED_F16=0` restores the MMQ path.

The dense kernel runs batched Q8_0 projections (attention query/KV, attention
output A/B, shared expert) for 32-2048 rows from `ds4_gpu_dsv41_q8_projection_rows`
and `ds4_gpu_dsv41_attention_output_batch` (`rocm/ds4_rocm_v41.cuh`), with the same
operand rounding as the WMMA row-tile kernels it replaces;
`DS4_ROCM_V41_DENSE_F16=0` restores those.

`ds41_v41.hip.cpp` holds the rest of that project's V4.1 kernels. Two of them are
called:

- The BF16 matrix-core product (`ds41_v41_matmul_bf16`), for the router projection of
  16+ rows (`ds4_gpu_dsv41_router_bf16_rows`, `rocm/ds4_rocm_v41.cuh`): the F32 router
  weights are exactly representable in BF16 (checked for all 40 layers of the 1017 Q2
  file) and are converted once per layer; activations are split into three BF16 terms,
  so no precision is lost. `DS4_ROCM_V41_ROUTER_BF16=0` restores hipBLAS SGEMM.
- The prefill attention kernel (`ds41_v41_attention`) with its packed compressed rows
  (`ds41_v41_fp4_store`), for batched attention of 32+ rows
  (`ds4_gpu_dsv41_attention_fp4_batch`, `rocm/ds4_rocm_v41.cuh`). Not bit-identical to
  the kernel it replaces (heads differ at the 1e-5 level), gated by the official scorer.
  `DS4_ROCM_V41_ATTN_FP4=0` restores the previous kernel.

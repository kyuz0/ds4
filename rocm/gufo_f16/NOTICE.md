# Routed FP16 expert kernel source notice

`ds41_routed_f16.hip.cpp` and `ds41_v41.h` are copied unchanged from
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

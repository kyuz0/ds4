# V4.1 IQ2 gate/up source notice

This shape-specific adapter derives from the qualified V4 co-computed gate/up implementation in `rocm/halo/moe`. Its IQ2 grid, codecs and native quantization helpers derive from DS4/ggml at the pinned sources in `docs/halo/sources.json`. Preserve the repository MIT license and its DS4, ggml and DeepSeek author notices. HIP and rocWMMA are separately licensed external dependencies.

The adapter uses V4.1 dimensions and routing capacity while preserving native MMQ quantization and the current fused SwiGLU multiplication order. It serves the resident ROCm backend's 2048-row physical workspace. Other shapes retain the native fallback.

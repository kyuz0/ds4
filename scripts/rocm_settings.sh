#!/usr/bin/env bash
# Shared ROCm runtime presets for DS4 launch/bench scripts.
# Source this file from scripts after ROOT_DIR/cd has been established.

if [[ -d "/opt/rocm/bin" ]]; then
  export PATH="/opt/rocm/bin:$PATH"
fi

# Export a DS4_CUDA_* and DS4_HIP_* boolean alias when the caller-facing
# DS4_SERVER_* knob is enabled.
ds4_rocm_export_pair_flag() {
  local suffix="$1"
  local value="${2:-0}"
  if [[ "$value" == "1" ]]; then
    export "DS4_CUDA_${suffix}=1"
    export "DS4_HIP_${suffix}=1"
  fi
}

# Export a DS4_CUDA_* and DS4_HIP_* value alias when the caller-facing
# DS4_SERVER_* knob is non-empty.
ds4_rocm_export_pair_value() {
  local suffix="$1"
  local value="${2:-}"
  if [[ -n "$value" ]]; then
    export "DS4_CUDA_${suffix}=$value"
    export "DS4_HIP_${suffix}=$value"
  fi
}

# Max-performance upstream-shaped ROCm defaults. Caller may pass a prefill chunk
# default; server uses 4096, bench/CLI use 2048 unless overridden.
ds4_rocm_apply_fast_full_defaults() {
  local prefill_chunk_default="${1:-2048}"

  export DS4_SERVER_PERFLEVEL="${DS4_SERVER_PERFLEVEL-high}"
  export DS4_SERVER_PREFILL_CHUNK="${DS4_SERVER_PREFILL_CHUNK:-$prefill_chunk_default}"
  export DS4_SERVER_DYNAMIC_PREFILL="${DS4_SERVER_DYNAMIC_PREFILL:-1}"
  export DS4_SERVER_DEVICE_TENSORS="${DS4_SERVER_DEVICE_TENSORS:-1}"
  export DS4_SERVER_COPY_MODEL="${DS4_SERVER_COPY_MODEL:-1}"
  export DS4_SERVER_COPY_MODEL_CHUNK_MB="${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"

  export DS4_SERVER_PREFILL_RAW_FAST="${DS4_SERVER_PREFILL_RAW_FAST:-1}"
  export DS4_SERVER_PREFILL_MIXED_FAST="${DS4_SERVER_PREFILL_MIXED_FAST:-1}"
  export DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP="${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-4}"
  export DS4_SERVER_INDEXED_HEADS32="${DS4_SERVER_INDEXED_HEADS32:-1}"

  export DS4_SERVER_ATTN_Q_B_CUBLAS="${DS4_SERVER_ATTN_Q_B_CUBLAS:-1}"
  export DS4_SERVER_ATTN_Q_B_PRELOAD="${DS4_SERVER_ATTN_Q_B_PRELOAD:-1}"
  export DS4_SERVER_ATTN_Q_B_F16_OUT="${DS4_SERVER_ATTN_Q_B_F16_OUT:-1}"
  export DS4_SERVER_ATTENTION_OUTPUT_F16_OUT="${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-1}"
  export DS4_SERVER_SHARED_DOWN_F16_OUT="${DS4_SERVER_SHARED_DOWN_F16_OUT:-1}"

  export DS4_SERVER_Q8_BATCH_FAST="${DS4_SERVER_Q8_BATCH_FAST:-1}"
  export DS4_SERVER_Q8_BATCH_SHARED_X="${DS4_SERVER_Q8_BATCH_SHARED_X:-1}"
  export DS4_SERVER_Q8_BATCH_TILE="${DS4_SERVER_Q8_BATCH_TILE:-32}"
  export DS4_SERVER_Q8_BATCH_RPB="${DS4_SERVER_Q8_BATCH_RPB:-32}"
  export DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS="${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-32}"
  export DS4_SERVER_Q8_GROUPED_BATCH_TILE="${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-32}"
  export DS4_SERVER_Q8_REPACK="${DS4_SERVER_Q8_REPACK:-0}"
  export DS4_SERVER_Q8_REPACK_SPLIT16="${DS4_SERVER_Q8_REPACK_SPLIT16:-0}"
  export DS4_SERVER_Q8_WMMA_FAST="${DS4_SERVER_Q8_WMMA_FAST:-1}"
  export DS4_SERVER_Q8_WMMA_4W="${DS4_SERVER_Q8_WMMA_4W:-1}"
  export DS4_SERVER_Q8_PREQUANT_DECODE="${DS4_SERVER_Q8_PREQUANT_DECODE:-1}"
  export DS4_SERVER_Q8_DECODE_RPB="${DS4_SERVER_Q8_DECODE_RPB:-1}"
  export DS4_SERVER_Q8_HC_DECODE_RPB="${DS4_SERVER_Q8_HC_DECODE_RPB:-16}"
  export DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB="${DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB:-32}"

  export DS4_SERVER_OLDHIP_ATTENTION_DECODE="${DS4_SERVER_OLDHIP_ATTENTION_DECODE:-1}"
  export DS4_SERVER_QKV_PAIR_DECODE="${DS4_SERVER_QKV_PAIR_DECODE:-1}"
  export DS4_SERVER_MOE_DECODE_RPB="${DS4_SERVER_MOE_DECODE_RPB:-1}"
  export DS4_SERVER_MOE_DECODE_Q8K_GATEUP="${DS4_SERVER_MOE_DECODE_Q8K_GATEUP:-1}"
  export DS4_SERVER_MOE_DECODE_Q8K_DOWN="${DS4_SERVER_MOE_DECODE_Q8K_DOWN:-1}"
  export DS4_SERVER_ATTN_OUT_LOW_SPLITK="${DS4_SERVER_ATTN_OUT_LOW_SPLITK:-0}"
  export DS4_SERVER_SHARED_GATE_UP_FUSED_W32="${DS4_SERVER_SHARED_GATE_UP_FUSED_W32:-0}"
  export DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR_SWIGLU="${DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR_SWIGLU:-1}"
  export DS4_SERVER_OVERLAP_SHARED_GATE_UP="${DS4_SERVER_OVERLAP_SHARED_GATE_UP:-1}"

  export DS4_SERVER_MOE_EXPERT_BATCH="${DS4_SERVER_MOE_EXPERT_BATCH:-1}"
  export DS4_SERVER_MOE_GATE_TILE="${DS4_SERVER_MOE_GATE_TILE:-4}"
  export DS4_SERVER_MOE_DOWN_TILE="${DS4_SERVER_MOE_DOWN_TILE:-4}"
  export DS4_SERVER_MOE_GATE_RPB="${DS4_SERVER_MOE_GATE_RPB:-16}"
  export DS4_SERVER_MOE_DOWN_RPB="${DS4_SERVER_MOE_DOWN_RPB:-16}"
  export DS4_SERVER_MOE_EXPERT_SHARED_X="${DS4_SERVER_MOE_EXPERT_SHARED_X:-1}"
  export DS4_SERVER_MOE_EXPERT_SHARED_MID="${DS4_SERVER_MOE_EXPERT_SHARED_MID:-1}"
  export DS4_SERVER_MOE_WMMA_HOT="${DS4_SERVER_MOE_WMMA_HOT:-1}"
  export DS4_SERVER_MOE_WMMA_GATE_HOT="${DS4_SERVER_MOE_WMMA_GATE_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_DOWN_HOT="${DS4_SERVER_MOE_WMMA_DOWN_HOT:-8}"
  export DS4_SERVER_MOE_WMMA_MTILES="${DS4_SERVER_MOE_WMMA_MTILES:-16}"
  export DS4_SERVER_MOE_WMMA_F16_DOWN_ALL="${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT="${DS4_SERVER_MOE_WMMA_F16_SPLIT:-1}"
  export DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN="${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-64}"
  export DS4_SERVER_MOE_WMMA_X_F16="${DS4_SERVER_MOE_WMMA_X_F16:-1}"

  # Upstream-shaped ROCm-only extras from the fastest CLI/bench recipe.
  export DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED="${DS4_CUDA_SHARED_GATE_UP_BATCH_FUSED:-1}"
  export DS4_CUDA_SHARED_DOWN_CUBLAS="${DS4_CUDA_SHARED_DOWN_CUBLAS:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL="${DS4_CUDA_ATTENTION_OUTPUT_CUBLAS_ALL:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_PACKED_B_CUBLAS="${DS4_CUDA_ATTENTION_OUTPUT_PACKED_B_CUBLAS:-1}"
  export DS4_CUDA_ATTENTION_OUTPUT_INTERLEAVED_B_CUBLAS="${DS4_CUDA_ATTENTION_OUTPUT_INTERLEAVED_B_CUBLAS:-1}"
}

# Quality-gated decode defaults used by parity/eval-style launchers. This keeps
# experimental prefill/WMMA paths opt-in while enabling production decode fixes.
ds4_rocm_apply_quality_decode_defaults() {
  export DS4_CUDA_COPY_MODEL="${DS4_CUDA_COPY_MODEL:-1}"
  export DS4_HIP_COPY_MODEL="${DS4_HIP_COPY_MODEL:-$DS4_CUDA_COPY_MODEL}"
  export DS4_CUDA_Q8_PREQUANT_DECODE="${DS4_CUDA_Q8_PREQUANT_DECODE:-1}"
  export DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW="${DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW:-1}"
  export DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32="${DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32:-1}"
}

# Translate DS4_SERVER_* knobs to the historical DS4_CUDA_* and DS4_HIP_* names
# consumed by the upstream-shaped ROCm backend.
ds4_rocm_export_runtime_env() {
  ds4_rocm_export_pair_flag PREFILL_RAW_FAST "${DS4_SERVER_PREFILL_RAW_FAST:-0}"
  ds4_rocm_export_pair_flag PREFILL_MIXED_FAST "${DS4_SERVER_PREFILL_MIXED_FAST:-0}"
  ds4_rocm_export_pair_value ATTENTION_INDEXED_FUSED_VALUE_GROUP "${DS4_SERVER_ATTENTION_INDEXED_FUSED_VALUE_GROUP:-}"
  ds4_rocm_export_pair_value ATTENTION_INDEXED_SPLIT_VALUE_GROUP "${DS4_SERVER_ATTENTION_INDEXED_SPLIT_VALUE_GROUP:-}"
  ds4_rocm_export_pair_flag INDEXED_HEADS32 "${DS4_SERVER_INDEXED_HEADS32:-0}"

  ds4_rocm_export_pair_flag ATTN_Q_B_CUBLAS "${DS4_SERVER_ATTN_Q_B_CUBLAS:-0}"
  ds4_rocm_export_pair_flag ATTN_Q_B_PRELOAD "${DS4_SERVER_ATTN_Q_B_PRELOAD:-0}"
  ds4_rocm_export_pair_flag ATTN_Q_B_F16_OUT "${DS4_SERVER_ATTN_Q_B_F16_OUT:-0}"
  ds4_rocm_export_pair_flag ATTENTION_OUTPUT_F16_OUT "${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT:-0}"
  ds4_rocm_export_pair_value ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS "${DS4_SERVER_ATTENTION_OUTPUT_F16_OUT_MIN_TOKENS:-}"
  ds4_rocm_export_pair_flag SHARED_DOWN_F16_OUT "${DS4_SERVER_SHARED_DOWN_F16_OUT:-0}"
  ds4_rocm_export_pair_value SHARED_DOWN_F16_OUT_MIN_TOKENS "${DS4_SERVER_SHARED_DOWN_F16_OUT_MIN_TOKENS:-}"

  ds4_rocm_export_pair_flag Q8_BATCH_FAST "${DS4_SERVER_Q8_BATCH_FAST:-0}"
  ds4_rocm_export_pair_value Q8_BATCH_TILE "${DS4_SERVER_Q8_BATCH_TILE:-}"
  ds4_rocm_export_pair_value Q8_BATCH_RPB "${DS4_SERVER_Q8_BATCH_RPB:-}"
  ds4_rocm_export_pair_flag Q8_BATCH_SHARED_X "${DS4_SERVER_Q8_BATCH_SHARED_X:-0}"
  ds4_rocm_export_pair_value Q8_BATCH_SHARED_X_BLOCKS "${DS4_SERVER_Q8_BATCH_SHARED_X_BLOCKS:-}"
  ds4_rocm_export_pair_value Q8_GROUPED_BATCH_TILE "${DS4_SERVER_Q8_GROUPED_BATCH_TILE:-}"
  ds4_rocm_export_pair_flag Q8_REPACK "${DS4_SERVER_Q8_REPACK:-0}"
  ds4_rocm_export_pair_flag Q8_REPACK_SPLIT16 "${DS4_SERVER_Q8_REPACK_SPLIT16:-0}"
  ds4_rocm_export_pair_flag Q8_WMMA_FAST "${DS4_SERVER_Q8_WMMA_FAST:-0}"
  ds4_rocm_export_pair_flag Q8_WMMA_4W "${DS4_SERVER_Q8_WMMA_4W:-0}"
  ds4_rocm_export_pair_value Q8_WMMA_4W_MIN_TOKENS "${DS4_SERVER_Q8_WMMA_4W_MIN_TOKENS:-}"
  ds4_rocm_export_pair_value Q8_WMMA_4W_MIN_OUT "${DS4_SERVER_Q8_WMMA_4W_MIN_OUT:-}"
  ds4_rocm_export_pair_value Q8_DECODE_RPB "${DS4_SERVER_Q8_DECODE_RPB:-}"
  ds4_rocm_export_pair_value Q8_HC_DECODE_RPB "${DS4_SERVER_Q8_HC_DECODE_RPB:-}"
  ds4_rocm_export_pair_value ATTN_OUT_LOW_DECODE_RPB "${DS4_SERVER_ATTN_OUT_LOW_DECODE_RPB:-}"
  ds4_rocm_export_pair_flag Q8_HIPBLASLT "${DS4_SERVER_Q8_HIPBLASLT:-0}"
  ds4_rocm_export_pair_value Q8_HIPBLASLT_MAX_TOKENS "${DS4_SERVER_Q8_HIPBLASLT_MAX_TOKENS:-}"

  ds4_rocm_export_pair_flag OLDHIP_ATTENTION_DECODE "${DS4_SERVER_OLDHIP_ATTENTION_DECODE:-0}"
  ds4_rocm_export_pair_flag QKV_PAIR_DECODE "${DS4_SERVER_QKV_PAIR_DECODE:-0}"
  ds4_rocm_export_pair_flag OVERLAP_SHARED_GATE_UP "${DS4_SERVER_OVERLAP_SHARED_GATE_UP:-0}"

  if [[ "${DS4_SERVER_Q8_PREQUANT_DECODE:-0}" == "1" ]]; then
    export DS4_CUDA_Q8_PREQUANT_DECODE=1
  elif [[ -n "${DS4_SERVER_Q8_PREQUANT_DECODE:-}" ]]; then
    unset DS4_CUDA_Q8_PREQUANT_DECODE || true
  fi
  if [[ "${DS4_SERVER_ATTN_OUT_LOW_SPLITK:-0}" == "1" ]]; then
    unset DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW || true
  else
    export DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW="${DS4_CUDA_DISABLE_SPLITK_ATTN_OUT_LOW:-1}"
  fi
  if [[ "${DS4_SERVER_SHARED_GATE_UP_FUSED_W32:-1}" == "1" ]]; then
    unset DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32 || true
  else
    export DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32="${DS4_CUDA_DISABLE_SHARED_GATE_UP_FUSED_W32:-1}"
  fi

  ds4_rocm_export_pair_flag MOE_EXPERT_BATCH "${DS4_SERVER_MOE_EXPERT_BATCH:-0}"
  ds4_rocm_export_pair_value MOE_EXPERT_TILE "${DS4_SERVER_MOE_EXPERT_TILE:-}"
  ds4_rocm_export_pair_value MOE_GATE_TILE "${DS4_SERVER_MOE_GATE_TILE:-}"
  ds4_rocm_export_pair_value MOE_DOWN_TILE "${DS4_SERVER_MOE_DOWN_TILE:-}"
  ds4_rocm_export_pair_value MOE_GATE_RPB "${DS4_SERVER_MOE_GATE_RPB:-}"
  ds4_rocm_export_pair_value MOE_DOWN_RPB "${DS4_SERVER_MOE_DOWN_RPB:-}"
  ds4_rocm_export_pair_value MOE_DECODE_RPB "${DS4_SERVER_MOE_DECODE_RPB:-}"
  ds4_rocm_export_pair_flag MOE_DECODE_Q8K_GATEUP "${DS4_SERVER_MOE_DECODE_Q8K_GATEUP:-0}"
  ds4_rocm_export_pair_flag MOE_DECODE_Q8K_DOWN "${DS4_SERVER_MOE_DECODE_Q8K_DOWN:-0}"
  ds4_rocm_export_pair_flag MOE_EXPERT_SHARED_X "${DS4_SERVER_MOE_EXPERT_SHARED_X:-0}"
  ds4_rocm_export_pair_flag MOE_EXPERT_SHARED_MID "${DS4_SERVER_MOE_EXPERT_SHARED_MID:-0}"
  ds4_rocm_export_pair_flag MOE_Q8K_DOWN "${DS4_SERVER_MOE_Q8K_DOWN:-0}"
  ds4_rocm_export_pair_value MOE_Q8K_DOWN_LAYERS "${DS4_SERVER_MOE_Q8K_DOWN_LAYERS:-}"
  ds4_rocm_export_pair_flag MOE_Q8K_DOWN_DIRECT "${DS4_SERVER_MOE_Q8K_DOWN_DIRECT:-0}"
  ds4_rocm_export_pair_value MOE_Q8K_DOWN_TILE "${DS4_SERVER_MOE_Q8K_DOWN_TILE:-}"
  ds4_rocm_export_pair_flag MOE_WMMA_HOT "${DS4_SERVER_MOE_WMMA_HOT:-0}"
  ds4_rocm_export_pair_value MOE_WMMA_GATE_HOT "${DS4_SERVER_MOE_WMMA_GATE_HOT:-}"
  ds4_rocm_export_pair_value MOE_WMMA_DOWN_HOT "${DS4_SERVER_MOE_WMMA_DOWN_HOT:-}"
  ds4_rocm_export_pair_value MOE_WMMA_LAYERS "${DS4_SERVER_MOE_WMMA_LAYERS:-}"
  ds4_rocm_export_pair_value MOE_WMMA_MTILES "${DS4_SERVER_MOE_WMMA_MTILES:-}"
  ds4_rocm_export_pair_flag MOE_WMMA_SPLIT_HOT "${DS4_SERVER_MOE_WMMA_SPLIT_HOT:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_TILE_HOT "${DS4_SERVER_MOE_WMMA_TILE_HOT:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_F16_MID "${DS4_SERVER_MOE_WMMA_F16_MID:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_F16_MID_ALL "${DS4_SERVER_MOE_WMMA_F16_MID_ALL:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_F16_DOWN "${DS4_SERVER_MOE_WMMA_F16_DOWN:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_F16_DOWN_ALL "${DS4_SERVER_MOE_WMMA_F16_DOWN_ALL:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_F16_SPLIT "${DS4_SERVER_MOE_WMMA_F16_SPLIT:-0}"
  ds4_rocm_export_pair_value MOE_WMMA_F16_SPLIT_MIN "${DS4_SERVER_MOE_WMMA_F16_SPLIT_MIN:-}"
  ds4_rocm_export_pair_flag MOE_WMMA_X_F16 "${DS4_SERVER_MOE_WMMA_X_F16:-0}"
  ds4_rocm_export_pair_flag MOE_SLOT_PARTIAL "${DS4_SERVER_MOE_SLOT_PARTIAL:-0}"
  ds4_rocm_export_pair_flag MOE_WMMA_DIRECT_SUM "${DS4_SERVER_MOE_WMMA_DIRECT_SUM:-0}"
  ds4_rocm_export_pair_flag MOE_DENSE_HOT "${DS4_SERVER_MOE_DENSE_HOT:-0}"
  ds4_rocm_export_pair_value MOE_DENSE_HOT_TOP "${DS4_SERVER_MOE_DENSE_HOT_TOP:-}"
  ds4_rocm_export_pair_value MOE_DENSE_HOT_MIN "${DS4_SERVER_MOE_DENSE_HOT_MIN:-}"
  ds4_rocm_export_pair_value MOE_DENSE_HOT_CACHE_MB "${DS4_SERVER_MOE_DENSE_HOT_CACHE_MB:-}"
  ds4_rocm_export_pair_flag MOE_DENSE_HOT_NO_CACHE "${DS4_SERVER_MOE_DENSE_HOT_NO_CACHE:-0}"

  ds4_rocm_export_pair_flag COPY_MODEL "${DS4_SERVER_COPY_MODEL:-0}"
  ds4_rocm_export_pair_value COPY_MODEL_CHUNK_MB "${DS4_SERVER_COPY_MODEL_CHUNK_MB:-}"
  ds4_rocm_export_pair_flag CACHE_FINAL_Q8 "${DS4_SERVER_CACHE_FINAL_Q8:-0}"
}

# Select managed-vs-device tensor env from DS4_SERVER_DEVICE_TENSORS.
ds4_rocm_apply_tensor_env() {
  if [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" != "1" ]]; then
    export DS4_HIP_MANAGED_TENSORS=1
  else
    unset DS4_HIP_MANAGED_TENSORS || true
  fi
}

# Export shared prefill chunk env. If no explicit chunk exists and device tensors
# are enabled, use device_default. If managed_default is supplied, use it for the
# non-device fallback (server wants 512; bench/CLI leave it unset).
ds4_rocm_apply_prefill_env() {
  local device_default="${1:-128}"
  local managed_default="${2:-}"
  if [[ -n "${DS4_SERVER_PREFILL_CHUNK:-}" ]]; then
    export DS4_METAL_PREFILL_CHUNK="$DS4_SERVER_PREFILL_CHUNK"
    export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$DS4_SERVER_PREFILL_CHUNK}"
  elif [[ "${DS4_SERVER_DEVICE_TENSORS:-0}" == "1" ]]; then
    export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$device_default}"
    export DS4_METAL_PREFILL_CHUNK="${DS4_METAL_PREFILL_CHUNK:-$DS4_SESSION_PROGRESS_CHUNK_TOKENS}"
  elif [[ -n "$managed_default" ]]; then
    export DS4_SESSION_PROGRESS_CHUNK_TOKENS="${DS4_SESSION_PROGRESS_CHUNK_TOKENS:-$managed_default}"
  fi
}

# Normalize a perflevel variable in-place: off/skip/none => empty.
ds4_rocm_normalize_perflevel_var() {
  local var="$1"
  local value="${!var:-}"
  case "$value" in
    off|skip|none) printf -v "$var" '%s' '' ;;
  esac
}

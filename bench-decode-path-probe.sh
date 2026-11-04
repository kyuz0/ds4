#!/usr/bin/env bash
# Tiny probe: which decode driver runs, and does an opt-in hook actually engage?
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial 1-2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-decode-path-probe$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 900 env DS4_V41_DECODE_STAGE_PROFILE=1 DS4_ROCM_V41_PREFETCH_ROUTE=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 8 --show-output \
    --ssd-streaming --ssd-streaming-cache-experts 92GB > "$OUT/probe.log" 2>&1
echo "[rc] $?"
grep -E "pre-attention route\+fetch active|moe_stream_begin|moe_stream_wait" "$OUT/probe.log" | head -3
grep -c "pre-attention route+fetch active" "$OUT/probe.log" | sed 's/^/banner=/'
grep -c "v41 decode stage" "$OUT/probe.log" | sed 's/^/stage_lines=/'
grep -o 'decoded text: ".*"' "$OUT/probe.log" | head -1 | cut -c1-70

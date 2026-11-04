#!/usr/bin/env bash
# Read the model's real dense-tensor types/dims/bytes instead of assuming them.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial 1-2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-dense-tensors$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 900 env DS4_DUMP_DENSE_TENSORS=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB > "$OUT/dump.log" 2>&1
echo "[rc] $?"
grep -oE "dense tensor [^ ]+ type=[a-z0-9_]+ dims=[0-9]+,[0-9]+ bytes=[0-9]+" "$OUT/dump.log" | head -40

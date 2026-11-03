#!/usr/bin/env bash
# Tally tensor_copy traffic vs rocclr-internal copies during a 2K prefill.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-copy-tally-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 900 env DS4_ROCM_COPY_TALLY=1 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
  --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 0 \
  --ssd-streaming --ssd-streaming-cache-experts 92GB --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv"
grep -E "copy tally|prefill frontier" "$OUT/profile.log"
exit 0

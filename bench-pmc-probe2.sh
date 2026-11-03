#!/usr/bin/env bash
# Probe whether rocprofv3 can collect hardware counters on this gfx1151 host, and
# if so get instructions-per-MAC for the IQ2 MMQ MoE kernel.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-pmc-probe2-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/prof" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "=== avail ==="
timeout 120 rocprofv3-avail list --hw-counters 2>&1 | head -5
echo "=== pmc run ==="
timeout -k 60 900 rocprofv3 --pmc TCC_HIT TCC_MISS TCC_EA0_RDREQ TCC_EA0_RDREQ_DRAM GRBM_GUI_ACTIVE SQ_WAVES \
  --kernel-include-regex 'ds4_mmq_compact_mul_mat_q|moe_down_q2K_hotlist' \
  -f csv -d "$OUT/prof" -- \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
grep -iE "error|unsupported|not supported|pmc" "$OUT/profile.log" | head -5
ls "$OUT/prof" 2>/dev/null | head -5
find "$OUT/prof" -name '*.csv' | head -3 | while read f; do echo "--- $f"; head -2 "$f" | cut -c1-220; done
exit 0

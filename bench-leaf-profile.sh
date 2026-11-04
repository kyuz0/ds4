#!/usr/bin/env bash
# Per-leaf device timing (hipEvents) for the dense projections at 32K decode.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2|3) ;; *) echo "*** trial 1-3 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-leaf-profile$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 env DS4_V41_LEAF_PROFILE=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens 16 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
tail -1 "$OUT/profile.csv"
grep -E "leaf profile|ds4:   " "$OUT/profile.log" | head -12
python3 -B - "$OUT/profile.log" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
tokens = 16
rows = re.findall(r"ds4:   (\S+)\s+([0-9.]+) ms\s+(\d+) calls\s+([0-9.]+) MB total\s+([0-9.]+) GB/s", txt)
if not rows:
    print("no leaf rows"); sys.exit(0)
ceil = 223.0
print(f"\n{'leaf':26s}{'ms/token':>10s}{'MB/token':>10s}{'calls':>7s}{'GB/s':>8s}{'%ceil':>7s}")
for name, ms, calls, mb, gbs in rows:
    print(f"{name[:26]:26s}{float(ms)/tokens:10.2f}{float(mb)/tokens:10.1f}"
          f"{calls:>7s}{float(gbs):8.0f}{float(gbs)/ceil*100:7.0f}")
PY

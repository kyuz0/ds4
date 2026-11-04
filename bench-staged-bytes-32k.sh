#!/usr/bin/env bash
# Is the expert copy path saturated? Measure staged bytes and the read wait together.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial 1-2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-staged-bytes$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 env DS4_ROCM_MOE_DECODE_PROFILE=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens 48 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
grep -o "MoE profile.*" "$OUT/profile.log" | tail -1
tail -1 "$OUT/profile.csv"
python3 -B - "$OUT" <<'PY'
import csv, os, re, sys
out=sys.argv[1]; log=open(os.path.join(out,"profile.log")).read()
m=re.search(r"calls=(\d+).*?split_finish_missing=([0-9.]+) ms.*?staged_bytes=(\d+) staged_jobs=(\d+)", log, re.S)
if not m: print("no profile line"); sys.exit(0)
calls, stall, byts, jobs = int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))
tps=float(list(csv.DictReader(open(os.path.join(out,"profile.csv"))))[-1]["gen_tps"])
print(f"calls={calls} (layer-token samples) staged={byts/1e9:.2f} GB in {jobs} jobs")
print(f"  bytes per layer-token {byts/calls/1e6:.2f} MB; jobs per layer-token {jobs/calls:.1f}")
print(f"  wait per layer-token {stall/calls*1000:.0f} us; implied staging rate {byts/1e9/(stall/1e3):.1f} GB/s")
print(f"  gen {tps:.2f} tps -> {1000/tps:.0f} ms/token; stall share {40*stall/calls/(1000/tps)*100:.1f}%")
print(f"  staged bytes per token {byts/calls*40/1e6:.0f} MB")
PY

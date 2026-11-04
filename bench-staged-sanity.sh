#!/usr/bin/env bash
# Sanity probe for staged-byte accounting: values must be nonzero and plausible.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial 1-2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-staged-sanity$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 env DS4_ROCM_STREAM_CACHE_STATS=1 DS4_ROCM_MOE_DECODE_PROFILE=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 32 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
grep -o "layer reuse cleanup.*" "$OUT/profile.log" | tail -1
grep -o "MoE profile.*" "$OUT/profile.log" | tail -1 | cut -c1-120
tail -1 "$OUT/profile.csv"
python3 -B - "$OUT" <<'PY'
import csv, os, re, sys
out=sys.argv[1]; log=open(os.path.join(out,"profile.log")).read()
m=re.search(r"staged-bytes=(\d+) staged-jobs=(\d+)", log)
c=re.search(r"MoE profile calls=(\d+)", log)
if not m: print("SANITY FAIL: no staged-bytes field"); sys.exit(0)
byts,jobs=int(m.group(1)),int(m.group(2))
calls=int(c.group(1)) if c else 0
print(f"SANITY: staged {byts/1e9:.2f} GB in {jobs} jobs; calls={calls}")
if byts==0 or jobs==0: print("SANITY FAIL: zero values")
else:
    print(f"  bytes/job {byts/jobs/1e6:.2f} MB, jobs/call {jobs/calls if calls else 0:.1f}"
          f", bytes/call {byts/calls/1e6 if calls else 0:.2f} MB")
    print(f"  per token (40 layers) {byts/calls*40/1e6 if calls else 0:.0f} MB")
PY

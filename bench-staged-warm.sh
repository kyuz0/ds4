#!/usr/bin/env bash
# Decode-scoped expert-staging profile: per-call wait versus the call's own read time,
# bucketed by read-job count so decode (~1 job) and prefill (many jobs) separate.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2|3|4|5) ;; *) echo "*** trial 1-5 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-staged-warm$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 env DS4_ROCM_STREAM_CACHE_STATS=1 DS4_ROCM_MOE_DECODE_PROFILE=1 \
    DS4_ROCM_STREAM_READ_PROFILE=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 32 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/cold.csv" > "$OUT/cold.log" 2>&1
echo "[rc] $?"
tail -1 "$OUT/cold.csv"
python3 -B - "$OUT/cold.log" <<'PY'
import csv, glob, os, re, sys
log = open(sys.argv[1], errors="ignore").read()
m = re.search(r"staged-bytes=(\d+) staged-jobs=(\d+) staged-read-us=(\d+)(.*)", log)
if not m:
    print("no staged accounting line"); sys.exit(0)
byts, jobs, read_us = int(m.group(1)), int(m.group(2)), int(m.group(3))
print(f"staged {byts/1e6:.0f} MB in {jobs} jobs, aggregate read {read_us/1e6:.2f} s")
for name, blk in re.findall(r"(jobs0|jobs1-3|jobs4\+)\[([^\]]*)\]", m.group(4)):
    c, wu, wr, wl = (int(x) for x in blk.split(","))
    if not c:
        print(f"  {name}: no calls")
        continue
    print(f"  {name}: calls {c}, wait/call {wu/c:.0f} us, own-read/call {wr/c:.0f} us, "
          f"read>wait {100*wl/c:.0f}%, hidden {100*(1-(wu/wr)) if wr else 0:.0f}% of own read")
f = re.search(r"split_finish_missing=([0-9.]+) ms", log)
if f:
    print(f"  moe split_finish_missing {float(f.group(1))/1000:.2f} s")
csvp = sys.argv[1].replace(".log", ".csv")
if os.path.exists(csvp):
    rows = list(csv.DictReader(open(csvp)))
    if rows: print(f"  gen {float(rows[-1]['gen_tps']):.2f} tps")
PY

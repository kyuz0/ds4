#!/usr/bin/env bash
# Identify the rocclr copyBuffer issuer: memory-copy trace with sizes, so the
# per-copy byte count names the tensor being copied (e.g. 20 KB = 5120 floats).
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-api-trace-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/prof" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 rocprofv3 --runtime-trace -f csv -d "$OUT/prof" -- \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv" 2>/dev/null
for f in $(find "$OUT/prof" -name '*api*.csv' -o -name '*runtime*.csv'); do
  echo "== $f"; head -1 "$f" | cut -c1-260
done
python3 -B - "$OUT" <<'PY2'
import csv, glob, sys, collections
for f in glob.glob(sys.argv[1] + "/prof/*/*.csv"):
    if "memory_copy" in f or "agent_info" in f: continue
    rows = list(csv.DictReader(open(f)))
    if not rows: continue
    cols = list(rows[0])
    name_col = next((c for c in cols if c in ("Function","Api_Name","Name") or "Function" in c), None)
    if not name_col: continue
    c = collections.Counter(r[name_col] for r in rows)
    print("==", f.split("/")[-1], "rows", len(rows))
    for k, v in c.most_common(10):
        print("   %-32s %d" % (k, v))
PY2
exit 0

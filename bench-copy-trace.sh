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
OUT="$G/findings/bench-copy-trace-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/prof" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
timeout -k 60 1200 rocprofv3 --memory-copy-trace -f csv -d "$OUT/prof" -- \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv" 2>/dev/null
f=$(find "$OUT/prof" -name '*memory*copy*.csv' | head -1); echo "file: $f"
python3 -B - "$f" <<'PY'
import csv, sys, collections
try: rows=list(csv.DictReader(open(sys.argv[1])))
except Exception as e: print("no copy trace:", e); raise SystemExit(0)
print("rows", len(rows), "cols", list(rows[0])[:12] if rows else [])
sizes=collections.Counter(); tot=collections.Counter()
for r in rows:
    s=r.get("Size") or r.get("size") or r.get("Bytes")
    try: s=int(s)
    except (TypeError,ValueError): s=0
    sizes[s]+=1; tot[s]+=s
print("%-12s %8s %14s" % ("bytes","calls","total_MB"))
for s,c in sizes.most_common(8):
    print("%-12d %8d %14.1f" % (s, c, tot[s]/1e6))
PY
exit 0

#!/usr/bin/env bash
# Decode-phase kernel breakdown: rocprofv3 trace over 4 greedy tokens at 8K.
# Per-kernel count/total/mean separates the per-token decode kernels (40
# layers x 4 tokens = 160 dispatches) from the one-off prefill kernels.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-decode-rocprof-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/prof" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "=== trace start $(date -Is) clocks $(rocm-smi --showclocks 2>/dev/null | grep -m1 sclk) ==="
timeout -k 60 1800 rocprofv3 --kernel-trace -f csv -d "$OUT/prof" -- \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 4 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv"
python3 -B - "$OUT" <<'PY'
import csv, glob, os, sys
files = glob.glob(os.path.join(sys.argv[1], "prof", "*", "*kernel_trace.csv"))
agg = {}
for f in files:
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                s = int(r["Start_Timestamp"]); e = int(r["End_Timestamp"])
            except (KeyError, ValueError):
                continue
            n, t = r["Kernel_Name"], max(0, e - s)
            c, tot = agg.get(n, (0, 0))
            agg[n] = (c + 1, tot + t)
print("\n%-62s %7s %10s %10s" % ("kernel", "calls", "total_ms", "mean_us"))
for n, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:22]:
    print("%-62s %7d %10.1f %10.1f" % (n[:62], c, t / 1e6, t / c / 1e3))
PY
exit 0

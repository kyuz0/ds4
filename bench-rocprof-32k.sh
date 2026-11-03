#!/usr/bin/env bash
# rocprofv3 kernel trace over one short prefill: which kernels own the time,
# and how much of it is q2 wstage staging versus mma/dequant.
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
OUT="$G/findings/bench-rocprof-32k-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/prof" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "=== rocprofv3 trace start $(date -Is) ==="
timeout -k 60 1800 rocprofv3 --kernel-trace -f csv -d "$OUT/prof" -- \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
grep -E "prefill frontier" "$OUT/profile.log"
ls -l "$OUT/prof" | head -5
python3 -B - "$OUT" <<'PY'
import csv, glob, os, sys
out = sys.argv[1]
files = glob.glob(os.path.join(out, "prof", "*.csv"))
if not files:
    print("no trace csv found")
    raise SystemExit(0)
tot = {}
rows = 0
for f in files:
    with open(f, newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames:
            continue
        name_col = next((c for c in rd.fieldnames if "Kernel_Name" in c or c == "Name"), None)
        dur_col = next((c for c in rd.fieldnames if "Duration" in c and "Ns" in c), None) \
            or next((c for c in rd.fieldnames if "Duration" in c), None)
        if not name_col or not dur_col:
            continue
        for r in rd:
            try:
                d = float(r[dur_col])
            except (TypeError, ValueError):
                continue
            rows += 1
            tot[r[name_col]] = tot.get(r[name_col], 0.0) + d
print("\nsampled kernel dispatches:", rows)
grand = sum(tot.values()) or 1.0
print("%-70s %12s %7s" % ("kernel", "total_ms", "share"))
for name, d in sorted(tot.items(), key=lambda kv: -kv[1])[:18]:
    print("%-70s %12.1f %6.1f%%" % (name[:70], d / 1e6, 100.0 * d / grand))
PY
exit 0

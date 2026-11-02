#!/usr/bin/env bash
# Isolate the decode-phase kernel cost by subtracting a gen=0 trace from a
# gen=4 trace of the same 8K prefill. Prefill dispatches cancel out.
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
OUT="$G/findings/bench-decode-diff-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
traced() {
  local tag="$1" gen="$2"
  mkdir -p "$OUT/$tag/prof"
  timeout -k 60 1800 rocprofv3 --kernel-trace -f csv -d "$OUT/$tag/prof" -- \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens "$gen" \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --csv "$OUT/$tag/profile.csv" > "$OUT/$tag/profile.log" 2>&1
  echo "[$tag rc] $?  $(grep -c 'prefill frontier' "$OUT/$tag/profile.log")"
}
traced g0 0
traced g4 4
python3 -B - "$OUT" <<'PY'
import csv, glob, os, sys
def agg(tag):
    a = {}
    for f in glob.glob(os.path.join(sys.argv[1], tag, "prof", "*", "*kernel_trace.csv")):
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    s = int(r["Start_Timestamp"]); e = int(r["End_Timestamp"])
                except (KeyError, ValueError): continue
                n = r["Kernel_Name"]; c, t = a.get(n, (0, 0))
                a[n] = (c + 1, t + max(0, e - s))
    return a
a0, a4 = agg("g0"), agg("g4")
rows = []
for n in set(a0) | set(a4):
    c0, t0 = a0.get(n, (0, 0)); c4, t4 = a4.get(n, (0, 0))
    rows.append((t4 - t0, c4 - c0, n, t0, t4))
rows.sort(key=lambda r: -r[0])
tot = sum(r[0] for r in rows) or 1
print("\ndecode-only delta (gen4 - gen0), total %.1f ms over 4 tokens" % (tot / 1e6))
print("%-56s %7s %9s %9s" % ("kernel", "d_calls", "delta_ms", "us/token"))
for d, dc, n, t0, t4 in rows[:16]:
    if d <= 0 and dc <= 0: continue
    print("%-56s %7d %9.2f %9.1f" % (n[:56], dc, d / 1e6, d / 1e3 / 4))
PY
exit 0

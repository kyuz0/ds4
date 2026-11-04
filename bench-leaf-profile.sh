#!/usr/bin/env bash
# Decode-only per-leaf device timing by subtraction: identical 32K prefill with 0 vs 16
# generated tokens, so prefill contributions cancel and call counts must differ by 16x40.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2|3|4|5|6|7) ;; *) echo "*** trial 1-7 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-leaf-profile$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
run() {
  timeout -k 60 1200 env DS4_V41_LEAF_PROFILE=1 \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens "$2" \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --csv "$OUT/$1.csv" > "$OUT/$1.log" 2>&1
  echo "[$1 rc] $?"
}
run g0 0
run g16 16
python3 -B - "$OUT" <<'PY'
import csv, glob, os, re, sys
out = sys.argv[1]
def leaves(p):
    txt = open(os.path.join(out, p + ".log"), errors="ignore").read()
    d = {}
    for name, ms, calls, mb, gbs in re.findall(
            r"ds4:   (\S+)\s+([0-9.]+) ms\s+(\d+) calls\s+([0-9.]+) MB total\s+([0-9.]+) GB/s", txt):
        d[name] = (float(ms), int(calls), float(mb))
    return d
a, b = leaves("g0"), leaves("g16")
if not a or not b:
    print("missing leaf tables"); sys.exit(0)
tokens, layers, ceil = 16, 40, 223.0
print(f"\n{'leaf':26s}{'d_calls':>8s}{'ms/token':>10s}{'MB/token':>10s}{'GB/s':>8s}{'%ceil':>7s}"
      f"   (expect d_calls == {tokens*layers})")
for k in sorted(set(a) | set(b)):
    ma, ca, ba = a.get(k, (0.0, 0, 0.0))
    mb_, cb, bb = b.get(k, (0.0, 0, 0.0))
    dms, dc, dmb = mb_ - ma, cb - ca, bb - ba
    if dc <= 0 and dms <= 0: continue
    if dms <= 0:
        print(f"{k[:26]:26s}{dc:8d}{'(fused)':>10s}{dmb/tokens:10.1f}{'-':>8s}{'-':>7s}")
        continue
    gbs = dmb * 1e6 / (dms * 1e-3) / 1e9 if dms > 0 else 0
    print(f"{k[:26]:26s}{dc:8d}{dms/tokens:10.2f}{dmb/tokens:10.1f}{gbs:8.0f}{gbs/ceil*100:7.0f}")
PY

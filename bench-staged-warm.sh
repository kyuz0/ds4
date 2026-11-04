#!/usr/bin/env bash
# Is the per-layer expert stall cache-miss latency or throughput? Run the identical
# decode twice back to back: a cold-vs-warm comparison separates the two.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial 1-2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
OUT="$G/findings/bench-staged-warm$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
run() {
  local arm="$1"
  timeout -k 60 1200 env DS4_ROCM_STREAM_CACHE_STATS=1 DS4_ROCM_MOE_DECODE_PROFILE=1 \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 32 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --csv "$OUT/$arm.csv" > "$OUT/$arm.log" 2>&1
  echo "=== $arm rc=$? ==="
  grep -o "staged-bytes=[0-9]* staged-jobs=[0-9]* staged-read-us=[0-9]*" "$OUT/$arm.log" | tail -1
  grep -o "calls=[0-9]*.*split_finish_missing=[0-9.]* ms" "$OUT/$arm.log" | tail -1 | grep -o "calls=[0-9]*\|split_finish_missing=[0-9.]*"
  tail -1 "$OUT/$arm.csv"
}
run cold
run warm
python3 -B - "$OUT" <<'PY'
import csv, os, re, sys
out=sys.argv[1]
def get(arm):
    log=open(os.path.join(out,arm+".log")).read()
    s=re.search(r"staged-bytes=(\d+) staged-jobs=(\d+) staged-read-us=(\d+)", log)
    c=re.search(r"MoE profile calls=(\d+)", log)
    f=re.search(r"split_finish_missing=([0-9.]+) ms", log)
    rows=list(csv.DictReader(open(os.path.join(out,arm+".csv"))))
    return (int(s.group(1)), int(s.group(2)), int(s.group(3)), int(c.group(1)),
            float(f.group(1)), float(rows[-1]["gen_tps"]))
for arm in ("cold","warm"):
    b,j,rd,c,st,tps=get(arm)
    print(f"{arm}: staged {b/1e6:.0f} MB, jobs {j}, read-total {rd/1e6:.1f} s, "
          f"wait/call {st/c*1000:.0f} us, read/call {rd/c:.0f} us, "
          f"overhead/call {st/c*1e6/c - rd/c:.0f} us, gen {tps:.2f} tps")
cb,cj,cr,cc,cs,ct=get("cold"); wb,wj,wr,wc,ws,wt=get("warm")
if cc and wc:
    print(f"wait/call cold {cs/cc*1000:.0f} us -> warm {ws/wc*1000:.0f} us "
          f"({(ws/wc)/(cs/cc)-1:+.0%})")
    print(f"decode cold {ct:.2f} -> warm {wt:.2f} tps ({(wt/ct-1)*100:+.1f}%)")
PY

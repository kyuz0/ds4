#!/usr/bin/env bash
# The soa_channel_blocks plumbing must be inert: bit-identical logits and no
# timing change against the binary captured before the patch.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
PRE=/tmp/ds4-bench-pre-soa
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$PRE" ] && [ -r "$MODEL" ] || { echo "*** pre binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-soa-inert-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
for arm in pre new; do
  bin=$([ "$arm" = pre ] && echo "$PRE" || echo "$G/ds4-bench")
  mkdir -p "$OUT/$arm/logits"
  timeout -k 60 900 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 8 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --dump-frontier-logits-dir "$OUT/$arm/logits" --csv "$OUT/$arm/profile.csv" \
    > "$OUT/$arm/profile.log" 2>&1
  echo "=== $arm rc=$? ==="; tail -1 "$OUT/$arm/profile.csv"
done
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z)>len(best) and z and all(isinstance(t,(int,float)) for t in z): best=z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
A = sorted(glob.glob(os.path.join(out,"pre","logits","*")))
B = sorted(glob.glob(os.path.join(out,"new","logits","*")))
for i,(fa,fb) in enumerate(zip(A,B)):
    a = find(json.load(open(fa))); b = find(json.load(open(fb)))
    print("frontier %d exact=%s max_abs=%.6g" % (i, a==b, max(abs(x-y) for x,y in zip(a,b))))
PY
exit 0

#!/usr/bin/env bash
# One-run confirmation that the promoted default (no env) reclaims the prefill
# reserve and stays bit-identical to the measured opt-in runs.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
REF="$(ls -dt $G/findings/bench-grow-cache-ab-trial1-* | head -1)/k_off1/logits"
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend still running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** cockpit container running ***"; exit 1; }
[ -r "$MODEL" ] && [ -r "$PROMPT" ] || { echo "*** model/prompt unreadable ***"; exit 1; }
OUT="$G/findings/bench-grow-cache-default-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT/logits" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"; echo "reference logits $REF"; } > "$OUT/provenance.txt"
./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" --ctx-frontiers 32768,36864 \
  --ctx-alloc 40960 --gen-tokens 32 --show-output --ssd-streaming \
  --ssd-streaming-cache-experts 92GB --dump-frontier-logits-dir "$OUT/logits" \
  --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
RC=$?
echo "[rc] $RC"
grep -E "decode cache grew|prefill frontier" "$OUT/profile.log"
cat "$OUT/profile.csv"
python3 -B - "$OUT/logits" "$REF" <<'PY'
import glob, json, os, sys
def find(o):
    best, stack = [], [o]
    while stack:
        c = stack.pop()
        if isinstance(c, list):
            if len(c) > len(best) and c and all(isinstance(v, (int, float)) for v in c): best = c
            else: stack.extend(c)
        elif isinstance(c, dict): stack.extend(c.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
a=sorted(glob.glob(os.path.join(sys.argv[1],'*'))); b=sorted(glob.glob(os.path.join(sys.argv[2],'*')))
for i,(fa,fb) in enumerate(zip(a,b)):
    va,vb=load(fa),load(fb)
    print("frontier %d exact=%s max_abs=%.6g"%(i, va==vb, max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

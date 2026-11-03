#!/usr/bin/env bash
# A/B for the IQ2 dequant ksigns64-LUT change: pre-LUT vs LUT binary, 32K, three
# interleaved pairs; logits must be identical.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
PRE=/tmp/ds4-bench-prelut
NEW=/tmp/ds4-bench-lut
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$PRE" ] && [ -x "$NEW" ] && [ -r "$MODEL" ] || { echo "*** binaries or model missing ***"; exit 1; }
OUT="$G/findings/bench-lut-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "pre sha256 $(sha256sum $PRE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum $NEW | cut -d' ' -f1)"
  echo "case 32K+4K append gen32, 3 interleaved pairs, 20 s warm-up"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  timeout -k 60 1200 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$? ==="
}
for r in 1 2 3; do run_one "pre_$r" "$PRE"; run_one "lut_$r" "$NEW"; done
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def v(c,i,k):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return float("nan")
    d = list(csv.DictReader(open(p)))
    return float(d[i][k]) if len(d) > i else float("nan")
for i, lab in ((0,"32K prefill tps"), (1,"4K append tps")):
    rs=[]
    for r in (1,2,3):
        a, b = v("pre_%d"%r,i,"prefill_tps"), v("lut_%d"%r,i,"prefill_tps")
        if a==a and b==b:
            rs.append((b/a-1)*100); print("%s pair%d pre=%.2f lut=%.2f -> %+.1f%%" % (lab,r,a,b,rs[-1]))
    if rs: print("  mean %+.1f%% (%d/3 positive)\n" % (sum(rs)/len(rs), sum(1 for x in rs if x>0)))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z)>len(best) and z and all(isinstance(t,(int,float)) for t in z): best=z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
for r in (1,2,3):
    A = sorted(glob.glob(os.path.join(out,"pre_%d"%r,"logits","*")))
    B = sorted(glob.glob(os.path.join(out,"lut_%d"%r,"logits","*")))
    for i,(fa,fb) in enumerate(zip(A,B)):
        a,b = find(json.load(open(fa))), find(json.load(open(fb)))
        print("pair%d frontier %d exact=%s max_abs=%.6g" % (r,i,a==b,max(abs(x-y) for x,y in zip(a,b))))
PY
exit 0

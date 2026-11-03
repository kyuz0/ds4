#!/usr/bin/env bash
# Generalisation gate: the promoted changes must be bit-identical on a prompt
# that was never used for tuning. base binary vs current defaults, 8K, gen 8.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=/tmp/prompt-alt.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] && [ -r "$PROMPT" ] || { echo "*** inputs missing ***"; exit 1; }
OUT="$G/findings/bench-alt-prompt-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "prompt $PROMPT sha256 $(sha256sum $PROMPT | cut -d' ' -f1) bytes $(stat -c '%s' $PROMPT)"
  echo "case 8K frontier, gen 8, base vs defaults, unseen prompt"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  timeout -k 60 900 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 8 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$? ==="; grep -E "decoded text" "$dir/profile.log" | head -1 | cut -c1-120
}
run_one base "$BASE"; run_one new "$G/ds4-bench"
diff <(grep "decoded text" "$OUT/base/profile.log") <(grep "decoded text" "$OUT/new/profile.log") >/dev/null && echo "generated text identical" || echo "generated text DIFFERS"
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
for c in ("base","new"):
    p = os.path.join(out, c, "profile.csv")
    if os.path.exists(p):
        for r in list(csv.DictReader(open(p)))[-1:]:
            print("%-4s prefill %s tps decode %s tps" % (c, r["prefill_tps"], r["gen_tps"]))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z)>len(best) and z and all(isinstance(t,(int,float)) for t in z): best=z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
A = sorted(glob.glob(os.path.join(out,"base","logits","*")))
B = sorted(glob.glob(os.path.join(out,"new","logits","*")))
for i,(fa,fb) in enumerate(zip(A,B)):
    a,b = find(json.load(open(fa))), find(json.load(open(fb)))
    print("frontier %d exact=%s max_abs=%.6g argmax_same=%s" % (i, a==b,
          max(abs(x-y) for x,y in zip(a,b)),
          max(range(len(a)), key=lambda k: a[k]) == max(range(len(b)), key=lambda k: b[k])))
PY
exit 0

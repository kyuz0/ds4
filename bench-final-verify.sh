#!/usr/bin/env bash
# Long-generation equivalence for the promoted defaults: 128 greedy tokens at
# 8K against the pre-session binary, bit-identical logits plus identical text.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] || { echo "*** base binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-final-verify-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 8K, gen 128 greedy, defaults only; text and logits must match base"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  timeout -k 60 1200 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 128 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$?"
  grep -E "decoded text" "$dir/profile.log" | head -2
}
run_one base "$BASE"
run_one new  "$G/ds4-bench"
echo "=== generated text diff (base vs new) ==="
diff <(grep "decoded text" "$OUT/base/profile.log") <(grep "decoded text" "$OUT/new/profile.log") && echo "text identical"
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
for c in ("base", "new"):
    d = list(csv.DictReader(open(os.path.join(out, c, "profile.csv"))))
    for r in d:
        print("%-5s ctx=%-5s tps=%-7s gen_tps=%-6s" % (c, r["ctx_tokens"], r["prefill_tps"], r["gen_tps"]))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z) > len(best) and z and all(isinstance(t,(int,float)) for t in z): best = z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
A = sorted(glob.glob(os.path.join(out, "base", "logits", "*")))
B = sorted(glob.glob(os.path.join(out, "new", "logits", "*")))
for i,(fa,fb) in enumerate(zip(A,B)):
    va, vb = load(fa), load(fb)
    print("frontier %d exact=%s max_abs=%.6g" % (i, va==vb, max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

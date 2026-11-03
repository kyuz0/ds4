#!/usr/bin/env bash
# Two interleaved 110K pairs measuring decode: base binary vs promoted defaults.
# Single frontier to stay inside the supervisor window; logits dumped for the
# bit-exact gate.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] || { echo "*** base binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-110k-decode-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 110592 frontier, gen 32, two interleaved base/new pairs, 15 s warm-up"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  echo "=== $name start $(date -Is) ==="
  timeout -k 60 1200 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 110592 --ctx-alloc 118784 --gen-tokens 32 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$? $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
}
run_one b1 "$BASE"; run_one n1 "$G/ds4-bench"
run_one b2 "$BASE"; run_one n2 "$G/ds4-bench"
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def row(c):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return None
    d = list(csv.DictReader(open(p)))
    return d[-1] if d else None
for r in (1,2):
    b, n = row("b%d"%r), row("n%d"%r)
    if not b or not n: print("pair%d missing" % r); continue
    print("pair%d prefill %.2f -> %.2f tps; decode %.3f -> %.3f tps (%+.1f%%)"
          % (r, float(b["prefill_tps"]), float(n["prefill_tps"]),
             float(b["gen_tps"]), float(n["gen_tps"]),
             (float(n["gen_tps"])/float(b["gen_tps"])-1)*100))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z)>len(best) and z and all(isinstance(t,(int,float)) for t in z): best=z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
for r in (1,2):
    A = sorted(glob.glob(os.path.join(out,"b%d"%r,"logits","*")))
    B = sorted(glob.glob(os.path.join(out,"n%d"%r,"logits","*")))
    for i,(fa,fb) in enumerate(zip(A,B)):
        a,b = find(json.load(open(fa))), find(json.load(open(fb)))
        print("pair%d frontier %d exact=%s max_abs=%.6g" % (r, i, a==b, max(abs(x-y) for x,y in zip(a,b))))
PY
exit 0

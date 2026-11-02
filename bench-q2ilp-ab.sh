#!/usr/bin/env bash
# A/B: previous binary vs Q2 decode matvec with independent integer accumulators.
# Integer sums are exact/associative, so frontier logits must stay bit-identical.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
PREV=/tmp/ds4-bench-prev
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -x "$PREV" ] && [ -r "$MODEL" ] || { echo "*** prev binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-q2ilp-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "prev sha256 $(sha256sum $PREV | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32768+4096 gen32 cache92GB suffix off; order prev,new,prev,new"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  echo "=== $name $(basename $bin): start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one prev1 "$PREV"
run_one new1  "$G/ds4-bench"
run_one prev2 "$PREV"
run_one new2  "$G/ds4-bench"
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("prev1", "new1", "prev2", "new2"):
    d = rows(c)
    for r in d:
        print("%-6s ctx=%-6s tps=%-7s append_ms=%-10s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def gen(c):
    d = rows(c); return float(d[-1]["gen_tps"]) if d else float("nan")
p = (gen("prev1") + gen("prev2")) / 2; n = (gen("new1") + gen("new2")) / 2
print("\ndecode mean prev=%.3f new=%.3f (%+.1f%%)" % (p, n, (n / p - 1) * 100 if p == p else float("nan")))
def find(o):
    best, stack = [], [o]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            if len(x) > len(best) and x and all(isinstance(v, (int, float)) for v in x): best = x
            else: stack.extend(x)
        elif isinstance(x, dict): stack.extend(x.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
ref = sorted(glob.glob(os.path.join(out, "prev1", "logits", "*")))
for c in ("new1", "new2", "prev2"):
    cur = sorted(glob.glob(os.path.join(out, c, "logits", "*")))
    for i, (fa, fb) in enumerate(zip(ref, cur)):
        va, vb = load(fa), load(fb)
        print("%-6s frontier %d exact=%s max_abs=%.6g" % (c, i, va == vb, max(abs(x - y) for x, y in zip(va, vb))))
PY
exit 0

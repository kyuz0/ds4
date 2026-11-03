#!/usr/bin/env bash
# A/B of __launch_bounds__(128,4) on ds4_mmq_compact_mul_mat_q (192 VGPR -> lower
# register budget -> higher occupancy, spills possible). Values must stay
# bit-identical: register allocation does not change arithmetic order.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] || { echo "*** base binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-jlimit-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32K+4K gen32 cache92GB suffix off; order base,new,base,new"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   clock: /'; }
run_one() {
  local name="$1" bin="$2"
  local EXTRA=""
  case "$name" in jl*) EXTRA="DS4_ROCM_V41_COMPACT_JLIMIT=1";; esac; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  echo "=== $name $(basename $bin): start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 $EXTRA "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one base1 "$BASE"
run_one jl1  "$G/ds4-bench"
run_one base2 "$BASE"
run_one jl2  "$G/ds4-bench"
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("base1","jl1","base2","jl2"):
    for r in rows(c):
        print("%-5s ctx=%-6s prefill_tps=%-7s append_ms=%-10s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def v(c,i,k):
    d = rows(c); return float(d[i][k]) if len(d) > i else float("nan")
b = (v("base1",0,"prefill_tps") + v("base2",0,"prefill_tps"))/2
o = (v("jl1",0,"prefill_tps") + v("jl2",0,"prefill_tps"))/2
print("\n32K prefill base=%.2f jl=%.2f -> %+.1f%% disjoint_hi=%s" % (b, o, (o/b-1)*100, min(v("jl1",0,"prefill_tps"),v("jl2",0,"prefill_tps")) > max(v("base1",0,"prefill_tps"),v("base2",0,"prefill_tps"))))
def find(x):
    best, stack = [], [x]
    while stack:
        y = stack.pop()
        if isinstance(y, list):
            if len(y) > len(best) and y and all(isinstance(t,(int,float)) for t in y): best = y
            else: stack.extend(y)
        elif isinstance(y, dict): stack.extend(y.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
ref = sorted(glob.glob(os.path.join(out, "base1", "logits", "*")))
for c in ("jl1","jl2","base2"):
    cur = sorted(glob.glob(os.path.join(out, c, "logits", "*")))
    for i,(fa,fb) in enumerate(zip(ref,cur)):
        va, vb = load(fa), load(fb)
        print("%-5s frontier %d exact=%s max_abs=%.6g" % (c, i, va==vb, max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

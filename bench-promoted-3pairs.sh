#!/usr/bin/env bash
# Protocol-compliant validation of the promoted defaults: three interleaved
# 32K pairs against the pre-session binary (>=3 pairs in one session is the
# rule adopted after the PAD=8 and non-packed-indexer false positives).
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] || { echo "*** base binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-promoted-3pairs-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"; echo "new sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32K+4K gen32, 3 interleaved base/new pairs, 20 s warm-up each"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" bin="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  timeout -k 60 2400 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name $(basename $bin) rc=$? $(date -Is) ==="
}
for r in 1 2 3; do
  run_one "base_$r" "$BASE"
  run_one "new_$r" "$G/ds4-bench"
done
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def v(c,i,k):
    d = rows(c); return float(d[i][k]) if len(d) > i else float("nan")
for i, lab in ((0,"32K prefill tps"), (1,"4K append tps")):
    rs = []
    for r in (1,2,3):
        b, n = v("base_%d"%r,i,"prefill_tps"), v("new_%d"%r,i,"prefill_tps")
        rs.append((n/b-1)*100)
        print("%s pair%d base=%.2f new=%.2f -> %+.1f%%" % (lab, r, b, n, rs[-1]))
    m = sum(rs)/len(rs); sd = (sum((x-m)**2 for x in rs)/(len(rs)-1))**0.5
    print("  mean %+.1f%% sd %.1f positive %d/3\n" % (m, sd, sum(1 for x in rs if x>0)))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z)>len(best) and z and all(isinstance(t,(int,float)) for t in z): best=z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
for r in (1,2,3):
    A = sorted(glob.glob(os.path.join(out,"base_%d"%r,"logits","*")))
    B = sorted(glob.glob(os.path.join(out,"new_%d"%r,"logits","*")))
    for i,(fa,fb) in enumerate(zip(A,B)):
        va,vb = load(fa),load(fb)
        print("pair%d frontier %d exact=%s max_abs=%.6g" % (r,i,va==vb,max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

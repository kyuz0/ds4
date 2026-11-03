#!/usr/bin/env bash
# Runtime indexer batch: memory cost first (graph GiB from the plan), then the
# 32K A/B against the default 32 rows, with exact-logit checks.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-index-batch-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case memory probe at 2K then 32K A/B default(32) vs 128 vs 256"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
echo "=== memory probe (2K prefill, graph GiB from plan) ==="
for b in 32 128 256; do
  envs=(); [ "$b" != 32 ] && envs+=("DS4_ROCM_V41_INDEX_BATCH=$b")
  timeout -k 60 600 env "${envs[@]}" ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 40960 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB --csv "$OUT/mem_$b.csv" > "$OUT/mem_$b.log" 2>&1
  echo "batch=$b rc=$? $(grep -m1 'memory:' "$OUT/mem_$b.log" | sed 's/.*graph, //')"
  grep -m1 "memory:" "$OUT/mem_$b.log" | sed 's/^/   /'
done
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" b="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=(); [ "$b" != 32 ] && envs+=("DS4_ROCM_V41_INDEX_BATCH=$b")
  timeout -k 60 2400 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" ./ds4-bench --rocm -m "$MODEL" \
    --prompt-file "$PROMPT" --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name batch=$b rc=$? ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
}
run_one b32a 32
run_one b128 128
run_one b256 256
run_one b32b 32
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("b32a","b128","b256","b32b"):
    for r in rows(c):
        print("%-5s ctx=%-6s tps=%-7s append_ms=%-10s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def v(c,i,k):
    d = rows(c); return float(d[i][k]) if len(d) > i else float("nan")
base=(v("b32a",0,"prefill_tps")+v("b32b",0,"prefill_tps"))/2
for c in ("b128","b256"):
    x=v(c,0,"prefill_tps"); print("%s 32K prefill %.2f vs base %.2f -> %+.1f%%" % (c,x,base,(x/base-1)*100))
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
ref=sorted(glob.glob(os.path.join(out,"b32a","logits","*")))
for c in ("b128","b256","b32b"):
    cur=sorted(glob.glob(os.path.join(out,c,"logits","*")))
    for i,(fa,fb) in enumerate(zip(ref,cur)):
        va,vb=load(fa),load(fb)
        print("%-5s frontier %d exact=%s max_abs=%.6g argmax_same=%s" % (c,i,va==vb,
              max(abs(x-y) for x,y in zip(va,vb)),
              max(range(len(va)),key=lambda k:va[k])==max(range(len(vb)),key=lambda k:vb[k])))
PY
exit 0

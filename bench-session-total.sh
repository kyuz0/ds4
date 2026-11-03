#!/usr/bin/env bash
# Aggregate session delta: the pre-session binary (before the fused-epilogue
# MMQ promotion) versus the current default plus the three opt-in flags.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE=/tmp/ds4-bench-base
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -x "$BASE" ] && [ -r "$MODEL" ] || { echo "*** base binary or model missing ***"; exit 1; }
OUT="$G/findings/bench-session-total-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "base sha256 $(sha256sum $BASE | cut -d' ' -f1)"
  echo "new  sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32K pair + 110K pair; new arm enables Q8_FUSE_BF16, Q8_FUSE_QAKV, Q8_BLOCKS4_PREFETCH"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" bin="$2" flags="$3" frontiers="$4" alloc="$5"
  local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=()
  if [ "$flags" = 1 ]; then
    envs+=("DS4_ROCM_Q8_FUSE_BF16=1" "DS4_ROCM_V41_Q8_FUSE_QAKV=1" "DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=1")
  fi
  echo "=== $name flags=$flags ctx=$frontiers: start $(date -Is) ==="
  timeout -k 60 2400 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" "$bin" --rocm -m "$MODEL" \
      --prompt-file "$PROMPT" --ctx-frontiers "$frontiers" --ctx-alloc "$alloc" \
      --gen-tokens 64 --show-output --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$? $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
}
run_one k_base "$BASE" 0 32768,36864 40960
run_one k_new  "$G/ds4-bench" 1 32768,36864 40960
run_one h_base "$BASE" 0 110592,114688 118784
run_one h_new  "$G/ds4-bench" 1 110592,114688 118784
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("k_base","k_new","h_base","h_new"):
    for r in rows(c):
        print("%-6s ctx=%-6s prefill_tps=%-7s append_ms=%-10s gen_tps=%s" % (
            c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def v(c,i,k):
    d = rows(c); return float(d[i][k]) if len(d) > i else float("nan")
for tag, a, b in (("32K","k_base","k_new"), ("110K","h_base","h_new")):
    for i, lab in ((0,"prefill tps"), (1,"4K append tps")):
        x, y = v(a,i,"prefill_tps"), v(b,i,"prefill_tps")
        print("%s %-14s %.2f -> %.2f  %+.1f%%" % (tag, lab, x, y, (y/x-1)*100))
    print("%s decode tps   %.2f -> %.2f  %+.1f%%" % (tag, v(a,1,"gen_tps"), v(b,1,"gen_tps"), (v(b,1,"gen_tps")/v(a,1,"gen_tps")-1)*100))
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
for a,b in (("k_base","k_new"),("h_base","h_new")):
    A = sorted(glob.glob(os.path.join(out, a, "logits", "*")))
    B = sorted(glob.glob(os.path.join(out, b, "logits", "*")))
    for i,(fa,fb) in enumerate(zip(A,B)):
        va, vb = load(fa), load(fb)
        print("%s vs %s frontier %d exact=%s max_abs=%.6g argmax_same=%s" % (a,b,i,va==vb,
              max(abs(x-y) for x,y in zip(va,vb)),
              max(range(len(va)), key=lambda k: va[k]) == max(range(len(vb)), key=lambda k: vb[k])))
PY
exit 0

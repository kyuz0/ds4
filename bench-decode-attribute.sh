#!/usr/bin/env bash
# Attribute the decode gain: current default vs +fuse vs +fuse+unroll vs +all.
# 8K context, gen 64, one interleaved round of each (cheap, high token count).
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-decode-attr-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 8K gen64; arms default, +fuse, +fuse+unroll, +all(bf16 too); 2 rounds"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" mode="$2"; local dir="$OUT/$name"; mkdir -p "$dir" || return 1
  soak
  local envs=()
  case "$mode" in
    fuse) envs+=("DS4_ROCM_V41_Q8_FUSE_QAKV=1");;
    fuse_unroll) envs+=("DS4_ROCM_V41_Q8_FUSE_QAKV=1" "DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=1");;
    all) envs+=("DS4_ROCM_V41_Q8_FUSE_QAKV=1" "DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=1" "DS4_ROCM_Q8_FUSE_BF16=1");;
  esac
  timeout -k 60 900 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" ./ds4-bench --rocm -m "$MODEL" \
    --prompt-file "$PROMPT" --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 64 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name($mode) rc=$? ==="
}
for r in 1 2; do
  run_one "d_$r" default
  run_one "f_$r" fuse
  run_one "fu_$r" fuse_unroll
  run_one "a_$r" all
done
python3 -B - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
def gen(c):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return None
    d = list(csv.DictReader(open(p)))
    return float(d[-1]["gen_tps"]) if d else None
rows = {}
for tag in ("d", "f", "fu", "a"):
    for r in (1, 2):
        rows[(tag, r)] = gen("%s_%d" % (tag, r))
base = [rows[("d", r)] for r in (1, 2)]
print("default decode tps:", base)
for tag, label in (("f", "+fuse"), ("fu", "+fuse+unroll"), ("a", "+all(bf16)")):
    vals = [rows[(tag, r)] for r in (1, 2)]
    ratios = [(v / b - 1) * 100 for v, b in zip(vals, base) if v and b]
    print("%-14s tps=%s  vs default %s" % (label, vals, ["%+.1f%%" % x for x in ratios]))
PY
exit 0

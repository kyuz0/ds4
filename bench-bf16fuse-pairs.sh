#!/usr/bin/env bash
# Three interleaved 32K pairs for the Q8 bf16-epilogue fusion, to separate the
# +4.2% first reading from the +0.7% confirmation.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-bf16fuse-pairs-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32K+4K gen32, 3 interleaved off/on pairs, 20 s warm-up each"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   clock: /'; }
run_one() {
  local name="$1" fuse="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=(); [ "$fuse" = 1 ] && envs+=("DS4_ROCM_Q8_FUSE_BF16=1")
  timeout -k 60 1200 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name fuse=$fuse rc=$? ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
}
for r in 1 2 3; do
  run_one "off_$r" 0
  run_one "on_$r" 1
done
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def v(c, i, k):
    d = rows(c)
    return float(d[i][k]) if len(d) > i else float("nan")
for i, lab in ((0, "32K prefill tps"), (1, "4K append tps")):
    ratios = []
    for r in (1, 2, 3):
        o, n = v("off_%d" % r, i, "prefill_tps"), v("on_%d" % r, i, "prefill_tps")
        ratios.append((n / o - 1) * 100)
        print("%s pair%d off=%.2f on=%.2f -> %+.1f%%" % (lab, r, o, n, ratios[-1]))
    m = sum(ratios) / len(ratios)
    sd = (sum((x - m) ** 2 for x in ratios) / (len(ratios) - 1)) ** 0.5
    print("  mean %+.1f%% sd %.1f positive %d/3\n" % (m, sd, sum(1 for x in ratios if x > 0)))
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
for r in (1,2,3):
    A = sorted(glob.glob(os.path.join(out, "off_%d" % r, "logits", "*")))
    B = sorted(glob.glob(os.path.join(out, "on_%d" % r, "logits", "*")))
    for i, (fa, fb) in enumerate(zip(A, B)):
        va, vb = load(fa), load(fb)
        print("pair%d frontier %d exact=%s max_abs=%.6g" % (r, i, va == vb, max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

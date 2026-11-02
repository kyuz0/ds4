#!/usr/bin/env bash
# Confirms the promoted decode-cache growth under a clock-soaked protocol:
# fixed warm-up before every arm so both arms start from the same boost state,
# and rocm-smi clock/power samples recorded per arm. DVFS is 'auto' with a
# 2900 MHz ceiling, which is the most likely source of our 3-7% A/B spread.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-soak-grow-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "clocks idle: $(rocm-smi --showclocks 2>/dev/null | grep -m1 sclk || echo n/a)"
  echo "perflevel: $(rocm-smi --showperflevel 2>/dev/null | grep -m1 'Performance Level' || echo n/a)"
  echo "case 32768+4096 gen32 cache92GB suffix off; soak 20s before every arm; order grow,grow-off,grow,grow-off"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() {
  timeout 120 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   soak clock: /'
}
run_one() {
  local name="$1" grow="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=()
  [ "$grow" = 0 ] && envs+=("DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL=0")
  echo "=== $name grow=$grow: start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   end clock: /'
  grep -E "decode cache grew|prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one on_a 1
run_one off_a 0
run_one on_b 1
run_one off_b 0
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("on_a", "off_a", "on_b", "off_b"):
    d = rows(c)
    for r in d:
        print("%-6s ctx=%-6s tps=%-7s append_ms=%-10s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def gen(c):
    d = rows(c); return float(d[-1]["gen_tps"]) if d else float("nan")
on = [gen("on_a"), gen("on_b")]; off = [gen("off_a"), gen("off_b")]
print("\ndecode on(grow)=%s off=%s" % (on, off))
if all(x == x for x in on + off):
    mo, mf = sum(on) / 2, sum(off) / 2
    print("  mean %.3f vs %.3f (%+.1f%%), disjoint=%s" % (mo, mf, (mo / mf - 1) * 100, min(on) > max(off)))
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
ref = sorted(glob.glob(os.path.join(out, "off_a", "logits", "*")))
for c in ("on_a", "off_b", "on_b"):
    cur = sorted(glob.glob(os.path.join(out, c, "logits", "*")))
    for i, (fa, fb) in enumerate(zip(ref, cur)):
        va, vb = load(fa), load(fb)
        print("%-6s frontier %d exact=%s max_abs=%.6g" % (c, i, va == vb, max(abs(x - y) for x, y in zip(va, vb))))
PY
exit 0

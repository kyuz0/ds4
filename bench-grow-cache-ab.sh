#!/usr/bin/env bash
# bench-grow-cache-ab.sh - A/B/A/B verification of reclaiming the prefill expert
# reserve as decode cache (DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL).
# Alternating order cancels drift; frontier logits must stay bit-identical.
# Run ONLY through supervisor.sh with trial=1.
set -uo pipefail

G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"
case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1

if curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "*** DS4 backend is still running; run via supervisor.sh ***"; exit 1
fi
CN="${CONTAINER_NAME:-}"
if [ -z "$CN" ]; then
  CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
fi
if [ -n "$CN" ] && podman ps --format '{{.Names}}' 2>/dev/null | grep -qx -- "$CN"; then
  echo "*** DS4 cockpit container is still running; run via supervisor.sh ***"; exit 1
fi
[ -r "$MODEL" ] && [ -r "$PROMPT" ] || { echo "*** model/prompt unreadable ***"; exit 1; }

OUT="$G/findings/bench-grow-cache-ab-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1) mtime $(stat -c '%y' ds4-bench)"
  echo "model sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "memory $(free -h | sed -n '2p')"
  echo "case: 32768 base + 4096 append, gen 32, cache 92GB, suffix off; order off,on,off,on"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

run_one() {
  local name="$1" grow="$2" base="$3" append="$4" alloc="$5"
  local dir="$OUT/$name"
  mkdir -p "$dir/logits" || return 1
  local envs=()
  [ "$grow" = 1 ] && envs+=("DS4_ROCM_V41_GROW_CACHE_AFTER_PREFILL=1")
  local frontiers="$base,$append"
  echo "=== $name grow=$grow frontiers=$frontiers: start $(date -Is) ==="
  timeout -k 60 2000 env "${envs[@]}" \
    DS4_BENCH_DEV_STAT=1 \
    ./ds4-bench --rocm \
      -m "$MODEL" \
      --prompt-file "$PROMPT" \
      --ctx-frontiers "$frontiers" \
      --ctx-alloc "$alloc" \
      --gen-tokens 32 \
      --show-output \
      --ssd-streaming \
      --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" \
      --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?
  echo "=== $name: rc=$rc done $(date -Is) ==="
  grep -E "prefill frontier|decode cache grew|V4.1 SSD prepared" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}

# 32K base: four alternating runs, cheap enough to measure noise.
run_one k_off1 0 32768 36864 40960
run_one k_on1  1 32768 36864 40960
run_one k_off2 0 32768 36864 40960
run_one k_on2  1 32768 36864 40960
# 110K confirmation pair (primary workload).
run_one h_off  0 110592 114688 118784
run_one h_on   1 110592 114688 118784

echo "=== metrics ==="
python3 -B - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
def rows(cfg):
    p = os.path.join(out, cfg, "profile.csv")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))
data = {c: rows(c) for c in ("k_off1", "k_on1", "k_off2", "k_on2", "h_off", "h_on")}
for c, d in data.items():
    for r in d:
        print("%-7s ctx=%-6s prefill_ms=%-12s tps=%-7s gen_tps=%s"
              % (c, r.get("ctx_tokens"), r.get("prefill_ms"), r.get("prefill_tps"), r.get("gen_tps")))
def gen(c):
    d = data[c]
    return float(d[-1]["gen_tps"]) if d else float("nan")
def app(c):
    d = data[c]
    return float(d[-1]["prefill_ms"]) if len(d) > 1 else float("nan")
off = [gen("k_off1"), gen("k_off2")]
on = [gen("k_on1"), gen("k_on2")]
print("\n32K decode tps: off=%s on=%s" % (off, on))
if all(x == x for x in off + on):
    mo, mn = sum(off) / 2, sum(on) / 2
    print("  mean off %.3f on %.3f delta %+.1f%%" % (mo, mn, (mn - mo) / mo * 100))
print("32K append ms: off=%s on=%s" % ([app("k_off1"), app("k_off2")], [app("k_on1"), app("k_on2")]))
print("110K append ms: off=%.0f on=%.0f" % (app("h_off"), app("h_on")))
print("110K decode tps: off=%.3f on=%.3f" % (gen("h_off"), gen("h_on")))
PY

echo "=== numerics: logits must be bit-identical (off vs on) ==="
python3 -B - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
def find(o):
    best, stack = [], [o]
    while stack:
        c = stack.pop()
        if isinstance(c, list):
            if len(c) > len(best) and c and all(isinstance(v, (int, float)) for v in c):
                best = c
            else:
                stack.extend(c)
        elif isinstance(c, dict):
            stack.extend(c.values())
    return best
def load(p):
    with open(p) as fh:
        return find(json.load(fh))
for pair in (("k_off1", "k_on1"), ("h_off", "h_on")):
    a = sorted(glob.glob(os.path.join(out, pair[0], "logits", "*")))
    b = sorted(glob.glob(os.path.join(out, pair[1], "logits", "*")))
    for i, (fa, fb) in enumerate(zip(a, b)):
        va, vb = load(fa), load(fb)
        if len(va) != len(vb):
            print("%s frontier %d shape mismatch" % (pair, i)); continue
        arg_a = max(range(len(va)), key=lambda k: va[k])
        arg_b = max(range(len(vb)), key=lambda k: vb[k])
        print("%s frontier %d exact=%s argmax %d/%d same=%s max_abs=%.6g"
              % (pair, i, va == vb, arg_a, arg_b, arg_a == arg_b,
                 max(abs(x - y) for x, y in zip(va, vb))))
PY
exit 0

#!/usr/bin/env bash
# Register-prefetch A/B for moe_down_q2K_hotlist_wmma_n2_kernel.
# Opt-in via DS4_ROCM_V41_DOWN_PREFETCH. Values unchanged -> frontier logits
# must be bit-identical. Fixed warm-up + per-arm clock samples (DVFS is auto).
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
OUT="$G/findings/bench-q8wide-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "perflevel $(rocm-smi --showperflevel 2>/dev/null | grep -m1 'Performance Level')"
  echo "case 32768+4096 gen32 cache92GB suffix off; soak 20s; order off,on,off,on (q8 wide tile)" >/dev/null
  echo "case 32768+4096 gen32 cache92GB suffix off; 20s warm-up; order off,on,off,on (q8 wide tile)"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   clock: /'; }
run_one() {
  local name="$1" pf="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=()
  [ "$pf" = 1 ] && envs+=("DS4_ROCM_Q8_WIDE_TILE=1")
  echo "=== $name tile=$pf: start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one off_a 0
run_one on_a 1
run_one off_b 0
run_one on_b 1
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("off_a", "on_a", "off_b", "on_b"):
    d = rows(c)
    for r in d:
        print("%-6s ctx=%-6s tps=%-7s append_ms=%-10s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def val(c, i, k):
    d = rows(c)
    return float(d[i][k]) if len(d) > i else float("nan")
for i, label in ((0, "32K prefill"), (1, "4K append")):
    off = [val("off_a", i, "prefill_tps"), val("off_b", i, "prefill_tps")]
    on = [val("on_a", i, "prefill_tps"), val("on_b", i, "prefill_tps")]
    mo, mn = sum(off) / 2, sum(on) / 2
    print("%s tps: off=%s on=%s -> %+.1f%% disjoint=%s" % (label, off, on, (mn / mo - 1) * 100, min(on) > max(off)))
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

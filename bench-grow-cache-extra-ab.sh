#!/usr/bin/env bash
# A/B/A/B: headroom-only cache growth (extra=0) vs growth that also consumes
# genuinely free memory (extra=25%, lazily guard-limited). Resume/logits must
# stay bit-identical; only cache residency changes.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -r "$MODEL" ] && [ -r "$PROMPT" ] || { echo "*** model/prompt unreadable ***"; exit 1; }
OUT="$G/findings/bench-grow-cache-extra-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"; echo "memory $(free -h | sed -n '2p')"
  echo "case 32768+4096 gen32 cache92GB suffix off; order extra0, extra25, extra0, extra25"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1" pct="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  echo "=== $name extra_pct=$pct: start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 DS4_ROCM_V41_GROW_CACHE_EXTRA_PCT="$pct" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc done $(date -Is) ==="
  grep -E "decode cache grew|prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one e0_a 0
run_one e25_a 25
run_one e0_b 0
run_one e25_b 25
python3 -B - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
for c in ("e0_a", "e25_a", "e0_b", "e25_b"):
    d = rows(c)
    for r in d:
        print("%-6s ctx=%-6s tps=%-7s gen_tps=%s" % (c, r["ctx_tokens"], r["prefill_tps"], r["gen_tps"]))
def gen(c):
    d = rows(c); return float(d[-1]["gen_tps"]) if d else float("nan")
a = [gen("e0_a"), gen("e0_b")]; b = [gen("e25_a"), gen("e25_b")]
print("\ndecode tps extra0=%s extra25=%s" % (a, b))
if all(x == x for x in a + b):
    print("  mean %.3f -> %.3f  (%+.1f%%)" % (sum(a)/2, sum(b)/2, (sum(b)/2 - sum(a)/2) / (sum(a)/2) * 100))
PY
python3 -B - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
def find(o):
    best, stack = [], [o]
    while stack:
        c = stack.pop()
        if isinstance(c, list):
            if len(c) > len(best) and c and all(isinstance(v, (int, float)) for v in c): best = c
            else: stack.extend(c)
        elif isinstance(c, dict): stack.extend(c.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
for pair in (("e0_a", "e25_a"), ("e0_b", "e25_b")):
    A = sorted(glob.glob(os.path.join(out, pair[0], "logits", "*")))
    B = sorted(glob.glob(os.path.join(out, pair[1], "logits", "*")))
    for i, (fa, fb) in enumerate(zip(A, B)):
        va, vb = load(fa), load(fb)
        print("%s frontier %d exact=%s max_abs=%.6g" % (pair, i, va == vb, max(abs(x-y) for x, y in zip(va, vb))))
PY
exit 0

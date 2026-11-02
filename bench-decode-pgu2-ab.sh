#!/usr/bin/env bash
# Decode occupancy sweep: V4.1 GATE/DOWN wave count (2/4/8) and the six-slot
# pointer preload. Per the kernel contract these change only how many rows one
# block owns, so results must stay bit-identical; only time may move.
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
OUT="$G/findings/bench-decode-pgu2-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"; echo "memory $(free -h | sed -n '2p')"
  echo "case 32768+4096 gen32 cache92GB suffix off"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1"; shift; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  echo "=== $name env[$*]: start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 "$@" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768,36864 --ctx-alloc 40960 --gen-tokens 32 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "partial gate/up|prefill frontier|inactive job set|failed" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one base1
run_one pgu1 DS4_ROCM_V41_ENABLE_PARTIAL_GATEUP=1
run_one base2
run_one pgu2 DS4_ROCM_V41_ENABLE_PARTIAL_GATEUP=1
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
names = ["base1", "pgu1", "base2", "pgu2"]
for c in names:
    d = rows(c)
    for r in d:
        print("%-6s ctx=%-6s tps=%-7s append_ms=%-10s gen_tps=%s"
              % (c, r["ctx_tokens"], r["prefill_tps"], r["prefill_ms"], r["gen_tps"]))
def gen(c):
    d = rows(c); return float(d[-1]["gen_tps"]) if d else float("nan")
print()
b = (gen("base1") + gen("base2")) / 2
for c in ("pgu1", "pgu2"):
    print("%s vs base mean: decode %+.1f%%" % (c, (gen(c) / b - 1) * 100))
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
ref = sorted(glob.glob(os.path.join(out, "base1", "logits", "*")))
for c in ("pgu1", "pgu2", "base2"):
    cur = sorted(glob.glob(os.path.join(out, c, "logits", "*")))
    for i, (fa, fb) in enumerate(zip(ref, cur)):
        va, vb = load(fa), load(fb)
        print("%-6s frontier %d exact=%s max_abs=%.6g" % (c, i, va == vb, max(abs(x - y) for x, y in zip(va, vb))))
PY
exit 0

#!/usr/bin/env bash
# Cold-prefill TTFT diagnostic: a 2K prefill in a fresh process pays for expert
# cache population (33 s at 62 tps in the ascending run, vs 178 tps warm).
# Compare default auto-seed against explicit preload counts.
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
OUT="$G/findings/bench-cold-prefill-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case cold 2K prefill, gen 0, cache 92GB; preload default / 2048 / 0"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_one() {
  local name="$1" preload="$2"; local dir="$OUT/$name"; mkdir -p "$dir" || return 1
  local envs=()
  echo "=== $name preload=${preload:-default}: start $(date -Is) ==="
  if [ -n "$preload" ]; then
    timeout -k 60 900 env DS4_BENCH_DEV_STAT=1 \
      ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
        --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 0 \
        --ssd-streaming --ssd-streaming-cache-experts 92GB \
        --ssd-streaming-preload-experts "$preload" \
        --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  else
    timeout -k 60 900 env DS4_BENCH_DEV_STAT=1 \
      ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
        --ctx-frontiers 2048 --ctx-alloc 8192 --gen-tokens 0 \
        --ssd-streaming --ssd-streaming-cache-experts 92GB \
        --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  fi
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "prefill frontier|preload|cache" "$dir/profile.log" | sed 's/^/   /' | head -6
  return "$rc"
}
run_one d1 ""
run_one p2048 2048
run_one p0 0
run_one d2 ""
python3 -B - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
for c in ("d1", "p2048", "p0", "d2"):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): print(c, "missing"); continue
    for r in csv.DictReader(open(p)):
        print("%-6s ctx=%-5s tokens=%-5s ms=%-10s tps=%s" % (c, r["ctx_tokens"], r["prefill_tokens"], r["prefill_ms"], r["prefill_tps"]))
PY
exit 0

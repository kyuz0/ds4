#!/usr/bin/env bash
# Per-token prefill cost vs context in one ascending session. If the expert
# working set becomes cache-resident at small contexts, per-token cost should
# fall; if it stays flat, prefill is compute/issue bound, not memory bound.
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
OUT="$G/findings/bench-context-cost-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case ascending frontiers 2048..32768, gen 0, cache 92GB"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
timeout -k 60 2400 env DS4_BENCH_DEV_STAT=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048,4096,8192,16384,32768 --ctx-alloc 40960 --gen-tokens 0 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv"
echo "=== per-token cost ==="
python3 -B - "$OUT/profile.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    tok = int(r["prefill_tokens"]); ms = float(r["prefill_ms"])
    print("ctx=%-6s new_tokens=%-6d ms=%-10.1f tps=%-7s ms/token=%.3f"
          % (r["ctx_tokens"], tok, ms, r["prefill_tps"], ms / tok))
PY
exit 0

#!/bin/bash
# bench_server.sh TAG PROMPT_FILE NTOK [extra ds4-server args...]
# Starts ds4-server with the production flags of docs/STRIX_HALO_DSPARK.md on PORT (default 18096),
# sends the same greedy chat request twice (the second one runs on warm caches and a rewound
# live session), prints time to first token and decode tok/s, then stops the server.
# Pass --dspark --mtp-model FILE as extra args to measure DSpark.
# Environment: DS4_SERVER (binary), MODEL, VISION (optional), CACHE (default 72GB), PORT, KV_DIR.
set -u
TAG=$1; PROMPT=$2; NTOK=$3; shift 3
BIN=${DS4_SERVER:-./ds4-server}; PORT=${PORT:-18096}; CACHE=${CACHE:-72GB}
KV_DIR=${KV_DIR:-/tmp/ds4-bench-kv-$TAG}; mkdir -p "$KV_DIR"
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=server-$TAG.log
export DS4_ROCM_SELECTED_SPLIT=${DS4_ROCM_SELECTED_SPLIT:-1} DS4_SERVER_PREFILL_QUANTUM=${DS4_SERVER_PREFILL_QUANTUM:-8192}
"$BIN" --rocm -m "$MODEL" ${VISION:+--vision "$VISION"} \
  --ssd-streaming --ssd-streaming-cache-experts "$CACHE" --ctx 300000 --batched-session 1 \
  --host 127.0.0.1 --port "$PORT" --kv-disk-dir "$KV_DIR" --kv-disk-space-mb 8192 \
  --kv-cache-min-tokens 256 --kv-cache-cold-max-tokens 30000 --kv-cache-continued-interval-tokens 2048 \
  --kv-cache-boundary-align-tokens 512 --kv-cache-reject-different-quant "$@" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 240); do curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break; sleep 5; done
cp "$PROMPT" "prompt-$TAG.txt"
for r in 1 2; do python3 "$HERE/client.py" "$PORT" "prompt-$TAG.txt" "$NTOK" "$TAG-$r"; done
kill -TERM "$PID"; wait "$PID" 2>/dev/null
grep -E "DSpark stats" "$LOG" | tail -1

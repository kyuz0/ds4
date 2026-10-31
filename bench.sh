#!/usr/bin/env bash
# bench.sh - DeepSeek V4.1 Flash SSD-streaming baseline payload.
#
# Run ONLY through supervisor.sh. The supervisor stops Pi and the cockpit
# backend, then runs this payload with no GPU/RAM competitor, restarts the
# backend and resumes Pi.
#
# This reproduces the published gfx1151 V4.1 Flash SSD frontier sweep:
#   76GB expert cache, 34816 allocated context, 128 greedy tokens,
#   frontiers 2048 -> 32768 with appended intervals 2048, 2048, 4096,
#   8192, 16384. See docs/STRIX_HALO.md.
set -uo pipefail

G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
cd "$G" || exit 1

# Guard: refuse to overlap the backend with the benchmark.
if curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "*** DS4 backend is still running; run this payload via supervisor.sh ***"
  exit 1
fi
CONTAINER_NAME="${CONTAINER_NAME:-ds4-cockpit-server-984bffa2}"
if podman ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
  echo "*** DS4 cockpit container is still running; run this payload via supervisor.sh ***"
  exit 1
fi
if [ ! -r "$MODEL" ]; then
  echo "*** model not readable: $MODEL ***"
  exit 1
fi
if [ ! -r "$PROMPT" ]; then
  echo "*** prompt not readable: $PROMPT ***"
  exit 1
fi

RUN_DIR="$G/findings/bench-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR/frontiers" || exit 1

# Keep the baseline at defaults. Debug selectors are overrides; clear any that
# leaked into the supervisor environment.
for name in $(compgen -v | grep '^DS4_'); do
  unset "$name"
done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

{
  echo "provenance $(date -Is)"
  git log -1 --format='commit: %h %ci %s'
  echo "worktree:"
  git status --short | sed 's/^/  /'
  echo "binary ds4-bench: sha256 $(sha256sum ds4-bench | cut -d' ' -f1) mtime $(stat -c '%y' ds4-bench)"
  echo "binary ds4:       sha256 $(sha256sum ds4 | cut -d' ' -f1) mtime $(stat -c '%y' ds4)"
  echo "model: $MODEL"
  echo "model bytes: $(stat -c '%s' "$MODEL") mtime: $(stat -c '%y' "$MODEL")"
  echo "prompt: $PROMPT"
  echo "prompt sha256: $(sha256sum "$PROMPT" | cut -d' ' -f1)"
  echo "rocm: $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
  echo "kernel: $(uname -r)"
  echo "memory: $(free -h | sed -n '2p')"
  echo "container: $CONTAINER_NAME"
  echo "container image: $(podman inspect -f '{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown)"
  echo "container cmd: $(podman inspect -f '{{join .Config.Cmd " "}}' "$CONTAINER_NAME" 2>/dev/null || echo unknown)"
  echo "flags: ds4-bench --rocm, default selectors; cache 76GB; ctx_alloc 34816; snapshot cap 2 GiB"
} > "$RUN_DIR/provenance.txt" 2>&1
cat "$RUN_DIR/provenance.txt"

echo "=== frontier sweep: start $(date -Is) ==="
timeout 3600 ./ds4-bench --rocm \
  -m "$MODEL" \
  --prompt-file "$PROMPT" \
  --ctx-start 2048 --ctx-max 32768 --step-mul 2 \
  --ctx-alloc 34816 \
  --gen-tokens 128 \
  --ssd-streaming \
  --ssd-streaming-cache-experts 76GB \
  --show-output \
  --dump-frontier-logits-dir "$RUN_DIR/frontiers" \
  --csv "$RUN_DIR/baseline.csv" \
  > "$RUN_DIR/baseline.log" 2>&1
RC=$?
echo "=== frontier sweep: rc=$RC done $(date -Is) ==="

SUMMARY="$(
  [ -s "$RUN_DIR/baseline.csv" ] &&
  awk -F, 'NR > 1 {
    printf "- ctx %d (+%d): prefill %.2f tok/s, decode %.2f tok/s, steady %.2f tok/s, first-token %.1f ms\n", $1, $2, $3, $5, $8, $6
  }' "$RUN_DIR/baseline.csv"
)"

{
  echo
  echo "## Cycle: V4.1 Flash SSD baseline - $(date -Is)"
  echo
  echo "- artifacts: \`findings/$(basename "$RUN_DIR")/\` (baseline.csv, provenance.txt, baseline.log)"
  echo "- command: \`DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 ds4-bench --rocm -m $MODEL --prompt-file $PROMPT --ssd-streaming --ssd-streaming-cache-experts 76GB --ctx-start 2048 --ctx-max 32768 --step-mul 2 --ctx-alloc 34816 --gen-tokens 128 --show-output --dump-frontier-logits-dir frontiers --csv baseline.csv\`"
  echo "- rc: $RC"
  echo
  echo "### Measured"
  echo "$SUMMARY"
  echo
  echo "### Published gfx1151 V4.1 reference (docs/STRIX_HALO.md)"
  echo "| Context | Appended | Prefill | Decode |"
  echo "|---:|---:|---:|---:|"
  echo "| 2,048 | 2,048 | 47.32 | 10.18 |"
  echo "| 4,096 | 2,048 | 86.48 | 10.27 |"
  echo "| 8,192 | 4,096 | 154.58 | 10.46 |"
  echo "| 16,384 | 8,192 | 271.06 | 10.34 |"
  echo "| 32,768 | 16,384 | 333.84 | 10.34 |"
  echo
  echo "### Log tail"
  echo '```'
  tail -30 "$RUN_DIR/baseline.log" 2>/dev/null
  echo '```'
  echo
} >> "$G/RESULTS.md"

echo "BENCH DONE rc=$RC"
exit "$RC"

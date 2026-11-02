#!/usr/bin/env bash
# bench-suffix-down-replay.sh - minimal V4.1 down-kernel replay from captured
# layer-20 F16 mid and identical owned Q2 expert weights. Run only through
# supervisor.sh with trial=1. It repeats the identical shape/input, then varies
# only placement/path: compact vs hotlist, candidate vs reference shape.
set -uo pipefail

G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
RUN_BASE="$G/findings/bench-suffix-dependency-diag-8k4k-trial1-20261101-182746/base8k"
TRIAL="${1:-1}"
case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1

if curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "*** DS4 backend is still running; run via supervisor.sh ***"; exit 1
fi
CONTAINER_NAME="${CONTAINER_NAME:-}"
if [ -z "$CONTAINER_NAME" ]; then
  CONTAINER_NAME="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
fi
if [ -n "$CONTAINER_NAME" ] && podman ps --format '{{.Names}}' 2>/dev/null | grep -qx -- "$CONTAINER_NAME"; then
  echo "*** DS4 cockpit container is still running; run via supervisor.sh ***"; exit 1
fi

[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
[ -x tests/test_suffix_down_replay ] || { echo "*** replay binary missing; make tests/test_suffix_down_replay ***"; exit 1; }
for p in candidate/diag/moe_mid_f16.raw candidate/diag/moe_counts_u32.raw candidate/diag/moe_down.raw \
         candidate/diag/shared_routed_ffn_20_selected_8065_8066.raw \
         reference/diag/moe_mid_f16.raw reference/diag/moe_counts_u32.raw reference/diag/moe_down.raw \
         reference/diag/shared_routed_ffn_20_selected_8065_8066.raw; do
  [ -r "$RUN_BASE/$p" ] || { echo "*** captured input missing: $p ***"; exit 1; }
done

OUT_DIR="$G/findings/bench-suffix-down-replay-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "worktree:"; git status --short | sed 's/^/  /'
  echo "binary sha256 $(sha256sum tests/test_suffix_down_replay | cut -d' ' -f1) mtime $(stat -c '%y' tests/test_suffix_down_replay)"
  echo "model $MODEL bytes $(stat -c '%s' "$MODEL") sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "capture $RUN_BASE"
  echo "down_offset=85345599488 (GGUF blk.20.ffn_down_exps.weight abs offset) candidate n=366 row=239 reference n=2048 row=1921"
  echo "paths: candidate actual hotlist vs reference actual compact; forced compact/hotlist replay"
} > "$OUT_DIR/provenance.txt" 2>&1
cat "$OUT_DIR/provenance.txt"

./tests/test_suffix_down_replay "$MODEL" "$RUN_BASE" 366 239 2048 1921 85345599488 "$OUT_DIR" \
  > "$OUT_DIR/replay_summary.txt" 2> "$OUT_DIR/replay.log"
RC=$?
echo "[rc replay] $RC"
echo "=== replay summary ==="
cat "$OUT_DIR/replay_summary.txt"
echo "=== replay log tail ==="
tail -n 80 "$OUT_DIR/replay.log"
echo "=== fused gate/up shape replay ==="
./tests/test_suffix_gate_up_replay "$MODEL" "$RUN_BASE" "$OUT_DIR" 82984828928 84165214208 \
  > "$OUT_DIR/fused_shapes.txt" 2> "$OUT_DIR/fused_shapes.log"
RC_GU=$?
echo "[rc fused_gate_up] $RC_GU"
cat "$OUT_DIR/fused_shapes.txt"
echo "=== outputs ==="
ls -l "$OUT_DIR"/case_*_target_f16.raw 2>/dev/null
exit "$RC"

#!/usr/bin/env bash
# bench-suffix-correctness-8k4k-diag.sh - focused V4.1 decoder-suffix diagnostic.
# Run ONLY through supervisor.sh with trial=1.
# One 8192 prefill case: reference full decoder body vs DS4_METAL_V41_DECODER_SUFFIX=1.
# Dumps selected layer-20/21 intermediates, source publication and a 127-row KV tail.
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
CONTAINER_NAME="${CONTAINER_NAME:-}"
if [ -z "$CONTAINER_NAME" ]; then
  CONTAINER_NAME="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
fi
if [ -n "$CONTAINER_NAME" ] && podman ps --format '{{.Names}}' 2>/dev/null | grep -qx -- "$CONTAINER_NAME"; then
  echo "*** DS4 cockpit container is still running; run via supervisor.sh ***"; exit 1
fi
[ -r "$MODEL" ] && [ -r "$PROMPT" ] || { echo "*** model/prompt unreadable ***"; exit 1; }

RUN_DIR="$G/findings/bench-suffix-dependency-diag-8k4k-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done

{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "worktree:"; git status --short | sed 's/^/  /'
  echo "candidate source lines:"
  git diff -- ds4.c | grep -nE 'DIAG_DUMP|diag_dump|DECODER_SUFFIX|decoder_suffix' | head -30 | sed 's/^/  /'
  echo "binary ds4-bench sha256 $(sha256sum ds4-bench | cut -d' ' -f1) mtime $(stat -c '%y' ds4-bench)"
  echo "model $MODEL bytes $(stat -c '%s' "$MODEL") sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "prompt $PROMPT sha256 $(sha256sum "$PROMPT" | cut -d' ' -f1)"
  echo "case base8k frontier=8192 rows=8192 sampled_kv_tail=8065..8191"
  echo "reference: no suffix env, full decoder body"
  echo "candidate: DS4_METAL_V41_DECODER_SUFFIX=1 DS4_ROCM_V41_SUFFIX_HC_PAD=1"
  echo "flags: --rocm; gen 0; cache 92GB; diag dump selected rows"
} > "$RUN_DIR/provenance.txt" 2>&1
cat "$RUN_DIR/provenance.txt"

run_arm() {
  local arm="$1" cand="$2"
  local dir="$RUN_DIR/base8k/$arm"
  mkdir -p "$dir/diag" || return 1
  local envs=()
  [ "$cand" = 1 ] && envs+=("DS4_METAL_V41_DECODER_SUFFIX=1" "DS4_ROCM_V41_SUFFIX_HC_PAD=1")
  echo "=== base8k/$arm: start $(date -Is) ==="
  timeout -k 60 2400 env "${envs[@]}" \
    DS4_METAL_V41_DIAG_DUMP=1 \
    DS4_METAL_V41_DIAG_DUMP_DIR="$dir/diag" \
    DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 \
    DS4_BENCH_DEV_STAT=1 \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}" \
    ./ds4-bench --rocm \
      -m "$MODEL" \
      --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 \
      --ctx-alloc 24576 \
      --gen-tokens 0 \
      --show-output \
      --ssd-streaming \
      --ssd-streaming-cache-experts 92GB \
      --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?
  echo "=== base8k/$arm: rc=$rc done $(date -Is) ==="
  return "$rc"
}

run_arm reference 0; RC_REF=$?
run_arm candidate 1; RC_CAND=$?
echo "[rc base8k/reference] $RC_REF"
echo "[rc base8k/candidate] $RC_CAND"

STATUS=0
[ "$RC_REF" -eq 0 ] || STATUS=1
[ "$RC_CAND" -eq 0 ] || STATUS=1
python3 tools/compare-suffix-diag.py "$RUN_DIR/base8k" > "$RUN_DIR/diagnostic.txt" 2>&1
RC_DIAG=$?
cat "$RUN_DIR/diagnostic.txt"
echo "[rc diagnostic] $RC_DIAG"
if [ "$RC_DIAG" -eq 2 ]; then
  echo "*** sampled intermediates match; this does not prove full model equivalence ***"
elif [ "$RC_DIAG" -ne 0 ]; then
  STATUS=1
fi
echo "=== producer comparison (input, gate, mid, instrumentation) ==="
python3 -B tools/compare-suffix-producer.py "$RUN_DIR" 2>&1 | tee "$RUN_DIR/producer_compare.txt"
exit "$STATUS"

#!/usr/bin/env bash
# bench-suffix-state-append.sh - correctness qualification for the decoder suffix.
# One cycle, both arms, identical history:
#   1. prefill frontier 8192 (cold) then append frontier 12288 (same session),
#      teacher-forced continuation so both arms decode the same forced tokens;
#   2. full logit-distribution comparison at both frontiers;
#   3. persistent-state comparison from --dump-session-payload-dir: every
#      window[0..39], compressed[]/index_cache[]/previous_* span, span by span.
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

OUT="$G/findings/bench-suffix-state-append-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "worktree:"; git status --short | sed 's/^/  /'
  echo "binary ds4-bench sha256 $(sha256sum ds4-bench | cut -d' ' -f1) mtime $(stat -c '%y' ds4-bench)"
  echo "model $MODEL bytes $(stat -c '%s' "$MODEL") sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "prompt $PROMPT sha256 $(sha256sum "$PROMPT" | cut -d' ' -f1)"
  echo "case frontiers=8192,12288 ctx_alloc=24576 gen=8 teacher-forced cache=92GB"
  echo "reference: full decoder body; candidate: DS4_METAL_V41_DECODER_SUFFIX=1 DS4_ROCM_V41_SUFFIX_HC_PAD=1"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

run_arm() {
  local arm="$1" cand="$2"
  local dir="$OUT/$arm"
  mkdir -p "$dir/logits" "$dir/payload" || return 1
  local envs=()
  [ "$cand" = 1 ] && envs+=("DS4_METAL_V41_DECODER_SUFFIX=1" "DS4_ROCM_V41_SUFFIX_HC_PAD=1")
  echo "=== $arm: start $(date -Is) ==="
  timeout -k 60 3000 env "${envs[@]}" \
    DS4_BENCH_DEV_STAT=1 \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}" \
    ./ds4-bench --rocm \
      -m "$MODEL" \
      --prompt-file "$PROMPT" \
      --ctx-frontiers 8192,12288 \
      --ctx-alloc 24576 \
      --gen-tokens 8 \
      --teacher-forced-decode \
      --show-output \
      --ssd-streaming \
      --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" \
      --dump-session-payload-dir "$dir/payload" \
      --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?
  echo "=== $arm: rc=$rc done $(date -Is) ==="
  grep -E "prefill frontier|decoded text|teacher" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}

run_arm reference 0; RC_REF=$?
run_arm candidate 1; RC_CAND=$?
echo "[rc reference] $RC_REF"
echo "[rc candidate] $RC_CAND"

echo "=== per-arm prefill/append timings ==="
for a in reference candidate; do
  echo "-- $a"; cat "$OUT/$a/profile.csv" 2>/dev/null | sed -n '1,3p'
done

echo "=== frontier logit distributions ==="
python3 -B tools/compare-suffix-logits.py "$OUT/reference/logits" "$OUT/candidate/logits" \
  2>&1 | tee "$OUT/logits_compare.txt"
RC_LOGITS=${PIPESTATUS[0]}

echo "=== payload files ==="
ls -l "$OUT"/reference/payload "$OUT"/candidate/payload 2>/dev/null | sed -n '1,20p'

echo "=== persistent state span comparison (prefill 8192 and append 12288) ==="
for f in $(ls "$OUT"/reference/payload | sort); do
  if [ -r "$OUT/candidate/payload/$f" ]; then
    echo "---- $f"
    python3 -B tools/compare-suffix-payload.py "$OUT/reference/payload/$f" "$OUT/candidate/payload/$f" \
      2>&1 | tee -a "$OUT/state_compare.txt"
  fi
done

echo "[rc logits] $RC_LOGITS"
exit 0

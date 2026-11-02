#!/usr/bin/env bash
# bench-suffix-gate-up-equivalence.sh - one cycle, two diagnostics:
#   1. gate/up replay: identical layer-20 normalized input, ids and weights,
#      only n_tokens (MMQ shape) varies -> is the mid delta batch-shape rounding?
#   2. end-to-end equivalence: full-body reference vs decoder-suffix arm at
#      8192 frontier with 32 greedy tokens, comparing frontier logits + text.
# Run ONLY through supervisor.sh with trial=1.
set -uo pipefail

G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
RUN_BASE="$G/findings/bench-suffix-dependency-diag-8k4k-trial1-20261101-182746/base8k"
GATE_OFF=82984828928      # blk.20.ffn_gate_exps.weight
UP_OFF=84165214208        # blk.20.ffn_up_exps.weight
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
[ -x tests/test_suffix_gate_up_replay ] || { echo "*** gate/up replay binary missing ***"; exit 1; }
for p in candidate/diag/hc_ffn_norm_20_norm_8065_8066.raw candidate/diag/moe_mid_f32.raw \
         candidate/diag/shared_routed_ffn_20_selected_8065_8066.raw \
         candidate/diag/shared_routed_ffn_20_route_weights_8065_8066.raw \
         reference/diag/moe_mid_f32.raw; do
  [ -r "$RUN_BASE/$p" ] || { echo "*** captured input missing: $p ***"; exit 1; }
done

OUT="$G/findings/bench-suffix-gate-up-eq-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "worktree:"; git status --short | sed 's/^/  /'
  echo "model $MODEL bytes $(stat -c '%s' "$MODEL") sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "gate_offset=$GATE_OFF up_offset=$UP_OFF down_offset=85345599488"
  echo "capture $RUN_BASE"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

echo "=== gate/up batch-shape replay $(date -Is) ==="
./tests/test_suffix_gate_up_replay "$MODEL" "$RUN_BASE" "$OUT" "$GATE_OFF" "$UP_OFF" \
  > "$OUT/gate_up_replay.txt" 2> "$OUT/gate_up_replay.log"
RC_REPLAY=$?
echo "[rc gate_up_replay] $RC_REPLAY"
cat "$OUT/gate_up_replay.txt"

run_arm() {
  local arm="$1" cand="$2"
  local dir="$OUT/$arm"
  mkdir -p "$dir/logits" || return 1
  local envs=()
  [ "$cand" = 1 ] && envs+=("DS4_METAL_V41_DECODER_SUFFIX=1" "DS4_ROCM_V41_SUFFIX_HC_PAD=1")
  echo "=== $arm: start $(date -Is) ==="
  timeout -k 60 2400 env "${envs[@]}" \
    DS4_BENCH_DEV_STAT=1 \
    LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}" \
    ./ds4-bench --rocm \
      -m "$MODEL" \
      --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 \
      --ctx-alloc 24576 \
      --gen-tokens 32 \
      --show-output \
      --ssd-streaming \
      --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" \
      --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?
  echo "=== $arm: rc=$rc done $(date -Is) ==="
  return "$rc"
}

run_arm reference 0; RC_REF=$?
run_arm candidate 1; RC_CAND=$?
echo "[rc reference] $RC_REF"
echo "[rc candidate] $RC_CAND"

echo "=== frontier logits comparison ==="
python3 - "$OUT" <<'PY' 2>&1 | tee "$OUT/logits_compare.txt"
import json, os, sys, glob
out = sys.argv[1]
def biggest_numeric(obj, best):
    if isinstance(obj, list):
        if obj and all(isinstance(v, (int, float)) for v in obj):
            if len(obj) > len(best):
                return obj
        for v in obj:
            best = biggest_numeric(v, best)
    elif isinstance(obj, dict):
        for v in obj.values():
            best = biggest_numeric(v, best)
    return best
def load(p):
    with open(p) as f:
        data = json.load(f)
    return biggest_numeric(data, [])
ref = sorted(glob.glob(os.path.join(out, 'reference/logits/*')))
cand = sorted(glob.glob(os.path.join(out, 'candidate/logits/*')))
print('ref logit files', len(ref), 'cand logit files', len(cand))
for i, (a, b) in enumerate(zip(ref, cand)):
    try:
        va, vb = load(a), load(b)
    except Exception as exc:
        print('parse failed', os.path.basename(a), exc)
        continue
    if not va or len(va) != len(vb):
        print('shape mismatch', len(va), len(vb))
        continue
    top_a = sorted(range(len(va)), key=lambda k: -va[k])[:5]
    top_b = sorted(range(len(vb)), key=lambda k: -vb[k])[:5]
    md = max(abs(x - y) for x, y in zip(va, vb))
    print(f'frontier {i} n={len(va)} argmax ref={top_a[0]} cand={top_b[0]} '
          f'same={top_a[0]==top_b[0]} top5_overlap={len(set(top_a)&set(top_b))} '
          f'max_abs_delta={md:.6g}')
PY

echo "=== generated output tails (reference) ==="
tail -n 25 "$OUT/reference/profile.log"
echo "=== generated output tails (candidate) ==="
tail -n 25 "$OUT/candidate/profile.log"
echo "=== csv summary ==="
for a in reference candidate; do
  echo "-- $a"; sed -n '1,4p' "$OUT/$a/profile.csv" 2>/dev/null
done
exit 0

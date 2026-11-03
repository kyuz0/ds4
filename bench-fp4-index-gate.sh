#!/usr/bin/env bash
# Gate for the additive FP4 index-cache packing: the packed cache is allocated and
# filled by both writers, but every reader still uses the F32 cache, so frontier
# logits must be bit-identical to the previous promoted binary.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
BASE="${BASE_BIN:-/tmp/ds4-bench-declut}"
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial must be 1 or 2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
[ -x "$BASE" ] || { echo "*** base binary $BASE missing ***"; exit 1; }
OUT="$G/findings/bench-fp4-index-gate-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "base $BASE $(sha256sum "$BASE" | cut -d' ' -f1)"
  echo "cand ./ds4-bench $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 8K gen8 frontier-logits bit equality; packed cache written by both writers, readers still F32"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_arm() {
  local arm="$1" bin="$2"; local dir="$OUT/$arm"; mkdir -p "$dir/logits"
  echo "=== $arm start $(date -Is) ==="
  timeout -k 60 1500 "$bin" --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 8 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $arm rc=$rc $(date -Is) ==="
  grep -E "memory:|prefill frontier|admit|error|invalid" "$dir/profile.log" | head -6 | sed 's/^/   /'
  return "$rc"
}
run_arm base "$BASE" || echo "*** base arm failed ***"
run_arm cand ./ds4-bench || echo "*** cand arm failed ***"
python3 -B - "$OUT" <<'PY'
import glob, json, os, struct, sys
out = sys.argv[1]
def load(d):
    p = sorted(glob.glob(os.path.join(out, d, "logits", "*.logits.json")))
    if not p:
        return None
    return [json.load(open(x)) for x in p]
b, c = load("base"), load("cand")
if b is None or c is None:
    print("GATE: missing logits dumps"); sys.exit(1)
print(f"GATE: {len(b)} base frontier files, {len(c)} cand frontier files")
bad = 0
for fb, fc in zip(b, c):
    vb = fb.get("logits") or fb.get("values") or []
    vc = fc.get("logits") or fc.get("values") or []
    if len(vb) != len(vc):
        print(f"GATE: length mismatch {len(vb)} vs {len(vc)}"); bad += 1; continue
    diff = sum(1 for x, y in zip(vb, vc) if struct.pack('<f', x) != struct.pack('<f', y))
    ab = max(range(len(vb)), key=lambda i: vb[i]); ac = max(range(len(vc)), key=lambda i: vc[i])
    mx = max((abs(x - y) for x, y in zip(vb, vc)), default=0.0)
    print(f"  ctx={fb.get('ctx')} n={len(vb)} bit_mismatches={diff} max_abs={mx:g} argmax {ab}->{ac}")
    if diff or ab != ac:
        bad += 1
print("GATE: PASS" if not bad else "GATE: FAIL")
sys.exit(0 if not bad else 1)
PY
rc=$?; [ $rc -eq 0 ] || echo "*** GATE FAILED ***"
exit $rc

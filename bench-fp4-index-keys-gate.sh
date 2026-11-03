#!/usr/bin/env bash
# Gate for the packed-key decode scorer: same binary, arm A scores from the F32
# index cache, arm B (DS4_ROCM_V41_INDEX_FP4=1) scores from the packed 68-byte keys.
# Frontier logits must be bit-identical and the dumped selected indices byte-identical.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial must be 1 or 2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-fp4-index-keys-gate-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "cand ./ds4-bench $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 8K+112K gen8; arm off = F32 keys, arm on = packed 68-byte keys"
  echo "checks: frontier logits bit-equal + selected-index dump byte-equal"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_arm() {
  local arm="$1" on="$2"; local dir="$OUT/$arm"; mkdir -p "$dir/logits"
  local envs=()
  [ "$on" = 1 ] && envs+=("DS4_ROCM_V41_INDEX_FP4=1")
  envs+=("DS4_V41_INDEX_DUMP=$dir/selected.bin")
  envs+=("DS4_V41_INDEX_FP4_VERIFY=1")
  echo "=== $arm on=$on start $(date -Is) ==="
  timeout -k 60 2400 env "${envs[@]}" ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192,114688 --ctx-alloc 131072 --gen-tokens 8 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $arm rc=$rc $(date -Is) ==="
  grep -E "memory:|prefill frontier|admit|nan|invalid|error" "$dir/profile.log" | head -8 | sed 's/^/   /'
  return "$rc"
}
run_arm f32 0 || echo "*** f32 arm failed ***"
run_arm fp4 1 || echo "*** fp4 arm failed ***"
python3 -B - "$OUT" <<'PY'
import glob, json, os, struct, sys
out = sys.argv[1]
bad = 0
def load(d):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, d, "logits", "*.logits.json")))]
a, b = load("f32"), load("fp4")
if not a or not b:
    print("GATE: missing logits dumps"); sys.exit(1)
print(f"GATE: {len(a)} f32 frontier files, {len(b)} fp4 frontier files")
for fa, fb in zip(a, b):
    va = fa.get("logits") or fa.get("values") or []
    vb = fb.get("logits") or fb.get("values") or []
    if len(va) != len(vb):
        print(f"  ctx={fa.get('ctx')} length mismatch {len(va)} vs {len(vb)}"); bad += 1; continue
    diff = sum(1 for x, y in zip(va, vb) if struct.pack('<f', x) != struct.pack('<f', y))
    am = max(range(len(va)), key=lambda i: va[i]); bm = max(range(len(vb)), key=lambda i: vb[i])
    mx = max((abs(x - y) for x, y in zip(va, vb)), default=0.0)
    print(f"  ctx={fa.get('ctx')} n={len(va)} bit_mismatches={diff} max_abs={mx:g} argmax {am}->{bm}")
    if diff or am != bm: bad += 1
da = os.path.join(out, "f32", "selected.bin"); db = os.path.join(out, "fp4", "selected.bin")
sa = open(da, "rb").read() if os.path.exists(da) else b""
sb = open(db, "rb").read() if os.path.exists(db) else b""
if not sa or not sb:
    print(f"GATE: missing index dumps ({len(sa)}/{len(sb)} bytes)"); bad += 1
else:
    topk = 512
    rec = 8 + 4 * topk
    print(f"GATE: selected-index dump {len(sa)} vs {len(sb)} bytes ({len(sa)//rec} records of top-{topk})")
    if sa == sb:
        print("GATE: selected indices byte-identical")
    else:
        bad += 1
        n = min(len(sa), len(sb)) // rec
        diff = 0
        for i in range(n):
            ra = sa[i*rec:(i+1)*rec]; rb = sb[i*rec:(i+1)*rec]
            if ra == rb: continue
            diff += 1
            if diff == 1:
                ha = struct.unpack_from("<2I", ra, 0); hb = struct.unpack_from("<2I", rb, 0)
                ia = struct.unpack_from("<%di" % topk, ra, 8); ib = struct.unpack_from("<%di" % topk, rb, 8)
                nd = sum(1 for x, y in zip(ia, ib) if x != y)
                print(f"  first differing record {i} pos/layer {ha} vs {hb}, {nd}/{topk} ids differ,"
                      f" same_set={set(ia)==set(ib)}")
        print(f"  differing records {diff}/{n}")
print("GATE: PASS" if not bad else "GATE: FAIL")
sys.exit(0 if not bad else 1)
PY
rc=$?; [ $rc -eq 0 ] || echo "*** GATE FAILED ***"
exit $rc

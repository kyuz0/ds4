#!/usr/bin/env bash
# Decode stream accounting at 32K: how many expert bytes are fetched per token and
# what the hit rate is, to decide whether cross-layer prefetch can be justified.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial must be 1 or 2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-stream-accounting32k-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "case arm A 32768/gen32 (prefill-heavy), arm B 2048/gen256 (decode-dominated)"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
run_arm() {
  local arm="$1" frontier="$2" gen="$3" alloc="$4"; local dir="$OUT/$arm"; mkdir -p "$dir"
  echo "=== $arm frontier=$frontier gen=$gen start $(date -Is) ==="
  timeout -k 60 1800 env DS4_ROCM_STREAM_CACHE_STATS=1 DS4_ROCM_MOE_DECODE_PROFILE=1 \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers "$frontier" --ctx-alloc "$alloc" --gen-tokens "$gen" \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $arm rc=$? $(date -Is) ==="
  grep -E "prefill frontier|layer reuse cleanup|MoE profile" "$dir/profile.log" | tail -3 | cut -c1-150
}
run_arm a 32768 32 40960
run_arm b 2048 256 8192
report() {
  local arm="$1" gen="$2"
  python3 -B - "$OUT/$arm" "$gen" <<'PY2'
import csv, os, re, sys
d, gent = sys.argv[1], int(sys.argv[2])
log = open(os.path.join(d, "profile.log")).read()
gen_tps = None
p = os.path.join(d, "profile.csv")
if os.path.exists(p):
    rows = list(csv.DictReader(open(p)))
    if rows: gen_tps = float(rows[-1]["gen_tps"])
m = re.search(r"logical-bytes=(\d+) missing-bytes=(\d+).*?read-returned-bytes=(\d+)", log, re.S)
if not m:
    print(f"  {os.path.basename(d)}: no reuse stats"); sys.exit(0)
logical, missing, read_bytes = (int(x) for x in m.groups())
pf = re.search(r"prefill frontier=\d+ tokens=(\d+)", log)
pft = int(pf.group(1)) if pf else 0
# Attribute bytes: prefill streams per-layer expert sets for its token sweep, decode for
# the generated tokens only. With a small prefill and many gen tokens, missing/token is
# decode-dominated; report both normalisers so the arm's bias stays visible.
print(f"  {os.path.basename(d)}: prefill_tokens={pft} gen_tokens={gent} gen_tps={gen_tps}")
print(f"    logical={logical/1e9:.1f} GB missing={missing/1e9:.1f} GB")
if gent and gen_tps:
    secs = gent / gen_tps
    print(f"    decode wall ~{secs:.1f} s, missing/token {missing/gent/1e6:.1f} MB, "
          f"imputed bw {missing/secs/1e9:.1f} GB/s")
PY2
}
report a 32
report b 256
exit 0

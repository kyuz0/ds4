#!/usr/bin/env bash
# Apportion the 32K decode token: per-stage ms aggregated across layers and tokens.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2) ;; *) echo "*** trial must be 1 or 2 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-decode-apportion32k-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "case 32768 frontier, gen 16, DS4_V41_DECODE_STAGE_PROFILE=1"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
timeout -k 60 1800 env DS4_V41_DECODE_STAGE_PROFILE=1 DS4_BENCH_DEV_STAT=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens 16 \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
tail -1 "$OUT/profile.csv"
python3 -B - "$OUT/profile.log" <<'PY'
import collections, re, sys
tot = collections.Counter(); cnt = collections.Counter()
rx = re.compile(r"v41 decode stage layer=(\d+) label=(\S+) ms=([0-9.]+)")
for ln in open(sys.argv[1]):
    m = rx.search(ln)
    if not m: continue
    tot[m.group(2)] += float(m.group(3)); cnt[m.group(2)] += 1
if not tot:
    print("no stage lines found"); sys.exit(0)
grand = sum(tot.values())
print(f"total instrumented {grand:.1f} ms over {max(cnt.values())} tokens/layer-samples")
for label, ms in tot.most_common(30):
    print(f"  {label:34s} {ms:9.1f} ms  {100*ms/grand:5.1f}%  n={cnt[label]}")
PY
exit 0

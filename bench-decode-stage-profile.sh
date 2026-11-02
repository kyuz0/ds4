#!/usr/bin/env bash
# Aggregate the V4.1 decode stage profile: where the per-token time goes.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1) ;; *) echo "*** trial must be 1 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
CN="$(podman ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }')"
[ -n "$CN" ] && podman ps --format '{{.Names}}' | grep -qx -- "$CN" && { echo "*** container running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-decode-stage-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
echo "=== profile run start $(date -Is) ==="
timeout -k 60 1200 env DS4_V41_DECODE_STAGE_PROFILE=1 DS4_BENCH_DEV_STAT=1 \
  ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 8 --show-output \
    --ssd-streaming --ssd-streaming-cache-experts 92GB \
    --csv "$OUT/profile.csv" > "$OUT/profile.log" 2>&1
echo "[rc] $?"
cat "$OUT/profile.csv"
python3 -B - "$OUT/profile.log" <<'PY'
import collections, re, sys
tot = collections.Counter(); cnt = collections.Counter()
per_token = collections.defaultdict(float)
rx = re.compile(r"v41 decode stage layer=(\d+) label=(\S+) ms=([0-9.]+)")
for ln in open(sys.argv[1]):
    m = rx.search(ln)
    if not m: continue
    layer, label, ms = int(m.group(1)), m.group(2), float(m.group(3))
    tot[label] += ms; cnt[label] += 1
print("\n%-22s %10s %8s %10s" % ("stage", "total_ms", "n", "mean_ms"))
for label, t in tot.most_common():
    print("%-22s %10.2f %8d %10.3f" % (label, t, cnt[label], t / cnt[label]))
print("\nsum of all sampled stages: %.1f ms over %d samples" % (sum(tot.values()), sum(cnt.values())))
PY
exit 0

#!/usr/bin/env bash
# bench-append-110k-tuning.sh - numerics-neutral configuration sweep at the
# primary workload (4K append on a ~110K cached base, then 32 decode tokens).
# Configs differ only in expert-cache budget and SSD read chunk count, so every
# kernel and every weight byte is identical; frontier logits must be bit-equal.
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

OUT="$G/findings/bench-append-110k-tuning-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" || exit 1
for name in $(compgen -v | grep '^DS4_'); do unset "$name"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{
  echo "provenance $(date -Is) trial=$TRIAL"
  git log -1 --format='commit: %h %ci %s'
  echo "binary ds4-bench sha256 $(sha256sum ds4-bench | cut -d' ' -f1) mtime $(stat -c '%y' ds4-bench)"
  echo "model $MODEL bytes $(stat -c '%s' "$MODEL") sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
  echo "prompt $PROMPT sha256 $(sha256sum "$PROMPT" | cut -d' ' -f1)"
  echo "memory $(free -h | sed -n '2p')"
  echo "flags: frontiers 110592,114688 ctx_alloc 118784 gen 32 suffix off"
  echo "configs: a=92GB(default chunks) b=104GB c=92GB+READ_CHUNKS=16"
} > "$OUT/provenance.txt" 2>&1
cat "$OUT/provenance.txt"

run_cfg() {
  local name="$1" cache="$2" chunks="$3"
  local dir="$OUT/$name"
  mkdir -p "$dir/logits" || return 1
  local envs=()
  [ -n "$chunks" ] && envs+=("DS4_ROCM_V41_SELECTED_READ_CHUNKS=$chunks")
  echo "=== $name cache=$cache chunks=${chunks:-default}: start $(date -Is) ==="
  timeout -k 60 3000 env "${envs[@]}" \
    DS4_BENCH_DEV_STAT=1 \
    ./ds4-bench --rocm \
      -m "$MODEL" \
      --prompt-file "$PROMPT" \
      --ctx-frontiers 110592,114688 \
      --ctx-alloc 118784 \
      --gen-tokens 32 \
      --show-output \
      --ssd-streaming \
      --ssd-streaming-cache-experts "$cache" \
      --dump-frontier-logits-dir "$dir/logits" \
      --csv "$dir/profile.csv" \
      > "$dir/profile.log" 2>&1
  local rc=$?
  echo "=== $name: rc=$rc done $(date -Is) ==="
  grep -E "V4.1 SSD prepared|prefill frontier|decoded text|memory:" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}

run_cfg a 92GB "";         RC_A=$?
run_cfg b 104GB "";        RC_B=$?
run_cfg c 92GB 16;         RC_C=$?
echo "[rc] a=$RC_A b=$RC_B c=$RC_C"

echo "=== profile rows ==="
for n in a b c; do echo "-- $n"; sed -n '1,3p' "$OUT/$n/profile.csv" 2>/dev/null; done

echo "=== numerics: frontier logits must be bit-identical across configs ==="
python3 -B - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
def find(o):
    best, stack = [], [o]
    while stack:
        c = stack.pop()
        if isinstance(c, list):
            if len(c) > len(best) and c and all(isinstance(v, (int, float)) for v in c):
                best = c
            else:
                stack.extend(c)
        elif isinstance(c, dict):
            stack.extend(c.values())
    return best
def load(p):
    with open(p) as fh:
        return find(json.load(fh))
base = None
for cfg in ("a", "b", "c"):
    files = sorted(glob.glob(os.path.join(out, cfg, "logits", "*")))
    print(cfg, "logit files", len(files))
    for i, f in enumerate(files):
        try:
            v = load(f)
        except Exception as exc:
            print("  parse failed", os.path.basename(f), exc); continue
        if base is None or i >= len(base):
            base = base or []
            base.append(v); continue
        r = base[i]
        if len(r) != len(v):
            print("  frontier %d shape mismatch %d/%d" % (i, len(r), len(v))); continue
        mx = max(abs(x - y) for x, y in zip(r, v))
        arg_r = max(range(len(r)), key=lambda k: r[k])
        arg_v = max(range(len(v)), key=lambda k: v[k])
        print("  frontier %d exact=%s argmax %d/%d same=%s max_abs=%.6g"
              % (i, r == v, arg_r, arg_v, arg_r == arg_v, mx))
PY

echo "=== append + decode summary ==="
python3 -B - "$OUT" <<'PY'
import csv, os, sys
out = sys.argv[1]
rows = {}
for cfg in ("a", "b", "c"):
    p = os.path.join(out, cfg, "profile.csv")
    if not os.path.exists(p):
        continue
    with open(p) as fh:
        data = list(csv.DictReader(fh))
    rows[cfg] = data
for cfg, data in rows.items():
    for r in data:
        print("%s ctx=%s prefill_ms=%s tps=%s gen_tps=%s"
              % (cfg, r.get("ctx_tokens"), r.get("prefill_ms"),
                 r.get("prefill_tps"), r.get("gen_tps")))
if "a" in rows and len(rows["a"]) >= 2:
    base_append = float(rows["a"][1]["prefill_ms"])
    base_gen = float(rows["a"][1]["gen_tps"])
    for cfg in ("b", "c"):
        if cfg in rows and len(rows[cfg]) >= 2:
            ap = float(rows[cfg][1]["prefill_ms"])
            g = float(rows[cfg][1]["gen_tps"])
            print("%s vs a: append %+.1f%%  decode %+.1f%%"
                  % (cfg, (ap - base_append) / base_append * 100.0,
                     (g - base_gen) / base_gen * 100.0))
PY
exit 0

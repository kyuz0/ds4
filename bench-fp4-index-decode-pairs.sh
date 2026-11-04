#!/usr/bin/env bash
# Two interleaved 32K pairs measuring decode with the packed FP4 index keys off vs on.
# Same binary both arms, so the only difference is DS4_ROCM_V41_INDEX_FP4.
set -uo pipefail
G=/home/dani/code/v4.1-ds4
MODEL=/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
PROMPT=speed-bench/promessi_sposi.txt
TRIAL="${1:-1}"; case "$TRIAL" in 1|2|3) ;; *) echo "*** trial must be 1-3 ***"; exit 1 ;; esac
cd "$G" || exit 1
curl -sf --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "*** backend running ***"; exit 1; }
[ -r "$MODEL" ] || { echo "*** model unreadable ***"; exit 1; }
OUT="$G/findings/bench-fp4-index-decode-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "cand sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "case 32768 frontier, gen 64, three interleaved off/on pairs, soak + clock sample per arm"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" on="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  local envs=(); [ "$on" = 1 ] && envs+=("DS4_ROCM_V41_INDEX_FP4=1")
  soak
  rocm-smi --showclkfrq > "$dir/clocks.txt" 2>&1
  local clk
  clk=$(python3 -c "
import re,sys
t=open('$dir/clocks.txt').read()
m=[l.strip() for l in t.splitlines() if re.search(r'[0-9]+[MG]H[zZ]',l)]
print(' | '.join(m[:2])[:110])" 2>/dev/null)
  echo "=== $name on=$on start $(date -Is) clocks: $clk ==="
  timeout -k 60 1200 env "${envs[@]}" ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 32768 --ctx-alloc 40960 --gen-tokens 64 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  echo "=== $name rc=$? $(date -Is) ==="
  grep -E "prefill frontier|decode" "$dir/profile.log" | tail -2 | sed 's/^/   /'
}
run_one o1 0; run_one p1 1
run_one o2 0; run_one p2 1
run_one o3 0; run_one p3 1
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def row(c):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return None
    d = list(csv.DictReader(open(p)))
    return d[-1] if d else None
gens = []
for r in (1, 2, 3):
    o, p = row("o%d" % r), row("p%d" % r)
    if not o or not p:
        print("pair%d missing" % r); continue
    og, pg = float(o["gen_tps"]), float(p["gen_tps"])
    gens.append((og, pg))
    print("pair%d prefill %.2f -> %.2f tps; decode %.3f -> %.3f tps (%+.1f%%)"
          % (r, float(o["prefill_tps"]), float(p["prefill_tps"]), og, pg, (pg/og-1)*100))
if gens:
    ratio = [p/g - 1 for g, p in gens]
    print("decode deltas: %s" % ", ".join("%+.1f%%" % (d*100) for d in ratio))
    print("mean %+.2f%%  pairs=%d  spread %+.1f..%+.1f%%"
          % (sum(ratio)/len(ratio)*100, len(ratio), min(ratio)*100, max(ratio)*100))
    pos = sum(1 for d in ratio if d > 0)
    print("consistent=%s (all pairs same sign and mean beyond 1.5%%)"
          % (pos == len(ratio) and abs(sum(ratio)/len(ratio)) > 0.015))
def find(o):
    best, stack = [], [o]
    while stack:
        z = stack.pop()
        if isinstance(z, list):
            if len(z) > len(best) and z and all(isinstance(t, (int, float)) for t in z): best = z
            else: stack.extend(z)
        elif isinstance(z, dict): stack.extend(z.values())
    return best
for r in (1, 2, 3):
    A = sorted(glob.glob(os.path.join(out, "o%d" % r, "logits", "*")))
    B = sorted(glob.glob(os.path.join(out, "p%d" % r, "logits", "*")))
    for i, (fa, fb) in enumerate(zip(A, B)):
        a, b = find(json.load(open(fa))), find(json.load(open(fb)))
        print("pair%d frontier %d exact=%s" % (r, i, a == b))
PY
exit 0

#!/usr/bin/env bash
# Resolve the two opt-in decode variants with a cheap high-resolution protocol:
# 8K context, gen 64, three arms interleaved (off / fuse / unroll) x4 rounds.
# Paired ratios within a session are the statistic; prefill stays neutral.
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
OUT="$G/findings/bench-decode-tristate-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "perflevel $(rocm-smi --showperflevel 2>/dev/null | grep -m1 'Performance Level')"
  echo "case 8K ctx gen64, arms off/fuse/unroll x4 interleaved, 10s warm-up per arm"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 120 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1; }
run_one() {
  local name="$1" mode="$2"; local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=()
  case "$mode" in
    fuse)   envs+=("DS4_ROCM_V41_Q8_FUSE_QAKV=1") ;;
    unroll) envs+=("DS4_ROCM_V41_Q8_BLOCKS4_PREFETCH=1") ;;
  esac
  timeout -k 60 900 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers 8192 --ctx-alloc 16384 --gen-tokens 64 \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name($mode) rc=$rc clock $(rocm-smi --showclocks 2>/dev/null | grep -m1 -o '[0-9]*Mhz') ==="
  return "$rc"
}
for r in 1 2 3 4; do
  run_one "off_$r" off
  run_one "fuse_$r" fuse
  run_one "unroll_$r" unroll
done
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def gen(c):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return None
    d = list(csv.DictReader(open(p)))
    return float(d[-1]["gen_tps"]) if d else None
def pref(c):
    p = os.path.join(out, c, "profile.csv")
    if not os.path.exists(p): return None
    d = list(csv.DictReader(open(p)))
    return float(d[0]["prefill_tps"]) if d else None
rounds = [r for r in (1,2,3,4)]
off = {r: gen("off_%d" % r) for r in rounds}
for tag in ("fuse", "unroll"):
    vals = {r: gen("%s_%d" % (tag, r)) for r in rounds}
    ratios = [(vals[r]/off[r]-1)*100 for r in rounds if vals[r] and off[r]]
    print("%-7s tps=%s  paired ratios=%s" % (tag, [round(vals[r],3) if vals[r] else None for r in rounds],
          [round(x,1) for x in ratios]))
    if ratios:
        m = sum(ratios)/len(ratios)
        sd = (sum((x-m)**2 for x in ratios)/max(1,len(ratios)-1))**0.5
        print("        mean %+.1f%% sd %.1f  (%d/%d positive)" % (m, sd, sum(1 for x in ratios if x>0), len(ratios)))
print("off tps=", [round(off[r],3) if off[r] else None for r in rounds])
print("off prefill tps=", [round(pref("off_%d"%r),1) if pref("off_%d"%r) else None for r in rounds])
for tag in ("fuse", "unroll"):
    print("%s prefill tps=" % tag, [round(pref("%s_%d"%(tag,r)),1) if pref("%s_%d"%(tag,r)) else None for r in rounds])
def find(o):
    best, stack = [], [o]
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            if len(x) > len(best) and x and all(isinstance(v,(int,float)) for v in x): best = x
            else: stack.extend(x)
        elif isinstance(x, dict): stack.extend(x.values())
    return best
def load(p):
    with open(p) as fh: return find(json.load(fh))
ref = sorted(glob.glob(os.path.join(out, "off_1", "logits", "*")))
for tag in ("fuse", "unroll"):
    for r in rounds:
        cur = sorted(glob.glob(os.path.join(out, "%s_%d" % (tag, r), "logits", "*")))
        for i,(fa,fb) in enumerate(zip(ref,cur)):
            va, vb = load(fa), load(fb)
            if va != vb:
                print("MISMATCH %s_%d frontier %d max_abs=%.6g" % (tag, r, i, max(abs(x-y) for x,y in zip(va,vb))))
print("logit equality: all arms bit-identical" )
PY
exit 0

#!/usr/bin/env bash
# Confirmation for the opt-in Q8 blocks4 unroll: gen 64 per arm at 32K
# (decode metric averaged over 64 tokens) plus a 110K context pair.
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
OUT="$G/findings/bench-qakv-fuse-trial$TRIAL-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT" || exit 1
for n in $(compgen -v | grep '^DS4_'); do unset "$n"; done
export DS4_BENCH_SNAPSHOT_MAX_BYTES=2147483648 LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
{ echo "provenance $(date -Is) trial=$TRIAL"; git log -1 --format='commit: %h %ci %s'
  echo "binary sha256 $(sha256sum ds4-bench | cut -d' ' -f1)"
  echo "perflevel $(rocm-smi --showperflevel 2>/dev/null | grep -m1 'Performance Level')"
  echo "case 32K gen64 A/B/A/B then 110K gen64 pair; 20s warm-up per arm"; } > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"
soak() { timeout 200 ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
    --ctx-frontiers 2048 --ctx-alloc 4096 --gen-tokens 0 --ssd-streaming \
    --ssd-streaming-cache-experts 92GB > /dev/null 2>&1
  rocm-smi --showclocks 2>/dev/null | grep -m1 sclk | sed 's/^/   clock: /'; }
run_one() {
  local name="$1" pf="$2" frontiers="$3" alloc="$4"
  local dir="$OUT/$name"; mkdir -p "$dir/logits" || return 1
  soak
  local envs=(); [ "$pf" = 1 ] && envs+=("DS4_ROCM_V41_Q8_FUSE_QAKV=1")
  echo "=== $name fuse=$pf ctx=$frontiers: start $(date -Is) ==="
  timeout -k 60 2000 env DS4_BENCH_DEV_STAT=1 "${envs[@]}" \
    ./ds4-bench --rocm -m "$MODEL" --prompt-file "$PROMPT" \
      --ctx-frontiers "$frontiers" --ctx-alloc "$alloc" --gen-tokens 64 --show-output \
      --ssd-streaming --ssd-streaming-cache-experts 92GB \
      --dump-frontier-logits-dir "$dir/logits" --csv "$dir/profile.csv" > "$dir/profile.log" 2>&1
  local rc=$?; echo "=== $name rc=$rc $(date -Is) ==="
  grep -E "prefill frontier" "$dir/profile.log" | sed 's/^/   /'
  return "$rc"
}
run_one k_off_a 0 32768,36864 40960
run_one k_on_a  1 32768,36864 40960
run_one k_off_b 0 32768,36864 40960
run_one k_on_b  1 32768,36864 40960
run_one h_off   0 110592,114688 118784
run_one h_on    1 110592,114688 118784
python3 -B - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
def rows(c):
    p = os.path.join(out, c, "profile.csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
names = ("k_off_a","k_on_a","k_off_b","k_on_b","h_off","h_on")
for c in names:
    for r in rows(c):
        print("%-8s ctx=%-6s prefill_tps=%-7s gen_tps=%-6s gen_steady=%-6s" % (
            c, r["ctx_tokens"], r["prefill_tps"], r["gen_tps"], r.get("gen_steady_tps","")))
def g(c):
    d = rows(c); return float(d[-1]["gen_tps"]) if d else float("nan")
for tag, a, b in (("32K decode", ("k_off_a","k_off_b"), ("k_on_a","k_on_b")),
                  ("110K decode", ("h_off",), ("h_on",))):
    off = [g(x) for x in a]; on = [g(x) for x in b]
    if all(x == x for x in off + on):
        mo, mn = sum(off)/len(off), sum(on)/len(on)
        print("%s: off=%s on=%s -> %+.1f%% disjoint=%s" % (tag, off, on, (mn/mo-1)*100, min(on) > max(off)))
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
for ref, other in (("k_off_a","k_on_a"), ("k_off_a","k_on_b"), ("h_off","h_on")):
    A = sorted(glob.glob(os.path.join(out, ref, "logits", "*")))
    B = sorted(glob.glob(os.path.join(out, other, "logits", "*")))
    for i,(fa,fb) in enumerate(zip(A,B)):
        va, vb = load(fa), load(fb)
        print("%s vs %s frontier %d exact=%s max_abs=%.6g" % (ref, other, i, va==vb, max(abs(x-y) for x,y in zip(va,vb))))
PY
exit 0

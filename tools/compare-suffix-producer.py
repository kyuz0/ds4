#!/usr/bin/env python3
"""Compare the layer-20 MoE producer captures of both arms.

Files (per arm, in <run>/base8k/<arm>/diag/):
  moe_x.raw        5120 f32  input row the MMQ consumed
  moe_weights.raw  6 f32     route weights
  moe_gate.raw     6*2304 f32 gate projection (fused path writes this)
  moe_mid_sync.raw 6*2304 f32 weighted SwiGLU mid, drained before the copy
  moe_mid_f32.raw  6*2304 f32 mid as captured by the down-launcher probe
  moe_counts_u32.raw, moe_down.raw  as before
Checks, in order: input/weights identity, then gate/mid per slot, then whether
the two mid dumps inside one arm agree (instrumentation reliability).
"""
import array
import os
import sys

U, H, X = 6, 2304, 5120


def rd(path, count, fmt="f"):
    a = array.array(fmt)
    with open(path, "rb") as fh:
        a.frombytes(fh.read())
    assert len(a) == count, (path, len(a), count)
    return a


def slots(a):
    return [a[j * H:(j + 1) * H] for j in range(U)]


def cmp(name, a, b):
    fa, fb = slots(a), slots(b)
    out = []
    for j in range(U):
        diffs = 0
        mx = 0.0
        scale = 0.0
        for x, y in zip(fa[j], fb[j]):
            if abs(y) > scale:
                scale = abs(y)
            if x != y:
                diffs += 1
                d = abs(x - y)
                if d > mx:
                    mx = d
        out.append((j, diffs, mx, scale))
    print("%s:" % name)
    for j, diffs, mx, scale in out:
        rel = mx / scale if scale else 0.0
        print("  slot%d diffs=%5d/%d max_abs=%-11.4g scale=%-8.4g rel=%.3g"
              % (j, diffs, H, mx, scale, rel))


def main():
    run = sys.argv[1]
    cand = os.path.join(run, "base8k", "candidate", "diag")
    ref = os.path.join(run, "base8k", "reference", "diag")

    def have(d, n):
        return os.path.exists(os.path.join(d, n))

    for arm, d in (("candidate", cand), ("reference", ref)):
        print("%s: %s" % (arm, ",".join(sorted(os.path.basename(p) for p in
                                               [os.path.join(d, n) for n in
                                                ("moe_x.raw", "moe_gate.raw", "moe_mid_sync.raw",
                                                 "moe_mid_f32.raw", "moe_down.raw")] if os.path.exists(p)))))
    if not (have(cand, "moe_gate.raw") and have(ref, "moe_gate.raw")):
        print("*** fused-path probe files missing; cannot separate input from projection ***")
        return 1

    xc = rd(os.path.join(cand, "moe_x.raw"), X)
    xr = rd(os.path.join(ref, "moe_x.raw"), X)
    same_x = xc == xr
    mx = max((abs(a - b) for a, b in zip(xc, xr)), default=0.0)
    print("\nINPUT consumed by MMQ: identical=%s max_abs=%.4g ndiff=%d"
          % (same_x, mx, sum(1 for a, b in zip(xc, xr) if a != b)))

    wc = rd(os.path.join(cand, "moe_weights.raw"), U)
    wr = rd(os.path.join(ref, "moe_weights.raw"), U)
    print("ROUTE weights: identical=%s" % (wc == wr))

    cmp("GATE candidate vs reference", rd(os.path.join(cand, "moe_gate.raw"), U * H),
        rd(os.path.join(ref, "moe_gate.raw"), U * H))
    cmp("MID(sync) candidate vs reference", rd(os.path.join(cand, "moe_mid_sync.raw"), U * H),
        rd(os.path.join(ref, "moe_mid_sync.raw"), U * H))

    for arm, d in (("candidate", cand), ("reference", ref)):
        if have(d, "moe_mid_f32.raw"):
            cmp("MID down-probe vs MID sync inside %s" % arm,
                rd(os.path.join(d, "moe_mid_f32.raw"), U * H),
                rd(os.path.join(d, "moe_mid_sync.raw"), U * H))
    return 0


if __name__ == "__main__":
    sys.exit(main())

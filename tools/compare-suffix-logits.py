#!/usr/bin/env python3
"""Full logit-distribution comparison for two frontier logits dumps.

Reports absolute/relative quantiles, top-1 margin, selected-token probability,
top-k overlap and how far the perturbation is from flipping the ranking.
"""
import glob
import json
import math
import os
import sys


def find_logits(obj):
    best = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            if len(cur) > len(best) and cur and all(isinstance(v, (int, float)) for v in cur):
                best = cur
            else:
                stack.extend(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
    return best


def load(path):
    with open(path) as fh:
        return find_logits(json.load(fh))


def quantiles(vals, qs):
    s = sorted(vals)
    n = len(s)
    return [s[min(n - 1, int(q * n))] for q in qs]


def main():
    ref_dir, cand_dir = sys.argv[1], sys.argv[2]
    ref = sorted(glob.glob(os.path.join(ref_dir, "*")))
    cand = sorted(glob.glob(os.path.join(cand_dir, "*")))
    print("ref files=%d cand files=%d" % (len(ref), len(cand)))
    rc = 0
    for a_path, b_path in zip(ref, cand):
        la, lb = load(a_path), load(b_path)
        name = os.path.basename(a_path)
        if len(la) != len(lb) or not la:
            print("%s: shape mismatch %d/%d" % (name, len(la), len(lb)))
            rc = 1
            continue
        n = len(la)
        dabs = [abs(x - y) for x, y in zip(la, lb)]
        q = quantiles(dabs, (0.5, 0.9, 0.99, 0.999, 0.9999))
        rel = [abs(x - y) / max(1e-9, abs(y)) for x, y in zip(la, lb)]
        rel.sort()
        ta = sorted(range(n), key=lambda i: -la[i])[:10]
        tb = sorted(range(n), key=lambda i: -lb[i])[:10]
        margin = la[ta[0]] - la[ta[1]]
        margin_b = lb[tb[0]] - lb[tb[1]]
        ma = max(la)
        mb = max(lb)
        ea = [math.exp(x - ma) for x in la]
        eb = [math.exp(x - mb) for x in lb]
        za = sum(ea)
        zb = sum(eb)
        pa = ea[ta[0]] / za
        pb = eb[tb[0]] / zb
        rank_a = {t: i for i, t in enumerate(sorted(range(n), key=lambda i: -la[i]))}
        rank_b = {t: i for i, t in enumerate(sorted(range(n), key=lambda i: -lb[i]))}
        moved = sum(1 for t in rank_a if rank_a[t] != rank_b[t])
        print("\n== %s (vocab %d)" % (name, n))
        print("  abs delta: mean %.5f p50 %.5f p90 %.5f p99 %.5f p99.9 %.5f max %.5f"
              % (sum(dabs) / n, q[0], q[1], q[2], q[3], q[4]))
        print("  rel delta: p50 %.3g p90 %.3g p99 %.3g p99.9 %.3g max %.3g"
              % (rel[n // 2], rel[int(0.9 * n)], rel[int(0.99 * n)], rel[int(0.999 * n)], rel[-1]))
        print("  argmax ref=%d cand=%d same=%s logit %.6f/%.6f" % (ta[0], tb[0], ta[0] == tb[0],
                                                                   la[ta[0]], lb[tb[0]]))
        print("  top10 overlap=%d/10  top10_ref=%s" % (len(set(ta) & set(tb)), ta[:6]))
        print("  margin top1-top2 ref=%.6f cand=%.6f (ratio %.4f)"
              % (margin, margin_b, margin_b / margin if margin else float("nan")))
        print("  p(selected token) ref=%.6f cand=%.6f rel=%.3g" % (pa, pb, abs(pa - pb) / pa))
        print("  max delta / margin = %.4f  (>=1 means a possible flip)"
              % (max(dabs) / margin if margin else float("inf")))
        print("  tokens with changed rank %d (%.2f%%), max rank move %d"
              % (moved, 100.0 * moved / n, max(abs(rank_a[t] - rank_b[t]) for t in rank_a)))
        if ta[0] != tb[0]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

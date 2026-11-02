#!/usr/bin/env python3
"""Compare two V4.1 session payloads (--dump-session-payload-dir) span by span.

Layout from ds41_save_payload: 13 u32 header fields, `len` u32 checkpoint
tokens, VOCAB f32 logits, then ds41_state_spans():
  40 x window[0..39]                raw * 512 * 4 bytes, raw = min(pos,128)
  per i in 0..3: compressed[i]      live * 512 * 4, live = pos/(i<3?2:1)
                 index_cache[i]     live * 128 * 4
                 if i < 3 and pos odd: previous_kv[i], previous_score[i] (512*4)
"""
import array
import struct
import sys

VOCAB = 129280
HDR = 13


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    h = struct.unpack_from("<%dI" % HDR, raw, 0)
    pos, ctx = h[7], h[2]
    off = HDR * 4
    tokens = array.array("I")
    tokens.frombytes(raw[off:off + pos * 4])
    off += pos * 4
    logits = array.array("f")
    logits.frombytes(raw[off:off + VOCAB * 4])
    off += VOCAB * 4
    spans = []
    cur = min(pos, 128)
    for il in range(40):
        n = cur * 512
        spans.append(("window[%d]" % il, off, n))
        off += n * 4
    for i in range(4):
        live = pos // (2 if i < 3 else 1)
        n = live * 512
        spans.append(("compressed[%d]" % i, off, n)); off += n * 4
        n = live * 128
        spans.append(("index_cache[%d]" % i, off, n)); off += n * 4
        if i < 3 and (pos & 1):
            spans.append(("previous_kv[%d]" % i, off, 512)); off += 512 * 4
            spans.append(("previous_score[%d]" % i, off, 512)); off += 512 * 4
    return {"raw": raw, "pos": pos, "ctx": ctx, "tokens": tokens,
            "logits": logits, "spans": spans, "end": off}


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def ulp32(v):
    a = abs(v)
    if a == 0.0:
        return 1e-45
    bits = struct.unpack("<I", struct.pack("<f", a))[0]
    return f32(struct.unpack("<f", struct.pack("<I", bits + 1))[0]) - a


def cmp_span(name, a, b, oa, ob, n):
    fa = array.array("f"); fa.frombytes(a[oa:oa + n * 4])
    fb = array.array("f"); fb.frombytes(b[ob:ob + n * 4])
    mx = 0.0; scale = 0.0; nd = 0; big = 0
    for x, y in zip(fa, fb):
        ay = abs(y)
        if ay > scale:
            scale = ay
        if x != y:
            nd += 1
            d = abs(x - y)
            if d > mx:
                mx = d
            if d > 1e-6:
                big += 1
    rel = mx / scale if scale else 0.0
    ulps = mx / ulp32(scale) if scale else 0.0
    return name, mx, scale, rel, ulps, nd, big, n


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    a, b = load(pa), load(pb)
    print("payload A pos=%d ctx=%d bytes=%d" % (a["pos"], a["ctx"], len(a["raw"])))
    print("payload B pos=%d ctx=%d bytes=%d" % (b["pos"], b["ctx"], len(b["raw"])))
    layout_ok = (a["end"] == len(a["raw"]) == len(b["raw"]) == b["end"])
    print("layout check: parsed end A=%d B=%d file=%d/%d -> %s" %
          (a["end"], b["end"], len(a["raw"]), len(b["raw"]),
           "OK" if layout_ok else "MISMATCH (span names unreliable)"))
    if a["pos"] != b["pos"]:
        print("*** position mismatch; aborting span compare ***")
        return 1
    if not layout_ok:
        print("*** payload layout mismatch; only the header/logit part is comparable ***")
        return 1
    tok_same = a["tokens"] == b["tokens"]
    print("token history identical:", tok_same, "len", len(a["tokens"]))
    print("%-20s %11s %11s %10s %9s %9s %9s" %
          ("span", "max_abs", "scale", "max_rel", "f32_ulps", "diffs", "count"))
    worst = []
    for (name, oa, n) in a["spans"]:
        ob = dict((s[0], s[1]) for s in b["spans"])[name]
        row = cmp_span(name, a["raw"], b["raw"], oa, ob, n)
        worst.append(row)
        if row[1] != 0.0:
            print("%-20s %11.4g %11.4g %10.3g %9.3g %9d %9d" %
                  (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
    clean = [r[0] for r in worst if r[1] == 0.0]
    print("\nspans exactly equal (%d):" % len(clean))
    print("  " + " ".join(clean))
    w21 = [r for r in worst if r[0].startswith("window[") and 21 <= int(r[0][7:-1]) <= 39]
    print("\nwindow[21..39] check: %d spans, max_abs=%.4g, max_rel=%.4g, max_ulps=%.4g" %
          (len(w21), max(r[1] for r in w21), max(r[3] for r in w21), max(r[4] for r in w21)))
    for r in w21:
        print("  %-14s max_abs=%-11.4g max_rel=%-10.3g ulps=%-8.3g diffs=%d/%d" %
              (r[0], r[1], r[3], r[4], r[5], r[7]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

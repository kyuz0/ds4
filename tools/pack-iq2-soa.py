#!/usr/bin/env python3
"""Prototype IQ2_XXS SoA packer/verifier (sidecar layout, no engine changes).

Per tensor the sidecar holds one global dq plane (2 B per 66-byte block) followed
by one global qs plane (64 B per block), which is exactly what
load_tiles_iq2_xxs_soa expects (dq plane first, 64-byte aligned, then qs).
This tool packs ONE expert of one tensor and then re-interleaves it to prove the
transform is lossless, so the layout can be validated before any engine work.
"""
import struct
import sys

BLOCK = 66          # IQ2_XXS block: 2 B scale + 64 B quants
ROWS = 2304
BLOCKS_PER_ROW = 5120 // 256
EXPERT_BLOCKS = ROWS * BLOCKS_PER_ROW     # 46080
EXPERT_BYTES = EXPERT_BLOCKS * BLOCK      # 3041280 == 2304*1320

T = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
     10: "<Q", 11: "<q", 12: "<d"}


def gguf_tensor(path, want):
    f = open(path, "rb")
    f.read(4)
    struct.unpack("<I", f.read(4))
    nt, nkv = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [rval(et) for _ in range(n)]
        fmt = T[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    meta = {}
    for _ in range(nkv):
        k = rstr()
        vt, = struct.unpack("<I", f.read(4))
        meta[k] = rval(vt)
    off = None
    for _ in range(nt):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        struct.unpack("<I", f.read(4))
        rel, = struct.unpack("<Q", f.read(8))
        if name == want:
            off = rel
    pos = f.tell()
    al = meta.get("general.alignment", 32)
    base = (pos + al - 1) // al * al
    f.close()
    if off is None:
        raise SystemExit("tensor not found: " + want)
    return base + off


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: %s MODEL TENSOR EXPERT OUT" % sys.argv[0])
    model, tensor, expert, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    off = gguf_tensor(model, tensor) + expert * EXPERT_BYTES
    with open(model, "rb") as f:
        f.seek(off)
        raw = f.read(EXPERT_BYTES)
    if len(raw) != EXPERT_BYTES:
        raise SystemExit("short read")
    dq = b"".join(raw[i * BLOCK:i * BLOCK + 2] for i in range(EXPERT_BLOCKS))
    qs = b"".join(raw[i * BLOCK + 2:(i + 1) * BLOCK] for i in range(EXPERT_BLOCKS))
    if len(dq) != EXPERT_BLOCKS * 2 or len(qs) != EXPERT_BLOCKS * 64:
        raise SystemExit("bad plane sizes")
    with open(out, "wb") as f:
        f.write(dq + qs)
    back = bytearray(EXPERT_BYTES)
    for i in range(EXPERT_BLOCKS):
        back[i * BLOCK:i * BLOCK + 2] = dq[i * 2:i * 2 + 2]
        back[i * BLOCK + 2:(i + 1) * BLOCK] = qs[i * 64:(i + 1) * 64]
    ok = bytes(back) == raw
    print("tensor=%s expert=%d blocks=%d sidecar_bytes=%d roundtrip_identical=%s"
          % (tensor, expert, EXPERT_BLOCKS, EXPERT_BLOCKS * BLOCK, ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

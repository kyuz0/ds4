#!/usr/bin/env python3
"""Sparse mirror of the routed experts of a ds4 GGUF on a second drive.

The destination has the same size and byte offsets as the model file; only the
ffn_{gate,up,down}_exps ranges are written (holes elsewhere), so for DeepSeek V4.1
Flash Q2 it takes 153 GB of the 366 GB file. With DS4_ROCM_STREAM_MIRROR=<dest>
the ROCm engine alternates streamed expert reads between the two drives.

Reads and writes use O_DIRECT (no page-cache pollution). At the end 512 random
1 MiB blocks are compared, and <dest>.ok records the source size and mtime; the
engine ignores a mirror whose marker does not match the model file.

usage: mirror_experts.py MODEL.gguf DEST.gguf   (needs `pip install gguf`)
"""
import mmap, os, random, sys, time
import gguf

ALIGN = 4096
CHUNK = 64 << 20


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    reader = gguf.GGUFReader(src)
    ranges = []
    for t in reader.tensors:
        if t.name.split('.')[-2] in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
            a = int(t.data_offset) // ALIGN * ALIGN
            b = -(-(int(t.data_offset) + int(t.n_bytes)) // ALIGN) * ALIGN
            ranges.append([a, b])
    ranges.sort()
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    size = os.path.getsize(src)
    total = sum(b - a for a, b in merged)
    print(f"{len(ranges)} expert tensors, {len(merged)} ranges, {total / 1e9:.2f} GB to copy", flush=True)

    fs = os.open(src, os.O_RDONLY | os.O_DIRECT)
    fd = os.open(dst, os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o644)
    os.ftruncate(fd, size)
    buf = mmap.mmap(-1, CHUNK)
    done, last, t0 = 0, 0, time.time()
    for a, b in merged:
        off = a
        while off < b:
            n = min(CHUNK, b - off, size - off)
            if n % ALIGN:  # unaligned file tail: buffered copy
                with open(src, 'rb') as fin, open(dst, 'r+b') as fout:
                    fin.seek(off); fout.seek(off); fout.write(fin.read(n))
            else:
                got = os.preadv(fs, [memoryview(buf)[:n]], off)
                assert got == n, (got, n)
                put = os.pwritev(fd, [memoryview(buf)[:n]], off)
                assert put == n, (put, n)
            off += n
            done += n
            if done - last >= (8 << 30):
                last = done
                print(f"  {done / 1e9:7.1f}/{total / 1e9:.1f} GB  {done / (time.time() - t0) / 1e9:.2f} GB/s", flush=True)
    os.fsync(fd)
    print(f"copied {done / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)

    rnd, bad = random.Random(20260925), 0
    for _ in range(512):
        a, b = rnd.choice(merged)
        off = a + rnd.randrange(max(1, (b - a - (1 << 20)) // ALIGN)) * ALIGN
        n = min(1 << 20, b - off)
        os.preadv(fs, [memoryview(buf)[:n]], off); s1 = bytes(buf[:n])
        os.preadv(fd, [memoryview(buf)[:n]], off); s2 = bytes(buf[:n])
        bad += s1 != s2
    print(f"sampled check: {512 - bad}/512 blocks identical", flush=True)
    os.close(fs)
    os.close(fd)
    if bad:
        raise SystemExit("mirror differs from the model: no .ok marker written")
    with open(dst + '.ok', 'w') as f:
        f.write(f"src={src}\nsize={size}\nsrc_mtime={int(os.path.getmtime(src))}\n"
                f"ranges={len(merged)}\nbytes={total}\nsample_ok={512 - bad}/512\n")


if __name__ == '__main__':
    main()

"""Minimal GGUF header reader (stdlib only). Reads metadata and tensor infos, never data."""
import struct
from pathlib import Path

from .common import Refuse

# ggml type id -> (name, block elements, block bytes)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1),
    25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}
_KEEP_ARRAY = 64  # arrays longer than this are summarised, not stored


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise Refuse("truncated GGUF header")
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.unpack("<Q")
        if n > 1 << 24:
            raise Refuse("implausible GGUF string length %d" % n)
        return self.read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _SCALAR:
            return self.unpack(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.unpack("<I")
            n = self.unpack("<Q")
            if etype in _SCALAR and n > _KEEP_ARRAY:
                self.fh.seek(n * struct.calcsize(_SCALAR[etype]), 1)
                return {"array_len": n}
            items = [self.value(etype) for _ in range(n)]
            return items if n <= _KEEP_ARRAY else {"array_len": n}
        raise Refuse("unknown GGUF value type %d" % vtype)


def read_gguf(path):
    path = Path(path)
    if not path.is_file():
        raise Refuse("model file not found: %s" % path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.read(4) != b"GGUF":
            raise Refuse("%s is not a GGUF file" % path)
        version = r.unpack("<I")
        if version < 2:
            raise Refuse("GGUF v%d is not supported" % version)
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.unpack("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(nd)]
            ttype = r.unpack("<I")
            offset = r.unpack("<Q")
            tensors.append({"name": name, "dims": dims, "type_id": ttype, "offset": offset})
        align = int(meta.get("general.alignment", 32))
        data_start = (fh.tell() + align - 1) // align * align
    _fill_bytes(tensors, size - data_start, align)
    return {"path": str(path), "version": version, "file_bytes": size,
            "alignment": align, "data_start": data_start, "meta": meta, "tensors": tensors}


def _fill_bytes(tensors, data_bytes, align):
    """Nominal bytes from the type table, cross-checked against offset spacing."""
    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    for pos, i in enumerate(order):
        t = tensors[i]
        nxt = tensors[order[pos + 1]]["offset"] if pos + 1 < len(order) else data_bytes
        t["span_bytes"] = nxt - t["offset"]
        n = 1
        for d in t["dims"]:
            n *= d
        t["elements"] = n
        info = GGML_TYPES.get(t["type_id"])
        if info is None:
            t["type"] = "type%d" % t["type_id"]
            t["bytes"] = t["span_bytes"]
            t["bytes_source"] = "offset_span"
            continue
        name, be, bb = info
        t["type"] = name
        if t["dims"] and t["dims"][0] % be:
            t["bytes"] = t["span_bytes"]
            t["bytes_source"] = "offset_span(row not block aligned)"
            continue
        t["bytes"] = n // be * bb
        t["bytes_source"] = "type_table"
        if t["bytes"] > t["span_bytes"] or t["span_bytes"] - t["bytes"] >= max(align, 1) * 2:
            t["bytes_mismatch"] = True

"""ak facts: model shapes and byte budgets read from the GGUF, never typed by hand."""
import re

from .common import Refuse, parse_rate
from .gguf import read_gguf

_LAYER = re.compile(r"^blk\.(\d+)\.")
# Metadata suffixes that describe the architecture (prefix is the arch name).
_HPARAM_HINTS = ("block_count", "embedding_length", "feed_forward_length", "expert_count",
                 "expert_used_count", "expert_feed_forward_length", "head_count",
                 "head_count_kv", "context_length", "key_length", "value_length",
                 "q_lora_rank", "kv_lora_rank", "expert_shared_count", "vocab_size",
                 "leading_dense_block_count", "rope", "indexer", "compress", "window")


def _canon(name):
    return _LAYER.sub("blk.*.", name)


def _arch(meta):
    return meta.get("general.architecture", "")


def _meta_int(meta, suffix):
    arch = _arch(meta)
    for key in ("%s.%s" % (arch, suffix),):
        if isinstance(meta.get(key), int):
            return meta[key]
    for k, v in meta.items():
        if k.endswith("." + suffix) and isinstance(v, int):
            return v
    return None


def summary(args):
    g = read_gguf(args.model)
    meta = g["meta"]
    hp = {k: v for k, v in meta.items()
          if not k.startswith("tokenizer.") and any(h in k for h in _HPARAM_HINTS)
          and not isinstance(v, dict)}
    by_type = {}
    for t in g["tensors"]:
        e = by_type.setdefault(t["type"], {"tensors": 0, "bytes": 0})
        e["tensors"] += 1
        e["bytes"] += t["bytes"]
    mism = [t["name"] for t in g["tensors"] if t.get("bytes_mismatch")]
    total = sum(t["bytes"] for t in g["tensors"])
    return {"ok": True, "model": g["path"], "architecture": _arch(meta),
            "name": meta.get("general.name"), "gguf_version": g["version"],
            "file_GB": round(g["file_bytes"] / 1e9, 3), "tensor_GB": round(total / 1e9, 3),
            "n_tensors": len(g["tensors"]), "hparams": hp,
            "bytes_by_type_GB": {k: round(v["bytes"] / 1e9, 3) for k, v in
                                 sorted(by_type.items(), key=lambda kv: -kv[1]["bytes"])},
            "byte_mismatches": mism[:10], "n_byte_mismatches": len(mism)}


def tensors(args):
    g = read_gguf(args.model)
    pat = re.compile(args.match) if args.match else None
    rows = [t for t in g["tensors"] if not pat or pat.search(t["name"])]
    if args.group:
        groups = {}
        for t in rows:
            key = _canon(t["name"])
            e = groups.setdefault(key, {"name": key, "count": 0, "types": set(), "dims": set(),
                                        "bytes_each": set(), "bytes_total": 0})
            e["count"] += 1
            e["types"].add(t["type"])
            e["dims"].add("x".join(map(str, t["dims"])))
            e["bytes_each"].add(t["bytes"])
            e["bytes_total"] += t["bytes"]
        out = []
        for e in sorted(groups.values(), key=lambda e: -e["bytes_total"]):
            out.append({"name": e["name"], "count": e["count"], "type": "/".join(sorted(e["types"])),
                        "dims_ne0_first": sorted(e["dims"]),
                        "MB_each": sorted(round(b / 1e6, 3) for b in e["bytes_each"]),
                        "MB_total": round(e["bytes_total"] / 1e6, 2)})
        rows_out = out
    else:
        rows_out = [{"name": t["name"], "type": t["type"], "dims_ne0_first": t["dims"],
                     "MB": round(t["bytes"] / 1e6, 4), "bytes": t["bytes"]} for t in rows]
    limit = args.limit
    return {"ok": True, "matched": len(rows_out), "shown": min(limit, len(rows_out)),
            "note": "dims are GGUF order: ne0 (row length / input dim) first",
            "tensors": rows_out[:limit]}


def decode_bytes(args):
    """Bytes a single decode token must stream, by tensor role, with stated assumptions."""
    g = read_gguf(args.model)
    meta = g["meta"]
    n_expert = args.expert_count or _meta_int(meta, "expert_count")
    k_used = args.experts_used or _meta_int(meta, "expert_used_count")
    if not n_expert or not k_used:
        raise Refuse("expert_count / expert_used_count not found in metadata",
                     fix="pass --expert-count and --experts-used explicitly")
    excl = re.compile(args.exclude) if args.exclude else None
    exp_pat = re.compile(args.expert_pattern)
    roles = {}
    layers = set()
    excluded = []
    for t in g["tensors"]:
        name = t["name"]
        if excl and excl.search(name):
            excluded.append(name)
            continue
        if name.startswith("token_embd"):
            continue  # one row gathered per token
        m = _LAYER.match(name)
        if m:
            layers.add(int(m.group(1)))
        if exp_pat.search(name):
            if n_expert not in t["dims"]:
                raise Refuse("%s matches the expert pattern but has no dim == %d" % (name, n_expert),
                             fix="check --expert-count or --expert-pattern")
            read = t["bytes"] * k_used / n_expert
            kind = "routed_expert"
        else:
            read = t["bytes"]
            kind = "dense"
        key = _canon(name)
        e = roles.setdefault(key, {"role": key, "kind": kind, "layers": 0, "bytes_per_token": 0.0})
        e["layers"] += 1
        e["bytes_per_token"] += read
    total = sum(e["bytes_per_token"] for e in roles.values())
    if total <= 0:
        raise Refuse("no tensors counted")
    ranked = sorted(roles.values(), key=lambda e: -e["bytes_per_token"])
    out_roles = [{"role": e["role"], "kind": e["kind"], "layers": e["layers"],
                  "MB_per_layer": round(e["bytes_per_token"] / e["layers"] / 1e6, 3),
                  "MB_per_token": round(e["bytes_per_token"] / 1e6, 1),
                  "share_pct": round(100 * e["bytes_per_token"] / total, 1)}
                 for e in ranked[:args.limit]]
    by_kind = {}
    for e in roles.values():
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + e["bytes_per_token"]
    res = {"ok": True, "experts": {"count": n_expert, "used": k_used},
           "n_layers_seen": len(layers), "GB_per_token": round(total / 1e9, 3),
           "GB_per_token_by_kind": {k: round(v / 1e9, 3) for k, v in by_kind.items()},
           "roles": out_roles, "roles_omitted": max(0, len(ranked) - args.limit),
           "excluded_tensors": len(excluded),
           "assumptions": [
               "every non-expert tensor in a layer is read once per decode token",
               "routed experts read used/count of their bytes (no cross-token reuse)",
               "token_embd excluded (one row gathered); output/lm head included",
               "cache hits vs SSD reads are not modelled: this is GPU-side weight traffic"]}
    if args.ceiling:
        bps = parse_rate(args.ceiling)
        res["ceiling"] = {"rate_GBps": round(bps / 1e9, 1),
                          "min_ms_per_token": round(total / bps * 1e3, 2),
                          "max_tokens_per_s": round(bps / total, 2)}
    return res


def register(sub):
    p = sub.add_parser("facts", help="model shapes and bytes from the GGUF")
    s = p.add_subparsers(dest="facts_cmd", required=True)
    a = s.add_parser("summary", help="architecture metadata and bytes by type")
    a.add_argument("model")
    a.set_defaults(func=summary)
    a = s.add_parser("tensors", help="list tensors with type, dims and exact bytes")
    a.add_argument("model")
    a.add_argument("--match", help="regex on tensor name")
    a.add_argument("--group", action="store_true", help="collapse blk.N. into blk.*.")
    a.add_argument("--limit", type=int, default=40)
    a.set_defaults(func=tensors)
    a = s.add_parser("decode-bytes", help="weight bytes streamed per decode token, by role")
    a.add_argument("model")
    a.add_argument("--experts-used", type=int)
    a.add_argument("--expert-count", type=int)
    a.add_argument("--expert-pattern", default=r"_exps\b|_exps\.",
                   help="regex identifying routed-expert tensors")
    a.add_argument("--exclude", help="regex of tensors to leave out (e.g. engram, mtp)")
    a.add_argument("--ceiling", help="measured read bandwidth, e.g. 223GB/s")
    a.add_argument("--limit", type=int, default=25)
    a.set_defaults(func=decode_bytes)

"""ak rate / ak budget: bytes-vs-time arithmetic and time budgets that refuse double counting."""
import re

from .common import Refuse, count_expr, load_json, parse_bytes, parse_rate, parse_time

# How a timing was obtained decides what it may be added to.
KINDS = {
    "gpu_event": "device time between events around known launches",
    "gpu_trace": "kernel busy time from a trace (rocprof); may absorb dependency waits",
    "host_wall": "host blocking time; overlaps GPU work, never additive with gpu_* time",
    "stage_wall": "wall time between host-synchronised stage marks; inflated, shares only",
}
ADDITIVE = {"gpu_event", "gpu_trace"}


def _verdict_rate(bps, ceiling, tol):
    if ceiling is None:
        return "no_ceiling", None
    pct = 100.0 * bps / ceiling
    if pct > 100.0 + tol:
        return "impossible_above_ceiling", pct
    return ("at_ceiling" if pct >= 85 else "below_ceiling"), pct


def rate(args):
    nbytes = parse_bytes(args.bytes)
    secs = parse_time(args.time)
    if secs <= 0:
        raise Refuse("time must be positive")
    ceiling = parse_rate(args.ceiling) if args.ceiling else None
    errors = []
    if args.calls is not None or args.expect_calls is not None:
        if args.calls is None or args.expect_calls is None:
            raise Refuse("--calls and --expect-calls go together",
                         fix="e.g. --calls 640 --expect-calls 'tokens*layers' --var tokens=16 --var layers=40")
        names = {}
        for v in args.var or []:
            k, sep, val = v.partition("=")
            if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k) or not val.isdigit():
                raise Refuse("--var %r must be name=integer" % v, fix="e.g. --var tokens=16")
            names[k] = int(val)
        expected = count_expr(args.expect_calls, names)
        if expected != args.calls:
            errors.append("call count %d != expected %d (%s): the counter is not scoped to the "
                          "population you are reasoning about" % (args.calls, expected, args.expect_calls))
    bps = nbytes / secs
    verdict, pct = _verdict_rate(bps, ceiling, args.tolerance_pct)
    if verdict == "impossible_above_ceiling":
        errors.append("rate exceeds the measured ceiling: the byte count or the time span is wrong "
                      "(wrong count of experts/rows/layers, or the timer does not cover the work)")
    return {"ok": not errors, "GBps": round(bps / 1e9, 2),
            "pct_of_ceiling": None if pct is None else round(pct, 1), "verdict": verdict,
            "inputs": {"bytes": nbytes, "seconds": secs}, "errors": errors}


def budget(args):
    spec = load_json(args.file, "budget file")
    token_ms = spec.get("token_ms")
    if not isinstance(token_ms, (int, float)) or token_ms <= 0:
        raise Refuse("budget needs token_ms (measured wall ms per token, from the same config)")
    names = {k: v for k, v in spec.items() if isinstance(v, int) and not isinstance(v, bool)}
    ceiling = parse_rate(spec["ceiling"]) if spec.get("ceiling") else None
    scope = spec.get("scope", "decode")
    entries = spec.get("entries") or []
    by_label = {}
    errors, warnings = [], []
    for e in entries:
        lab = e.get("label")
        if not lab or lab in by_label:
            raise Refuse("every entry needs a unique label (duplicate or missing: %r)" % lab)
        if e.get("kind") not in KINDS:
            raise Refuse("entry %s: kind must be one of %s" % (lab, ", ".join(KINDS)),
                         fix="; ".join("%s = %s" % kv for kv in KINDS.items()))
        if not isinstance(e.get("ms"), (int, float)):
            raise Refuse("entry %s: ms (per token) is required" % lab)
        by_label[lab] = e
    for e in entries:
        lab = e["label"]
        if e.get("parent") and e["parent"] not in by_label:
            errors.append("%s: parent %r not in budget" % (lab, e["parent"]))
        if e.get("scope", scope) != scope:
            errors.append("%s: scope %r differs from budget scope %r (e.g. prefill rows in a "
                          "decode budget)" % (lab, e.get("scope"), scope))
        if "calls" in e or "expect_calls" in e:
            if "calls" not in e or "expect_calls" not in e:
                errors.append("%s: give both calls and expect_calls" % lab)
            else:
                exp = count_expr(e["expect_calls"], names)
                if exp != e["calls"]:
                    errors.append("%s: %d calls, expected %d (%s)" % (lab, e["calls"], exp, e["expect_calls"]))
        else:
            warnings.append("%s: no call-count reconciliation" % lab)
        if e.get("regime") == "serialized" and e["kind"] == "host_wall":
            warnings.append("%s: host time measured with per-leaf syncs is not representative "
                            "of the natural run" % lab)
        if "bytes" in e:
            bps = parse_bytes(e["bytes"]) / (e["ms"] / 1e3) if e["ms"] > 0 else 0
            v, pct = _verdict_rate(bps, ceiling, 5.0)
            e["_GBps"] = round(bps / 1e9, 1)
            e["_pct"] = None if pct is None else round(pct, 1)
            if v == "impossible_above_ceiling":
                errors.append("%s: %.1f GB/s exceeds ceiling; bytes or span wrong" % (lab, bps / 1e9))
    # children must not exceed parents
    for lab, e in by_label.items():
        kids = [k for k in entries if k.get("parent") == lab]
        if kids:
            s = sum(k["ms"] for k in kids)
            if s > e["ms"] * 1.05:
                errors.append("%s: children sum %.2f ms > parent %.2f ms" % (lab, s, e["ms"]))
            if any(k["kind"] != e["kind"] for k in kids):
                warnings.append("%s: children use a different timing kind than the parent" % lab)
    roots = [e for e in entries if not e.get("parent")]
    additive = [e for e in roots if e["kind"] in ADDITIVE]
    host = [e for e in roots if e["kind"] == "host_wall"]
    stage = [e for e in roots if e["kind"] == "stage_wall"]
    kinds_used = {e["kind"] for e in additive}
    if len(kinds_used) > 1:
        warnings.append("gpu_event and gpu_trace mixed in one sum; trace times may absorb waits")
    gpu_ms = sum(e["ms"] for e in additive)
    if gpu_ms > token_ms * 1.02:
        errors.append("attributed GPU time %.2f ms exceeds the token (%.2f ms): something is "
                      "double counted or from another config" % (gpu_ms, token_ms))
    if stage:
        warnings.append("stage_wall entries excluded from sums (host-sync inflated): %s"
                        % ", ".join(e["label"] for e in stage))
    host_ms = sum(e["ms"] for e in host)
    return {
        "ok": not errors,
        "token_ms": token_ms,
        "gpu_attributed_ms": round(gpu_ms, 2),
        "gpu_attributed_pct": round(100 * gpu_ms / token_ms, 1),
        "not_attributed_ms": round(token_ms - gpu_ms, 2),
        "host_wall_ms_not_additive": round(host_ms, 2),
        "note": ("host_wall time overlaps GPU work; the real idle time is token_ms minus GPU busy "
                 "time, measured in the natural (non-serialised) run"),
        "rows": [{"label": e["label"], "kind": e["kind"], "parent": e.get("parent"),
                  "ms": e["ms"], "GBps": e.get("_GBps"), "pct_ceiling": e.get("_pct")}
                 for e in entries],
        "errors": errors, "warnings": warnings,
    }


def register(sub):
    p = sub.add_parser("rate", help="bytes / time vs the measured ceiling, with call reconciliation")
    p.add_argument("--bytes", required=True, help="with unit: 36.5MB, 3.04GiB")
    p.add_argument("--time", required=True, help="with unit: 175us, 8.7ms")
    p.add_argument("--ceiling", help="measured read bandwidth, e.g. 223GB/s")
    p.add_argument("--calls", type=int)
    p.add_argument("--expect-calls", help="expression, e.g. 'tokens*layers'")
    p.add_argument("--var", action="append", help="name=int for --expect-calls")
    p.add_argument("--tolerance-pct", type=float, default=5.0)
    p.set_defaults(func=rate)
    p = sub.add_parser("budget", help="check a per-token time budget for double counting")
    p.add_argument("file", help="budget JSON (see README)")
    p.set_defaults(func=budget)

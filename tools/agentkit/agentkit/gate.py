"""ak gate: correctness gates (logit equality) and which gates a diff requires."""
import json
import math
import re
from pathlib import Path

from .common import Refuse, git, load_json, repo_root, write_json

_DEFAULT_RULES = Path(__file__).resolve().parent.parent / "gates.json"


def _load_logits(path):
    d = load_json(path, "logits dump")
    if isinstance(d, dict) and isinstance(d.get("logits"), list):
        return d.get("argmax_id"), d["logits"]
    if isinstance(d, list):
        return None, d
    raise Refuse("%s has no 'logits' array" % path)


def _topk(v, k):
    idx = [i for i, x in enumerate(v) if x is not None and not math.isnan(x)]
    idx.sort(key=lambda i: -v[i])
    return idx[:k]


def compare_files(fa, fb, k=10):
    _, a = _load_logits(fa)
    _, b = _load_logits(fb)
    if len(a) != len(b):
        return {"file": Path(fa).name, "verdict": "fail", "error": "vocab %d vs %d" % (len(a), len(b))}
    max_abs, n_diff, nonfinite = 0.0, 0, 0
    for x, y in zip(a, b):
        if x is None or y is None:
            if (x is None) != (y is None):
                nonfinite += 1
            continue
        if x != y:
            n_diff += 1
            max_abs = max(max_abs, abs(x - y))
    ta, tb = _topk(a, k), _topk(b, k)
    exact = n_diff == 0 and nonfinite == 0
    return {"file": Path(fa).name, "exact": exact, "n_diff": n_diff, "max_abs": max_abs,
            "nonfinite_mismatch": nonfinite, "argmax_equal": ta[:1] == tb[:1],
            "topk_order_equal": ta == tb, "topk_set_equal": set(ta) == set(tb)}


def _pair_files(da, db, pattern):
    A = {p.name: p for p in Path(da).glob(pattern)}
    B = {p.name: p for p in Path(db).glob(pattern)}
    if not A or not B:
        raise Refuse("no %s files in %s or %s" % (pattern, da, db),
                     fix="dump with ds4-bench --dump-frontier-logits-dir DIR")
    common = sorted(set(A) & set(B))
    missing = sorted(set(A) ^ set(B))
    return [(A[n], B[n]) for n in common], missing


def compare_dirs(da, db, mode="bit_exact", pattern="*.logits.json", k=10, tol=0.0):
    pairs, missing = _pair_files(da, db, pattern)
    rows = [compare_files(a, b, k) for a, b in pairs]
    fails = []
    for r in rows:
        if "error" in r:
            fails.append(r["file"])
        elif mode == "bit_exact" and not r["exact"]:
            fails.append(r["file"])
        elif mode == "argmax_stable" and not (r["argmax_equal"] and r["topk_set_equal"]
                                              and r["max_abs"] <= tol + 1e-30
                                              and r["nonfinite_mismatch"] == 0):
            fails.append(r["file"])
    kinds = sorted({re.sub(r"_\d+.*", "", r["file"]) for r in rows})
    verdict = "pass" if not fails and not missing else "fail"
    return {"ok": True, "gate": mode, "verdict": verdict, "dir_a": str(da), "dir_b": str(db),
            "files_compared": len(rows), "kinds": kinds, "missing_in_one_side": missing[:10],
            "failed": fails[:10], "worst_max_abs": max((r.get("max_abs", 0) for r in rows), default=0),
            "rows": rows[:8], "rows_omitted": max(0, len(rows) - 8)}


def cmd_logits(args):
    if args.mode == "argmax_stable" and args.tol is None:
        raise Refuse("argmax_stable needs an explicit --tol (max_abs allowed)",
                     fix="numerics-changing candidates must state their tolerance up front")
    res = compare_dirs(args.dir_a, args.dir_b, args.mode, args.pattern, args.topk, args.tol or 0.0)
    if "decode" not in " ".join(res["kinds"]):
        res["coverage_note"] = ("only prefill-frontier logits compared: this does NOT verify "
                                "decode-only code paths (single-row kernels)")
    if args.save:
        write_json(args.save, res)
        res["saved"] = args.save
    return res


def cmd_required(args):
    root = repo_root()
    rules = load_json(args.rules or _DEFAULT_RULES, "gate rules")["rules"]
    diff = git(root, "diff", "--unified=0", args.range, "--", *(args.paths or []))
    files, lines = [], {}
    cur, in_hunk = None, False
    for ln in diff.splitlines():
        head = re.match(r"^diff --git a/(.*) b/(.*)$", ln)
        if head:
            # b/ path covers added, modified, renamed and deleted files alike
            cur, in_hunk = head.group(2), False
            files.append(cur)
            lines[cur] = []
        elif ln.startswith("@@"):
            in_hunk = True
        elif cur and in_hunk and ln[:1] in ("+", "-"):
            lines[cur].append(ln[1:])   # content lines, including ones starting with ++ or --
    required = {}
    for r in rules:
        fre = re.compile(r.get("file", ".*"))
        lre = re.compile(r["line"]) if r.get("line") else None
        hits = []
        for f in files:
            if not fre.search(f):
                continue
            if lre is None or any(lre.search(l) for l in lines.get(f, [])):
                hits.append(f)
        if hits:
            e = required.setdefault(r["gate"], {"gate": r["gate"], "why": [], "files": set()})
            e["why"].append(r["why"])
            e["files"].update(hits)
    out = [{"gate": e["gate"], "why": sorted(set(e["why"])), "files": sorted(e["files"])[:8]}
           for e in required.values()]
    return {"ok": True, "range": args.range, "files_changed": len(files), "required": out,
            "note": "heuristic from gates.json; add rules when a gate was missed"}


def register(sub):
    p = sub.add_parser("gate", help="correctness gates")
    s = p.add_subparsers(dest="gate_cmd", required=True)
    a = s.add_parser("logits", help="compare two logits dump directories")
    a.add_argument("dir_a")
    a.add_argument("dir_b")
    a.add_argument("--mode", choices=["bit_exact", "argmax_stable"], default="bit_exact")
    a.add_argument("--tol", type=float, help="max_abs allowed in argmax_stable mode")
    a.add_argument("--pattern", default="*.logits.json")
    a.add_argument("--topk", type=int, default=10)
    a.add_argument("--save", help="write the full result JSON (evidence for ak ledger)")
    a.set_defaults(func=cmd_logits)
    a = s.add_parser("required", help="which gates a git range needs (from gates.json)")
    a.add_argument("range", help="e.g. main..HEAD or HEAD~1..HEAD")
    a.add_argument("paths", nargs="*")
    a.add_argument("--rules")
    a.set_defaults(func=cmd_required)

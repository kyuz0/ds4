"""ak ledger: append-only claims with evidence, retraction cascade, and staged promotion."""
import json
import re
import subprocess
import time
from pathlib import Path

from .ab import config
from .common import Refuse, git, load_json, repo_root

KINDS = ("hypothesis", "measurement", "change", "decision")
STATUSES = ("open", "confirmed", "rejected", "retracted", "suspect")
STAGES = ("proposed", "implemented", "gated", "measured", "confirmed", "promoted")
_NUMBER_WITH_UNIT = re.compile(
    r"[-+]?\d+(\.\d+)?\s*(%|ms|us|µs|s\b|GB/s|MB/s|GiB|MiB|GB|MB|tps|tok/s|x\b)", re.I)


def _path(root):
    p = Path(root) / config(root)["ledger_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _events(root):
    p = _path(root)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _append(root, ev):
    ev["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(_path(root), "a") as fh:
        fh.write(json.dumps(ev, separators=(",", ":")) + "\n")


def state(root):
    claims = {}
    for ev in _events(root):
        op = ev["op"]
        if op == "add":
            c = {k: ev.get(k) for k in ("id", "kind", "text", "evidence", "depends", "change")}
            c.update(status="open", stage="proposed" if ev["kind"] == "change" else None,
                     created=ev["at"], history=[], stage_evidence={})
            claims[ev["id"]] = c
        elif ev["id"] in claims:
            c = claims[ev["id"]]
            if op == "status":
                c["status"] = ev["status"]
                c["history"].append("%s %s: %s" % (ev["at"][:10], ev["status"], ev.get("reason", "")))
            elif op == "advance":
                c["stage"] = ev["stage"]
                c["stage_evidence"][ev["stage"]] = ev.get("evidence")
                c["history"].append("%s stage %s" % (ev["at"][:10], ev["stage"]))
    return claims


def _next_id(claims, kind):
    prefix = kind[0].upper()
    n = 1 + max([int(i[1:]) for i in claims if i.startswith(prefix)] or [0])
    return "%s%03d" % (prefix, n)


def _check_evidence(root, paths):
    out = []
    for p in paths or []:
        full = Path(p) if Path(p).is_absolute() else Path(root) / p
        if not full.exists():
            raise Refuse("evidence not found: %s" % p,
                         fix="evidence must be a file produced by ak (analysis.json, gate --save, budget)")
        out.append(str(full))
    return out


def cmd_add(args):
    root = repo_root()
    claims = state(root)
    if args.kind == "measurement" and not args.evidence:
        raise Refuse("a measurement needs --evidence (an ak output file)",
                     fix="numbers come from ak ab analyze / ak gate --save / ak budget, not from memory")
    if _NUMBER_WITH_UNIT.search(args.text) and not args.evidence:
        raise Refuse("claim text contains a quantity but no evidence file",
                     fix="attach --evidence, or record it as a hypothesis without numbers")
    for d in args.depends or []:
        if d not in claims:
            raise Refuse("depends on unknown claim %s" % d)
    cid = _next_id(claims, args.kind)
    _append(root, {"op": "add", "id": cid, "kind": args.kind, "text": args.text,
                   "evidence": _check_evidence(root, args.evidence), "depends": args.depends or [],
                   "change": args.change})
    return {"ok": True, "id": cid}


def _dependents(claims, cid):
    out, todo = set(), [cid]
    while todo:
        cur = todo.pop()
        for c in claims.values():
            if cur in (c["depends"] or []) and c["id"] not in out:
                out.add(c["id"])
                todo.append(c["id"])
    return out


def cmd_set(args):
    root = repo_root()
    claims = state(root)
    if args.id not in claims:
        raise Refuse("unknown claim %s" % args.id)
    if not args.reason:
        raise Refuse("--reason is required")
    if args.status == "confirmed" and claims[args.id]["kind"] == "measurement" and not claims[args.id]["evidence"]:
        raise Refuse("cannot confirm a measurement without evidence")
    _append(root, {"op": "status", "id": args.id, "status": args.status, "reason": args.reason})
    flagged = []
    if args.status in ("retracted", "rejected"):
        for d in sorted(_dependents(claims, args.id)):
            if claims[d]["status"] not in ("retracted", "rejected"):
                _append(root, {"op": "status", "id": d, "status": "suspect",
                               "reason": "depends on %s which was %s" % (args.id, args.status)})
                flagged.append(d)
    return {"ok": True, "id": args.id, "status": args.status, "dependents_marked_suspect": flagged}


def _require_ab(root, path, change_commit, cfg_sessions=None):
    a = load_json(path, "A/B analysis")
    if a.get("kind") != "ab":
        raise Refuse("evidence is not an A/B analysis (kind=%s)" % a.get("kind"))
    if not all(v["anchored"] for v in a["arms"].values()):
        raise Refuse("A/B arms are not commit-anchored", fix="use 'commit' arms built by ak ab build")
    wins = [k for k, v in a["verdicts"].items() if v == "win"]
    if not wins:
        raise Refuse("A/B shows no 'win' verdict: %s" % a["verdicts"])
    gate = a.get("gate")
    if not gate:
        raise Refuse("A/B analysis has no logits gate (dump_logits was off)",
                     fix="re-run the A/B with dump_logits enabled")
    if not gate.get("pass"):
        raise Refuse("A/B logits gate failed")
    if a.get("session_source") != "run":
        raise Refuse("A/B session identity was not recorded at run time (no host.txt)",
                     fix="use an analysis of a run produced by an 'ak ab plan' payload")
    b = a["arms"]["B"]["commit"]
    if change_commit and b:
        rc = subprocess.run(["git", "merge-base", "--is-ancestor", change_commit, b],
                            cwd=root, capture_output=True).returncode
        if rc != 0:
            raise Refuse("candidate arm %s does not contain the change commit %s"
                         % (b[:12], change_commit[:12]))
    return a, wins


def cmd_advance(args):
    root = repo_root()
    claims = state(root)
    c = claims.get(args.id)
    if not c or c["kind"] != "change":
        raise Refuse("%s is not a change claim" % args.id, fix="ak ledger add --kind change ...")
    if c["status"] in ("retracted", "rejected", "suspect"):
        raise Refuse("%s is %s and cannot advance" % (args.id, c["status"]),
                     fix="resolve it first: ak ledger show %s" % args.id)
    if args.stage in ("gated", "measured", "confirmed") and not args.evidence:
        raise Refuse("stage %s needs --evidence (an ak output file)" % args.stage)
    cur = STAGES.index(c["stage"])
    want = STAGES.index(args.stage)
    if want != cur + 1:
        raise Refuse("cannot go from %s to %s" % (c["stage"], args.stage),
                     fix="next stage is %s" % (STAGES[cur + 1] if cur + 1 < len(STAGES) else "none"))
    ev = {"op": "advance", "id": args.id, "stage": args.stage}
    if args.stage == "implemented":
        if not args.commit:
            raise Refuse("implemented needs --commit (the opt-in implementation)")
        ev["evidence"] = {"commit": git(root, "rev-parse", "--verify", args.commit + "^{commit}")}
    elif args.stage == "gated":
        g = load_json(args.evidence, "gate result")
        if g.get("verdict") != "pass":
            raise Refuse("gate verdict is %s" % g.get("verdict"))
        ev["evidence"] = {"file": str(Path(args.evidence).resolve()), "gate": g.get("gate"),
                          "kinds": g.get("kinds")}
        if args.decode_path and "decode" not in " ".join(g.get("kinds") or []):
            raise Refuse("change touches a decode path but the gate compared no decode logits",
                         fix="dump decode-step logits and gate them (see README)")
    elif args.stage in ("measured", "confirmed"):
        commit = (c["stage_evidence"].get("implemented") or {}).get("commit")
        a, wins = _require_ab(root, args.evidence, commit)
        ev["evidence"] = {"file": str(Path(args.evidence).resolve()), "wins": wins,
                          "session": a["session"], "date": a["date"]}
        if args.stage == "confirmed":
            first = c["stage_evidence"]["measured"]
            if a["session"] == first["session"] and a["date"] == first["date"]:
                raise Refuse("confirmation must come from a different session (reboot or another day)",
                             fix="repeat the A/B after a reboot or on another day")
    elif args.stage == "promoted":
        if not args.commit:
            raise Refuse("promoted needs --commit (the commit flipping the default)")
        ev["evidence"] = {"commit": git(root, "rev-parse", "--verify", args.commit + "^{commit}")}
    _append(root, ev)
    return {"ok": True, "id": args.id, "stage": args.stage}


def cmd_check_promote(args):
    root = repo_root()
    c = state(root).get(args.id)
    if not c or c["kind"] != "change":
        raise Refuse("%s is not a change claim" % args.id)
    need = STAGES[:STAGES.index("confirmed") + 1]
    missing = [s for s in need if s != "proposed" and s not in c["stage_evidence"]]
    return {"ok": True, "id": args.id, "stage": c["stage"], "may_promote": not missing
            and c["status"] not in ("retracted", "rejected", "suspect"),
            "missing_stages": missing, "status": c["status"]}


def cmd_show(args):
    root = repo_root()
    claims = state(root)
    if args.id:
        if args.id not in claims:
            raise Refuse("unknown claim %s" % args.id)
        return {"ok": True, "claim": claims[args.id]}
    rows = [c for c in claims.values() if (not args.status or c["status"] == args.status)
            and (not args.kind or c["kind"] == args.kind)]
    return {"ok": True, "count": len(rows),
            "claims": [{"id": c["id"], "kind": c["kind"], "status": c["status"], "stage": c["stage"],
                        "text": c["text"][:160]} for c in rows[-args.limit:]]}


def cmd_truth(args):
    root = repo_root()
    claims = state(root)
    L = ["# Current state (generated by `ak ledger truth`; do not edit)", ""]

    def sec(title, rows, fmt):
        L.append("## " + title)
        L.extend(fmt(c) for c in rows) if rows else L.append("- none")
        L.append("")
    ch = [c for c in claims.values() if c["kind"] == "change"]
    sec("Changes", ch, lambda c: "- %s [%s/%s] %s" % (c["id"], c["stage"], c["status"], c["text"]))
    live = [c for c in claims.values() if c["kind"] != "change" and c["status"] in ("confirmed", "open")]
    sec("Confirmed and open claims", live,
        lambda c: "- %s %s (%s): %s%s" % (c["id"], c["kind"], c["status"], c["text"],
                                         " — evidence: " + ", ".join(Path(e).name for e in c["evidence"])
                                         if c["evidence"] else ""))
    sus = [c for c in claims.values() if c["status"] == "suspect"]
    sec("Suspect (depends on something retracted)", sus, lambda c: "- %s: %s" % (c["id"], c["text"]))
    dead = [c for c in claims.values() if c["status"] in ("retracted", "rejected")]
    sec("Rejected / retracted (do not reuse)", dead,
        lambda c: "- %s %s: %s — %s" % (c["id"], c["status"], c["text"][:120],
                                        c["history"][-1] if c["history"] else ""))
    out = Path(root) / (args.out or "ledger/CURRENT.md")
    out.write_text("\n".join(L))
    return {"ok": True, "written": str(out), "changes": len(ch), "live": len(live),
            "suspect": len(sus), "dead": len(dead)}


def register(sub):
    p = sub.add_parser("ledger", help="claims with evidence, retraction cascade, staged promotion")
    s = p.add_subparsers(dest="ledger_cmd", required=True)
    a = s.add_parser("add")
    a.add_argument("--kind", choices=KINDS, required=True)
    a.add_argument("--text", required=True)
    a.add_argument("--evidence", nargs="*")
    a.add_argument("--depends", nargs="*")
    a.add_argument("--change", help="short name of the env knob / change")
    a.set_defaults(func=cmd_add)
    a = s.add_parser("set", help="set status; retract/reject marks dependents suspect")
    a.add_argument("id")
    a.add_argument("--status", choices=STATUSES, required=True)
    a.add_argument("--reason", required=True)
    a.set_defaults(func=cmd_set)
    a = s.add_parser("advance", help="move a change to its next stage with evidence")
    a.add_argument("id")
    a.add_argument("stage", choices=STAGES[1:])
    a.add_argument("--evidence")
    a.add_argument("--commit")
    a.add_argument("--decode-path", action="store_true", help="change touches decode-only code")
    a.set_defaults(func=cmd_advance)
    a = s.add_parser("check-promote")
    a.add_argument("id")
    a.set_defaults(func=cmd_check_promote)
    a = s.add_parser("show")
    a.add_argument("id", nargs="?")
    a.add_argument("--status", choices=STATUSES)
    a.add_argument("--kind", choices=KINDS)
    a.add_argument("--limit", type=int, default=30)
    a.set_defaults(func=cmd_show)
    a = s.add_parser("truth", help="regenerate ledger/CURRENT.md")
    a.add_argument("--out")
    a.set_defaults(func=cmd_truth)

"""ak ab: commit-anchored binaries, counterbalanced A/B payloads, and statistical verdicts."""
import csv
import json
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import time
from math import comb, exp, log
from pathlib import Path

from .common import (Refuse, boot_id, git, load_json, repo_root, sha256_file, state_dir,
                     write_json)
from .gate import compare_dirs

_DEFAULTS = Path(__file__).resolve().parent.parent / "config.json"
# Flags ak sets itself; a spec must not override them (ds4_bench.c accepts -m and --model).
_RESERVED_FLAGS = {"-m", "--model", "--prompt-file", "--csv", "--dump-frontier-logits-dir"}
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
AK = Path(__file__).resolve().parent.parent / "ak"


def config(root):
    cfg = load_json(_DEFAULTS, "agentkit config")
    local = state_dir(root) / "config.json"
    if local.exists():
        cfg.update(load_json(local, "local agentkit config"))
    return cfg


# ---------------------------------------------------------------- build

def bin_dir(root, sha):
    return state_dir(root) / "bins" / sha


def cmd_build(args):
    root = repo_root()
    cfg = config(root)
    sha = git(root, "rev-parse", "--verify", args.commit + "^{commit}")
    out = bin_dir(root, sha)
    binary = out / cfg["bench_binary"]
    if binary.exists() and not args.force:
        meta = load_json(out / "meta.json", "binary meta")
        return {"ok": True, "commit": sha, "binary": str(binary), "sha256": meta["sha256"],
                "cached": True}
    wt = state_dir(root) / "worktrees" / sha
    if not wt.exists():
        git(root, "worktree", "add", "--detach", str(wt), sha)
    cmd = args.build_cmd or cfg["build_cmd"]
    t0 = time.time()
    log = out / "build.log"
    out.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, shell=True, cwd=wt, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            raise Refuse("build of %s exceeded %ds" % (sha[:10], args.timeout),
                         fix="raise --timeout or check %s" % log)
    built = wt / cfg["bench_binary"]
    if rc != 0 or not built.exists():
        tail = log.read_text(errors="replace").splitlines()[-15:]
        raise Refuse("build failed (rc=%d) for %s" % (rc, sha[:10]), fix="see %s" % log,
                     log_tail=tail)
    shutil.copy2(built, binary)
    meta = {"commit": sha, "subject": git(root, "log", "-1", "--format=%s", sha),
            "build_cmd": cmd, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0), "sha256": sha256_file(binary)}
    write_json(out / "meta.json", meta)
    if args.remove_worktree:
        git(root, "worktree", "remove", "--force", str(wt))
    return {"ok": True, "commit": sha, "binary": str(binary), "sha256": meta["sha256"],
            "cached": False, "seconds": meta["seconds"]}


# ---------------------------------------------------------------- plan

def _resolve_arm(root, cfg, name, arm):
    env = {str(k): str(v) for k, v in (arm.get("env") or {}).items()}
    for k in env:
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k):
            raise Refuse("arm %s: bad env name %r" % (name, k))
    if "commit" in arm:
        sha = git(root, "rev-parse", "--verify", str(arm["commit"]) + "^{commit}")
        binary = bin_dir(root, sha) / cfg["bench_binary"]
        if not binary.exists():
            raise Refuse("arm %s: no binary built for %s" % (name, sha[:10]),
                         fix="run: ak ab build %s" % sha[:12])
        meta = load_json(binary.parent / "meta.json", "binary meta")
        if sha256_file(binary) != meta["sha256"]:
            raise Refuse("arm %s: cached binary changed since build" % name,
                         fix="ak ab build %s --force" % sha[:12])
        return {"name": name, "commit": sha, "binary": str(binary), "sha256": meta["sha256"],
                "env": env, "anchored": True}
    if "binary" in arm:
        b = Path(arm["binary"])
        if not b.is_file():
            raise Refuse("arm %s: binary not found: %s" % (name, b))
        return {"name": name, "commit": None, "binary": str(b.resolve()),
                "sha256": sha256_file(b), "env": env, "anchored": False}
    raise Refuse("arm %s needs 'commit' (preferred) or 'binary'" % name)


def _order(pairs):
    """ABBA counterbalancing: pair i runs A,B when i is even and B,A when odd."""
    return [(i, ("A", "B") if i % 2 == 0 else ("B", "A")) for i in range(pairs)]


def cmd_plan(args):
    root = repo_root()
    cfg = config(root)
    spec = load_json(args.spec, "A/B spec")
    for key in ("name", "model", "prompt", "bench_args", "arms"):
        if key not in spec:
            raise Refuse("spec is missing %r" % key, fix="see tools/agentkit/README.md (ab spec)")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", spec["name"]):
        raise Refuse("spec name must be [A-Za-z0-9._-]+")
    if set(spec["arms"]) != {"A", "B"}:
        raise Refuse("arms must be exactly A (baseline) and B (candidate)")
    pairs = int(spec.get("pairs", 4))
    if pairs < 4 or pairs % 2:
        raise Refuse("pairs must be even and >= 4 so the ABBA order is balanced")
    common = list(spec.get("common_args", []))
    bench = list(spec["bench_args"])
    for arg in bench + common:
        name = str(arg).split("=", 1)[0]
        if name in _RESERVED_FLAGS:
            raise Refuse("%s is set by ak; remove it from the spec" % name,
                         fix="model and prompt go in the spec's 'model' and 'prompt' fields")
    arms = {n: _resolve_arm(root, cfg, n, spec["arms"][n]) for n in ("A", "B")}
    aa = arms["A"]["sha256"] == arms["B"]["sha256"] and arms["A"]["env"] == arms["B"]["env"]
    model = str(Path(spec["model"]).expanduser())
    fit, warns = _fit_supervisor(root, args.profile, spec, pairs)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    runs_root = state_dir(root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    for n in range(100):
        run_dir = runs_root / ("%s-%s%s" % (spec["name"], stamp, "-%d" % n if n else ""))
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise Refuse("could not create a unique run directory in %s" % runs_root)
    warm = list(spec.get("warmup_args", cfg["warmup_args"]))
    runs = []
    for i, order in _order(pairs):
        for pos, arm in enumerate(order):
            runs.append({"idx": len(runs), "pair": i, "position": pos, "arm": arm,
                         "dir": "r%02d_%s" % (len(runs), arm)})
    case = {"model": model, "prompt": spec["prompt"], "common_args": common, "bench_args": bench,
            "global_env": spec.get("env", {}), "model_id": _model_id(model),
            "prompt_sha256": _prompt_sha(root, spec["prompt"])}
    manifest = {"name": spec["name"], "kind": "aa" if aa else "ab", "created": stamp,
                "repo_head": git(root, "rev-parse", "HEAD"), "arms": arms, "case": case,
                "case_signature": case_signature(case), "pairs": pairs, "runs": runs,
                "metrics": spec.get("metrics", cfg["metrics"]),
                "dump_logits": bool(spec.get("dump_logits", True)),
                "warmup_args": warm, "timeout_s": int(spec.get("timeout_s", 2400))}
    write_json(run_dir / "manifest.json", manifest)
    shutil.copy2(args.spec, run_dir / "spec.json")
    script = _payload(root, cfg, run_dir, manifest)
    payload = Path(args.out) if args.out else run_dir / "payload.sh"
    payload.write_text(script)
    payload.chmod(0o755)
    return {"ok": True, "kind": manifest["kind"], "run_dir": str(run_dir), "payload": str(payload),
            "order": "".join(r["arm"] for r in runs), "anchored": {n: a["anchored"] for n, a in arms.items()},
            "runtime": fit, "warnings": warns,
            "next": "launch the payload through run_benchmark; after resume run "
                    "'ak ab analyze %s' if result.json is missing" % run_dir}


def _profile_timeout(root, profile):
    """PAYLOAD_TIMEOUT from supervisor.d/<profile>.env (the supervisor kills the payload after it)."""
    path = Path(profile) if ("/" in profile or profile.endswith(".env")) else \
        Path(root) / "supervisor.d" / (profile + ".env")
    if not path.exists():
        return None, str(path)
    m = None
    for line in path.read_text().splitlines():
        hit = re.match(r"\s*PAYLOAD_TIMEOUT=([0-9]+)\s*(#.*)?$", line)
        if hit:
            m = int(hit.group(1))
    return m, str(path)


def _fit_supervisor(root, profile, spec, pairs):
    limit, path = _profile_timeout(root, profile)
    limit = limit or 3600
    runs = pairs * 2
    warm = int(spec.get("warmup_est_s", 60))
    warns = []
    if "est_run_s" in spec:
        est = runs * (int(spec["est_run_s"]) + warm)
        if est > 0.9 * limit:
            fit_pairs = max(0, int(0.9 * limit // (2 * (int(spec["est_run_s"]) + warm))) // 2 * 2)
            raise Refuse("plan needs ~%ds but the supervisor kills payloads after %ds (%s)"
                         % (est, limit, path),
                         fix="use pairs=%d per cycle (repeat cycles), shorten the case, or raise "
                             "PAYLOAD_TIMEOUT in the profile" % fit_pairs if fit_pairs >= 4 else
                             "shorten the case or raise PAYLOAD_TIMEOUT in the profile")
    else:
        est = None
        warns.append("no est_run_s in spec: cannot check the %ds supervisor limit; set it from "
                     "run_seconds_median of an earlier analysis" % limit)
    return {"supervisor_limit_s": limit, "estimated_s": est, "profile": path}, warns


def _model_id(model):
    """Size + mtime: hashing a 100+ GB model is not practical, but a re-quantised file changes these."""
    try:
        st = Path(model).stat()
        return "%d:%d" % (st.st_size, st.st_mtime_ns)
    except OSError:
        return "missing"


def _prompt_sha(root, prompt):
    p = Path(prompt) if Path(prompt).is_absolute() else Path(root) / prompt
    try:
        return sha256_file(p)
    except OSError:
        return "missing"


def case_signature(case):
    """Noise floors are only valid for the same workload: args, env, prompt content, model file."""
    keep = {k: case[k] for k in ("prompt", "common_args", "bench_args", "global_env")}
    keep["model"] = Path(case["model"]).name
    keep["model_id"] = case.get("model_id")
    keep["prompt_sha256"] = case.get("prompt_sha256")
    return json.dumps(keep, sort_keys=True)


def _payload(root, cfg, run_dir, m):
    q = shlex.quote
    L = ["#!/usr/bin/env bash",
         "# Generated by 'ak ab plan' for %s. Do not edit: change the spec and re-plan." % m["name"],
         "set -uo pipefail", "cd %s || exit 1" % q(str(root)), "RUN=%s" % q(str(run_dir)),
         "for n in $(compgen -v | grep '^DS4_'); do unset \"$n\"; done"]
    for k, v in m["case"]["global_env"].items():
        L.append("export %s=%s" % (k, q(str(v))))
    if cfg.get("busy_url"):
        L.append("curl -sf --max-time 3 %s >/dev/null 2>&1 && { echo '*** backend running ***'; exit 1; }"
                 % q(cfg["busy_url"]))
    L += ["[ -r %s ] || { echo '*** model missing ***'; exit 1; }" % q(m["case"]["model"]),
          "sclk() { cat %s 2>/dev/null | sed -n 's/^[0-9]*: *\\([0-9]*\\)[Mm]hz *\\*.*/\\1/p' | head -1; }"
          % cfg["sclk_glob"],
          "temp() { cat %s 2>/dev/null | head -1; }" % cfg["temp_glob"],
          "{ date -Is; uname -r; cat /proc/cmdline; cat /sys/firmware/acpi/platform_profile 2>/dev/null;"
          " tuned-adm active 2>/dev/null; cat /proc/sys/kernel/random/boot_id; } > \"$RUN/host.txt\""]
    base = [m["case"]["common_args"], ["-m", m["case"]["model"], "--prompt-file", m["case"]["prompt"]]]
    for r in m["runs"]:
        arm = m["arms"][r["arm"]]
        env = " ".join("%s=%s" % (k, q(v)) for k, v in sorted(arm["env"].items()))
        warm = " ".join(q(x) for x in [arm["binary"]] + base[0] + base[1] + m["warmup_args"])
        meas = " ".join(q(x) for x in [arm["binary"]] + base[0] + base[1] + m["case"]["bench_args"])
        meas += ' --csv "$D/profile.csv"'
        if m["dump_logits"]:
            meas += ' --dump-frontier-logits-dir "$D/logits"'
        L += ["echo \"=== run %d arm %s pair %d $(date -Is) ===\"" % (r["idx"], r["arm"], r["pair"]),
              "D=\"$RUN/%s\"; mkdir -p \"$D/logits\"" % r["dir"],
              "env %s timeout 600 %s > \"$D/warmup.log\" 2>&1" % (env, warm),
              "( while :; do s=$(sclk); t=$(temp); echo \"$(date +%s.%N) ${s:--} ${t:--}\"; sleep 1; done )"
              " > \"$D/clock.tsv\" & CLK=$!",
              "env %s timeout -k 60 %d %s > \"$D/bench.log\" 2>&1; echo $? > \"$D/rc\"" % (env, m["timeout_s"], meas),
              "kill $CLK 2>/dev/null; wait $CLK 2>/dev/null"]
    L += ["python3 -I %s ab analyze \"$RUN\" --write-result" % q(str(AK)), "exit 0", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- analyze

def _read_csv(path):
    if not path.exists():
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _clock_median(path):
    vals = []
    if path.exists():
        for ln in path.read_text().splitlines():
            parts = ln.split()
            # "<ts> <sclk MHz|-> <temp mC|->": exactly three fields, so an empty clock
            # reading can never shift the temperature into the clock column
            if len(parts) == 3 and parts[1].isdigit():
                vals.append(int(parts[1]))
    return statistics.median(vals) if vals else None


def _run_session(run_dir):
    """Session identity from host.txt, written by the payload when the benchmark ran."""
    host = Path(run_dir) / "host.txt"
    if host.exists():
        lines = host.read_text(errors="replace").splitlines()
        ids = [l.strip() for l in lines if _UUID.fullmatch(l.strip())]
        date = lines[0][:10] if lines and re.match(r"\d{4}-\d{2}-\d{2}", lines[0]) else None
        if ids and date:
            return {"session": ids[-1], "date": date, "session_source": "run"}
    return {"session": boot_id(), "date": time.strftime("%Y-%m-%d"), "session_source": "analysis_time"}


def _run_seconds(path):
    """Measured-run duration from the first and last clock sample."""
    if not path.exists():
        return None
    ts = []
    for ln in path.read_text().splitlines():
        try:
            ts.append(float(ln.split()[0]))
        except (ValueError, IndexError):
            pass
    return round(ts[-1] - ts[0]) if len(ts) >= 2 else None


def _bootstrap_ci(xs, n=10000, seed=1):
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _sign_p(pos, n):
    """Two-sided exact sign test p-value."""
    k = max(pos, n - pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _better(metric):
    return -1 if metric.endswith("_ms") else 1


def _noise_path(root):
    return state_dir(root) / "noise.json"


def analyze_dir(run_dir, root=None):
    run_dir = Path(run_dir)
    m = load_json(run_dir / "manifest.json", "manifest")
    root = root or repo_root()
    warnings = []
    data = {}
    for r in m["runs"]:
        rd = run_dir / r["dir"]
        rc_txt = (rd / "rc").read_text().strip() if (rd / "rc").exists() else "missing"
        rows = _read_csv(rd / "profile.csv")
        if rc_txt != "0" or not rows:
            warnings.append("run %s rc=%s rows=%s" % (r["dir"], rc_txt, len(rows or [])))
        data[r["idx"]] = {"rows": rows or [], "clock": _clock_median(rd / "clock.tsv"),
                          "seconds": _run_seconds(rd / "clock.tsv")}
    metrics = {}
    for metric in m["metrics"]:
        sign = _better(metric)
        n_frontiers = max((len(d["rows"]) for d in data.values()), default=0)
        for fi in range(n_frontiers):
            arm_d, order_d, label = [], [], None
            for p in range(m["pairs"]):
                rs = [r for r in m["runs"] if r["pair"] == p]
                vals = {}
                for r in rs:
                    rows = data[r["idx"]]["rows"]
                    if fi < len(rows) and rows[fi].get(metric) not in (None, ""):
                        vals[r["arm"]] = float(rows[fi][metric])
                        vals["pos%d" % r["position"]] = vals[r["arm"]]
                        label = "%s@%s" % (metric, rows[fi].get("ctx_tokens", fi))
                if "A" in vals and "B" in vals and min(vals["A"], vals["B"]) > 0:
                    # log ratios: symmetric, so a pure order effect cannot leak into the arm effect
                    arm_d.append(sign * log(vals["B"] / vals["A"]))
                    order_d.append(sign * log(vals["pos1"] / vals["pos0"]))
            if not label or len(arm_d) < 2:
                continue
            pct = lambda x: round((exp(x) - 1) * 100, 2)  # noqa: E731
            lo, hi = _bootstrap_ci(arm_d)
            pos = sum(1 for x in arm_d if x > 0)
            metrics[label] = {"n": len(arm_d), "mean_pct": pct(statistics.mean(arm_d)),
                              "median_pct": pct(statistics.median(arm_d)),
                              "sd_pct": round(statistics.stdev(arm_d) * 100, 2),
                              "ci95_pct": [pct(lo), pct(hi)], "positive": pos,
                              "sign_p": round(_sign_p(pos, len(arm_d)), 3),
                              "order_effect_pct": pct(statistics.mean(order_d)),
                              "pairs_pct": [pct(x) for x in arm_d]}
    if not metrics:
        raise Refuse("no complete pairs to analyze", fix="inspect bench.log in %s" % run_dir,
                     warnings=warnings[:10])
    # clocks
    clocks = {a: [data[r["idx"]]["clock"] for r in m["runs"] if r["arm"] == a and data[r["idx"]]["clock"]]
              for a in ("A", "B")}
    clk = {a: (statistics.median(v) if v else None) for a, v in clocks.items()}
    clock_skew = None
    if clk["A"] and clk["B"]:
        clock_skew = round((clk["B"] / clk["A"] - 1) * 100, 2)
        if abs(clock_skew) > 3:
            warnings.append("median GPU clock differs %.1f%% between arms" % clock_skew)
    else:
        warnings.append("no clock samples (check sclk_glob in config)")
    # noise floor
    noise_all = load_json(_noise_path(root), "noise") if _noise_path(root).exists() else {}
    floors = noise_all.get(m["case_signature"], {})
    verdicts = {}
    if m["kind"] == "aa":
        for k, s in metrics.items():
            f = max(abs(s["ci95_pct"][0]), abs(s["ci95_pct"][1]), abs(s["mean_pct"]))
            prev = floors.get(k, {"floor_pct": 0, "runs": 0})
            floors[k] = {"floor_pct": round(max(prev["floor_pct"], f), 2), "runs": prev["runs"] + 1,
                         "updated": time.strftime("%Y-%m-%d")}
            verdicts[k] = "noise_floor_recorded"
        noise_all[m["case_signature"]] = floors
        write_json(_noise_path(root), noise_all)
    else:
        for k, s in metrics.items():
            f = floors.get(k)
            if f is None:
                verdicts[k] = "inconclusive_no_noise_floor"
                continue
            lo, hi = s["ci95_pct"]
            if lo > f["floor_pct"] and s["positive"] == s["n"]:
                verdicts[k] = "win"
            elif hi < -f["floor_pct"]:
                verdicts[k] = "loss"
            else:
                verdicts[k] = "inconclusive"
        if clock_skew is not None and abs(clock_skew) > 3:
            verdicts = {k: ("inconclusive_clock_skew" if v in ("win", "loss") else v)
                        for k, v in verdicts.items()}
        if not floors:
            warnings.append("no A/A run for this case: plan one with arms A and B identical")
    # correctness: compare each B run's logits with the first A run
    gate = None
    if m.get("dump_logits"):
        a0 = next(r for r in m["runs"] if r["arm"] == "A")
        results, worst = [], 0.0
        for r in m["runs"]:
            if r["arm"] == "B":
                try:
                    g = compare_dirs(run_dir / a0["dir"] / "logits", run_dir / r["dir"] / "logits")
                    results.append(g["verdict"] == "pass")
                    worst = max(worst, g["worst_max_abs"])
                except Refuse as e:
                    results.append(False)
                    warnings.append("logits gate: " + e.error)
        gate = {"mode": "bit_exact_prefill_frontiers", "pass": all(results) and bool(results),
                "runs_checked": len(results), "worst_max_abs": worst,
                "note": "frontier logits are prefill-end; decode-only paths need a decode gate"}
    arms = {n: {"commit": a["commit"], "anchored": a["anchored"],
                "env": a["env"]} for n, a in m["arms"].items()}
    if not all(a["anchored"] for a in m["arms"].values()):
        warnings.append("an arm is not commit-anchored; ledger will not accept this as promotion evidence")
    sess = _run_session(run_dir)
    if sess["session_source"] != "run":
        warnings.append("host.txt missing or incomplete: session/date taken at analysis time; "
                        "the ledger will not accept this as session evidence")
    return {"ok": True, "run_id": run_dir.name, "kind": m["kind"], "pairs": m["pairs"],
            "order": "".join(r["arm"] for r in m["runs"]), **sess, "arms": arms, "verdicts": verdicts,
            "metrics": metrics, "noise_floor_pct": {k: v["floor_pct"] for k, v in floors.items()},
            "clock_median_mhz": clk, "clock_skew_pct": clock_skew, "gate": gate,
            "run_seconds_median": (statistics.median(secs) if (secs := [d["seconds"] for d in
                                   data.values() if d["seconds"]]) else None),
            "warnings": warnings[:12]}


def _compact(res):
    """result.json for the supervisor handoff must stay under 4 KiB."""
    c = dict(res)
    c["metrics"] = {k: {kk: v[kk] for kk in ("n", "mean_pct", "ci95_pct", "positive", "order_effect_pct")}
                    for k, v in res["metrics"].items()}
    return c


def cmd_analyze(args):
    res = analyze_dir(args.run_dir)
    out = Path(args.run_dir) / "analysis.json"
    write_json(out, res)
    res["analysis_file"] = str(out)
    if args.write_result:
        compact = _compact(res)
        compact["analysis_file"] = str(out)
        write_json(Path(args.run_dir) / "result.json", compact)
        pi = os.environ.get("PI_RUN_DIR")
        if pi:
            write_json(Path(pi) / "result.json", compact)
    return res


def register(sub):
    p = sub.add_parser("ab", help="commit-anchored, counterbalanced A/B benchmarks")
    s = p.add_subparsers(dest="ab_cmd", required=True)
    a = s.add_parser("build", help="build the bench binary for a commit in a git worktree")
    a.add_argument("commit")
    a.add_argument("--build-cmd")
    a.add_argument("--force", action="store_true")
    a.add_argument("--remove-worktree", action="store_true")
    a.add_argument("--timeout", type=int, default=7200)
    a.set_defaults(func=cmd_build)
    a = s.add_parser("plan", help="validate a spec and generate an ABBA payload script")
    a.add_argument("spec")
    a.add_argument("--out", help="write the payload here (e.g. a bench-*.sh the supervisor accepts)")
    a.add_argument("--profile", default=os.environ.get("SUP_PROFILE", "default"),
                   help="supervisor profile whose PAYLOAD_TIMEOUT the plan must fit")
    a.set_defaults(func=cmd_plan)
    a = s.add_parser("analyze", help="paired statistics, noise floor, clocks and logits gate")
    a.add_argument("run_dir")
    a.add_argument("--write-result", action="store_true",
                   help="also write result.json (and $PI_RUN_DIR/result.json)")
    a.set_defaults(func=cmd_analyze)

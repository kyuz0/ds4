"""Synthetic-data tests for agentkit. Run: python3 -m unittest discover -s tools/agentkit/tests"""
import csv
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from agentkit import ab, common, gguf  # noqa: E402

AK = HERE / "ak"


def ak(*args, cwd=None, env=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run([sys.executable, str(AK), *map(str, args)], cwd=cwd, env=e,
                       capture_output=True, text=True)
    out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
    return p.returncode, out, p.stderr


# ---------------------------------------------------------------- synthetic GGUF

def _s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, meta, tensors):
    """tensors: list of (name, dims, type_id); data is zero-filled with 32-byte alignment."""
    out = io.BytesIO()
    out.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for k, v in meta.items():
        out.write(_s(k))
        if isinstance(v, str):
            out.write(struct.pack("<I", 8) + _s(v))
        else:
            out.write(struct.pack("<I", 4) + struct.pack("<I", v))
    offset = 0
    sizes = []
    for name, dims, t in tensors:
        _, be, bb = gguf.GGML_TYPES[t]
        n = 1
        for d in dims:
            n *= d
        size = n // be * bb
        sizes.append(size)
        out.write(_s(name) + struct.pack("<I", len(dims)))
        for d in dims:
            out.write(struct.pack("<Q", d))
        out.write(struct.pack("<IQ", t, offset))
        offset += (size + 31) // 32 * 32
    pad = (-out.tell()) % 32
    out.write(b"\0" * pad + b"\0" * offset)
    Path(path).write_bytes(out.getvalue())
    return sizes


class Units(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(common.parse_bytes("36.5MB"), 36.5e6)
        self.assertEqual(common.parse_bytes("1GiB"), 2 ** 30)
        self.assertAlmostEqual(common.parse_time("175us"), 175e-6)
        self.assertEqual(common.parse_rate("223GB/s"), 223e9)

    def test_unitless_refused(self):
        with self.assertRaises(common.Refuse):
            common.parse_bytes("36.5")
        with self.assertRaises(common.Refuse):
            common.parse_time("175")

    def test_count_expr(self):
        self.assertEqual(common.count_expr("tokens*layers", {"tokens": 16, "layers": 40}), 640)
        with self.assertRaises(common.Refuse):
            common.count_expr("__import__('os')", {})


class Facts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name) / "m.gguf"
        self.sizes = write_gguf(self.model, {
            "general.architecture": "deepseek41", "general.alignment": 32,
            "deepseek41.expert_count": 8, "deepseek41.expert_used_count": 2,
            "deepseek41.block_count": 2},
            [("token_embd.weight", [64, 100], 0),
             ("blk.0.attn_q_b.weight", [64, 32], 8), ("blk.1.attn_q_b.weight", [64, 32], 8),
             ("blk.0.ffn_gate_exps.weight", [256, 16, 8], 16),
             ("blk.1.ffn_gate_exps.weight", [256, 16, 8], 16),
             ("output.weight", [64, 100], 1)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_bytes_exact(self):
        g = gguf.read_gguf(self.model)
        by = {t["name"]: t for t in g["tensors"]}
        self.assertEqual(by["blk.0.attn_q_b.weight"]["bytes"], 64 * 32 // 32 * 34)
        self.assertEqual(by["blk.0.ffn_gate_exps.weight"]["bytes"], 256 * 16 * 8 // 256 * 66)
        self.assertFalse(any(t.get("bytes_mismatch") for t in g["tensors"]))

    def test_decode_bytes(self):
        rc, out, err = ak("facts", "decode-bytes", self.model, "--ceiling", "1GB/s")
        self.assertEqual(rc, 0, err)
        dense = 2 * 2176 + 64 * 100 * 2           # two q_b + f16 output, embd excluded
        experts = 2 * (256 * 16 * 8 // 256 * 66) * 2 / 8
        self.assertAlmostEqual(out["GB_per_token"], round((dense + experts) / 1e9, 3))
        self.assertEqual(out["experts"], {"count": 8, "used": 2})

    def test_group(self):
        rc, out, _ = ak("facts", "tensors", self.model, "--group", "--match", "blk")
        self.assertEqual(rc, 0)
        names = {t["name"]: t for t in out["tensors"]}
        self.assertEqual(names["blk.*.attn_q_b.weight"]["count"], 2)


class Attribution(unittest.TestCase):
    def test_rate_above_ceiling(self):
        # the session's 8x expert error: 60.6 MB in 175 us vs a 223 GB/s ceiling
        rc, out, _ = ak("rate", "--bytes", "60.6MB", "--time", "175us", "--ceiling", "223GB/s")
        self.assertEqual(rc, 0)
        self.assertFalse(out["ok"])
        self.assertEqual(out["verdict"], "impossible_above_ceiling")

    def test_rate_call_mismatch(self):
        rc, out, _ = ak("rate", "--bytes", "1GB", "--time", "10ms", "--ceiling", "223GB/s",
                        "--calls", "32832", "--expect-calls", "tokens*layers",
                        "--var", "tokens=16", "--var", "layers=40")
        self.assertFalse(out["ok"])
        self.assertIn("call count", out["errors"][0])

    def test_budget_host_drain_not_additive(self):
        spec = {"token_ms": 83.0, "tokens": 16, "layers": 40, "ceiling": "223GB/s",
                "entries": [
                    {"label": "leaves", "kind": "gpu_event", "ms": 61.1},
                    {"label": "moe_routed", "kind": "gpu_event", "ms": 11.3, "parent": "leaves"},
                    {"label": "host_drain", "kind": "host_wall", "ms": 11.58, "calls": 640,
                     "expect_calls": "tokens*layers", "regime": "serialized"},
                    {"label": "attn_stage", "kind": "stage_wall", "ms": 30.0}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(spec, fh)
        rc, out, _ = ak("budget", fh.name)
        os.unlink(fh.name)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["gpu_attributed_ms"], 61.1)
        self.assertEqual(out["host_wall_ms_not_additive"], 11.58)
        self.assertTrue(any("serialized" in w or "per-leaf" in w for w in out["warnings"]))

    def test_budget_overcount(self):
        spec = {"token_ms": 50.0, "entries": [
            {"label": "a", "kind": "gpu_event", "ms": 40.0},
            {"label": "b", "kind": "gpu_event", "ms": 20.0}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(spec, fh)
        rc, out, _ = ak("budget", fh.name)
        os.unlink(fh.name)
        self.assertFalse(out["ok"])


def write_logits(d, name, values):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({"argmax_id": max(range(len(values)), key=values.__getitem__),
                                      "logits": values}))


class Gate(unittest.TestCase):
    def test_exact_and_fail(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            v = [0.1 * i for i in range(50)]
            write_logits(t / "a", "frontier_000064.logits.json", v)
            write_logits(t / "b", "frontier_000064.logits.json", v)
            w = list(v)
            w[3] += 1e-6
            write_logits(t / "c", "frontier_000064.logits.json", w)
            rc, out, _ = ak("gate", "logits", t / "a", t / "b")
            self.assertEqual(out["verdict"], "pass")
            self.assertIn("coverage_note", out)
            rc, out, _ = ak("gate", "logits", t / "a", t / "c")
            self.assertEqual(out["verdict"], "fail")
            rc, out, _ = ak("gate", "logits", t / "a", t / "c", "--mode", "argmax_stable")
            self.assertEqual(rc, 3)  # tolerance must be explicit
            rc, out, _ = ak("gate", "logits", t / "a", t / "c", "--mode", "argmax_stable",
                            "--tol", "1e-5")
            self.assertEqual(out["verdict"], "pass")


def make_run(root, kind, pairs, values, clocks=None, commits=("a" * 40, "b" * 40)):
    """values[(pair, arm)] -> (prefill_tps, gen_steady_tps). Writes a manifest + run dirs."""
    run = Path(root) / ("run-" + kind)
    runs = []
    for i, order in ab._order(pairs):
        for pos, arm in enumerate(order):
            runs.append({"idx": len(runs), "pair": i, "position": pos, "arm": arm,
                         "dir": "r%02d_%s" % (len(runs), arm)})
    case = {"model": "/m.gguf", "prompt": "p.txt", "common_args": [], "bench_args": ["x"],
            "global_env": {}}
    arms = {"A": {"commit": commits[0], "binary": "/a", "sha256": "1", "env": {}, "anchored": True},
            "B": {"commit": commits[1], "binary": "/b", "sha256": "1" if kind == "aa" else "2",
                  "env": {}, "anchored": True}}
    m = {"name": "t", "kind": kind, "arms": arms, "case": case,
         "case_signature": ab.case_signature(case), "pairs": pairs, "runs": runs,
         "metrics": ["prefill_tps", "gen_steady_tps"], "dump_logits": True}
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(m))
    for r in runs:
        d = run / r["dir"]
        d.mkdir()
        (d / "rc").write_text("0\n")
        pf, gen = values[(r["pair"], r["arm"])]
        with open(d / "profile.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ctx_tokens", "prefill_tps", "gen_steady_tps"])
            w.writerow([32768, pf, gen])
        clk = (clocks or {}).get(r["arm"], 2900)
        (d / "clock.tsv").write_text("".join("%d %d 50000\n" % (i, clk) for i in range(5)))
        write_logits(d / "logits", "frontier_032768.logits.json", [1.0, 2.0, 3.0])
    return run


class AB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "state"
        os.environ["AK_STATE_DIR"] = str(self.state)

    def tearDown(self):
        os.environ.pop("AK_STATE_DIR", None)
        self.tmp.cleanup()

    def test_abba_order(self):
        self.assertEqual("".join("".join(o) for _, o in ab._order(4)), "ABBAABBA")

    def test_aa_then_ab(self):
        noise = {(p, a): (180 + (0.5 if a == "A" else -0.5) * (1 if p % 2 else -1), 11.0)
                 for p in range(4) for a in "AB"}
        aa = make_run(self.tmp.name, "aa", 4, noise)
        res = ab.analyze_dir(aa, root=self.tmp.name)
        self.assertTrue(all(v == "noise_floor_recorded" for v in res["verdicts"].values()))
        win = {(p, a): (180.0 if a == "A" else 186.0, 11.0) for p in range(4) for a in "AB"}
        r = ab.analyze_dir(make_run(self.tmp.name, "ab", 4, win), root=self.tmp.name)
        self.assertEqual(r["verdicts"]["prefill_tps@32768"], "win")
        self.assertEqual(r["verdicts"]["gen_steady_tps@32768"], "inconclusive")
        self.assertTrue(r["gate"]["pass"])

    def test_no_floor_is_inconclusive(self):
        win = {(p, a): (180.0 if a == "A" else 186.0, 11.0) for p in range(4) for a in "AB"}
        r = ab.analyze_dir(make_run(self.tmp.name, "ab", 4, win), root=self.tmp.name)
        self.assertEqual(r["verdicts"]["prefill_tps@32768"], "inconclusive_no_noise_floor")

    def test_order_effect_separated(self):
        # no arm effect, but whichever runs second is 3% faster
        vals = {}
        for p, order in ab._order(4):
            vals[(p, order[0])] = (100.0, 10.0)
            vals[(p, order[1])] = (103.0, 10.0)
        r = ab.analyze_dir(make_run(self.tmp.name, "ab", 4, vals), root=self.tmp.name)
        m = r["metrics"]["prefill_tps@32768"]
        self.assertAlmostEqual(m["mean_pct"], 0.0, places=6)
        self.assertAlmostEqual(m["order_effect_pct"], 3.0, places=6)

    def test_clock_skew_downgrades(self):
        common.write_json(self.state / "noise.json", {})
        noise = {(p, a): (180.0, 11.0) for p in range(4) for a in "AB"}
        ab.analyze_dir(make_run(self.tmp.name, "aa", 4, noise), root=self.tmp.name)
        win = {(p, a): (180.0 if a == "A" else 190.0, 11.0) for p in range(4) for a in "AB"}
        r = ab.analyze_dir(make_run(self.tmp.name, "ab", 4, win, clocks={"A": 2800, "B": 2950}),
                           root=self.tmp.name)
        self.assertEqual(r["verdicts"]["prefill_tps@32768"], "inconclusive_clock_skew")


class Repo(unittest.TestCase):
    """build / plan / ledger against a throwaway git repo."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        self.env = {"AK_STATE_DIR": str(self.root / ".agentkit"), "GIT_AUTHOR_NAME": "t",
                    "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        g = lambda *a: subprocess.run(["git", *a], cwd=self.root, check=True, capture_output=True,
                                      env=dict(os.environ, **self.env))
        g("init", "-q")
        (self.root / "x.c").write_text("int x;\n")
        g("add", ".")
        g("commit", "-qm", "base")
        (self.root / "x.c").write_text("int x; /* decode blocks4 */\n")
        g("commit", "-qam", "change")
        self.git = g
        (self.root / ".agentkit").mkdir()
        (self.root / ".agentkit" / "config.json").write_text(json.dumps(
            {"build_cmd": "printf '#!/bin/sh\\necho bench\\n' > ds4-bench && chmod +x ds4-bench",
             "busy_url": ""}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_ak(self, *a):
        return ak(*a, cwd=self.root, env=self.env)

    def test_build_plan_payload(self):
        rc, b1, err = self.run_ak("ab", "build", "HEAD~1")
        self.assertEqual(rc, 0, (b1, err))
        rc, b2, _ = self.run_ak("ab", "build", "HEAD")
        rc, again, _ = self.run_ak("ab", "build", "HEAD")
        self.assertTrue(again["cached"])
        spec = {"name": "t1", "model": "/nonexistent.gguf", "prompt": "p.txt",
                "common_args": ["--rocm"], "bench_args": ["--ctx-frontiers", "2048"],
                "arms": {"A": {"commit": "HEAD~1"}, "B": {"commit": "HEAD", "env": {"DS4_X": "1"}}},
                "pairs": 4}
        sp = self.root / "spec.json"
        sp.write_text(json.dumps(spec))
        rc, plan, err = self.run_ak("ab", "plan", sp)
        self.assertEqual(rc, 0, (plan, err))
        self.assertEqual(plan["order"], "ABBAABBA")
        self.assertEqual(plan["kind"], "ab")
        syntax = subprocess.run(["bash", "-n", plan["payload"]], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        text = Path(plan["payload"]).read_text()
        self.assertIn("DS4_X=1", text)
        self.assertIn('--csv "$D/profile.csv"', text)
        spec["pairs"] = 3
        sp.write_text(json.dumps(spec))
        rc, out, _ = self.run_ak("ab", "plan", sp)
        self.assertEqual(rc, 3)

    def test_plan_requires_build(self):
        spec = {"name": "t2", "model": "/m", "prompt": "p", "bench_args": [],
                "arms": {"A": {"commit": "HEAD~1"}, "B": {"commit": "HEAD"}}}
        sp = self.root / "spec.json"
        sp.write_text(json.dumps(spec))
        rc, out, _ = self.run_ak("ab", "plan", sp)
        self.assertEqual(rc, 3)
        self.assertIn("ak ab build", out["fix"])

    def test_gate_required(self):
        rc, out, _ = self.run_ak("gate", "required", "HEAD~1..HEAD")
        gates = {r["gate"] for r in out["required"]}
        self.assertIn("decode_step_logits", gates)
        self.assertIn("prefill_frontier_logits", gates)

    def test_ledger_flow(self):
        rc, out, _ = self.run_ak("ledger", "add", "--kind", "hypothesis",
                                 "--text", "drain costs 11.6 ms per token")
        self.assertEqual(rc, 3)  # number without evidence
        rc, h, _ = self.run_ak("ledger", "add", "--kind", "hypothesis",
                               "--text", "per-layer drain leaves the GPU idle")
        rc, dep, _ = self.run_ak("ledger", "add", "--kind", "hypothesis",
                                 "--text", "skipping drains is the top decode lever",
                                 "--depends", h["id"])
        rc, out, _ = self.run_ak("ledger", "set", h["id"], "--status", "retracted",
                                 "--reason", "drain time overlaps GPU work")
        self.assertEqual(out["dependents_marked_suspect"], [dep["id"]])
        rc, c, _ = self.run_ak("ledger", "add", "--kind", "change", "--text", "skip idle drains",
                               "--change", "DS4_X")
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "measured")
        self.assertEqual(rc, 3)  # cannot skip stages
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True,
                              text=True).stdout.strip()
        base = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=self.root, capture_output=True,
                              text=True).stdout.strip()
        rc, out, err = self.run_ak("ledger", "advance", c["id"], "implemented", "--commit", "HEAD")
        self.assertEqual(rc, 0, (out, err))
        gate_file = self.root / "gate.json"
        gate_file.write_text(json.dumps({"verdict": "pass", "gate": "bit_exact",
                                         "kinds": ["frontier"]}))
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "gated", "--evidence", gate_file,
                                 "--decode-path")
        self.assertEqual(rc, 3)  # decode change, no decode logits
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "gated", "--evidence", gate_file)
        self.assertEqual(rc, 0, out)
        analysis = {"kind": "ab", "session": "s1", "date": "2026-11-04",
                    "arms": {"A": {"commit": base, "anchored": True},
                             "B": {"commit": head, "anchored": True}},
                    "verdicts": {"prefill_tps@32768": "win"}, "gate": {"pass": True}}
        a1 = self.root / "a1.json"
        a1.write_text(json.dumps(analysis))
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "measured", "--evidence", a1)
        self.assertEqual(rc, 0, out)
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "confirmed", "--evidence", a1)
        self.assertEqual(rc, 3)  # same session
        analysis.update(session="s2", date="2026-11-05")
        a2 = self.root / "a2.json"
        a2.write_text(json.dumps(analysis))
        rc, out, _ = self.run_ak("ledger", "advance", c["id"], "confirmed", "--evidence", a2)
        self.assertEqual(rc, 0, out)
        rc, out, _ = self.run_ak("ledger", "check-promote", c["id"])
        self.assertTrue(out["may_promote"], out)
        rc, out, _ = self.run_ak("ledger", "truth")
        text = Path(out["written"]).read_text()
        self.assertIn("Suspect", text)
        self.assertIn(dep["id"], text)


if __name__ == "__main__":
    unittest.main()

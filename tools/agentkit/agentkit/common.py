"""Shared helpers: strict unit parsing, JSON output, refusals, repo paths."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


class Refuse(Exception):
    """An input the tool will not compute on. Carries a concrete fix."""

    def __init__(self, error, fix=None, **extra):
        super().__init__(error)
        self.error = error
        self.fix = fix
        self.extra = extra


def emit(obj, stream=None):
    stream = stream or sys.stdout
    stream.write(json.dumps(obj, separators=(",", ":"), sort_keys=False) + "\n")


def refuse_payload(exc):
    out = {"ok": False, "error": exc.error}
    if exc.fix:
        out["fix"] = exc.fix
    out.update(exc.extra)
    return out


# ---------------------------------------------------------------- units

_BYTE_UNITS = {
    "B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
    "KIB": 1024, "MIB": 1024 ** 2, "GIB": 1024 ** 3, "TIB": 1024 ** 4,
}
_TIME_UNITS = {"NS": 1e-9, "US": 1e-6, "MS": 1e-3, "S": 1.0}
_NUM = r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def _split(text, what):
    m = re.fullmatch(r"\s*" + _NUM + r"\s*([A-Za-z/µμ]+)\s*", str(text))
    if not m:
        raise Refuse("%s %r has no explicit unit" % (what, text),
                     fix="write the unit, e.g. 36.5MB, 3.2GiB, 175us, 8.7ms, 223GB/s")
    return float(m.group(1)), m.group(2)


def parse_bytes(text):
    value, unit = _split(text, "byte quantity")
    u = unit.upper()
    if u not in _BYTE_UNITS:
        raise Refuse("unknown byte unit %r" % unit,
                     fix="use B, KB, MB, GB (decimal) or KiB, MiB, GiB (binary)")
    return value * _BYTE_UNITS[u]


def parse_time(text):
    value, unit = _split(text, "time")
    u = unit.replace("µ", "u").replace("μ", "u").upper()  # micro sign and Greek mu
    if u not in _TIME_UNITS:
        raise Refuse("unknown time unit %r" % unit, fix="use ns, us, ms or s")
    return value * _TIME_UNITS[u]


def parse_rate(text):
    value, unit = _split(text, "rate")
    if not unit.upper().endswith("/S"):
        raise Refuse("rate %r is not per second" % text, fix="e.g. 223GB/s")
    return parse_bytes("%r%s" % (value, unit[:-2]))


def count_expr(expr, names):
    """Evaluate a small integer expression such as 'tokens*layers' or '16*40'."""
    text = str(expr)
    if not re.fullmatch(r"[0-9a-zA-Z_+*() ]+", text):
        raise Refuse("call-count expression %r has unsupported characters" % text,
                     fix="use integers, names, + * and parentheses")
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text):
        if name not in names:
            raise Refuse("call-count expression uses unknown name %r" % name,
                         fix="known names: %s" % ", ".join(sorted(names)))
    try:
        return int(eval(text, {"__builtins__": {}}, dict(names)))  # noqa: S307 (validated)
    except (SyntaxError, TypeError, ValueError) as e:
        raise Refuse("call-count expression %r is not valid (%s)" % (text, e.__class__.__name__),
                     fix="e.g. 'tokens*layers' or '16*40'")


# ---------------------------------------------------------------- paths

def repo_root(start=None):
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start or os.getcwd(),
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Refuse("not inside a git repository", fix="run ak from the ds4 checkout")


def state_dir(root=None):
    """Machine-local state (built binaries, noise floors, runs). Not committed."""
    d = Path(os.environ.get("AK_STATE_DIR") or (Path(root or repo_root()) / ".agentkit"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def git(root, *args):
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refuse("git %s failed: %s" % (" ".join(args), out.stderr.strip()[:300]))
    return out.stdout.strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path, what="file"):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise Refuse("%s not found: %s" % (what, path))
    except json.JSONDecodeError as e:
        raise Refuse("%s is not valid JSON: %s (%s)" % (what, path, e))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown"

# DS4 optimization — Strix Halo / V4.1 Flash

Improve measured prefill/decode on `gfx1151`. Preserve numerical correctness,
SSD streaming, tool calling and unrelated backends. Change one measured bottleneck
at a time. Keep `balanced` power settings and preserve unrelated dirty files.
Never restore whole files containing other work or delete models/caches.

## Continue without re-discovery

The latest session handoff and subsequent results carry current work. Continue
its next pending action; don't repeat established checks or reread unchanged
files. Newer recorded results supersede older summaries. Read an artifact only
to resolve a specific missing fact. Mark checks as passed, failed or pending.

STATE.md is a retired bootstrap snapshot, not current state. Older instructions
to read/update it or maintain EXPERIMENTS.md are superseded. No separate state
ledger is required. NOTE.md, RESULTS.md and EXPERIMENTS.md are historical evidence;
search a relevant section only when needed. AGENTS.md contains durable rules;
do not copy these rules into every handoff.

## Tools

The current DeepSeek response limit is **8,192 generated tokens total**, shared
by reasoning, visible text and all tool-call arguments in that response. The
131,072-token context window is separate; free context does not increase the
response limit. File bytes and line counts are not token budgets. Leave room for
complete tool arguments instead of spending the response on lengthy planning.
For a large new file, write a small initial portion, then extend it through
separate complete edit calls; do not execute it until complete and validated.
If output-limit exhaustion rejects a call, it was not executed: issue a smaller
complete call with brief reasoning. Never continue truncated JSON or assume a
partial write succeeded. Prefer targeted edits for existing files.

Prefer native read/grep/edit for source work. Scope searches by path/glob, use
short unique edit anchors, and inspect edit feedback before another read.
Use bash for builds, processes and structured-data analysis. Keep each output
focused on one question: don't combine a source dump with a large process listing
or log dump. Parse traces/logs locally and return relevant aggregates or errors.
A truncated result is incomplete; retrieve only the missing range, not the whole
request again. Use native offset/limit for source reads. Batch disjoint edits
in one file when appropriate. Resolve uncertain facts before extended reasoning.

## Benchmarks

Use `run_benchmark` to launch a prepared payload through this checkout's supervisor:

    {"payload":"bench-append-4k-long-ab.sh","args":["1"]}

The tool captures the current Pi PID/session/pane, validates the payload, and
launches detached. Do not construct shell launch commands, search for PIDs, or
re-derive the supervisor lifecycle. After launch issue no further tools; the
supervisor stops Pi/backend, runs the payload and resumes this same session.
If launch fails, resolve that concrete error; do not bypass it with manual launch.
Inspect supervisor internals only to diagnose or change specific behavior.

Review the run-specific summary.json and necessary result fields after resume.
Exit code zero alone proves neither correctness nor a speedup. No separate
state-file update is required before continuing. Never overlap supervisors or
model processes, or manually restart the backend during a cycle.

Current long-context API work uses 92GB/128K; historical baseline used 76GB/32K.
Keep model, prompt, cache, context, generation and build fixed within paired A/B.
The GB suffix matters. Separate prefill/decode and GPU/I/O/profile timings.
Run focused correctness checks, preserve test coverage, and confirm executable
build success. Keep candidates opt-in until numerics and repeated balanced
measurements support promotion. Reject unexplained drift and unverified wins.

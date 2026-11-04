# Agent workflow for the V4.1 Strix Halo optimization

This document explains what was added to this branch to help a smaller model drive the
V4.1 / Strix Halo optimization work without repeating the mistakes of the earlier
sessions. It covers:

- why the change was needed;
- the pieces: `ak`, the supervisor and AGENTS.md;
- how they fit together and how to start using them;
- what is still open.

| Commit | What |
|---|---|
| `d1c9fe1` | `tools/agentkit/`: the `ak` command-line tools |
| `156cfd5` | `supervisor.sh` + `supervisor.d/`: profile-driven supervisor |
| `31408f6` | agentkit fixes from a code review |
| `1090bb3` | AGENTS.md rewritten around the tools |

Detailed references: `tools/agentkit/README.md`, `supervisor.d/README.md`, `AGENTS.md`.

---

## 1. Why: what the review of the previous work found

The branch's optimization work (about 105 commits since 2026-10-31, summarised in
`findings/v41-session-summary.md`) was reviewed statically.

**What was good**
- Strong correctness discipline: bit-identical frontier-logit gates, exhaustive proofs (the 128-of-128 IQ2 sign-table check), and an FP4 pack verifier that caught real bugs.
- Small, opt-out-able promotions. The verified gains were about +3.8 % prefill, +4 % append and +5–7 % decode at 32K, with every arm bit-identical.
- A complete map of decode weight streaming: every weight-streaming kernel runs at 87–102 % of the measured 223 GB/s read ceiling.
- Dead ends closed on evidence: the FP4 index cache, stream rescheduling, and decode-cache growth (whose gain turned out to be clock-boost drift).

**What went wrong.** These are the patterns the new tooling is built to prevent:

1. **Double counting.** The final "host drain = 11.58 ms/token (14 %)" figure is host time spent in `cudaDeviceSynchronize` while the GPU finishes the layer's routed-MoE work. It matches the MoE GPU time already counted (284 vs 290 µs per layer), and it was measured with a profiler that synchronizes after every leaf. The real target is about 22 ms/token of GPU idle time, which the earlier rocprof gap measurement (21.4 ms) already showed.
2. **Wrong populations and units.** The findings file contains at least six retractions: the 7168×2048 V4 Pro shape used for V4.1 Flash, 8 experts instead of 6, prefill rows counted in a decode counter, MB/µs vs GB/s, and rates above the bandwidth ceiling.
3. **A/B design.** The baseline always ran first in each pair (no counterbalancing) and there was never an A/A test. Two "+2.5 %" wins did not reproduce.
4. **Gate coverage.** Decode-path promotions (the q_a+kv fusion and the blocks4 unroll) were gated with prefill-end frontier logits, which never run decode kernels.
5. **Cross-backend leak.** The IQ2 sign-table change in `cuda/mmq/mmq.cuh` is unguarded, so it is compiled into the NVIDIA build, where it was never measured.
6. **Reproducibility.**
   - 76 of 77 payloads hardcode `/home/dani/code/v4.1-ds4`.
   - The baseline binary in `/tmp` matches no commit.
   - Raw evidence (`findings/`, `*.log`, NOTE.md, RESULTS.md) is gitignored.
   - `41bb7cf` is a 2,933-line commit with an empty message, containing default-on changes never measured against upstream.
7. **Findings as a log.** A 1,351-line append-only file whose top table disagrees with its later sections.

**Most promising next optimizations, ranked**
1. **GPU idle time, about 22 ms/token.** Every layer has two host syncs: a blocking read of `selected` and the end-of-layer drain. Steps, cheapest first:
   - skip the drain for layers that streamed nothing;
   - pinned readback with spin-waiting;
   - `HSA_ENABLE_SDMA=0` for the tiny copies;
   - an expert-residency table on the GPU, so the host only syncs on a cache miss;
   - kernel fusion to cut the 2,408 dispatches/token.
2. **DSpark speculative decoding together with SSD streaming.** Upstream antirez/ds4 documents this for V4.1 on ROCm; this fork's docs say it is unsupported.
3. **Fewer bytes per token.** Q8_0 attention projections are 59 % of decode traffic. This changes numerics, so it must be opt-in with a quality evaluation.
4. **Prefill.** Recheck the indexer GEMM throughput and tune hipBLASLt; batch the 1.27 M dispatches of a 32K prefill.

Ceiling for reference: 9.07 GB/token at 223 GB/s is 40.7 ms, i.e. about 24.6 tok/s, against about 12 tok/s today.

---

## 2. The pieces

### 2.1 `ak`: tools that compute instead of the model (`tools/agentkit/`)

Python standard library only, plus git. Every command prints one JSON object. Exit codes:
0 computed, 3 refused (with `error` and `fix`), 2 usage error, 4 internal bug.
**Refusals are the point:** each one turns a mistake from section 1 into an error message.

| Command | Purpose | Prevents |
|---|---|---|
| `ak facts summary\|tensors\|decode-bytes` | Shapes, types and exact bytes read from the GGUF; weight bytes per decode token, by role, with a ceiling | Wrong shapes, wrong expert counts, guessed byte figures |
| `ak rate` | Bytes / time against the measured ceiling, with mandatory units and call-count reconciliation | Unit errors, impossible rates, wrong populations |
| `ak budget` | Per-token time budget with timing kinds (`gpu_event`, `gpu_trace`, `host_wall`, `stage_wall`) and parent/child nesting | Double counting host wait and nested stages |
| `ak ab build\|plan\|analyze` | Binaries built per commit in git worktrees; ABBA payloads with warm-up, clock and temperature samples and host details; log-ratio statistics with CI, sign test, order effect, A/A noise floor, clock-skew check and logits gate | Fixed arm order, baselines matching no commit, false "wins" |
| `ak gate logits\|required` | Bit-exact or explicit-tolerance logit comparison; which gates a diff needs (decode, CUDA, streaming, promotion…) | Wrong gate for the path changed, unnoticed cross-backend edits |
| `ak ledger add\|set\|advance\|check-promote\|show\|truth` | Append-only claims with evidence; retraction cascade; staged promotion; generates `ledger/CURRENT.md` | Numbers without evidence, unreproduced promotions, stale conclusions |

Safeguards worth knowing:
- **"Win" verdict:** a result is a `win` only if the 95 % CI is above the A/A noise floor **and** every pair is positive. Without an A/A run for that exact case, the verdict is always `inconclusive_no_noise_floor`.
- **Noise floor key:** the floor is tied to the case: arguments, env, the prompt's sha256, and the model's size and mtime.
- **Promotion order:** promotion requires `implemented → gated → measured → confirmed`. The `confirmed` stage must come from another session (a different boot id or day), as recorded by the payload **when the benchmark ran**. Re-analysing an old run does not count.
- **Evidence rules:**
  - an A/B with no logits gate is not evidence;
  - a retracted, rejected or suspect change cannot advance;
  - the candidate commit must contain the implementation commit.
- **Decode path:** `--decode-path` on `gated` requires the gate to have compared decode logits.

Tests: `python3 -m unittest discover -s tools/agentkit/tests` (27 tests, synthetic data).

### 2.2 Supervisor: configurable, pluggable and exact on restart (`supervisor.sh`, `supervisor.d/`)

The supervisor still runs the same 7 steps: stop Pi, stop the backend, wait for memory, run the
payload, restart the backend, resume Pi. Now:
- **Settings live in profiles** (`supervisor.d/<name>.env`, sourced by bash). `default.env` reproduces the current Cockpit/podman setup.
- **Per-launch overrides:** `--profile`, `--set KEY=VALUE`, `--env NAME=VALUE` (payload), `--backend-env NAME=VALUE` (server) and `--dry-run` (preflight and the plan, touching nothing).
- **Backends:** `podman`, `command` (host `ds4-server`), `systemd`, `none`.
- **Exact restart.** Before stopping the container, the supervisor captures how it was created (`podman inspect` CreateCommand and image) and replays that on restart. The profile's run arguments are only a fallback. This fixed a real bug: preflight used to check a different image than the one recovery started.
- **Run directory.** Each run directory holds the effective settings (`supervisor.env`), the payload log, `summary.json`, `result.json` and the podman capture. `PAYLOAD_TIMEOUT` is exported to the payload.
- **Compatibility:** the old positional interface is unchanged, so the existing `run_benchmark` tool keeps working.

It was tested locally with a stub `podman`, the `none` and `command` backends, a payload timeout and bad inputs. It has **not** been run against real podman, systemd or the GPU.

### 2.3 AGENTS.md: rules the model reads every session

Rewritten around the tools:
- **State:** current state is `ledger/CURRENT.md`; the old findings files are history.
- **Tools:** a table of which `ak` command to use for each need, with the instruction to follow a refusal's `fix`.
- **Lessons:** five rules from the session's mistakes: units, population-scoped counters, host wait vs GPU time, nested stages, the ceiling check.
- **Changes:** promotion through the ledger, decode-gate coverage, guarding ROCm-only edits in shared files, hypotheses before code.
- **Benchmarks:** `ak ab plan --out bench-ak-NAME.sh` then `run_benchmark`; only an `ak` "win" is a speedup; an A/A run for every new case.
- **Response budget:** the 8K response-budget guidance, condensed.

---

## 3. How it fits together

```
 hypothesis ──► ak ledger add --kind hypothesis           (mechanism, expected size, no numbers)
     │
 implement (opt-in env flag), commit
     │──────► ak ledger advance C… implemented --commit SHA
     │
 ak gate required HEAD~1..HEAD  ──► run the listed gates ──► ak gate logits A B --save gate.json
     │──────► ak ledger advance C… gated --evidence gate.json [--decode-path]
     │
 ak ab build BASE ; ak ab build SHA
 ak ab plan spec.json --out bench-ak-NAME.sh      (checks PAYLOAD_TIMEOUT from supervisor.d)
     │
 run_benchmark {"payload":"bench-ak-NAME.sh"} ──► supervisor.sh (profile) ──► payload
     │                                              └─ ak ab analyze --write-result
     │                                                 └─ $PI_RUN_DIR/result.json → summary.json
 Pi resumes, reads verdicts/gate/warnings
     │──────► ak ledger advance C… measured --evidence .agentkit/runs/…/analysis.json
     │        (later, another session) … confirmed
     │
 ak ledger check-promote C… ──► flip the default ──► ak ledger advance C… promoted --commit SHA
 ak ledger truth  ──► ledger/CURRENT.md
```

Attribution work (deciding *what* to optimize) uses `ak facts`, `ak rate` and `ak budget`
before anything is written into a claim.

---

## 4. Getting started on the Strix Halo machine

1. **Sync the branch.** The bench payloads still `cd` into `/home/dani/code/v4.1-ds4`. Either check out this branch there, or update old payloads before reusing them. New `ak` payloads use the checkout they were planned in.
2. **Check the supervisor without interrupting anything:**
   ```sh
   ./supervisor.sh --dry-run "$PI_PID" "$SESSION" "$PANE" bench.sh
   ```
   Then read the printed plan and `findings/supervisor-*/supervisor.env`.
3. **Build the baseline once:**
   ```sh
   ak ab build 41bb7cf   # or upstream 815e8fb for a clean pre-session reference
   ak ab build HEAD
   ```
4. **Run an A/A for the main case** (both arms the same commit, 4 pairs, `est_run_s` set) to record the noise floor. Until then, every A/B is inconclusive by design.
5. **Seed the ledger** with the current state, so `ledger/CURRENT.md` exists:
   - the promoted set (fused-epilogue MMQ tile, q_a+kv fusion, blocks4 unroll, IQ2 sign table);
   - the closed lines;
   - the retracted host-drain claim.

## 5. Hooking into Pi (planned)

- **Tools:** expose `ak` subcommands as Pi tools with typed schemas (`pi.registerTool`). Outputs are already compact JSON.
- **Supervisor tool:** passes `--profile` / `--set` / `--env` and launches detached; `--dry-run` first when settings change.
- **Guards:** `pi.on("tool_call")` hooks that block:
  - direct `ds4-bench` / `supervisor.sh` runs (A/B must go through `ak ab plan`);
  - a default flip without `ak ledger check-promote`;
  - rates or ms figures written into findings that didn't come from `ak`.

## 6. Open items and limitations

- **No decode gate yet.** `ds4-bench` only dumps prefill-frontier logits. A per-step decode dump (e.g. `decode_<frontier>_<step>.logits.json`, same format) is needed before decode-path changes can be promoted. `ak gate` already compares any `*.logits.json`.
- **Payload location is assumed.** `run_benchmark` was assumed to accept repo-root `bench-ak-*.sh` payloads; that tool is not in this checkout.
- **Untested on real hardware.** Everything was tested on synthetic data (a generated GGUF, fake runs, a throwaway git repo, a stub podman), never against the real model, GPU or supervisor cycle.
- **Builds are slow but cached.** The default `ak ab build` command (`make strix-halo`) is a full `-B` rebuild, done once per commit.
- **`gate required` is a heuristic** (regexes in `tools/agentkit/gates.json`). Add a rule whenever a gate was missed.
- **Problems from section 1 not fixed by this work:**
  - the unguarded CUDA change in `cuda/mmq/mmq.cuh`;
  - the hardcoded paths in old payloads;
  - the unmeasured default-on changes in `41bb7cf`.

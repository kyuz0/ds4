# agentkit (`ak`)

Command-line tools that do the error-prone parts of performance work, so an agent
(or a person) never computes them by hand. The tools read model shapes from the GGUF,
check rates and time budgets, run counterbalanced A/B benchmarks, gate correctness, and
keep claims in a ledger.

Each tool either computes a verdict or **refuses with a concrete fix**. The refusals are
the point: they turn the mistakes from the V4.1 session into errors instead of findings.

| Session mistake | What now stops it |
|---|---|
| 7168×2048 "Pro" shape used for Flash; 8 experts instead of 6 | `ak facts` reads dims, types and bytes from the GGUF |
| MB/µs vs GB/s; rates above the 223 GB/s ceiling | `ak rate` requires units and rejects rates above the ceiling |
| Prefill rows counted in a decode leaf (32 832 calls) | `--calls` / `--expect-calls tokens*layers` reconciliation |
| Nested stages added twice; host drain added to GPU time | `ak budget`: parents/children and timing kinds |
| Baseline always run first; +2.5 % wins that didn't reproduce | `ak ab`: ABBA order, A/A noise floor, clock check |
| Baseline binary in `/tmp` matching no commit | `ak ab build`: binaries built from commits in git worktrees |
| Prefill-frontier logits used to gate decode-only changes | `ak gate` coverage note; `ak ledger advance --decode-path` |
| 1 351-line findings log full of retractions | `ak ledger`: claims with evidence; retractions mark dependents suspect |

Requirements: Python ≥ 3.9, stdlib only, `git`. Run from anywhere inside the ds4 checkout.

```sh
alias ak=tools/agentkit/ak
python3 -m unittest discover -s tools/agentkit/tests    # synthetic-data tests
```

## Conventions

- **Output:** every command prints one compact JSON object on stdout.
- **Exit codes:**
  - `0` means the command computed a result. Read `ok`, `verdict` / `verdicts`, `errors` and `warnings`. `ok:false` with exit 0 means the inputs were valid but the result failed a check, such as a rate above the ceiling.
  - `3` means refused. Read `error` and `fix`.
  - `2` means a usage error.
- **Units:** every quantity must carry a unit. Bytes are `B KB MB GB` (decimal) or `KiB MiB GiB`; times are `ns us ms s`; rates are `GB/s`. A bare number is refused.
- **Machine-local state** lives in `.agentkit/` (gitignored): built binaries, worktrees, runs, noise floors and an optional `config.json` that overrides `tools/agentkit/config.json`. Override the location with `AK_STATE_DIR`.
- **Committed state** is `ledger/claims.jsonl` (append-only) and the generated `ledger/CURRENT.md`.

## `ak facts`: model facts from the GGUF

```sh
ak facts summary  MODEL.gguf                       # arch hparams, bytes by type
ak facts tensors  MODEL.gguf --group --match attn  # blk.*.X collapsed: type, dims, MB each/total
ak facts decode-bytes MODEL.gguf --ceiling 223GB/s [--exclude 'engram|mtp']
```

- **Dims order:** `dims_ne0_first` is GGUF order, with ne0 (the row length, i.e. the input dimension) first.
- **How bytes are computed:** from the ggml type table, then cross-checked against the spacing between tensor offsets. Mismatches are reported. Unknown types fall back to the offset span.
- **`decode-bytes`:** weight bytes streamed per decode token, by role. Every non-expert tensor is counted as read once per token. Routed experts (tensors matching `_exps` with a dim equal to `expert_count`) are counted at `used/count` of their bytes. `token_embd` is excluded and `output` is included. The assumptions are printed with the result. `--ceiling` adds the minimum ms/token and maximum tok/s.

## `ak rate`: bytes / time against the measured ceiling

```sh
ak rate --bytes 36.5MB --time 175us --ceiling 223GB/s \
        --calls 640 --expect-calls 'tokens*layers' --var tokens=16 --var layers=40
```

Verdict is one of `at_ceiling` (≥85 %), `below_ceiling`, or `impossible_above_ceiling` (more than 5 % over the ceiling, which gives `ok:false`). A call count that differs from the expected population is an error: the counter is measuring a different set of calls than you think.

## `ak budget`: per-token time budget without double counting

```json
{
  "token_ms": 83.0, "tokens": 16, "layers": 40, "ceiling": "223GB/s", "scope": "decode",
  "entries": [
    {"label": "attn_output_pair", "kind": "gpu_event", "ms": 15.49, "calls": 640,
     "expect_calls": "tokens*layers", "bytes": "3208.6MB"},
    {"label": "moe_routed", "kind": "gpu_event", "ms": 11.3},
    {"label": "gate_up", "kind": "gpu_event", "ms": 7.0, "parent": "moe_routed"},
    {"label": "host_drain", "kind": "host_wall", "ms": 11.58, "regime": "serialized"}
  ]
}
```

`ak budget budget.json` applies these rules:
- **Timing kinds:**
  - `gpu_event`: events around launches.
  - `gpu_trace`: rocprof busy time.
  - `host_wall`: host blocking time. It overlaps GPU work, so it is reported separately and **never added**.
  - `stage_wall`: synchronised stage marks. These are inflated, so they are excluded from sums.
- **Nesting:** entries with a `parent` are not added. A sum of children larger than the parent is an error.
- **Call counts:** mismatched `calls` are an error. A missing call count is a warning.
- **Rates:** a rate above the ceiling is an error.
- **Totals:** attributed GPU time above `token_ms` is an error (something is double counted).
- **Regime:** `"regime":"serialized"` on host time triggers a warning that it doesn't represent the natural run.

The real GPU idle time is `token_ms` minus GPU busy time, measured in the natural run, not the sum of host waits.

## `ak ab`: commit-anchored, counterbalanced A/B

1. **Build each arm from a commit** (once per commit; cached by SHA with its sha256):
   ```sh
   ak ab build 41bb7cf
   ak ab build HEAD
   ```
   This runs `build_cmd` (default `make -j$(nproc) strix-halo`) in `.agentkit/worktrees/<sha>` and copies `ds4-bench` to `.agentkit/bins/<sha>/`. Your working tree is never touched.

2. **Write a spec:**
   ```json
   {
     "name": "qakv-fuse",
     "model": "/home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf",
     "prompt": "speed-bench/promessi_sposi.txt",
     "common_args": ["--rocm", "--ssd-streaming", "--ssd-streaming-cache-experts", "92GB"],
     "bench_args": ["--ctx-frontiers", "32768,36864", "--ctx-alloc", "40960", "--gen-tokens", "64"],
     "env": {"DS4_BENCH_SNAPSHOT_MAX_BYTES": "2147483648"},
     "arms": {
       "A": {"commit": "41bb7cf"},
       "B": {"commit": "HEAD", "env": {"DS4_ROCM_V41_Q8_FUSE_QAKV": "1"}}
     },
     "pairs": 4
   }
   ```
   - `common_args` go to both the warm-up run and the measured run; `bench_args` go to the measured run only.
   - `-m`, `--prompt-file`, `--csv` and the logits directory are set by `ak` and are refused in the spec.
   - `pairs` must be even and at least 4.
   - A spec whose two arms have the same binary and env is an **A/A run**.

3. **Plan:**
   ```sh
   ak ab plan spec.json [--profile default] [--out bench-ak-qakv.sh]
   ```
   The plan must fit the supervisor's `PAYLOAD_TIMEOUT` from `supervisor.d/<profile>.env`. Add `"est_run_s"`
   (seconds per measured run; take it from `run_seconds_median` of an earlier analysis) and a plan that
   needs more than 90 % of the timeout is refused, with a smaller pair count as the fix. See
   `supervisor.d/README.md`.

   This generates a payload that runs in ABBA order (`ABBAABBA`). It unsets all `DS4_*` variables and checks that the backend is down. Each run gets a fixed warm-up and clock/temperature samples taken every second from sysfs, and the host details (kernel, cmdline, platform profile, boot id) are recorded. At the end the payload runs `ak ab analyze --write-result`, which also writes `$PI_RUN_DIR/result.json` (under 4 KiB) when launched by the supervisor. Launch the payload through `run_benchmark` / the supervisor as usual. Use `--out` if the supervisor only accepts payloads at a specific path.

4. **Analyze** (automatic at the end of the payload; can be re-run):
   ```sh
   ak ab analyze .agentkit/runs/<name>-<stamp>
   ```
   - **Per metric and frontier** (`prefill_tps@32768`, `gen_steady_tps@36864`, …) it reports per-pair %, mean, median, sd, bootstrap 95 % CI, sign test, and `order_effect_pct`, the bias toward whichever arm runs second.
   - **Log ratios:** all statistics use log ratios, so an order effect cannot leak into the arm effect.
   - **Verdicts:**
     - `win` needs CI low > noise floor and every pair positive.
     - `loss` needs CI high < −floor.
     - Otherwise the verdict is `inconclusive`.
     - Without an A/A run for the same case, the verdict is always `inconclusive_no_noise_floor`.
     - A clock skew above 3 % between arms downgrades win or loss to `inconclusive_clock_skew`.
   - **Noise floor:** an A/A run records it per case signature in `.agentkit/noise.json` (the most conservative value seen).
   - **Logits gate:** frontier logits from every B run are compared bit-exactly with the first A run.

   **Run an A/A first for every new case.** That is the only way to know what +2 % means on this machine.

## `ak gate`: correctness gates

```sh
ak gate logits DIR_A DIR_B [--save gate.json]                     # bit_exact (default)
ak gate logits DIR_A DIR_B --mode argmax_stable --tol 1e-3        # numerics-changing candidates
ak gate required main..HEAD                                        # which gates a diff needs
```

- **`logits`:** compares every `*.logits.json` pair: exact equality, `max_abs`, argmax, and top-k order and set. `argmax_stable` requires an explicit `--tol`. When only prefill frontiers were compared, the result carries a `coverage_note` saying decode-only paths are **not** verified.
- **`required`:** maps a git diff to gates using `tools/agentkit/gates.json`. The rules are regexes on file paths and changed lines. Examples:
  - decode / `_rows` / `blocks4` code requires `decode_step_logits`;
  - `cuda/` requires `cuda_backend_build_and_bench`;
  - a flipped `getenv(...) == '0'` default requires `promotion_requires_ledger`;
  - streaming code requires `ssd_streaming_check`.

  The rules are heuristic: add one whenever a gate was missed.

**Gap in ds4 itself:** `ds4-bench` only dumps prefill-frontier logits. A decode gate needs per-step decode logits, for example `decode_<frontier>_<step>.logits.json` in the same JSON format. `ak gate logits` compares any `*.logits.json`, so adding that dump to `ds4_bench.c` is all that's needed.

## `ak ledger`: claims, evidence, promotion

```sh
ak ledger add --kind hypothesis --text "per-layer drain leaves the GPU idle"
ak ledger add --kind measurement --text "..." --evidence .agentkit/runs/x/analysis.json
ak ledger add --kind change --text "skip drains for layers that streamed nothing" --change DS4_X
ak ledger set H003 --status retracted --reason "drain overlaps GPU work"   # dependents become suspect
ak ledger advance C001 implemented --commit <sha>
ak ledger advance C001 gated --evidence gate.json [--decode-path]
ak ledger advance C001 measured  --evidence .agentkit/runs/<run1>/analysis.json
ak ledger advance C001 confirmed --evidence .agentkit/runs/<run2>/analysis.json
ak ledger check-promote C001
ak ledger advance C001 promoted --commit <sha flipping the default>
ak ledger show [ID] [--status suspect] ; ak ledger truth   # writes ledger/CURRENT.md
```

**What it refuses:**
- **Claims without evidence:** text containing a quantity (`%`, ms, GB/s, tps…) without an evidence file, and `measurement` claims without evidence.
- **Out-of-order stages:** stages go `proposed → implemented → gated → measured → confirmed → promoted`, with no skipping.
- **Weak gates:**
  - `gated` needs a gate file with verdict `pass`;
  - with `--decode-path`, the gate must have compared decode logits.
- **Weak A/B evidence:** `measured` and `confirmed` need an `ak ab` analysis that:
  - has kind `ab` and commit-anchored arms;
  - shows at least one `win`;
  - passed its logits gate;
  - has a candidate arm that contains the implementation commit.
- **Same-session confirmation:** `confirmed` must come from a different session, i.e. a different boot id or a different day.

**Retraction cascade:** retracting or rejecting a claim marks everything that depends on it (`--depends`) as `suspect`. `ledger/CURRENT.md` is the short current-truth page; the old findings files become history.

## Suggested agent loop

1. `ak facts …` for any shape or byte figure. Never type a dimension.
2. `ak ledger add --kind hypothesis` with the mechanism and expected size, no numbers.
3. Implement opt-in, commit, then `ak ledger advance … implemented`.
4. `ak gate required HEAD~1..HEAD`, run the listed gates, then `advance … gated`.
5. `ak ab build` both commits, then `ak ab plan` and launch. Run an A/A first if the case is new.
6. `advance … measured`. Later, in another session, `advance … confirmed`.
7. Promote only when `ak ledger check-promote` says `may_promote:true`.
8. Use `ak rate` / `ak budget` for every attribution before writing it anywhere.

## Hooking into Pi later

- **Tools:** wrap each subcommand as a Pi tool with a typed schema and pass the JSON through. Outputs are already compact, which suits the 8K response limit.
- **Guards:** add `tool_call` hooks:
  - block direct `ds4-bench` / `supervisor.sh` invocations, so A/Bs go through `ak ab plan`;
  - block `git commit` of a default flip unless `ak ledger check-promote` passes;
  - block writes to findings files containing rates or ms figures that didn't come from `ak`.

## Limitations

- **Build:** the default `build_cmd` does a full `-B` rebuild. It is done once per commit and cached. Override it in `.agentkit/config.json`.
- **Clocks:** samples come from the first `pp_dpm_sclk` entry marked `*`. Adjust `sclk_glob` if the GPU is on a different card.
- **Gates:** `gate required` rules are pattern heuristics, not program analysis.
- **Tested:** with synthetic data only (a generated GGUF, fake runs, a throwaway git repo). Not yet run against the real model or supervisor.

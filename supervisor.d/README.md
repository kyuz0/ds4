# Supervisor and profiles

`supervisor.sh` (repo root) runs a benchmark or other payload with the whole machine to itself, then
puts everything back. Every machine-specific setting lives in a **profile** in this
directory. To change the setup, edit or add a profile; the script itself never needs
editing.

## What one cycle does

1. **Preflight.** Checks the profile, Pi, the tmux pane, the payload, the model, the GPU devices and the backend. If anything fails, it exits **without touching anything**.
2. Stops Pi. The session id is kept for the resume.
3. Stops the backend (the model server Pi talks to), plus any stray host `ds4-server` on the backend port.
4. Waits until `MIN_FREE_GIB` of memory is available.
5. Runs the payload with `PAYLOAD_ENV`, killing it after `PAYLOAD_TIMEOUT` seconds.
6. Restarts the backend **exactly as it was** and waits for its health URL.
7. Resumes Pi in the original tmux pane with a message pointing at the run's `summary.json`.

If anything fails after step 2, an emergency handler stops the payload, brings the backend back and resumes Pi with an "aborted" message.

## Running it

```sh
supervisor.sh [options] PI_PID SESSION_ID TMUX_PANE [PAYLOAD] [-- PAYLOAD_ARGS...]
```

| Option | Meaning |
|---|---|
| `--profile NAME\|PATH` | Profile to load. `NAME` means `supervisor.d/NAME.env`. Default: `$SUP_PROFILE`, else `default`. |
| `--set KEY=VALUE` | Override one scalar setting, e.g. `--set PAYLOAD_TIMEOUT=7200`. Repeatable. |
| `--env NAME=VALUE` | Add an environment variable for the payload. Repeatable. |
| `--backend-env NAME=VALUE` | Add an environment variable for the backend server (see the replay note below). Repeatable. |
| `--dry-run` | Run preflight and print what would happen. Nothing is stopped or run. |

`PAYLOAD` is relative to the repo (default `DEFAULT_PAYLOAD`, i.e. `bench.sh`). Arguments after `--` are passed to it unchanged. The old form, with a single fifth argument holding whitespace-separated arguments, still works.

Exit code: the payload's exit code (124 means it hit `PAYLOAD_TIMEOUT`); 1 for a supervisor failure; 75 if another cycle holds the lock.

Examples:

```sh
# Check a new profile without interrupting anything
./supervisor.sh --profile default --dry-run "$PI_PID" "$SESSION" "$PANE" bench.sh

# Longer run with an extra variable for this payload only
./supervisor.sh --set PAYLOAD_TIMEOUT=7200 --env DS4_ROCM_V41_X=1 \
    "$PI_PID" "$SESSION" "$PANE" .agentkit/runs/qakv-…/payload.sh
```

## Profiles

A profile is a bash file. It is `source`d, so you can use variables, `${VAR:-}` and
arrays. Anything you leave out keeps the default shown below. Start from
`default.env`, which reproduces the Strix Halo / Cockpit setup.

### General

| Setting | Default | Meaning |
|---|---|---|
| `REPO` | the script's directory | Checkout the payload runs in (exported as `G` for old payloads). |
| `PI_BIN` | `command -v pi` | Pi executable used for the resume. |
| `PI_RESUME_ARGS` | `(--tui-mode regular)` | Extra arguments for the resume command. |
| `DEFAULT_PAYLOAD` | `bench.sh` | Payload when none is given. |
| `RUNS_DIR` | `$REPO/findings` | Where run directories are created. |
| `MODEL` | empty | When set, preflight checks it is readable. Exported to the payload as `SUP_MODEL`. |
| `MIN_FREE_GIB` | `100` | Free memory (GiB) required before the payload starts. |
| `MEMORY_WAIT_S` | `240` | How long to wait for that memory. |
| `PAYLOAD_TIMEOUT` | `3600` | The payload is killed after this many seconds. `ak ab plan` reads this to refuse plans that won't fit. |
| `PAYLOAD_ENV` | `()` | Array of `NAME=VALUE` set for every payload. `--env` appends to it. |
| `CLEAR_DS4_ENV` | `1` | Unset inherited `DS4_*` variables before the payload. Variables in `PAYLOAD_ENV` are applied after, so they survive. |
| `REQUIRE_GPU_DEVICES` | `1` | Require `/dev/kfd` and `/dev/dri` (skipped for `BACKEND=none`). |
| `NOTIFY` | `1` | Desktop notification on failure. |

### Backend

| Setting | Default | Meaning |
|---|---|---|
| `BACKEND` | `podman` | `podman`, `command`, `systemd` or `none`. |
| `HEALTH_URL` | `http://127.0.0.1:8000/v1/models` | Polled after the restart. Empty means no health check. |
| `HEALTH_WAIT` | `900` | Seconds to wait for health. |
| `BACKEND_PORT` | `8000` | Used to find the container and any stray host server. |
| `STOP_STRAY_SERVER` | `1` | Stop a host `ds4-server` listening on `BACKEND_PORT`. |
| `BACKEND_ENV` | `()` | `NAME=VALUE` entries for the server. `--backend-env` appends to it. |
| `SERVER_ARGS` | `()` | Server arguments: the container command, or the host binary's arguments. |

**`BACKEND=podman`**

| Setting | Default | Meaning |
|---|---|---|
| `CONTAINER_NAME` | empty | Empty means use the container that publishes `BACKEND_PORT`. |
| `CONTAINER_NAME_FALLBACK` | empty | Name to use when nothing is running, i.e. the backend is already down. |
| `CONTAINER_RECREATE` | `replay` | `replay`: recreate a removed container from the create command captured in preflight. `profile`: always build it from the settings below. |
| `CONTAINER_IMAGE` | empty | Image for profile-based recreation. |
| `CONTAINER_RUN_ARGS` | `()` | `podman run` options for profile-based recreation (devices, ports, volumes…). |

**How the restart works.** Before stopping the container, preflight saves
`podman inspect` → `.Config.CreateCommand` and `.ImageName` into the run directory, and
checks that the image still exists.
- If the container still exists after the stop (no `--rm`), it is simply started again.
- If it was removed (Cockpit uses `--rm`), it is recreated from the captured command. `-i`/`-t` are dropped and `--detach` is added; everything else is identical.
- The profile's `CONTAINER_IMAGE` / `CONTAINER_RUN_ARGS` / `SERVER_ARGS` / `BACKEND_ENV` are used only when nothing was running at preflight, or with `CONTAINER_RECREATE=profile`.

So **`--backend-env` and `BACKEND_ENV` do not change a replayed container**. That is deliberate: the backend comes back exactly as it was. Use `--set CONTAINER_RECREATE=profile` to apply them.

**`BACKEND=command`**: a host server started by the supervisor.

| Setting | Default | Meaning |
|---|---|---|
| `SERVER_BIN` | — | Executable, e.g. `$REPO/ds4-server`. Started with `BACKEND_ENV` and `SERVER_ARGS`, detached, PID in `$REPO/ds4-server.pid`. |
| `SERVER_LOG` | run dir | Server log path. |

**`BACKEND=systemd`**

| Setting | Default | Meaning |
|---|---|---|
| `SYSTEMD_UNIT` | — | Unit to stop and start. |
| `SYSTEMD_USER` | `1` | Use `systemctl --user`. |

**`BACKEND=none`**: nothing is stopped or started and there is no health check. Use it for payloads that don't compete with a server, and for testing a profile.

## Run directory

Each cycle creates `RUNS_DIR/supervisor-<UTC time>-XXXXXX/`. The path is exported to the
payload as `PI_RUN_DIR`.

| File | Contents |
|---|---|
| `supervisor.env` | Effective settings after the profile, `--set` and `--env` were applied. |
| `payload.log` | Payload stdout and stderr. `$REPO/payload.log` is a symlink to the latest one. |
| `summary.json` | Handoff written by `pi-run-summary.py` (status, rc, elapsed, `result.json` if small). |
| `result.json` | Optional, written by the payload (`ak ab` payloads do this). |
| `container-create.json`, `container-image.txt` | Captured podman state used for the restart. |
| `server.log` | `BACKEND=command` server output. |

Variables exported to the payload: `PI_RUN_DIR`, `G` (= `REPO`), `SUP_PROFILE`, `SUP_PAYLOAD_TIMEOUT`, `SUP_MODEL`, `RESULTS_OFFSET`, plus `PAYLOAD_ENV`. The payload runs with `REPO` as its working directory.

## With agentkit

`ak ab plan spec.json --profile NAME` reads `PAYLOAD_TIMEOUT` from the same profile.
- If the spec has `est_run_s` (seconds per measured run) and the whole plan would not fit in 90 % of the timeout, the plan is refused and the fix suggests a smaller even pair count per cycle.
- Without `est_run_s` it only warns.
- `ak ab analyze` reports `run_seconds_median`; use it as `est_run_s` next time.

## Notes for a Pi tool

- Pass the profile name and any `--set` / `--env` from the tool's arguments; the rest stays in the profile.
- Run `--dry-run` first when a profile is new or the agent has changed settings. It is cheap and interrupts nothing.
- Launch it detached (`setsid nohup … &`). The supervisor stops Pi itself, so the tool call must return immediately after launching.
- Only one cycle can run at a time (lock file `.supervisor.lock` next to the script). A second launch exits with 75.

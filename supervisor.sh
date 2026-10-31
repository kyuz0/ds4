#!/usr/bin/env bash
# supervisor.sh
#
# Usage:
#   setsid nohup /home/dani/code/v4.1-ds4/supervisor.sh \
#       "$PI_PID" "$SESSION_ID" "$(tmux display-message -p '#{pane_id}')" [payload] \
#       >/home/dani/code/v4.1-ds4/supervisor.log 2>&1 &
#
# Flow (all inside the same tmux session):
#   1. stop Pi, keeping the session id for the resume
#   2. stop the DS4 cockpit container (Pi's V4.1 backend)
#   3. stop any stray host DS4 server on :8000
#   4. wait for enough free memory
#   5. run the DS4 payload (default: bench.sh)
#   6. restart the container and wait for /v1/models
#   7. resume the Pi session in the original tmux pane
#
# The container name is detected from the process listening on :8000 so a
# cockpit restart with a new name does not need a supervisor edit. Missing
# temporary containers are recreated with the Podman command below. Set CONTAINER_NAME
# explicitly when starting while the backend is already down.

set -uo pipefail

PI_PID="${1:?usage: supervisor.sh PI_PID SESSION_ID TMUX_PANE [payload]}"
SESSION="${2:?usage: supervisor.sh PI_PID SESSION_ID TMUX_PANE [payload]}"
TMUX_PANE="${3:?usage: supervisor.sh PI_PID SESSION_ID TMUX_PANE [payload]}"
PAYLOAD="${4:-/home/dani/code/v4.1-ds4/bench.sh}"
PAYLOAD_ARGS="${5:-}"

G="${G:-/home/dani/code/v4.1-ds4}"

# Hold one lock for the complete cycle. --close keeps the lock descriptor out
# of the backend and resumed Pi, so they cannot accidentally retain it.
if [ "${DS4_SUPERVISOR_LOCKED:-}" != 1 ]; then
  exec flock -n -E 75 --close "$G/.supervisor.lock" \
    env DS4_SUPERVISOR_LOCKED=1 bash "$0" "$@"
fi
unset DS4_SUPERVISOR_LOCKED

# The tool passes an argv array after --. Historical callers retain the old
# whitespace/glob argument convention through the fifth positional parameter.
if [ "${5:-}" = -- ]; then
  PAYLOAD_ARGV=("${@:6}")
  PAYLOAD_ARGS=""
  if [ "${#PAYLOAD_ARGV[@]}" -gt 0 ]; then
    printf -v PAYLOAD_ARGS '%q ' "${PAYLOAD_ARGV[@]}"
  fi
else
  PAYLOAD_ARGV=( $PAYLOAD_ARGS )
fi

SERVER_PID_FILE="$G/ds4-server.pid"

CONTAINER_NAME="${CONTAINER_NAME:-}"
export CONTAINER_NAME
CONTAINER_HEALTH="${CONTAINER_HEALTH:-http://127.0.0.1:8000/v1/models}"
CONTAINER_WAIT="${CONTAINER_WAIT:-900}"

PI=/home/dani/.local/share/pi-node/node-v22.23.2-linux-x64/bin/pi

TMUX_SOCK="${TMUX:-}"
TMUX_SOCK="${TMUX_SOCK%%,*}"
TMUX_SOCK="${TMUX_SOCK:-/tmp/tmux-$(id -u)/default}"

export PATH="/home/dani/.local/share/pi-node/node-v22.23.2-linux-x64/bin:$PATH"

log() {
  echo "[sup $(date '+%H:%M:%S')] $*"
}


# --------------------------------------------------------------------------
# DS4 SERVER (cleanup only; never started here)
# --------------------------------------------------------------------------

find_server_pid() {
  # Prefer our stored PID.
  if [ -f "$SERVER_PID_FILE" ]; then
    local pid
    pid="$(cat "$SERVER_PID_FILE")"

    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi

    rm -f "$SERVER_PID_FILE"
  fi

  # Fallback for a DS4 server that was started manually.
  local pid
  pid="$(
    ss -ltnp 2>/dev/null |
      awk '/127\.0\.0\.1:8000|0\.0\.0\.0:8000/ {
        if (match($0, /pid=[0-9]+/))
          print substr($0, RSTART+4, RLENGTH-4)
      }' |
      head -1
  )"

  if [ -n "$pid" ] && ps -p "$pid" -o comm= 2>/dev/null | grep -qx ds4-server; then
    echo "$pid"
    return 0
  fi

  return 1
}


stop_server() {
  local pid

  if ! pid="$(find_server_pid)"; then
    log "DS4 server already stopped"
    return 0
  fi

  log "stopping DS4 server pid=$pid"

  kill -TERM "$pid" 2>/dev/null || true

  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$SERVER_PID_FILE"
      log "DS4 server stopped"
      return 0
    fi

    sleep 1
  done

  log "DS4 server did not stop gracefully, sending SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true

  rm -f "$SERVER_PID_FILE"

  sleep 2
}


# --------------------------------------------------------------------------
# PI BACKEND CONTAINER (podman)
# --------------------------------------------------------------------------

# Validate recovery before interrupting Pi or stopping its current backend.
preflight() {
  if [ ! -x "$PI" ]; then
    log "*** Pi executable is unavailable: $PI ***"
    return 1
  fi
  if ! tmux -S "$TMUX_SOCK" display-message -p -t "$TMUX_PANE" '#{pane_id}' >/dev/null; then
    log "*** cannot locate the tmux pane needed to resume Pi ***"
    return 1
  fi
  if ! [[ "$CONTAINER_WAIT" =~ ^[0-9]+$ ]] || [ "$CONTAINER_WAIT" -lt 3 ]; then
    log "*** CONTAINER_WAIT must be an integer of at least 3 seconds ***"
    return 1
  fi
  podman info >/dev/null || return 1
  if ! podman image exists docker.io/kyuz0/strix-halo-ds4-toolbox:rocm-10.0; then
    log "*** DS4 image unavailable locally; update it in Cockpit first ***"
    return 1
  fi
  if [ ! -r /home/dani/models/deepseek-v4.1/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf ]; then
    log "*** DS4 model is not readable ***"
    return 1
  fi
  if [ ! -c /dev/kfd ] || [ ! -d /dev/dri ]; then
    log "*** run on the host with /dev/kfd and /dev/dri exposed ***"
    return 1
  fi

  local containers
  containers="$(podman ps --format '{{.Names}} {{.Ports}}')" || return 1
  if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME="$(awk '/127\.0\.0\.1:8000->8000/ { print $1; exit }' <<< "$containers")"
  fi
  CONTAINER_NAME="${CONTAINER_NAME:-ds4-cockpit-server}"
  export CONTAINER_NAME
  log "backend container: $CONTAINER_NAME"
}


container_is_running() {
  local names
  names="$(podman ps --format '{{.Names}}')" || return 2
  grep -Fxq -- "$CONTAINER_NAME" <<< "$names"
}


container_is_healthy() {
  curl -sf --max-time 3 -o /dev/null "$CONTAINER_HEALTH"
}


container_diagnostics() {
  podman logs --tail 80 "$CONTAINER_NAME" 2>&1 || true
}


stop_container() {
  local status
  if container_is_running; then
    log "stopping container $CONTAINER_NAME"
  else
    status=$?
    if [ "$status" -ne 1 ]; then
      log "*** cannot inspect Podman; refusing to run the payload ***"
      return 1
    fi
    log "container $CONTAINER_NAME already stopped"
    return 0
  fi

  # Cockpit uses --rm, so its container may disappear after this stop.
  podman stop -t 30 "$CONTAINER_NAME" || return 1
  sleep 2

  if container_is_running; then
    log "*** container $CONTAINER_NAME still running after stop ***"
    return 1
  else
    status=$?
    if [ "$status" -ne 1 ]; then
      log "*** cannot verify that the container stopped ***"
      return 1
    fi
  fi
  log "container $CONTAINER_NAME stopped"
}


start_container() {
  local status i
  if container_is_running; then
    log "container $CONTAINER_NAME already running"
  else
    status=$?
    if [ "$status" -ne 1 ]; then
      log "*** cannot inspect Podman; backend recovery failed ***"
      return 1
    fi
    if podman container exists "$CONTAINER_NAME"; then
      log "starting existing container $CONTAINER_NAME"
      podman start "$CONTAINER_NAME" || {
        log "*** failed to start container $CONTAINER_NAME ***"
        container_diagnostics
        return 1
      }
    else
      status=$?
      if [ "$status" -ne 1 ]; then
        log "*** cannot determine whether container $CONTAINER_NAME exists ***"
        return 1
      fi
      log "recreating removed Cockpit container $CONTAINER_NAME"
      # Run detached and retain the container so later cycles can restart it.
      podman run --detach --name "$CONTAINER_NAME" \
        --device /dev/dri \
        --device /dev/kfd \
        --group-add keep-groups \
        --security-opt seccomp=unconfined \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt label=disable \
        --userns=keep-id \
        --env "DS4_LOCK_FILE=/tmp/$CONTAINER_NAME.lock" \
        --env DS4_ROCM_V41_SELECTIVE_STAGE=1 \
        -p 127.0.0.1:8000:8000 \
        -v /home/dani/models/deepseek-v4.1:/models:ro \
        localhost/ds4-v41-optimized:latest \
        -m /models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
        --ctx 131072 \
        --host 0.0.0.0 \
        --port 8000 \
        --ssd-streaming \
        --ssd-streaming-cache-experts 92GB || {
        log "*** failed to recreate container $CONTAINER_NAME ***"
        container_diagnostics
        return 1
      }
    fi
  fi

  for i in $(seq 1 $((CONTAINER_WAIT / 3))); do
    if ! container_is_running; then
      log "*** container $CONTAINER_NAME exited or Podman became unavailable during startup ***"
      container_diagnostics
      return 1
    fi
    if container_is_healthy; then
      log "container $CONTAINER_NAME healthy"
      return 0
    fi
    sleep 3
  done

  log "*** container $CONTAINER_NAME NOT healthy after ${CONTAINER_WAIT}s ***"
  container_diagnostics
  return 1
}


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

wait_memory() {
  # Wait until at least 100 GiB are available. $1 = max iterations of 2s.
  local iters="${1:-120}" i=0 AVAIL=0

  while [ "$i" -lt "$iters" ]; do
    i=$((i + 1))
    AVAIL=$(
      awk '/^MemAvailable:/ {
        printf "%.0f", $2 / 1048576
      }' /proc/meminfo
    )

    if [ "$AVAIL" -ge 100 ]; then
      log "memory available: ${AVAIL} GiB"
      return 0
    fi

    if [ "$((i % 10))" -eq 1 ] && [ "$i" -gt 1 ]; then
      log "memory available: ${AVAIL} GiB, still waiting"
    fi

    sleep 2
  done

  log "*** memory still at ${AVAIL} GiB after $((iters * 2))s; refusing to run the payload ***"
  return 1
}


resume_pi() {
  local msg="$1"
  local quoted_g quoted_pi quoted_session quoted_msg cmd
  printf -v quoted_g '%q' "$G"
  printf -v quoted_pi '%q' "$PI"
  printf -v quoted_session '%q' "$SESSION"
  printf -v quoted_msg '%q' "$msg"
  cmd="cd $quoted_g && $quoted_pi --session $quoted_session --tui-mode regular $quoted_msg"

  log "resuming Pi session ${SESSION:0:8} in tmux pane $TMUX_PANE"

  # Clear any partially typed command.
  tmux -S "$TMUX_SOCK" send-keys \
    -t "$TMUX_PANE" C-u 2>/dev/null || true

  sleep 1

  if tmux -S "$TMUX_SOCK" send-keys \
      -t "$TMUX_PANE" "$cmd" Enter \
      2>>"$G/supervisor.log"; then
    log "Pi resume command sent successfully"
    return 0
  fi

  log "*** FAILED to resume Pi in tmux pane $TMUX_PANE ***"
  log "manual command: $cmd"
  return 1
}

write_run_summary() {
  local status="$1" rc="$2"
  if [ -n "${PAYLOAD_PID:-}" ] && [ -f "$G/payload.log" ]; then
    cp "$G/payload.log" "$PI_RUN_DIR/payload.log" || return 1
  fi
  python3 "$G/pi-run-summary.py" "$PI_RUN_DIR" "$status" "$rc" \
    "$PAYLOAD" "$PAYLOAD_ARGS" "$SESSION" "$G" "${START:-0}"
}


# --------------------------------------------------------------------------
# FAILURE RECOVERY
# --------------------------------------------------------------------------

emergency() {
  local rc=$?

  # Prevent recursive EXIT trap.
  trap - EXIT INT TERM

  log "supervisor aborted rc=$rc"

  # Kill a payload still running in the background.
  if [ -n "${PAYLOAD_PID:-}" ] && kill -0 "$PAYLOAD_PID" 2>/dev/null; then
    log "stopping payload pid=$PAYLOAD_PID"
    kill -TERM "$PAYLOAD_PID" 2>/dev/null || true

    for _ in $(seq 1 10); do
      kill -0 "$PAYLOAD_PID" 2>/dev/null || break
      sleep 1
    done

    kill -KILL "$PAYLOAD_PID" 2>/dev/null || true
  fi

  # Make sure no DS4 server is holding memory.
  stop_server

  wait_memory 60

  # Bring the container back up (Pi's backend).
  if ! container_is_healthy; then
    log "attempting emergency container restart"
    CONTAINER_WAIT=60 start_container || {
      log "*** MANUAL CONTAINER RECOVERY REQUIRED; see $G/supervisor.log ***"
    }
  fi

  # Best-effort resume of Pi with an abort message (only if Pi is gone).
  if ! kill -0 "$PI_PID" 2>/dev/null; then
    if container_is_running && container_is_healthy; then
      write_run_summary aborted "$rc" || log "could not write run summary; inspect $PI_RUN_DIR/payload.log"
      resume_pi "DS4 cycle aborted rc=$rc. Read $PI_RUN_DIR/summary.json (or its payload.log if summary is absent). Review this run before another run. The backend is healthy again."
    else
      log "*** Pi remains stopped because its local backend is unavailable ***"
    fi
  fi

  notify-send -u critical \
    "DS4 cycle aborted rc=$rc" \
    "Recovery attempted. See $G/supervisor.log" \
    2>/dev/null || true

  exit "$rc"
}

if ! preflight; then
  log "*** preflight failed; Pi and the backend have not been interrupted ***"
  exit 1
fi

# Prepare unique artifacts before interrupting anything. Legacy payload.log stays
# available for old payloads; the run-specific copy is the handoff authority.
PI_RUN_DIR=$(mktemp -d "$G/findings/supervisor-$(date -u '+%Y%m%dT%H%M%SZ')-XXXXXX") || exit 1
export PI_RUN_DIR
write_run_summary prepared -1 || exit 1

trap emergency EXIT INT TERM

log "PI_SUPERVISOR_READY=$PI_RUN_DIR"

# --------------------------------------------------------------------------
# 1. RETIRE CURRENT PI (session id is kept for the resume)
# --------------------------------------------------------------------------

log "waiting 5s before terminating Pi pid=$PI_PID"
sleep 5

if kill -0 "$PI_PID" 2>/dev/null; then
  kill -TERM "$PI_PID" 2>/dev/null || true

  for _ in $(seq 1 20); do
    kill -0 "$PI_PID" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$PI_PID" 2>/dev/null; then
    log "Pi did not exit, sending SIGKILL"
    kill -KILL "$PI_PID" 2>/dev/null || true
  fi

  log "Pi terminated"
else
  log "Pi already gone"
fi

sleep 3


# --------------------------------------------------------------------------
# 2. STOP THE DS4 COCKPIT BACKEND (must be down before benchmark work)
# --------------------------------------------------------------------------

if ! stop_container; then
  log "*** could not stop container $CONTAINER_NAME; aborting ***"
  exit 1
fi


# --------------------------------------------------------------------------
# 3. STOP ANY STRAY DS4 SERVER
# --------------------------------------------------------------------------

stop_server


# --------------------------------------------------------------------------
# 4. WAIT FOR MEMORY
# --------------------------------------------------------------------------

log "waiting for memory to become available"
wait_memory 120 || exit 1


# --------------------------------------------------------------------------
# 5. RUN THE DS4 PAYLOAD
# --------------------------------------------------------------------------

log "running payload: $PAYLOAD $PAYLOAD_ARGS"

START=$(date +%s)
RESULTS_OFFSET=$(stat -c %s "$G/RESULTS.md" 2>/dev/null || echo 0)
export RESULTS_OFFSET

timeout -k 60 3600 \
  bash "$PAYLOAD" "${PAYLOAD_ARGV[@]}" \
  >"$G/payload.log" 2>&1 &

PAYLOAD_PID=$!

wait "$PAYLOAD_PID"
RC=$?

log "payload rc=$RC in $(( $(date +%s) - START ))s"
write_run_summary completed "$RC" || exit 1


# --------------------------------------------------------------------------
# 6. RESTART THE DS4 COCKPIT BACKEND
# --------------------------------------------------------------------------

if ! start_container; then
  log "*** container $CONTAINER_NAME NOT healthy; attempting recovery before Pi can resume ***"

  notify-send -u critical \
    "DS4 cockpit container not healthy" \
    "Check: podman logs $CONTAINER_NAME" \
    2>/dev/null || true
  exit 1
fi

# Everything the supervisor manages is back online.
trap - EXIT INT TERM


# --------------------------------------------------------------------------
# 7. RESUME PI IN ORIGINAL TMUX PANE
# --------------------------------------------------------------------------

MSG="DS4 cycle finished payload rc=$RC. Read $PI_RUN_DIR/summary.json. Review correctness and measurements before another run; rc=0 alone proves neither. Backend is online; host DS4 server is intentionally stopped."

resume_pi "$MSG" || exit 1

exit "$RC"


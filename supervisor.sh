#!/usr/bin/env bash
# supervisor.sh - run a DS4 payload with exclusive use of the machine, then resume Pi.
#
# Usage:
#   supervisor.sh [options] PI_PID SESSION_ID TMUX_PANE [PAYLOAD] [-- PAYLOAD_ARGS...]
#
# Options (before the positional arguments):
#   --profile NAME|PATH   settings file (default: $SUP_PROFILE or "default");
#                         NAME resolves to supervisor.d/NAME.env next to this script
#   --set KEY=VALUE       override one scalar setting from the profile
#   --env NAME=VALUE      add an environment variable for the payload
#   --backend-env N=V     add an environment variable for the backend server
#   --dry-run             preflight + print the plan; stop, run and change nothing
#
# Flow (all inside the same tmux session):
#   1. stop Pi, keeping the session id for the resume
#   2. stop the backend (podman | command | systemd | none, from the profile)
#   3. stop any stray host ds4-server on the backend port
#   4. wait for enough free memory
#   5. run the payload with the profile's PAYLOAD_ENV
#   6. restart the backend exactly as it was and wait for health
#   7. resume the Pi session in the original tmux pane
#
# See supervisor.d/README.md for every setting.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"

# One lock for the complete cycle. --close keeps the descriptor out of the
# backend and the resumed Pi.
if [ "${DS4_SUPERVISOR_LOCKED:-}" != 1 ]; then
  exec flock -n -E 75 --close "$SCRIPT_DIR/.supervisor.lock" \
    env DS4_SUPERVISOR_LOCKED=1 bash "$0" "$@"
fi
unset DS4_SUPERVISOR_LOCKED

log() { echo "[sup $(date '+%H:%M:%S')] $*"; }
die() { log "*** $* ***"; exit 1; }

# --------------------------------------------------------------------------
# ARGUMENTS
# --------------------------------------------------------------------------

PROFILE="${SUP_PROFILE:-default}"
DRY_RUN=0
SETS=()
EXTRA_PAYLOAD_ENV=()
EXTRA_BACKEND_ENV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
    --set) SETS+=("${2:?--set needs KEY=VALUE}"); shift 2 ;;
    --env) EXTRA_PAYLOAD_ENV+=("${2:?--env needs NAME=VALUE}"); shift 2 ;;
    --backend-env) EXTRA_BACKEND_ENV+=("${2:?--backend-env needs NAME=VALUE}"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    --) shift; break ;;
    -*) die "unknown option $1" ;;
    *) break ;;
  esac
done

USAGE="usage: supervisor.sh [options] PI_PID SESSION_ID TMUX_PANE [payload] [-- args...]"
PI_PID="${1:?$USAGE}"
SESSION="${2:?$USAGE}"
TMUX_PANE="${3:?$USAGE}"
PAYLOAD_ARG="${4:-}"
PAYLOAD_ARGV=()
if [ "${5:-}" = -- ]; then
  PAYLOAD_ARGV=("${@:6}")
elif [ -n "${5:-}" ]; then
  read -r -a PAYLOAD_ARGV <<< "$5"   # historical whitespace-separated form
fi
PAYLOAD_ARGS=""
if [ "${#PAYLOAD_ARGV[@]}" -gt 0 ]; then
  printf -v PAYLOAD_ARGS '%q ' "${PAYLOAD_ARGV[@]}"
fi

# --------------------------------------------------------------------------
# SETTINGS: defaults, then the profile, then --set / --env overrides
# --------------------------------------------------------------------------

REPO="$SCRIPT_DIR"
PI_BIN="$(command -v pi || true)"
PI_RESUME_ARGS=(--tui-mode regular)
DEFAULT_PAYLOAD="bench.sh"
RUNS_DIR=""                     # default: $REPO/findings
MODEL=""                        # checked for readability when set
MIN_FREE_GIB=100
MEMORY_WAIT_S=240
PAYLOAD_TIMEOUT=3600
PAYLOAD_ENV=()
CLEAR_DS4_ENV=1                 # unset inherited DS4_* before the payload
NOTIFY=1
REQUIRE_GPU_DEVICES=1          # require /dev/kfd and /dev/dri unless BACKEND=none

BACKEND=podman                  # podman | command | systemd | none
HEALTH_URL="http://127.0.0.1:8000/v1/models"
HEALTH_WAIT=900
BACKEND_PORT=8000
STOP_STRAY_SERVER=1             # stop a host ds4-server holding BACKEND_PORT
BACKEND_ENV=()

# podman
CONTAINER_NAME=""               # empty: detect the container publishing BACKEND_PORT
CONTAINER_NAME_FALLBACK=""      # used when nothing publishes the port (backend already down)
CONTAINER_RECREATE=replay       # replay the captured create command | profile
CONTAINER_IMAGE=""
CONTAINER_RUN_ARGS=()           # podman run options (fallback recreate)
SERVER_ARGS=()                  # server arguments (container command or host binary)

# command
SERVER_BIN=""
SERVER_LOG=""                   # default: run dir

# systemd
SYSTEMD_UNIT=""
SYSTEMD_USER=1

case "$PROFILE" in
  */*|*.env) PROFILE_FILE="$PROFILE" ;;
  *) PROFILE_FILE="$SCRIPT_DIR/supervisor.d/$PROFILE.env" ;;
esac
[ -r "$PROFILE_FILE" ] || die "profile not found: $PROFILE_FILE"
# shellcheck disable=SC1090
source "$PROFILE_FILE" || die "profile failed to load: $PROFILE_FILE"

for kv in "${SETS[@]}"; do
  key="${kv%%=*}"
  [[ "$kv" == *=* && "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "--set expects KEY=VALUE, got $kv"
  if declare -p "$key" 2>/dev/null | grep -q '^declare -a'; then
    die "--set cannot replace array $key; use --env / --backend-env or the profile"
  fi
  printf -v "$key" '%s' "${kv#*=}"
done
PAYLOAD_ENV+=("${EXTRA_PAYLOAD_ENV[@]}")
BACKEND_ENV+=("${EXTRA_BACKEND_ENV[@]}")
for kv in "${PAYLOAD_ENV[@]}" "${BACKEND_ENV[@]}"; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || die "environment entry must be NAME=VALUE: $kv"
done

G="$REPO"                       # historical name used by payloads
RUNS_DIR="${RUNS_DIR:-$REPO/findings}"
PAYLOAD="${PAYLOAD_ARG:-$DEFAULT_PAYLOAD}"
[[ "$PAYLOAD" = /* ]] || PAYLOAD="$REPO/$PAYLOAD"
SERVER_PID_FILE="$REPO/ds4-server.pid"

TMUX_SOCK="${TMUX:-}"
TMUX_SOCK="${TMUX_SOCK%%,*}"
TMUX_SOCK="${TMUX_SOCK:-/tmp/tmux-$(id -u)/default}"
[ -n "$PI_BIN" ] && PATH="$(dirname "$PI_BIN"):$PATH"
export PATH

# --------------------------------------------------------------------------
# STRAY HOST SERVER (cleanup only)
# --------------------------------------------------------------------------

port_pid() {
  ss -ltnp 2>/dev/null |
    awk -v p=":$BACKEND_PORT" '$4 ~ p"$" { if (match($0, /pid=[0-9]+/)) print substr($0, RSTART+4, RLENGTH-4) }' |
    head -1
}

find_server_pid() {
  if [ -f "$SERVER_PID_FILE" ]; then
    local pid; pid="$(cat "$SERVER_PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then echo "$pid"; return 0; fi
    rm -f "$SERVER_PID_FILE"
  fi
  local pid; pid="$(port_pid)"
  if [ -n "$pid" ] && ps -p "$pid" -o comm= 2>/dev/null | grep -qx ds4-server; then
    echo "$pid"; return 0
  fi
  return 1
}

stop_pid() {  # $1 pid, $2 label
  local pid="$1"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || { log "$2 stopped"; return 0; }
    sleep 1
  done
  log "$2 did not stop gracefully, sending SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  sleep 2
}

stop_server() {
  [ "$STOP_STRAY_SERVER" = 1 ] || return 0
  local pid
  if ! pid="$(find_server_pid)"; then log "no host ds4-server on :$BACKEND_PORT"; return 0; fi
  log "stopping host ds4-server pid=$pid"
  stop_pid "$pid" "host ds4-server"
  rm -f "$SERVER_PID_FILE"
}

# --------------------------------------------------------------------------
# BACKENDS: each defines preflight / stop / start / running
# --------------------------------------------------------------------------

backend_healthy() {
  [ -z "$HEALTH_URL" ] && return 0
  curl -sf --max-time 3 -o /dev/null "$HEALTH_URL"
}

wait_healthy() {  # $1 seconds
  local i
  for i in $(seq 1 $(( ($1 + 2) / 3 ))); do
    backend_running || { log "*** backend exited during startup ***"; return 1; }
    backend_healthy && { log "backend healthy"; return 0; }
    sleep 3
  done
  log "*** backend NOT healthy after ${1}s ***"
  return 1
}

# ---- podman

podman_running() {
  local names; names="$(podman ps --format '{{.Names}}')" || return 2
  grep -Fxq -- "$CONTAINER_NAME" <<< "$names"
}

podman_preflight() {
  command -v podman >/dev/null || { log "*** podman not found ***"; return 1; }
  podman info >/dev/null || return 1
  if [ -z "$CONTAINER_NAME" ]; then
    CONTAINER_NAME="$(podman ps --format '{{.Names}} {{.Ports}}' |
      awk -v p=":$BACKEND_PORT->" 'index($0, p) { print $1; exit }')"
  fi
  CONTAINER_NAME="${CONTAINER_NAME:-$CONTAINER_NAME_FALLBACK}"
  [ -n "$CONTAINER_NAME" ] || { log "*** no container publishes :$BACKEND_PORT; set CONTAINER_NAME ***"; return 1; }
  log "backend container: $CONTAINER_NAME"
  # Capture how the running container was created so recovery replays it exactly.
  if podman_running; then
    podman inspect --format '{{json .Config.CreateCommand}}' "$CONTAINER_NAME" \
      > "$CAPTURE_DIR/container-create.json" || return 1
    podman inspect --format '{{.ImageName}}' "$CONTAINER_NAME" \
      > "$CAPTURE_DIR/container-image.txt" || return 1
    log "captured create command and image $(cat "$CAPTURE_DIR/container-image.txt")"
  elif ! podman container exists "$CONTAINER_NAME"; then
    [ "${#CONTAINER_RUN_ARGS[@]}" -gt 0 ] && [ -n "$CONTAINER_IMAGE" ] || {
      log "*** $CONTAINER_NAME is not running and the profile has no CONTAINER_IMAGE/CONTAINER_RUN_ARGS ***"
      return 1; }
  fi
  local image
  image="$(cat "$CAPTURE_DIR/container-image.txt" 2>/dev/null || echo "$CONTAINER_IMAGE")"
  if [ -n "$image" ] && ! podman image exists "$image"; then
    log "*** image $image is not available locally ***"; return 1
  fi
}

podman_stop() {
  if podman_running; then :; else
    [ $? -eq 1 ] || { log "*** cannot inspect podman ***"; return 1; }
    log "container $CONTAINER_NAME already stopped"; return 0
  fi
  log "stopping container $CONTAINER_NAME"
  podman stop -t 30 "$CONTAINER_NAME" || return 1
  sleep 2
  if podman_running; then log "*** $CONTAINER_NAME still running ***"; return 1; fi
  log "container $CONTAINER_NAME stopped"
}

# Rebuild the captured create command as a detached run.
replay_args() {
  python3 - "$CAPTURE_DIR/container-create.json" <<'PY'
import json, sys
cmd = json.load(open(sys.argv[1]))
if not isinstance(cmd, list) or "run" not in cmd[:3]:
    sys.exit(1)
i = cmd.index("run")
opts = [a for a in cmd[i + 1:] if a not in ("-i", "-t", "-it", "-ti", "--interactive", "--tty")]
out = ["run", "--detach"] + [a for a in opts if a not in ("-d", "--detach")]
sys.stdout.write("\0".join(out) + "\0")
PY
}

podman_start() {
  if podman_running; then log "container $CONTAINER_NAME already running"; return 0; fi
  if podman container exists "$CONTAINER_NAME"; then
    log "starting existing container $CONTAINER_NAME"
    podman start "$CONTAINER_NAME" || return 1
    return 0
  fi
  local args=() e
  if [ "$CONTAINER_RECREATE" = replay ] && [ -s "$CAPTURE_DIR/container-create.json" ] &&
     mapfile -d '' args < <(replay_args) && [ "${#args[@]}" -gt 0 ]; then
    log "recreating $CONTAINER_NAME from its captured create command"
  else
    log "recreating $CONTAINER_NAME from the profile"
    args=(run --detach --name "$CONTAINER_NAME" "${CONTAINER_RUN_ARGS[@]}")
    for e in "${BACKEND_ENV[@]}"; do args+=(--env "$e"); done
    args+=("$CONTAINER_IMAGE" "${SERVER_ARGS[@]}")
  fi
  podman "${args[@]}" || { podman logs --tail 80 "$CONTAINER_NAME" 2>&1 || true; return 1; }
}

# ---- command (host server started by this script)

command_running() {
  [ -f "$SERVER_PID_FILE" ] && kill -0 "$(cat "$SERVER_PID_FILE")" 2>/dev/null
}

command_preflight() {
  [ -x "$SERVER_BIN" ] || { log "*** SERVER_BIN is not executable: $SERVER_BIN ***"; return 1; }
}

command_stop() { stop_server; }

command_start() {
  command_running && { log "host server already running"; return 0; }
  local logf="${SERVER_LOG:-$PI_RUN_DIR/server.log}"
  log "starting host server $SERVER_BIN (log $logf)"
  ( cd "$REPO" && exec env "${BACKEND_ENV[@]}" setsid nohup "$SERVER_BIN" "${SERVER_ARGS[@]}" \
      >"$logf" 2>&1 < /dev/null ) &
  echo $! > "$SERVER_PID_FILE"
}

# ---- systemd

systemctl_() { if [ "$SYSTEMD_USER" = 1 ]; then systemctl --user "$@"; else systemctl "$@"; fi; }
systemd_running() { systemctl_ is-active --quiet "$SYSTEMD_UNIT"; }
systemd_preflight() {
  [ -n "$SYSTEMD_UNIT" ] || { log "*** SYSTEMD_UNIT is not set ***"; return 1; }
  systemctl_ cat "$SYSTEMD_UNIT" >/dev/null 2>&1 || { log "*** unit $SYSTEMD_UNIT not found ***"; return 1; }
}
systemd_stop() { log "stopping unit $SYSTEMD_UNIT"; systemctl_ stop "$SYSTEMD_UNIT"; }
systemd_start() { log "starting unit $SYSTEMD_UNIT"; systemctl_ start "$SYSTEMD_UNIT"; }

# ---- none

none_running() { return 0; }
none_preflight() { HEALTH_URL=""; }
none_stop() { log "backend: none"; }
none_start() { :; }

backend_running() { "${BACKEND}_running"; }
backend_stop() { "${BACKEND}_stop"; }
backend_start() { "${BACKEND}_start" && wait_healthy "$HEALTH_WAIT"; }

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

wait_memory() {  # $1 seconds
  local deadline=$(( $(date +%s) + $1 )) avail=0 n=0
  while :; do
    avail=$(awk '/^MemAvailable:/ { printf "%.0f", $2 / 1048576 }' /proc/meminfo)
    if [ "$avail" -ge "$MIN_FREE_GIB" ]; then log "memory available: ${avail} GiB"; return 0; fi
    [ "$(date +%s)" -ge "$deadline" ] && break
    n=$((n + 1)); [ $((n % 10)) -eq 1 ] && log "memory available: ${avail} GiB, waiting for ${MIN_FREE_GIB}"
    sleep 2
  done
  log "*** memory still at ${avail} GiB after ${1}s; refusing to run the payload ***"
  return 1
}

notify() { [ "$NOTIFY" = 1 ] && notify-send -u critical "$1" "$2" 2>/dev/null || true; }

resume_pi() {
  local msg="$1" cmd q_repo q_pi q_session q_msg q_extra=""
  printf -v q_repo '%q' "$REPO"
  printf -v q_pi '%q' "$PI_BIN"
  printf -v q_session '%q' "$SESSION"
  printf -v q_msg '%q' "$msg"
  [ "${#PI_RESUME_ARGS[@]}" -gt 0 ] && printf -v q_extra '%q ' "${PI_RESUME_ARGS[@]}"
  cmd="cd $q_repo && $q_pi --session $q_session $q_extra$q_msg"
  log "resuming Pi session ${SESSION:0:8} in tmux pane $TMUX_PANE"
  tmux -S "$TMUX_SOCK" send-keys -t "$TMUX_PANE" C-u 2>/dev/null || true
  sleep 1
  if tmux -S "$TMUX_SOCK" send-keys -t "$TMUX_PANE" "$cmd" Enter; then
    log "Pi resume command sent"; return 0
  fi
  log "*** FAILED to resume Pi; manual command: $cmd ***"
  return 1
}

write_run_summary() {
  python3 "$REPO/pi-run-summary.py" "$PI_RUN_DIR" "$1" "$2" \
    "$PAYLOAD" "$PAYLOAD_ARGS" "$SESSION" "$REPO" "${START:-0}"
}

write_effective_config() {
  {
    echo "# effective supervisor settings ($(date -Is))"
    echo "# profile: $PROFILE_FILE"
    local v
    for v in REPO PI_BIN MODEL MIN_FREE_GIB MEMORY_WAIT_S PAYLOAD_TIMEOUT BACKEND HEALTH_URL \
             HEALTH_WAIT BACKEND_PORT CONTAINER_NAME CONTAINER_RECREATE CONTAINER_IMAGE \
             SERVER_BIN SYSTEMD_UNIT PAYLOAD; do
      printf '%s=%q\n' "$v" "${!v}"
    done
    for v in PAYLOAD_ENV BACKEND_ENV SERVER_ARGS CONTAINER_RUN_ARGS PAYLOAD_ARGV; do
      declare -p "$v" | sed 's/^declare -a //'
    done
  } > "$1"
}

# --------------------------------------------------------------------------
# PREFLIGHT (nothing is interrupted before this passes)
# --------------------------------------------------------------------------

preflight() {
  case "$BACKEND" in podman|command|systemd|none) ;; *) log "*** unknown BACKEND=$BACKEND ***"; return 1 ;; esac
  [ -x "$PI_BIN" ] || { log "*** Pi executable unavailable: ${PI_BIN:-<not found>} (set PI_BIN) ***"; return 1; }
  tmux -S "$TMUX_SOCK" display-message -p -t "$TMUX_PANE" '#{pane_id}' >/dev/null ||
    { log "*** cannot locate tmux pane $TMUX_PANE ***"; return 1; }
  [ -r "$PAYLOAD" ] || { log "*** payload not readable: $PAYLOAD ***"; return 1; }
  [ -r "$REPO/pi-run-summary.py" ] || { log "*** $REPO/pi-run-summary.py missing ***"; return 1; }
  [[ "$PAYLOAD_TIMEOUT" =~ ^[0-9]+$ && "$HEALTH_WAIT" =~ ^[0-9]+$ && "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] ||
    { log "*** PAYLOAD_TIMEOUT, HEALTH_WAIT and MIN_FREE_GIB must be integers ***"; return 1; }
  if [ -n "$MODEL" ] && [ ! -r "$MODEL" ]; then log "*** model not readable: $MODEL ***"; return 1; fi
  if [ "$BACKEND" != none ] && [ "$REQUIRE_GPU_DEVICES" = 1 ] && { [ ! -c /dev/kfd ] || [ ! -d /dev/dri ]; }; then
    log "*** /dev/kfd and /dev/dri are required ***"; return 1
  fi
  "${BACKEND}_preflight"
}

# --------------------------------------------------------------------------
# FAILURE RECOVERY
# --------------------------------------------------------------------------

emergency() {
  local rc=$?
  trap - EXIT INT TERM
  log "supervisor aborted rc=$rc"
  if [ -n "${PAYLOAD_PID:-}" ] && kill -0 "$PAYLOAD_PID" 2>/dev/null; then
    log "stopping payload pid=$PAYLOAD_PID"
    stop_pid "$PAYLOAD_PID" payload
  fi
  stop_server
  wait_memory 120 || true
  if ! backend_healthy || ! backend_running; then
    log "attempting emergency backend restart"
    HEALTH_WAIT=120 backend_start || log "*** MANUAL BACKEND RECOVERY REQUIRED; see $PI_RUN_DIR ***"
  fi
  if ! kill -0 "$PI_PID" 2>/dev/null; then
    if backend_running && backend_healthy; then
      write_run_summary aborted "$rc" || log "could not write run summary"
      resume_pi "DS4 cycle aborted rc=$rc. Read $PI_RUN_DIR/summary.json (or payload.log there). Review this run before another run. The backend is healthy again."
    else
      log "*** Pi remains stopped because its backend is unavailable ***"
    fi
  fi
  notify "DS4 cycle aborted rc=$rc" "Recovery attempted. See $PI_RUN_DIR"
  exit "$rc"
}

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

mkdir -p "$RUNS_DIR" || die "cannot create $RUNS_DIR"
CAPTURE_DIR="$(mktemp -d "$RUNS_DIR/supervisor-$(date -u '+%Y%m%dT%H%M%SZ')-XXXXXX")" ||
  die "cannot create a run directory in $RUNS_DIR"

if ! preflight; then
  log "*** preflight failed; Pi and the backend have not been interrupted ***"
  rm -rf "$CAPTURE_DIR"
  exit 1
fi
PI_RUN_DIR="$CAPTURE_DIR"
export PI_RUN_DIR
write_effective_config "$PI_RUN_DIR/supervisor.env"

if [ "$DRY_RUN" = 1 ]; then
  log "DRY RUN profile=$PROFILE_FILE backend=$BACKEND ${CONTAINER_NAME:+container=$CONTAINER_NAME}"
  log "would stop Pi pid=$PI_PID, stop the backend, wait for ${MIN_FREE_GIB} GiB free,"
  log "run: $PAYLOAD $PAYLOAD_ARGS (timeout ${PAYLOAD_TIMEOUT}s)"
  log "payload env: ${PAYLOAD_ENV[*]:-<none>}"
  log "then restart the backend, wait for ${HEALTH_URL:-<no health check>}, resume Pi session ${SESSION:0:8}"
  log "effective settings: $PI_RUN_DIR/supervisor.env"
  exit 0
fi

write_run_summary prepared -1 || exit 1
trap emergency EXIT INT TERM
log "PI_SUPERVISOR_READY=$PI_RUN_DIR"

# 1. stop Pi (the session id is kept for the resume)
log "waiting 5s before terminating Pi pid=$PI_PID"
sleep 5
if kill -0 "$PI_PID" 2>/dev/null; then
  kill -TERM "$PI_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$PI_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$PI_PID" 2>/dev/null; then
    log "Pi did not exit, sending SIGKILL"; kill -KILL "$PI_PID" 2>/dev/null || true
  fi
  log "Pi terminated"
else
  log "Pi already gone"
fi
sleep 3

# 2-3. stop the backend and any stray host server
backend_stop || { log "*** could not stop the backend; aborting ***"; exit 1; }
stop_server

# 4. memory
log "waiting for memory to become available"
wait_memory "$MEMORY_WAIT_S" || exit 1

# 5. payload
log "running payload: $PAYLOAD $PAYLOAD_ARGS"
START=$(date +%s)
RESULTS_OFFSET=$(stat -c %s "$REPO/RESULTS.md" 2>/dev/null || echo 0)
export RESULTS_OFFSET G SUP_PROFILE="$PROFILE_FILE" SUP_PAYLOAD_TIMEOUT="$PAYLOAD_TIMEOUT"
[ -n "$MODEL" ] && export SUP_MODEL="$MODEL"
if [ "$CLEAR_DS4_ENV" = 1 ]; then
  for n in $(compgen -e | grep '^DS4_'); do unset "$n"; done
fi
ln -sfn "$PI_RUN_DIR/payload.log" "$REPO/payload.log" 2>/dev/null || true
( cd "$REPO" && exec env "${PAYLOAD_ENV[@]}" timeout -k 60 "$PAYLOAD_TIMEOUT" \
    bash "$PAYLOAD" "${PAYLOAD_ARGV[@]}" ) > "$PI_RUN_DIR/payload.log" 2>&1 &
PAYLOAD_PID=$!
wait "$PAYLOAD_PID"
RC=$?
PAYLOAD_PID=""
[ "$RC" -eq 124 ] && log "*** payload hit PAYLOAD_TIMEOUT=${PAYLOAD_TIMEOUT}s ***"
log "payload rc=$RC in $(( $(date +%s) - START ))s"
write_run_summary completed "$RC" || exit 1

# 6. backend
if ! backend_start; then
  notify "DS4 backend not healthy" "See $PI_RUN_DIR"
  exit 1
fi
trap - EXIT INT TERM

# 7. resume Pi
resume_pi "DS4 cycle finished payload rc=$RC. Read $PI_RUN_DIR/summary.json. Review correctness and measurements before another run; rc=0 alone proves neither. The backend is online." || exit 1
exit "$RC"

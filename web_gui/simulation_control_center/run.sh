#!/usr/bin/env bash
#
# MABEL SIM CONTROL CENTER — one command: the MuJoCo sim server + the studio.
#
#   ./simulation_control_center/run.sh                # sim (home scene) + GUI
#   ./simulation_control_center/run.sh --view          # + desktop MuJoCo window (=--viewer)
#   ./simulation_control_center/run.sh -s kitchen     # start in another scene
#   ./simulation_control_center/run.sh --gui-only     # attach to a running sim
#   ./simulation_control_center/run.sh --mock         # GUI only, synthetic data
#
# What it does:
#   * launches the SAME sim stack as server/run.sh (sim_teleop_bridge :9090 +
#     cameras :8080, home scene by default) — supervised by the backend, so the
#     GUI's "Scene ▾" switcher can restart it into any environment
#   * serves the studio at http://<this-mac>:8097
#   * macOS: no ROS — mapping (SLAM/Nav2) is skipped; pointcloud/LiDAR/odom
#     come from the built-in MuJoCo mirror instead
#   * Ubuntu with the built ROS workspace: launches the FULL ROS sim stack
#     (server/sim_ros_server.sh — cuVSLAM + nvblox + Nav2) and taps ROS
#
# Ctrl-C stops everything (backend tears the sim stack down with it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

SCENE=""
VIEWER=0
GUI_ONLY=0
MOCK=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scene)  SCENE="$2"; shift 2 ;;
    -v|--view|--viewer)  VIEWER=1; shift ;;
    --gui-only)  GUI_ONLY=1; shift ;;
    --mock)      MOCK=1; shift ;;
    -h|--help)   sed -n '2,21p' "$0"; exit 0 ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

# ── scene: accept a name from scenes.json ("kitchen") or an .xml path ────────
SCENES_DIR="$REPO/simulation/mabel_mujoco/scenes"
if [[ -n "$SCENE" && ! -f "$SCENE" ]]; then
  for cand in "$SCENES_DIR/$SCENE" "$SCENES_DIR/${SCENE}.xml" \
              "$SCENES_DIR/mabel_${SCENE}.xml"; do
    [[ -f "$cand" ]] && SCENE="$cand" && break
  done
  [[ -f "$SCENE" ]] || { echo "ERROR: scene not found: $SCENE"; exit 1; }
fi
[[ -n "$SCENE" ]] && export MABEL_SCENE="$SCENE"

# ── platform: Ubuntu with a built ROS workspace gets the FULL ROS sim stack ──
OS="$(uname -s)"
if [[ "$OS" == "Linux" && -d "$REPO/ros/install" && $MOCK -eq 0 ]]; then
  echo "▸ Linux + built ros detected → ROS sim stack (SLAM + Nav2 + RViz)"
  # shellcheck disable=SC1091
  [ -f "$REPO/ros/scripts/mabel_env.sh" ] && \
    source "$REPO/ros/scripts/mabel_env.sh" 2>/dev/null || true
  export MABEL_SCC_SIM_LAUNCHER="$REPO/server/sim_ros_server.sh"
  export MABEL_MAP_PORT="${MABEL_MAP_PORT:-9092}"   # the bridge's ROS map plane
else
  export MABEL_MAP_PORT="${MABEL_MAP_PORT:-0}"      # built-in MuJoCo mirror plane
fi

# ── launch policy flags → backend env ────────────────────────────────────────
export MABEL_SCC_VIEWER="$VIEWER"
if [[ $MOCK -eq 1 ]]; then
  export MABEL_SCC_SOURCE=mock
  export MABEL_SCC_MANAGE_SIM=0
  export MABEL_SCC_MIRROR=0
elif [[ $GUI_ONLY -eq 1 ]]; then
  export MABEL_SCC_MANAGE_SIM=0
else
  # run.sh OWNS the sim: always (re)launch the stack, taking over any stale
  # one (start_teleop's pre-clean kills it) — attaching to leftovers silently
  # runs OLD code and disables scene switching. --gui-only opts into attach.
  export MABEL_SCC_MANAGE_SIM=1
fi

# ── interpreter: needs fastapi + uvicorn (+ mujoco & websockets for the sim
#    telemetry/mirror) — the anaconda python used by the sim stack has all four
PY="${MABEL_SCC_PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in "$HOME/anaconda3/bin/python3" python3; do
    if "$cand" -c 'import fastapi, uvicorn' 2>/dev/null; then PY="$cand"; break; fi
  done
fi
[ -n "$PY" ] || { echo "ERROR: no python with fastapi/uvicorn found" \
  "(pip install fastapi 'uvicorn[standard]' websockets psutil)"; exit 1; }

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null \
  || hostname -I 2>/dev/null | awk '{print $1}' || echo 127.0.0.1)"
PORT="${MABEL_SCC_PORT:-8097}"

# ── free the ports: never fail to start because a previous run is still up ────
# (that "[Errno 48] address already in use" on :8097). Kill any stale SIM studio
# holding the GUI port and — unless --gui-only, which deliberately ATTACHES to a
# running sim — its bridge/cameras too. Scoped to the SIM's own local ports +
# process names; the real robot lives on `thor`, so nothing here can touch it.
_free_port() {                       # $1 = tcp port -> kill whatever LISTENs on it
  local pids; pids="$(lsof -nP -tiTCP:"$1" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] && { echo "  ▸ freeing :$1 (kill $pids)"; kill -9 $pids 2>/dev/null || true; }
}
if [ "$GUI_ONLY" = 1 ]; then
  _free_port "$PORT"                 # attach mode: only reclaim the studio port
else
  echo "▸ clearing any previous sim stack so the ports are free…"
  _free_port "$PORT"; _free_port 9090; _free_port 8080
  pkill -9 -f "simulation_control_center\.server" 2>/dev/null || true
  pkill -9 -f "sim_teleop_bridge\.py"             2>/dev/null || true
  pkill -9 -f "sim_camera_server\.py"             2>/dev/null || true
  sleep 0.3                          # let the OS reclaim the sockets before we bind
fi

cat <<EOF

  ──────────────────────────────────────────────────────────────────────────
    M A B E L   S I M   C O N T R O L   C E N T E R
    the control_center studio, entirely on simulation
  ──────────────────────────────────────────────────────────────────────────
    studio     http://${LAN_IP}:${PORT}          (this machine: http://localhost:${PORT})
    sim planes ws://${LAN_IP}:9090/teleop · cameras http://${LAN_IP}:8080
    scene      ${MABEL_SCENE:-mabel_home.xml (default)}
    viewer     $([ "$VIEWER" = 1 ] && echo "ON — desktop MuJoCo window (true robot state)" || echo "off (relaunch with --viewer)")
    mode       $([ "$MOCK" = 1 ] && echo "MOCK (no sim)" || { [ "$GUI_ONLY" = 1 ] && echo "GUI only (attach to a running sim)" || echo "sim supervised by the studio (Scene ▾ switches worlds)"; })

    Ctrl-C here stops the studio AND the sim it launched.
  ──────────────────────────────────────────────────────────────────────────

EOF

# pop the studio in the default browser as soon as it answers (non-blocking;
# MABEL_NO_BROWSER=1 suppresses — headless/SSH launches, and server/run.sh sets
# it because it pops the tab itself). Backgrounded BEFORE the exec below.
if [ "${MABEL_NO_BROWSER:-0}" != 1 ]; then
  ( for _ in $(seq 1 60); do
      if curl -s -o /dev/null -m 1 "http://localhost:${PORT}/" 2>/dev/null; then
        open "http://localhost:${PORT}/" 2>/dev/null \
          || xdg-open "http://localhost:${PORT}/" 2>/dev/null || true
        exit 0
      fi
      sleep 0.5
    done ) >/dev/null 2>&1 &
fi

# (bash 3.2 + set -u: expand the array only when non-empty)
exec "$PY" -m simulation_control_center.server ${EXTRA[@]+"${EXTRA[@]}"}

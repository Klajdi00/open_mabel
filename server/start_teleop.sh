#!/usr/bin/env bash
#
# start_teleop.sh — one-command launcher for the MABEL teleop stack on the Mac:
# sim/real robot on one side, any client (iOS, Vision Pro, web, gamepads) on
# the other.
#
# It starts three things and ties their lifecycles together:
#
#   1. sim_camera_server.py   (background)  renders the robot cameras (STEREO head
#                                           by default), streams them on
#                                           udp://<mac>:9091 (low-latency, headset)
#                                           + MJPEG http://<mac>:8080 (fallback/web/iOS)
#   2. view_robot.py          (background)  native desktop window mirroring the
#                                           robot                       (only with --viewer)
#   3. sim_teleop_bridge.py   (FOREGROUND)  physics + IK + hand-tracking WebSocket
#                                           on ws://<mac>:9090
#
# The bridge runs in the FOREGROUND on purpose: that is where you press ENTER to
# (re)anchor the clutch, and a single Ctrl-C there cleanly tears the whole stack
# down. The two background helpers wait for the bridge's published state, so the
# start order does not matter.
#
# Usage:
#   ./start_teleop.sh                        default scene, stereo head, quality 85 (LAN)
#   ./start_teleop.sh --viewer               also open the Mac desktop viewer
#                                            (WARNING: shares the GPU with the camera
#                                             renderer — lowers video fps. Leave OFF for
#                                             lowest-latency / smoothest web teleop.)
#   ./start_teleop.sh --mono                 single head eye (saves render budget)
#   ./start_teleop.sh --head-only            max head fps (no wrists, no stereo)
#   ./start_teleop.sh --real                 stream the REAL USB cameras (3D stereo head
#                                            split+rectified, global-shutter wrist) — not the sim
#   ./start_teleop.sh --quality 60           JPEG quality (default 85, tuned for LAN /
#                                            local network — crisp, low latency. Drop to
#                                            60 for a lossy/remote link: smaller frames =
#                                            fewer UDP packets = loss-robust)
#   (the Command Monitor GUI opens by default, pinned to BACKGROUND QoS / efficiency
#    cores so it never slows the sim — monitor_gui.py, README.md)
#   ./start_teleop.sh --no-monitor           don't open the Command Monitor window
#   ./start_teleop.sh --mock                 synthetic hands (test without a headset)
#   ./start_teleop.sh -s path/to/scene.xml   use a different MuJoCo scene
#
# Override the interpreters / scene without editing the file:
#   MABEL_PY=/path/to/python  MABEL_MJPYTHON=/path/to/mjpython  MABEL_SCENE=... ./start_teleop.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- configuration (env-overridable) ----------------------------------------
PY="${MABEL_PY:-$HOME/anaconda3/bin/python3}"
MJPY="${MABEL_MJPYTHON:-mjpython}"
SCENE="${MABEL_SCENE:-$SCRIPT_DIR/../simulation/mabel_mujoco/scenes/mabel_home.xml}"
HTTP_PORT=8080
WS_PORT=9090
UDP_VIDEO_PORT="${MABEL_UDP_VIDEO_PORT:-9091}"
H264_PORT="${MABEL_SIM_H264_PORT:-9092}"   # H.264 over UDP (LAN, match-source res) +
                                           # /camera/<name>/h264 WebSocket (browsers).
                                           # Same codec the real robot emits, so every
                                           # client decodes one stream sim OR hardware.
JPEG_QUALITY="${MABEL_JPEG_QUALITY:-85}"   # LAN default: 85 = crisp teleop. On a
                                           # clean local network packet loss is ~0,
                                           # so big frames are fine and glass-to-glass
                                           # latency is encode+decode-bound, not
                                           # packet-count-bound. For a lossy/remote
                                           # link pass --quality 60 (smaller frames =
                                           # fewer UDP chunks to lose).
STATE=/tmp/mabel_teleop_state.f64
CAM_LOG=/tmp/mabel_cameras.log
VIEW_LOG=/tmp/mabel_viewer.log
# 96xx read-only data plane (telemetry / pointclouds / SLAM / dedicated video).
# Port contract lives in data_streams.py + docs/data_streams.md; MABEL_DATA_STREAMS=0
# turns the whole plane off (legacy behavior, zero new listeners).
DATA_STREAMS="${MABEL_DATA_STREAMS:-1}"
TELEMETRY_PORT="${MABEL_TELEMETRY_PORT:-9600}"
CLOUD_PORTS=(9601 9602 9603 9604)            # head, wrist_l, wrist_r, front
SLAM_STREAM_PORT="${MABEL_SLAM_STREAM_PORT:-9605}"
DEDI_VIDEO_PORTS=(9620 9621 9622 9623)       # head SBS (hi-res), wrist_l, wrist_r, front

VIEWER=0
SIM_BRIDGE=1  # simulation_bridge (the MuJoCo HAL) owns physics; the teleop
              # server runs as its gateway (sim:bridge) — the exact mirror of
              # the real robot's hardware_bridge split. --legacy-physics
              # restores the old in-process mj_step.
TUI_BG=0      # --tui-bg (run.sh's cockpit mode): run the BRIDGE in the
              # background too — stdin from $BRIDGE_FIFO (the cockpit forwards
              # ENTER/R through it), console to $BRIDGE_CONSOLE_LOG — and hold
              # this script on `wait` so its cleanup trap still owns teardown.
              # The terminal itself then belongs to run.sh's cockpit TUI.
BRIDGE_FIFO="${MABEL_BRIDGE_FIFO:-/tmp/mabel_bridge_stdin.fifo}"
BRIDGE_CONSOLE_LOG=/tmp/mabel_bridge_console.log
HEAD_ONLY=0
MOCK=0
REAL_CAMS=0  # --real: serve the operator's USB cameras (real_camera_server.py) instead
             # of the rendered sim cameras. Head = stitched 3D stereo cam (split + rectified);
             # wrist = global-shutter cam. The bridge still runs teleop/IK on the sim robot.
STEREO=1     # both head eyes by default — the Vision Pro wrap needs them for 3D
MONITOR=1    # Command Monitor GUI ON by default, but launched at BACKGROUND QoS (efficiency
             # cores / low priority) so it never steals CPU from the sim; --no-monitor to skip
MON_LOG=/tmp/mabel_monitor.log
# Optional access code. Unset (default) = open, which is what every LAN client
# expects. Set MABEL_TOKEN=<your-code> to make the bridge require ?key=<code>
# (e.g. when the robot is reachable over a shared tailnet).
APP_TOKEN="${MABEL_TOKEN:-}"

usage() { sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'; }

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scene)  SCENE="$2"; shift 2 ;;
    --viewer)    VIEWER=1; shift ;;
    --head-only) HEAD_ONLY=1; STEREO=0; shift ;;
    --mono)      STEREO=0; shift ;;
    --real|--real-cameras) REAL_CAMS=1; shift ;;
    --quality)   JPEG_QUALITY="$2"; shift 2 ;;
    --monitor)   MONITOR=1; shift ;;
    --no-monitor) MONITOR=0; shift ;;
    --mock)      MOCK=1; shift ;;
    --tui-bg)    TUI_BG=1; shift ;;            # cockpit mode: bridge bg + stdin FIFO
    --legacy-physics) SIM_BRIDGE=0; shift ;;   # old in-process mj_step (no HAL)
    -h|--help)   usage; exit 0 ;;
    *) echo "unknown arg: $1"; echo; usage; exit 1 ;;
  esac
done

# resolve scene to an absolute path
case "$SCENE" in
  /*) ;;
  *)  SCENE="$(cd "$(dirname "$SCENE")" 2>/dev/null && pwd)/$(basename "$SCENE")" ;;
esac

# ---- preflight checks -------------------------------------------------------
[[ -x "$PY" ]]                || { echo "ERROR: Python not found/executable at '$PY'. Set MABEL_PY=…"; exit 1; }
[[ -f "$SCRIPT_DIR/sim_teleop_bridge.py" ]] || { echo "ERROR: run this from the server/ dir."; exit 1; }
[[ -f "$SCENE" ]]             || { echo "ERROR: scene not found: $SCENE"; exit 1; }
if [[ $VIEWER -eq 1 ]] && ! command -v "$MJPY" >/dev/null 2>&1; then
  echo "ERROR: '$MJPY' not found — needed for --viewer (the native window needs mjpython)."
  echo "       Run without --viewer, or set MABEL_MJPYTHON=/path/to/mjpython."
  exit 1
fi

# Pretty status lines (dim arrow prefix, matching the Python servers' look).
say() { printf '  \033[2m▸\033[0m %s\n' "$*"; }

# ---- clear any previous run (avoids 'port in use') --------------------------
say "clearing any previous run…"
# Kill any OLD launcher shell FIRST, with SIGKILL, so its EXIT trap never runs:
# the trap pkills the servers BY NAME ~0.4 s after its foreground bridge dies,
# which would take down the processes THIS run is about to start. ($$ is this
# script, $PPID the run.sh that called it — both excluded.)
for _p in $(pgrep -f 'bash .*start_teleop\.sh' 2>/dev/null); do
  [[ "$_p" == "$$" || "$_p" == "$PPID" ]] && continue
  kill -9 "$_p" >/dev/null 2>&1 || true
done
pkill -9 -f sim_camera_server  >/dev/null 2>&1 || true
pkill -9 -f real_camera_server >/dev/null 2>&1 || true
pkill -9 -f sim_teleop_bridge >/dev/null 2>&1 || true
pkill -9 -f simulation_bridge >/dev/null 2>&1 || true
pkill -9 -f view_robot        >/dev/null 2>&1 || true
pkill -9 -f monitor_gui       >/dev/null 2>&1 || true
# Helpers a previous run leaves behind when its shell died uncleanly (SIGKILL →
# no EXIT trap): stale dns-sd ads duplicate the Bonjour record (32 once piled
# up).
pkill -9 -f "dns-sd -R MABEL Sim" >/dev/null 2>&1 || true
# Free ONLY the LISTENERS we are about to (re)bind — never a client socket.
# `lsof -ti :PORT` lists every process with a socket on that port, INCLUDING
# clients: the studio connects to :8080 (camera proxy) and :9090 (telemetry) as a
# client, so a bare port-kill would take the studio down with us on every scene
# switch. `-sTCP:LISTEN` restricts the kill to the actual servers.
_free_listener() { lsof -ti :"$1" -sTCP:LISTEN 2>/dev/null | xargs kill -9 >/dev/null 2>&1 || true; }
_free_listener "$WS_PORT"                       # the sim teleop bridge (we own it)
_free_listener "${MABEL_SIM_BRIDGE_PORT:-8788}" # the simulation_bridge RPC
if [[ "${MABEL_NO_CAMERAS:-0}" == "1" ]]; then
  # Unity-render mode: the Python camera server is OFF and :8080 is owned by the
  # PERSISTENT Unity render (survives scene switches for instant re-attach) — do
  # NOT free it. UDP :9091 is unused here too.
  say "unity-render mode: leaving :$HTTP_PORT (Unity render) untouched"
else
  _free_listener "$HTTP_PORT"                   # the MJPEG camera server
  lsof -ti udp:"$UDP_VIDEO_PORT" 2>/dev/null | xargs kill -9 >/dev/null 2>&1 || true
  lsof -ti udp:"$H264_PORT"      2>/dev/null | xargs kill -9 >/dev/null 2>&1 || true
fi
if [[ "$DATA_STREAMS" == "1" ]]; then
  # 96xx data plane: telemetry + SLAM live in the bridge, clouds + dedicated
  # video in the camera server — free them all (listeners only, like :9090).
  _free_listener "$TELEMETRY_PORT"
  _free_listener "$SLAM_STREAM_PORT"
  for _p in "${CLOUD_PORTS[@]}" "${DEDI_VIDEO_PORTS[@]}"; do _free_listener "$_p"; done
fi
rm -f "$STATE" "$STATE.scene"
sleep 1

# Verify the ports actually came free — never start a new stack that silently
# loses its bind while an old server keeps streaming underneath it. kill -9
# releases a listener instantly, so the first check normally passes.
_wait_free() {  # _wait_free PORT LABEL
  for _i in $(seq 1 20); do
    [[ -z "$(lsof -ti :"$1" -sTCP:LISTEN 2>/dev/null)" ]] && return 0
    sleep 0.25
  done
  echo "ERROR: port :$1 ($2) is still held by:"
  lsof -i :"$1" -sTCP:LISTEN 2>/dev/null | sed 's/^/    /'
  echo "       kill that process and re-run."
  exit 1
}
_wait_free "$WS_PORT" "teleop bridge"
_wait_free "${MABEL_SIM_BRIDGE_PORT:-8788}" "simulation_bridge RPC"
[[ "${MABEL_NO_CAMERAS:-0}" == "1" ]] || _wait_free "$HTTP_PORT" "camera server"

# ---- lifecycle: one Ctrl-C tears everything down ----------------------------
PIDS=()
_cleaned=0
cleanup() {
  [[ $_cleaned -eq 1 ]] && return; _cleaned=1
  echo; say "shutting down…"
  for p in "${PIDS[@]:-}"; do kill "$p" >/dev/null 2>&1 || true; done
  sleep 0.4
  pkill -9 -f sim_camera_server  >/dev/null 2>&1 || true
  pkill -9 -f real_camera_server >/dev/null 2>&1 || true
  pkill -9 -f sim_teleop_bridge >/dev/null 2>&1 || true
  pkill -9 -f simulation_bridge >/dev/null 2>&1 || true
  pkill -9 -f view_robot        >/dev/null 2>&1 || true
  pkill -9 -f monitor_gui       >/dev/null 2>&1 || true
  pkill -9 -f "dns-sd -R MABEL Sim" >/dev/null 2>&1 || true
  rm -f "$STATE" "$STATE.scene"
  [[ $TUI_BG -eq 1 ]] && rm -f "$BRIDGE_FIFO"
  say "done."
}
trap cleanup EXIT INT TERM

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '127.0.0.1')"

# The 96xx data plane is read by BOTH child processes (bridge: telemetry+SLAM;
# camera server: clouds + dedicated video) — export the gate so the default-on
# behavior of this launcher reaches them even when the caller never set it.
export MABEL_DATA_STREAMS="$DATA_STREAMS"

say "scene   $SCENE"
say "python  $PY"
if [[ "$DATA_STREAMS" == "1" ]]; then
  say "data plane     :$TELEMETRY_PORT telemetry · :${CLOUD_PORTS[0]}-${CLOUD_PORTS[3]} pointclouds · :$SLAM_STREAM_PORT slam/map · :${DEDI_VIDEO_PORTS[0]}-${DEDI_VIDEO_PORTS[3]} video (head hi-res SBS)   [MABEL_DATA_STREAMS=0 to disable]"
else
  say "data plane     OFF (MABEL_DATA_STREAMS=0)"
fi

# ---- firewall preflight -----------------------------------------------------
# The #1 cause of "video works but the headset says disconnected": the macOS
# application firewall blocks INCOMING connections to the bridge (:9090) for the
# Python binary, while loopback (127.0.0.1) and already-allowed paths still get
# through. Detect it and offer the one-time fix so discovery + control work.
FW_STATE="$(defaults read /Library/Preferences/com.apple.alf globalstate 2>/dev/null || echo 0)"
if [[ "$FW_STATE" != "0" ]]; then
  SFW=/usr/libexec/ApplicationFirewall/socketfilterfw
  BLOCKED=0
  if [[ -x "$SFW" ]]; then
    "$SFW" --getappblocked "$PY" 2>/dev/null | grep -qi "blocked" && BLOCKED=1
  fi
  if [[ $BLOCKED -eq 1 ]] || [[ "$FW_STATE" == "2" ]]; then
    printf '\n  \033[33m⚠ macOS firewall is ON and may block the teleop port (:9090).\033[0m\n'
    say "If the headset gets VIDEO but says DISCONNECTED, allow Python once:"
    say "    sudo $SFW --add \"$PY\""
    say "    sudo $SFW --unblockapp \"$PY\""
    say "(or System Settings ▸ Network ▸ Firewall ▸ Options ▸ allow python3)"
    # Try to fix it automatically if we already have sudo (no password prompt).
    if sudo -n true 2>/dev/null && [[ -x "$SFW" ]]; then
      sudo -n "$SFW" --add "$PY" >/dev/null 2>&1 || true
      sudo -n "$SFW" --unblockapp "$PY" >/dev/null 2>&1 || true
      say "auto-allowed Python through the firewall (had sudo)."
    fi
    echo
  fi
fi

# ---- camera-server scheduling priority --------------------------------------
# The camera box renders one GPU thread PER camera (stereo head + both wrists),
# each a ~41 ms GPU-readback stall. At DEFAULT priority those threads contend with
# the bridge's real-time physics loop and drag it BELOW real-time — which is exactly
# what makes the robot's joints lag and look out of sync on every client. Measured
# on an M1 Max (mabel_home, stereo + 2 wrists streaming, 2 wrist-teleop clients):
#
#     camera priority            bridge loop Hz     video fps
#     default (the old behaviour)   ~95  (dips 81)    ~71
#     taskpolicy -c utility         ~162 (159-180)    ~70      ← chosen
#     taskpolicy -b (E-cores)       ~176             ~0        (starves video)
#
# `utility` QoS keeps the cameras on the PERFORMANCE cores (full video fps) but tells
# the scheduler to favour the latency-critical bridge — so the control loop stays
# real-time AND video is unaffected. We deliberately do NOT use background QoS
# (`taskpolicy -b`, the monitor's launcher) here: it pins to the efficiency cores and
# collapses video to ~0 fps. `nice` is a weak fallback (macOS barely honours it).
if command -v taskpolicy >/dev/null 2>&1; then
  CAM_PRIO=(taskpolicy -c utility)
elif command -v nice >/dev/null 2>&1; then
  CAM_PRIO=(nice -n 5)
else
  CAM_PRIO=()
fi

# ---- 0.9) simulation_bridge — the MuJoCo HAL (owns ALL physics) -------------
# The one process that talks to the simulated robot (mj_step), mirroring
# hardware_bridge on the real robot. It publishes [t,qpos,qvel,ctrl] to the
# state memmap (cameras/viewer/mirrors read it) and takes commands over its
# ROS-free RPC; the teleop server below runs as its GATEWAY (sim:bridge).
if [[ ${SIM_BRIDGE:-1} -eq 1 ]]; then
  SIMBRIDGE_LOG="${TMPDIR:-/tmp}/mabel_simulation_bridge.log"
  say "simulation_bridge  MuJoCo HAL (single physics owner) → rpc :${MABEL_SIM_BRIDGE_PORT:-8788}   (log: $SIMBRIDGE_LOG)"
  ( cd "$SCRIPT_DIR/.." && exec "$PY" -u -m simulation_bridge.simulation_bridge \
      --scene "$SCENE" ) >"$SIMBRIDGE_LOG" 2>&1 &
  PIDS+=($!)
  # wait for the bridge's state memmap so every consumer attaches first try.
  # 60 s window: under heavy host load (parallel studios/simulators) the cold
  # import + model load alone can take 20 s+ — the loop breaks on first success.
  for _i in $(seq 1 120); do [[ -s "$STATE.scene" ]] && break; sleep 0.5; done
  [[ -s "$STATE.scene" ]] || { echo "ERROR: simulation_bridge never published state — $SIMBRIDGE_LOG:"; tail -5 "$SIMBRIDGE_LOG"; exit 1; }
  export MABEL_DISPATCH_TARGET="${MABEL_DISPATCH_TARGET:-sim:bridge}"
fi

# ---- 1) camera server (background) ------------------------------------------
# MABEL_NO_CAMERAS=1 skips the Python camera server so the Unity render server can
# own :$HTTP_PORT (photoreal drop-in — same /camera/<name>/stream.mjpg paths).
if [[ "${MABEL_NO_CAMERAS:-0}" == "1" ]]; then
  say "camera server  SKIPPED (MABEL_NO_CAMERAS=1) — Unity render server serves :$HTTP_PORT"
elif [[ $REAL_CAMS -eq 1 ]]; then
  # Real USB cameras: head = stitched 3D stereo (auto-split + rectified per the
  # ros calibration), wrist = global-shutter cam. Same MJPEG/UDP paths the
  # Vision Pro already pulls (main = left eye, head_right = right eye, wrist_*),
  # so the headset shows real stereo with no app change.
  CAM_FLAGS=(--jpeg-quality "$JPEG_QUALITY" --udp-video-port "$UDP_VIDEO_PORT"
             --h264-port "$H264_PORT")
  say "REAL cameras   USB head (stereo, split+rectified) + wrist  → http://$LAN_IP:$HTTP_PORT + udp://$LAN_IP:$UDP_VIDEO_PORT   (log: $CAM_LOG)"
  "${CAM_PRIO[@]}" "$PY" real_camera_server.py "${CAM_FLAGS[@]}" >"$CAM_LOG" 2>&1 &
  PIDS+=($!)
else
  CAM_FLAGS=(--scene "$SCENE" --jpeg-quality "$JPEG_QUALITY"
             --udp-video-port "$UDP_VIDEO_PORT" --h264-port "$H264_PORT")
  [[ $HEAD_ONLY -eq 1 ]] && CAM_FLAGS+=(--head-only)
  [[ $STEREO -eq 1 ]] && CAM_FLAGS+=(--stereo)
  say "camera server  → udp://$LAN_IP:$UDP_VIDEO_PORT (low-latency) + http://$LAN_IP:$HTTP_PORT (fallback)   (log: $CAM_LOG)"
  [[ $STEREO -eq 1 ]] && say "stereo head    both eyes rendered — true 3D in the Vision Pro wrap"
  "${CAM_PRIO[@]}" "$PY" sim_camera_server.py "${CAM_FLAGS[@]}" >"$CAM_LOG" 2>&1 &
  PIDS+=($!)
fi

# ---- 2) desktop viewer (background, optional) -------------------------------
if [[ $VIEWER -eq 1 ]]; then
  say "desktop viewer  (steals ~25% of camera fps; close it while teleoperating)"
  "$MJPY" view_robot.py --scene "$SCENE" >"$VIEW_LOG" 2>&1 &
  PIDS+=($!)
fi

# ---- 2.5) LAN discovery (Bonjour) -------------------------------------------
# The BRIDGE advertises itself (_advertise_bonjour in sim_teleop_bridge.py), on
# every launch path — run.sh, this script, or a direct python launch — so we do
# NOT register a second dns-sd here: duplicate ads for the same service piled
# up (32 once) because a SIGKILLed run never reaps them. The clear-previous-run
# sweep above collects any strays.
say "Bonjour  _mabel-teleop._tcp on :$WS_PORT   (registered by the bridge itself)"

# ---- 2.7) Command Monitor GUI (background, on by default) -------------------
# Read-only desktop window of the live operator-intent stream (monitor_gui.py →
# ws://localhost:9090/monitor). Auto-reconnects, so starting it before the
# bridge is fine — it links up the moment :9090 is live. Skipped with
# --no-monitor, and gracefully disabled if tkinter / a display is unavailable
# (e.g. a headless SSH run) so it never blocks the stack.
#
# CRITICAL for latency: the GUI runs in its OWN process at BACKGROUND QoS, so the
# OS keeps it OFF the performance cores the sim needs. On Apple Silicon
# `taskpolicy -b` pins it to the efficiency cores (perfect — the sim keeps every
# P-core); elsewhere we fall back to `nice` (lowest scheduling priority). Either
# way the monitor only ever uses cycles the sim isn't.
if [[ $MONITOR -eq 1 ]]; then
  if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
    say "NOTE: tkinter not available in '$PY' — Command Monitor disabled (--no-monitor to silence)."
  else
    MON_FLAGS=(--host 127.0.0.1 --port "$WS_PORT")
    [[ -n "$APP_TOKEN" ]] && MON_FLAGS+=(--token "$APP_TOKEN")   # match the bridge's token re-check
    # The monitor auto-discovers other bridges on the LAN via Bonjour/Avahi. For a
    # robot reachable only across a network (e.g. its Tailscale name/IP), list it
    # in MABEL_HOSTS so it always shows up as a host: export MABEL_HOSTS=thor.local
    [[ -n "${MABEL_HOSTS:-}" ]] && MON_FLAGS+=(--hosts "$MABEL_HOSTS")
    # Pick the lowest-impact launcher available on this OS.
    if command -v taskpolicy >/dev/null 2>&1; then
      LOWPRIO=(taskpolicy -b)            # macOS: background QoS → efficiency cores
    elif command -v nice >/dev/null 2>&1; then
      LOWPRIO=(nice -n 19)               # POSIX: lowest scheduling priority
    else
      LOWPRIO=()
    fi
    say "command monitor → desktop window on ws://127.0.0.1:$WS_PORT/monitor  (background QoS · log: $MON_LOG)"
    "${LOWPRIO[@]}" "$PY" monitor_gui.py "${MON_FLAGS[@]}" >"$MON_LOG" 2>&1 &
    PIDS+=($!)
  fi
fi

# ---- 3) bridge (foreground) -------------------------------------------------
BRIDGE_FLAGS=(--scene "$SCENE")
[[ $MOCK -eq 1 ]] && BRIDGE_FLAGS+=(--mock)
# With real cameras, tell the bridge the REAL stereo head's true field of view
# (computed from its calibration) so the hello carries it to the Vision Pro and
# the immersive panel subtends the real FOV → the world is at real-life scale.
if [[ $REAL_CAMS -eq 1 ]]; then
  HEAD_FOV_KV="$("$PY" -c "import real_cameras as rc; f=rc.head_fov(); print(f'{f[0]:.3f} {f[1]:.5f}') if f else None" 2>/dev/null)"
  if [[ -n "$HEAD_FOV_KV" ]]; then
    export MABEL_HEAD_FOV="${HEAD_FOV_KV%% *}"
    export MABEL_HEAD_ASPECT="${HEAD_FOV_KV##* }"
    say "head FOV       fovy=${MABEL_HEAD_FOV}° aspect=${MABEL_HEAD_ASPECT} → Vision Pro panel (real-life scale)"
  fi
fi
# Token policy: enforce only when the operator set MABEL_TOKEN explicitly.
if [[ -n "$APP_TOKEN" ]]; then
  export MABEL_TOKEN="$APP_TOKEN"
else
  unset MABEL_TOKEN
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
  MABEL teleop is starting — connect any client:
    • same Wi-Fi         apps auto-discover via Bonjour (or Host = $LAN_IP)
    • Tailscale          same as Wi-Fi, using this machine's 100.x address
    • web console        http://$LAN_IP:$HTTP_PORT/console
    • video check        http://$LAN_IP:$HTTP_PORT   (any browser)
    • monitor window  $([[ $MONITOR -eq 1 ]] && echo "live command-stream GUI (read-only; --no-monitor to skip)" || echo "off (--monitor to open the command-stream GUI)")
  Video: MJPEG :$HTTP_PORT/camera/<name>/stream.mjpg (per-client transcoded, drop-to-latest,
         low latency) · UDP-JPEG :$UDP_VIDEO_PORT.  H.264 is OFF in the sim
         (videotoolbox adds ~400 ms); MABEL_SIM_H264=1 re-enables.
  Here in this window:
    • ENTER   (re)anchor the clutch once hand tracking is live
    • Ctrl-C  stop EVERYTHING (bridge, cameras$([[ $VIEWER -eq 1 ]] && echo , viewer)$([[ $MONITOR -eq 1 ]] && echo , monitor))
  Camera log:  tail -f $CAM_LOG
────────────────────────────────────────────────────────────────────────────

EOF

if [[ $TUI_BG -eq 1 ]]; then
  # Cockpit mode (run.sh --tui): the bridge runs in the BACKGROUND with its
  # stdin on a FIFO — the cockpit forwards ENTER (clutch re-anchor) and R
  # (sim reset) through it — and its console in a log the cockpit tails. The
  # FIFO is held open READ-WRITE by this shell (fd 8) so the bridge's stdin
  # reader never sees EOF between cockpit writes. `wait` keeps this script
  # alive as the lifecycle owner: killing it still fires the cleanup trap.
  rm -f "$BRIDGE_FIFO"; mkfifo "$BRIDGE_FIFO" 2>/dev/null \
    || { echo "ERROR: cannot create $BRIDGE_FIFO"; exit 1; }
  exec 8<>"$BRIDGE_FIFO"
  : > "$BRIDGE_CONSOLE_LOG"
  say "bridge (bg)   console → $BRIDGE_CONSOLE_LOG · stdin ← $BRIDGE_FIFO (cockpit ENTER/R)"
  "$PY" sim_teleop_bridge.py "${BRIDGE_FLAGS[@]}" <&8 >"$BRIDGE_CONSOLE_LOG" 2>&1 &
  BRIDGE_PID=$!
  PIDS+=($BRIDGE_PID)
  wait "$BRIDGE_PID"
else
  "$PY" sim_teleop_bridge.py "${BRIDGE_FLAGS[@]}"
fi
# when the bridge exits (Ctrl-C), the EXIT trap cleans up the background helpers.

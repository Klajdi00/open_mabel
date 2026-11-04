#!/usr/bin/env bash
#
# MABEL/server/run.sh — one command to bring the MuJoCo teleop stack up so any
# client (iOS app, Vision Pro, web dashboard, future gamepads/joysticks) can
# drive the simulated robot and watch its cameras.
#
# It launches the teleop server (camera MJPEG on :8080 + teleop WebSocket on
# :9090) that lives right here in this folder, then prints the exact host to
# type into the apps. The SAME server serves every client at once.
#
# It ALSO brings up the Simulation Control Center web studio on :8097 (attach
# mode — this script keeps owning the sim; the studio is the free control GUI
# on top of it). --no-studio skips it; for the studio-supervised experience
# (Scene ▾ switching restarts the sim) run
# web_gui/simulation_control_center/run.sh instead.
#
# Usage:
#   ./run.sh                 # default home scene + control studio :8097
#                            #   (interactive terminal → the retro COCKPIT TUI:
#                            #    pixel MABEL + joints/radar/logs; q quits all)
#   ./run.sh --no-tui        # plain scrolling output (no cockpit); also auto
#                            #   when there is no TTY (CI, piped, launchd)
#   ./run.sh --viewer        # also open the Mac desktop MuJoCo window
#   ./run.sh --no-studio     # teleop stack only (skip the web control GUI)
#   ./run.sh --test          # WBC/retargeting sandbox: 120x120 m open field, a
#                            #   manip table dead ahead with graspable objects
#                            #   (mugs/bowl/teapot/cereal + cube/can/ball, grasp-
#                            #   tuned friction) + the MuJoCo viewer auto-opens
#   ./run.sh -s path/to/scene.xml
#
# Requirements: MuJoCo in your Python (the script defaults to
# ~/anaconda3/bin/python3 — override with MABEL_PY=/path/to/python).
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

[[ -f "$HERE/start_teleop.sh" ]] || { echo "ERROR: start_teleop.sh not found in $HERE"; exit 1; }

# ── args: strip --no-studio/--test/--no-tui (ours); the rest passes through ───
STUDIO=1; ARGS=(); TEST=0; HAS_SCENE=0; TUI="${MABEL_TUI:-1}"
for a in "$@"; do
  case "$a" in
    --no-studio) STUDIO=0 ;;
    --test)      TEST=1 ;;
    --no-tui|--plain) TUI=0 ;;
    --mock)      TUI=0; ARGS+=("$a") ;;   # mock = "no clients": the bridge serves
                                          # NO network (no :9090/:9600), so the
                                          # cockpit would wait forever — plain out.
    -s|--scene)  HAS_SCENE=1; ARGS+=("$a") ;;
    *)           ARGS+=("$a") ;;
  esac
done

# ── the retro cockpit TUI (tui/mabel_cockpit.py) ──────────────────────────────
# Default when this terminal is interactive: the stack runs in the background
# and the terminal becomes the arcade-style live cockpit (pixel MABEL, joints,
# radar, logs; q quits everything). --no-tui / MABEL_TUI=0 restores the plain
# scrolling output. Headless (no TTY: CI, ssh pipes, launchd) auto-disables.
TUI_PY="${MABEL_PY:-$HOME/anaconda3/bin/python3}"
[[ -x "$TUI_PY" ]] || TUI_PY="$(command -v python3 || true)"
if [[ "$TUI" = 1 ]]; then
  { [[ -t 0 && -t 1 ]] && [[ -n "$TUI_PY" ]] \
      && "$TUI_PY" -c 'import curses' 2>/dev/null; } || TUI=0
fi

# ── --test: whole-body-control / retargeting sandbox ──────────────────────────
# Spawn the robot on a 120x120 m open grid field (scenes/mabel_open_field.xml —
# directional sun + skybox so the cameras never run out of lit floor at a
# boundary) with ONE manipulation table dead ahead carrying grasp-tuned objects
# (drive the base ~20 cm forward to reach them), and pop the native MuJoCo
# viewer so you can watch the body from outside while teleoperating.
# start_teleop.sh doesn't know --test, so expand it here. An explicit
# -s/--scene still wins over the test scene.
if [[ "$TEST" = 1 ]]; then
  TEST_SCENE="$ROOT/simulation/mabel_mujoco/scenes/mabel_open_field.xml"
  if [[ "$HAS_SCENE" = 0 ]]; then
    [[ -f "$TEST_SCENE" ]] || { echo "ERROR: test scene not found: $TEST_SCENE"; exit 1; }
    ARGS+=(-s "$TEST_SCENE")
  fi
  ARGS+=(--viewer)
fi

STUDIO_SH="$ROOT/web_gui/simulation_control_center/run.sh"
STUDIO_PORT="${MABEL_SCC_PORT:-8097}"
STUDIO_LOG="/tmp/mabel_sim_studio.log"
if [[ "$STUDIO" = 1 && ! -x "$STUDIO_SH" ]]; then
  echo "WARN: sim control studio launcher missing at $STUDIO_SH — continuing without the web GUI."
  STUDIO=0
fi

LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo '127.0.0.1')"
cat <<EOF

  ──────────────────────────────────────────────────────────────────────────
    M A B E L   T E L E O P   S E R V E R
    one server, every client — iOS · Vision Pro · web · gamepads (soon)
  ──────────────────────────────────────────────────────────────────────────

    endpoints
      teleop    ws://${LAN_IP}:9090/teleop
      video     udp://${LAN_IP}:9092  (H.264, LAN: crisp + low-latency)
                ws://${LAN_IP}:8080/camera/<name>/h264  (H.264, browsers)
                udp://${LAN_IP}:9091  (UDP-JPEG fallback) · http://${LAN_IP}:8080/camera/*  (MJPEG fallback)
      console   http://${LAN_IP}:8080/console
$( [[ "$STUDIO" = 1 ]] && echo "      studio    http://${LAN_IP}:${STUDIO_PORT}  (Sim Control Center — full web control GUI; log: ${STUDIO_LOG})" \
                        || echo "      studio    off (--no-studio)" )

    connect
      same Wi-Fi         apps auto-discover (Bonjour); or Host = ${LAN_IP}
      Tailscale          Host = this Mac's Tailscale IP (100.x), same protocol
      browser            open the console URL above

    Ctrl-C here stops everything.
  ──────────────────────────────────────────────────────────────────────────

EOF
if [[ "$TEST" = 1 ]]; then
  echo "  ▸ TEST MODE: 120x120 m open field + manip table with graspable objects"
  echo "    (mabel_open_field.xml) — drive ~20 cm forward to reach them; free space"
  echo "    for whole-body-control / retargeting; the MuJoCo viewer opens itself."
  echo
fi

# Sim Control Center studio (:8097) in ATTACH mode — --gui-only means the studio
# does NOT supervise the sim (this script owns it via start_teleop.sh); it kills
# any stale studio squatting on its port, then serves the GUI and reconnects to
# the bridge/cameras as they come up. Backgrounded; reaped when this script exits.
STUDIO_PID=""
if [[ "$STUDIO" = 1 ]]; then
  MABEL_NO_BROWSER=1 "$STUDIO_SH" --gui-only >"$STUDIO_LOG" 2>&1 &   # this script pops the tab itself
  STUDIO_PID=$!
  trap '[[ -n "$STUDIO_PID" ]] && kill "$STUDIO_PID" 2>/dev/null' EXIT
  # pop the studio in the default browser as soon as it answers (non-blocking;
  # MABEL_NO_BROWSER=1 suppresses — e.g. headless/SSH launches)
  if [[ "${MABEL_NO_BROWSER:-0}" != 1 ]]; then
    ( for _ in $(seq 1 60); do
        if curl -s -o /dev/null -m 1 "http://localhost:${STUDIO_PORT}/" 2>/dev/null; then
          open "http://localhost:${STUDIO_PORT}/" 2>/dev/null \
            || xdg-open "http://localhost:${STUDIO_PORT}/" 2>/dev/null || true
          exit 0
        fi
        sleep 0.5
      done ) >/dev/null 2>&1 &
  fi
fi

# start_teleop.sh handles the Bonjour advertisement (so a direct launch is
# discoverable too) and tears it down on exit.
if [[ "$TUI" = 1 ]]; then
  # ── cockpit mode: stack in the background, this terminal IS the cockpit ─────
  LAUNCH_LOG=/tmp/mabel_launch.log
  : > "$LAUNCH_LOG"
  echo "  ▸ COCKPIT MODE — the stack starts in the background; this terminal"
  echo "    becomes the live cockpit (q quits everything · --no-tui for plain)."
  "$HERE/start_teleop.sh" --tui-bg ${ARGS[@]+"${ARGS[@]}"} >"$LAUNCH_LOG" 2>&1 &
  TELEOP_PID=$!
  # bash runs an INT trap only after the foreground child exits — so Ctrl-C
  # lands in the cockpit (clean exit) and teardown below still runs.
  # Log paths go via ENV, not argv: start_teleop's clear-previous-run sweep
  # pkill -f's names like `simulation_bridge`, and an argv containing
  # …/mabel_simulation_bridge.log would get the cockpit itself SIGKILLed.
  trap ':' INT
  MABEL_COCKPIT_LOGS="launch=$LAUNCH_LOG,bridge=/tmp/mabel_bridge_console.log,physics=${TMPDIR:-/tmp}/mabel_simulation_bridge.log,cameras=/tmp/mabel_cameras.log,studio=$STUDIO_LOG" \
    "$TUI_PY" "$HERE/tui/mabel_cockpit.py" --mode sim \
      --fifo "${MABEL_BRIDGE_FIFO:-/tmp/mabel_bridge_stdin.fifo}"
  trap - INT
  echo "  ▸ cockpit closed — shutting the whole stack down…"
  kill -TERM "$TELEOP_PID" 2>/dev/null                # its trap reaps the stack
  for _ in $(seq 1 40); do kill -0 "$TELEOP_PID" 2>/dev/null || break; sleep 0.25; done
  kill -9 "$TELEOP_PID" 2>/dev/null
  # After a studio Scene ▾ switch the sim runs under the STUDIO's supervisor,
  # not our start_teleop — sweep it too (same set the stay-resident path kills).
  [[ -n "$STUDIO_PID" ]] && kill "$STUDIO_PID" 2>/dev/null
  pkill -9 -f sim_teleop_bridge   >/dev/null 2>&1 || true
  pkill -9 -f 'simulation_bridge.simulation_bridge' >/dev/null 2>&1 || true
  pkill -9 -f sim_camera_server   >/dev/null 2>&1 || true
  trap - EXIT
  echo "  ▸ done."
  exit 0
fi
"$HERE/start_teleop.sh" ${ARGS[@]+"${ARGS[@]}"}
TELEOP_RC=$?

# A SCENE SWITCH relaunches the sim stack UNDER THE STUDIO's supervisor — which
# pkills OUR foreground start_teleop above. Without this guard, run.sh then
# exited and its EXIT trap killed the studio too, so scene switching and the
# MuJoCo viewer died after the FIRST switch. If the studio is alive and now
# MANAGES a sim, stay resident instead: Ctrl-C here still stops everything.
if [[ -n "$STUDIO_PID" ]] && kill -0 "$STUDIO_PID" 2>/dev/null; then
  if curl -s -m 2 "http://localhost:${STUDIO_PORT}/api/sim/status" 2>/dev/null \
      | grep -q '"managed": *true'; then
    echo "▸ scene switch detected: the control studio now owns the sim stack —"
    echo "  run.sh stays resident so Ctrl-C still stops EVERYTHING."
    trap - EXIT
    stop_all() {
      kill "$STUDIO_PID" 2>/dev/null
      pkill -9 -f sim_teleop_bridge   >/dev/null 2>&1 || true
      pkill -9 -f 'simulation_bridge.simulation_bridge' >/dev/null 2>&1 || true
      pkill -9 -f sim_camera_server   >/dev/null 2>&1 || true
      exit 0
    }
    trap stop_all INT TERM
    wait "$STUDIO_PID"
  fi
fi
exit $TELEOP_RC

#!/usr/bin/env bash
# MABEL Control Lab — the mabel-library test bench (:8115).
#   ./run.sh                # empty-field scene, all benches
#   ./run.sh -s <scene>     # another scenes/*.xml
# Vision Pro teleop binds :9090 when enabled from the GUI — stop the main
# sim bridge (start_teleop.sh) first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

VIEWER=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scene) export MABEL_CTS_SCENE="$2"; shift 2;;
    --no-viewer) VIEWER=0; shift;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;;
    *) echo "unknown flag $1"; exit 2;;
  esac
done

PORT="${MABEL_CTS_PORT:-8115}"
# free the studio port (never touch 9090 — the user may run the bridge)
if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
  lsof -ti tcp:"$PORT" | xargs kill 2>/dev/null || true; sleep 0.5
fi
pkill -f "control_lab.server" 2>/dev/null || true

PY="python3"
if [ -x "$HOME/anaconda3/bin/python3" ]; then PY="$HOME/anaconda3/bin/python3"; fi
"$PY" -c "import fastapi, uvicorn, mujoco" || {
  echo "need fastapi+uvicorn+mujoco in $PY"; exit 1; }

echo "──────────────────────────────────────────────"
echo "  MABEL · Control Lab   http://localhost:$PORT"
echo "  benches: motion · controllers · wrist IK · retarget (hand / whole-body)"
echo "──────────────────────────────────────────────"
cd "$REPO/web_gui"

# pop the studio in the default browser as soon as it answers (non-blocking;
# MABEL_NO_BROWSER=1 suppresses — e.g. headless/SSH launches). Backgrounded
# BEFORE the exec below, which replaces this shell.
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

# native MuJoCo viewer mirror (macOS passive viewer needs mjpython)
if [ "$VIEWER" = 1 ]; then
  if command -v mjpython >/dev/null 2>&1; then
    pkill -f "control_lab.server.viewer_mirror" 2>/dev/null || true
    ( sleep 3; exec mjpython -m control_lab.server.viewer_mirror ) \
      > /tmp/control_lab_viewer.log 2>&1 &
    echo "  + MuJoCo viewer mirror launching (mjpython)…"
  else
    echo "  ! mjpython not found — skipping the native MuJoCo viewer"
  fi
fi

exec "$PY" -m control_lab.server

#!/usr/bin/env bash
# MABEL Retarget Lab — human data + offline retargeting comparison (:8120).
#   ./run.sh               studio on :8120, Vision Pro gateway on :9090 (LAN)
#   ./run.sh --no-browser  don't open the browser (the Mac app uses this)
# The VP gateway takes :9090 — stop the sim/real gateway (start_teleop.sh) first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BROWSER=1
for a in "$@"; do case "$a" in --no-browser) BROWSER=0;; -h|--help) sed -n '2,5p' "$0"; exit 0;; esac; done
PORT="${MABEL_RTL_PORT:-8120}"
if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then lsof -ti tcp:"$PORT" | xargs kill 2>/dev/null || true; sleep 0.5; fi
PY="${MABEL_RTL_PYTHON:-python3}"
if [ -z "${MABEL_RTL_PYTHON:-}" ] && [ -x "$HOME/anaconda3/bin/python3" ]; then PY="$HOME/anaconda3/bin/python3"; fi
"$PY" -c "import fastapi, uvicorn, mujoco, websockets, mabel" 2>/dev/null || \
"$PY" -c "import sys; sys.path.insert(0, '$REPO/controller'); import fastapi, uvicorn, mujoco, websockets, mabel" || {
  echo "need fastapi uvicorn mujoco websockets + the mabel package in $PY"; exit 1; }
if [ "$BROWSER" = 1 ]; then
  ( for _ in $(seq 1 60); do
      curl -s -o /dev/null -m 1 "http://localhost:$PORT/api/health" && { open "http://localhost:$PORT/"; exit 0; }
      sleep 0.5; done ) >/dev/null 2>&1 &
fi
cd "$REPO/web_gui"
exec "$PY" -m retarget_lab.server

#!/usr/bin/env bash
# MABEL Control Center — rviz-style visualizer + teleop client + server monitor.
# Runs on the robot host (mabel-jetson) next to real_run.sh; browsers anywhere then get the
# full studio. Falls back to a synthetic mock plant when ROS is absent.
#
#   ./control_center/run.sh                 # auto: ROS if available, else mock
#   MABEL_CC_SOURCE=mock ./control_center/run.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# source the ROS env if present so the rclpy tap can attach. $REPO is
# web_gui/ since the folder-by-role reorg, so ros lives one level
# up; keep the old repo-root-relative path as a fallback for stale checkouts.
if [ "${MABEL_CC_SOURCE:-auto}" != "mock" ]; then
  for _env in "$REPO/../ros/scripts/mabel_env.sh" "$REPO/ros/scripts/mabel_env.sh"; do
    [ -f "$_env" ] || continue
    # shellcheck disable=SC1091
    source "$_env" 2>/dev/null || true
    break
  done
fi

# interpreter: explicit override > any python3 with fastapi > mabel conda env
PY="${MABEL_CC_PYTHON:-}"
if [ -z "$PY" ]; then
  if python3 -c 'import fastapi' 2>/dev/null; then PY=python3
  elif [ -x "$HOME/miniconda3/envs/mabel/bin/python" ]; then
    PY="$HOME/miniconda3/envs/mabel/bin/python"
  else
    echo "control_center: no python with fastapi found (pip install fastapi 'uvicorn[standard]' psutil)" >&2
    exit 1
  fi
fi
# Operator credentials (MABEL_CC_USER / MABEL_CC_PASSKEY) come from a machine-local,
# gitignored file so they are NEVER committed (this repo may be made public). Create
# ~/.mabel_cc_env with:  export MABEL_CC_USER=... ; export MABEL_CC_PASSKEY=...
# Without it the login is fail-closed (config defaults are empty).
# shellcheck disable=SC1091
[ -f "$HOME/.mabel_cc_env" ] && source "$HOME/.mabel_cc_env"

# App-level login (styled popup gate) session secret. Persist a machine-local
# secret so operator sessions survive service restarts; never committed to git.
if [ -z "${MABEL_CC_AUTH_SECRET:-}" ]; then
  SECRET_FILE="$HOME/.mabel_cc_secret"
  if [ ! -f "$SECRET_FILE" ]; then
    (umask 077; head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$SECRET_FILE")
  fi
  export MABEL_CC_AUTH_SECRET="$(cat "$SECRET_FILE")"
fi

# pop the studio in the default browser as soon as it answers (non-blocking;
# MABEL_NO_BROWSER=1 suppresses — headless/SSH launches, and real_run.sh sets
# it because it pops the tab itself). Backgrounded BEFORE the exec below.
PORT="${MABEL_CC_PORT:-8094}"
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

cd "$REPO"     # env scripts may chdir; the package import needs the repo root
exec "$PY" -m control_center.server "$@"

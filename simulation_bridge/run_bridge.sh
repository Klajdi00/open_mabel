#!/usr/bin/env bash
# MABEL simulation_bridge — the MuJoCo HAL (mirror of hardware_bridge/run_bridge.sh).
# ROS-free on macOS; on Ubuntu source your ROS env first for the /sim DDS mirror.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${MABEL_PY:-$HOME/anaconda3/bin/python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
cd "$HERE/.."
exec "$PY" -u -m simulation_bridge.simulation_bridge "$@"

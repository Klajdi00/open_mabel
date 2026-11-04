#!/usr/bin/env bash
# Launch the MABEL Motor Calibration GUI.
#   ./run_gui.sh
#   ./run_gui.sh --motor DM-J4340P --id 1 --port /dev/cu.usbmodem1101
# On macOS the serial (DM-USB2FDCAN) backend is the default; on Linux, SocketCAN.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(command -v python || command -v python3)"
exec "$PY" motor_calib_gui.py "$@"

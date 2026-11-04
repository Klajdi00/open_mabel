#!/usr/bin/env bash
# bring_up_cameras.sh -- connect to the depth cameras and visualize them. NO ROS.
#
# Brings up whatever is plugged in (Intel RealSense D435i + Stereolabs ZED-M),
# reads RGB + depth + point cloud + IMU, and shows a terminal dashboard, RGB +
# depth windows, and a live 3D point cloud.
#
#   bash bring_up_cameras.sh                  # all cameras, full viz
#   bash bring_up_cameras.sh --cameras realsense
#   bash bring_up_cameras.sh --no-cloud       # skip the 3D window
#   bash bring_up_cameras.sh --no-window      # headless dashboard only
#
# Any extra args are passed straight to bringup.py (see --help there).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

# 1. first-run setup
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "▸ first run -- setting up dependencies"
  bash "$HERE/install_deps.sh" || exit 1
fi

# 2. show the USB link health (catches the ZED-on-USB-2 gotcha up front)
bash "$HERE/usb_check.sh" || true
echo

# 3. launch the viewer in the venv
# shellcheck disable=SC1091
source "$VENV/bin/activate"
exec python "$HERE/bringup.py" "$@"

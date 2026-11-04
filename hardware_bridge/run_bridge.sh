#!/usr/bin/env bash
# run_bridge.sh — launch the MABEL hardware bridge (the single HAL).
#
# Sources the SAME ROS environment the rest of MABEL runs under so the bridge is on
# the same DDS graph as the ROS bring-up + the app server. Preference order:
#   1. ros/scripts/mabel_env.sh  — the conda RoboStack ROS 2 the whole stack uses
#      (ros2, rclpy, AND the camera SDKs pyrealsense2/pyzed live here). This is what
#      real_run.sh / hardware.launch.py use, so the bridge must match it for DDS interop.
#   2. /opt/ros/jazzy + the colcon overlay — fallback if the conda env is absent.
# Used by the systemd unit and for manual runs.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MABEL_ROOT="${MABEL_ROOT:-$(cd "$HERE/.." && pwd)}"
ENV_SH="${MABEL_ENV_SH:-$MABEL_ROOT/ros/scripts/mabel_env.sh}"

if [ -f "$ENV_SH" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SH"
else
    ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    # shellcheck disable=SC1090
    [ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
    WS_SETUP="${WS_SETUP:-$MABEL_ROOT/ros/install/setup.bash}"
    # shellcheck disable=SC1090
    [ -f "$WS_SETUP" ] && source "$WS_SETUP"
fi

# Match the stack's RMW + domain so the bridge shares its DDS graph (CycloneDDS is the
# MABEL RMW — see slam_env / mabel notes). honor any inherited override.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# RSUSB pyrealsense2 (default ON when the local build exists; MABEL_RS_RSUSB=0 to
# force the pip wheel). WHY: the pip wheel's V4L2/IIO backend cannot see the D435i's
# motion module on the Tegra kernel (no hid-sensor IIO modules -> "HID Motion Sensor
# Failure"), so /camera/front/imu — cuVSLAM's VIO fusion input — stays silent. The
# RSUSB build reads the IMU over libusb directly (verified on thor: motion module
# enumerates, gyro 200 Hz clean stamps). It switches the WHOLE bridge (video too)
# onto the libusb backend — if video ever misbehaves, set MABEL_RS_RSUSB=0 and
# restart the service. Build (once): cd ~/librealsense_rsusb/librealsense/build &&
#   cmake -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) . &&
#   make -j8 pyrealsense2
if [ "${MABEL_RS_RSUSB:-1}" = 1 ]; then
    RSUSB_DIR="${RSUSB_PYREALSENSE2_DIR:-$HOME/librealsense_rsusb/librealsense/build/Release}"
    if compgen -G "$RSUSB_DIR/pyrealsense2*.so" > /dev/null; then
        export PYTHONPATH="$RSUSB_DIR${PYTHONPATH:+:$PYTHONPATH}"
        export LD_LIBRARY_PATH="$RSUSB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[run_bridge] RSUSB pyrealsense2 from $RSUSB_DIR (D435i IMU enabled)"
    else
        echo "[run_bridge] MABEL_RS_RSUSB=1 but no pyrealsense2 build in $RSUSB_DIR — using pip wheel (no IMU)" >&2
    fi
fi

cd "$HERE"

# Optional LOCAL, git-ignored env overrides (MABEL_HW_* etc.) — a sudo-free way to
# retune the systemd-launched bridge without editing the unit. Sourced LAST so it
# wins over the service's Environment= lines. e.g. `export MABEL_HW_BASE=0` here
# parks a subsystem without touching the unit. Delete the file to revert.
[ -f "$HERE/.env.local" ] && source "$HERE/.env.local"

exec "${PYTHON:-python3}" -u hardware_bridge.py "$@"

#!/usr/bin/env bash
# control_center.sh — launch the MABEL Control Center GUI.
#
# Sources the SAME ROS environment as the hardware bridge (run_bridge.sh) so the GUI is
# on the same DDS graph as the bridge and the rest of the stack, then runs the Tk app.
# The GUI talks ONLY to the bridge over DDS — it never opens a device.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="$(cd "$HERE/.." && pwd)"
export MABEL_ROOT="${MABEL_ROOT:-$(cd "$BRIDGE_DIR/.." && pwd)}"
ENV_SH="${MABEL_ENV_SH:-$MABEL_ROOT/ros/scripts/mabel_env.sh}"

if [ -f "$ENV_SH" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SH"
else
    ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    # shellcheck disable=SC1090
    [ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
fi

# Match the stack's RMW + domain so the GUI shares the bridge's DDS graph.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

cd "$BRIDGE_DIR"
exec "${PYTHON:-python3}" -u -m gui "$@"

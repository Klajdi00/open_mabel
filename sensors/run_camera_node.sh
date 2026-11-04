#!/usr/bin/env bash
# run_camera_node.sh -- launch the single camera-hardware node (ros_camera_node.py)
# in the conda `mabel` ROS env, so its /camera/* topics land directly on the same
# conda ROS graph as RViz / nvblox / cuVSLAM (NO cross-env DDS bridge needed -- the
# node runs natively in conda with pyrealsense2 + pyzed installed there).
#
#   ./run_camera_node.sh --ros                    # ROS publish only (bring-up)
#   ./run_camera_node.sh --ros --serve            # ROS + MJPEG to the network
#   ./run_camera_node.sh --serve --cameras zed    # server only, ZED only
#
# The ROS hardware bring-up launches this same node automatically; use this for a
# standalone run / debugging.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$HERE/../ros"

# conda mabel env = ROS 2 Jazzy (RoboStack) + rclpy + pyrealsense2 + pyzed, and it
# pins CycloneDDS / domain 0 so this shares the graph with the rest of the stack.
# shellcheck disable=SC1091
source "$WS/scripts/mabel_env.sh"

exec python "$HERE/ros_camera_node.py" "$@"

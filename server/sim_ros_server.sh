#!/usr/bin/env bash
# =============================================================================
# sim_ros_server.sh — ONE command: MuJoCo ROS sim + visual SLAM + app server.
# =============================================================================
# Brings up, on Ubuntu, everything the iOS app needs against the SIMULATED robot:
#   • mabel_sim         the ROS MuJoCo robot (physics, sensors, wheel /odom)
#   • visual SLAM       cuVSLAM + nvblox  → live 3D mesh + map→odom correction
#   • the app server    teleop control · robot state · control modes · camera
#                       video (MJPEG/UDP) · AND the live mesh+odom on /map
#
# Drive from the iOS app (Drive tab) and watch the map build in Maps ▸ Live SLAM.
#
# In sim this also opens, on this desktop, RViz (robot + sensors + the live nvblox
# SLAM mesh) AND the MuJoCo mirror window (view_robot.py — the robot's TRUE state),
# and prints a ✓/⚠ confirming the mesh + SLAM came up. Use --no-rviz / --no-viewer
# to suppress the windows on a headless box.
#
#   ./sim_ros_server.sh                 # full stack + RViz + MuJoCo window
#   ./sim_ros_server.sh --no-rviz       # ... without RViz
#   ./sim_ros_server.sh --no-viewer     # ... without the MuJoCo window
#   ./sim_ros_server.sh -s scene.xml    # a different MuJoCo scene
#   ./sim_ros_server.sh --no-slam       # sim + server only (no mesh)
#   ./sim_ros_server.sh -h              # all options
#
# Prereqs: the workspace is built (ros/dev_up.sh) and, for visual SLAM, the
# Isaac ROS apt stack is installed (scripts/slam_env.sh / project_isaac_native_jazzy).
# =============================================================================
MODE=sim
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ros_server_common.sh"
ros_server_main "$@"

# MABEL Isaac-SLAM environment -- SOURCE this in the terminal that runs the GPU
# SLAM stack (visual_slam.launch.py = cuVSLAM + nvblox):
#
#     source scripts/slam_env.sh
#     ros2 launch mabel_slam visual_slam.launch.py        # nvblox 3D mesh (+ cuVSLAM)
#     ros2 launch mabel_slam lidar_slam.launch.py          # 2D slam_toolbox map
#
# WHY THIS EXISTS (the conda<->apt split):
#   The sim + the rest of the stack run in the RoboStack *conda* env (mabel_env.sh),
#   which deliberately SCRUBS /opt/ros to stay hermetic. But the NVIDIA Isaac ROS GPU
#   packages -- nvblox_ros, isaac_ros_visual_slam, cuVSLAM -- are NOT available on
#   RoboStack/conda; they ship only as apt binaries under /opt/ros/jazzy. So NO single
#   conda shell can see both `mabel_slam` (conda) AND `nvblox` (apt) at once, and
#   launching SLAM from the conda env silently runs WITHOUT nvblox -> no 3D mesh.
#
#   This script sets up the OTHER side: the apt /opt/ros/jazzy ROS (which HAS nvblox +
#   cuVSLAM) with the conda-built `mabel_slam` package OVERLAID on top (it is pure
#   Python with a /usr/bin/python3 shebang, so it runs fine under apt rclpy). The sim
#   keeps running in its conda terminal; the two ROS graphs talk over DDS because BOTH
#   ends use CycloneDDS on ROS_DOMAIN_ID 0 (forced below). nvblox then subscribes to the
#   sim's camera topics and publishes /nvblox_node/mesh -> mesh_to_pointcloud ->
#   /nvblox_node/mesh_pointcloud, which the sim's RViz (sim_sensors.rviz) already shows.

# --- locate the workspace (this file lives in <ws>/scripts/) -----------------
_MABEL_WS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# --- start from a CLEAN slate: drop any active conda/RoboStack ROS overlay ----
# A conda env on PATH (mabel/base) leaves PYTHONPATH/AMENT_PREFIX_PATH/LD_LIBRARY_PATH
# pointing at RoboStack ROS; mixing those with apt /opt/ros causes ABI/typesupport
# clashes. Wipe them so apt ROS is the ONLY ROS underlay.
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_PACKAGE_PATH LD_LIBRARY_PATH CONDA_PREFIX

# --- apt ROS 2 Jazzy (the one carrying Isaac ROS / nvblox / cuVSLAM) ----------
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "slam_env: /opt/ros/jazzy not found. The Isaac ROS GPU stack (nvblox, cuVSLAM)" >&2
  echo "          installs there via apt -- see project_isaac_native_jazzy." >&2
  return 1 2>/dev/null || exit 1
fi
source /opt/ros/jazzy/setup.bash

# --- overlay the conda-built mabel_slam package (launch files + pure-py nodes) -
# Only mabel_slam is overlaid (NOT the whole conda workspace, whose other pkgs are
# linked against conda libs). mabel_slam's nodes shebang /usr/bin/python3 and use only
# rclpy + numpy, so they run cleanly under apt ROS. nvblox_msgs is on apt, so the
# mesh_to_pointcloud converter imports fine here (it could NOT in the conda env).
_MS="$_MABEL_WS/install/mabel_slam"
if [ -d "$_MS" ]; then
  export AMENT_PREFIX_PATH="$_MS:$AMENT_PREFIX_PATH"
  # Make `import mabel_slam` (sim_detect, the node modules) resolvable. The dir depends
  # on the colcon build mode, so handle all three:
  #   copy install   -> the package sits directly in lib/pythonX/site-packages
  #   symlink install -> only an `mabel-slam.egg-link` is there, pointing at build/ (a
  #                      plain PYTHONPATH entry does NOT activate an egg-link), so add
  #                      the egg-link's TARGET dir instead
  #   neither         -> fall back to the source tree
  _added=""
  for _sp in "$_MS"/lib/python3.*/site-packages; do
    [ -d "$_sp" ] || continue
    if [ -d "$_sp/mabel_slam" ]; then                      # copy install
      export PYTHONPATH="$_sp${PYTHONPATH:+:$PYTHONPATH}"; _added=1
    elif [ -f "$_sp/mabel-slam.egg-link" ]; then           # symlink install
      _tgt="$(head -n1 "$_sp/mabel-slam.egg-link")"
      [ -d "$_tgt/mabel_slam" ] && { export PYTHONPATH="$_tgt${PYTHONPATH:+:$PYTHONPATH}"; _added=1; }
    fi
  done
  if [ -z "$_added" ] && [ -d "$_MABEL_WS/src/mabel_slam/mabel_slam" ]; then
    export PYTHONPATH="$_MABEL_WS/src/mabel_slam${PYTHONPATH:+:$PYTHONPATH}"  # source fallback
  fi
  # mabel_slam is colcon-built under CONDA, so its node executables get a conda
  # python shebang (#!/.../anaconda3/envs/mabel/bin/python3). Run under THIS apt ROS
  # that's an ABI clash: conda python + apt rosidl typesupport -> "undefined symbol
  # ...convert_to_py" and the pure-python nodes (mesh_to_pointcloud) crash on import,
  # so /nvblox_node/mesh_pointcloud never appears. Rewrite their shebangs to the apt
  # /usr/bin/python3 (they only use rclpy + numpy + nvblox_msgs, all on apt).
  # Idempotent + re-applied on every source, so it survives a conda rebuild.
  if [ -d "$_MS/lib/mabel_slam" ]; then
    for _exe in "$_MS"/lib/mabel_slam/*; do
      [ -f "$_exe" ] || continue
      case "$(head -1 "$_exe" 2>/dev/null)" in
        '#!'*conda*|'#!'*miniforge*|'#!'*mambaforge*|'#!'*/envs/*)
          sed -i '1s|.*|#!/usr/bin/python3|' "$_exe" 2>/dev/null || true ;;
      esac
    done
  fi
else
  echo "slam_env: $_MS missing -- build it first (in the conda env):" >&2
  echo "          source scripts/mabel_env.sh && colcon build --packages-select mabel_slam" >&2
fi

# --- DDS: must MATCH the conda sim so the two ROS graphs interoperate ----------
# RoboStack's mabel env defaults to CycloneDDS on domain 0; force the same here, else
# nvblox sees none of the sim's camera/clock topics (FastDDS<->Cyclone don't mix, and a
# different domain is isolated). This is also the host<->container fix in
# project_slam_host_container_dds, applied to the conda<->apt split.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
# LOCALHOST-only discovery: identical to scripts/mabel_env.sh so the apt SLAM and
# the conda env reliably discover each other on this multi-interface box (docker0
# + WiFi otherwise scatter participants -> "ros2 topic list" misses nodes).
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Shared DDS tuning (big socket receive buffers) — the SAME file mabel_env.sh uses,
# so both stacks' sockets stop dropping the heavy point-cloud/mesh traffic.
export CYCLONEDDS_URI="file://$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )/cyclonedds.xml"

echo "slam_env: apt ROS Jazzy + mabel_slam overlay | RMW=$RMW_IMPLEMENTATION DOMAIN=$ROS_DOMAIN_ID DISCOVERY=$ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "          nvblox/cuVSLAM available; talks to the conda sim over DDS."
echo "          sim running?  ros2 launch mabel_slam visual_slam.launch.py use_sim_time:=true"

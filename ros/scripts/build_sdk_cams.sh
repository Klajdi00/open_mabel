#!/usr/bin/env bash
# Build the SDK-bound camera ROS packages (ZED wrapper + cob_srvs) into the conda
# `mabel` workspace. RealSense uses the prebuilt robostack `ros-jazzy-realsense2-
# camera` conda package, so it is NOT built here.
#
# The ZED SDK (4.2.5, /usr/local/zed) requires CUDA 12 -- the box's default
# /usr/local/cuda is 13.0, which the SDK's cmake REJECTS (find_package(CUDA 12)).
# Pin CUDA 12.6 for this build only.
#
#   ./scripts/build_sdk_cams.sh
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$WS/scripts/mabel_env.sh"

# ---------------------------------------------------------------------------
# Patch the ZED wrapper for ROS 2 Jazzy (idempotent).
# The pinned humble-v4.2.5 tag's CMake knows jazzy (-DFOUND_JAZZY) but the C++
# include guard only lists HUMBLE/IRON/FOXY and #error's on jazzy, which skips
# the tf2_geometry_msgs include and cascades into "no matching toMsg()" errors.
# Jazzy uses the same .hpp header as Humble/Iron, so fold it into that branch.
# Re-runnable: a re-clone (vcs import) resets the file, so we re-apply here.
# ---------------------------------------------------------------------------
ZED_CC="$WS/src/zed-ros2-wrapper/zed_components/src/zed_camera/src/zed_camera_component.cpp"
if [ -f "$ZED_CC" ] && grep -q '^#ifdef FOUND_HUMBLE$' "$ZED_CC"; then
  echo "[patch] enabling FOUND_JAZZY in zed_camera_component.cpp"
  perl -0pi -e 's/#ifdef FOUND_HUMBLE\n#include <tf2_geometry_msgs\/tf2_geometry_msgs.hpp>\n#elif defined FOUND_IRON\n#include <tf2_geometry_msgs\/tf2_geometry_msgs.hpp>\n#elif defined FOUND_FOXY/#if defined FOUND_HUMBLE || defined FOUND_IRON || defined FOUND_JAZZY\n#include <tf2_geometry_msgs\/tf2_geometry_msgs.hpp>\n#elif defined FOUND_FOXY/' "$ZED_CC"
fi

# cob_common ships several packages; we only want cob_srvs. Ignore the siblings
# so a full `colcon build` (dev_up.sh) doesn't try to build the ROS1-ish extras.
for d in cob_common cob_msgs cob_actions; do
  p="$WS/src/cob_common/$d"
  [ -d "$p" ] && touch "$p/COLCON_IGNORE"
done

export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd "$WS"
colcon build --symlink-install \
  --packages-select cob_srvs zed_msgs zed_components zed_wrapper zed_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc

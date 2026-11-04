#!/usr/bin/env bash
# =============================================================================
# real_ros_server.sh — DEPRECATED ALIAS for ./real_run.sh
# =============================================================================
# The real-robot app server was MERGED into real_run.sh: that one script now
# brings up the HAL + visual/2D SLAM + Nav2 + RViz AND serves the iOS / Vision Pro
# / web app (teleop + live /map mesh+odom on :9090/:9092 + MJPEG video on :8080).
# So this file is just a thin forwarder to real_run.sh, kept so existing docs /
# muscle-memory / the cross-reaper keep working.
#
#   ./real_ros_server.sh            ==  ./real_run.sh
#   ./real_ros_server.sh --no-slam  ==  ./real_run.sh --no-slam
#
# The few flags that were UNIQUE to the old server are translated below; prefer
# calling real_run.sh directly (run `./real_run.sh -h` for its full flag set).
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '  \033[33m⚠ real_ros_server.sh is now an alias for real_run.sh (merged). Forwarding…\033[0m\n' >&2

# Translate the old server-only flags to real_run.sh's. Shared flags
# (--no-slam, --no-rviz, --no-nav, --no-web, --lidar-slam, --hands, …) pass through.
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -s|--scene)  export MABEL_SCENE="$2"; shift 2 ;;   # real_run reads MABEL_SCENE
    --lidar)     shift ;;                               # lidar is ON by default in real_run
    --rviz)      shift ;;                               # RViz is ON by default in real_run
    --no-video)  printf '  \033[33m⚠ --no-video has no real_run equivalent (camera_hal serves :8080); ignoring.\033[0m\n' >&2; shift ;;
    --no-backend)printf '  \033[33m⚠ --no-backend is not supported by real_run (it always brings up the HAL); ignoring.\033[0m\n' >&2; shift ;;
    *)           args+=("$1"); shift ;;
  esac
done

exec "$SCRIPT_DIR/real_run.sh" "${args[@]}"

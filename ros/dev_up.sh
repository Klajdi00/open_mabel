#!/usr/bin/env bash
# =============================================================================
# MABEL -- one-command dev bring-up
# =============================================================================
# Run this ONE file from a fresh clone (Ubuntu 24.04) and it will:
#   1. find conda (or install Miniforge if none is present)
#   2. create / update the reproducible `mabel` env from environment.yml
#      (RoboStack ROS 2 Jazzy + all python deps -- no apt, no system ROS)
#   3. colcon-build the workspace
#   4. source the workspace in every pane
#   5. split into 4 panes laid out side-by-side (horizontal), each with its
#      command PRE-TYPED + a clear comment (you just press Enter in each,
#      starting with the sim):
#        pane 0  sim            ros2 launch mabel_sim sim.launch.py ...
#        pane 1  LiDAR SLAM     ros2 launch mabel_slam lidar_slam.launch.py ...
#                               (the pane also prints the one-liner to switch
#                                to visual SLAM; or run ./dev_up.sh --visual)
#        pane 2  keyboard teleop ros2 run mabel_teleop keyboard_teleop
#        pane 3  exploration    ros2 launch mabel_navigation exploration.launch.py slam:=none
#
# Usage:
#   ./dev_up.sh                 # full: env + build + tmux (LiDAR SLAM, GPU-free)
#   ./dev_up.sh --visual        # use visual SLAM (cuVSLAM+nvblox) in pane 2 instead
#   ./dev_up.sh --no-build      # skip colcon build (env + tmux only)
#   ./dev_up.sh --no-tmux       # set up env + build, don't open the session
#   ./dev_up.sh --with-sdk      # also vcs-import + build ZED/RealSense source pkgs
#                               #   (needs the ZED SDK + librealsense installed)
#   ./dev_up.sh --env-only      # just create/update the conda env, nothing else
# =============================================================================
set -euo pipefail

# Keep the env hermetic: ignore ~/.local/lib/pythonX user-site packages, which are
# on sys.path and can shadow the conda env's ROS/python at build & runtime.
export PYTHONNOUSERSITE=1

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=mabel
ENV_FILE="$WS/environment.yml"
SESSION=mabel

DO_BUILD=1; DO_TMUX=1; WITH_SDK=0; SLAM=lidar; CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --no-build) DO_BUILD=0 ;;
    --no-tmux)  DO_TMUX=0 ;;
    --with-sdk) WITH_SDK=1 ;;
    --visual)   SLAM=visual ;;
    --lidar)    SLAM=lidar ;;
    --clean)    CLEAN=1 ;;
    --env-only) DO_BUILD=0; DO_TMUX=0 ;;
    -h|--help)  sed -n '2,44p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# -----------------------------------------------------------------------------
# 1. find or install conda
# -----------------------------------------------------------------------------
find_conda() {
  if command -v conda >/dev/null 2>&1; then command -v conda; return 0; fi
  [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ] && { echo "$CONDA_EXE"; return 0; }
  local d
  for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" \
           "$HOME/miniconda3" "/opt/conda"; do
    [ -x "$d/bin/conda" ] && { echo "$d/bin/conda"; return 0; }
  done
  return 1
}

CONDA="$(find_conda || true)"
if [ -z "$CONDA" ]; then
  log "No conda found -- installing Miniforge3 into \$HOME/miniforge3"
  arch="$(uname -m)"
  url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/miniforge.sh" "$url"
  bash "$tmp/miniforge.sh" -b -p "$HOME/miniforge3"
  rm -rf "$tmp"
  CONDA="$HOME/miniforge3/bin/conda"
fi
log "Using conda: $CONDA"

# RoboStack requires strict conda-forge priority (esp. on an Anaconda base, whose
# `defaults` channel would otherwise shadow the conda-forge / robostack builds).
"$CONDA" config --env --set channel_priority strict >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 2. create / update the env  (idempotent)
# -----------------------------------------------------------------------------
if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Updating existing env '$ENV_NAME'"
  "$CONDA" env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  log "Creating env '$ENV_NAME' from environment.yml (this pulls ROS Jazzy -- a few minutes)"
  "$CONDA" env create -f "$ENV_FILE"
fi

# activate it for the rest of this script. RoboStack's activation scripts assume
# nounset is OFF (they reference $CONDA_BUILD etc. unbound), so relax -u around it.
eval "$("$CONDA" shell.bash hook)"
set +u
conda activate "$ENV_NAME"
set -u

# Drop foreign ROS overlays (apt /opt/ros + any sibling colcon workspace the login
# profile sourced) so the BUILD links against conda ROS only and the resulting
# overlay chains $CONDA_PREFIX -- not /opt/ros or another workspace.
MABEL_WS="$WS"
source "$WS/scripts/_scrub_ros_env.sh"
mabel_scrub_ros_env

log "Active env: $CONDA_PREFIX   (ros2: $(command -v ros2 || echo MISSING))"

# -----------------------------------------------------------------------------
# 2.5 camera SDK python APIs (pyrealsense2 + pyzed) INTO the conda env
# -----------------------------------------------------------------------------
# sensors/ros_camera_node.py -- the SINGLE owner of the USB depth cameras -- runs
# in THIS conda env and imports pyrealsense2 (Intel D435i) + pyzed (Stereolabs
# ZED-M). pyrealsense2 is on PyPI (already in environment.yml); pyzed is NOT -- it
# ships per ZED-SDK version and is installed from the LOCAL SDK via its
# get_python_api.py (which downloads the matching wheel). environment.yml can only
# DOCUMENT that, so do it here: a fresh `dev_up.sh` then brings the head ZED up
# with no manual step (this is exactly what was missing on the Thor Jetson).
# Both are NON-FATAL: a box with no ZED SDK / no librealsense still builds the
# workspace and runs the sim; only the matching real camera stays offline.
install_camera_sdks() {
  log "Camera SDK python APIs (pyrealsense2 + pyzed) for the camera-hardware node"
  # D435i front cam: pyrealsense2 (also in environment.yml -- ensure importable).
  if python -c "import pyrealsense2" >/dev/null 2>&1; then
    echo "  pyrealsense2: present ($(python -c 'import pyrealsense2 as r; print(r.__version__)' 2>/dev/null))"
  else
    echo "  pyrealsense2: installing from PyPI"
    python -m pip install --quiet pyrealsense2 \
      || echo "  WARN: pyrealsense2 install failed -- D435 front cam (+ visual SLAM) unavailable."
  fi
  # ZED-M head cam: pyzed from the LOCAL ZED SDK. Idempotent: skip if importable;
  # warn (never fail) if no SDK is installed on this box.
  if python -c "import pyzed.sl" >/dev/null 2>&1; then
    echo "  pyzed: present"
  elif [ -f /usr/local/zed/get_python_api.py ]; then
    echo "  pyzed: installing from local ZED SDK (/usr/local/zed/get_python_api.py)"
    ( cd /usr/local/zed && python get_python_api.py ) \
      || echo "  WARN: pyzed install failed -- ZED head cam unavailable."
  else
    echo "  WARN: no ZED SDK at /usr/local/zed -- install it from"
    echo "        https://www.stereolabs.com/developers then re-run; ZED head cam stays offline."
  fi
  # get_python_api.py / pyzed pull numpy>=2.0, which breaks the numpy-1.x ABI the
  # REST of the env needs (numpy-quaternion, scipy, AND the ZED C-ext at capture).
  # Re-pin numpy<2: pyzed 5.x still imports + grabs colour/depth/cloud fine at
  # numpy 1.26 (its >=2.0 wheel metadata is advisory, not a hard runtime need --
  # verified on the Thor ZED-M). See the numpy pin rationale in environment.yml.
  python -m pip install --quiet "numpy>=1.24,<2" || true
  echo "  numpy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null) (pinned <2 for the camera/pose ABI)"
}
install_camera_sdks

# -----------------------------------------------------------------------------
# 3. optional: pull the source SDK packages (ZED / RealSense)
# -----------------------------------------------------------------------------
if [ "$WITH_SDK" -eq 1 ]; then
  log "Importing source SDK packages (thirdparty.repos)"
  vcs import "$WS/src" < "$WS/thirdparty.repos"
fi

# -----------------------------------------------------------------------------
# 4. build
# -----------------------------------------------------------------------------
# The ZED / RealSense ROS packages need their native SDKs; ignore EVERY package
# found under their source trees (discovered dynamically so nothing slips through)
# unless --with-sdk was given. Absent on a fresh clone -> empty list -> no-op.
SKIP_SDK=()
if [ "$WITH_SDK" -ne 1 ]; then
  while IFS= read -r name; do
    [ -n "$name" ] && SKIP_SDK+=("$name")
  done < <(find src/realsense-ros src/zed-ros2-wrapper src/zed-ros2-interfaces \
             -name package.xml 2>/dev/null \
             -exec grep -hoP '(?<=<name>)[^<]+' {} \; | sort -u)
fi

if [ "$DO_BUILD" -eq 1 ]; then
  cd "$WS"
  # A workspace previously built against the apt ROS records /opt/ros/jazzy as its
  # underlay; sourcing that overlay would re-chain system ROS and shadow the conda
  # one. Rebuild clean against the conda env so the overlay chains $CONDA_PREFIX.
  if [ "$CLEAN" -eq 1 ] || { [ -f install/setup.bash ] && grep -qs "/opt/ros" install/setup.bash; }; then
    log "Removing stale build/ install/ log/ (clean conda rebuild)"
    rm -rf build install log
  fi
  log "Building workspace with colcon"
  if [ "${#SKIP_SDK[@]}" -gt 0 ]; then
    log "Ignoring SDK packages (no --with-sdk): ${SKIP_SDK[*]}"
    colcon build --symlink-install --packages-ignore "${SKIP_SDK[@]}" \
      --cmake-args -DCMAKE_BUILD_TYPE=Release
  else
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
  fi
  log "Build complete."
fi

# -----------------------------------------------------------------------------
# 5. the 4-pane tmux dev session (commands PRE-TYPED, awaiting Enter)
# -----------------------------------------------------------------------------
ENV_SRC="source \"$WS/scripts/mabel_env.sh\""

# --- the four pane commands, in launch order -------------------------------
# 1. sim            2. LiDAR SLAM      3. keyboard teleop      4. exploration
CMD_SIM='ros2 launch mabel_sim sim.launch.py front_only:=true sensor_rate:=15'
CMD_TELEOP='ros2 run mabel_teleop keyboard_teleop'
# exploration owns nav; slam:=none because the dedicated SLAM pane (1) runs SLAM.
CMD_EXPLORE='ros2 launch mabel_navigation exploration.launch.py slam:=none use_sim_time:=true'

# pane 1: LiDAR SLAM by default (GPU-free, slam_toolbox -- no Docker / Isaac).
# --visual swaps in cuVSLAM+nvblox (needs the Isaac ROS stack; native on Jazzy).
CMD_SLAM_VISUAL='ros2 launch mabel_slam visual_slam.launch.py env:=navigation use_sim_time:=true'
if [ "$SLAM" = "visual" ]; then
  CMD_SLAM="$CMD_SLAM_VISUAL"
  SLAM_TITLE="1 VISUAL SLAM (needs Isaac ROS)"
  SLAM_HINT="# switch to LiDAR SLAM: ros2 launch mabel_slam lidar_slam.launch.py use_sim_time:=true"
else
  CMD_SLAM='ros2 launch mabel_slam lidar_slam.launch.py use_sim_time:=true'
  SLAM_TITLE="1 LIDAR SLAM"
  SLAM_HINT="# switch to VISUAL SLAM: $CMD_SLAM_VISUAL"
fi

setup_pane() {  # <pane-index> <title> <command-to-pretype> [hint-comment]
  local p="$1" title="$2" cmd="$3" hint="${4:-}"
  tmux send-keys -t "$SESSION:dev.$p" "$ENV_SRC" C-m            # source the env + workspace
  tmux send-keys -t "$SESSION:dev.$p" "clear; echo '[$title] press Enter to run:'" C-m
  [ -n "$hint" ] && \
    tmux send-keys -t "$SESSION:dev.$p" "echo '$hint'" C-m     # print the alt-command hint
  tmux send-keys -t "$SESSION:dev.$p" "$cmd"                    # PRE-TYPE only, no Enter
}

if [ "$DO_TMUX" -eq 1 ]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux not found in env"; exit 1; }
  tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

  log "Opening tmux session '$SESSION' (4 panes side-by-side, commands pre-typed)"
  tmux new-session -d -s "$SESSION" -n dev
  tmux split-window -h -t "$SESSION:dev.0"   # pane 1  (split horizontally = side by side)
  tmux split-window -h -t "$SESSION:dev.0"   # pane 2
  tmux split-window -h -t "$SESSION:dev.0"   # pane 3
  tmux select-layout -t "$SESSION:dev" even-horizontal   # 4 equal columns, left -> right
  tmux set-option -t "$SESSION" pane-border-status top
  tmux set-option -t "$SESSION" mouse on

  tmux select-pane -t "$SESSION:dev.0" -T "0 SIM"
  tmux select-pane -t "$SESSION:dev.1" -T "$SLAM_TITLE"
  tmux select-pane -t "$SESSION:dev.2" -T "2 TELEOP"
  tmux select-pane -t "$SESSION:dev.3" -T "3 EXPLORATION"

  setup_pane 0 "SIM"          "$CMD_SIM"
  setup_pane 1 "SLAM"         "$CMD_SLAM"      "$SLAM_HINT"
  setup_pane 2 "TELEOP"       "$CMD_TELEOP"
  setup_pane 3 "EXPLORATION"  "$CMD_EXPLORE"

  tmux select-pane -t "$SESSION:dev.0"

  cat <<EOF

  tmux session '$SESSION' is ready -- 4 panes side-by-side, each command typed and waiting.
  Order, left -> right:  0 SIM  ->  1 SLAM  ->  2 TELEOP  ->  3 EXPLORATION.
  Start the SIM (pane 0) first, then SLAM / teleop / exploration -- just press Enter in each.
  Switch panes: Ctrl-b then arrow keys  |  detach: Ctrl-b d  |  kill: tmux kill-session -t $SESSION

EOF
  if [ -t 1 ]; then
    tmux attach -t "$SESSION"
  else
    echo "  (non-interactive shell) attach with:  tmux attach -t $SESSION"
  fi
fi

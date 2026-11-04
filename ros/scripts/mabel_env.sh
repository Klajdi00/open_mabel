# MABEL dev environment -- SOURCE this file in any shell to start working:
#
#     source scripts/mabel_env.sh
#
# It activates the `mabel` conda env (RoboStack ROS 2 Jazzy -- this also puts
# `ros2`, rviz2, etc. on PATH) and overlays the colcon workspace. Every terminal
# that runs `ros2 ...` must do this first; dev_up.sh runs it automatically in
# each tmux pane.

# --- locate the workspace (this file lives in <ws>/scripts/) -----------------
_MABEL_WS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_MABEL_ENV="${MABEL_ENV_NAME:-mabel}"

# --- find a conda, even if it isn't on PATH yet ------------------------------
_mabel_find_conda() {
  if command -v conda >/dev/null 2>&1; then command -v conda; return 0; fi
  if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then echo "$CONDA_EXE"; return 0; fi
  local d
  for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/anaconda3" \
           "$HOME/miniconda3" "/opt/conda" "/opt/miniforge3"; do
    [ -x "$d/bin/conda" ] && { echo "$d/bin/conda"; return 0; }
  done
  return 1
}

_MABEL_CONDA="$(_mabel_find_conda)" || {
  echo "mabel_env: could not find conda. Run ./dev_up.sh first to install it." >&2
  return 1 2>/dev/null || exit 1
}

# Keep the env hermetic: ignore ~/.local user-site packages that shadow the env.
export PYTHONNOUSERSITE=1

# --- activate the env (loads ROS via RoboStack activation hooks) -------------
# RoboStack activation assumes nounset is OFF; relax it around activate so a
# caller running `set -u` doesn't abort on $CONDA_BUILD being unbound.
eval "$("$_MABEL_CONDA" shell.bash hook)"
# RoboStack's activation hooks AND colcon's install/setup.bash both reference
# unbound vars ($CONDA_BUILD, $COLCON_TRACE); relax nounset across both, restore
# the caller's setting afterwards.
_mabel_u=0; case "$-" in *u*) _mabel_u=1; set +u ;; esac

conda activate "$_MABEL_ENV" || {
  echo "mabel_env: env '$_MABEL_ENV' not found. Run ./dev_up.sh to create it." >&2
  [ "$_mabel_u" = 1 ] && set -u
  return 1 2>/dev/null || exit 1
}

# --- overlay the built workspace ---------------------------------------------
if [ -f "$_MABEL_WS/install/setup.bash" ]; then
  source "$_MABEL_WS/install/setup.bash"
else
  echo "mabel_env: workspace not built yet ($_MABEL_WS/install missing)." >&2
  echo "           run ./dev_up.sh  (or: colcon build --symlink-install)" >&2
fi

# --- scrub foreign ROS overlays (apt /opt/ros + sibling workspaces) ----------
# Done AFTER the overlay so anything it re-chains is also removed; leaves only the
# conda env + this workspace on PATH / LD_LIBRARY_PATH / PYTHONPATH / *_PREFIX_PATH.
MABEL_WS="$_MABEL_WS"
source "$_MABEL_WS/scripts/_scrub_ros_env.sh"
mabel_scrub_ros_env

# --- ONE DDS for the whole machine -------------------------------------------
# This box has docker0 + WiFi + lo interfaces, and the apt ROS defaults to a
# DIFFERENT RMW (FastDDS) than this conda env (CycloneDDS). Left alone you get
# "two DDS": `ros2 topic list` from one shell can't see nodes started in another
# (or sees only some), because participants scatter across interfaces / RMWs.
# Pin EVERYTHING: CycloneDDS, domain 0, and LOCALHOST-only discovery (a built-in
# ROS 2 Jazzy mechanism) so every local node -- this env AND the apt Isaac SLAM
# via slam_env.sh, which sets the identical trio -- shares one graph and reliably
# discovers each other. (Clients connect over websockets, not DDS, so localhost
# discovery costs nothing. For a future multi-machine setup, widen this to SUBNET.)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# Shared DDS tuning (big socket receive buffers so the heavy point-cloud/mesh
# topics stop dropping — see the WHY inside the xml). rmw_cyclonedds merges this
# with the LOCALHOST discovery config it generates. slam_env.sh points at the SAME
# file so both stacks' sockets get the identical treatment.
export CYCLONEDDS_URI="file://$_MABEL_WS/scripts/cyclonedds.xml"

[ "$_mabel_u" = 1 ] && set -u
unset _mabel_u

cd "$_MABEL_WS"
unset -f _mabel_find_conda

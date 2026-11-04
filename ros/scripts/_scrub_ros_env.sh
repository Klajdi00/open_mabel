# Shared helper (source me): make a conda ROS env hermetic.
#
# The login profile on this box auto-sources BOTH apt ROS (/opt/ros/jazzy) and a
# sibling colcon workspace (/home/mabel/open_mabel/install). Either one prepends its
# ros2 / .so's / python site-packages ahead of the conda env, so `ros2` and
# `import rclpy` resolve to system ROS and ABI-clash with conda's. This drops
# every foreign ROS overlay from the ROS-related env vars, leaving ONLY the conda
# env and THIS workspace ($MABEL_WS).
#
# Requires MABEL_WS to point at this workspace root. Call mabel_scrub_ros_env
# AFTER conda activate (build) and AFTER sourcing the overlay (runtime).
mabel_scrub_ros_env() {
  local ws="${MABEL_WS:-}"
  local v cur out p keep
  for v in PATH LD_LIBRARY_PATH PYTHONPATH CMAKE_PREFIX_PATH AMENT_PREFIX_PATH \
           COLCON_PREFIX_PATH PKG_CONFIG_PATH AMENT_CURRENT_PREFIX; do
    cur="$(eval "printf '%s' \"\${$v:-}\"")"
    [ -z "$cur" ] && continue
    out=""
    while IFS= read -r p; do
      [ -z "$p" ] && continue
      keep=1
      case "$p" in
        /opt/ros/*) keep=0 ;;                      # apt ROS
      esac
      case "$p" in                                 # a colcon install/ tree...
        */install|*/install/*)
          case "$p" in
            "$ws"/install|"$ws"/install/*) ;;      # ...ours -> keep
            *) keep=0 ;;                           # ...someone else's -> drop
          esac ;;
      esac
      [ "$keep" = 1 ] && out="${out:+$out:}$p"
    done < <(printf '%s\n' "$cur" | tr ':' '\n')
    export "$v=$out"
  done
}

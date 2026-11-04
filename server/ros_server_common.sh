#!/usr/bin/env bash
# =============================================================================
# ros_server_common.sh — shared bring-up for the ROS-backed MABEL teleop server
# =============================================================================
# SOURCED by sim_ros_server.sh / real_ros_server.sh, which set MODE=sim|real then
# call `ros_server_main "$@"`. One command brings up the full stack on Ubuntu so
# the iOS app (and Vision Pro / web) gets EVERYTHING — teleop control, robot
# state, control modes, camera video — PLUS the live SLAM map (nvblox mesh +
# odom) streamed back over the new /map channel.
#
# THREE process groups, each launched in its OWN environment subshell, because
# the two ROS environments are mutually exclusive:
#
#   ┌ BACKEND (conda RoboStack ROS — scripts/mabel_env.sh) ────────────────────┐
#   │  sim : ros2 launch mabel_sim sim.launch.py physics:=false                  │
#   │        → sensors (cams/LiDAR) + swerve_odometry + robot_state_publisher    │
#   │        NO physics node — the SERVER owns physics (see below)               │
#   │  real: ros2 launch mabel_bringup hardware.launch.py   real HAL + cameras  │
#   ├ SLAM (apt Isaac ROS — scripts/slam_env.sh, which WIPES conda) ────────────┤
#   │  ros2 launch mabel_slam visual_slam.launch.py   cuVSLAM + nvblox → mesh   │
#   │  → /nvblox_node/mesh_pointcloud  +  map→odom correction                   │
#   ├ SERVER (conda RoboStack ROS — scripts/mabel_env.sh) ──────────────────────┤
#   │  sim : sim_teleop_bridge.py  (MABEL_DISPATCH_TARGET=sim:mujoco+ros)        │
#   │        OWNS MuJoCo physics — identical controller / motion model / WBC     │
#   │        torque engine / mj_step pacing to the Mac sim — AND publishes the   │
#   │        canonical ROS state: /clock /joint_states /odom_ground_truth /imu   │
#   │  real: sim_teleop_bridge.py  (MABEL_DISPATCH_TARGET=real:ros) drives HAL   │
#   │    → :9090  /teleop (drive) · /map (mesh+odom) · robot_state · control     │
#   │    the bridge SUBSCRIBES to /nvblox_node/mesh_pointcloud + /odom and fans  │
#   │    them out to /map clients (see map_stream.py)                           │
#   │  {sim,real}_camera_server.py → :8080 MJPEG + :9091 UDP  (iOS video tiles) │
#   └───────────────────────────────────────────────────────────────────────────┘
#
# Sim = the Mac sim, on Linux. To make the ROS sim behave EXACTLY like running the
# MuJoCo sim on the Mac (same controller, motion model, runtime, all control modes),
# the bridge OWNS the physics in sim too (sim:mujoco+ros, server/plant.py SimRosPlant):
# it steps MuJoCo + runs the full mabel controller just like the Mac, and additionally
# publishes the canonical ROS state so the sensor renderer / SLAM / Nav2 / RViz layer
# on top unchanged. There is exactly ONE physics + /clock authority (the bridge), so
# mabel_sim's mujoco_node physics is NOT launched (physics:=false). On the REAL robot
# the bridge instead mirrors encoders and drives the HAL over the same contract.
#
# All three groups talk over CycloneDDS on ROS_DOMAIN_ID 0 (forced by the env
# scripts), so the conda sim/server and the apt SLAM interoperate.
#
# Config (env-overridable, same names as start_teleop.sh where they overlap):
#   MABEL_ROOT     repo root            (default: this file's ../.. )
#   MABEL_WS       colcon workspace     (default: $MABEL_ROOT/ros)
#   MABEL_SCENE    sim/mirror MJCF      (default: simulation/.../mabel_home.xml)
#   MABEL_SLAM     visual | lidar | both (default: visual — only visual makes a mesh;
#                  both = visual primary + lidar secondary /map_lidar, no map->odom fight)
#   MABEL_TOKEN    teleop access token  (default: unset / open)
#   MABEL_JPEG_QUALITY                  (default: 85)
#   MABEL_STEREO   1|0 stereo head cam  (default: 1)
#   MABEL_HTTP_PORT / MABEL_WS_PORT / MABEL_UDP_VIDEO_PORT  (8080 / 9090 / 9091)
#
# Flags: -s|--scene PATH · --lidar · --rviz · --no-slam · --no-backend ·
#        --no-video · -h|--help
# =============================================================================

set -o pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MABEL_ROOT="${MABEL_ROOT:-$(cd "$SERVER_DIR/.." && pwd)}"
WS="${MABEL_WS:-$MABEL_ROOT/ros}"

# ── config defaults ───────────────────────────────────────────────────────────
SCENE="${MABEL_SCENE:-$MABEL_ROOT/simulation/mabel_mujoco/scenes/mabel_home.xml}"
SLAM="${MABEL_SLAM:-visual}"
JPEG_QUALITY="${MABEL_JPEG_QUALITY:-85}"
STEREO="${MABEL_STEREO:-1}"
HTTP_PORT="${MABEL_HTTP_PORT:-8080}"
WS_PORT="${MABEL_WS_PORT:-9090}"
UDP_VIDEO_PORT="${MABEL_UDP_VIDEO_PORT:-9091}"
APP_TOKEN="${MABEL_TOKEN:-}"

DO_SLAM=1; DO_BACKEND=1; DO_VIDEO=1; DO_NAV=1
DO_MONITOR=1                   # Command Monitor GUI on by default (like run.sh)
DO_CAM_ROS=1                   # camera render also publishes ROS Image topics
# Visualisations: in SIM both come up by DEFAULT so one launch gives you the full
# picture — RViz (the ROS view: robot model + sensors + the live nvblox SLAM mesh,
# layout sim_sensors.rviz) AND the desktop MuJoCo mirror window (view_robot.py,
# which shows the robot's TRUE simulated state — it mirrors the bridge's qpos and
# never steps its own physics). Opt out with --no-rviz / --no-viewer. On the REAL
# robot (often headless) both default OFF; turn them on with --rviz / --viewer.
# RViz comes up by DEFAULT in both sim and real (real_run.sh-style visualization:
# robot + sensors + the nvblox SLAM mesh/2D map). The MuJoCo mirror window is
# sim-only. Suppress RViz on a headless robot with --no-rviz.
if [ "$MODE" = sim ]; then DO_RVIZ=1; DO_VIEWER=1; else DO_RVIZ=1; DO_VIEWER=0; fi

LOG_DIR=/tmp
LOG_BACKEND="$LOG_DIR/mabel_backend.log"
LOG_SLAM="$LOG_DIR/mabel_slam.log"
LOG_BRIDGE="$LOG_DIR/mabel_bridge.log"
LOG_CAM="$LOG_DIR/mabel_cameras.log"
LOG_MON="$LOG_DIR/mabel_monitor.log"
LOG_VIEW="$LOG_DIR/mabel_viewer.log"
LOG_NAV="$LOG_DIR/mabel_nav.log"

PIDS=()

# ── pretty printing ────────────────────────────────────────────────────────────
say()  { printf '  \033[2m▸\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
$(basename "$0") — bring up the ROS ($MODE) backend + SLAM + the MABEL app server.

  ./sim_ros_server.sh        # MuJoCo ROS sim + visual SLAM + app server
  ./real_ros_server.sh       # real robot HAL + visual SLAM + app server

Options:
  -s, --scene PATH   MuJoCo scene / mirror model (default: mabel_home.xml)
  --lidar            use lidar_slam instead of visual_slam (NOTE: no 3D mesh)
  --both-slam        run BOTH: visual_slam (primary: mesh + map->odom + Nav2 costmap)
                     AND lidar_slam as a secondary (map_lidar frame + /map_lidar topic)
                     so the two never fight over map->odom. Nav2 still uses visual's /map.
  --rviz / --no-rviz open / suppress the backend's RViz (robot + sensors + nvblox
                     SLAM mesh).  DEFAULT: on in sim, off on the real robot.
  --viewer/--no-viewer  open / suppress the desktop MuJoCo mirror window
                     (view_robot.py — the robot's TRUE simulated state).
                     DEFAULT: on in sim, off on the real robot.
  --no-backend       don't launch sim/hardware (assume it's already running)
  --no-slam          don't launch SLAM (assume it's already running, or none)
  --no-nav           don't launch Nav2 (NavFn A* + MPPI for web/app goals)
  --no-video         don't launch the MJPEG/UDP camera server
  --no-cam-ros       camera renders to MJPEG only (don't also publish ROS Image topics)
  --no-monitor       don't open the Command Monitor GUI (on by default, like run.sh)
  -h, --help         this help

Env overrides: MABEL_ROOT MABEL_WS MABEL_SCENE MABEL_SLAM MABEL_TOKEN
               MABEL_JPEG_QUALITY MABEL_STEREO MABEL_{HTTP,WS,UDP_VIDEO}_PORT
EOF
}

# ── env subshell launchers ─────────────────────────────────────────────────────
# Each group runs `( source <env>; exec <cmd> )` so its environment never leaks
# into the others. Because of `exec`, $! is the launched process itself, so a
# later `kill` reaches `ros2 launch` (which then shuts its child nodes down).
_spawn_conda() {  # logfile workdir -- cmd...
  local logf="$1" wd="$2"; shift 2
  ( source "$WS/scripts/mabel_env.sh" >>"$logf" 2>&1 || exit 1
    cd "$wd" || exit 1
    exec "$@" ) >>"$logf" 2>&1 &
  PIDS+=($!)
  echo $!
}

_spawn_slam() {   # logfile -- cmd...
  local logf="$1"; shift
  ( source "$WS/scripts/slam_env.sh" >>"$logf" 2>&1 || exit 1
    exec "$@" ) >>"$logf" 2>&1 &
  PIDS+=($!)
  echo $!
}

# Wait until a topic appears (sources the conda env ONCE, then polls).
_wait_topic() {   # topic timeout_s
  local topic="$1" timeout="${2:-40}"
  ( source "$WS/scripts/mabel_env.sh" >/dev/null 2>&1 || exit 1
    local i=0
    while [ "$i" -lt "$timeout" ]; do
      ros2 topic list 2>/dev/null | grep -qx "$topic" && exit 0
      sleep 1; i=$((i + 1))
    done
    exit 1 )
}

# ── lifecycle ──────────────────────────────────────────────────────────────────
_cleaned=0
cleanup() {
  [ "$_cleaned" = 1 ] && return; _cleaned=1
  echo; say "shutting down…"
  # SIGINT first so `ros2 launch` tears its nodes down cleanly.
  for p in "${PIDS[@]:-}"; do kill -INT "$p" >/dev/null 2>&1 || true; done
  sleep 1.5
  for p in "${PIDS[@]:-}"; do kill -TERM "$p" >/dev/null 2>&1 || true; done
  sleep 0.5
  # backstop: reap anything that ignored the signals (matched narrowly).
  for pat in sim_teleop_bridge sim_camera_server real_camera_server \
             "mabel_sim/mujoco" "mabel_base_hardware/swerve" mesh_to_pointcloud \
             nvblox_node cuvslam visual_slam async_slam_toolbox_node \
             "ros2 launch mabel_navigation" \
             controller_server planner_server bt_navigator cmd_vel_bridge; do
    pkill -9 -f "$pat" >/dev/null 2>&1 || true
  done
  ok "done."
}

# Reap any previous run BEFORE launching, so re-running never stacks instances
# (which fight over :9090/:8080 and the ROS topics — the camera server then can't
# bind :8080 and the app gets no video). Mirrors start_teleop.sh's pre-clean.
# NOTE: the patterns below are chosen so NONE match this launcher's own cmdline
# ("bash …/sim_ros_server.sh") — pkill -f would otherwise kill us.
_reap_stale() {
  say "clearing any previous run …"
  local pat
  for pat in sim_teleop_bridge sim_camera_server real_camera_server udp_video \
             "mabel_sim/lib/mabel_sim" "mabel_base_hardware/swerve" \
             mesh_to_pointcloud nvblox_node visual_slam_container async_slam_toolbox_node \
             "ros2 launch mabel_sim" "ros2 launch mabel_slam" "ros2 launch mabel_bringup" \
             "ros2 launch mabel_navigation" controller_server planner_server \
             bt_navigator cmd_vel_bridge \
             monitor_gui view_robot.py; do
    pkill -9 -f "$pat" >/dev/null 2>&1 || true
  done
  # Stale launchers (other instances of these scripts) — never ourselves ($$) or
  # our parent ($PPID, e.g. a wrapper shell), so we can't kill the process that
  # started us.
  local pid
  for pid in $(pgrep -f "sim_ros_server.sh|real_ros_server.sh" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    [ "$pid" = "$PPID" ] && continue
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  # Free the ports if anything is still wedged on them.
  if command -v lsof >/dev/null 2>&1; then
    for port in "$WS_PORT" "$HTTP_PORT" "$UDP_VIDEO_PORT"; do
      for pid in $(lsof -ti :"$port" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        kill -9 "$pid" >/dev/null 2>&1 || true
      done
    done
  fi
  sleep 1
}

_preflight() {
  [ -f "$WS/scripts/mabel_env.sh" ] || die "missing $WS/scripts/mabel_env.sh — set MABEL_WS."
  [ -f "$WS/install/setup.bash" ]   || die "workspace not built — run ros/dev_up.sh first."
  [ -f "$SERVER_DIR/sim_teleop_bridge.py" ] || die "run this from the server/ dir."
  if [ "$DO_SLAM" = 1 ] && [ "$SLAM" = visual ] && [ ! -f "$WS/scripts/slam_env.sh" ]; then
    die "visual SLAM needs $WS/scripts/slam_env.sh (Isaac ROS apt env)."
  fi
  [ "$MODE" = sim ] && { [ -f "$SCENE" ] || die "scene not found: $SCENE"; }
  _check_server_deps
}

# The bridge runs in the conda env, so that env needs BOTH rclpy (workspace) AND
# the server's deps (server/requirements.txt). Catch a gap here with a clear fix
# instead of a silent bridge crash logged to a file. Skip with MABEL_SKIP_DEPCHECK=1.
_check_server_deps() {
  [ "${MABEL_SKIP_DEPCHECK:-0}" = 1 ] && return
  say "checking the conda env has the server + ROS deps …"
  local missing
  missing="$(
    source "$WS/scripts/mabel_env.sh" >/dev/null 2>&1 || { echo "__ENV__"; exit 0; }
    python3 - <<'PY' 2>/dev/null
import importlib.util as u
need = ["rclpy","numpy","websockets","aiohttp","mujoco","cv2","quadprog","osqp",
        "sensor_msgs.msg","nav_msgs.msg"]
print(" ".join(m for m in need if u.find_spec(m) is None))
PY
  )"
  if [ "$missing" = "__ENV__" ]; then
    die "couldn't activate the '$WS' conda env — run ros/dev_up.sh first."
  fi
  if [ -n "$missing" ]; then
    warn "missing Python modules in the conda env: $missing"
    say  "fix (pick one):"
    say  "    conda env update -n mabel -f $WS/environment.yml --prune   # reproducible"
    say  "    OR:  conda run -n mabel pip install -r $SERVER_DIR/requirements.txt"
    die  "install the missing deps, then re-run. (Override with MABEL_SKIP_DEPCHECK=1.)"
  fi
  ok "conda env has all server + ROS deps."
}

# ── argument parsing ───────────────────────────────────────────────────────────
_parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -s|--scene)   SCENE="$2"; shift 2 ;;
      --lidar)      SLAM="lidar"; shift ;;
      --both-slam|--slam-both) SLAM="both"; shift ;;
      --rviz)       DO_RVIZ=1; shift ;;
      --no-rviz)    DO_RVIZ=0; shift ;;
      --no-backend) DO_BACKEND=0; shift ;;
      --no-slam)    DO_SLAM=0; shift ;;
      --no-nav)     DO_NAV=0; shift ;;
      --no-video)   DO_VIDEO=0; shift ;;
      --no-cam-ros) DO_CAM_ROS=0; shift ;;
      --monitor)    DO_MONITOR=1; shift ;;
      --no-monitor) DO_MONITOR=0; shift ;;
      --viewer)     DO_VIEWER=1; shift ;;
      --no-viewer)  DO_VIEWER=0; shift ;;
      -h|--help)    usage; exit 0 ;;
      *) warn "unknown arg: $1"; usage; exit 1 ;;
    esac
  done
}

# ── group launchers ────────────────────────────────────────────────────────────
_launch_backend() {
  [ "$DO_BACKEND" = 1 ] || { say "backend   skipped (--no-backend)"; return; }
  : > "$LOG_BACKEND"
  if [ "$MODE" = sim ]; then
    local rviz="false"; [ "$DO_RVIZ" = 1 ] && rviz="true"
    # physics:=false — the teleop SERVER (sim_teleop_bridge in sim:mujoco+ros) OWNS the
    # MuJoCo physics + /clock, byte-identical to the Mac sim. This launch runs only the
    # camera/LiDAR renderer + swerve_odometry + robot_state_publisher (+ RViz), all fed
    # by the bridge's state. So /joint_states + /odom come up once the server starts —
    # the readiness wait is deferred to _launch_server (below). ONE /clock publisher.
    say "backend   MuJoCo sensors + odom + TF (physics from the server)   (log: $LOG_BACKEND)"
    _spawn_conda "$LOG_BACKEND" "$WS" \
      ros2 launch mabel_sim sim.launch.py physics:=false rviz:="$rviz" scene:="$SCENE" >/dev/null
    say "          sensors waiting for the server's physics to publish /joint_states …"
  else
    local rviz="false"; [ "$DO_RVIZ" = 1 ] && rviz="true"
    # hardware.launch.py can bring up SLAM itself, but in the SERVER the SLAM group
    # runs as its OWN process (apt Isaac env, dedicated $LOG_SLAM + readiness ✓ via
    # _verify_slam) exactly like sim — so pass slam:=false here to avoid a SECOND
    # cuVSLAM/nvblox. RViz uses hardware.rviz (the launch default, Fixed Frame
    # base_link) which already carries the nvblox mesh + 2D /map displays.
    local hw_args=(slam:=false rviz:="$rviz")
    say "backend   real hardware HAL (mabel_bringup)   (log: $LOG_BACKEND)"
    _spawn_conda "$LOG_BACKEND" "$WS" \
      ros2 launch mabel_bringup hardware.launch.py "${hw_args[@]}" >/dev/null
    say "          waiting for /odom …"
    if _wait_topic /odom 45; then ok "backend up — /odom is publishing."
    else warn "/odom not seen in 45s — check $LOG_BACKEND (continuing anyway)."; fi
  fi
}

_launch_slam() {
  [ "$DO_SLAM" = 1 ] || { say "slam      skipped (--no-slam)"; return; }
  : > "$LOG_SLAM"
  # visual = cuVSLAM+nvblox (3D mesh + map->odom + a 2D /map). lidar = slam_toolbox
  # (2D /map). both = visual PRIMARY (owns map->odom + the Nav2 costmap source) plus
  # lidar as a NON-CONFLICTING secondary (its own map_lidar frame + /map_lidar topic)
  # so the two never fight over map->odom — see the explicit WARNING in each launch
  # file. nav_2d_map:=true makes nvblox emit the ESDF 2D /map for RViz + Nav2's costmap.
  # MABEL_SLAM_OWNS_ODOM=1 -> cuVSLAM publishes odom->base_link itself (bench with no
  # wheel odom); default 0 = wheel odom owns odom->base_link, cuVSLAM corrects map->odom.
  local owns_odom="false"; [ "${MABEL_SLAM_OWNS_ODOM:-0}" = 1 ] && owns_odom="true"
  case "$SLAM" in
    lidar)
      warn "lidar SLAM makes a 2D map, NOT the 3D mesh the app's Live tab needs."
      say "slam      lidar_slam (apt Isaac env)   (log: $LOG_SLAM)"
      _spawn_slam "$LOG_SLAM" ros2 launch mabel_slam lidar_slam.launch.py >/dev/null
      ;;
    both)
      say "slam      BOTH → visual_slam PRIMARY (mesh + map->odom + 2D /map)   (log: $LOG_SLAM)"
      _spawn_slam "$LOG_SLAM" ros2 launch mabel_slam visual_slam.launch.py \
        nav_2d_map:=true publish_odom_to_base_tf:="$owns_odom" >/dev/null
      local log2="${LOG_SLAM%.log}_lidar.log"; : > "$log2"
      say "slam      + lidar_slam SECONDARY → map_lidar frame + /map_lidar (no TF fight)   (log: $log2)"
      _spawn_slam "$log2" ros2 launch mabel_slam lidar_slam.launch.py \
        map_frame:=map_lidar map_topic:=/map_lidar >/dev/null
      say "          SLAM starting — drive the robot and the mesh + both maps build live."
      ;;
    *)
      say "slam      visual_slam: cuVSLAM + nvblox mesh + 2D map (apt Isaac env)   (log: $LOG_SLAM)"
      _spawn_slam "$LOG_SLAM" ros2 launch mabel_slam visual_slam.launch.py \
        nav_2d_map:=true publish_odom_to_base_tf:="$owns_odom" >/dev/null
      # Don't BLOCK on the mesh — nvblox only emits geometry as the robot drives and
      # sees new surfaces, so a stationary start would just time out. Teleop + video
      # are already live; the mesh fills into Maps ▸ Live SLAM once you move.
      say "          SLAM starting — drive the robot and the mesh builds live."
      ;;
  esac
}

# Nav2: NavFn (A*) global planner + MPPI controller. Web/app taps go to /goal_pose
# and the bridge feeds Nav2's /mabel_cmd to the base (autonomous-with-takeover) —
# this replaces the browser's old local A*/pure-pursuit. Runs in the conda env
# (mabel_navigation + ros-jazzy-navigation2). Needs a costmap source from SLAM, so
# launch it after the backend + SLAM. --no-nav to skip.
_launch_nav() {
  [ "$DO_NAV" = 1 ] || { say "nav       skipped (--no-nav)"; return; }
  : > "$LOG_NAV"
  local sim_time="true"; [ "$MODE" = real ] && sim_time="false"
  say "nav       Nav2 NavFn A* + MPPI (mabel_navigation)   (log: $LOG_NAV)"
  _spawn_conda "$LOG_NAV" "$WS" \
    ros2 launch mabel_navigation navigation.launch.py use_sim_time:="$sim_time" rviz:=false >/dev/null
}

# Confirm SLAM actually came up — the user asked "is the mesh + SLAM activated?".
# nvblox loads the heavy Isaac ROS stack, so its topics take a few seconds to
# advertise; we poll the graph (conda env — CycloneDDS/domain 0 bridges to the apt
# SLAM, so the conda shell SEES the nvblox topics) and report a clear ✓ / ⚠.
#   visual : the mesh is on /nvblox_node/mesh_pointcloud — the topic advertising
#            means nvblox is ALIVE; the surfaces themselves only fill in as the
#            robot drives and sees new geometry (RViz: "nvblox Mesh (3D)").
#   lidar  : no 3D mesh — check the 2D /map OccupancyGrid instead.
_verify_slam() {
  [ "$DO_SLAM" = 1 ] || { say "slam      not launched (--no-slam) — mesh check skipped"; return; }
  local wait_s="${MABEL_SLAM_WAIT:-40}"
  if [ "$SLAM" = lidar ]; then
    say "slam      verifying lidar SLAM /map (up to ${wait_s}s) …"
    if _wait_topic /map "$wait_s"; then ok "lidar SLAM active — /map is publishing (2D grid, no 3D mesh)."
    else warn "lidar SLAM /map not seen in ${wait_s}s — still loading or failed; check $LOG_SLAM."; fi
    return
  fi
  if [ "$SLAM" = both ]; then
    say "slam      verifying BOTH: nvblox mesh + lidar /map_lidar (up to ${wait_s}s) …"
    if _wait_topic /nvblox_node/mesh_pointcloud "$wait_s"; then ok "visual SLAM active — nvblox mesh up (fills in as you drive)."
    else warn "nvblox mesh not seen in ${wait_s}s — visual SLAM loading/failed; check $LOG_SLAM."; fi
    if _wait_topic /map_lidar "$wait_s"; then ok "lidar SLAM active — /map_lidar publishing (secondary 2D grid)."
    else warn "lidar /map_lidar not seen in ${wait_s}s — check ${LOG_SLAM%.log}_lidar.log."; fi
    return
  fi
  say "slam      verifying nvblox mesh /nvblox_node/mesh_pointcloud (up to ${wait_s}s) …"
  if _wait_topic /nvblox_node/mesh_pointcloud "$wait_s"; then
    ok "SLAM + mesh ACTIVE — nvblox is up; the mesh fills in as you drive (RViz ▸ 'nvblox Mesh (3D)', app ▸ Maps ▸ Live SLAM)."
  else
    warn "nvblox mesh topic not seen in ${wait_s}s — SLAM may still be loading or failed."
    say  "          check $LOG_SLAM (Isaac ROS apt env / scripts/slam_env.sh). Teleop + video are unaffected."
  fi
}

_launch_server() {
  : > "$LOG_BRIDGE"
  # sim  : the bridge OWNS MuJoCo physics (sim:mujoco+ros) — identical controller, motion
  #        model, WBC torque engine and mj_step pacing as the Mac — and ALSO publishes the
  #        canonical ROS state (/clock, /joint_states, /odom_ground_truth, IMU) so the
  #        sensor renderer / SLAM / Nav2 / RViz compose on top unchanged.
  # real : the bridge mirrors the robot's encoders in and publishes the command contract.
  local target="real:ros"; [ "$MODE" = sim ] && target="sim:mujoco+ros"
  say "server    teleop + /map bridge ($target) on ws://0.0.0.0:$WS_PORT   (log: $LOG_BRIDGE)"
  local env_args=(MABEL_DISPATCH_TARGET="$target" MABEL_MAP_STREAM=1
                  MABEL_WS_PORT="$WS_PORT" MABEL_HTTP_PORT="$HTTP_PORT")
  [ -n "$APP_TOKEN" ] && env_args+=(MABEL_TOKEN="$APP_TOKEN")
  _spawn_conda "$LOG_BRIDGE" "$SERVER_DIR" \
    env "${env_args[@]}" python3 sim_teleop_bridge.py --scene "$SCENE" >/dev/null

  # In sim the bridge is the physics authority, so /clock + /joint_states + /odom only
  # appear now. Gate here so SLAM/Nav (launched next) start against a live sim.
  if [ "$MODE" = sim ]; then
    say "          waiting for the sim physics + /odom …"
    if _wait_topic /odom 45; then ok "sim physics up — /clock + /joint_states + /odom live."
    else warn "/odom not seen in 45s — check $LOG_BRIDGE + $LOG_BACKEND (continuing)."; fi
  fi

  if [ "$DO_VIDEO" = 1 ]; then
    : > "$LOG_CAM"
    if [ "$MODE" = sim ]; then
      say "cameras   rendered sim cameras → http://0.0.0.0:$HTTP_PORT + udp:$UDP_VIDEO_PORT   (log: $LOG_CAM)"
      local cam_args=(--scene "$SCENE" --jpeg-quality "$JPEG_QUALITY" --udp-video-port "$UDP_VIDEO_PORT"
                      --max-fps "${MABEL_CAM_FPS:-30}")   # cap render fps so the GPU stays free for nvblox
      [ "$STEREO" = 1 ] && cam_args+=(--stereo)
      # One MuJoCo render → MJPEG (iOS) AND ROS image topics (--ros). --no-cam-ros to skip.
      [ "$DO_CAM_ROS" = 1 ] && cam_args+=(--ros)
      # Render via EGL on the NVIDIA GPU (headless, no display needed) — the same
      # backend the sim sensor node uses. sim_camera_server defaults to MUJOCO_GL=
      # glfw, which (a) needs an X display (fails silently in a service → "no
      # video") and (b) on this box benchmarks ~90 fps vs EGL's ~600 fps. Pinning
      # the NVIDIA EGL vendor dir keeps it off the Intel/llvmpipe software GL.
      # Override the backend with MABEL_CAM_GL / the vendor dir with MABEL_EGL_VENDOR_DIR.
      _spawn_conda "$LOG_CAM" "$SERVER_DIR" \
        env MUJOCO_GL="${MABEL_CAM_GL:-egl}" \
            __EGL_VENDOR_LIBRARY_DIRS="${MABEL_EGL_VENDOR_DIR:-/usr/share/glvnd/egl_vendor.d}" \
        python3 sim_camera_server.py "${cam_args[@]}" >/dev/null
    else
      # REAL robot: the ROBOT's cameras are served by the camera_hal node that
      # hardware.launch.py already started (sensors/ros_camera_node.py --serve): it
      # owns the front D435i + head ZED-M and fans them out as MJPEG on :$HTTP_PORT
      # -> /camera/main + /camera/head_left (ZED head) + /camera/front. We do NOT
      # launch real_camera_server.py here: that opens the OPERATOR's local USB cams
      # via OpenCV (a Mac teleop station), which on the robot grabs the wrong
      # /dev/video, FIGHTS camera_hal for :$HTTP_PORT, and serves bogus streams.
      # The web/iOS pull /camera/main (head) -> live; /camera/wrist_left|wrist_right
      # -> NOT served by camera_hal, so the client shows "not connected" (correct
      # when the wrist cams aren't wired). Bridging the wrist/body v4l cams' ROS
      # image topics to MJPEG is a separate enhancement.
      say "cameras   robot cameras via camera_hal (hardware.launch --serve) → http://0.0.0.0:$HTTP_PORT"
    fi
  else
    say "cameras   skipped (--no-video)"
  fi
}

# Desktop GUIs — same as run.sh: the Command Monitor (live operator-intent +
# client view, ws://127.0.0.1:WS_PORT/monitor) on by default, plus an optional
# MuJoCo mirror window. Both run at lowest priority (nice -n 19) so they never
# steal cycles from the sim/SLAM, and both degrade gracefully (no display / no
# tkinter → they just log and exit; the stack keeps running).
_launch_gui() {
  if [ "$DO_MONITOR" = 1 ]; then
    : > "$LOG_MON"
    local mon=(--host 127.0.0.1 --port "$WS_PORT")
    [ -n "$APP_TOKEN" ] && mon+=(--token "$APP_TOKEN")
    [ -n "${MABEL_HOSTS:-}" ] && mon+=(--hosts "$MABEL_HOSTS")
    say "monitor   Command Monitor GUI → ws://127.0.0.1:$WS_PORT/monitor   (log: $LOG_MON)"
    _spawn_conda "$LOG_MON" "$SERVER_DIR" nice -n 19 python3 monitor_gui.py "${mon[@]}" >/dev/null
  fi
  if [ "$DO_VIEWER" = 1 ]; then
    : > "$LOG_VIEW"
    say "viewer    desktop MuJoCo mirror window (view_robot.py)   (log: $LOG_VIEW)"
    # Force the NVIDIA GPU for the GLFW window on hybrid-graphics laptops
    # (Optimus) — else the viewer can segfault picking the Intel/llvmpipe GL.
    _spawn_conda "$LOG_VIEW" "$SERVER_DIR" \
      env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
      python3 view_robot.py --scene "$SCENE" >/dev/null
  fi
}

_advertise() {
  # Best-effort LAN discovery so the iOS app finds this box without typing an IP.
  command -v avahi-publish-service >/dev/null 2>&1 || return
  avahi-publish-service "MABEL $( [ "$MODE" = sim ] && echo Sim || echo Robot ) ($(hostname -s))" \
    _mabel-teleop._tcp "$WS_PORT" >/dev/null 2>&1 & PIDS+=($!)
  say "bonjour   advertising _mabel-teleop._tcp on :$WS_PORT (iOS auto-discovers)"
}

_lan_ip() {
  hostname -I 2>/dev/null | awk '{print $1}' | grep -E '^[0-9]' || echo 127.0.0.1
}

_banner() {
  local ip; ip="$(_lan_ip)"
  cat <<EOF

  ──────────────────────────────────────────────────────────────────────────
    M A B E L   R O S   S E R V E R   ·   ${MODE^^}   ·   ${SLAM} SLAM
  ──────────────────────────────────────────────────────────────────────────
    iOS / clients
      teleop         ws://${ip}:${WS_PORT}/teleop
      live map       ws://${ip}:${MABEL_MAP_PORT:-9092}   (mesh + odom, own port)
      video          http://${ip}:${HTTP_PORT}/camera/*   (udp://${ip}:${UDP_VIDEO_PORT})
      in the app     Settings ▸ set the host to  ${ip}   (or it auto-discovers)
      then           Maps ▸ Live SLAM   shows the live mesh + robot odom

    on this desktop
      RViz           $( [ "$DO_RVIZ" = 1 ] && echo "OPEN — robot + sensors + live nvblox SLAM mesh" || echo "off (--rviz to open)")
      MuJoCo window  $( [ "$DO_VIEWER" = 1 ] && echo "OPEN — view_robot.py, the robot's TRUE state" || echo "off (--viewer to open)")
      SLAM / mesh    $( [ "$DO_SLAM" = 1 ] && echo "${SLAM} SLAM running — mesh builds as you drive" || echo "off (--no-slam)")

    logs:  backend=$LOG_BACKEND  slam=$LOG_SLAM  bridge=$LOG_BRIDGE  cam=$LOG_CAM
    Ctrl-C here stops everything.
  ──────────────────────────────────────────────────────────────────────────

EOF
}

# ── entrypoint ─────────────────────────────────────────────────────────────────
ros_server_main() {
  [ -n "${MODE:-}" ] || die "MODE not set (use sim_ros_server.sh / real_ros_server.sh)."
  _parse_args "$@"
  _preflight
  trap cleanup EXIT INT TERM

  say "root    $MABEL_ROOT"
  say "ws      $WS"
  [ "$MODE" = sim ] && say "scene   $SCENE"

  _reap_stale            # kill any previous run first (frees :9090/:8080, ROS topics)
  _launch_backend        # robot + sensors + /odom
  _launch_server         # bridge (/teleop + /map) + cameras → video up FAST
  _launch_gui            # Command Monitor GUI (+ optional --viewer)
  _launch_slam           # SLAM: nvblox mesh + map→odom
  _launch_nav            # Nav2 (NavFn A* + MPPI) — web/app goals drive the base
  _verify_slam           # confirm the mesh + SLAM actually activated (✓ / ⚠)
  _advertise
  _banner

  # Hold the foreground; cleanup() runs on Ctrl-C / any exit.
  while true; do sleep 3600 & wait $!; done
}

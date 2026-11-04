#!/usr/bin/env bash
# =============================================================================
# real_run.sh — bring up the REAL MABEL robot (HARDWARE) on THIS machine.
# =============================================================================
# The robot-SIDE bring-up — the same thing you'd run on the Thor Jetson, runnable
# here on this Ubuntu box (standing in for the robot) to test that every module's
# ROS connection comes alive and to see the sensors + robot state in RViz:
#
#   • base      swerve drive                   (mabel_base_hardware — a THIN relay;
#               the bridge owns the Teensy. It also FORWARDS /lift/command onward.)
#   • lift      standing-desk column           (hardware_bridge lift subsystem -> Pico;
#               mabel_base_hardware has NO lift driver — swerve_node only RELAYS
#               /mabel_cmd.lift_pos -> /lift/command (Float64, metres of travel).
#               OWNED SINCE 2026-07-18: this script now launches the bridge with
#               MABEL_HW_LIFT=true (--no-lift opts out; MABEL_HW_LIFT_PORT sets the
#               Pico serial, default /dev/ttyACM0). Idle-safe with no Pico plugged;
#               auto_home stays false so bring-up never moves it. ⚠ the SYSTEMD unit
#               still leaves MABEL_HW_LIFT unset — add it there once verified live.)
#   • arms      bimanual arms                  (mabel_arms_hardware — thin relay)
#   • body      neck / head                    (mabel_body_hardware — thin relay now:
#               the hardware_bridge HAL owns the neck's Dynamixel bus (MABEL_HW_NECK)
#               and serves /body/*; this ROS node only monitors + mirrors e-stop.)
#   • hands     both ORCA hands                (mabel_hand_hardware)
#   • cameras   front D435i + head ZED-M + wrist D405/UVC   (hardware_bridge cameras/)
#   • audio     USB sound card + speakers/mic  (hardware_bridge audio subsystem —
#               JMTek dongle + 8Ω/5W speaker pair, VID:PID 0c76:1203. Found by USB
#               IDENTITY, not by port, so it reconnects on ANY socket.
#               mabel_audio_hardware is a thin client.)
#   • SLAM      visual cuVSLAM + nvblox (front cam) → 3D mesh + 2D /map + map→odom.
#               When a LiDAR is DETECTED (a /scan appears), 2D slam_toolbox takes
#               over the 2D /map + map→odom (the mature CPU 2D-nav source) and
#               nvblox drops to 3D-mesh-only — so the two never fight over map→odom.
#   • Nav2      NavFn A* + MPPI                (mabel_navigation)
#   • web/iOS   teleop + live /map (mesh+odom) bridge :9090/:9092 + MJPEG video :8080
#   • studio    Control Center web GUI on :8094 (web_gui/control_center —
#               visualizer + teleop client + server monitor). If thor's systemd
#               studio service already serves :8094 it is REUSED, not duplicated.
#   • RViz      robot model + TF + every camera + the SLAM mesh/map
#
# A module with no hardware plugged in stays IDLE (it never crashes the bring-up);
# the pinned status dashboard + RViz show which subsystems are live. Ctrl-C here
# stops everything.
#
# This IS the unified real-robot server (real_ros_server.sh is now a thin alias to
# it). One command brings up the robot HAL + SLAM + Nav2 + RViz AND everything the
# iOS / Vision Pro / web app needs: teleop control, robot state, control modes, the
# MJPEG camera video (:8080, served by camera_hal), and the live SLAM mesh+odom on
# /map (:9090/:9092). The gateway architecture is preserved: ROS does NOT own the
# hardware directly — each subsystem driver (camera_hal, swerve/lift, arms, neck,
# hands) owns its OWN device and fans out over DDS (+ MJPEG for video); the app
# bridge (real:ros) only MIRRORS that state, it is never in the motor loop.
#
# Run it from the repo root (or from server/):
#   ./server/real_run.sh                      # everything: HW + SLAM + Nav2 + RViz + app server
#   ./server/real_run.sh --no-slam            # skip the GPU SLAM (just motors + cameras)
#   ./server/real_run.sh --lidar-slam on      # FORCE 2D slam_toolbox (wait for /scan)
#   ./server/real_run.sh --no-lidar-slam      # never run slam_toolbox (visual owns 2D)
#   ./server/real_run.sh --no-web             # skip the iOS/web teleop + /map bridge
#   ./server/real_run.sh --no-studio          # skip the Control Center web GUI (:8094)
#   ./server/real_run.sh --no-watch           # skip the periodic hardware status watchdog
#   ./server/real_run.sh --no-tui             # plain scrolling output — skip the retro
#                                             #   cockpit TUI this terminal becomes by
#                                             #   default (pixel MABEL + joints/radar/
#                                             #   hw-watchdog/logs; q = Ctrl-C). Auto-
#                                             #   off without a TTY (systemd boot).
#   ./server/real_run.sh --no-audio           # skip the speaker/mic (:9610 says why)
#   ./server/real_run.sh --no-nav             # skip Nav2
#   ./server/real_run.sh --rviz / --no-rviz   # force RViz on / off. Default AUTO: RViz
#                                             #   opens only when a usable display answers
#                                             #   (a human debugging at the desk); headless
#                                             #   (SSH / systemd boot) runs without it and
#                                             #   nothing else changes.
#   ./server/real_run.sh --hands left         # FORCE only the LEFT hand (default: auto —
#                                             #   detect plugged-in adapters by USB serial)
#   ./server/real_run.sh --only-hands         # ONLY the hands (base/arms/body/cams off)
#   ./server/real_run.sh --only-cameras       # ONLY the cameras + SLAM
#   ./server/real_run.sh --no-base --no-arms  # e.g. test neck + hands + cameras
#   ./server/real_run.sh -h                   # all options
#
# Auto-start arbitration: Thor boots mabel-hardware-bridge.service (the single HAL)
# and mabel-bringup.service (THIS script, headless). A MANUAL run must not fight
# them, so it takes over cleanly: it writes /tmp/mabel_manual_run.lock, SIGTERMs the
# service's real_run instance (its cleanup reaps its ROS nodes + app server), and
# systemd's respawned instance sees the lock and STANDS BY (idle, owns nothing)
# until the manual run exits — then it resumes the normal headless bring-up on its
# own. The hardware bridge service is NEVER duplicated: both runs reuse it, and any
# stray process squatting on the hand serial adapters or the base Teensy link is
# reaped so exactly ONE bridge owns the hardware.
#
# Prereqs: the workspace is built (ros/dev_up.sh) and, for SLAM, the Isaac
# ROS apt stack is installed (scripts/slam_env.sh / project_isaac_native_jazzy).
# =============================================================================
set -o pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this lives in server/
MABEL_ROOT="${MABEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WS="${MABEL_WS:-$MABEL_ROOT/ros}"
export MABEL_WS="$WS"                 # hardware.launch.py's SLAM shell-out reads this
ENV_SH="$WS/scripts/mabel_env.sh"     # conda RoboStack ROS 2 (bring-up + Nav2)

# ── l_j4 note (2026-07-28): the original left-elbow motor was hardware-faulty
# (miscommutation — see hardware_bridge/experiments/exp_j4_shake.py) and was
# QUARANTINED here (gains zeroed + MABEL_ARM_FF_SKIP=l_j4). The motor was REPLACED
# and recalibrated the same day, so the quarantine is lifted; the env knobs remain
# available for the next faulty joint (MABEL_HW_<SIDE>_ARM_KP_J<n>=0 +
# MABEL_ARM_FF_SKIP=<fw_joint> — plant.py drops that joint's gravity FF).

# ── config (env-overridable) ───────────────────────────────────────────────────
DO_SLAM=1; DO_NAV=1
DO_RVIZ="${MABEL_RVIZ:-auto}"  # auto: RViz iff a usable display answers (resolved below);
                               # --rviz / --no-rviz (or MABEL_RVIZ=1/0) force it.
DO_WATCH=1                     # periodic hardware status watchdog (+ RealSense uvcvideo self-heal)
WATCH_INTERVAL="${MABEL_WATCH_INTERVAL:-10}"   # seconds between hardware status reports
DO_TUI="${MABEL_TUI:-1}"       # retro cockpit TUI in THIS terminal (tui/mabel_cockpit.py):
                               # pixel MABEL + joints/radar/hw-watchdog/logs; q quits the
                               # whole bring-up. Auto-disabled without a TTY (systemd/SSH
                               # pipes), --no-tui/MABEL_TUI=0 force plain output.
LOG_WATCH=/tmp/mabel_hw_watch.log   # watchdog output in TUI mode (cockpit tails+parses it)
DO_WEB=1                       # also serve the web/iOS app: the teleop + /map (mesh
                               # +odom) bridge on :9090/:9092 (camera_hal already
                               # serves the video MJPEG on :8080). --no-web to skip.
DO_STUDIO=1                    # also serve the Control Center web GUI on :$STUDIO_PORT
                               # (reuses an already-listening studio, e.g. thor's
                               # systemd service). --no-studio to skip.
STUDIO_PORT="${MABEL_CC_PORT:-8094}"
SCENE="${MABEL_SCENE:-$MABEL_ROOT/simulation/mabel_mujoco/scenes/mabel_home.xml}"
WS_PORT="${MABEL_WS_PORT:-9090}"
HTTP_PORT="${MABEL_HTTP_PORT:-8080}"
APP_TOKEN="${MABEL_TOKEN:-}"
LOG_BRIDGE="/tmp/mabel_bridge.log"
SLAM_ODOM="${MABEL_SLAM_ODOM:-auto}"   # who owns odom->base_link.
SLAM_ODOM_FORCED=0; [ "$SLAM_ODOM" != auto ] && SLAM_ODOM_FORCED=1   # env/flag forced the owner
                               #   auto (default): resolved from the base below —
                               #     base ON  -> wheel swerve_odometry owns odom->base_link
                               #                 (cuVSLAM/slam_toolbox only correct map->odom);
                               #     base OFF -> cuVSLAM owns odom->base_link (camera-only bench).
                               #   --wheel-odom / --slam-odom force false / true.
                               # NOTE: 2D slam_toolbox REQUIRES wheel odom->base_link, so the
                               # LiDAR dual-SLAM path only engages when the base owns odom.
LIDAR_SLAM="${MABEL_LIDAR_SLAM:-auto}" # auto: run 2D slam_toolbox IFF a /scan is detected
                               # (LiDAR onboard); on: force it (wait longer for /scan);
                               # off: never (visual cuVSLAM/nvblox owns the 2D /map too).
BASE=true; ARMS=true; BODY=true; HANDS=true; SENSORS=true; LIDAR=true
AUDIO=true
# LIFT: bridge-owned standing-desk column (Pico serial, /lift/command metres).
# Default ON since 2026-07-18 (VP whole-body teleop's head-height follow needs
# it); SAFE BY DESIGN — with no Pico on MABEL_HW_LIFT_PORT the subsystem
# advertises the /lift contract and stays idle, and auto_home defaults false so
# bring-up never moves it. --no-lift opts out.
LIFT_HW=true
HAND_SIDES="${MABEL_HAND_SIDES:-auto}"        # auto | left | right | both
TEENSY_IP="${MABEL_TEENSY_IP:-192.168.124.10}"
# ORCA hand adapters are told apart BY USB SERIAL NUMBER (the two adapters are
# identical CH340s; the serial is the only stable discriminator). The bridge resolves
# port 'auto' through its serial->side registry (hardware_bridge/config.py HANDS
# adapter_serials), so with one adapter plugged in the connected hand always comes up
# as the CORRECT side — never first-come-first-served. The serials here mirror that
# registry so the preflight can REPORT which hand is physically present. Find a new
# adapter's serial with: ls -l /dev/serial/by-id/  (usb-1a86_USB_Single_Serial_<SERIAL>).
# An explicit device path still pins a side; 'none' turns it off.
LEFT_PORT="${MABEL_LEFT_PORT:-auto}"          # auto (registry) | by-id path | none
RIGHT_PORT="${MABEL_RIGHT_PORT:-auto}"
HAND_SIDES_FORCED=0; [ "$HAND_SIDES" != auto ] && HAND_SIDES_FORCED=1   # env/--hands forced sides
LEFT_SERIAL="${MABEL_HW_HANDS_LEFT_SERIAL:-}"                # left adapter serial (unknown yet)
RIGHT_SERIAL="${MABEL_HW_HANDS_RIGHT_SERIAL:-0000000002}"    # right adapter serial (verified 17/17)
NAV_PREPOSE="${MABEL_NAV_PREPOSE:-false}"     # false: do NOT auto-tuck arms on the bench
# The sound card is identified by USB VID:PID, NOT by serial or port: the JMTek dongle
# (8Ω/5W speaker pair + mic; replaced the MZ-631M bar 2026-07-27) reports the hard-coded
# serial "000000000000" (not unique), so the hands' serial-registry trick would mis-bind
# a second dongle. The bridge maps VID:PID -> ALSA card id through sysfs
# (mabel_hw/drivers/audio.py card_for_usb_id), which is why it reconnects on ANY port.
AUDIO_USB_ID="${MABEL_HW_AUDIO_USB_ID:-0c76:1203}"   # JMTek USB PnP Audio Device
AUDIO_CARD="${MABEL_HW_AUDIO_CARD:-}"                # blank = resolve by VID:PID

LOG_DIR=/tmp
LOG_HW="$LOG_DIR/mabel_real_hw.log"           # bring-up (HAL + cameras + lidar + RViz)
LOG_NAV="$LOG_DIR/mabel_real_nav.log"
LOG_SLAM="$LOG_DIR/mabel_real_slam.log"        # visual SLAM (cuVSLAM/nvblox)
LOG_SLAM_2D="$LOG_DIR/mabel_real_slam2d.log"   # 2D LiDAR SLAM (slam_toolbox)
LOG_LODOM="$LOG_DIR/mabel_real_laser_odom.log" # LiDAR odometry (odom->base_link)
LOG_HWBRIDGE="$LOG_DIR/mabel_hw_bridge.log"    # the hardware bridge (HAL) — base Teensy link
LOG_STUDIO="$LOG_DIR/mabel_control_center.log" # the Control Center web GUI (:8094)
HWBRIDGE_DIR="$MABEL_ROOT/hardware_bridge"
STUDIO_SH="$MABEL_ROOT/web_gui/control_center/run.sh"
BRIDGE_OURS=0                                  # 1 if THIS run started the bridge (so we reap it)
STUDIO_OURS=0                                  # 1 if THIS run started the studio (so we reap it)

# SLAM + the lidar driver live in the apt Isaac ROS stack (cuVSLAM/nvblox/slam_toolbox
# are NOT in the conda RoboStack env this script's bring-up uses), so they are spawned
# through scripts/slam_env.sh (apt ROS + CycloneDDS/domain 0 so they discover the conda
# camera/scan topics over DDS) instead of mabel_env.sh.
SLAM_ENV="$WS/scripts/slam_env.sh"

PIDS=()

# ── pretty printing ─────────────────────────────────────────────────────────────
say()  { printf '  \033[2m▸\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  # Print the header comment (lines 3.. up to the closing # === banner). Starting at
  # 5 skips the OPENING # === banner on line 4, which would otherwise end the range.
  sed -n '3p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  sed -n '5,/^# ===/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

# ── argument parsing ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --no-slam)     DO_SLAM=0; shift ;;
    --no-nav)      DO_NAV=0; shift ;;
    --rviz)        DO_RVIZ=1; shift ;;
    --no-rviz)     DO_RVIZ=0; shift ;;
    --no-web)      DO_WEB=0; shift ;;            # don't serve the web/iOS bridge
    --no-studio)   DO_STUDIO=0; shift ;;         # don't serve the Control Center web GUI
    --no-audio)   AUDIO=false; shift ;;
    --no-watch)    DO_WATCH=0; shift ;;          # don't run the periodic hw status watchdog
    --wheel-odom)  SLAM_ODOM=false; SLAM_ODOM_FORCED=1; shift ;;  # force: wheel owns odom->base_link
    --slam-odom)   SLAM_ODOM=true;  SLAM_ODOM_FORCED=1; shift ;;  # force: cuVSLAM owns odom->base_link
    --lidar-slam)  LIDAR_SLAM="$2"; shift 2 ;;  # auto|on|off — 2D slam_toolbox engagement
    --no-lidar-slam|--no-slam-toolbox) LIDAR_SLAM=off; shift ;;
    --no-tui|--plain) DO_TUI=0; shift ;;         # plain scrolling output (no cockpit)
    --no-base)     BASE=false; shift ;;
    --no-arms)     ARMS=false; shift ;;
    --no-lift)     LIFT_HW=false; shift ;;
    --no-body|--no-neck) BODY=false; shift ;;
    --no-hands)    HANDS=false; shift ;;
    --no-cameras|--no-sensors) SENSORS=false; shift ;;
    --no-lidar)    LIDAR=false; shift ;;
    --hands)       HAND_SIDES="$2"; HAND_SIDES_FORCED=1; shift 2 ;;
    --only-hands)    BASE=false; ARMS=false; BODY=false; SENSORS=false; LIDAR=false; AUDIO=false; DO_SLAM=0; DO_NAV=0; shift ;;
    --only-cameras)  BASE=false; ARMS=false; BODY=false; HANDS=false; LIDAR=false; AUDIO=false; DO_NAV=0; shift ;;
    --only-base)     ARMS=false; BODY=false; HANDS=false; SENSORS=false; LIDAR=false; DO_SLAM=0; DO_NAV=0; shift ;;
    --prepose)     NAV_PREPOSE=true; shift ;;
    --no-prepose)  NAV_PREPOSE=false; shift ;;
    --teensy-ip)   TEENSY_IP="$2"; shift 2 ;;
    --left-port)   LEFT_PORT="$2"; shift 2 ;;
    --right-port)  RIGHT_PORT="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *) warn "unknown arg: $1"; echo "  (try -h)"; exit 1 ;;
  esac
done

# ── single-instance arbitration (systemd auto-start vs a manual run) ────────────
# Thor auto-launches this script at boot (mabel-bringup.service, Restart=always).
# A manual run cannot just pkill the service instance: systemd respawns it 5 s later
# and the two runs trade :9090/:8080 + the ROS graph back and forth forever. So the
# two instances arbitrate through a lock file (no root needed — same user):
#   • MANUAL run: writes $MANUAL_LOCK (its pid), SIGTERMs the service instance (its
#     cleanup reaps its ROS nodes + app server), and waits for it to die. systemd
#     respawns the service, whose NEW instance sees the live lock and STANDS BY.
#   • SERVICE instance: while a LIVE manual run holds the lock, it idles (owns
#     nothing); when the lock is released (or its pid is dead = stale), it proceeds
#     with the normal headless bring-up. So ending a manual run automatically
#     returns the robot to the auto-started stack.
# The hardware bridge service is NOT touched — it is the single HAL and every run
# reuses it (_ensure_bridge); duplicates/strays are reaped separately.
BRINGUP_SVC="mabel-bringup.service"
MANUAL_LOCK="${MABEL_MANUAL_LOCK:-/tmp/mabel_manual_run.lock}"
MANUAL_LOCK_OURS=0
_svc_mainpid() { systemctl show -p MainPID --value "$BRINGUP_SVC" 2>/dev/null | tr -dc '0-9'; }

if [ "$(_svc_mainpid)" = "$$" ]; then
  # We ARE the systemd service instance: stand by while a live manual run owns the robot.
  _standby_said=0
  while :; do
    _lockpid="$(cat "$MANUAL_LOCK" 2>/dev/null | tr -dc '0-9')"
    [ -n "$_lockpid" ] || break
    if ! kill -0 "$_lockpid" 2>/dev/null; then
      rm -f "$MANUAL_LOCK"
      say "stale manual-run lock (pid $_lockpid is gone) — removed; proceeding with bring-up."
      break
    fi
    if [ "$_standby_said" = 0 ]; then
      say "manual real_run.sh (pid $_lockpid) owns the robot — service instance STANDING BY until it exits."
      _standby_said=1
    fi
    sleep 5
  done
  [ "$_standby_said" = 1 ] && say "manual run exited — service instance resuming the headless bring-up."
else
  # Manual run: grab the lock, then take over from the auto-start service if it is up.
  echo "$$" > "$MANUAL_LOCK" 2>/dev/null && MANUAL_LOCK_OURS=1
  # Minimal trap so an early `die` still releases the lock (replaced by the full
  # cleanup trap in the run section below, which also removes it).
  trap '[ "$MANUAL_LOCK_OURS" = 1 ] && rm -f "$MANUAL_LOCK"' EXIT
  _svc_pid="$(_svc_mainpid)"
  if [ -n "$_svc_pid" ] && [ "$_svc_pid" != 0 ] && [ "$_svc_pid" != "$$" ] && kill -0 "$_svc_pid" 2>/dev/null; then
    say "auto-start $BRINGUP_SVC is running (pid $_svc_pid) — taking over; it stands by until this run exits."
    kill -TERM "$_svc_pid" 2>/dev/null || true
    _i=0
    while kill -0 "$_svc_pid" 2>/dev/null && [ "$_i" -lt 30 ]; do sleep 1; _i=$((_i + 1)); done
    if kill -0 "$_svc_pid" 2>/dev/null; then
      warn "service instance (pid $_svc_pid) ignored SIGTERM for 30 s — SIGKILLing it (systemd respawns it into standby)."
      kill -9 "$_svc_pid" 2>/dev/null || true
      sleep 1
    else
      ok "service instance exited (its nodes reaped) — the respawned instance will stand by."
    fi
  fi
fi

# ── resolve RViz (auto: on iff a usable display) ────────────────────────────────
# Headless (SSH session, systemd boot: no DISPLAY) → RViz OFF, so the bring-up never
# wastes a node on a Qt "could not connect to display" crash loop. A human at the
# desk (DISPLAY set and the X server answering) → RViz ON. A set-but-dead DISPLAY
# (stale SSH forward) counts as headless. --rviz / --no-rviz force either way.
_display_ok() {
  if [ -n "${DISPLAY:-}" ]; then
    command -v xset >/dev/null 2>&1 || return 0            # can't probe — trust $DISPLAY
    timeout 3 xset -display "$DISPLAY" q >/dev/null 2>&1 && return 0
    return 1                                               # DISPLAY set but not answering
  fi
  [ -n "${WAYLAND_DISPLAY:-}" ] && [ -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/$WAYLAND_DISPLAY" ]
}
if [ "$DO_RVIZ" = auto ]; then
  if _display_ok; then
    DO_RVIZ=1; say "display detected (${DISPLAY:-wayland $WAYLAND_DISPLAY}) — RViz ON (--no-rviz to skip)."
  else
    DO_RVIZ=0; say "no usable display — RViz OFF (headless; pass --rviz to force)."
  fi
fi

# ── resolve the cockpit TUI (needs an interactive terminal + python curses) ─────
# The systemd bring-up has no TTY, so the service instance always stays plain.
if [ "$DO_TUI" = 1 ]; then
  if [ -t 0 ] && [ -t 1 ] && command -v python3 >/dev/null 2>&1 \
       && python3 -c 'import curses' >/dev/null 2>&1; then
    say "cockpit   this terminal becomes the retro cockpit TUI once bring-up is launched (q quits; --no-tui for plain)."
  else
    DO_TUI=0
  fi
fi

# ── hand auto-detection (preflight report + HAND_SIDES=auto resolution) ─────────
# Scan /dev/serial/by-id for the registered adapter serials so the bring-up REPORTS
# which hand is physically present, and resolve HAND_SIDES=auto to what is plugged in.
# This is informational + side selection only — the authoritative detection (and the
# safety that a side never grabs the other side's adapter) lives in the bridge's
# serial->side registry; ports stay 'auto' and the bridge hot-plugs a late adapter.
_hand_dev() {   # serial -> by-id path (empty if absent)
  [ -n "$1" ] || return 0
  ls /dev/serial/by-id/*"$1"* 2>/dev/null | head -1
}
_detect_hands() {
  [ "$HANDS" = true ] || return 0
  local ldev rdev
  ldev="$(_hand_dev "$LEFT_SERIAL")"; rdev="$(_hand_dev "$RIGHT_SERIAL")"
  [ -n "$ldev" ] && ok "left hand adapter detected  (serial $LEFT_SERIAL → $ldev)"
  [ -n "$rdev" ] && ok "right hand adapter detected (serial $RIGHT_SERIAL → $rdev)"
  if [ "$HAND_SIDES" = auto ]; then
    if   [ -n "$ldev" ] && [ -n "$rdev" ]; then HAND_SIDES=both
    elif [ -n "$rdev" ];                   then HAND_SIDES=right
    elif [ -n "$ldev" ];                   then HAND_SIDES=left
    else
      HAND_SIDES=both     # nothing yet: keep both armed so a hot-plug is picked up
      warn "no hand adapter detected — hands stay IDLE; plugging one in hot-connects (bridge 2s watchdog)."
    fi
    say "hands     auto → $HAND_SIDES"
  elif [ -z "$ldev" ] && [ -z "$rdev" ]; then
    warn "hands($HAND_SIDES) requested but no registered adapter is on USB — idle until plugged in."
  fi
}

# ── resolve odom ownership ──────────────────────────────────────────────────────
# odom->base_link must have EXACTLY one owner. auto: the base owns it when it's on
# (so wheel swerve_odometry feeds slam_toolbox + cuVSLAM corrects map->odom on top);
# only a camera-only bench (base off) lets cuVSLAM own odom from the static camera.
if [ "$SLAM_ODOM" = auto ]; then
  [ "$BASE" = true ] && SLAM_ODOM=false || SLAM_ODOM=true
fi

# ── audio device (USB sound card) ───────────────────────────────────────────────
# Identified by USB VID:PID and NOT by port or serial — this dongle reports the
# hard-coded serial "000000000000", and its ALSA card INDEX is arrival-order. The
# card *id* comes from the device's own product string, so the same card answers to
# the same name on any socket; the bridge maps VID:PID -> card id through sysfs and
# re-resolves every 2 s, which is what makes a replug reconnect with no restart.
_audio_card() {   # -> ALSA card id of AUDIO_USB_ID (empty if absent), any port
  local vid="${AUDIO_USB_ID%%:*}" pid="${AUDIO_USB_ID##*:}" c d id
  for c in /sys/class/sound/card*; do
    [ -e "$c/device" ] || continue
    id="$(cat "$c/id" 2>/dev/null)"
    d="$(readlink -f "$c/device")"
    while [ "$d" != "/" ] && [ -n "$d" ]; do
      if [ -f "$d/idVendor" ]; then
        if [ "$(cat "$d/idVendor" 2>/dev/null)" = "$vid" ] &&
           [ "$(cat "$d/idProduct" 2>/dev/null)" = "$pid" ]; then
          echo "$id"; return 0
        fi
        break                      # this card's USB device isn't ours
      fi
      d="$(dirname "$d")"
    done
  done
  return 0
}

_audio_port() {   # -> the USB path it happens to sit on today (display only)
  local d
  for d in /sys/bus/usb/devices/*/; do
    [ -f "$d/idVendor" ] || continue
    [ "$(cat "$d/idVendor" 2>/dev/null):$(cat "$d/idProduct" 2>/dev/null)" = "$AUDIO_USB_ID" ] \
      && { basename "$d"; return 0; }
  done
  return 0
}

_detect_audio() {
  [ "$AUDIO" = true ] || return 0
  local card port
  card="$(_audio_card)"; port="$(_audio_port)"
  if [ -n "$card" ]; then
    AUDIO_CARD="${AUDIO_CARD:-$card}"
    ok "audio card detected       (USB $AUDIO_USB_ID on port $port → ALSA card '$card')"
    say "  found by USB identity, not by port — replug it anywhere and it reconnects."
  elif lsusb 2>/dev/null | grep -qi "$AUDIO_USB_ID"; then
    # Enumerated but no PCM bound: snd-usb-audio didn't claim it. lsusb alone
    # would call this healthy, which is why the probe checks BOTH.
    warn "audio card is on USB ($AUDIO_USB_ID) but NO ALSA card bound to it —"
    warn "  → snd-usb-audio did not claim it. Replug it; if it persists: dmesg | grep -i snd"
    warn "  continuing — audio stays IDLE and the :9610 plane explains why."
  else
    warn "no USB audio card detected ($AUDIO_USB_ID) — audio stays IDLE; plugging"
    warn "  one in hot-connects it (the bridge re-resolves every 2s, any port)."
  fi
}

# ── env subshell launcher ───────────────────────────────────────────────────────
# Each group runs `( source <env>; exec <cmd> )` so $! is the launched process
# itself — a later SIGINT reaches `ros2 launch`, which tears its nodes down cleanly.
# _spawn uses the conda RoboStack env (mabel_env.sh: HAL, Nav2, app bridge);
# _spawn_slam uses the apt Isaac ROS env (slam_env.sh: cuVSLAM/nvblox/slam_toolbox).
_spawn() {   # logfile -- cmd...
  local logf="$1"; shift
  : > "$logf"
  ( source "$ENV_SH" >>"$logf" 2>&1 || exit 1
    exec "$@" ) >>"$logf" 2>&1 &
  PIDS+=($!)
}

_spawn_slam() {   # logfile -- cmd...   (sources the apt Isaac ROS env, not conda)
  local logf="$1"; shift
  : > "$logf"
  ( source "$SLAM_ENV" >>"$logf" 2>&1 || exit 1
    exec "$@" ) >>"$logf" 2>&1 &
  PIDS+=($!)
}

# ── hardware bridge (the single HAL) ─────────────────────────────────────────────
# The base now goes THROUGH ~/open_mabel/hardware_bridge: it OWNS the Teensy link + the
# validated ps5_teleop IK/calibration and republishes encoder state on
# /mabel/hw/base/joint_states, which the ROS base_bridge_client (in the bring-up)
# relays into the stack. Start it here (base subsystem) unless it is already up
# (systemd auto-start or a manual run) — idempotent so we never double-own the Teensy.
# Arms/body/hands/cameras still come up via their own launch nodes for now.
# Preflight the base Ethernet link so a NO-CARRIER cable is reported LOUDLY here
# (with the fix) instead of surfacing later as a vague "/odom not seen" timeout. The
# bridge auto-reconnects, so we only WARN and continue — plug in the cable and it picks
# the Teensy back up with no relaunch.
# The Teensys are PORT-AGNOSTIC (2026-07-23): each one may be plugged into the direct
# PCIe NIC (enP2p1s0), one of Thor's QSFP28 break-out lanes (mgbe0_0..mgbe3_0), or the
# Ethernet EXPANSION BOARD — a switch that puts several devices behind ONE port (it
# negotiates 10G on the lane, so a 100M link speed means a Teensy is plugged straight
# in, 10G means the expansion board is). We therefore never trust a pinned interface:
# `_ensure_lane` probes every carrier-up wired port for the Teensy's ARP reply — which
# works even before the host has an address in that subnet — then assigns the host /24
# on the answering port with `nmcli device modify` (RUNTIME-ONLY: the persistent NM
# profiles, "Wired connection 1"/"teensy-base"/"teensy-arms", are never edited; on
# reboot mabel-bringup re-runs this script, which re-discovers the port). If the routed
# ping still fails after that, another carrier-up port is holding the same /24 (a stale
# profile on the old home port) and shadowing the route — we strip the subnet there,
# runtime-only, and retest.
BASE_NM_CON="${MABEL_BASE_NM_CON:-Wired connection 1}"
BASE_IFACE="${MABEL_BASE_IFACE:-enP2p1s0}"      # home-port HINT for diagnostics only

_wired_ifaces() {
  # Physical wired candidates with carrier — PCIe NIC, QSFP lanes, expansion board alike.
  local n
  for n in /sys/class/net/*; do
    n="$(basename "$n")"
    case "$n" in lo|can*|wl*|usb*|docker*|veth*|l4tbr*|tailscale*|tun*) continue ;; esac
    [ -e "/sys/class/net/$n/device" ] || continue
    [ "$(cat "/sys/class/net/$n/carrier" 2>/dev/null)" = 1 ] && echo "$n"
  done
}

_iface_answers() {
  # Does $2 ARP-answer on port $1? The ping's ARP who-has goes out even when the host
  # has no address in the target subnet, and the Teensy's lwIP replies regardless of
  # the probe's source — so this finds the device without touching any config.
  local ifc="$1" tip="$2"
  ping -c1 -W1 -I "$ifc" "$tip" >/dev/null 2>&1 && return 0
  ip neigh show dev "$ifc" 2>/dev/null | grep -q "^$tip .*lladdr"
}

_ensure_lane() {
  # $1 label  $2 teensy_ip  $3 host_cidr — make $2 reachable, whatever port it is on.
  # Reachability is pinned per-DEVICE, not per-subnet: a /32 host route on the
  # answering port, with an explicit src= (longest-prefix beats any /24 held by
  # another port). Both matter when the two Teensys share one /24 but sit on
  # DIFFERENT ports (e.g. base behind the expansion board, arm on the direct NIC):
  # the shared /24 route can only point one way, and without src= the kernel may
  # stamp another subnet's address as source (192.168.125.2 on the QSFP lane) —
  # the Teensy's /24 stack cannot route the reply and the ping dies silently.
  local label="$1" tip="$2" cidr="$3" hip="${3%/*}" ifc found=""
  ping -c1 -W1 "$tip" >/dev/null 2>&1 && return 0
  for ifc in $(_wired_ifaces); do
    if _iface_answers "$ifc" "$tip"; then found="$ifc"; break; fi
  done
  [ -n "$found" ] || return 1
  # MOVE the pin, never accumulate it: a /32 left behind on a previous port (the
  # Teensy re-plugged elsewhere) BLACKHOLES the IP — longest-prefix beats the /24
  # even while the stale port is linkdown (ignore_routes_with_linkdown=0; seen
  # live 2026-07-28: a persistent pin on enP2p1s0 hid the Teensy from every other
  # port). Strip the /32 from every other wired port (runtime) and from any NM
  # profile that stored it persistently.
  local other prof
  for other in /sys/class/net/*; do
    other="$(basename "$other")"
    case "$other" in lo|can*|wl*|usb*|docker*|veth*|l4tbr*|tailscale*|tun*) continue ;; esac
    [ -e "/sys/class/net/$other/device" ] || continue
    [ "$other" = "$found" ] && continue
    nmcli device modify "$other" -ipv4.routes "$tip/32 0.0.0.0 0 src=$hip" >/dev/null 2>&1 || true
  done
  while IFS= read -r prof; do
    [ -n "$prof" ] || continue
    nmcli -g ipv4.routes connection show "$prof" 2>/dev/null | grep -q "$tip/32" || continue
    nmcli connection modify "$prof" -ipv4.routes "$tip/32 0 src=$hip" >/dev/null 2>&1 || \
      nmcli connection modify "$prof" -ipv4.routes "$tip/32 0.0.0.0 0 src=$hip" >/dev/null 2>&1 || true
  done < <(nmcli -t -f NAME connection show 2>/dev/null)
  nmcli device modify "$found" -ipv4.routes "$tip/32 0.0.0.0 0 src=$hip" >/dev/null 2>&1 || true
  nmcli device modify "$found" \
    +ipv4.addresses "$cidr" \
    +ipv4.routes "$tip/32 0.0.0.0 0 src=$hip" >/dev/null 2>&1 || true
  sleep 1
  if ping -c1 -W1 "$tip" >/dev/null 2>&1; then
    ok "$label Teensy $tip found on port $found — host $cidr + /32 route assigned (runtime)"
    return 0
  fi
  return 1
}

_check_base_link() {
  [ "$BASE" = true ] || return 0
  if ping -c1 -W1 "$TEENSY_IP" >/dev/null 2>&1; then
    ok "base Teensy reachable at $TEENSY_IP"
    return 0
  fi
  # Port-agnostic self-heal: find the port the Teensy actually answers on.
  _ensure_lane "base" "$TEENSY_IP" "192.168.124.2/24" && return 0
  # No port answered ARP — try the persistent profile once (covers a link that came up
  # so late the carrier flag itself lagged), then report.
  if nmcli -t -f NAME connection show 2>/dev/null | grep -qx "$BASE_NM_CON"; then
    say "base Teensy not reachable yet — activating NM profile '$BASE_NM_CON'…"
    nmcli connection up "$BASE_NM_CON" >/dev/null 2>&1 || true
    sleep 1
    if ping -c1 -W1 "$TEENSY_IP" >/dev/null 2>&1; then
      ok "base Teensy reachable at $TEENSY_IP (via '$BASE_NM_CON')"
      return 0
    fi
  fi
  local up_ifaces
  up_ifaces="$(_wired_ifaces | tr '\n' ' ')"
  warn "BASE TEENSY UNREACHABLE at $TEENSY_IP — the base will not move until the link returns."
  if [ -z "$up_ifaces" ]; then
    warn "  → NO wired port has carrier. Plug the Teensy (or the expansion board) into any"
    warn "    Ethernet port ($BASE_IFACE, a mgbe lane, or the expansion board) and power it."
  else
    warn "  → wired port(s) up ($up_ifaces) but the Teensy answers ARP on none of them."
    warn "    Check Teensy power / its cable into the expansion board. Verify with:"
    warn "       ping -I <port> $TEENSY_IP ; ip neigh show   (want: $TEENSY_IP … lladdr …)"
  fi
  warn "  continuing — the bridge auto-reconnects when the link comes back (no relaunch needed)."
}

# The ARMS Teensy (openarm-dual firmware, 192.168.124.11) is port-agnostic exactly like
# the base: direct NIC, QSFP lane, or the expansion board — `_ensure_lane` finds it.
# The QSFP28 `teensy-arms` profile's 192.168.125.x subnet stays reserved for the
# in-repo ArmLink gateway firmware once it's flashed.
ARMS_NM_CON="${MABEL_ARMS_NM_CON:-Wired connection 1}"
ARMS_IFACE="${MABEL_ARMS_IFACE:-enP2p1s0}"      # home-port HINT for diagnostics only
ARMS_TEENSY_IP="${MABEL_ARMS_TEENSY_IP:-192.168.124.11}"
_check_arms_link() {
  [ "$ARMS" = true ] || return 0
  if ping -c1 -W1 "$ARMS_TEENSY_IP" >/dev/null 2>&1; then
    ok "arms Teensy reachable at $ARMS_TEENSY_IP"
    return 0
  fi
  _ensure_lane "arms" "$ARMS_TEENSY_IP" "192.168.124.2/24" && return 0
  if nmcli -t -f NAME connection show 2>/dev/null | grep -qx "$ARMS_NM_CON"; then
    say "arms Teensy not reachable yet — activating NM profile '$ARMS_NM_CON'…"
    nmcli connection up "$ARMS_NM_CON" >/dev/null 2>&1 || true
    sleep 1
    if ping -c1 -W1 "$ARMS_TEENSY_IP" >/dev/null 2>&1; then
      ok "arms Teensy reachable at $ARMS_TEENSY_IP (via '$ARMS_NM_CON')"
      return 0
    fi
  fi
  local up_ifaces
  up_ifaces="$(_wired_ifaces | tr '\n' ' ')"
  warn "ARMS TEENSY UNREACHABLE at $ARMS_TEENSY_IP — arms stay limp/idle until the link returns."
  if [ -z "$up_ifaces" ]; then
    warn "  → NO wired port has carrier. Plug/power the arms Teensy on any Ethernet port"
    warn "    ($ARMS_IFACE, a mgbe lane, or the expansion board)."
  else
    warn "  → wired port(s) up ($up_ifaces) but the arms Teensy answers ARP on none of them."
    warn "    Check Teensy power / firmware (env mabel_armlink) / its cable into the expansion board."
  fi
  warn "  continuing — the bridge auto-reconnects when the link comes back (no relaunch needed)."
}

_ensure_bridge() {
  # The bridge is the single hardware owner for the base AND the ORCA hands (Feetech
  # TTL serial). ROS never opens those devices — the ROS hand package is a bridge
  # CLIENT (the relay republishes the bridge's /<side>_hand/joint_states into the
  # URDF /joint_states so RViz mirrors the real hands). Start the bridge if either the
  # base or the hands are enabled.
  # AUDIO counts too: the bridge is the sole owner of the sound card, so an
  # audio-only bench (everything else off) still needs it — without this,
  # `--only-cameras`-style flag sets would silently leave the speaker dead.
  # ARMS/BODY/LIFT count since the HAL refactor: the bridge owns the arms link,
  # the neck buses (Dynamixel + yaw FDCAN dongle), the torso dongle and the lift
  # Pico — a body-only bench (--no-base --no-hands --no-cameras) previously
  # skipped the bridge entirely and every /arms|/body|/lift row sat "waiting".
  if [ "$BASE" != true ] && [ "$HANDS" != true ] && [ "$SENSORS" != true ] \
     && [ "$AUDIO" != true ] && [ "$ARMS" != true ] && [ "$BODY" != true ] \
     && [ "$LIFT_HW" != true ]; then
    say "bridge    every bridge-owned subsystem off — hardware_bridge not started"; return
  fi
  # TAKEOVER: a debugging real_run.sh is the SOLE hardware talker. Kill any
  # existing bridge (the systemd instance goes into lock-standby and resumes
  # ownership automatically when this run exits — see hardware_bridge.py's
  # _acquire_singleton) and establish OUR bridge as the one owner.
  local existing
  existing="$(pgrep -f '[h]ardware_bridge.py' 2>/dev/null || true)"
  if [ -n "$existing" ]; then
    warn "taking hardware ownership: stopping existing bridge (pid(s): $(echo $existing))."
    local p pg
    for p in $existing; do
      # INT first (rclpy's clean-shutdown path — the bridge ignores TERM)
      kill -INT "$p" 2>/dev/null || true
    done
    # wait for a clean exit (flock releases); escalate if it lingers
    local i
    for i in $(seq 1 20); do
      pgrep -f '[h]ardware_bridge.py' >/dev/null 2>&1 || break
      sleep 0.5
    done
    if pgrep -f '[h]ardware_bridge.py' >/dev/null 2>&1; then
      warn "existing bridge ignored SIGTERM — SIGKILLing (lock recovery handles the rest)."
      for p in $(pgrep -f '[h]ardware_bridge.py'); do
        pg="$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')"
        kill -KILL -- "-${pg:-$p}" 2>/dev/null || kill -KILL "$p" 2>/dev/null || true
      done
      sleep 1
      rm -f "${MABEL_HW_LOCK:-/tmp/mabel_hardware_bridge.lock}"   # ghost-flock guard
    fi
    ok "previous bridge stopped — this run owns the hardware."
  fi
  if [ ! -x "$HWBRIDGE_DIR/run_bridge.sh" ]; then
    warn "hardware_bridge missing at $HWBRIDGE_DIR — base/hands have NO driver; clients idle."
    return
  fi

  # Per-subsystem env. Base → Teensy; hands → the connected Feetech adapter(s),
  # honouring --hands left|right|both and the --left-port/--right-port flags. A side
  # not selected is forced 'none' (idle). auto-torque defaults ON for a real run but
  # can be overridden (MABEL_HW_HANDS_AUTO_TORQUE=false → bring up in free-drive).
  # MABEL_HW_STANDBY=0: OUR debug bridge must fail loudly if it cannot own the
  # hardware (standby is for the systemd instance only)
  # CAMERAS live in the bridge too (front D435i + head ZED + wrist D405/UVC + the
  # MJPEG/UDP video server on :8080/:9091/:9092) — the bridge is the ONLY process
  # that touches camera hardware, in this mode exactly like under systemd. The old
  # split (debug bridge with MABEL_HW_CAMERAS=false + the conda ros_camera_node
  # owning the cams) is gone: ros_camera_node has no D405 driver and double-owned
  # the wrist /dev nodes against the bridge.
  # LIFT is bridge-owned (mabel_hw lift subsystem -> Pico, /lift/command metres;
  # the gateway's /mabel_cmd.lift_pos is relayed by swerve_node). ON by default
  # since 2026-07-18 — the VP whole-body teleop's head-height follow was
  # publishing lift commands nobody consumed. Idle-safe with no Pico plugged.
  # NECK is bridge-owned now (serial Dynamixel neck, mabel_hw/drivers/dynamixel.py):
  # the HAL owns /dev/ttyUSB0 and serves /body/* (MABEL_HW_BODY = the DDS adapter,
  # MABEL_HW_NECK = the device). SAFE BY DESIGN: XC330 servos boot torque-OFF and
  # connect never energises them; only a /body/torque_enable true moves the neck.
  # Gate both on $BODY so --no-neck cedes the device too.
  # TORSO is bridge-owned too (DM-J10422P on the HDSC USB-CAN serial dongle,
  # mabel_hw/drivers/damiao_serial.py behind the drivers/solo.py seam adapter) and
  # the neck's yaw DaMiao (DM-USB2FDCAN, drivers/damiao_fdcan.py) rides the NECK
  # subsystem as a composite second bus. SAFE BY DESIGN: both dongle backends
  # connect WITHOUT energizing; only /body/torque_enable moves them. Gated on
  # $BODY like the neck so --no-body cedes both dongles to the setup studio.
  local body_on="$([ "$BODY" = true ] && echo true || echo false)"
  local bridge_env=(MABEL_HW_STANDBY=0
                    MABEL_HW_BODY="$body_on"
                    MABEL_HW_NECK="$body_on"
                    MABEL_HW_TORSO="$body_on"
                    MABEL_HW_LIFT="$([ "$LIFT_HW" = true ] && echo true || echo false)")
  # ARMS are bridge-owned (arms Teensy, mabel_hw/drivers/openarm_link.py) and follow the
  # --arms flag. SAFE BY DESIGN: the subsystem boots torque-OFF (limp) and only a
  # /arms/torque_enable true energises it — bringing the subsystem up sends NO
  # motion commands, it only listens for telemetry.
  if [ "$ARMS" = true ]; then
    bridge_env+=(MABEL_HW_ARMS=true MABEL_HW_ARMS_TEENSY_IP="$ARMS_TEENSY_IP")
  else
    bridge_env+=(MABEL_HW_ARMS=false)
  fi
  if [ "$SENSORS" = true ]; then
    bridge_env+=(MABEL_HW_CAMERAS=true)
  else
    bridge_env+=(MABEL_HW_CAMERAS=false)
  fi
  # RPLIDAR A2M12: the bridge owns the CP2102 serial and publishes /scan directly
  # (the ROS-side sllidar node is gone — hw_args passes lidar:=false below).
  bridge_env+=(MABEL_HW_LIDAR="$([ "$LIDAR" = true ] && echo true || echo false)")
  bridge_env+=(MABEL_HW_AUDIO="$([ "$AUDIO" = true ] && echo true || echo false)")
  [ -n "$AUDIO_CARD" ] && bridge_env+=(MABEL_HW_AUDIO_CARD="$AUDIO_CARD")
  if [ "$BASE" = true ]; then
    bridge_env+=(MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP="$TEENSY_IP")
  else
    bridge_env+=(MABEL_HW_BASE=false)
  fi
  # Honest label: list every bridge-owned subsystem this run enables (the old
  # binary "base → …"/"cameras only" lied for body/arms/lift-only benches).
  local what=""
  [ "$BASE" = true ]    && what+="base→$TEENSY_IP "
  [ "$ARMS" = true ]    && what+="arms "
  [ "$BODY" = true ]    && what+="neck+torso "
  [ "$LIFT_HW" = true ] && what+="lift "
  [ "$HANDS" = true ]   && what+="hands "
  [ "$SENSORS" = true ] && what+="cameras "
  [ "$AUDIO" = true ]   && what+="audio "
  what="${what% }"
  if [ "$HANDS" = true ]; then
    local lport="$LEFT_PORT" rport="$RIGHT_PORT"
    # Only a USER-forced side selection (--hands / MABEL_HAND_SIDES) turns the other
    # side off. Auto-resolved sides keep both ports 'auto': the bridge's serial->side
    # registry makes that safe, and the absent side hot-plugs when its adapter appears.
    if [ "$HAND_SIDES_FORCED" = 1 ]; then
      case "$HAND_SIDES" in
        left)  rport="none" ;;
        right) lport="none" ;;
      esac
    fi
    # FREE-DRIVE by default: bring the hands up limp so launching real_run.sh never
    # energises/commands them (a powered Feetech servo snaps to its stale goal register
    # = an unasked-for motion). Engage torque later from the control center / teleop, or
    # set MABEL_HW_HANDS_AUTO_TORQUE=true to arm at startup.
    bridge_env+=(MABEL_HW_HANDS=true
                 MABEL_HW_HANDS_LEFT_PORT="${lport:-}"
                 MABEL_HW_HANDS_RIGHT_PORT="${rport:-none}"
                 MABEL_HW_HANDS_AUTO_TORQUE="${MABEL_HW_HANDS_AUTO_TORQUE:-false}")
    what="${what} ; hands($HAND_SIDES) → Feetech TTL"
    [ "$BASE" = true ] || what="hands($HAND_SIDES) → Feetech TTL"
  else
    bridge_env+=(MABEL_HW_HANDS=false)
  fi
  if [ "$AUDIO" = true ]; then
    what="${what} ; audio → ${AUDIO_CARD:-USB $AUDIO_USB_ID}"
  fi

  say "bridge    hardware_bridge: $what   (log: $LOG_HWBRIDGE)"
  : > "$LOG_HWBRIDGE"
  env "${bridge_env[@]}" "$HWBRIDGE_DIR/run_bridge.sh" >>"$LOG_HWBRIDGE" 2>&1 &
  PIDS+=($!); BRIDGE_OURS=1
  sleep 2     # let it advertise /mabel/hw/base/* + /<side>_hand/* before clients subscribe
}

# Post-condition: the bridge must be the SOLE commander of the Teensy. hardware_bridge.py's
# flock already makes a SECOND bridge exit(3), but only if both use the same lock path, and
# nothing stops a non-bridge talker (ps5_teleop, a live_*_diag probe, an old out-of-repo
# bridge) from sending to the Teensy in parallel. Assert exactly one bridge holds the lock,
# kill any duplicate bridge (keep the lock holder), and reap any stray non-bridge commander.
_verify_single_base_owner() {
  [ "$BASE" = true ] || return 0
  local lock="${MABEL_HW_LOCK:-/tmp/mabel_hardware_bridge.lock}"
  local owner pids n p
  owner="$(cat "$lock" 2>/dev/null | tr -dc '0-9')"
  pids="$(pgrep -f "[h]ardware_bridge.py" 2>/dev/null)"
  n="$(printf '%s\n' "$pids" | grep -c .)"
  if [ "${n:-0}" -gt 1 ]; then
    warn "MULTIPLE ($n) hardware_bridge processes own the base — they FIGHT for the Teensy."
    warn "  keeping the lock holder (pid ${owner:-?}); killing the rest."
    for p in $pids; do [ "$p" = "$owner" ] || kill -9 "$p" 2>/dev/null || true; done
  fi
  # Stray NON-bridge commanders that bypass the flock (reaped at startup too; re-check here
  # in case one was launched between _reap_stale and now).
  local stray
  stray="$(pgrep -af "ps5_teleop\.py|tools/live_.*_diag|tools/live_optimize_fix|maxswerve.*ps5|mabel_swerve_bridge\.py" 2>/dev/null)"
  if [ -n "$stray" ]; then
    warn "stray base commander(s) bypassing the bridge — killing so the bridge is SOLE owner:"
    printf '%s\n' "$stray" | sed 's/^/      /'
    pkill -9 -f "ps5_teleop\.py" 2>/dev/null || true
    pkill -9 -f "tools/live_.*_diag" 2>/dev/null || true
    pkill -9 -f "tools/live_optimize_fix" 2>/dev/null || true
    pkill -9 -f "mabel_swerve_bridge\.py" 2>/dev/null || true   # retired out-of-repo direct-Teensy bridge
  fi
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    ok "single base owner confirmed — hardware_bridge pid $owner holds the Teensy lock."
  elif [ "${n:-0}" -ge 1 ]; then
    ok "base owner present (hardware_bridge running; lock pid ${owner:-unknown})."
  else
    warn "no hardware_bridge owns the base — the Teensy has no driver (clients will idle)."
  fi
}

# Catch-all for hardware squatters the named patterns above miss: ANY process (an old
# bench script, a stray python REPL, a half-dead node) that holds a hand serial
# adapter or a socket to the base Teensy blocks the ONE bridge from owning that
# hardware. Reap by RESOURCE, not by name — the bridge itself and this shell are
# never touched. Best effort: fuser/ss see same-user processes, which is all of ours.
_kill_hw_squatters() {
  local bridgepids dev real pid
  bridgepids=" $(pgrep -f '[h]ardware_bridge.py' 2>/dev/null | tr '\n' ' ') "
  _is_ours() { [ "$1" = "$$" ] || [ "$1" = "$PPID" ] || case "$bridgepids" in *" $1 "*) true ;; *) false ;; esac; }
  # hand serial adapters (Feetech TTL) — the bridge must be the only holder
  if [ "$HANDS" = true ] && command -v fuser >/dev/null 2>&1; then
    for dev in "$LEFT_PORT" "$RIGHT_PORT" /dev/serial/by-id/usb-1a86_USB_Single_Serial_*; do
      case "$dev" in none|auto|"") continue ;; esac
      [ -e "$dev" ] || continue
      real="$(readlink -f "$dev")"
      for pid in $(fuser "$real" 2>/dev/null); do
        _is_ours "$pid" && continue
        warn "pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null)) squats on $real — killing it so the bridge is sole owner."
        kill -9 "$pid" 2>/dev/null || true
      done
    done
  fi
  # base Teensy ethernet link — no non-bridge process may talk to it
  if [ "$BASE" = true ]; then
    for pid in $(ss -tunap 2>/dev/null | grep -F "$TEENSY_IP" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
      _is_ours "$pid" && continue
      warn "pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null)) holds a socket to the base Teensy $TEENSY_IP — killing it."
      kill -9 "$pid" 2>/dev/null || true
    done
  fi
  # arms Teensy ethernet link — same rule (two owners would fight over MIT targets)
  if [ "$ARMS" = true ]; then
    for pid in $(ss -tunap 2>/dev/null | grep -F "$ARMS_TEENSY_IP" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
      _is_ours "$pid" && continue
      warn "pid $pid ($(ps -o comm= -p "$pid" 2>/dev/null)) holds a socket to the arms Teensy $ARMS_TEENSY_IP — killing it."
      kill -9 "$pid" 2>/dev/null || true
    done
  fi
}

# Wait until a topic appears (sources the conda env ONCE, then polls).
_wait_topic() {   # topic timeout_s
  local topic="$1" timeout="${2:-45}"
  ( source "$ENV_SH" >/dev/null 2>&1 || exit 1
    local i=0
    while [ "$i" -lt "$timeout" ]; do
      ros2 topic list 2>/dev/null | grep -qx "$topic" && exit 0
      sleep 1; i=$((i + 1))
    done
    exit 1 )
}

# Like _wait_topic but requires an actual MESSAGE, not just an advertised topic:
# proves a publisher is really emitting data — e.g. the swerve base is connected
# and /odom is LIVE, vs. /odom merely advertised by a node with no hardware behind
# it (a camera-only bench, where /odom is silent and odom->base_link never appears).
_topic_has_data() {   # topic timeout_s
  local topic="$1" timeout="${2:-6}"
  ( source "$ENV_SH" >/dev/null 2>&1 || exit 1
    timeout "$timeout" ros2 topic echo "$topic" --once >/dev/null 2>&1 )
}

# ── hardware status watchdog ─────────────────────────────────────────────────
# Periodically (every $WATCH_INTERVAL s) print EVERY subsystem's liveness so a
# camera you hot-SWAP (head ZED <-> base D435 on the one good USB-3 cable) is SEEN
# the moment it re-binds — no relaunch. It also SELF-HEALS the classic Tegra trap:
# the D435i enumerates fine on USB-3 but its uvcvideo interfaces come up UNBOUND
# (the librealsense RSUSB 'unbind' udev rule, or a hot-plug race) so the V4L2
# pyrealsense2 sees 0 devices and the front cam never starts. When we spot the
# D435 on USB with NO RealSense /dev/video node, we rebind uvcvideo (best effort,
# needs root) so camera_hal's 2 s reconnect loop picks it straight up.
_realsense_present() { lsusb 2>/dev/null | grep -qiE '8086:0b3a'; }
_zed_present()       { lsusb 2>/dev/null | grep -qiE '2b03:f68'; }
_realsense_bound() {   # true if a RealSense V4L2 node exists (uvcvideo is bound)
  local v
  for v in /sys/class/video4linux/video*/name; do
    [ -r "$v" ] && grep -qi 'realsense' "$v" && return 0
  done
  return 1
}
_rebind_realsense() {  # bind uvcvideo to the D435 UVC interfaces (needs root)
  local dev intf
  for dev in /sys/bus/usb/devices/*; do
    [ -r "$dev/idProduct" ] || continue
    [ "$(cat "$dev/idVendor"  2>/dev/null)" = 8086 ] || continue
    [ "$(cat "$dev/idProduct" 2>/dev/null)" = 0b3a ] || continue
    for intf in "$dev":*; do
      [ -r "$intf/bInterfaceClass" ] || continue
      [ "$(cat "$intf/bInterfaceClass" 2>/dev/null)" = 0e ] || continue   # 0e = UVC video
      [ -e "$intf/driver" ] && continue                                   # already bound
      sudo -n sh -c "echo -n '$(basename "$intf")' > /sys/bus/usb/drivers/uvcvideo/bind" 2>/dev/null || return 1
    done
  done
  return 0
}
_hw_watchdog() {
  source "$ENV_SH" >/dev/null 2>&1 || true
  local warned_rs=0 cam rs_usb rs_node zed_usb topics
  while true; do
    sleep "$WATCH_INTERVAL"
    # Teensy lane self-heal is CONTINUOUS (2026-07-28): the boards are port-agnostic
    # and get (re)plugged/powered mid-run, so the startup-only _check_*_link left a
    # board unreachable until the next relaunch. Quietly re-discover any missing
    # Teensy each tick; _ensure_lane is cheap when the board is absent (one bounded
    # ARP ping per carrier-up port) and logs ok only when it actually lands a lane.
    if [ "$BASE" = true ] && ! ping -c1 -W1 "$TEENSY_IP" >/dev/null 2>&1; then
      _ensure_lane "base" "$TEENSY_IP" "192.168.124.2/24" || true
    fi
    if [ "$ARMS" = true ] && ! ping -c1 -W1 "$ARMS_TEENSY_IP" >/dev/null 2>&1; then
      _ensure_lane "arms" "$ARMS_TEENSY_IP" "192.168.124.2/24" || true
    fi
    # cameras — cheap: reuse the camera node's own 5 s status line in the HW log
    cam="$(grep -aoE '\[camera_node\] [^|]*' "$LOG_HW" 2>/dev/null | tail -1 | sed 's/\[camera_node\] //')"
    # RealSense uvcvideo bind health + self-heal
    rs_usb=off; rs_node='—'
    if _realsense_present; then
      rs_usb=on
      if _realsense_bound; then rs_node=bound; warned_rs=0
      elif _rebind_realsense && _realsense_bound; then rs_node=rebound; warned_rs=0
      else
        rs_node=UNBOUND
        if [ "$warned_rs" = 0 ]; then
          warn "D435 on USB-3 but uvcvideo UNBOUND (no /dev/video → pyrealsense sees 0). The V4L2 pyrealsense2 needs uvcvideo bound. Permanently fix it: sudo mv /etc/udev/rules.d/99-realsense-rsusb-unbind.rules{,.disabled} && sudo udevadm control --reload && sudo udevadm trigger"
          warned_rs=1
        fi
      fi
    fi
    zed_usb=off; _zed_present && zed_usb=on
    # subsystems — ONE topic-list snapshot, then membership tests
    topics="$(ros2 topic list 2>/dev/null)"
    _has() { printf '%s\n' "$topics" | grep -qx "$1" && printf '✓' || printf '·'; }
    # CONNECTED != ADVERTISED. Every HAL node calls create_publisher() in __init__, so
    # /arms/joint_states, /body/joint_states, /lift/joint_states and the wrist /camera/*
    # topics all EXIST the instant the node starts — even with NO hardware plugged in
    # (each node's _stream does `if not names: return`, so a MESSAGE only flows when a
    # real device streams). So `_has` (topic present) reads ✓ on an empty bench and LIES
    # about arms/neck/torso/lift/wrist being "connected". For every subsystem whose
    # connection claim must be trustworthy — arms, neck, torso, lift, hands, wrist cams,
    # audio — require a real MESSAGE (✓ = live data from the physical device, · =
    # idle/absent). Only odom/scan/map, which have no idle-advertise trap here, stay `_has`.
    # `-k 1`, NOT a bare `timeout 2`. This status line wedged the entire bring-up on
    # thor for 30+ minutes (2026-07-29): plain `timeout` sends SIGTERM, `ros2 topic
    # echo` blocks inside rclpy waiting for a message that a disconnected camera will
    # never publish and does not service the signal, and `timeout` then waits FOREVER
    # for a child it cannot kill. The loop never returned, so the teleop ports (9090
    # command, 9610 voice, 8080 video) never opened — which reads to the operator as
    # "the app connects to nothing" and "the audio stopped working", with the robot
    # pingable the whole time. Twelve of these zombies had accumulated. `-k 1`
    # escalates to SIGKILL one second after the SIGTERM, so the bound is real.
    _live() { timeout -k 1 2 ros2 topic echo "$1" --once >/dev/null 2>&1 && printf '✓' || printf '·'; }
    # The body is THREE independent motor buses sharing ONE /body/joint_states topic;
    # the lift is a FOURTH connection on its OWN topic (/lift/joint_states, Pico serial):
    #   neck dynamixel = serial Dynamixel (mabel_hw/drivers/dynamixel.py) — pitch/roll
    #                    (names neck_pitch/neck_roll, or motor_* in the FK-down fallback)
    #   neck damiao    = DM-J4310 on the DM-USB2FDCAN dongle (drivers/damiao_fdcan.py
    #                    behind drivers/solo.py) — yaw (name neck_yaw)
    #   torso          = DM-J10422P on the arms Teensy's CAN bus 3 (drivers/openarm.py
    #                    group over the shared openarm_link; TORSO_BACKEND=serial falls
    #                    back to the HDSC dongle + damiao_serial.py) — name 'torso'
    # Each HAL node names a joint ONLY while its motor streams live feedback, so grep the
    # live message to tell each bus apart — one can be up while another is unplugged.
    # (One echo, not three.)  lift is probed separately via _live /lift/joint_states.
    neck_dxl='·'; neck_dm='·'; torso='·'
    _body="$(timeout -k 1 2 ros2 topic echo /body/joint_states --once 2>/dev/null)"   # -k: see _live
    printf '%s' "$_body" | grep -qE 'neck_pitch|neck_roll|motor_' && neck_dxl='✓'
    printf '%s' "$_body" | grep -qE 'neck_yaw'                     && neck_dm='✓'
    printf '%s' "$_body" | grep -qE '[[:space:]-]torso([[:space:]]|$)' && torso='✓'
    aud_usb=off; [ -n "$(_audio_card)" ] && aud_usb=on
    printf '  \033[2m[hw %s]\033[0m %s | RS-usb:%s(%s) ZED-usb:%s aud-usb:%s | odom:%s arms:%s neck-dxl:%s neck-damiao:%s torso:%s lift:%s handL:%s handR:%s wristL:%s wristR:%s scan:%s audio:%s | SLAM map:%s mesh:%s\n' \
      "$(date +%H:%M:%S)" "${cam:-cameras:—}" "$rs_usb" "$rs_node" "$zed_usb" "$aud_usb" \
      "$(_has /odom)" "$(_live /arms/joint_states)" "$neck_dxl" "$neck_dm" "$torso" "$(_live /lift/joint_states)" \
      "$(_live /left_hand/joint_states)" "$(_live /right_hand/joint_states)" \
      "$(_live /camera/wrist_left/image_raw)" "$(_live /camera/wrist_right/image_raw)" \
      "$(_has /scan)" "$(_live /audio/device_health)" \
      "$(_has /map)" "$(_has /nvblox_node/mesh_pointcloud)"
  done
}

# ── lifecycle ──────────────────────────────────────────────────────────────────
_cleaned=0
cleanup() {
  [ "$_cleaned" = 1 ] && return; _cleaned=1
  echo; say "shutting down…"
  for p in "${PIDS[@]:-}"; do kill -INT "$p" >/dev/null 2>&1 || true; done
  sleep 1.5
  for p in "${PIDS[@]:-}"; do kill -TERM "$p" >/dev/null 2>&1 || true; done
  sleep 0.5
  # backstop: reap anything that ignored the signals (narrowly matched). Include
  # the FULL Nav2 node set so no lifecycle_manager/cmd_vel_bridge/etc. is orphaned
  # (an orphaned node keeps a CycloneDDS participant -> domain-0 index exhaustion).
  for pat in "ros2 launch mabel_bringup" "ros2 launch mabel_navigation" \
             "ros2 launch mabel_slam" component_container_mt nvblox_node \
             visual_slam mesh_to_pointcloud slice_to_occupancy async_slam_toolbox_node \
             "mabel_slam/laser_odometry" "lib/mabel_slam/map_relay" \
             "lib/mabel_base_hardware" "lib/mabel_arms_hardware" \
             "lib/mabel_body_hardware" "lib/mabel_hand_hardware" \
             swerve_odometry sensors/ros_camera_node.py \
             controller_server planner_server bt_navigator behavior_server \
             smoother_server velocity_smoother collision_monitor \
             lifecycle_manager_navigation cmd_vel_bridge goal_relay base_link_nav \
             "rviz2.*hardware" robot_state_publisher joint_state_publisher bringup_status \
             sim_teleop_bridge; do
    pkill -9 -f "$pat" >/dev/null 2>&1 || true
  done
  # Reap the hardware bridge ONLY if THIS run started it (never a systemd-owned one).
  [ "${BRIDGE_OURS:-0}" = 1 ] && pkill -f "hardware_bridge/hardware_bridge.py" >/dev/null 2>&1 || true
  # Same for the Control Center studio (the leading [ /] keeps the pattern off
  # simulation_control_center.server — that one is '_'-joined, never space/slash).
  [ "${STUDIO_OURS:-0}" = 1 ] && pkill -9 -f "[ /]control_center\.server" >/dev/null 2>&1 || true
  # Release the manual-run lock so the standing-by service instance resumes the
  # headless auto-start bring-up on its own.
  if [ "${MANUAL_LOCK_OURS:-0}" = 1 ]; then
    rm -f "$MANUAL_LOCK"
    say "manual-run lock released — $BRINGUP_SVC resumes the headless bring-up."
  fi
  ok "done."
  exit 0
}

# Reap a previous run so this one owns the topic graph. hardware.launch.py's own
# reaper clears the HW + camera + RViz + RSP nodes, but ORPHANED Nav2/SLAM nodes
# (whose launcher already died) survive by name — and they each hold a CycloneDDS
# participant, so across runs they pile up until domain 0 runs out of participant
# indices ("Failed to find a free participant index") and NEW nodes (e.g. the
# camera) can't start. So reap the FULL Nav2 + SLAM node set here by executable,
# plus a stray real_ros_server / sim bridge (the alternative stack — never run both
# against the same hardware). None of these patterns match THIS launcher's cmdline
# ("bash …/real_run.sh"), and the real_run.sh loop skips $$ / $PPID.
_reap_stale() {
  say "clearing any previous run (incl. orphaned Nav2/SLAM nodes) …"
  for pat in "ros2 launch mabel_" "ros2 launch nav" \
             component_container_mt nvblox_node visual_slam mesh_to_pointcloud \
             slice_to_occupancy async_slam_toolbox_node \
             nav2_controller nav2_planner nav2_bt_navigator nav2_behaviors \
             nav2_smoother nav2_velocity_smoother nav2_collision_monitor \
             nav2_lifecycle_manager controller_server planner_server bt_navigator \
             behavior_server smoother_server velocity_smoother collision_monitor \
             lifecycle_manager_navigation cmd_vel_bridge goal_relay frontier_explorer \
             "mabel_slam/laser_odometry" "lib/mabel_slam/map_relay" \
             nav_pose_setter base_link_nav "rviz2.*hardware" \
             sim_teleop_bridge real_ros_server.sh \
             "ps5_teleop\.py" "tools/live_.*_diag" "tools/live_optimize_fix" \
             "maxswerve.*ps5" "mabel_swerve_bridge\.py" \
             "studio/launch\.sh" "-m studio\.server" "-m calib\.server" \
             "-m arm_bridge\.bridge" "-m hand_bridge\.bridge" "-m neck_bridge\.bridge"; do
    # ^ ps5/live_diag/maxswerve are STRAY BASE COMMANDERS (send straight to the base
    #   Teensy, bypassing the bridge's flock). The modules/* patterns are the SETUP
    #   STUDIO stack (2026-07-28): its arm_bridge/hand_bridge/neck_bridge are UDP/serial
    #   clients of the SAME devices as the HAL — the arms Teensy firmware is
    #   last-Hello-wins, so studio + bridge running together makes the arms flap
    #   limp/jittery (seen live during VP teleop). Studio and robot runs are mutually
    #   exclusive; a fresh robot run reaps the studio, exactly as the studio's own
    #   run.sh stops the HAL. None match hardware_bridge.py / real_run.sh.
    pkill -9 -f "$pat" >/dev/null 2>&1 || true
  done
  local pid svcpid
  # NEVER kill the systemd service instance here: it is either us ($$) or already
  # standing by on the manual-run lock (killing it only makes systemd respawn it).
  svcpid="$(_svc_mainpid)"
  for pid in $(pgrep -f "real_run.sh" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue; [ "$pid" = "$PPID" ] && continue
    [ -n "$svcpid" ] && [ "$pid" = "$svcpid" ] && continue
    kill -9 "$pid" >/dev/null 2>&1 || true
  done
  sleep 2
}

# The IP the iOS app / browser should connect to. hostname -I lists ALL addresses and
# the base-Teensy ethernet (192.168.124.x), the Jetson USB gadget (192.168.55.x), and
# docker (172.17.x) are NOT reachable from a phone -- skip them and return the real
# Wi-Fi/LAN address (e.g. 10.20.x). Falls back to 127.0.0.1 if nothing else.
_lan_ip() {
  hostname -I 2>/dev/null | tr ' ' '\n' \
    | grep -E '^[0-9]' \
    | grep -vE '^(127\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[01]\.|192\.168\.124\.|192\.168\.55\.|100\.)' \
    | head -1 || echo 127.0.0.1
}
# Tailscale IP (100.x) for remote control off the LAN, if present.
_tailscale_ip() { hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^100\.' | head -1; }

_banner() {
  local on="\033[32mon\033[0m" off="\033[2moff\033[0m"
  local b="$([ "$BASE" = true ] && echo -e "$on" || echo -e "$off")"
  local a="$([ "$ARMS" = true ] && echo -e "$on" || echo -e "$off")"
  # Body is THREE buses: neck-dxl = Dynamixel serial (pitch/roll), neck-damiao = DaMiao
  # yaw, torso = DaMiao CAN. All gate on $BODY (--no-neck cedes them). Shown separately
  # so the three connections are visible (each can be up/down independently).
  local y="$([ "$BODY" = true ] && echo -e "$on" || echo -e "$off")"
  local lf="$([ "$LIFT_HW" = true ] && echo -e "$on" || echo -e "$off")"
  local h="$([ "$HANDS" = true ] && echo -e "$on ($HAND_SIDES)" || echo -e "$off")"
  local c="$([ "$SENSORS" = true ] && echo -e "$on" || echo -e "$off")"
  local l="$([ "$LIDAR" = true ] && echo -e "$on" || echo -e "$off")"
  # audio's pill reports the DEVICE, not just the flag: "on" with no bar plugged in
  # would read as a working speaker (the mic-mute lesson — say what is true).
  local au="$off"
  if [ "$AUDIO" = true ]; then
    local _ac; _ac="$(_audio_card)"
    au="$([ -n "$_ac" ] && echo -e "$on ($_ac)" || echo -e "$on (no device)")"
  fi
  local s="$([ "$DO_SLAM" = 1 ] && echo -e "$on" || echo -e "$off")"
  local n="$([ "$DO_NAV" = 1 ] && echo -e "$on" || echo -e "$off")"
  local v="$([ "$DO_RVIZ" = 1 ] && echo -e "$on" || echo -e "$off")"
  local wv="$([ "$DO_WEB" = 1 ] && echo -e "$on" || echo -e "$off")"
  local st="$([ "$DO_STUDIO" = 1 ] && echo -e "$on" || echo -e "$off")"
  local ip; ip="$(_lan_ip)"
  cat <<EOF

  ──────────────────────────────────────────────────────────────────────────
    M A B E L   ·   R E A L   R O B O T   B R I N G - U P   ($(hostname -s))
  ──────────────────────────────────────────────────────────────────────────
EOF
  local odomsrc="$([ "$SLAM_ODOM" = true ] && echo 'cuVSLAM owns odom (self-contained)' || echo 'wheel odom owns; cuVSLAM corrects map->odom')"
  local slammode
  if [ "${use_lidar_slam:-0}" = 1 ]; then
    slammode='2D slam_toolbox (LiDAR /map+map->odom) + nvblox 3D mesh'
  else
    slammode='visual cuVSLAM + nvblox (3D mesh + 2D /map + map->odom)'
  fi
  echo -e "    base $b   arms $a   neck-dxl $y   neck-damiao $y   torso $y   lift $lf   hands $h   cameras $c   lidar $l"
  echo -e "    audio $au"
  echo -e "    \033[2m(pills = ENABLED/requested, not proof of a live device. The periodic"
  echo -e "     [hw HH:MM:SS …] line below shows what is actually CONNECTED — streaming"
  echo -e "     real data — per bus: arms · neck-dxl · neck-damiao · torso · lift · wristL/R.)\033[0m"
  echo -e "    SLAM $s   Nav2 $n   RViz $v   web $wv   studio $st"
  if [ "$DO_SLAM" = 1 ]; then
    echo -e "    SLAM mode:   $slammode"
    echo -e "    odom source: $odomsrc"
  fi
  if [ "$DO_WEB" = 1 ]; then
    local tsip; tsip="$(_tailscale_ip)"
    echo -e "    iOS app / browser — control $(hostname -s):"
    echo -e "      • same Wi-Fi:   add robot host  \033[1m${ip}\033[0m   (teleop ws://${ip}:${WS_PORT}/teleop)"
    [ -n "$tsip" ] && echo -e "      • Tailscale:    add robot host  \033[1m${tsip}\033[0m   (off-LAN, same tailnet)"
    echo -e "      video  http://${ip}:${HTTP_PORT}/camera/main/stream.mjpg   ·   Nav/map  ws://${ip}:${WS_PORT}/map"
    echo -e "      (the app auto-discovers $(hostname -s) on the same Wi-Fi; otherwise add the host above)"
  fi
  if [ "$DO_STUDIO" = 1 ]; then
    echo -e "    control studio (browser GUI):  http://${ip}:${STUDIO_PORT}"
  fi
  cat <<EOF

    check each module's ROS connection (run in another shell, after sourcing
    the env:  source $WS/scripts/mabel_env.sh):
      all topics      ros2 topic list
      all nodes       ros2 node list
      base/odom       ros2 topic echo /odom            ·  /base/joint_states
      drive a module  ros2 topic pub /mabel_cmd  …      ·  /arms/command  /body/command
      arms            ros2 topic echo /arms/joint_states
      neck/body       ros2 topic echo /body/joint_states
      hands           ros2 topic echo /left_hand/joint_states  /right_hand/joint_states
      cameras         ros2 topic hz /camera/front/color/image_raw   (+ /camera/zed/left/image_raw)
      front IR/SLAM   ros2 topic hz /camera/front/infra1/image_rect_raw
      SLAM mesh/map   ros2 topic hz /nvblox_node/mesh_pointcloud   ·  /map   (tf: map→odom)
      teleop by hand  ros2 run mabel_teleop keyboard_teleop

    RViz shows the robot model, TF, every camera cloud/image, the LaserScan,
    and (SLAM on) the nvblox 3D mesh + 2D map.
    logs: $LOG_HW  $LOG_NAV  $LOG_SLAM  $LOG_SLAM_2D
    Ctrl-C here stops everything.
  ──────────────────────────────────────────────────────────────────────────

EOF
}

# ── preflight ──────────────────────────────────────────────────────────────────
[ -f "$ENV_SH" ]                  || die "missing $ENV_SH — set MABEL_WS / build the workspace."
[ -f "$WS/install/setup.bash" ]   || die "workspace not built — run $WS/dev_up.sh first."
if [ "$DO_SLAM" = 1 ] && [ ! -f "$WS/scripts/slam_env.sh" ]; then
  warn "SLAM requested but $WS/scripts/slam_env.sh is missing — continuing with --no-slam."
  DO_SLAM=0
fi
# The librealsense RSUSB 'unbind' udev rule rips uvcvideo off the D435i on every
# plug, but the conda V4L2 pyrealsense2 needs uvcvideo BOUND — with the rule active
# the front cam never starts (and a hot-swap never re-detects). Warn early; the
# watchdog self-heals each plug, but disabling the rule is the permanent fix.
if [ "$SENSORS" = true ] && [ -f /etc/udev/rules.d/99-realsense-rsusb-unbind.rules ]; then
  warn "active rule /etc/udev/rules.d/99-realsense-rsusb-unbind.rules UNBINDS uvcvideo from the D435i — the V4L2 pyrealsense2 needs it bound. Disable it: sudo mv /etc/udev/rules.d/99-realsense-rsusb-unbind.rules{,.disabled} && sudo udevadm control --reload && sudo udevadm trigger"
fi

# ── run ────────────────────────────────────────────────────────────────────────
# EXIT runs cleanup for ANY exit; INT/TERM just `exit`, which fires the EXIT trap
# (so a single Ctrl-C tears the whole robot down and ACTUALLY exits — without the
# explicit exit the `while` hold below would just loop after cleanup ran).
trap cleanup EXIT
trap 'exit 130' INT TERM
say "root   $MABEL_ROOT"
say "ws     $WS"
_reap_stale

# 0) Hardware bridge (the single HAL) — must be up so the base_bridge_client in the
#    bring-up has a driver behind /mabel/hw/base/*. Idempotent (skips if already up).
#    Preflight the base link first so a NO-CARRIER cable is flagged loudly + actionably,
#    and report which hand adapter(s) are on USB (resolves --hands auto).
_detect_hands
_detect_audio
_check_base_link
_check_arms_link
_ensure_bridge
_verify_single_base_owner
_kill_hw_squatters

# 0.5) LiDAR-vs-VISUAL nav architecture — decided BEFORE the hardware launch so odom
#    ownership is set correctly from the start. The bridge (step 0) owns the RPLIDAR
#    and brings /scan up within ~10 s; if it appears, the LiDAR architecture runs:
#      /scan → laser_odometry (scan-matching)   owns odom->base_link — the robot
#              odom is DERIVED FROM THE LIDAR (+ /odom_laser); the wheel /odom is
#              twist feedback only (cmd_vel_bridge /odom_nav) and never owns TF.
#      /scan + odom → slam_toolbox              /map_lidar (→ /map) + map->odom;
#              Nav2 localises + plans on the LiDAR grid.
#      front cam (if present) → nvblox MESH-ONLY: the 3D mesh keeps streaming to the
#              apps; cuVSLAM stays off (slam_toolbox owns map->odom).
#    No /scan → the VISUAL architecture: cuVSLAM + nvblox → /map_visual (→ /map) +
#    map->odom + 3D mesh (odom owner per SLAM_ODOM auto). Either way the canonical
#    /map grid + the mesh stream to the apps on :9090/:9092 UNCHANGED (map_stream
#    grid plane rides /map — provenance lives on /map_lidar vs /map_visual).
LIDAR_MODE=0
if [ "$DO_SLAM" = 1 ] && [ "$LIDAR" = true ] && [ "$LIDAR_SLAM" != off ]; then
  scan_to=$([ "$LIDAR_SLAM" = on ] && echo 30 || echo 15)
  say "lidar     probing /scan (${scan_to}s) — LiDAR nav architecture iff it appears …"
  if _wait_topic /scan "$scan_to"; then
    LIDAR_MODE=1
    SLAM_ODOM=true   # base must NOT publish odom->base_link: laser_odometry owns it
    ok "LiDAR /scan live → slam_toolbox 2D map + LiDAR-derived odom own the nav frames."
  else
    say "          no /scan — VISUAL architecture (cuVSLAM + nvblox)."
  fi
fi

# 1) Hardware bring-up — base/arms/body/hands/cameras (camera_hal also serves the
#    MJPEG video on :8080) + robot_state_publisher + joint_state_publisher + status
#    dashboard + the RPLIDAR /scan driver + RViz. SLAM is NOT launched here
#    (slam:=false): real_run owns SLAM below so it can pick visual-vs-LiDAR after it
#    knows whether a /scan is live. One launch does the whole robot/HAL side.
local_rviz="false"; [ "$DO_RVIZ" = 1 ] && local_rviz="true"
say "hardware  bring-up (base/arms/neck/hands/cameras+MJPEG/lidar) + RViz   (log: $LOG_HW)"
# Build the launch args. ros2 launch REJECTS an empty value ('left_port:='), so
# only pass left_port/right_port when set — otherwise let hardware.launch.py use
# its own defaults (left_port '' = auto-detect, right_port 'none' = off).
# slam_odom is still passed even with slam:=false: it sets the base's odom->base_link
# TF ownership (false = wheel swerve_odometry publishes it, needed by slam_toolbox).
# Camera ownership: the hardware_bridge (ours or systemd's) is the SINGLE camera
# owner — front D435i + head ZED + wrist D405/UVC — publishing /camera/* over DDS
# AND serving the MJPEG/UDP video (:8080/:9091/:9092). The old conda camera node
# (sensors/ros_camera_node.py) is fully redundant behind it and must NOT run: it has
# no D405 driver and its default wrist-UVC path double-owns the wrist /dev nodes
# against the bridge. camera_node:=false keeps it off in every mode.
# lidar:=false ALWAYS: the RPLIDAR moved into the hardware_bridge (LIDAR subsystem,
# /scan + lidar->lidar_link TF) — the old apt sllidar_ros2 shell-out must not fight
# it for the CP2102 serial. real_run's SLAM arbitration still just watches /scan.
hw_args=(
  base:="$BASE" arms:="$ARMS" body:="$BODY" hands:="$HANDS" hand_sides:="$HAND_SIDES"
  sensors:="$SENSORS" lidar:=false status:=true slam:=false
  slam_odom:="$SLAM_ODOM" rviz:="$local_rviz" teensy_ip:="$TEENSY_IP"
  camera_node:=false
)
[ -n "$LEFT_PORT" ]  && hw_args+=(left_port:="$LEFT_PORT")
[ -n "$RIGHT_PORT" ] && hw_args+=(right_port:="$RIGHT_PORT")
_spawn "$LOG_HW" ros2 launch mabel_bringup hardware.launch.py "${hw_args[@]}"

# Wait for the TF tree / odom so SLAM + Nav2 start against a live graph.
say "          waiting for /odom (base odometry / TF) …"
if _wait_topic /odom 45; then ok "bring-up up — /odom is advertised."
else warn "/odom not seen in 45s (base hardware not connected?) — continuing; check $LOG_HW."; fi

# 1.5) SLAM — the architecture was decided at step 0.5 (LIDAR_MODE).
#    LiDAR mode: laser_odometry owns odom->base_link (odom DERIVED FROM THE LIDAR),
#    slam_toolbox owns /map (via /map_lidar) + map->odom, Nav2 runs on the LiDAR
#    grid, and nvblox runs MESH-ONLY so the front cam's 3D mesh keeps streaming to
#    the apps (cuVSLAM off — one map->odom owner only).
#    Visual mode: cuVSLAM + nvblox own the full picture (3D mesh + /map via
#    /map_visual + map->odom); odom->base_link goes to the WHEEL base only when it
#    is ACTUALLY streaming /odom, else cuVSLAM owns it (camera-only bench).
#    Once odom->base_link
#    exists, EVERY robot part follows from the URDF static/joint TFs (robot_state_
#    publisher; base_link is the URDF root). --slam-odom / --wheel-odom force it.
#    Both stacks run in the apt Isaac ROS env via slam_env.sh.
if [ "$DO_SLAM" = 1 ]; then
  if [ "$LIDAR_MODE" = 1 ]; then
    # ── LiDAR architecture (decided at step 0.5) ────────────────────────────────
    # laser_odometry OWNS odom->base_link (the robot odom is DERIVED FROM THE
    # LIDAR); slam_toolbox maps + localises (/map_lidar → /map + map->odom); Nav2
    # runs on the LiDAR grid; nvblox runs MESH-ONLY so the front cam's 3D mesh
    # keeps streaming to the apps (cuVSLAM off — one map->odom owner only).
    say "slam      LiDAR odometry (scan-matching → odom->base_link + /odom_laser)   (log: $LOG_LODOM)"
    _spawn "$LOG_LODOM" python3 "$WS/src/mabel_slam/mabel_slam/laser_odometry.py"
    say "slam      2D slam_toolbox (LiDAR → /map_lidar → /map + map->odom)   (log: $LOG_SLAM_2D)"
    _spawn_slam "$LOG_SLAM_2D" ros2 launch mabel_slam lidar_slam.launch.py use_sim_time:=false
    say "slam      nvblox 3D mesh only (front cam → app mesh stream; no cuVSLAM/2D)   (log: $LOG_SLAM)"
    _spawn_slam "$LOG_SLAM" ros2 launch mabel_slam visual_slam.launch.py \
      env:=${MABEL_SLAM_PROFILE:-navigation} use_sim_time:=false cuvslam:=false nvblox:=true nav_2d_map:=false
  else
    # ── VISUAL architecture ─────────────────────────────────────────────────────
    # odom->base_link owner (unless forced): wheel when it is really streaming
    # odom, else VISUAL cuVSLAM — the default the operator asked for.
    wheel_live=0
    _topic_has_data /odom 6 && wheel_live=1
    if [ "$SLAM_ODOM_FORCED" = 0 ]; then
      [ "$wheel_live" = 1 ] && SLAM_ODOM=false || SLAM_ODOM=true
    fi
    if [ "$SLAM_ODOM" = true ]; then
      say "          VISUAL cuVSLAM owns odom->base_link + 3D mesh + 2D /map_visual → /map + map->odom (no wheel odom)."
    else
      say "          wheel owns odom->base_link; visual cuVSLAM publishes map->odom + 3D mesh + 2D /map_visual → /map."
    fi
    say "slam      visual cuVSLAM + nvblox (front cam → 3D mesh + /map_visual → /map + map->odom)   (log: $LOG_SLAM)"
    _spawn_slam "$LOG_SLAM" ros2 launch mabel_slam visual_slam.launch.py \
      env:=${MABEL_SLAM_PROFILE:-navigation} use_sim_time:=false nav_2d_map:=true publish_odom_to_base_tf:="$SLAM_ODOM"
  fi
else
  say "slam      skipped (--no-slam)"
fi

# 2) Nav2 — NavFn A* + MPPI on the real clock. prepose OFF by default so the arms
#    are NOT auto-tucked on the bench (pass --prepose to enable that safety fold).
if [ "$DO_NAV" = 1 ]; then
  say "nav       Nav2 NavFn A* + MPPI (mabel_navigation)   (log: $LOG_NAV)"
  _spawn "$LOG_NAV" ros2 launch mabel_navigation navigation.launch.py \
    use_sim_time:=false rviz:=false prepose:="$NAV_PREPOSE"
else
  say "nav       skipped (--no-nav)"
fi

# 3) Web/iOS bridge — teleop control + the live /map (nvblox mesh + odom + Nav2
#    path) the app's Navigation tab renders (the in-browser/iOS "RViz"). The camera
#    MJPEG video is already served by camera_hal on :$HTTP_PORT (hardware.launch
#    --serve), so this just adds the teleop + map WebSocket on :$WS_PORT (+ :9092).
#    real:ros => the bridge mirrors the robot's state, it does NOT own physics.
if [ "$DO_WEB" = 1 ]; then
  : > "$LOG_BRIDGE"
  # The robot's pose on the /map comes from odom: on the real robot that's the
  # wheel /odom; on a bench running self-contained SLAM (slam_odom=true, no wheel
  # odom) use cuVSLAM's odometry so the Nav tab still shows the robot moving.
  # LiDAR mode: the pose comes from the LiDAR-derived odometry (/odom_laser) —
  # cuVSLAM is off, and the wheel /odom is twist-only feedback.
  map_odom="/odom"; [ "$SLAM_ODOM" = true ] && map_odom="/visual_slam/tracking/odometry"
  [ "$LIDAR_MODE" = 1 ] && map_odom="/odom_laser"
  benv=(MABEL_DISPATCH_TARGET=real:ros MABEL_MAP_STREAM=1
        MABEL_WS_PORT="$WS_PORT" MABEL_HTTP_PORT="$HTTP_PORT"
        MABEL_MAP_ODOM_TOPIC="$map_odom"
        # 96xx data plane (docs/data_streams.md): :9600 telemetry, :9601-9604 clouds
        # (RealSense/ZED PointCloud2 → CLOUD records), :9605 SLAM mesh+pose+grid.
        # Read-only, demand-driven, LAN/Tailscale direct only.
        # :9620-9623 dedicated video is served by the camera HAL, not this bridge.
        MABEL_DATA_STREAMS="${MABEL_DATA_STREAMS:-1}"
        # shaded TRIANGLE surfaces (nvblox mesh_marker) instead of the point
        # cloud fallback — the Control Center / apps render real mesh geometry
        MABEL_MAP_MESH_MODE="${MABEL_MAP_MESH_MODE:-mesh}"
        # PROCESS SPLIT (2026-07-13): the heavy map/localization/cloud/SLAM planes
        # run in a SEPARATE process (perception_server.py, below) so their rclpy
        # executor threads never share this process's GIL with the real-time joint
        # planes. Before the split, that GIL contention starved robot_state (:9090)
        # and proprio (:9600) to ~0.5 Hz; after it they hold 90+/50 Hz. This env
        # tells the gateway to skip creating map/:9092, slam/:9605 and clouds/:9601-4
        # (perception_server owns them). MABEL_PERCEPTION_EXTERNAL=0 reverts to the
        # old single-process layout.
        MABEL_PERCEPTION_EXTERNAL="${MABEL_PERCEPTION_EXTERNAL:-1}"
        # ── Jetson CPU relief for real-time whole-body retarget (2026-07-21) ──
        # Thor (14 cores) hit load ~40 during teleop; the gateway alone spiked to 500%
        # CPU / 5 Hz in RETARGET mode because each mj_forward / numpy-BLAS call spawns
        # threads that OVERSUBSCRIBE the cores (3 concurrent mj_forward callers, each
        # fanning to 14 OpenMP threads). Cap every math thread-pool to 1 so each call
        # stays on one core; solve both arms INLINE (the ThreadPool just thrashed the
        # GIL for two ~1 ms solves); HALVE the gravity + feedback mj_forward rates
        # (30 Hz tracks the slow dynamics they serve). Keeps the retarget loop real-time.
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
        MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
        VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
        MABEL_ARM_SEQUENTIAL="${MABEL_ARM_SEQUENTIAL:-1}"
        MABEL_GRAV_HZ="${MABEL_GRAV_HZ:-30}"
        MABEL_FEEDBACK_HZ="${MABEL_FEEDBACK_HZ:-30}")
  [ -n "$APP_TOKEN" ] && benv+=(MABEL_TOKEN="$APP_TOKEN")
  say "web       teleop + joint state (robot_state :$WS_PORT, proprio :9600) — video on :$HTTP_PORT   (log: $LOG_BRIDGE)"
  [ "${MABEL_DATA_STREAMS:-1}" = 1 ] && \
    say "data      :9600 telemetry (joints/control) · :9620-9623 video (HAL)   [MABEL_DATA_STREAMS=0 off]"
  # stdin from /dev/null: the gateway starts a stdin-reader thread (sim ENTER/R),
  # and a BACKGROUND process reading the controlling TTY gets SIGTTIN-stopped —
  # fatal once the cockpit TUI owns the terminal in raw mode. systemd runs have
  # no TTY anyway, so this only makes the manual run match that behavior.
  ( source "$ENV_SH" >>"$LOG_BRIDGE" 2>&1 || exit 1
    cd "$SCRIPT_DIR" || exit 1
    exec env "${benv[@]}" python3 sim_teleop_bridge.py --scene "$SCENE" ) </dev/null >>"$LOG_BRIDGE" 2>&1 &
  PIDS+=($!)

  # Perception process: map (mesh+2D grid+odom) :9092, clouds :9601-9604, SLAM
  # :9605 — the localization/map streaming, isolated from the joint planes above.
  if [ "${MABEL_PERCEPTION_EXTERNAL:-1}" = 1 ] && [ "${MABEL_DATA_STREAMS:-1}" = 1 ]; then
    LOG_PERCEPTION="$LOG_DIR/mabel_real_perception.log"
    say "percept   map+odom :9092 · clouds :9601-9604 · slam :9605  (own process/GIL; log: $LOG_PERCEPTION)"
    ( source "$ENV_SH" >>"$LOG_PERCEPTION" 2>&1 || exit 1
      cd "$SCRIPT_DIR" || exit 1
      exec env "${benv[@]}" python3 perception_server.py ) >>"$LOG_PERCEPTION" 2>&1 &
    PIDS+=($!)
  fi
else
  say "web       skipped (--no-web)"
fi

# 3.5) Control Center web studio — the browser control GUI (visualizer + teleop
#      client + server monitor, web_gui/control_center) on :$STUDIO_PORT, so
#      launching the robot also hands out a control surface with zero extra steps.
#      IDEMPOTENT: thor normally runs the studio as a user service
#      (mabel-control-center.service) — if ANYTHING already listens on the port it
#      is REUSED, never duplicated; only a studio THIS run started is reaped on exit.
if [ "$DO_STUDIO" = 1 ]; then
  if ss -ltn 2>/dev/null | grep -qE "[:.]${STUDIO_PORT}([[:space:]]|$)"; then
    ok "studio    Control Center already serving :$STUDIO_PORT (systemd service or a previous run) — reusing it."
  elif [ -x "$STUDIO_SH" ]; then
    say "studio    Control Center web GUI → http://$(_lan_ip):$STUDIO_PORT   (log: $LOG_STUDIO)"
    _spawn "$LOG_STUDIO" env MABEL_CC_PORT="$STUDIO_PORT" MABEL_NO_BROWSER=1 \
      MABEL_WS_PORT="$WS_PORT" MABEL_HTTP_PORT="$HTTP_PORT" "$STUDIO_SH"
    STUDIO_OURS=1
  else
    warn "studio    launcher missing at $STUDIO_SH — no web control GUI this run."
  fi
else
  say "studio    skipped (--no-studio)"
fi

# Pop the studio in a browser when a human is at the desk (headless/SSH launches
# skip it). Non-blocking: waits for the studio to answer, then opens the window.
# MABEL_NO_BROWSER=1 suppresses. Cross-platform: macOS always has a GUI (`open`);
# Linux needs a live X/Wayland display + `xdg-open`. (We pass MABEL_NO_BROWSER=1 to
# the studio launcher above, so this parent is the SOLE opener — it must cover
# macOS too, else nothing pops the tab there.)
_BROWSER_OPENER=""
if [ "$(uname -s)" = "Darwin" ] && command -v open >/dev/null 2>&1; then
  _BROWSER_OPENER="open"                                   # macOS: GUI always present
elif _display_ok && command -v xdg-open >/dev/null 2>&1; then
  _BROWSER_OPENER="xdg-open"                               # Linux: only with a live display
fi
if [ "$DO_STUDIO" = 1 ] && [ "${MABEL_NO_BROWSER:-0}" != 1 ] && [ -n "$_BROWSER_OPENER" ]; then
  say "studio    opening http://127.0.0.1:$STUDIO_PORT in the browser once it answers."
  ( for _ in $(seq 1 60); do
      if curl -s -o /dev/null -m 1 "http://127.0.0.1:$STUDIO_PORT/" 2>/dev/null; then
        "$_BROWSER_OPENER" "http://127.0.0.1:$STUDIO_PORT/" >/dev/null 2>&1 || true
        exit 0
      fi
      sleep 0.5
    done ) >/dev/null 2>&1 &
  PIDS+=($!)
fi

# 4) Hardware status watchdog — periodic liveness of EVERY subsystem + automatic
#    RealSense uvcvideo rebind, so a hot-swapped camera (head<->base on one cable)
#    is detected with no relaunch. Runs in THIS shell so warn()/say() colours work.
if [ "$DO_WATCH" = 1 ]; then
  say "watch     periodic hardware status every ${WATCH_INTERVAL}s (cameras/base/SLAM + RealSense uvcvideo self-heal)"
  if [ "$DO_TUI" = 1 ]; then
    # cockpit mode: the watchdog must not print over the curses screen — it
    # writes to $LOG_WATCH and the cockpit tails it, parsing the newest
    # `[hw HH:MM:SS] …` line into its live systems row.
    : > "$LOG_WATCH"
    _hw_watchdog >>"$LOG_WATCH" 2>&1 &
  else
    _hw_watchdog &
  fi
  PIDS+=($!)
else
  say "watch     skipped (--no-watch)"
fi

_banner

# Hold the foreground; cleanup() runs on Ctrl-C / any exit.
# Cockpit mode: the retro TUI owns the terminal — telemetry from the gateway's
# :9600 plane, hardware liveness from the watchdog's [hw] line, logs tailed
# from /tmp. Quitting it (q or Ctrl-C) IS the shutdown: this script then exits
# and the EXIT trap tears the whole robot down, exactly like plain Ctrl-C.
# Log paths ride $MABEL_COCKPIT_LOGS (env, NOT argv): several of these paths
# contain process names (…sim_teleop_bridge…-adjacent patterns) that our own
# pkill -f reap sweeps match, and pkill sees argv — env is invisible to it.
if [ "$DO_TUI" = 1 ]; then
  MABEL_COCKPIT_LOGS="watch=$LOG_WATCH,bridge=$LOG_BRIDGE,hw=$LOG_HW,hal=$LOG_HWBRIDGE,slam=$LOG_SLAM,nav=$LOG_NAV,studio=$LOG_STUDIO" \
    python3 "$SCRIPT_DIR/tui/mabel_cockpit.py" --mode real
  say "cockpit closed — shutting the robot bring-up down…"
  exit 0
fi
while true; do sleep 3600 & wait $!; done

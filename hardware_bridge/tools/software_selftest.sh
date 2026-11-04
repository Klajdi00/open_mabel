#!/usr/bin/env bash
# software_selftest.sh — validate the WHOLE base path in software, NO real Teensy.
#
# Brings up a mock Teensy on 127.0.0.1, points the hardware bridge at it (so NOTHING
# reaches the real robot at 192.168.124.10), runs the ROS base_bridge_client +
# swerve_odometry exactly as real_run.sh wires them, and asserts BOTH directions:
#   • state   (Teensy → ROS):   /mabel/hw/base/joint_states → /base/joint_states → /odom LIVE
#   • control (ROS → Teensy):   /mabel_cmd → /mabel/hw/base/cmd_vel → IK+calib → mock RX
#   • stability: a constant command stays STEADY at the Teensy (no oscillation)
#   • estop + singleton lock.
# Runs in an isolated ROS_DOMAIN_ID so it can't touch any running stack.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HWB="$(cd "$HERE/.." && pwd)"
MABEL_ROOT="$(cd "$HWB/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-44}"
FAKE_LOG=/tmp/selftest_fake_teensy.log
BR_LOG=/tmp/selftest_bridge.log
CL_LOG=/tmp/selftest_client.log
OD_LOG=/tmp/selftest_odom.log
PIDS=()
PASS=0; FAIL=0
ok(){ echo "  ✓ $*"; PASS=$((PASS+1)); }
no(){ echo "  ✗ $*"; FAIL=$((FAIL+1)); }
cleanup(){ for p in "${PIDS[@]:-}"; do kill -INT "$p" 2>/dev/null; done; sleep 1
           for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null; done
           pkill -9 -f "[f]ake_teensy.py" 2>/dev/null; pkill -9 -f "[h]ardware_bridge.py" 2>/dev/null; }
# GUARD (before the trap, so this exit never triggers the name-based cleanup): refuse
# to run if a bridge is already up — the cleanup below kills bridges by name and would
# take down a REAL one. Stop the real bridge first.
if pgrep -f "[h]ardware_bridge.py" >/dev/null 2>&1; then
  echo "✗ ABORT: a hardware bridge is already running (the real robot?). Stop it first"
  echo "  (sudo systemctl stop mabel-hardware-bridge, or stop real_run.sh) — refusing to"
  echo "  run so this test does not kill it."
  exit 1
fi
trap cleanup EXIT

source "$MABEL_ROOT/ros/scripts/mabel_env.sh" >/dev/null 2>&1

echo "== 1. mock Teensy on 127.0.0.1:5800 (no real hardware touched) =="
python3 "$HERE/fake_teensy.py" --host 127.0.0.1 --port 5800 >"$FAKE_LOG" 2>&1 & PIDS+=($!)
sleep 1
grep -q "listening" "$FAKE_LOG" && ok "mock Teensy up" || no "mock Teensy failed to start"

echo "== 2. hardware bridge -> 127.0.0.1 (base only) =="
( cd "$HWB"; MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP=127.0.0.1 \
  MABEL_HW_ARMS=false MABEL_HW_BODY=false MABEL_HW_HANDS=false MABEL_HW_LIFT=false \
  MABEL_HW_CAMERAS=false MABEL_HW_LOCK=/tmp/selftest_hwb.lock ./run_bridge.sh ) >"$BR_LOG" 2>&1 & PIDS+=($!)
sleep 4
grep -q "base online" "$BR_LOG" && ok "bridge base subsystem online" || no "bridge did not come online"

echo "== 3. base_bridge_client + swerve_odometry (as real_run wires them) =="
ros2 run mabel_base_hardware swerve_node --ros-args -r __node:=base_bridge_client \
  -r /joint_states:=/base/joint_states >"$CL_LOG" 2>&1 & PIDS+=($!)
ros2 run mabel_base_hardware swerve_odometry --ros-args \
  -r /joint_states:=/base/joint_states >"$OD_LOG" 2>&1 & PIDS+=($!)
sleep 3

echo "== 4. STATE path (Teensy → ROS): joint_states + /odom must be LIVE =="
timeout 5 ros2 topic echo /mabel/hw/base/joint_states --once >/dev/null 2>&1 \
  && ok "/mabel/hw/base/joint_states LIVE (bridge reads mock encoders)" \
  || no "/mabel/hw/base/joint_states silent"
timeout 5 ros2 topic echo /base/joint_states --once >/dev/null 2>&1 \
  && ok "/base/joint_states LIVE (client relayed it)" || no "/base/joint_states silent"
# This is EXACTLY real_run.sh's base-detection probe (_topic_has_data /odom):
if timeout 6 ros2 topic echo /odom --once >/dev/null 2>&1; then
  ok "/odom LIVE → real_run.sh WOULD DETECT the base"
else no "/odom silent → real_run.sh would NOT detect the base"; fi

echo "== 5. CONTROL path (ROS → Teensy) + STABILITY: steady cmd → steady at Teensy =="
ros2 topic pub -r 10 /mabel_cmd mabel_interfaces/msg/MabelSwerveCmd \
  "{vx: 0.2, vy: 0.0, omega: 0.0, lift_pos: 0.0, hardware_state: 0}" >/dev/null 2>&1 & CMDPID=$!
sleep 4
kill "$CMDPID" 2>/dev/null
# The mock logs DISTINCT commands/sec. A single non-steady window is just a setpoint
# CHANGE (idle→cmd); REAL oscillation is SUSTAINED — many consecutive non-steady
# windows while the input is constant. So fail only on >=2 consecutive OSCILLATING.
MAXRUN=$(awk '/OSCILLATING/{r++; if(r>m)m=r; next}{r=0} END{print m+0}' "$FAKE_LOG")
NBLIP=$(grep -c OSCILLATING "$FAKE_LOG")
if [ "${MAXRUN:-0}" -ge 2 ]; then
  no "SUSTAINED oscillation: $MAXRUN consecutive non-steady windows"; grep OSCILLATING "$FAKE_LOG" | tail -3
else
  ok "no sustained oscillation (max consecutive=$MAXRUN; $NBLIP isolated setpoint-change blip(s), steady otherwise)"
fi
LASTDRIVE=$(grep -oE "drive=\[[^]]*\]" "$FAKE_LOG" | tail -1)
echo "    last command at mock: $LASTDRIVE"
echo "$LASTDRIVE" | grep -qE "0\.2|-0\.2" && ok "mock received the expected |drive|≈0.2 for vx=0.2" \
  || no "mock drive not ~0.2 (got $LASTDRIVE)"

echo "== 6. ESTOP: hardware_state=1 → drive zeroed at Teensy =="
ros2 topic pub -r 10 /mabel_cmd mabel_interfaces/msg/MabelSwerveCmd "{hardware_state: 1}" >/dev/null 2>&1 & EPID=$!
sleep 3; kill "$EPID" 2>/dev/null
LASTDRIVE=$(grep -oE "drive=\[[^]]*\]" "$FAKE_LOG" | tail -1)
echo "$LASTDRIVE" | grep -qE "drive=\[(-?0\.0,? ?){3}\]|\[0\.0, 0\.0, 0\.0\]" && ok "estop zeroed drive at mock ($LASTDRIVE)" \
  || no "estop did not zero drive (got $LASTDRIVE)"

echo "== 7. SINGLETON: a 2nd bridge must REFUSE (no double-commanding) =="
( cd "$HWB"; MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP=127.0.0.1 \
  MABEL_HW_LOCK=/tmp/selftest_hwb.lock ./run_bridge.sh ) >/tmp/selftest_bridge2.log 2>&1
RC=$?
[ "$RC" = 3 ] && ok "2nd bridge refused (exit 3, singleton lock held)" \
  || no "2nd bridge did NOT refuse (rc=$RC) — would double-command"

echo
echo "================  RESULT: $PASS passed, $FAIL failed  ================"
[ "$FAIL" = 0 ] && echo "ALL GREEN — two-way base comms stable in software." || echo "FAILURES above."
exit "$FAIL"

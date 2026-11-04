#!/usr/bin/env bash
# reconnect_test.sh — prove the bridge auto-recovers when the Teensy is unplugged
# then reconnected. Uses the mock Teensy on 127.0.0.1 (never the real hardware):
#   1 bridge up, NO Teensy        -> no telemetry (idle, correct)
#   2 Teensy appears              -> bridge reconnects, /joint_states go LIVE
#   3 Teensy disconnected         -> telemetry stops, bridge logs link LOST
#   4 Teensy reconnected          -> bridge picks it back up, LIVE again
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HWB="$(cd "$HERE/.." && pwd)"
MABEL_ROOT="$(cd "$HWB/.." && pwd)"
source "$MABEL_ROOT/ros/scripts/mabel_env.sh" >/dev/null 2>&1
export ROS_DOMAIN_ID=47
LOCK=/tmp/reconnect_hwb.lock
BR_LOG=/tmp/rc_bridge.log
FAKE_LOG=/tmp/rc_fake.log
TEENSY=127.0.0.1
PASS=0; FAIL=0
ok(){ echo "  ✓ $*"; PASS=$((PASS+1)); }
no(){ echo "  ✗ $*"; FAIL=$((FAIL+1)); }
cleanup(){ pkill -9 -f "[h]ardware_bridge.py" 2>/dev/null; pkill -9 -f "[f]ake_teensy.py" 2>/dev/null; }
# GUARD (before the trap): cleanup name-kills bridges, so refuse to run if a REAL bridge
# is already up. Stop the real bridge first.
if pgrep -f "[h]ardware_bridge.py" >/dev/null 2>&1; then
  echo "✗ ABORT: a hardware bridge is already running (real robot?). Stop it first"
  echo "  (sudo systemctl stop mabel-hardware-bridge) — refusing so this test can't kill it."
  exit 1
fi
trap cleanup EXIT
cleanup; rm -f "$LOCK"; sleep 1

js_live(){ timeout "${1:-3}" ros2 topic echo /mabel/hw/base/joint_states --once >/dev/null 2>&1; }
start_fake(){ python3 "$HERE/fake_teensy.py" --host "$TEENSY" --port 5800 >"$FAKE_LOG" 2>&1 & echo $!; }

echo "== 1. bridge up with the Teensy ABSENT =="
( cd "$HWB"; MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP=$TEENSY MABEL_HW_ARMS=false \
  MABEL_HW_BODY=false MABEL_HW_HANDS=false MABEL_HW_LIFT=false MABEL_HW_CAMERAS=false \
  MABEL_HW_LOCK=$LOCK ./run_bridge.sh ) >"$BR_LOG" 2>&1 &
sleep 5
if js_live 3; then no "joint_states LIVE with no Teensy (unexpected)"; else ok "idle while Teensy absent (no false telemetry)"; fi

echo "== 2. Teensy APPEARS -> bridge must reconnect =="
FPID=$(start_fake); sleep 5
if js_live 4; then ok "bridge picked up the Teensy after it appeared (auto-connect)"; else no "bridge did NOT pick up the appearing Teensy"; fi

echo "== 3. Teensy DISCONNECTED (kill mock) -> telemetry stops =="
kill -9 "$FPID" 2>/dev/null; sleep 4
if js_live 3; then no "joint_states still LIVE after disconnect"; else ok "telemetry stopped on disconnect (loss detected)"; fi
grep -q "link LOST" "$BR_LOG" && ok "bridge logged 'link LOST'" || echo "    (no LOST log line yet — timing; functional check above is what matters)"

echo "== 4. Teensy RECONNECTED -> bridge must pick it back up =="
FPID=$(start_fake); sleep 5
if js_live 5; then ok "bridge RE-ACQUIRED the Teensy after reconnect (refresh works)"; else no "bridge did NOT re-acquire after reconnect"; fi
grep -q "link UP" "$BR_LOG" && ok "bridge logged 'link UP' on recovery" || echo "    (no UP log line — non-fatal)"

echo
echo "================  RECONNECT RESULT: $PASS passed, $FAIL failed  ================"
[ "$FAIL" = 0 ] && echo "ROBUST — bridge refreshes and picks the Teensy back up after a disconnect." || echo "FAILURES above."
exit "$FAIL"

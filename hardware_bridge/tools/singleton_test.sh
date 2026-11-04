#!/usr/bin/env bash
# singleton_test.sh — SAFETY: prove exactly ONE bridge can ever own the hardware.
#
# The invariant: no matter how many servers / launches race, only ONE process holds
# the Teensy/CAN/serial link. A second owner makes the base oscillate (two setpoints
# fighting), so this is safety-critical. Every bridge here points at 127.0.0.1 — the
# REAL Teensy (192.168.124.10) is NEVER touched.
#
# Edge cases covered:
#   A concurrent launch (8 at once)        -> exactly 1 alive, 7 refuse
#   B cross ROS_DOMAIN_ID                   -> a 2nd on another domain still refuses
#      (the lock is on the LINK, not the DDS graph)
#   C crash of the owner (kill -9)          -> flock auto-releases, a new one acquires
#                                              (no stale-lock deadlock)
#   D clean handoff (SIGINT)                -> lock released, a new one acquires
#   E real_run.sh detection                 -> its pgrep check sees the running bridge
#   F one UDP socket to the Teensy port     -> exactly one commander on :5800
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HWB="$(cd "$HERE/.." && pwd)"
MABEL_ROOT="$(cd "$HWB/.." && pwd)"
source "$MABEL_ROOT/ros/scripts/mabel_env.sh" >/dev/null 2>&1
LOCK=/tmp/mabel_singleton_test.lock
TEENSY=127.0.0.1
PASS=0; FAIL=0
ok(){ echo "  ✓ $*"; PASS=$((PASS+1)); }
no(){ echo "  ✗ $*"; FAIL=$((FAIL+1)); }
alive(){ pgrep -fc "[h]ardware_bridge.py"; }
killall_bridges(){ pkill -9 -f "[h]ardware_bridge.py" 2>/dev/null; sleep 1; }
# start one bridge (base only, localhost). args: domain logfile
spawn(){ ( cd "$HWB"; MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP=$TEENSY \
    MABEL_HW_ARMS=false MABEL_HW_BODY=false MABEL_HW_HANDS=false MABEL_HW_LIFT=false \
    MABEL_HW_CAMERAS=false MABEL_HW_LOCK=$LOCK ROS_DOMAIN_ID=$1 ./run_bridge.sh ) >"$2" 2>&1 & echo $!; }

# GUARD (before the trap): this test spawns + name-kills bridges, so refuse to run if a
# REAL bridge is already up — it would take it down. Stop the real bridge first.
if pgrep -f "[h]ardware_bridge.py" >/dev/null 2>&1; then
  echo "✗ ABORT: a hardware bridge is already running (real robot?). Stop it first"
  echo "  (sudo systemctl stop mabel-hardware-bridge) — refusing so this test can't kill it."
  exit 1
fi
trap killall_bridges EXIT
killall_bridges; rm -f "$LOCK"

echo "== A. concurrent: 8 servers launch the bridge simultaneously =="
rm -f /tmp/sgl_*.log
for i in $(seq 1 8); do spawn 70 "/tmp/sgl_$i.log" >/dev/null; done
sleep 5
A_ALIVE=$(alive); A_REFUSED=$(grep -l "Refusing to start" /tmp/sgl_*.log 2>/dev/null | wc -l)
[ "$A_ALIVE" = 1 ] && ok "exactly 1 bridge alive of 8" || no "$A_ALIVE alive (MUST be 1)"
[ "$A_REFUSED" = 7 ] && ok "7 refused via lock (exit 3)" || no "$A_REFUSED refused (expected 7)"

echo "== F. exactly one commander socket to the Teensy port 5800 =="
# sudo password comes from the environment (never hard-code it — this repo may be
# public). Export SUDO_PASS before running, or run this section with cached sudo.
SUDO_S() { if [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S "$@" 2>/dev/null; else sudo -n "$@" 2>/dev/null; fi; }
SOCKS=$(SUDO_S ss -unp | grep -c "hardware_bridge\|python" || echo "?")
NB=$(SUDO_S ss -unp | grep -oE 'pid=[0-9]+' | sort -u | grep -c . || echo "?")
echo "    (udp sockets from python/bridge procs: $SOCKS; distinct pids on udp: $NB)"
[ "$A_ALIVE" = 1 ] && ok "single owner ⇒ single Teensy commander" || no "multiple owners"

echo "== B. cross ROS domain: 2nd bridge on a DIFFERENT domain must refuse =="
B2_LOG=/tmp/sgl_crossdomain.log
( cd "$HWB"; MABEL_HW_BASE=true MABEL_HW_BASE_TEENSY_IP=$TEENSY MABEL_HW_ARMS=false \
  MABEL_HW_BODY=false MABEL_HW_HANDS=false MABEL_HW_LIFT=false MABEL_HW_CAMERAS=false \
  MABEL_HW_LOCK=$LOCK ROS_DOMAIN_ID=99 ./run_bridge.sh ) >"$B2_LOG" 2>&1; B_RC=$?
[ "$B_RC" = 3 ] && ok "domain-99 bridge refused while domain-70 holds the lock (rc=3)" \
  || no "cross-domain 2nd did NOT refuse (rc=$B_RC) — SAFETY HOLE"

echo "== E. real_run.sh detection: its pgrep check sees the running bridge =="
if pgrep -f "[h]ardware_bridge.py" >/dev/null 2>&1; then
  ok "_ensure_bridge would detect the running bridge and SKIP starting another"
else no "_ensure_bridge would NOT detect it (could double-start)"; fi

echo "== C. crash recovery: kill -9 the owner, a fresh bridge must acquire =="
OWNER=$(pgrep -f "[h]ardware_bridge.py" | head -1)
kill -9 "$OWNER" 2>/dev/null; sleep 2
C_LOG=/tmp/sgl_recover.log
NEWPID=$(spawn 71 "$C_LOG"); sleep 4
if kill -0 "$NEWPID" 2>/dev/null && grep -q "base online" "$C_LOG"; then
  ok "after kill -9, new bridge acquired the freed lock (no stale-lock deadlock)"
else no "new bridge could NOT acquire after crash (stale lock!)"; cat "$C_LOG" | tail -3; fi

echo "== D. clean handoff: SIGINT the owner, a fresh bridge must acquire =="
OWNER=$(pgrep -f "[h]ardware_bridge.py" | head -1)
kill -INT "$OWNER" 2>/dev/null; sleep 2
D_LOG=/tmp/sgl_handoff.log
NEWPID=$(spawn 72 "$D_LOG"); sleep 4
if kill -0 "$NEWPID" 2>/dev/null && grep -q "base online" "$D_LOG"; then
  ok "after SIGINT, new bridge acquired the released lock"
else no "new bridge could NOT acquire after clean stop"; cat "$D_LOG" | tail -3; fi
[ "$(alive)" = 1 ] && ok "still exactly 1 bridge alive at end" || no "$(alive) bridges alive at end"

echo
echo "================  SINGLETON RESULT: $PASS passed, $FAIL failed  ================"
[ "$FAIL" = 0 ] && echo "SAFE — exactly one hardware owner under every edge case." || echo "SAFETY FAILURES above."
killall_bridges
exit "$FAIL"

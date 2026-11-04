#!/usr/bin/env bash
# calibrate_hand.sh — bridge-owned end-stop calibration for an ORCA hand.
#
# Calibration MUST own the TTL serial bus exclusively, so the running hardware bridge
# (which also owns the bus) is stopped for the duration and restarted afterwards. This is
# the one safe, single-command way to (re)calibrate: it stops the HAL, runs the
# stall-detection sweep (which WRITES <side>_calibration.yaml that the bridge then reloads
# on its next connect), and brings the HAL back up.
#
#   ./tools/calibrate_hand.sh right                 # calibrate all joints, right hand
#   ./tools/calibrate_hand.sh right --joints wrist  # only some joints (merged)
#   ./tools/calibrate_hand.sh right --finger index
#
# SAFETY: the hand physically moves each tendon to its hard-stops. Keep fingers clear and
# ensure the tendons are tensioned first. The sweep prompts for confirmation (pass --yes
# to skip). Torque is released on exit, including Ctrl-C.
set -uo pipefail

SIDE="${1:-right}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MABEL_ROOT="${MABEL_ROOT:-$(cd "$HERE/.." && pwd)}"
UNIT="mabel-hardware-bridge.service"

ENV_SH="${MABEL_ENV_SH:-$MABEL_ROOT/ros/scripts/mabel_env.sh}"
# shellcheck disable=SC1090
[ -f "$ENV_SH" ] && source "$ENV_SH"

# Free the bus: stop the bridge if the systemd service is running, remember to restart it.
RESTART=0
if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
    echo "[calibrate] stopping $UNIT to free the hand bus..."
    sudo systemctl stop "$UNIT" && RESTART=1
    sleep 1
fi
restore() {
    if [ "$RESTART" = "1" ]; then
        echo "[calibrate] restarting $UNIT (reloads the new calibration)..."
        sudo systemctl start "$UNIT"
    fi
}
trap restore EXIT

cd "$HERE"
echo "[calibrate] running stall-detection sweep on the $SIDE hand..."
"${PYTHON:-python3}" -m mabel_hw.drivers.orcahand.calibrate --model "$SIDE" "$@"

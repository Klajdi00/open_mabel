#!/usr/bin/env bash
# =============================================================================
# run_serial.sh  —  REAL-HARDWARE actuator calibration over the DM-USB2FDCAN
#                   *serial* adapter (works on macOS AND Linux)
# =============================================================================
# Counterpart to run_hardware.sh (which drives SocketCAN can0, Linux-only).
# This path needs no `ip link`/SocketCAN — it talks to the DM-USB2FDCAN over its
# USB serial port, so it is the way to calibrate from a Mac.
#
#     detect motor  ->  collect (staircase + chirp)  ->  identify
#       ->  export MuJoCo attrs  ->  plot
#
# USAGE
#   ./run_serial.sh <MOTOR> <ID> [VEL_FRAC] [-- extra collect.py args]
#
#   MOTOR     motor profile name (DM-J4310, DM-J4340P, DM-J8009P, ...)
#   ID        CAN id of the motor (integer)
#   VEL_FRAC  optional; max staircase speed as fraction of rated v_max (default 0.10)
#
# Override the port with PORT=/dev/cu.usbmodemXXXX (else auto-detected).
#
# SAFETY: bench-mount with the OUTPUT SHAFT FREE; hand on the e-stop. collect.py
# enables on start and ALWAYS disables on exit (incl. Ctrl-C).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python || command -v python3)"
MOTOR="${1:-}"; ID="${2:-}"; VEL_FRAC="${3:-0.10}"
if [ -z "$MOTOR" ] || [ -z "$ID" ]; then
  grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//' | sed -n '1,30p'
  exit 1
fi
EXTRA=()
for a in "$@"; do [ "${SEEN:-}" = 1 ] && EXTRA+=("$a"); [ "$a" = "--" ] && SEEN=1; done
PORT_ARG=(); [ -n "${PORT:-}" ] && PORT_ARG=(--port "$PORT")

echo "############################################################"
echo "# SERIAL calibration: motor=$MOTOR  id=$ID  vel_frac=$VEL_FRAC"
echo "############################################################"

echo "[1] detecting motor on the serial adapter (non-energizing)..."
"$PY" detect_serial.py "${PORT_ARG[@]}" --ids "$ID" || {
  echo "    ERROR — motor $ID did not answer. Checks: 24V on? CAN_H/L wiring? port?"
  exit 2; }

echo "[2] collecting (shaft must be free)..."
"$PY" collect.py --motor "$MOTOR" --id "$ID" --backend serial "${PORT_ARG[@]}" \
      --vel-frac "$VEL_FRAC" "${EXTRA[@]}"

RES="../results/${MOTOR}"
STAIR="${RES}/logs/staircase.csv"
CHIRP="${RES}/logs/chirp.csv"
DYN_ARG=""; [ -f "$CHIRP" ] && DYN_ARG="--dynamic $CHIRP"
echo "[3] identifying friction + inertia..."
"$PY" identify.py --staircase "$STAIR" $DYN_ARG --motor "$MOTOR" --model stribeck

echo "[4] exporting MuJoCo attributes..."
"$PY" export_mujoco.py --params "${RES}/params.json" --joint r_elbow || true

echo "[5] plotting raw data..."
"$PY" plot_data.py --motor "$MOTOR" || true

echo
echo "DONE. Outputs in $(cd "$(dirname "$0")/.." && pwd)/results/${MOTOR}/"

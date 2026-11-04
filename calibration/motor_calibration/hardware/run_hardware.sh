#!/usr/bin/env bash
# =============================================================================
# run_hardware.sh  —  REAL-HARDWARE actuator calibration (DaMiao on can0)
# =============================================================================
# Unlike ./run_selftest.sh (which is the SIMULATION self-test and never touches
# the motor), this script drives a real motor end-to-end:
#
#     bring up can0  ->  non-energizing scan  ->  collect (staircase + chirp)
#       ->  identify friction+inertia  ->  export MuJoCo attrs  ->  plot
#
# USAGE
#   ./run_hardware.sh <MOTOR> <ID> [VEL_FRAC] [-- extra act_01 args]
#
#   MOTOR     motor profile name (DM-J4310, DM-J4340P, DM-J8009P, ...)
#   ID        CAN id of the motor (integer)
#   VEL_FRAC  optional; max staircase speed as a fraction of rated v_max
#             (default 0.10 — gentle; raise only after a clean low-speed run)
#
# EXAMPLES
#   ./run_hardware.sh DM-J4310  2                 # gentle default run
#   ./run_hardware.sh DM-J4340P 1 0.30            # faster, more speed range
#   ./run_hardware.sh DM-J4310  2 0.10 -- --no-chirp --levels 6   # friction only
#
# SAFETY
#   * Bench-mount the motor with the OUTPUT SHAFT FREE; keep a hand on the e-stop.
#   * collect.py enables on start and ALWAYS disables on exit (incl Ctrl-C).
#   * Start with a low VEL_FRAC; the position-ramp staircase forces the motor
#     through stiction so it WILL move at every level.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="$(command -v python || command -v python3)"
CHANNEL="${CHANNEL:-can0}"
BITRATE="${BITRATE:-1000000}"

MOTOR="${1:-}"; ID="${2:-}"; VEL_FRAC="${3:-0.10}"
if [ -z "$MOTOR" ] || [ -z "$ID" ]; then
  grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//' | sed -n '1,30p'
  exit 1
fi
# collect any args after a literal "--" and forward them to collect.py
EXTRA=()
for a in "$@"; do [ "${SEEN:-}" = 1 ] && EXTRA+=("$a"); [ "$a" = "--" ] && SEEN=1; done

echo "############################################################"
echo "# HARDWARE calibration: motor=$MOTOR  id=$ID  vel_frac=$VEL_FRAC"
echo "#   channel=$CHANNEL @ ${BITRATE}bps"
echo "############################################################"

# --- 1. bring up can0 (no-op if already up) ---------------------------------
if ! ip link show "$CHANNEL" | grep -q "state UP"; then
  echo "[1] bringing up $CHANNEL @ ${BITRATE} (needs sudo)..."
  sudo ip link set "$CHANNEL" up type can bitrate "$BITRATE"
else
  echo "[1] $CHANNEL already UP"
fi
# The default CAN txqueuelen is only 10 frames -- at the 500Hz command rate a
# brief bus hiccup overflows it instantly (ENOBUFS). Bump it for headroom.
sudo ip link set "$CHANNEL" txqueuelen 1000 2>/dev/null || true

# --- 2. non-energizing scan: confirm the motor answers ----------------------
echo "[2] probing id $ID (non-energizing disable frame)..."
HEXID=$(printf '%03X' "$ID")
( timeout 1.5 candump -t z "$CHANNEL" > /tmp/hwcalib_probe.log 2>&1 & sleep 0.3
  cansend "$CHANNEL" "${HEXID}#FFFFFFFFFFFFFFFD"; sleep 0.2
  cansend "$CHANNEL" "${HEXID}#FFFFFFFFFFFFFFFD"; wait ) || true
if grep -qvi "FF FF FF FF FF FF FF FD" /tmp/hwcalib_probe.log 2>/dev/null; then
  echo "    OK — motor replied:"; grep -vi "FF FF FF FF FF FF FF FD" /tmp/hwcalib_probe.log | sed 's/^/      /'
else
  echo "    ERROR — no reply from id $ID. Checks: 24V on? CAN_H/L wiring? correct id?"
  echo "    Tip: scan all ids with:  ./scan.sh"
  exit 2
fi

# --- 3. collect (staircase + chirp) -----------------------------------------
echo "[3] collecting (shaft must be free)..."
"$PY" collect.py --motor "$MOTOR" --id "$ID" \
      --channel "$CHANNEL" --vel-frac "$VEL_FRAC" "${EXTRA[@]}"

# --- 4. identify ------------------------------------------------------------
RES="../results/${MOTOR}"
STAIR="${RES}/logs/staircase.csv"
CHIRP="${RES}/logs/chirp.csv"
DYN_ARG=""; [ -f "$CHIRP" ] && DYN_ARG="--dynamic $CHIRP"
echo "[4] identifying friction + inertia..."
"$PY" identify.py --staircase "$STAIR" $DYN_ARG \
      --motor "$MOTOR" --model stribeck

# --- 5. export MuJoCo attrs -------------------------------------------------
echo "[5] exporting MuJoCo attributes..."
"$PY" export_mujoco.py --params "${RES}/params.json" --joint r_elbow || true

# --- 6. plot ----------------------------------------------------------------
echo "[6] plotting raw data..."
"$PY" plot_data.py --motor "$MOTOR" || true

echo
echo "DONE. Outputs in $(cd "$(dirname "$0")/.." && pwd)/results/${MOTOR}/"

#!/usr/bin/env bash
# Flash the lift firmware to a Raspberry Pi Pico over USB with mpremote.
#
# Copies every module in firmware/ to the Pico's filesystem and resets it.
# Usage:
#   ./tools/deploy.sh            # copy all modules + reset
#   ./tools/deploy.sh repl       # copy, reset, then drop into the REPL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FW_DIR="$(cd "${SCRIPT_DIR}/../firmware" && pwd)"

if ! command -v mpremote >/dev/null 2>&1; then
  echo "error: mpremote not found. Install with: pip install mpremote" >&2
  exit 1
fi

# Resolve the lift Pico EXPLICITLY — never 'connect auto'. On MABEL's USB hub the
# first serial port is the base Teensy (16c0), so 'auto' would flash the wrong
# board. Target the RP2040/Pico by its USB identity (vendor 0x2E8A), preferring a
# stable /dev/serial/by-id symlink. Override with LIFT_PICO_DEV=/dev/... if needed.
PICO_DEV="${LIFT_PICO_DEV:-}"
if [[ -z "${PICO_DEV}" ]]; then
  PICO_DEV="$(ls /dev/serial/by-id/usb-MicroPython* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${PICO_DEV}" ]]; then
  PICO_DEV="$(mpremote connect list 2>/dev/null | awk '/ 2e8a:/{print $1; exit}')"
fi
if [[ -z "${PICO_DEV}" ]]; then
  echo "error: no RP2040/Pico (USB VID 2e8a) found — is the lift Pico plugged in," \
       "and is the port free (stop the hardware bridge first)?" >&2
  exit 1
fi
echo "Target Pico: ${PICO_DEV}"

echo "Copying firmware modules from ${FW_DIR} ..."
for f in "${FW_DIR}"/*.py; do
  name="$(basename "$f")"
  echo "  -> :${name}"
  mpremote connect "${PICO_DEV}" fs cp "$f" ":${name}"
done

echo "Resetting Pico ..."
mpremote connect "${PICO_DEV}" reset

if [[ "${1:-}" == "repl" ]]; then
  echo "Opening REPL (Ctrl-] to exit) ..."
  sleep 0.5
  mpremote connect "${PICO_DEV}" repl
fi

echo "Done."

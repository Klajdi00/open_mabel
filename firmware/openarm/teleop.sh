#!/usr/bin/env bash
# teleop.sh — bring up can0 and launch the DaMiao teleop GUI
#
# Recovery ladder (automatic, no manual steps needed):
#   1. Interface already exists → bring it UP cleanly
#   2. Interface missing → reload gs_usb kernel module
#   3. Still missing → udevadm trigger (re-enumerate USB)
#   4. Still missing → instructions to replug

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── macOS: gs_usb backend (no SocketCAN, no `ip link`, no kernel module) ──────
# The CANable/candleLight adapter is spoken to directly over USB via libusb, so
# there is nothing to bring up — just launch the GUI, which auto-selects gs_usb.
if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected — launching DaMiao teleop GUI (auto: serial adapter → gs_usb)…"
    # pyserial covers the DaMiao serial adapter (the reliable path); python-can +
    # gs_usb + libusb cover the CANable/candleLight fallback.
    if ! python3 -c "import serial" 2>/dev/null; then
        echo "Installing pyserial…"; python3 -m pip install --quiet pyserial
    fi
    if ! python3 -c "import can, gs_usb, usb" 2>/dev/null; then
        echo "Installing gs_usb CAN deps (libusb + python-can + gs_usb + pyusb)…"
        command -v brew >/dev/null && { brew list libusb >/dev/null 2>&1 || brew install libusb; }
        python3 -m pip install --quiet python-can gs_usb pyusb
    fi
    exec python3 "$SCRIPT_DIR/scripts/teleop_gui.py"
fi

CHANNEL="${1:-can0}"

# ── Helpers ───────────────────────────────────────────────────────────────────
_bring_up() {
    if ip link show "$CHANNEL" 2>/dev/null | grep -q "UP"; then
        return 0   # already up — leave it alone
    fi
    echo "Bringing up $CHANNEL at 1 Mbps…"
    # Bounce down first to clear stale CAN params (avoids RTNETLINK Protocol error)
    sudo ip link set "$CHANNEL" down 2>/dev/null || true
    sudo ip link set "$CHANNEL" up type can bitrate 1000000
}

_launch() {
    echo "Starting DaMiao teleop GUI on $CHANNEL"
    exec python3 "$SCRIPT_DIR/scripts/teleop_gui.py" --channel "$CHANNEL"
}

_wait_for_iface() {
    local deadline=$(( SECONDS + $1 ))
    while (( SECONDS < deadline )); do
        ip link show "$CHANNEL" &>/dev/null && return 0
        sleep 0.3
    done
    return 1
}

# ── Fast path: interface already exists ───────────────────────────────────────
if ip link show "$CHANNEL" &>/dev/null; then
    _bring_up
    _launch
fi

# ── Step 1: reload gs_usb kernel module ───────────────────────────────────────
echo "'$CHANNEL' not found — reloading gs_usb driver…"
sudo modprobe -r gs_usb 2>/dev/null || true
sleep 0.3
sudo modprobe gs_usb 2>/dev/null || true

if _wait_for_iface 2; then
    echo "  → '$CHANNEL' recovered via module reload."
    _bring_up
    _launch
fi

# ── Step 2: udevadm re-trigger (device present but udev missed it) ────────────
echo "  → gs_usb reload didn't help — triggering USB re-enumeration…"
sudo udevadm trigger --action=add --subsystem-match=usb 2>/dev/null || true
sudo udevadm settle --timeout=3 2>/dev/null || true
sudo modprobe gs_usb 2>/dev/null || true   # ensure module is loaded for any newly seen device

if _wait_for_iface 3; then
    echo "  → '$CHANNEL' recovered via udevadm trigger."
    _bring_up
    _launch
fi

# ── Still missing: diagnose and exit ─────────────────────────────────────────
echo ""
echo "USB devices currently detected:"
lsusb 2>/dev/null | sed 's/^/  /'
echo ""

# Check whether any candleLight/OpenMoko/GS USB adapter appears in lsusb
if lsusb 2>/dev/null | grep -qiE "1d50:606f|0483:1234|OpenMoko|candleLight|gs_usb"; then
    echo "INFO: A CAN adapter IS visible over USB but '$CHANNEL' still didn't appear."
    echo "      Try:  sudo modprobe gs_usb && sleep 2 && ip link show"
else
    echo "WARNING: No CAN adapter found in USB device list."
    echo "         The adapter may have lost power or come unplugged."
fi
echo ""
echo "ERROR: '$CHANNEL' still not found after all recovery attempts."
echo ""
echo "  → Unplug the USB CAN adapter, wait 3 s, replug, then re-run."
exit 1

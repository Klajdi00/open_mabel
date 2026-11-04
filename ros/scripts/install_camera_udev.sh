#!/usr/bin/env bash
# Install the USB udev rules the camera drivers need to CLAIM the devices.
#
# Symptom this fixes (RealSense): the node logs
#   ERROR (handle-libusb.h) failed to open usb interface: 0, error: RS2_USB_STATUS_ACCESS
#   [camera.front]: The requested device ... is NOT found. Will Try again.
# because the raw USB node is root:root mode 0664 -- the `plugdev` user can read
# (enumerate) but not WRITE (claim the streaming interface). The rule below sets
# the D4xx devices to 0666/plugdev so librealsense can open them with no sudo.
#
# The ZED rules (99-slabs.rules) ship with the ZED SDK and are usually already
# present; this script installs them too if a copy is found, else just notes it.
#
#   sudo ./scripts/install_camera_udev.sh        # then UNPLUG/REPLUG each camera
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo:  sudo $0" >&2
  exit 1
fi

RS_SRC="$WS/src/mabel_sensors/udev/99-realsense-libusb.rules"
RS_DST=/etc/udev/rules.d/99-realsense-libusb.rules
if [ -f "$RS_SRC" ]; then
  install -m 0644 "$RS_SRC" "$RS_DST"
  echo "[ok] installed $RS_DST"
else
  echo "[!!] missing $RS_SRC -- re-run the camera bring-up setup" >&2
fi

# Make sure the invoking user is in plugdev (the rule grants that group access).
REAL_USER="${SUDO_USER:-$USER}"
if ! id -nG "$REAL_USER" | tr ' ' '\n' | grep -qx plugdev; then
  usermod -aG plugdev "$REAL_USER"
  echo "[ok] added $REAL_USER to 'plugdev' (LOG OUT/IN for it to take effect)"
fi

udevadm control --reload-rules
udevadm trigger
echo
echo "udev rules reloaded. NOW UNPLUG AND REPLUG each camera (RealSense + ZED)"
echo "so the new permissions apply to the live device nodes."

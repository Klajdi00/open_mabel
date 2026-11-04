#!/usr/bin/env bash
# install.sh — install the MABEL hardware bridge as a boot service (headless).
#
# No GUI / xterm popups. Copies the unit, enables it for boot, and starts it now.
# Re-runnable (idempotent). Run with sudo, or it will re-exec itself with sudo.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT="mabel-hardware-bridge.service"
DEST="/etc/systemd/system/$UNIT"

if [ "$(id -u)" -ne 0 ]; then
    exec sudo -E bash "$0" "$@"
fi

echo "[install] making run_bridge.sh executable"
chmod +x "$HERE/../run_bridge.sh"

echo "[install] copying $UNIT -> $DEST"
install -m 0644 "$HERE/$UNIT" "$DEST"

echo "[install] reloading systemd + enabling at boot"
systemctl daemon-reload
systemctl enable "$UNIT"

echo "[install] (re)starting the bridge now"
systemctl restart "$UNIT"

echo
echo "[install] done. Check status with:"
echo "    systemctl status $UNIT"
echo "    journalctl -u $UNIT -f"
echo
echo "Enable more subsystems by uncommenting the MABEL_HW_* lines in $DEST,"
echo "then: sudo systemctl daemon-reload && sudo systemctl restart $UNIT"

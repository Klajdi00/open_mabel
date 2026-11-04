#!/usr/bin/env bash
# can_down.sh — cleanly tear down a SocketCAN interface (and any slcand attached
# to it). Safe to run whether the iface is native (gs_usb) or serial (slcan).
#
# Usage:
#   ./can_down.sh            # take down can0
#   ./can_down.sh -i can1
set -euo pipefail

IFACE="${CAN_IFACE:-can0}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--iface) IFACE="$2"; shift 2 ;;
        -h|--help)  echo "usage: $0 [-i can0]"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

if ip link show "$IFACE" &>/dev/null; then
    sudo ip link set "$IFACE" down 2>/dev/null || true
    echo -e "${GREEN}$IFACE set down.${NC}"
else
    echo -e "${YELLOW}$IFACE not present.${NC}"
fi

# Kill any slcand that owns this iface (serial adapters). -c on slcand closes the
# tty, but kill it explicitly so a re-attach with a new bitrate is clean.
if pgrep -af "slcand.* $IFACE\$" &>/dev/null || pgrep -af "slcand.*$IFACE" &>/dev/null; then
    sudo pkill -f "slcand.*$IFACE" 2>/dev/null || true
    echo -e "${GREEN}stopped slcand for $IFACE.${NC}"
fi

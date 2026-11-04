#!/usr/bin/env bash
# setup_arms_lane.sh — create the persistent NetworkManager profile for the ARMS
# Teensy ethernet lane, mirroring the base's `teensy-base` profile.
#
# The arms Teensy 4.1 sits on one of Thor's QSFP28 break-out 10 GbE lanes
# (mgbe0_0..mgbe3_0; the base owns mgbe3_0). Reserved plan (# Hardware/Thor.md): arms/body Teensys take 192.168.125-127.x, one /24 per lane.
#   host (Thor):  192.168.125.2/24   on the arms lane
#   arms Teensy:  192.168.125.10:5800 (UDP, docs/ARMLINK.md)
#
# Usage:  sudo ./setup_arms_lane.sh [iface]     # default iface: mgbe2_0
#
# `ipv4.never-default` keeps this lane from ever stealing the wifi/Tailscale
# default route; `autoconnect` makes it self-assert whenever the link carries.

set -euo pipefail

IFACE="${1:-${MABEL_ARMS_IFACE:-mgbe2_0}}"
CON="${MABEL_ARMS_NM_CON:-teensy-arms}"
HOST_IP="${MABEL_ARMS_HOST_IP:-192.168.125.2/24}"

if nmcli -t -f NAME con show | grep -qx "$CON"; then
    echo "[arms-lane] profile '$CON' already exists — updating iface/IP."
    nmcli con modify "$CON" connection.interface-name "$IFACE" \
        ipv4.method manual ipv4.addresses "$HOST_IP" ipv4.never-default yes \
        connection.autoconnect yes
else
    nmcli con add type ethernet con-name "$CON" ifname "$IFACE" \
        ipv4.method manual ipv4.addresses "$HOST_IP" ipv4.never-default yes \
        connection.autoconnect yes
fi
nmcli con up "$CON" || echo "[arms-lane] link not up yet (no carrier?) — profile" \
    "will auto-assert when the Teensy lane carries. Check: ip -br addr show $IFACE"
echo "[arms-lane] done: $CON on $IFACE = $HOST_IP (Teensy expected at 192.168.125.10:5800)"

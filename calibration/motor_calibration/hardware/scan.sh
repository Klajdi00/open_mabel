#!/usr/bin/env bash
# scan.sh — non-energizing scan for DaMiao motors on a SocketCAN bus.
#
# Sends the MIT "disable" frame (FF..FD) to each CAN id; a motor that is powered
# replies with ONE feedback frame while staying de-energized. Decodes the reply
# to show which id answered and its fault state. Moves nothing.
#
#   ./scan.sh            # scan ids 1..16 on can0
#   ./scan.sh 1 8        # scan ids 1..8
#   CHANNEL=can1 ./scan.sh
set -euo pipefail
CHANNEL="${CHANNEL:-can0}"
LO="${1:-1}"; HI="${2:-16}"

if ! ip link show "$CHANNEL" | grep -q "state UP"; then
  echo "bringing up $CHANNEL @ 1Mbps (needs sudo)..."
  sudo ip link set "$CHANNEL" up type can bitrate 1000000
fi

LOG=$(mktemp)
timeout $(echo "($HI-$LO+1)*0.16+1" | bc) candump -t z "$CHANNEL" > "$LOG" 2>&1 &
sleep 0.3
for id in $(seq "$LO" "$HI"); do
  cansend "$CHANNEL" "$(printf '%03X' "$id")#FFFFFFFFFFFFFFFD"; sleep 0.15
done
sleep 0.3; wait || true

echo "=== motors found on $CHANNEL ==="
FOUND=0
# unique feedback lines; decode byte0 (field 5): low nibble=id, high nibble=state
while read -r line; do
  d0=$(echo "$line" | awk '{print $5}')
  [ -z "$d0" ] && continue
  v=$((16#$d0)); id=$((v & 15)); err=$((v >> 4))
  case $err in 0) st="disabled";; 1) st="enabled";; *) st=$(printf 'FAULT 0x%X' "$err");; esac
  raw=$(echo "$line" | awk '{print $5,$6,$7,$8,$9,$10,$11,$12}')
  printf "  motor ID %d  (%s)  raw: %s\n" "$id" "$st" "$raw"; FOUND=1
done < <(grep -vi "FF FF FF FF FF FF FF FD" "$LOG" | sort -u)
[ "$FOUND" = 1 ] || echo "  (none — check 24V power, CAN_H/L wiring, ids)"
rm -f "$LOG"

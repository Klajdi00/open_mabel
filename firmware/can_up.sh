#!/usr/bin/env bash
# can_up.sh — auto-detect a USB-CAN adapter and bring up a SocketCAN interface.
#
# Handles BOTH adapter families used on MABEL:
#   * native SocketCAN (gs_usb: candleLight / OpenMoko / CANable in "candleLight"
#     firmware)  -> a canN netdev appears by itself; we just configure bitrate.
#   * serial-line  (slcan: CANable in "slcan" fw, CH340/CP210x-based dongles)
#     -> a /dev/ttyACM* or /dev/ttyUSB* appears; we attach it with slcand first.
#
# Usage:
#   ./can_up.sh                 # bring up can0 at 1 Mbit (DaMiao default)
#   ./can_up.sh -i can0 -b 1000000
#   ./can_up.sh --ping          # also ping motor IDs 1..15 after bring-up
#   CAN_BITRATE=1000000 ./can_up.sh
#
# Exit status: 0 if the interface is UP, non-zero otherwise.
set -euo pipefail

IFACE="${CAN_IFACE:-can0}"
BITRATE="${CAN_BITRATE:-1000000}"   # DaMiao bus is a fixed 1 Mbit
DO_PING=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}$*${NC}"; }
ok()    { echo -e "${GREEN}$*${NC}"; }
warn()  { echo -e "${YELLOW}$*${NC}"; }
err()   { echo -e "${RED}$*${NC}" >&2; }

# slcan bitrate code: -s8 = 1 Mbit, -s6 = 500k, -s5 = 250k, -s4 = 125k
slcan_speed_flag() {
    case "$1" in
        1000000) echo "-s8" ;;
         500000) echo "-s6" ;;
         250000) echo "-s5" ;;
         125000) echo "-s4" ;;
              *) echo "" ;;   # unknown -> let slcand use adapter default
    esac
}

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; /^set -euo/d'
    exit "${1:-0}"
}

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--iface)   IFACE="$2"; shift 2 ;;
        -b|--bitrate) BITRATE="$2"; shift 2 ;;
        -p|--ping)    DO_PING=1; shift ;;
        -h|--help)    usage 0 ;;
        *) err "unknown arg: $1"; usage 1 ;;
    esac
done

for t in ip; do
    command -v "$t" &>/dev/null || { err "'$t' not found (sudo apt install iproute2)"; exit 1; }
done

echo -e "${BOLD}== can_up: target ${IFACE} @ ${BITRATE} bps ==${NC}"

# ── helpers (defined before any use) ──────────────────────────────────────────
have_netdev() { ip link show "$IFACE" &>/dev/null; }

bring_up_native() {
    info "Configuring native SocketCAN device $IFACE..."
    sudo ip link set "$IFACE" down 2>/dev/null || true   # clear bus-off
    sleep 0.3
    sudo ip link set "$IFACE" up type can bitrate "$BITRATE"
    sleep 0.5
}

# ── 2. otherwise look for a serial adapter and attach slcand (slcan path) ─────
find_serial_port() {
    local p
    for p in /dev/ttyACM* /dev/ttyUSB*; do
        [[ -e "$p" ]] && { echo "$p"; return 0; }
    done
    return 1
}

bring_up_slcan() {
    local port="$1" flag
    command -v slcand &>/dev/null || { err "'slcand' not found (sudo apt install can-utils)"; return 1; }
    flag="$(slcan_speed_flag "$BITRATE")"
    info "Attaching serial adapter $port -> $IFACE via slcand ($flag)..."
    # -o open, -c close-on-exit, -f read status flags; name the netdev $IFACE
    sudo slcand -o -c -f $flag "$port" "$IFACE"
    sleep 0.7
    have_netdev || { err "slcand did not create $IFACE"; return 1; }
    sudo ip link set "$IFACE" up
    sleep 0.3
}

# ── ping the DaMiao motors (read-register broadcast to IDs 1..15) ─────────────
ping_motors() {
    command -v cansend &>/dev/null && command -v candump &>/dev/null || {
        warn "can-utils (cansend/candump) missing — skipping motor ping."; return 0; }
    echo -e "\n${BOLD}== pinging DaMiao motor IDs 1..15 on $IFACE ==${NC}"
    local tmp; tmp=$(mktemp)
    candump "$IFACE" -T 1500 > "$tmp" 2>&1 &
    local dpid=$!
    sleep 0.3
    for id in $(seq 1 15); do
        printf -v lo "%02X" "$id"
        cansend "$IFACE" "7FF#${lo}003300" 2>/dev/null || true
        sleep 0.05
    done
    sleep 0.4
    wait "$dpid" 2>/dev/null || true

    # DaMiao read-reply: CAN id 000, CMD byte (field 5) == 33; D[0]=ID|ERR<<4
    local found=0
    while read -r line; do
        f=($line)
        if [[ ${#f[@]} -ge 6 && "${f[1]}" == "000" && "${f[5]}" == "33" ]]; then
            mid=$(( 0x${f[3]} & 0x0F )); errn=$(( (0x${f[3]} >> 4) & 0x0F ))
            if [[ $errn -eq 0 ]]; then
                ok "  Motor ID $mid responding (no fault)."
            else
                warn "  Motor ID $mid responding but ERR=$errn (see datasheet fault table)."
            fi
            found=$((found+1))
        fi
    done < "$tmp"
    if [[ $found -eq 0 ]]; then
        warn "  No motor replies. Raw frames captured:"
        if [[ -s "$tmp" ]]; then sed 's/^/    /' "$tmp"; else echo "    (none — check 24 V power & CAN_H/CAN_L wiring)"; fi
    fi
    rm -f "$tmp"
}

# ── 0. already up? just (optionally) ping and exit ────────────────────────────
if have_netdev && [[ "$(ip -br link show "$IFACE" | awk '{print $2}')" == "UP" ]]; then
    ok "$IFACE is already UP."
    ip -details link show "$IFACE" | sed 's/^/  /'
    [[ $DO_PING -eq 1 ]] && ping_motors
    exit 0
fi

# ── decide which path applies ─────────────────────────────────────────────────
if have_netdev; then
    bring_up_native
else
    info "$IFACE not present yet — looking for a serial USB-CAN adapter..."
    if port="$(find_serial_port)"; then
        ok "Found serial adapter: $port"
        bring_up_slcan "$port"
    else
        err "No CAN interface and no /dev/ttyACM*|ttyUSB* serial adapter found."
        err "Plug in the USB-CAN adapter (and its 24 V motor supply), then:"
        err "  watch the kernel:   sudo dmesg -w"
        err "  list usb devices:   lsusb        (count should rise from baseline)"
        err "  list netdevs:       ip -br link show"
        exit 1
    fi
fi

# ── verify ────────────────────────────────────────────────────────────────────
if [[ "$(ip -br link show "$IFACE" 2>/dev/null | awk '{print $2}')" == "UP" ]]; then
    ok "$IFACE is UP."
    ip -details link show "$IFACE" | sed 's/^/  /'
else
    err "Failed to bring $IFACE UP."
    exit 1
fi

[[ $DO_PING -eq 1 ]] && ping_motors
exit 0

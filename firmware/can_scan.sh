#!/usr/bin/env bash
# Scans all CAN interfaces for DaMiao motors.
# Brings up any down interfaces, pings motor IDs 1-15 on each bus,
# and reports which motors responded.

BITRATE=${CAN_BITRATE:-1000000}
PING_DELAY=0.06   # seconds between pings
LISTEN_MS=5000    # candump timeout in ms

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

require() {
    command -v "$1" &>/dev/null || { echo -e "${RED}Error: '$1' not found. Run: sudo apt install can-utils${NC}"; exit 1; }
}

require candump
require cansend
require ip

# ── Find all CAN interfaces ───────────────────────────────────────────────────
mapfile -t CAN_IFACES < <(ip link show | awk -F': ' '/^[0-9]+: can[0-9]+/{print $2}')

if [[ ${#CAN_IFACES[@]} -eq 0 ]]; then
    echo -e "${RED}No CAN interfaces found. Check that your USB-CAN adapter is plugged in.${NC}"
    exit 1
fi

echo -e "${BOLD}Found ${#CAN_IFACES[@]} CAN interface(s): ${CAN_IFACES[*]}${NC}\n"

# ── Bring up any interfaces that are not UP ───────────────────────────────────
declare -a ACTIVE_IFACES=()

for iface in "${CAN_IFACES[@]}"; do
    state=$(ip link show "$iface" | grep -o 'state [A-Z]*' | awk '{print $2}')

    if [[ "$state" == "UP" ]]; then
        echo -e "${GREEN}$iface already UP.${NC}"
        ACTIVE_IFACES+=("$iface")
    else
        echo -e "${YELLOW}Bringing up $iface at ${BITRATE} bps...${NC}"
        # Cycle down first to clear any bus-off state
        sudo ip link set "$iface" down 2>/dev/null || true
        sleep 0.3
        if sudo ip link set "$iface" up type can bitrate "$BITRATE"; then
            sleep 1.0   # give motor and adapter time to settle
            echo -e "${GREEN}$iface is ready.${NC}"
            ACTIVE_IFACES+=("$iface")
        else
            echo -e "${RED}Failed to bring up $iface — skipping.${NC}"
        fi
    fi
done

echo ""

if [[ ${#ACTIVE_IFACES[@]} -eq 0 ]]; then
    echo -e "${RED}No interfaces could be brought up.${NC}"
    exit 1
fi

# ── Scan each interface for motors ───────────────────────────────────────────
total_motors=0

for iface in "${ACTIVE_IFACES[@]}"; do
    echo -e "${CYAN}${BOLD}[$iface] Scanning motor IDs 1–15...${NC}"

    tmp=$(mktemp)
    candump "$iface" -T "$LISTEN_MS" > "$tmp" 2>&1 &
    DUMP_PID=$!
    sleep 0.3   # let candump start listening before first ping

    for id in $(seq 1 15); do
        lo=$(printf "%02X" "$id")
        cansend "$iface" "7FF#${lo}003300" 2>/dev/null || true
        sleep "$PING_DELAY"
    done

    sleep 0.5
    wait "$DUMP_PID" 2>/dev/null || true

    # candump line format: "  can0  000   [8]  ID_L ID_H CMD ..."
    # fields (0-indexed bash array): [0]=can0 [1]=000 [2]=[8] [3]=ID_L [4]=ID_H [5]=CMD
    # DaMiao read response: CAN ID 000, CMD byte (fields[5]) == 33
    declare -a motor_ids=()
    while read -r line; do
        fields=($line)
        if [[ ${#fields[@]} -ge 6 \
           && "${fields[1]}" == "000" \
           && "${fields[5]}" == "33" ]]; then
            motor_id=$(printf "%d" "0x${fields[3]}")
            motor_ids+=("$motor_id")
        fi
    done < "$tmp"

    # Debug: show raw capture if nothing found
    if [[ ${#motor_ids[@]} -eq 0 ]]; then
        echo -e "  ${YELLOW}No motors found on $iface.${NC}"
        if [[ -s "$tmp" ]]; then
            echo -e "  Raw CAN frames captured:"
            sed 's/^/    /' "$tmp"
        else
            echo -e "  (No frames received at all — check wiring and power)"
        fi
    else
        echo -e "  ${GREEN}${#motor_ids[@]} motor(s) detected:${NC}"
        for mid in "${motor_ids[@]}"; do
            echo -e "    ${GREEN}Motor ID $mid${NC}"
        done
        total_motors=$(( total_motors + ${#motor_ids[@]} ))
    fi

    rm -f "$tmp"
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}────────────────────────────────────────${NC}"
echo -e "${BOLD}Total motors found across all buses: $total_motors${NC}"
echo -e "${BOLD}────────────────────────────────────────${NC}"

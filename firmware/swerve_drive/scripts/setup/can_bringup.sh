#!/usr/bin/env bash
# MABEL swerve drive — low-level CAN bus bringup & diagnostics.
#
# REV SPARK Flex / SPARK MAX requirements:
#   - 29-bit extended IDs       (handled in software, not here)
#   - 1 Mbit/s bitrate          (set by this script)
#   - txqueuelen >= 1000        (set by this script, prevents drops
#                                during heartbeat + setpoint floods)
#
# Subcommands:
#   up        Bring the interface up at the configured bitrate (default)
#   down      Bring the interface down
#   restart   Cycle the interface (down -> up); use after bus-off / wiring fix
#   status    Print link state, bitrate, error counters, queue stats
#   scan      List every CAN-capable netdev currently visible to the kernel
#   doctor    Diagnose why an interface is missing (USB, modules, dmesg)
#   install-autoup
#             Install a udev rule that auto-configures the OpenMoko/gs_usb
#             adapter to 1 Mbit/s the moment it's plugged in. Survives reboots.
#
# Environment overrides:
#   CAN_IFACE     Interface name      (default: can0)
#   CAN_BITRATE   Bus bitrate in bps  (default: 1000000)
#   CAN_TXQLEN    TX queue length     (default: 1000)
#
# Examples:
#   ./setup_can.sh                    # bring can0 up at 1 Mbit/s
#   ./setup_can.sh restart            # cycle (recover from bus-off)
#   ./setup_can.sh status             # show link + error stats
#   ./setup_can.sh doctor             # why is can0 missing?
#   CAN_BITRATE=500000 ./setup_can.sh # non-default bitrate

set -u   # no -e: we want to print helpful messages on failure ourselves

# ─── Colors ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
    RED=; GREEN=; YELLOW=; CYAN=; BOLD=; DIM=; NC=
fi

# ─── Config ──────────────────────────────────────────────────────────────────
IFACE="${CAN_IFACE:-can0}"
BITRATE="${CAN_BITRATE:-1000000}"
TXQLEN="${CAN_TXQLEN:-1000}"

# OpenMoko / Geschwister Schneider USB-CAN adapter (gs_usb driver)
# (This matches the ID found in your lsusb logs for the Canable 2)
GS_USB_VID="1d50"
GS_USB_PID="606f"

# ─── Helpers ─────────────────────────────────────────────────────────────────
info()  { echo -e "${CYAN}$*${NC}"; }
ok()    { echo -e "${GREEN}$*${NC}"; }
warn()  { echo -e "${YELLOW}$*${NC}"; }
err()   { echo -e "${RED}$*${NC}" >&2; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        err "Missing tool: $1. Install with:  sudo apt install $2"
        exit 1
    }
}

iface_exists() { ip link show "$1" >/dev/null 2>&1; }

iface_state() {
    # Returns: UP / DOWN / MISSING
    if ! iface_exists "$1"; then echo "MISSING"; return; fi
    ip link show "$1" | grep -qE '\bstate UP\b|\bUP[,>]' && echo "UP" || echo "DOWN"
}

# ─── Subcommands ─────────────────────────────────────────────────────────────

cmd_scan() {
    info "${BOLD}CAN-capable interfaces visible to the kernel:${NC}"
    local found
    found=$(ip -br link show type can 2>/dev/null)
    if [[ -z "$found" ]]; then
        warn "  (none — kernel does not see any CAN netdev)"
        echo
        info "Quick checks:"
        echo "    1. USB adapter plugged in?         lsusb | grep -i can"
        echo "    2. gs_usb driver loaded?           lsmod | grep gs_usb"
        echo "    3. Run the doctor for full diag:   $0 doctor"
        return 1
    fi
    echo "$found" | sed 's/^/  /'
    return 0
}

cmd_status() {
    if ! iface_exists "$IFACE"; then
        err "Interface ${BOLD}$IFACE${NC}${RED} does not exist."
        echo
        cmd_scan
        return 1
    fi
    info "${BOLD}== $IFACE detailed state ==${NC}"
    ip -details -statistics link show "$IFACE"
    echo
    # Highlight critical fields
    local state bitrate canstate
    state=$(iface_state "$IFACE")
    bitrate=$(ip -details link show "$IFACE" | awk '/bitrate/ {for(i=1;i<=NF;i++) if($i=="bitrate") print $(i+1); exit}')
    canstate=$(ip -details link show "$IFACE" | awk '/can state/ {print $3; exit}')

    echo -e "  Summary: link=${BOLD}$state${NC}  bitrate=${BOLD}${bitrate:-unknown}${NC}  bus=${BOLD}${canstate:-unknown}${NC}"
    case "$canstate" in
        ERROR-ACTIVE)  ok   "  Bus is healthy (ERROR-ACTIVE)." ;;
        ERROR-WARNING) warn "  Bus is degraded (ERROR-WARNING) — check wiring/termination." ;;
        ERROR-PASSIVE) warn "  Bus is failing (ERROR-PASSIVE) — many TX errors." ;;
        BUS-OFF)       err  "  Bus is OFF — run '$0 restart' to recover." ;;
        STOPPED)       warn "  Interface is administratively DOWN. Run '$0 up'." ;;
    esac
}

cmd_down() {
    if ! iface_exists "$IFACE"; then
        warn "$IFACE does not exist; nothing to bring down."
        return 0
    fi
    info "Bringing $IFACE down..."
    sudo ip link set "$IFACE" down || { err "ip link set down failed"; return 1; }
    ok "$IFACE is DOWN."
}

cmd_up() {
    if ! iface_exists "$IFACE"; then
        err "Interface ${BOLD}$IFACE${NC}${RED} not found."
        echo
        warn "Possible causes:"
        echo "  • USB-CAN adapter not plugged in"
        echo "  • Adapter just plugged in — kernel hasn't enumerated yet (wait 1–2 s)"
        echo "  • gs_usb / can_dev kernel modules not loaded"
        echo
        info "Run:  $0 doctor"
        return 1
    fi

    # If already up, cycle it cleanly so bitrate/txqueuelen get re-applied.
    if [[ "$(iface_state "$IFACE")" == "UP" ]]; then
        info "$IFACE is already UP — cycling to re-apply config..."
        sudo ip link set "$IFACE" down 2>/dev/null || true
        sleep 0.2
    fi

    info "Configuring $IFACE at ${BOLD}$BITRATE${NC}${CYAN} bps, txqueuelen=$TXQLEN..."

    # Set bitrate while down (kernel requires this).
    if ! sudo ip link set "$IFACE" type can bitrate "$BITRATE"; then
        err "Failed to set bitrate. Possible reasons:"
        echo "  • $IFACE is busy (another process holds it)"
        echo "  • Bitrate $BITRATE not supported by this adapter"
        return 1
    fi

    if ! sudo ip link set "$IFACE" up; then
        err "Failed to bring $IFACE up."
        return 1
    fi

    sudo ip link set "$IFACE" txqueuelen "$TXQLEN" || warn "Failed to set txqueuelen (non-fatal)."

    # Give the controller a moment to settle and verify.
    sleep 0.3
    if [[ "$(iface_state "$IFACE")" != "UP" ]]; then
        err "$IFACE did not stay UP after bringup (likely bus problem)."
        cmd_status
        return 1
    fi

    ok "$IFACE is UP at $BITRATE bps."
    echo
    cmd_status
}

cmd_restart() {
    cmd_down
    sleep 0.3
    cmd_up
}

cmd_doctor() {
    info "${BOLD}== CAN doctor — diagnosing $IFACE ==${NC}"

    # 1. Netdev presence
    echo
    info "1. Kernel netdev for $IFACE:"
    if iface_exists "$IFACE"; then
        ok "   ✓ $IFACE exists (state: $(iface_state "$IFACE"))"
    else
        err "   ✗ $IFACE does not exist"
    fi

    # 2. Other CAN devices
    echo
    info "2. All CAN-capable netdevs:"
    cmd_scan 2>/dev/null | grep -vE '^\s*$' | sed 's/^/   /' || true

    # 3. Required kernel modules
    echo
    info "3. Kernel modules:"
    for mod in can can_raw can_dev gs_usb; do
        if lsmod | awk '{print $1}' | grep -qx "$mod"; then
            ok "   ✓ $mod loaded"
        else
            warn "   ✗ $mod NOT loaded  (try: sudo modprobe $mod)"
        fi
    done

    # 4. USB adapter presence
    echo
    info "4. USB-CAN adapter (OpenMoko / gs_usb):"
    if lsusb | grep -qi "${GS_USB_VID}:${GS_USB_PID}\|geschwister\|openmoko"; then
        ok   "   ✓ Adapter enumerated on USB:"
        lsusb | grep -iE "${GS_USB_VID}:${GS_USB_PID}|geschwister|openmoko" | sed 's/^/      /'
    else
        err  "   ✗ No OpenMoko / gs_usb adapter found in lsusb"
        echo "      Try unplugging + replugging the USB cable."
    fi

    # 5. Recent dmesg
    echo
    info "5. Recent kernel messages (CAN/USB, last 5 min):"
    if sudo -n true 2>/dev/null; then
        local msgs
        msgs=$(sudo dmesg --since '5 min ago' 2>/dev/null \
               | grep -Ei 'can|gs_usb|usb [0-9]+-' | tail -15)
        if [[ -n "$msgs" ]]; then
            echo "$msgs" | sed 's/^/      /'
        else
            echo "      (nothing recent — adapter not plugged in / replugged in last 5 min)"
        fi
    else
        warn "   (sudo not cached — run 'sudo dmesg | tail' manually)"
    fi

    # 6. Suggested next step
    echo
    info "${BOLD}Suggested fix order:${NC}"
    echo "   a. If adapter is unplugged → plug it in, wait 2 s, retry"
    echo "   b. If modules missing      → sudo modprobe gs_usb can_raw"
    echo "   c. If adapter is dead      → power-cycle USB / try another port"
    echo "   d. If $IFACE shows up      → $0 up"
}

cmd_install_autoup() {
    local rule="/etc/udev/rules.d/90-mabel-can.rules"
    info "Installing udev rule that auto-configures the gs_usb adapter at plug-in..."

    sudo tee "$rule" >/dev/null <<EOF
# MABEL swerve drive — auto-configure OpenMoko / gs_usb CAN adapter
# Sets bitrate to ${BITRATE} bps and brings the interface up the moment
# the adapter is plugged in. Survives reboots.
ACTION=="add", SUBSYSTEM=="net", KERNELS=="*", \
    ATTRS{idVendor}=="${GS_USB_VID}", ATTRS{idProduct}=="${GS_USB_PID}", \
    RUN+="/sbin/ip link set %k type can bitrate ${BITRATE}", \
    RUN+="/sbin/ip link set %k up", \
    RUN+="/sbin/ip link set %k txqueuelen ${TXQLEN}"
EOF

    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=net
    ok "Installed:  $rule"
    info "Replug the adapter to test. Verify with:  $0 status"
}

# ─── Main ────────────────────────────────────────────────────────────────────
require_cmd ip iproute2

case "${1:-up}" in
    up)              shift; [[ $# -gt 0 ]] && IFACE="$1"; cmd_up ;;
    down)            shift; [[ $# -gt 0 ]] && IFACE="$1"; cmd_down ;;
    restart|cycle)   shift; [[ $# -gt 0 ]] && IFACE="$1"; cmd_restart ;;
    status|info)     shift; [[ $# -gt 0 ]] && IFACE="$1"; cmd_status ;;
    scan|list)       cmd_scan ;;
    doctor|diag)     cmd_doctor ;;
    install-autoup)  cmd_install_autoup ;;
    -h|--help|help)
        # Parse lines 2 through 32 of this script to dynamically print the help block
        sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        err "Unknown subcommand: $1"
        echo "Run '$0 help' for usage."
        exit 2
        ;;
esac

exit 0
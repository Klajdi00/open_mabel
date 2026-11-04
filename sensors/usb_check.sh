#!/usr/bin/env bash
# usb_check.sh -- is each depth camera on a link the SDK can actually use?
#
# The Intel D435i streams over USB-2 (it just drops bandwidth), but Stereolabs
# ZED cameras expose their video interface ONLY over a USB-3 SuperSpeed link --
# on USB-2 the ZED enumerates HID-only (its IMU) and NO /dev/video appears, so
# the ZED SDK cannot open it. This tells you, per camera, the negotiated link
# speed and whether a usable video node exists, then prints a GREEN/RED verdict.
#
#   bash usb_check.sh          # one-shot
#   watch -c -n1 bash usb_check.sh   # live, while you try different ports
set -uo pipefail

bold=$'\e[1m'; dim=$'\e[2m'; red=$'\e[31m'; grn=$'\e[32m'; ylw=$'\e[33m'
cyn=$'\e[36m'; rst=$'\e[0m'

# vendor:product -> label
declare -A CAMS=(
  ["8086"]="Intel RealSense D435i"
  ["2b03"]="Stereolabs ZED-M"
)

speed_label() { # Mbps -> name
  case "$1" in
    12)    echo "USB 1.1 (12M)"   ;;
    480)   echo "USB 2.0 (480M)"  ;;
    5000)  echo "USB 3.0 (5G SuperSpeed)"  ;;
    10000) echo "USB 3.1 (10G SuperSpeed)" ;;
    20000) echo "USB 3.2 (20G SuperSpeed)" ;;
    *)     echo "${1}M" ;;
  esac
}

printf '%s\n' "${bold}${cyn}┌─ USB camera link check ─────────────────────────────────────┐${rst}"

for vid in "${!CAMS[@]}"; do
  label="${CAMS[$vid]}"
  # find the sysfs node(s) whose idVendor matches, then the highest-speed one
  dev="" ; spd=0
  for d in /sys/bus/usb/devices/*; do
    [[ -r "$d/idVendor" ]] || continue
    [[ "$(cat "$d/idVendor" 2>/dev/null)" == "$vid" ]] || continue
    s=$(cat "$d/speed" 2>/dev/null || echo 0); s=${s%.*}
    if [[ -z "$dev" || "$s" -ge "$spd" ]]; then spd="$s"; dev="$d"; fi
  done

  printf '%s│%s %s%-22s%s ' "$cyn" "$rst" "$bold" "$label" "$rst"
  if [[ -z "$dev" ]]; then
    printf '%s%s%s\n' "$dim" "not connected" "$rst"
    continue
  fi

  # count video nodes belonging to this vendor
  nvideo=0
  for v in /dev/video*; do
    [[ -e "$v" ]] || continue
    vv=$(udevadm info --query=property --name="$v" 2>/dev/null | sed -n 's/^ID_VENDOR_ID=//p')
    [[ "$vv" == "$vid" ]] && nvideo=$((nvideo+1))
  done

  link="$(speed_label "$spd")"
  if (( spd >= 5000 )); then col="$grn"; else col="$ylw"; fi
  printf '%slink=%s%s  %svideo nodes=%d%s\n' "$col" "$link" "$rst" "$dim" "$nvideo" "$rst"

  # per-camera verdict
  printf '%s│%s   ' "$cyn" "$rst"
  if [[ "$vid" == "2b03" ]]; then
    if (( spd >= 5000 )) && (( nvideo > 0 )); then
      printf '%s✓ READY%s — SuperSpeed + video present; the ZED SDK can open it.\n' "$grn" "$rst"
    elif (( spd >= 5000 )); then
      printf '%s… SuperSpeed but no /dev/video yet%s — re-plug or wait a beat.\n' "$ylw" "$rst"
    else
      printf '%s✗ USB-2 / HID-only%s — move to a SuperSpeed port (5G+) w/ the SS cable.\n' "$red" "$rst"
    fi
  else
    if (( nvideo == 0 )); then
      printf '%s✗ no video node%s — replug / check librealsense.\n' "$red" "$rst"
    elif (( spd >= 5000 )); then
      printf '%s✓ READY%s — SuperSpeed; full framerate.\n' "$grn" "$rst"
    else
      printf '%s! USB-2: enumerates but streams at ~0 fps here%s — move to a '\
'SuperSpeed port for full rate.\n' "$ylw" "$rst"
    fi
  fi
done

printf '%s├─ SuperSpeed-capable buses on this machine ──────────────────┤%s\n' "$cyn" "$rst"
while read -r line; do
  printf '%s│%s %s%s%s\n' "$cyn" "$rst" "$dim" "$line" "$rst"
done < <(lsusb -t | grep -E '5000M|10000M|20000M' || echo "(none found — no SuperSpeed ports?!)")
printf '%s└─────────────────────────────────────────────────────────────┘%s\n' "$cyn" "$rst"

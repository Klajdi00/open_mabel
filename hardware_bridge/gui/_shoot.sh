#!/usr/bin/env bash
# Wrapper: source the ROS env, then run the screenshot harness on a nested Xephyr.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BR="$(cd "$HERE/.." && pwd)"
export OUT="${OUT:-$HERE/_shots}"
mkdir -p "$OUT"
export DISPLAY="${DISPLAY:-:0}"
source "$BR/../ros/scripts/mabel_env.sh" >/dev/null 2>&1 || true
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0
cd "$BR"
python3 -u gui/_shoot.py

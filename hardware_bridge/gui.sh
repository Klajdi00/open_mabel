#!/usr/bin/env bash
# gui.sh — launch the MABEL Control Center (the ultimate GUI on top of the bridge).
#
# One window for the whole hardware stack — base, arms, body, hands, lift — talking ONLY
# to the hardware bridge over DDS (never to a device directly). Bring the bridge up first
# (run_bridge.sh or the systemd unit); this is just a client.
#
#   ./gui.sh            # launch the control center
#
# Sources the SAME ROS env as the bridge so they share the DDS graph. Thin wrapper over
# gui/control_center.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/gui/control_center.sh" "$@"

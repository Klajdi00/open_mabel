#!/usr/bin/env bash
# view.sh — open a LIVE side-by-side video of the ZED-M's left + right eyes in a
# browser (MJPEG plays as continuous video; no extra deps, no device contention).
#
#   bash view.sh                 # view on this machine (localhost:8080)
#   bash view.sh 192.168.1.42    # view a remote thor by IP
#   bash view.sh 100.64.0.2   # via Tailscale
#   PORT=8080 bash view.sh HOST
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-127.0.0.1}"
PORT="${PORT:-8080}"
URL="file://$HERE/live.html?host=$HOST&port=$PORT"

echo "live left+right  →  $URL"
echo "(or open directly in any browser/phone:  http://$HOST:$PORT/camera/head_left/stream.mjpg )"

if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 &
elif command -v firefox  >/dev/null 2>&1; then firefox  "$URL" >/dev/null 2>&1 &
else echo "no browser launcher found — open the URL above manually."; fi

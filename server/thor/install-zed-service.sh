#!/usr/bin/env bash
# install-zed-service.sh — auto-start the ZED-M camera stream on thor at boot.
#
# Creates a systemd service `mabel-zed-cam.service` that runs
# `thor/zed_stream.sh --watch`: it WAITS for the ZED-M to be connected (USB-3),
# then streams its stereo footage to the network on the SAME ports every client
# already subscribes to —
#     http://<thor>:8080/camera/head_left|head_right|main/stream.mjpg   (MJPEG)
#     udp://<thor>:9091                                                 (headset)
#     http://<thor>:8080/                                               (status index)
# so Vision Pro / iPhone / web clients just connect, no app change.
#
# - enabled (start at boot), Restart=always (comes back if it crashes / on replug),
#   ordered After=network-online so the socket binds cleanly.
# - If the ZED isn't plugged in, the service sits idle waiting; plug it in and it
#   streams within ~1 s. Unplug + replug → it reconnects.
#
# Idempotent: safe to re-run.
#
# Usage:   bash thor/install-zed-service.sh            # install + enable + start
#          bash thor/install-zed-service.sh --no-start # install + enable only
set -euo pipefail

USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(eval echo "~$USER_NAME")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "$HERE/.." && pwd)"
CONDA_PY="$HOME_DIR/miniconda3/envs/base-module-env/bin/python"

START=1
[[ "${1:-}" == "--no-start" ]] && START=0

echo "▸ user=$USER_NAME  server=$SERVER_DIR"
[[ -x "$CONDA_PY" ]]                         || echo "  ⚠ conda python not found at $CONDA_PY"
[[ -f "$SERVER_DIR/real_camera_server.py" ]] || echo "  ⚠ real_camera_server.py not found in $SERVER_DIR"
[[ -f "$HERE/zed_stream.sh" ]]               || { echo "  ✗ zed_stream.sh missing"; exit 1; }

# A running manual instance would hold :8080 — stop it so the service can bind.
if pgrep -f "real_camera_server.py" >/dev/null 2>&1; then
  echo "▸ stopping a manually-launched real_camera_server (frees :8080)"
  pkill -f "real_camera_server.py" || true
  sleep 1
fi

# ── service ──────────────────────────────────────────────────────────────────
sudo tee /etc/systemd/system/mabel-zed-cam.service >/dev/null <<EOF
[Unit]
Description=MABEL ZED-M stereo camera stream (MJPEG :8080 + UDP :9091)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$SERVER_DIR
# --watch: wait for the ZED-M /dev/video* to appear, then stream. (Default mode
#  2560x720 → 1280x720/eye, native, 30fps, raw split — see thor/zed_stream.sh.)
ExecStart=/usr/bin/env bash $SERVER_DIR/thor/zed_stream.sh --watch
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
echo "▸ wrote mabel-zed-cam.service"

# ── open the firewall so clients on the LAN/Tailscale can subscribe ──────────
if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -qi "Status: active"; then
  sudo ufw allow 8080/tcp >/dev/null 2>&1 && echo "▸ ufw: allowed 8080/tcp (MJPEG)"
  sudo ufw allow 9091/udp >/dev/null 2>&1 && echo "▸ ufw: allowed 9091/udp (video tee)"
fi

# ── enable (+ start) ─────────────────────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl enable mabel-zed-cam.service >/dev/null
echo "▸ enabled on boot"
if [[ $START -eq 1 ]]; then
  sudo systemctl restart mabel-zed-cam.service
  sleep 2
  echo "▸ started"
fi

echo
echo "Status:"
systemctl is-enabled mabel-zed-cam | sed 's/^/  enabled: /'
systemctl is-active  mabel-zed-cam | sed 's/^/  active:  /'
echo
echo "Manage:   systemctl status mabel-zed-cam   ·   journalctl -u mabel-zed-cam -f"
echo "          sudo systemctl restart|stop mabel-zed-cam"
echo "View:     bash $SERVER_DIR/../sensors/view.sh        (left+right live, on this box)"
echo "          http://<thor-ip>:8080/                     (status + streams, any client)"

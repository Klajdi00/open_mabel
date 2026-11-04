#!/usr/bin/env bash
# Install the Control Center as a PERMANENT user service (survives reboots,
# no sudo needed). Reach it at http://<robot-host>:8094 on the LAN or over
# Tailscale.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/mabel-control-center.service <<EOF
[Unit]
Description=MABEL Control Center studio (:8094 — viewer + teleop client + server monitor)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=$REPO/control_center/run.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now mabel-control-center.service
loginctl enable-linger "$USER" 2>/dev/null || true
echo "installed: mabel-control-center (:8094)."

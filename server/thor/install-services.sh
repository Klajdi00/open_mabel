#!/usr/bin/env bash
# install-services.sh — install/refresh the MABEL real-robot auto-start on thor
# (the Jetson). Run it ON thor; it creates systemd services that start the
# teleop server automatically every time the Jetson powers on:
#
#   mabel-bridge.service  — the teleop WebSocket bridge on :9090 (drives the real
#                           MAXSwerve base; talks to the Teensy on the robot net).
#   mabel-diag.service    — a one-shot boot-time connectivity report.
#
# The bridge is `enabled` (start at boot), `Restart=always` (comes back if it
# crashes), and ordered After=network-online so the network is up first.
#
# Idempotent: safe to re-run (e.g. after a Jetson reflash or a path change).
#
# Usage:   bash install-services.sh            # install/refresh + enable + start
#          bash install-services.sh --no-start # install + enable, don't (re)start
set -euo pipefail

# ── paths (edit if thor's layout changes) ────────────────────────────────────
USER_NAME="${SUDO_USER:-$USER}"
HOME_DIR="$(eval echo "~$USER_NAME")"
MAXSWERVE_DIR="$HOME_DIR/projects/upstream/maxswerve-host"
CONDA_PY="$HOME_DIR/miniconda3/envs/base-module-env/bin/python"
DIAG_PY="$MAXSWERVE_DIR/diagnose.py"          # base/wheel connectivity check

START=1
[[ "${1:-}" == "--no-start" ]] && START=0

echo "▸ user=$USER_NAME  home=$HOME_DIR"
[[ -x "$CONDA_PY" ]]        || echo "  ⚠ conda python not found at $CONDA_PY"
[[ -f "$MAXSWERVE_DIR/mabel_swerve_bridge.py" ]] || echo "  ⚠ bridge not found in $MAXSWERVE_DIR"

# ── 1) bridge service ────────────────────────────────────────────────────────
sudo tee /etc/systemd/system/mabel-bridge.service >/dev/null <<EOF
[Unit]
Description=MABEL real swerve teleop bridge (control WebSocket :9090)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$MAXSWERVE_DIR
Environment=PYTHONPATH=$MAXSWERVE_DIR
ExecStart=$CONDA_PY -u mabel_swerve_bridge.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
echo "▸ wrote mabel-bridge.service"

# ── 2) diagnosis: a `mabel-diag` command + a boot-time logging service ───────
install -d -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/bin"
cat > "$HOME_DIR/bin/mabel-diag" <<EOF
#!/usr/bin/env bash
# Check which parts of the robot are connected (base + wheels today).
exec "$CONDA_PY" "$DIAG_PY" "\$@"
EOF
chmod +x "$HOME_DIR/bin/mabel-diag"; chown "$USER_NAME:$USER_NAME" "$HOME_DIR/bin/mabel-diag"
echo "▸ wrote ~/bin/mabel-diag (run anytime)"

# Runs once at boot, ~10 s after the bridge (lets the Teensy link settle), and
# logs the report to the journal:  journalctl -u mabel-diag
sudo tee /etc/systemd/system/mabel-diag.service >/dev/null <<EOF
[Unit]
Description=MABEL robot diagnosis at boot (base + wheel connectivity → journal)
After=mabel-bridge.service
Wants=mabel-bridge.service

[Service]
Type=oneshot
User=$USER_NAME
Environment=NO_COLOR=1
ExecStartPre=/bin/sleep 10
ExecStart=$HOME_DIR/bin/mabel-diag
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
echo "▸ wrote mabel-diag.service (runs at boot)"

# ── 3) enable (+ optionally start) ───────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl enable mabel-bridge.service mabel-diag.service >/dev/null
echo "▸ enabled on boot"
if [[ $START -eq 1 ]]; then
  sudo systemctl restart mabel-bridge.service
  sleep 2
  echo "▸ started"
fi

echo
echo "Status:"
systemctl is-enabled mabel-bridge mabel-diag | sed 's/^/  enabled: /'
systemctl is-active  mabel-bridge | sed 's/^/  active:  /'
echo
echo "Diagnose:  mabel-diag                    (which parts are connected — base + wheels)"
echo "           journalctl -u mabel-diag      (the boot-time report)"
echo "Manage:    systemctl status mabel-bridge   ·   journalctl -u mabel-bridge -f"
echo "           sudo systemctl restart|stop mabel-bridge"
echo "Arm to drive:  ~/bin/mabel-arm 60   (motion stays disabled until armed)"

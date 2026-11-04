#!/usr/bin/env bash
# install_deps.sh -- one-time setup for the ROS-free camera bring-up.
#
# Creates sensors/.venv (Python 3.12, the same ABI as /opt/ros/jazzy so this can
# later feed a ROS bridge) and installs:
#   * pyrealsense2   -- Intel RealSense SDK python bindings  (pip)
#   * pyzed          -- Stereolabs ZED SDK python bindings   (from /usr/local/zed)
#   * open3d, rich   -- point-cloud viewer + pretty dashboard (pip)
#   numpy + OpenCV are inherited from the system via --system-site-packages.
#
#   bash install_deps.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY=/usr/bin/python3                       # 3.12, matches ROS Jazzy ABI
ZED_SDK=/usr/local/zed

c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_b=$'\e[1m'; c_d=$'\e[2m'; c_0=$'\e[0m'
step(){ printf '%s\n' "${c_b}▸ $*${c_0}"; }
ok(){   printf '%s\n' "  ${c_ok}✓ $*${c_0}"; }
warn(){ printf '%s\n' "  ${c_warn}! $*${c_0}"; }

step "Creating venv at .venv (system site-packages for numpy/OpenCV)"
"$PY" -m venv --system-site-packages "$VENV" || { echo "venv failed"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q && ok "pip ready ($(python --version))"

step "Installing pyrealsense2 + open3d + rich (pip)"
if pip install -q pyrealsense2 open3d rich requests; then
  ok "pip packages installed"
else
  warn "one or more pip packages failed (open3d is heavy; cloud view may be off)"
fi

step "Installing pyzed from the ZED SDK ($ZED_SDK)"
if [[ -f "$ZED_SDK/get_python_api.py" ]]; then
  ( cd "$(mktemp -d)" && python "$ZED_SDK/get_python_api.py" ) \
    && ok "pyzed installed" \
    || warn "pyzed install failed -- ZED features will be skipped (RealSense still works)"
else
  warn "ZED SDK not found at $ZED_SDK -- install it from stereolabs.com if you need the ZED"
fi

step "Verifying imports"
python - <<'PY'
mods = ["numpy", "cv2", "pyrealsense2", "open3d", "rich", "pyzed.sl"]
for m in mods:
    try:
        __import__(m); print(f"  \033[32m✓\033[0m {m}")
    except Exception as e:
        print(f"  \033[33m!\033[0m {m}: {type(e).__name__}")
PY
echo
ok "done -- run:  bash bring_up_cameras.sh"

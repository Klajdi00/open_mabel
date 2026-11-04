#!/usr/bin/env bash
# MABEL Hardware Setup Studio (:8300) — per-module status · control · calibration.
#   ./run.sh            # real hardware (stop the hardware bridge / HAL first)
#   ./run.sh --sim      # mabel_hw mock plants, no hardware; saves go to a scratch copy
# Extra flags pass through to studio.py (--http PORT, --calib-root DIR, --no-open).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"
exec "$PY" "$HERE/studio.py" "$@"

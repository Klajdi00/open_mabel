#!/usr/bin/env bash
# Launch the MABEL Swerve Studio GUI.
#
# IMPORTANT: this runs under the project venv (.venv), built from the *system*
# Python. The Anaconda Python's Tcl/Tk has no fontconfig/Xft, so it can only
# render ancient X11 bitmap fonts (the "dated" look). The system Tk renders
# real anti-aliased iOS-style fonts. The venv has sparkcan_py compiled for it,
# so hardware control works too.
#
# First-time / after a clean checkout, build the venv with:  ./setup_gui_venv.sh
#
#   ./swerve_gui.sh            # connect to can0 (falls back to SIM if down)
#   ./swerve_gui.sh --mock     # force simulation mode, no hardware needed
#   ./swerve_gui.sh --iface can1
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python -m gui "$@"
fi

echo "warning: .venv not found — run ./setup_gui_venv.sh first for proper fonts." >&2
echo "falling back to system python3 (sparkcan_py may be unavailable → SIM mode)." >&2
exec python3 -m gui "$@"

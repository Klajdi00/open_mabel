#!/usr/bin/env bash
# Launch the MABEL Lift control GUI.
#
# IMPORTANT: runs under the project .venv (built from the *system* python3),
# whose Tcl/Tk has fontconfig/Xft and renders real anti-aliased iOS fonts.
# Anaconda's python3 only has X11 bitmap fonts and looks dated — so we avoid it.
# First time:  ./setup_gui_venv.sh
#
#   ./lift_gui.sh                 # auto-connect to the Pico
#   ./lift_gui.sh -p /dev/ttyACM0 # force a port
set -euo pipefail
cd "$(dirname "$0")"

# 1) project venv (best fonts + pyserial)
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python gui/gui.py "$@"
fi

# 2) system python3 if it already has pyserial + tkinter (good fonts)
if /usr/bin/python3 -c "import serial, tkinter" >/dev/null 2>&1; then
    exec /usr/bin/python3 gui/gui.py "$@"
fi

# 3) last resort: whatever python3 is on PATH (may be Anaconda → dated fonts)
echo "warning: no .venv — run ./setup_gui_venv.sh for proper anti-aliased fonts." >&2
echo "falling back to '$(command -v python3)' (may be Anaconda Tk = dated fonts)." >&2
exec python3 gui/gui.py "$@"

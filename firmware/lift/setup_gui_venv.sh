#!/usr/bin/env bash
# Build the lift GUI venv from the SYSTEM python3 (NOT Anaconda).
#
# Why: the system Tcl/Tk is built with fontconfig/Xft, so it renders real
# anti-aliased iOS-style fonts. Anaconda's Tk only has X11 bitmap fonts and
# looks dated. The GUI only needs pyserial (Tkinter ships with system python3).
set -euo pipefail
cd "$(dirname "$0")"

SYS_PY="${SYS_PY:-/usr/bin/python3}"
echo "Building .venv from $SYS_PY ($("$SYS_PY" --version))"

rm -rf .venv
"$SYS_PY" -m venv --without-pip .venv

if [[ ! -x .venv/bin/pip ]]; then
    echo "Bootstrapping pip…"
    TMP="$(mktemp /tmp/get-pip-XXXX.py)"
    if command -v curl >/dev/null; then
        curl -sSL https://bootstrap.pypa.io/pip/get-pip.py -o "$TMP"
    else
        wget -qO "$TMP" https://bootstrap.pypa.io/pip/get-pip.py
    fi
    .venv/bin/python "$TMP"
    rm -f "$TMP"
fi

echo "Installing pyserial…"
.venv/bin/pip install -q -U pip pyserial

echo
echo "Done. Launch the GUI with:  ./lift_gui.sh"

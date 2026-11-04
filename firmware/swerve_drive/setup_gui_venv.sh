#!/usr/bin/env bash
# Build the GUI venv from the SYSTEM python3 (not Anaconda).
#
# Why a venv from /usr/bin/python3?  The system Tcl/Tk is built with
# fontconfig/Xft, so it renders real anti-aliased iOS-style fonts. Anaconda's Tk
# only has X11 bitmap fonts and looks dated. The venv also gets sparkcan_py
# compiled for it, so hardware control works.
#
# Re-run any time you change the C++ bindings (sparkcan_py/...).
set -euo pipefail
cd "$(dirname "$0")"

SYS_PY="${SYS_PY:-/usr/bin/python3}"
echo "Building .venv from $SYS_PY ($("$SYS_PY" --version))"

rm -rf .venv
# Ubuntu's system python ships without ensurepip, so create venv without pip
# then bootstrap pip via get-pip.
"$SYS_PY" -m venv --without-pip .venv

if [[ ! -x .venv/bin/pip ]]; then
    echo "Bootstrapping pip…"
    TMP_GETPIP="$(mktemp /tmp/get-pip-XXXX.py)"
    if command -v curl >/dev/null; then
        curl -sSL https://bootstrap.pypa.io/pip/get-pip.py -o "$TMP_GETPIP"
    else
        wget -qO "$TMP_GETPIP" https://bootstrap.pypa.io/pip/get-pip.py
    fi
    .venv/bin/python "$TMP_GETPIP"
    rm -f "$TMP_GETPIP"
fi

echo "Installing build deps…"
.venv/bin/pip install -q -U pip scikit-build-core pybind11 cmake ninja

echo "Building + installing sparkcan_py for the venv…"
.venv/bin/pip install ./sparkcan_py --no-build-isolation

echo
echo "Done. Launch the GUI with:  ./swerve_gui.sh"

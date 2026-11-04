"""Put the calibration package on sys.path and define result directories, so
scripts run from anywhere without installation.

Auto-locates the *domain root* (the nearest parent directory that contains a
``src/`` folder), so scripts may live at any depth below it
(``hardware/``, ``simulation/``, ``scripts/`` ...).
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_ROOT = _here
while _ROOT != os.path.dirname(_ROOT):
    if os.path.isdir(os.path.join(_ROOT, "src")):
        break
    _ROOT = os.path.dirname(_ROOT)

_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CALIB_ROOT = _ROOT
MODEL_PATH = os.path.join(_ROOT, "models", "right_arm_sysid.xml")
RESULTS_DIR = os.path.join(_ROOT, "results")
EXP_DIR = RESULTS_DIR  # backward-compatible alias
os.makedirs(RESULTS_DIR, exist_ok=True)


def result_dir(*parts):
    """Return (creating it) a subdirectory under ``results/``.

    e.g. ``result_dir("DM-J4310")`` -> ``<root>/results/DM-J4310/``.
    """
    d = os.path.join(RESULTS_DIR, *parts)
    os.makedirs(d, exist_ok=True)
    return d

#!/usr/bin/env bash
# One-command driver for the MABEL arm system-ID demo.
#
#   ./run.sh                 # full pipeline (model + sysid + movie + plots)
#   ./run.sh --no-video      # skip the MuJoCo movie (faster)
#   ./run.sh --noise 0.1     # any 01_run_sysid.py flag is forwarded
#
# Produces, in results/:
#   excitation.gif, excitation_trajectory.png, torque_fit.png,
#   sysid_results.png, sysid_summary.json, identified_params.json
set -e
cd "$(dirname "$0")"

# Headless software GL so the movie renders without a display.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

# Use python3 if a bare `python` is not on PATH.
PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "no python interpreter found"; exit 1; fi

echo "############################################################"
echo "# MABEL arm system identification"
echo "############################################################"

echo
echo ">>> [1/3] building the standalone arm model from mabel_full.xml"
"$PY" tools/extract_arm_model.py

echo
echo ">>> [2/3] sanity-checking model + Lagrangian regressor"
"$PY" scripts/00_check_model.py

echo
echo ">>> [3/3] running identification (generate -> scramble -> identify -> plot)"
"$PY" scripts/01_run_sysid.py "$@"

echo
echo ">>> using the identified parameters for gravity compensation"
"$PY" scripts/02_gravity_comp_demo.py

echo
echo "Done. Open the plots/movie in:  $(pwd)/results/"

#!/usr/bin/env bash
# Simulation self-test for the MABEL actuator friction/inertia calibration.
# NO HARDWARE is touched — this validates the pipeline end to end:
#
#   ./run_selftest.sh                 # run the sim self-test
#   ./run_selftest.sh --model cv      # any simulate_check.py flag is forwarded
#
# It builds a ground-truth Damiao-class actuator, synthesises a velocity
# staircase + chirp, identifies the friction + reflected inertia, and checks the
# recovery against the known truth.
#
# To calibrate REAL hardware instead, use ../hardware/run_hardware.sh, e.g.:
#   ../hardware/run_hardware.sh DM-J4310 2 0.10
#
# Produces, in results/simulation/:
#   friction_curve.png, torque_fit.png, sim_summary.json
set -e
cd "$(dirname "$0")"

PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then echo "no python interpreter found"; exit 1; fi

echo "############################################################"
echo "# MABEL actuator friction & inertia calibration (SIM test)"
echo "############################################################"
echo
"$PY" simulate_check.py "$@"
echo
echo "Done. Plots in:  $(cd .. && pwd)/results/simulation/"
echo "Docs: ../docs/ACTUATOR_CALIBRATION.pdf"

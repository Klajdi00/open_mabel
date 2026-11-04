# calibration

System identification for the parameters CAD does not give: per-actuator friction/damping/inertia and per-link arm mass/CoM.

| Package | Identifies | MuJoCo attributes | Report |
|---|---|---|---|
| [`motor_calibration/`](motor_calibration/) | Coulomb friction, viscous damping, reflected inertia | `frictionloss`, `damping`, `armature` | [ACTUATOR_CALIBRATION.pdf](motor_calibration/docs/ACTUATOR_CALIBRATION.pdf) |
| [`arm_calibration/`](arm_calibration/) | 7-DOF arm link mass and centre of mass | `<inertial>` | [ARM_CALIBRATION.pdf](arm_calibration/docs/ARM_CALIBRATION.pdf) |

Deployed encoder offsets and gravity feed-forward data used by the HAL live in [`hardware_bridge/calibration_data/`](../hardware_bridge/calibration_data/).

## Run

```bash
pip install -r calibration/requirements.txt          # hardware: + pyserial python-can

calibration/motor_calibration/simulation/run_selftest.sh     # no hardware
calibration/motor_calibration/gui/run_gui.sh --backend sim   # GUI, synthetic motor
calibration/motor_calibration/hardware/scan.sh               # list motors on can0 (non-energising)
calibration/arm_calibration/run.sh --no-video                # arm identification in simulation
```

Safety: only `motor_calibration/hardware/` scripts (`run_hardware.sh`, `run_serial.sh`, `collect.py`, `hold_test.py`, `serial_demo.py` with a motion flag) and the GUI's Calibrate button energise a motor. Bench-mount with the shaft free and keep a hand on the e-stop.

Check the motor's programmed MIT ranges `(p_max, v_max, t_max)` match `motor_calibration/src/mabel_actid/motors.py` first, or every value is silently rescaled.

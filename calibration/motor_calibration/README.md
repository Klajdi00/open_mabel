# motor_calibration (`mabel_actid`)

Bench identification of a DaMiao actuator's Coulomb friction, viscous damping and reflected inertia (free output shaft), exported as MuJoCo joint attributes, an Isaac Lab `DCMotorCfg` snippet and a feed-forward table.

Report: [docs/ACTUATOR_CALIBRATION.pdf](docs/ACTUATOR_CALIBRATION.pdf).

## Run

```bash
simulation/run_selftest.sh                     # synthetic actuator; moves nothing
gui/run_gui.sh --backend sim                   # GUI without hardware
gui/run_gui.sh --motor DM-J4340P --id 1        # GUI on a real motor

hardware/scan.sh                               # Linux/SocketCAN, non-energising scan of can0
hardware/run_hardware.sh DM-J4310 2            # MOTOR ID [VEL_FRAC]
python hardware/detect_serial.py --ids 1       # serial adapter, non-energising
hardware/run_serial.sh DM-J4340P 1 0.30        # serial adapter calibration
```

Step by step (from `hardware/`): `collect.py` → `identify.py` → `export_mujoco.py` → `plot_data.py`. Results land in `results/<MOTOR>/` (`params.json`, `plots/`, and for GUI runs `calibration.json`, `mujoco_joint.xml`).

Backends: `serial` (DaMiao USB-CAN), `gs_usb` (CANable-class), `socketcan` (Linux `can0`), `sim`. Force with `--backend`. Only one process can hold a USB adapter.

## Safety

- Telemetry uses the non-energising refresh (`0xCC`); enable (`0xFC`) is sent only to calibrate. `collect.py` always disables on exit, including Ctrl-C.
- Keep clear of the shaft, keep a hand on the e-stop, start with a low `--vel-frac`.
- Confirm the motor's MIT ranges match `src/mabel_actid/motors.py` before trusting a fit.

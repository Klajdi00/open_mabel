# Setup Studio (`web_gui/setup_studio`)

Hardware bring-up page for MABEL: per-module **status · control · calibration** with the 3-D rig
(`web_gui/shared_assets/`). Modules: Arm · L/R, Torso, Neck, Lift, Hand · L/R, Base (read-only).

```bash
./run.sh                 # real hardware  ->  http://127.0.0.1:8300
./run.sh --sim           # mabel_hw mock plants, no hardware
./run.sh --http 8301 --calib-root /path/to/dir --no-open
```

It opens the hardware through `hardware_bridge/mabel_hw` (`MabelHardware`, the same subsystems
and drivers the HAL uses) and takes the HAL's one-owner lock, so it **refuses to start while the
hardware bridge is running** — stop the HAL first. Calibration is written through mabel_hw's
`CalibrationStore` into the existing slices of `hardware_bridge/calibration_data/mabel.json`
(`arm/openarm-dual`, `torso/damiao`, `neck/teensy`, `hand/orca-<side>`, `lift/pico/calib/_travel`),
centre-anchored (`raw_min`, `raw_max`, `dir`, `model_mid`, `reliability`); the HAL hot-reloads it.
With `--sim` the store is a fresh scratch copy, so the robot's file is never touched.

Safety:
- Nothing is energised except from the **Control** tab. At rest every joint group is torque off:
  the zero-torque damping hold on the arms/torso/neck, free-drive on the hands.
- **E-STOP** (button, Esc or Space) drops every group to that state, halts the lift and drops its
  torque; RE-ENGAGE only unlocks the controls. Losing the page for 1.5 s does the same (deadman).
- Calibration is by hand: support the arm, move each joint to both stops, Save. Lift calibration
  drives the column into both stops — keep the path clear. The base is kept e-stopped.

# orcahand — ORCA hand driver (Feetech bus)

Python driver for the ORCA hands: 17 Feetech servos per hand (IDs 1–17, 16 fingers + wrist) on a 1 Mbaud TTL bus through a USB adapter. It gives a joint-space API in radians with clamping to the calibrated end-stops. The HAL vendors this package (`hardware_bridge/mabel_hw/drivers/orcahand/`) and owns both buses on the robot, so for bench work stop the HAL's hand subsystem first (the port is opened exclusively).

Docs: [`../docs/orcahand_api.pdf`](../docs/orcahand_api.pdf), [`../docs/orca_firmware.pdf`](../docs/orca_firmware.pdf).

## Install and use

```bash
cd firmware/orca_hand/feetech
pip install -e .            # + console scripts orcahand-scan, orcahand-calibrate; .[dev] for pytest
python3 -m pytest tests -q  # hardware-free
```

```python
from orcahand import OrcaHand
hand = OrcaHand("left").connect(enable_torque=False)    # free-drive; OrcaHand.mock("left") for no hardware
hand.enable_torque()
hand.set_joint_positions({"index_mcp": 0.5})            # radians, partial dict ok
q = hand.get_joint_positions()
hand.disconnect()                                       # disables torque first
```

## Tools (all take `--model left|right`, `--port`, `--mock`)

```bash
python3 gui/gui.py --hand left                # ORCA Hand Studio
python3 scripts/scan.py      --model left     # list servos by joint
python3 scripts/bringup.py   --model left --watch --no-torque
python3 scripts/tension.py   --model left     # 1. seat the tendons
python3 scripts/calibrate.py --model left     # 2. end-stop sweep (--joints ..., --sim self-test)
python3 control/torque_off.py --hand left     # release all servos
```

Config: `orcahand/configs/<side>_hand.yaml` (joint ↔ ID, angle range, `invert`) and `<side>_calibration.json` (raw-tick end-stops). The robot's copy is described in `hardware_bridge/calibration_data/README.md`.

## Safety

- Every command is clamped to the calibrated tick range. Calibrate a newly assembled hand before enabling torque.
- `connect()` enables torque by default, and an energized servo jumps to its stored goal. Pass `enable_torque=False` (or `--no-torque`). The HAL always brings hands up in free-drive.

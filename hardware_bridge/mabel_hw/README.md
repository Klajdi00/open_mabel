# `mabel_hw` — hardware access layer

The installable package behind the HAL and the in-process API for every device. Details: [`../docs/mabel_hw_api.pdf`](../docs/mabel_hw_api.pdf).

```python
from mabel_hw import MabelHardware

with MabelHardware() as hw:                    # lock -> calibration -> connect; mock=True for no hardware
    state = hw.read()                          # frozen RobotState snapshot, SI units
    with hw.left_arm.claim("teleop") as arm:   # exclusive writer lease
        arm.set_torque(True)                   # hold at the measured pose
        arm.set_joint_positions({"l_j1": 0.3}) # model-frame radians
    hw.estop()                                 # lease-free, always allowed
```

Install (optional): `pip install -e .` from `hardware_bridge/` (`[can]` for SocketCAN). Tests: `python3 -m pytest tests -q` (hardware-free, fakes in `tests/fakes/`).

| Path | Role |
|---|---|
| `api.py`, `arbitration.py` | facade; one writer per actuator group |
| `subsystems/` | calibrated ROS-free cores (joint_group, neck, base, hands, lift, lidar, audio) |
| `calibration/` | `CalibrationStore` over `../calibration_data/` |
| `drivers/` | protocol only (arms/base Teensy links, DaMiao bench backends, Dynamixel, ORCA, lift, lidar, audio, joystick) |
| `ros/`, `cameras/`, `rpc/` | DDS adapters + supervisor; video plane; hand RPC |
| `config.py`, `types.py`, `mock.py`, `singleton.py` | tunables (`MABEL_HW_<NAME>`), types, mock plants, one-owner lock |

## Rules

- Mutating calls exist only on `claim()`ed writers. Never add a write path that bypasses the broker.
- Nothing energizes on connect by default. Backends that do (`teensy`, `can`) must ship with their group disabled.
- `rclpy` is allowed only in `ros/` and `cameras/` (`tests/test_mabel_hw_pure.py`).
- Calibration is applied once, in the subsystems: model frame above the drivers, raw motor units below.

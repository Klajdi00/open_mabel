# hardware_bridge — the HAL

The only process that opens a device on the robot: swerve base, arms, torso, neck, hands, lift, lidar, audio and cameras. It runs on the robot computer as a systemd service, outside the ROS workspace. See the root [Architecture](../README.md#architecture).

| Surface | Used by |
|---|---|
| ROS 2 DDS topics (contract frozen in `tests/contract/contract.json`) | `ros/`, the `server/` gateway, `gui/` |
| Video plane (MJPEG `:8080`, UDP `:9091`/`:9092`) | apps, browser studios ([docs/video_streaming.md](docs/video_streaming.md)) |
| Hand RPC (TCP `127.0.0.1:8779`, `mabel_hw/rpc/hand_server.py`) | hand calibration, ROS-free hand bench |
| Python API (`import mabel_hw`, [mabel_hw/README.md](mabel_hw/README.md)) | scripts run while the service is stopped |

Motion commands reach the HAL only after passing through `controller/`.

## Run

```bash
cd hardware_bridge
./run_bridge.sh                                # foreground; sources ros/scripts/mabel_env.sh
MABEL_HW_BASE_JOYSTICK=true ./run_bridge.sh    # also drive the base from a gamepad (hold L1)
./gui.sh                                       # Tk Control Center (gui/README.md)

sudo ./install/install.sh                      # boot service; edit paths/User in install/mabel-hardware-bridge.service first
journalctl -u mabel-hardware-bridge -f
```

Local overrides go in git-ignored `hardware_bridge/.env.local`. One owner only: a lock on `/tmp/mabel_hardware_bridge.lock` refuses a second manual start.

## Subsystems

| Subsystem | Device | Transport | Default |
|---|---|---|---|
| `base` | 3 MAXSwerve modules behind a Teensy 4.1 | UDP + protobuf | on |
| `left_arm`, `right_arm`, `torso`, `neck` | DaMiao actuators + Dynamixel neck behind the arms Teensy (`firmware/openarm_dual`) | UDP + protobuf | on |
| `hands` | 2 ORCA hands (Feetech, 17 servos each) | USB TTL serial, picked by adapter serial number | on |
| `lift` | DC column on a Pico (`firmware/lift`) | USB serial | off |
| `lidar` | RPLIDAR A2M12 | serial | on |
| `audio`, `cameras` | USB sound card; D435i, ZED Mini, 2× D405 | ALSA; vendor SDKs | on |

A missing device never stops the bridge. That subsystem goes idle and reconnects when the device comes back. Each tunable in `mabel_hw/config.py` can be overridden as `MABEL_HW_<NAME>`, e.g. `MABEL_HW_LIFT=true`, `MABEL_HW_CAMERAS=false`, `MABEL_HW_BASE_TEENSY_IP=...`. Calibration is applied only in `mabel_hw/subsystems/joint_group.py`, from [`calibration_data/`](calibration_data/).

## Safety

- Nothing energizes at startup: arms, torso and neck come up limp (damping hold). Torque is enabled explicitly (`/<group>/torque_enable`), latching a hold at the measured pose.
- When the arms Teensy rejoins, every group goes back to limp. The firmware disables the motors after 250 ms with no host traffic. Support the arms when you stop the bridge.
- Base: zero drive when `cmd_vel` is older than `MABEL_HW_BASE_CMD_TIMEOUT` (0.5 s). `/mabel/hw/base/estop` latches a stop. `MabelHardware.estop()` bypasses write leases.
- The control law (gravity FF, impedance) is in `controller/mabel/arm_control`, not here. The HAL only clamps `t_ff` to `t_max` and targets to the calibrated window.
- Stop the service before running any `tools/` script that commands hardware: `sudo systemctl stop mabel-hardware-bridge`.

## Tests and tools

```bash
python3 -m pytest tests -q                                       # hardware-free
tools/software_selftest.sh                                       # base path against a mock Teensy (needs ROS)
python3 tools/sync_calibration.py --check
python3 -m mabel_hw.drivers.orcahand.calibrate --model left      # hand end-stop sweep (hand subsystem released)
python3 -m mabel_hw.rpc.hand_server                              # ROS-free hand RPC on :8779
```

Optional package install: `pip install -e .` (`.[can]` for SocketCAN).

Reports ([`docs/`](docs/)): [hardware_bridge.pdf](docs/hardware_bridge.pdf), [mabel_hw_api.pdf](docs/mabel_hw_api.pdf), [arm_controllers.pdf](docs/arm_controllers.pdf), [gravity_ff_tracking.pdf](docs/gravity_ff_tracking.pdf).

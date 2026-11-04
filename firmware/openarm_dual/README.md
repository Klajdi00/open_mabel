# openarm_dual — arms, torso and neck firmware (Teensy 4.1)

A configuration-driven actuator gateway, derived from the upstream OpenArm DAQ firmware. It drives DaMiao motors (MIT mode, 3 CAN buses) and ROBOTIS Dynamixel servos (half-duplex serial) behind one UDP + protobuf host interface. The whole layout comes from `/robot.json` on the SD card, and the board advertises it to the host on connect (`RobotInfo`). Host side: `hardware_bridge/mabel_hw/drivers/openarm_link.py`.

## Build and flash

```bash
cd firmware/openarm_dual
cp config/robot.json /path/to/sdcard/robot.json   # MABEL's layout; set net.ip for your LAN
pio run -e teensy41 -t upload                     # if the first upload fails, retry once (never after a success)
pio device monitor -b 115200                      # lists every actuator, live or not
```

Diagnostic envs (each replaces the control firmware until you reflash `teensy41`): `teensy41_test_config` (parse the SD config), `teensy41_test_regs` (read-only register dump), `teensy41_test_j1` (single-motor CAN check), `teensy41_provision` (set motor IDs), `teensy41_test_dxl` (Dynamixel probe).

## Layout (`config/robot.json`)

| Group | Bus | Actuators |
|---|---|---|
| Left arm | CAN3 (pins 30/31) | `l_j1`..`l_j7` (DM8009 ×2, DM4340 ×2, DM4310 ×3) |
| Right arm | CAN1 (22/23) | `r_j1`..`r_j7` |
| Spine | CAN2 (0/1) | `torso` (DM10422), `neck_yaw` (DM4310) |
| Neck | Serial3 pin 14, half-duplex | `neck_pitch`, `neck_roll` (2XL430) |

Conventions: `MST_ID = CAN_ID + 0x10`, IDs unique per bus, never `0x7FF`. The Dynamixel bus is 5 V logic and Teensy pins are not 5 V tolerant, so use a level shifter. Each motor's `PMAX/VMAX/TMAX` are read from its registers, not from the config.

Set motor IDs from a host with a DaMiao USB-CAN dongle: `python3 scripts/dm_configurator.py scan` / `setid 1 2`.

## Safety

- At boot every motor is disabled, and a bad config means nothing is ever enabled (fast-blinking LED).
- When a host connects, motors get a damping hold (`kp = 0`): limp and back-drivable until the host enables torque.
- No fresh command for `cmd_ms` (150 ms) → damping hold. No host packet for `conn_ms` (250 ms) → all motors disabled. Support the arms before stopping the host.

Docs: [`docs/HOST_PROTOCOL.md`](docs/HOST_PROTOCOL.md) (wire protocol, timing, safety), [`docs/DXL_ACTUATOR_TRANSPORT.md`](docs/DXL_ACTUATOR_TRANSPORT.md), [`docs/CanBus-tutorial.md`](docs/CanBus-tutorial.md), schema `proto/robot.proto`.

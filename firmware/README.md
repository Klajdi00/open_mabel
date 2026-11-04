# firmware

On-device firmware and host drivers, one folder per subsystem. Each exposes a host-side driver that [`../hardware_bridge/`](../hardware_bridge/) wraps.

| Folder | Runs on | Bus | Drives |
|---|---|---|---|
| [`swerve_drive/`](swerve_drive/) | Teensy 4.1 | REV SPARK CAN, 1 Mbit | 3 MAXSwerve modules |
| [`openarm_dual/`](openarm_dual/) | Teensy 4.1 | DaMiao MIT CAN ×3 + Dynamixel serial; UDP/protobuf to host | both arms, torso, neck |
| `openarm/` | Teensy 4.1 | DaMiao MIT CAN | single-bus arm firmware + CAN tools (`scripts/`) |
| [`lift/`](lift/) | Pico (MicroPython) | USB serial | lift column |
| [`orca_hand/`](orca_hand/feetech/) | host Python | Feetech TTL, 1 Mbit | two ORCA hands |

## Build and flash

```bash
cd firmware/swerve_drive && pio run -e teensy41 -t upload
cd firmware/openarm_dual && pio run -t upload            # reads /robot.json from the SD card
pip install mpremote && firmware/lift/tools/deploy.sh    # Pico lift

python3 firmware/orca_hand/feetech/scripts/scan.py       # hands: no flashing; check IDs 1..17
python3 firmware/orca_hand/feetech/scripts/calibrate.py

firmware/can_up.sh --ping                                # SocketCAN up + ping motors
firmware/can_scan.sh
firmware/can_down.sh
```

- Check each DaMiao motor's `(p_max, v_max, t_max)` in the vendor tool before first use. A mismatch silently rescales every value on the wire.
- macOS has no SocketCAN, so the CAN tools fall back to `gs_usb` over libusb.
- References: `openarm_dual/docs/HOST_PROTOCOL.md`, `swerve_drive/REV_CAN_PROTOCOL.md`, [`orca_hand/docs/orcahand_api.pdf`](orca_hand/docs/orcahand_api.pdf), [`orca_hand/docs/orca_firmware.pdf`](orca_hand/docs/orca_firmware.pdf).

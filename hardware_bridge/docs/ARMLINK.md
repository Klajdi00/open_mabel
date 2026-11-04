# ArmLink — wire contract of the `firmware/openarm` gateway

> **Status:** reference only. The robot's arms Teensy runs `firmware/openarm_dual`
> (UDP + protobuf, host side `mabel_hw/drivers/openarm_link.py`). ArmLink is an
> alternative, lighter gateway firmware in `firmware/openarm` (PlatformIO env
> `mabel_armlink`) that has not been validated on the robot and has no host client in
> `mabel_hw`. A future client would implement the
> `mabel_hw.drivers.seam.JointGroupDriver` interface (as `drivers/openarm.py` does) and
> take calibration from `calibration_data/`.

The firmware header that mirrors this document is
`firmware/openarm/include/armlink_protocol.h`.

## Topology

A Teensy 4.1 on a dedicated Ethernet link owns both arm CAN buses (FlexCAN CAN1 = left
arm, CAN2 = right arm; DaMiao MIT mode, 1 Mbit/s) and acts as a transparent MIT gateway:
the host streams per-motor position targets, the Teensy forwards them on its 500 Hz
loop and streams every motor's feedback back. The Teensy is the UDP server on port
5800 (static IP set at the top of `firmware/openarm/src/main_armlink.cpp`); it learns the host
endpoint from inbound packets. `hardware_bridge/install/setup_arms_lane.sh` creates a
NetworkManager profile for such a dedicated lane.

## Framing

One UDP datagram per message. Byte 0 is a type tag; the rest is a little-endian packed
struct (no protobuf on either side).

| Tag | Name | Direction | Body |
|---|---|---|---|
| `0x03` | HELLO | host → Teensy | `<H>` stream_hz (0 = firmware default, 100 Hz) |
| `0x01` | COMMAND | host → Teensy | `<IBB>` seq, flags (bit0 = TORQUE), count; then count × `<BBfff>` bus, id, pos_rad, kp, kd |
| `0x10` | STATE | Teensy → host | `<IIBB>` seq_echo, t_us, flags (bit0 = torque on), count; then count × `<BBfffBBB>` bus, id, pos_rad, vel_rad_s, tau_Nm, t_mos_C, t_rotor_C, err |

Positions on the wire are in the motor frame. The host applies calibration in both
directions.

## Safety contract

- **Boot is torque-off.** HELLO never energizes anything; only a COMMAND with the
  TORQUE flag set does.
- **Torque flag transitions:** set → enable (`0xFC`) every motor in the table; cleared
  → disable (`0xFD`) every motor (explicit limp, the e-stop path).
- Motors in the table but absent from a COMMAND are held at their measured position
  while torque is on, never snapped to a stale or zero target.
- **Command watchdog (~150 ms):** torque on but no fresh COMMAND → hold the measured
  position (a dead commander must not drop the arms).
- **Connection watchdog:** no host datagrams → keep holding and stop streaming STATE
  after ~1 s; telemetry resumes on the next HELLO.
- STATE streams at `stream_hz` with every mapped motor's latest feedback, including
  MOSFET and rotor temperature and the DaMiao error nibble.

## Build

```bash
cd firmware/openarm
pio run -e mabel_armlink            # build
pio run -e mabel_armlink -t upload  # flash
```

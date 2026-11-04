# Dynamixel actuators behind the same UDP API

MABEL's neck pitch and roll are one ROBOTIS 2XL430-W250-T (two axes, IDs 1 and 2) on a
half-duplex serial line. Every other actuator on the arms Teensy is a DaMiao DM-J
speaking MIT over CAN. The firmware drives both behind one interface; nothing above the
firmware knows the difference.

Code: `include/DxlBus.h` (bus scheduling, framing, baud detection) and
`include/DxlActuator.h` (one servo axis). Configuration: the `dxl_ports` block and
`"transport": "dynamixel"` actuators in `/robot.json` (see `../README.md`).

## The rule

> **The UDP API is per joint, never per transport.**

`proto/robot.proto` is unchanged. A host sends `pos/vel/kp/kd/t_ff` and receives
`position/velocity/torque` and health for every index, whatever wire is underneath.
Adding the neck was a `/robot.json` edit.

**A capability the transport lacks is published as 0, never approximated.** This reuses
the existing `RobotInfo` convention for an offline actuator, where 0 means "this value
does not exist". On the Dynamixel axes `t_ff`, `vel`, `torque` and `t_mos` are 0, and
`p_max`/`v_max`/`t_max` are 0 because a Dynamixel has no MIT fixed-point scale. Mapping
`vel` onto Profile Velocity (a speed cap, not a setpoint) or `torque` onto Present Load
(a percentage, not N·m) would put a plausible number in a field that means something
else.

## Where the transports differ

| | DM-J / CAN | Dynamixel / serial |
|---|---|---|
| Duplex | full, hardware FIFO | one wire; request and reply cannot overlap |
| Send | fire-and-forget frame | must be scheduled; `sendCommand` caches a setpoint |
| Dispatch | by CAN ID, in hardware | software matches each reply to the outstanding request |
| Ranges | read from the motor (`PMAX`/`VMAX`/`TMAX`) | none; reported as 0 |
| `enabled` | DaMiao error nibble | Torque Enable (address 64), read back |
| `kp == 0` | develops no torque | must release torque explicitly |

`kp == 0` means "do not hold this joint" everywhere else on the robot. A position servo
has no such continuum, so the firmware translates `kp == 0` into a torque release and
`kp > 0` into "hold the commanded goal".

The same asymmetry affects the stale-command fail-safe. The DaMiao damping hold
(`kp = 0`, `kd = default`, `pos = 0`) is safe for a DM-J but would mean "drive to raw
zero" on a position servo. The hold is therefore a per-transport operation
(`Robot::sendHold`); on a Dynamixel axis it leaves the last goal in place.

## Implementation notes

Issues that produced plausible-looking telemetry until handled explicitly:

- **Byte stuffing on receive.** Protocol 2.0 stuffs in both directions, and the CRC
  covers the stuffed form, so a stuffed reply validates but every later field is
  shifted by one byte. It appears only for certain data patterns (typically while
  moving). Replies are de-stuffed before parsing.
- **Read/write fairness.** Writes and reads strictly alternate per device; otherwise a
  write re-armed at loop rate starves feedback as soon as torque is on.
- **Status Return Level.** At the factory level (2) the servo replies to every write,
  and the next packet collides with that reply. The firmware sets level 1 and verifies
  it; an axis whose level will not stick is reported offline.
- **Two channels.** Torque-enable and mode transitions go through a guaranteed-delivery
  FIFO; setpoints go through a coalescing latest-wins slot.
- **Reply attribution.** The bus records which device and address are in flight, and a
  reply ends a wait only if it comes from the awaited device.
- **Module-wide EEPROM gate.** On a 2XL430, EEPROM is writable only while Torque Enable
  is 0 on every axis of the module. Bring-up reads before writing and refuses to drive
  an axis whose mode did not take.

## Operating notes

- **Baud rate is configuration.** `/robot.json` `dxl_ports[].baud` sets it; a missing or
  wrong value falls back to a sweep of the standard rates, keeping the rate at which
  the most servos answer. If nothing answers at boot, the port is parked at the
  configured rate and re-swept every 5 s while no host is connected.
- **Cadence scales with line rate.** A 21-byte state read takes about 0.5 ms at 1 Mbaud
  and about 8 ms at 57600 baud; `DxlBus::roundTripMs` derives the polling period from
  the measured byte time and device count.
- **The two axes of a 2XL430 share one comm module:** one baud rate, one protocol
  setting, two IDs.
- **Electrical:** the X-series TTL bus is 5 V logic and Teensy 4.x pins are 3.3 V and
  not 5 V tolerant. Use a level shifter on the data line (pin 14, TX3); a 1 kΩ series
  resistor is only a stopgap.

## Bench tools

| PlatformIO env | Purpose |
|---|---|
| `teensy41_test_dxl`, `teensy40_test_dxl` | protocol poker: `scan`, `info`, `baud`, `move`; touches only the serial bus. A factory servo (57600 baud) is invisible to the control firmware at 1 Mbaud until `baud` re-provisions it |
| `teensy41_dxl_rig`, `teensy40_dxl_rig` | runs the real `DxlBus`/`DxlActuator`/`Robot` classes through the calls `host_link` makes |

`test/native/run.sh` checks CRC, length, stuffing and status parsing on the host with
no hardware, anchored to the PING packet documented in the ROBOTIS e-manual.

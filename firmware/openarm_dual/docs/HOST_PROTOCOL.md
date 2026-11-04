# Host protocol — UDP + protobuf link to the actuator Teensy

This document specifies the link between a host computer and the `openarm_dual`
firmware: transport, messages, timing and the safety model, with the design decisions
behind them.

| Piece | Location |
|---|---|
| Firmware side | `src/host_link.h` (`RobotHostLink`), `src/main.cpp` |
| Canonical schema | `proto/robot.proto` (nanopb sizing in `proto/robot.options`) |
| Robot definition | `/robot.json` on the SD card, parsed by `include/config_loader.h` |
| MABEL host client | `hardware_bridge/mabel_hw/drivers/openarm_link.py` (hand-rolled codec, one link shared by all groups) |
| Generic host client | `hardware_bridge/mabel_hw/drivers/robot_host/client.py` (`RobotClient`) |

## 1. Summary

One Teensy drives a set of actuators defined by `/robot.json`. The host talks to it over
UDP on a direct Ethernet link. On connect the host sends `Hello`; the Teensy replies with
`RobotInfo` describing every actuator, so the host builds its actuator table at run
time. The Teensy then streams `State` at a fixed rate, the host sends MIT `Command`s
that the Teensy applies on its control tick, and `Health` is a separate low-rate
request/reply. Motor output is gated on the host being alive.

## 2. Network

The Teensy's identity comes from the `net` block of `/robot.json` (IP, netmask, gateway,
UDP port; default port 5800). The host uses any address on the same subnet. The Teensy
never needs to be told the host address: it replies and streams to the endpoint that
last sent it a packet. Changing the address is a config edit, not a rebuild. Once
cabled, the Teensy answers ICMP ping.

## 3. Transport and framing

- One UDP datagram per message. A dropped `State` is simply superseded by the next one;
  there is no TCP head-of-line blocking.
- Every datagram is `[tag byte][protobuf body]`. The tag is not part of protobuf.

| Tag | Direction | Message |
|---:|---|---|
| `0x01` | host → Teensy | `Command` |
| `0x02` | host → Teensy | `HealthRequest` |
| `0x03` | host → Teensy | `Hello` |
| `0x10` | Teensy → host | `State` |
| `0x11` | Teensy → host | `Health` |
| `0x12` | Teensy → host | `RobotInfo` (reply to `Hello`) |

All per-actuator repeated fields have a run-time count (at most `MAX_ACTUATORS = 32`).
Units are motor side: position rad, velocity rad/s, torque N·m.

## 4. Messages

### `Command` (host → Teensy)

MIT setpoints, one entry per actuator: τ = kp·(pos − q) + kd·(vel − dq) + t_ff.

| Field | Type | Meaning |
|---|---|---|
| `seq` | uint32 | echoed in `State.seq` (RTT and loss tracking) |
| `pos`, `vel` | repeated float | targets, rad and rad/s |
| `kp`, `kd` | repeated float | gains; `kd` must be > 0 when commanding a position |
| `t_ff` | repeated float | feedforward torque, N·m |
| `priority`, `commander_id` | uint32 | used by host-side arbitration only; ignored by the firmware |

The firmware caches the latest `Command` and pushes it to the actuators on the next
control tick, so command latency is up to one stream period (2.5 ms at 400 Hz).

### `Hello` (host → Teensy)

Registers the host endpoint, sets the stream rate and serves as the idle keep-alive.

| Field | Type | Meaning |
|---|---|---|
| `stream_hz` | uint32 | requested State rate; 0 = `stream.default_hz`; clamped to `stream.max_hz` |

### `HealthRequest` (host → Teensy)

`seq` (uint32), echoed in the reply.

### `RobotInfo` (Teensy → host)

| Field | Type | Meaning |
|---|---|---|
| `name` | string | robot name from `/robot.json` |
| `stream_default_hz`, `stream_max_hz` | uint32 | rates |
| `actuators` | repeated `ActuatorInfo` | one per actuator; defines the index order used everywhere |

`ActuatorInfo`: `name`, `model`, `bus`, `can_id`, `mst_id`; `p_max`, `v_max`, `t_max`
(read from the motor's own registers at bring-up; 0 = offline or not applicable, e.g.
a Dynamixel axis); `kp_max`, `kd_max` (MIT protocol constants); `default_kp`,
`default_kd` (damping-hold gains).

### `State` (Teensy → host)

| Field | Type | Meaning |
|---|---|---|
| `seq` | uint32 | last applied `Command.seq` |
| `t_us` | uint32 | Teensy `micros()` at send (relative timing only; no clock sync) |
| `live_mask` | uint32 | bit *i* set if actuator *i* returned feedback within the last 50 ms |
| `actuators` | repeated `ActuatorState` | `position`, `velocity`, `torque` |

### `Health` (Teensy → host)

`seq`, `t_us`, and per actuator: `err` (DaMiao error nibble, 1 = enabled OK, 8-E =
fault), `t_mos` and `t_rotor` (°C), `enabled`, `live`, `stale_ms` (age of the newest
feedback frame).

## 5. Actuator indexing

There is one flat actuator array. Its order and names come from the `actuators` list in
`/robot.json`, echoed verbatim in `RobotInfo`. Index *i* in `Command`, `State`, `Health`
and `live_mask` always means `RobotInfo.actuators[i]`. Hosts address actuators by index
or name; there is no built-in count and no per-role special case (a gripper is just
another named actuator). A disconnected actuator reads not-live and keeps its last
cached `err`.

## 6. Timing

1. **State is streamed** at `stream_hz` (MABEL default 400 Hz, cap 500 Hz). A host read
   of the latest State never blocks on the network.
2. **Commands are fire-and-forget**; the latest one is applied on the next tick.
3. **Health is request/reply**, kept off the hot path so `State` stays small.

DaMiao motors answer each command frame with exactly one feedback frame, so the control
tick both drives the motors and produces the next `State`: command rate and State rate
are the same number. The rate cap is set by the busiest CAN bus, since the buses
transmit in parallel. One bus carrying 8 actuators needs 16 frames per tick, about
1.8 ms at 1 Mbit/s, which is why the MABEL layout caps at 500 Hz. To raise the rate,
rebalance actuators across buses in `/robot.json`.

## 7. Safety model

Two watchdogs, both configured in the `watchdog` block of `/robot.json`:

| Watchdog | Default | Effect |
|---|---|---|
| Connection | `conn_ms` = 250 ms | no host packet (including `Hello` keep-alives) → disable all motors and stop streaming |
| Command | `cmd_ms` = 150 ms | connected but no fresh `Command` → damping hold instead of the stale setpoint |

Session behavior:

- **Boot:** load `/robot.json`, bring up every actuator, then disable all. No host, no
  output. If the config fails to load, no motor is ever enabled.
- **Connect:** the first host packet enables the motors and `Hello` triggers
  `RobotInfo`.
- **Connected, not commanding:** each actuator receives its damping hold
  (`Robot::sendHold`): for a DaMiao motor `kp = 0`, `kd = default_kd`, `t_ff = 0`
  (damped, not driven to a target); a Dynamixel axis keeps its last goal and torque
  state unchanged (a position servo already resists disturbance, whereas the DaMiao
  tuple would mean "drive to raw zero").
- **Connected, commanding:** each tick applies the latest setpoints.
- **Disconnect, crash or close:** the connection watchdog expires and all motors are
  disabled. The arms go limp; keep them supported.

The DaMiao motors' own comm-loss timeout is disabled (register 9 = 0), so without the
explicit disable above a motor would hold its last MIT command indefinitely. Possible
additions are re-enabling the motor timeout as a second layer, or a timed damped
ramp-down instead of an immediate limp.

The protocol intentionally exposes no mode switching, zeroing or ID provisioning; those
are done with the provisioning sketch or `scripts/dm_configurator.py`.

## 8. Quick reference

```text
robot defined by /robot.json on SD (actuators, buses, IDs, gains, net, rates); MIT ranges read from motors
Teensy UDP server: net.ip : net.port (default 5800); host on the same subnet
host -> Teensy : 0x01 Command | 0x02 HealthRequest | 0x03 Hello
Teensy -> host : 0x10 State   | 0x11 Health        | 0x12 RobotInfo (reply to Hello)
handshake      : Hello -> RobotInfo -> host builds its actuator table
State 400 Hz (cap 500) | conn watchdog 250 ms | cmd watchdog 150 ms | live window 50 ms
no host -> disabled | connected, idle -> damping hold | connected, commanding -> MIT
```

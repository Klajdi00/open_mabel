# Teleoperation wire protocol

Every command source (the Vision Pro app in `apps/visionos/`, the browser studios in
`web_gui/`, keyboard and gamepad clients, a learned policy) speaks one contract to the gateway
(`server/sim_teleop_bridge.py`). The contract is defined in `server/wire_protocol.py` and mirrored
in Swift (`apps/visionos/MabelTeleop/Networking/Protocol/WireEnvelope.swift`) and JavaScript
(`web_gui/*/web/js/`). Clients differ only in how the operator produces intent; all of them
receive the same `robot_state` stream and the same camera streams, and the same messages drive
the simulated or the physical robot.

The technical report `server/docs/teleop_protocol.pdf` contains the message-flow diagram and the
channel-rate analysis. The read-only data plane on ports 9600–9623 is specified separately in
[`data_streams.md`](data_streams.md).

## Endpoints

| Endpoint | Direction | Purpose |
|---|---|---|
| `ws://<host>:9090/teleop` | both | Commands up, `robot_state` down |
| `ws://<host>:9090/monitor` | down | Read-only fan-out of every client's command stream, for `server/monitor_gui.py` |
| `ws://<host>:9090/map`, `ws://<host>:9092/map` | down | Live SLAM map records ([`data_streams.md`](data_streams.md#binary-records)); the dedicated `:9092` socket is preferred |
| `http://<host>:8080/camera/<name>/stream.mjpg` | down | MJPEG video |
| `ws://<host>:8080/camera/<name>/h264` | down | H.264 video over WebSocket (browsers) |
| `udp://<host>:9091` | down | Latest-frame-wins JPEG video |
| `udp://<host>:9092` | down | H.264 video over UDP (LAN / Tailscale) |

Query parameters on the WebSocket URL:

- `client=<tag>` names the device to the gateway and the Command Monitor (`visionpro`, `web`).
  Without it the gateway falls back to the upgrade request's User-Agent and Origin.
- `key=<token>` is required on every WebSocket endpoint when the gateway runs with
  `MABEL_TOKEN` set; unset (the default) means open.

## Envelope

Every message is a two-layer JSON object:

```json
{ "type": "<message type>", "payload": { } }
```

Field names are the Swift `Codable` camelCase defaults, so nothing is renamed on the wire. All
angles are radians, the prismatic lift is in metres, and every pose is a row-major 4x4 matrix
(16 floats) in the ARKit world frame (x right, y up, −z forward). Unknown message types are
rejected at decode, so adding a type is an explicit change on both sides (`WireMessageType`).

## Message types

Client to server:

| Type | Payload | Meaning |
|---|---|---|
| `teleop_frame` | see below | Head and hand poses, mode, navigation joystick |
| `joint_command` | `{joints: {id: rad}}` | Direct joint targets |
| `control_mode` | see below | Teleoperation method, controller, region, channel switches |
| `recalibrate` | — | Re-anchor the retargeting clutch at the operator's current pose |
| `reset` | — | Return the robot to its home pose |
| `estop` | `{active: bool}` | Engage or release the emergency stop; on the real robot release also energises the hands |
| `external_force` | `{forces: {body: [fx, fy, fz]}}` | Operator push for compliance modes |
| `record` | `{active: bool, task?}` or `{select: task}` | Start or stop a demonstration episode (HDF5), or pick the task folder |
| `data_collect` | `{active: bool, label?}` | Start or stop a ROS 2 bag (MCAP) session |
| `demo_record`, `demo_replay`, `demo_delete`, `demo_list` | see `wire_protocol.py` | Record and replay command clips through the same controller |
| `nav_goal`, `nav_goal_preview`, `nav_cancel` | `{x, y, yaw?}` | Nav2 goal in the map/odom frame, a staged preview, or cancel |
| `list_maps`, `load_map`, `save_map`, `delete_map` | `{name}` | Saved-map library (`""` or `"live"` returns to live SLAM) |
| `body_calibration` | `{action: start\|cancel\|apply, user}` | Guided operator anthropometry capture |
| `policy_select`, `animation_select` | `{name: str\|null}` | Select (not run) a learned policy or scripted motion; echoed in `robot_state` |
| `operator_points` | `{on: bool}` | Opt in to the retargeting debug overlay |
| `audio_config` | `{micEnabled: bool}` | Mute or unmute the robot microphone (the audio itself rides the data plane) |
| `set_robot_name` | `{name}` | Rename the robot; persisted and pushed to all clients |
| `ping` | `{t}` | Liveness and round-trip measurement |

Server to client:

| Type | Meaning |
|---|---|
| `hello` | Handshake: robot identity, `kind` (`sim` or `real`), ports, video and head field-of-view parameters |
| `robot_state` | Realized robot state at 100 Hz (below) |
| `pong` | `{t, srv}`: echoes the client's `t`; the client displays `now − t` as round-trip latency |
| `nav_result` | `{status: reached\|unreachable\|aborted, msg}` |
| `map_list` | `{maps: [{name, has_mesh, has_reloc}], loaded}` |
| `demo_clips` | The command-clip library |
| `operator_points` | Wrist targets, elbow hints and hand skeletons the arm IK consumed, in the robot frame |
| `error` | Diagnostic string |

Clients choose the robot model to render from `hello` (`robot`, `kind`, and the per-side hand
version in `hands`), never from a renamable display name.

## `teleop_frame`

```json
{ "type": "teleop_frame",
  "payload": {
    "sequence": 412, "timestamp": 7.71e8,
    "head": { "transform": { "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] },
              "trackingState": "normal" },
    "leftHand": null, "rightHand": null,
    "mode": "Base Driving",
    "navJoystick": { "lx": 0.0, "ly": 0.8, "rx": -0.2, "ry": 0.0 } } }
```

- `head` is required; `trackingState` is `normal`, `limited` or `notAvailable`.
- A hand is `{chirality, anchorTransform, joints: [{joint, localTransform, isTracked}], isTracked}`
  with the 27 ARKit hand joints. The world pose of a joint is `anchorTransform @ localTransform`.
- `mode` is `Arms & Hands` or `Base Driving`.
- `navJoystick` axes are normalized to [−1, 1]: the left stick translates the base, the right
  stick yaws and drives the lift. `server/nav_axes.py` holds the exact conversion to a body twist.

A client drives manipulation by filling the hands and base motion by filling the joystick; the
message is the same either way. The gateway consumes frames newest-only (a one-slot buffer per
client), so control latency is bounded by the client's send period plus one control tick, never
by a queue.

## `control_mode`

```json
{ "type": "control_mode",
  "payload": { "method": "retarget", "controlType": "impedance", "region": "whole_body",
               "stiffness": 1.0, "scope": "global",
               "tracking": { "neck": true, "arms": true, "hands": true, "torso": true, "lift": true },
               "gravityComp": true, "wholeBodyRetarget": false } }
```

| Key | Values |
|---|---|
| `method` | `joint`, `navigation`, `wrist`, `retarget` |
| `controlType` | `impedance` (track the target, gravity-compensated) or `compliance` (yield to external force) |
| `region` | `arm`, `upper_body`, `whole_body` |
| `scope` | `local` (base driven by joystick only) or `global` (base joins the retargeting loop); absent means `local` |
| `tracking` | Per-channel switches; a channel switched off eases to a neutral pose. Keys are merged, so omitted keys keep their value |
| `gravityComp`, `wholeBodyRetarget` | Optional booleans, merged |

Payloads with only a legacy `mode` key are still accepted.

## `robot_state`

Sent per connection at 100 Hz from the plant's measured state (MuJoCo `qpos` in simulation,
`/joint_states` on the robot). It is the realized state, not an echo of the command, so clients
render a true mirror of the robot.

```json
{ "type": "robot_state",
  "payload": {
    "jointPositions": { "left_arm_1": 0.004, "torso": 0.0, "neck_yaw": 0.0, "lift_lower": 0.01,
                        "left_wrist": 0.15, "left_index_mcp": 0.15 },
    "base": { "x": 0.0, "y": 0.0, "z": 0.0, "yaw": -0.006, "quat": [1, 0, 0, 0],
              "steer": [0, 0, 0], "drive": [0, 0, 0] },
    "battery": { "available": false, "reason": "not applicable (sim)" },
    "latencyMs": 12.0, "stateAgeMs": 4.0, "sentAt": 1780000000.123,
    "estop": false, "controlMode": { }, "torque": { },
    "recording": { "task": "pick_place", "isRecording": false, "total": 6, "episodeIndex": null },
    "server": { "name": "MABEL Sim", "kind": "sim" } } }
```

- `jointPositions` covers arms, torso, neck, lift, swerve steering, wrists and every finger joint.
  Keys equal the joint ids in the app rigs; values are absolute MuJoCo `qpos`. Renderers must draw
  `qpos − ref` (the reference pose from `rig.json` or `mabel_joints.json`), which differs from zero
  only for the ORCA hand joints.
- `base.quat` is the full chassis orientation (`[w, x, y, z]`); `steer` and `drive` are the three
  swerve modules (front-left, front-right, back).
- `sentAt` is the server wall clock at send; only same-host clients can use it to measure one-way
  staleness.
- Additive keys (`estop`, `controlMode`, `torque`, `dataCollection`, `policy`, `animation`,
  `audio`, `modules`) are ignored by clients that do not know them.

Optional fields appear only on ticks with a fresh sample:

| Field | Rate | Contents |
|---|---|---|
| `lidar` | ~5 Hz | `{angleMin, angleInc, rangeMax, stamp, ranges[360]}`, base frame, ROS `/scan` compatible |
| `jointVelocities`, `jointTorques`, `jointTemps`, `motorHealth` | ~10 Hz | Same keys as `jointPositions`; only joints that are actually measured appear |
| `calibrationStatus`, `operatorProfile` | on change | Operator body-calibration progress and result |

## Video

Cameras are served by a process separate from the gateway (the simulated camera server or the
camera HAL), so a heavy decode never delays control. Stream names are listed in
[`../README.md`](../README.md#camera-streams).

**UDP-JPEG, `:9091` (`server/udp_video.py`).** The client sends `{"sub": ["main", ...]}` as a
JSON datagram every second (subscriptions expire after 3 s) and receives a stream-name-to-id map.
Each JPEG is split into datagrams of at most 1196 bytes, so they pass a 1280-byte WireGuard MTU
unfragmented, each with a 14-byte big-endian header:

```text
"MV" | ver u8 | stream u8 | frame_seq u16 | chunk u16 | chunks u16 | ts_ms u32 | payload (<= 1182 B)
```

The receiver reassembles per stream; a datagram of a newer frame (wrap-aware compare) evicts any
incomplete older frame. There are no retransmissions: a lost packet costs one frame instead of
stalling the stream behind TCP head-of-line blocking.

**H.264, `:9092` UDP and `/camera/<name>/h264` WebSocket.** The same codec on the simulated and
physical robot, so clients decode one stream type. Apps use the UDP path (LAN or Tailscale);
browsers, which cannot receive UDP, use the WebSocket path.

**MJPEG, `:8080`.** Standard `multipart/x-mixed-replace` at
`http://<host>:8080/camera/<name>/stream.mjpg`. Each subscriber holds a one-slot queue, so a
slow client drops frames instead of accumulating lag. The Vision Pro app falls back to MJPEG when
no UDP datagram arrives within 2.5 s and returns to UDP when datagrams resume.

**Head stereo.** `head_stereo` is one side-by-side frame (left eye, right eye) with both halves
from the same capture. Headsets should use it rather than pairing two independently paced eye
streams, which drift apart by whole frames.

## Simulated and physical robot

| | Simulation | Physical robot |
|---|---|---|
| Launch | `bash server/run.sh` | `bash server/real_run.sh` |
| Plant | `sim:bridge` (`simulation_bridge/`) or `sim:mujoco` (in-process) | `real:ros` (`ros/src/mabel_control/mabel_control/ros_plant.py`) |
| `robot_state` source | MuJoCo `qpos` | `/joint_states` and the per-subsystem topics |
| Command sink | `simulation_bridge` RPC or MuJoCo actuators | `/arms/command`, `/body/command`, `/<side>_hand/command`, `/mabel_cmd`, `/goal_pose` |
| Cameras | Rendered by `server/sim_camera_server.py` | Camera HAL in `hardware_bridge/` |

Joint keys, units and message schema are identical on both paths, so clients need no change to
switch between them; only the host differs.

## Protocol parameters

| Parameter | Value | Source |
|---|---|---|
| Command transport | `ws://<host>:9090/teleop` | `server/config.py` |
| `robot_state` rate | 100 Hz, newest-only per client | `server/sim_teleop_bridge.py` |
| Control loop | 50 Hz target (`MABEL_CONTROL_HZ`) | `server/sim_teleop_bridge.py` |
| Frame consumption | Newest-only, one slot per client | `server/sim_teleop_bridge.py` |
| UDP datagram / payload | ≤ 1196 B / ≤ 1182 B | `server/udp_video.py` |
| UDP subscription keepalive / expiry | 1 s / 3 s | `server/udp_video.py` |
| MJPEG fallback trigger | No UDP datagram for 2.5 s | `apps/visionos/MabelTeleop/Networking/UDPVideoReceiver.swift` |
| Discovery | Bonjour `_mabel-teleop._tcp` | `server/sim_teleop_bridge.py` |

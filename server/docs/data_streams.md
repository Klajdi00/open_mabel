# Data plane and binary records

Besides the `:9090` control socket ([`protocol.md`](protocol.md)) and the camera ports, the
gateway exposes a read-only data plane on dedicated ports 9600–9623. Port numbers are identical
in simulation and on the physical robot; only the process that binds each port differs. The
single source of the numbers is `server/data_streams.py`. `MABEL_DATA_STREAMS=0` disables the
whole plane (no listeners are opened; the control plane is unchanged).

The binary records used here are the same records the live SLAM map stream (`/map`) uses; they
are specified in [Binary records](#binary-records) below. The technical report
`server/docs/data_streams.pdf` contains the measured rates and the contention analysis.

## Rules

- **Read-only.** No 96xx port accepts a command. Commands, including starting a recording or
  selecting a policy, stay on `:9090`.
- **One duplex exception: voice on `:9610`.** It carries operator speech down to the robot
  speaker. This is allowed because audio has no actuation authority: it cannot move a joint, a
  wheel or a gripper and never reaches the `controller/` gate. Mute and unmute are control and
  stay on `:9090` as `audio_config`.
- **No depth images are streamed.** Cameras contribute RGB video and pointclouds only; in
  simulation depth is rendered internally to synthesize the clouds.
- **Every pointcloud is in the `odom` frame.** A cloud without a usable transform is dropped,
  never sent in the camera's optical frame (`map_stream.CloudFramer` is the single
  implementation).
- **Direct links only.** Every plane is dialled directly at the robot host's LAN or Tailscale
  address; nothing is exposed to the public internet.
- **An absent source answers with a notice, not a refused connection.** WebSocket planes send a
  JSON text frame `{"type": "notice", "available": false, "reason": ...}`; MJPEG planes answer
  HTTP 503 with the same body.
- **Same authentication as `/teleop`.** Every WebSocket plane enforces `?key=<MABEL_TOKEN>` when
  the token is set.

## Port table

| Port | Plane | Content | Simulation | Physical robot |
|---|---|---|---|---|
| 9600 | Telemetry (WebSocket, JSON) | proprio and control at 50 Hz, status at 5 Hz, lidar at ~5 Hz, collection and network at 1 Hz | gateway (MuJoCo `qpos`, `qvel`, `qfrc_actuator`; ray-cast lidar) | gateway (`joint_states`, hand temperatures, motor health, `/scan`) |
| 9601 | Pointcloud, head | CLOUD `0x05` | camera server (rendered depth) | gateway (ZED Mini depth points) |
| 9602 | Pointcloud, left wrist | CLOUD `0x05` | camera server | gateway (RealSense D405) |
| 9603 | Pointcloud, right wrist | CLOUD `0x05` | camera server | gateway (RealSense D405) |
| 9604 | Pointcloud, front base | CLOUD `0x05` | camera server | gateway (RealSense D435i) |
| 9605 | SLAM and 2-D map | MESH `0x01`, ODOM `0x02`, GRID `0x04`, PATH `0x03`, JSON status at 1 Hz | notice (SLAM runs on the robot only) | gateway (`slam_stream.py`) |
| 9610 | Voice (duplex) | AUDIO `0x07`, 16 kHz mono S16LE, 20 ms frames | notice (no sound card) | gateway (`audio_streams.py`) |
| 9620 | Video, head stereo | MJPEG side-by-side, the only high-resolution stream | camera server | camera HAL |
| 9621 | Video, left wrist | MJPEG, standard resolution | camera server | camera HAL |
| 9622 | Video, right wrist | MJPEG, standard resolution | camera server | camera HAL |
| 9623 | Video, front base | MJPEG, standard resolution | camera server | camera HAL |

Per-camera clouds default to 1 Hz and at most 6000 points (`MABEL_CLOUD_HZ`,
`MABEL_CLOUD_MAX_POINTS`, `MABEL_CLOUD_MAX_RANGE` in `server/cloud_streams.py`). The `hello` on
`:9090` and on `:9600` carries the full port map under `ports`.

## Telemetry messages (`:9600`)

One JSON object per text frame. Every frame has `type` and `t` (server Unix time). A missing
value is `null`; a missing subsystem is `{"available": false, "reason": ...}`. Joint keys are the
same ids as `robot_state.jointPositions`.

```jsonc
// on connect
{"type":"hello","t":0,"robot":{"name":"...","kind":"sim","server":"...","version":"..."},
 "schema":1,"ports":{},
 "rates":{"proprioHz":50,"controlHz":50,"statusHz":5,"lidarHz":5,"collectionHz":1,"networkHz":1}}

// 50 Hz: proprioception
{"type":"proprio","t":0,"stamp":0,
 "joints":{"left_arm_1":{"q":0.12,"dq":0.01,"tau":1.3,"temp":null}},
 "base":{"pos":[0,0,0],"quat":[1,0,0,0],"steer":[0,0,0],"drive":[0,0,0]}}

// 50 Hz: every control input into the robot
{"type":"control","t":0,"mode":{"method":"...","type":"...","region":"...","stiffness":1.0},
 "desired":{"<joint>":0.0},"baseTwist":{"vx":0,"vy":0,"wz":0},"ctrl":[],"engaged":true,"estop":false}

// 5 Hz: status
{"type":"status","t":0,"estop":false,"controlMode":{},
 "latency":{"cmdStalenessMs":0,"stateAgeMs":0},
 "battery":{"available":false,"reason":"not applicable (sim)"},
 "torque":{"left_hand":true,"right_hand":true,"arms":null,"base":null},
 "policy":{"selected":null,"running":false,"available":[]},
 "animation":{"selected":null,"playing":false},
 "driver":{"clients":1},"recording":{},"dataCollection":{}}

// ~5 Hz, only on a fresh sweep: 2-D lidar
{"type":"lidar","t":0,"angleMin":0,"angleInc":0.0175,"rangeMax":12,"stamp":0,"ranges":[]}

// 1 Hz: recordings on disk
{"type":"collection","t":0,"episodes":{"dir":"...","tasks":[{"name":"...","count":0}],"total":0},
 "bags":{"dir":"...","active":false,"count":0,"sessions":[]}}

// 1 Hz: this socket's transport
{"type":"network","t":0,"hostname":"...","ips":{"lan":"...","tailscale":"..."},
 "transport":"LAN","secure":false,"tokenGated":false,"portMap":{}}

// client ping, echoed
{"type":"ping","t":0}  ->  {"type":"pong","t":0,"serverT":0}
```

In simulation `dq` is `qvel`, `tau` is `qfrc_actuator` and `temp` is `null`. On the robot `dq`
and `tau` come from the hardware `joint_states`, and `temp` is the hand servo or arm driver
temperature in °C.

## Voice

Voice is `ws://<robot>:9610/audio` (LAN or Tailscale), served by `data_streams.serve_hub_ws`
on `:9610` and keyed only if `MABEL_TOKEN` is set. It lands on `audio_streams.AudioHub`. On the robot the gateway bridges it to the
HAL over ROS 2 (`/audio/mic`, `/audio/speaker`, `/audio/enable`, `/audio/state`,
`/audio/device_health`); `hardware_bridge/` is the only process that opens the sound card. In
simulation the plane serves a notice; it does not loop the operator's voice back.

Audio is a sequence, not a snapshot. Every other hub here is latest-wins; the audio hub keeps a
bounded FIFO and, on overflow, drops the whole backlog to resynchronize, which the client sees
as one `seq` jump. Frames whose rate, channel count or codec differ from what the server
advertised in `hello` are refused, never resampled.

Diagnostics:

```bash
python3 server/tools/exp_data_streams.py --probe      # the plane, including duplex voice
python3 server/tools/exp_audio_duplex.py              # voice round trip
ros2 topic echo /audio/device_health                  # the ROS leg (robot)
```

## Binary records

Every binary WebSocket frame on the map stream (`/map` on `:9090` and `:9092`), the raw LiDAR
socket (`:9093`) and the 96xx binary planes carries exactly one record, tagged by its first
byte. All fields are little-endian. Text frames are JSON (`hello`, `notice`, status) and are
distinguished by the WebSocket frame type. The producer is `server/map_stream.py`; reference
decoders are `web_gui/control_center/web/js/maplink.js` and
`apps/visionos/MabelTeleop/Networking/MapStreamLink.swift`.

All positions are in the `odom` frame (REP-103: x forward, y left, z up). Clients rebase to their
own up-axis; producers must not pre-rotate.

| Tag | Record | Replaces client state | Default rate |
|---|---|---|---|
| `0x01` | MESH | yes (snapshot) | 0.5 Hz (`MABEL_MAP_MESH_HZ`) |
| `0x02` | ODOM | yes | ≤ 30 Hz (`MABEL_MAP_ODOM_HZ`) |
| `0x03` | PATH | yes | ≤ 5 Hz (`MABEL_MAP_PATH_HZ`) |
| `0x04` | GRID | yes | ≤ 2 Hz (`MABEL_MAP_GRID_HZ`) |
| `0x05` | CLOUD | yes | ≤ 2 Hz on `/map` (`MABEL_MAP_CLOUD_HZ`) |
| `0x06` | SCAN | yes | ≤ 5 Hz (`MABEL_MAP_SCAN_HZ`) |
| `0x07` | AUDIO | no (sequence) | 50 frames/s |

### Vertex layout (MESH and CLOUD)

16 bytes per vertex: `f32 x, f32 y, f32 z, u8 r, u8 g, u8 b, u8 a` (alpha unused, 255).

### MESH (`0x01`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x01` |
| 1 | u8 | flags: bit0 = final chunk, bit1 = an index buffer follows |
| 2 | u16 | reserved |
| 4 | u32 | vertexCount |
| 8 | 16 B × n | vertices |
| … | u32 | indexCount (only if bit1), = 3 × triangles |
| … | u32 × n | indices into this chunk's vertices (only if bit1) |

A snapshot too large for one frame is sent as several frames with bit0 = 0 and a final frame with
bit0 = 1; the client swaps the accumulated snapshot in on the final frame. Indices are
chunk-local; the client rebases them by its running vertex count. The producer emits a triangle
soup (three vertices per face) from `nvblox_msgs/Mesh`, or vertices only (bit1 = 0, rendered as a
point cloud) when that message type is unavailable. Caps: `MABEL_MAP_CHUNK` (60000 points per
frame), `MABEL_MAP_MAX_POINTS` (400000), `MABEL_MAP_MAX_TRIS` (15000).

### ODOM (`0x02`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x02` |
| 1 | u8 × 3 | reserved |
| 4 | f32 × 3 | x, y, z (m) |
| 16 | f32 × 4 | qx, qy, qz, qw (ROS order) |

32 bytes. Source: the odometry topic (`MABEL_MAP_ODOM_TOPIC`, default `/odom`).

### PATH (`0x03`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x03` |
| 1 | u8 × 3 | reserved |
| 4 | u32 | count (0 clears the route) |
| 8 | f32 × 3 × n | waypoints x, y, z |

The Nav2 plan (`/plan`, published in `map`) composed into `odom`. The goal travels the other way,
as a `nav_goal` message on `:9090`.

### GRID (`0x04`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x04` |
| 1 | u8 | flags (reserved) |
| 2 | u16 | reserved |
| 4 | u32 | width (cells) |
| 8 | u32 | height (cells) |
| 12 | f32 | resolution (m/cell) |
| 16 | f32 × 3 | origin x, y, z of cell (0, 0) in `odom` |
| 28 | f32 | origin yaw (rad) |
| 32 | u8 × w·h | cells, row-major, x fastest |

Cell values: 0 free to 100 occupied, 255 unknown (ROS −1). Source: `nav_msgs/OccupancyGrid`
(`MABEL_MAP_GRID_TOPIC`, default `/map`).

### CLOUD (`0x05`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x05` |
| 1 | u8 | flags: bit0 = final chunk |
| 2 | u16 | reserved |
| 4 | u32 | vertexCount (0 in a single final chunk clears the overlay) |
| 8 | 16 B × n | vertices |

A live sensor cloud that the client replaces every frame (as opposed to the accumulated MESH).
Chunked like MESH. On `/map` the source is `MABEL_MAP_CLOUD_TOPIC` (default
`/camera/front/depth/points`), capped at `MABEL_MAP_CLOUD_MAX_POINTS` (40000).

### SCAN (`0x06`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x06` |
| 1 | u8 | version (0) |
| 2 | u16 | reserved |
| 4 | u32 | n (ray count) |
| 8 | f32 | angle_min (rad; ray i at angle_min + i · angle_increment, CCW) |
| 12 | f32 | angle_increment (rad) |
| 16 | f32 | range_min (m) |
| 20 | f32 | range_max (m) |
| 24 | f32 × 2 | lidar mount x, y in `base_link` (m) |
| 32 | f32 | lidar mount yaw (rad) |
| 36 | u16 × n | ranges in millimetres (0 = no return) |

Rays are in the lidar frame: `p_world = T_odom_base · T_base_lidar · [r cos θ, r sin θ]`, with
`T_odom_base` from ODOM. Sent inline on `/map` and alone on the raw LiDAR socket
`ws://<host>:9093` (`MABEL_LIDAR_PORT`), which replays the latest sweep on connect.

### AUDIO (`0x07`)

| Offset | Type | Field |
|---:|---|---|
| 0 | u8 | type = `0x07` |
| 1 | u8 | codec: 0 = PCM16 (implemented), 1 = Opus (reserved) |
| 2 | u16 | channels |
| 4 | u32 | seq (wraps at 2³²; the gap detector) |
| 8 | u32 | rate (Hz) |
| 12 | u32 | nBytes |
| 16 | … | payload (PCM16: interleaved S16LE) |

The only record that travels in both directions. Encoder and decoder: `map_stream.encode_audio`
and `decode_audio`. Apple and Web Audio playback must convert S16 samples to Float32
(`sample / 32768`) before scheduling.

### Map stream configuration

The map producer runs inside the gateway when it can import `rclpy`; otherwise `/map` stays idle.
On the robot `server/perception_server.py` runs it in a separate process. In simulation the
gateway feeds `/map` itself with the robot pose, the front-camera cloud and the lidar sweep, only
while a client is connected.

| Variable | Default | Meaning |
|---|---|---|
| `MABEL_MAP_STREAM` | `1` | `0` disables the producer |
| `MABEL_MAP_PORT` | `9092` | Dedicated map WebSocket port |
| `MABEL_MAP_MESH_TOPIC` | `/nvblox_node/mesh_pointcloud` | Mesh source |
| `MABEL_MAP_ODOM_TOPIC` | `/odom` | Pose source |
| `MABEL_MAP_PLAN_TOPIC` | `/plan` | Planned path |
| `MABEL_MAP_GRID_TOPIC` | `/map` | Occupancy grid |
| `MABEL_MAP_CLOUD_TOPIC` | `/camera/front/depth/points` | Live cloud |
| `MABEL_MAP_SCAN_TOPIC` | `/scan` | LiDAR |
| `MABEL_MAP_FRAME` | `odom` | Frame the records are expressed in |

## Measuring

```bash
python3 server/tools/exp_data_streams.py --probe        # rates, notices, duplex voice
python3 server/tools/exp_data_streams.py --contention   # load matrix across all planes
python3 server/tools/exp_data_streams.py --figures      # figures for docs/data_streams.pdf
python3 server/map_latency_probe.py                     # /map throughput and latency
```

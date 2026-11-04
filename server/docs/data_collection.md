# Data collection

The gateway records demonstrations in two formats:

| Format | Recorder | Used for |
|---|---|---|
| Curation HDF5 (`episode_XXXX.h5`) | `server/curation_recorder.py` (in the gateway), `server/real_curation_collector.py` (standalone, real robot) | Curation and policy training (`robot_learning/`) |
| ROS 2 bag (MCAP) | `server/data_collection.py` | Raw archive of every topic; replay and inspection in Rerun |

Both formats record locally; recording directories are git-ignored. The curation and training
steps that follow are described in [`../../robot_learning/PIPELINE.md`](../../robot_learning/PIPELINE.md). The
technical report is `server/docs/data_collection.pdf`.

## Episodes from the gateway (`record`)

A client starts and stops an episode with the `record` message on `:9090`
(`{"active": true, "task": "<task>"}` / `{"active": false}`; `{"select": "<task>"}` picks the
task folder). The gateway writes

```text
<episodes-dir>/<task folder>/episode_XXXX.h5   (+ dataset_meta.json)
```

with `--episodes-dir` defaulting to `dataset/simulation/ios` under the repo root (override with
`MABEL_EPISODES_DIR`; Vision Pro clients are routed to the sibling `vision_os` folder) and the sample rate set by `--record-fps` (default 10 Hz,
`MABEL_RECORD_FPS`). The HDF5 writer is
`robot_learning/imitation_learning/data_collection/recorder.py`, the same writer every MABEL recorder
uses, so the folder is a Data Curation dataset as-is. Keys include `time/sim`, `time/wall`,
`obs/cam/<name>/rgb` (JPEG), `obs/cam/<name>/depth` (uint16 mm), `obs/lidar`,
`obs/joint_pos|joint_vel|joint_torque`, `obs/qpos|qvel`, `obs/pose/*`, `obs/base_twist`,
`action/ctrl`, and the active control configuration under `obs/control/*`.

Each episode also produces a replayable command clip in the `demo_list` library, named after the
episode. Camera capture runs off the control loop (`server/record_pump.py`), so recording does
not stall the simulation.

## Real-robot collector

`server/real_curation_collector.py` is a standalone ROS 2 node. It subscribes to every live
camera, depth, lidar, joint-state, command and odometry topic, latches the newest message of
each, and appends one frame per tick of a fixed clock to a curation-format episode:

```text
<root>/<client>/<TASK>_<NNN>_<YYYYMMDD>_<HHMMSS>/episode_XXXX.h5
```

with `<root>` defaulting to `real/` under the repo root and `<client>` one of `ios`,
`vision_os`, `real`.

- **Every topic is optional.** A missing camera, an unpowered hand or no lidar drops that channel;
  the episode still records.
- **One clock.** Images, proprioception and actions are all sampled on the same `--fps` grid
  (default 15 Hz). Record actions at the rate the policy will run.
- **Small on disk.** RGB is JPEG at `--rgb-max` (640 px long side), depth is uint16 millimetres at
  `--depth-max` (320 px), JPEG quality `--jpeg-quality` (88).
- **High-rate side stream** (`--hr-sidestream`, on by default): native-rate timestamped
  proprioception and actions in `episode_XXXX_hr.npz`, for analysis only.
- **Control** is a WebSocket (`--control-port`, default 9095) that accepts the same `record`
  message shape as the gateway.
- **DDS.** The collector defaults `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` to match the HAL stack;
  source `ros/scripts/mabel_env.sh` so `CYCLONEDDS_URI` points at `ros/scripts/cyclonedds.xml`.
  Run exactly one collector at a time.

### Runbook

1. Bring up sensing without energising any motor:

   ```bash
   cd hardware_bridge
   env MABEL_HW_CAMERAS=true MABEL_HW_LIDAR=true \
       MABEL_HW_BASE=false MABEL_HW_ARMS=false MABEL_HW_BODY=false \
       MABEL_HW_HANDS=false MABEL_HW_LIFT=false MABEL_HW_AUDIO=false \
       MABEL_HW_CAM_WRIST_D405_DEPTH=1 ./run_bridge.sh &
   ```

   For live joint feedback with the joints still limp, add `MABEL_HW_ARMS=true MABEL_HW_HANDS=true`.

2. Start the collector:

   ```bash
   source ros/scripts/mabel_env.sh
   python3 server/real_curation_collector.py --client visionos --control-port 9095 &
   ```

3. Collect: point the client's data-collection control at `ws://<robot>:9095`. For a dry run that
   never moves the robot, inject synthetic command targets and drive the start/stop button from a
   script:

   ```bash
   python3 server/tools/mock_control_publisher.py &
   python3 server/tools/mock_vp_collect.py --client visionos --task pick_place --duration 60 --count 10
   ```

4. Curate, convert and train: see [`../../robot_learning/PIPELINE.md`](../../robot_learning/PIPELINE.md).

## ROS 2 bags (`data_collect`)

The `data_collect` message (`{"active": true, "label": "..."}`) makes the gateway run
`ros2 bag record` on a curated topic set (`server/data_collection.py`): TF, all measured joint
states, the operator command topics (`/mabel_cmd`, `/arms/command`, `/body/command`,
`/<side>_hand/command`, `/goal_pose`, `/cmd_vel`), control-mode status, odometry, IMU, camera
color, depth, point clouds and `camera_info`, `/scan`, `/map` and costmap, the nvblox mesh and
`/plan`. Topics that are not being published are simply absent from the bag.

Sessions are written to `recordings/<YYYYmmdd_HHMMSS>_<label>/` under the repo root (MCAP,
`metadata.yaml` and a `session.json` with the commit, topic list and timing); override with
`MABEL_RECORDINGS_DIR`. `MABEL_BAG_ALL=1` records every topic; `MABEL_BAG_TOPICS="/a /b"`
replaces the list. Stopping sends SIGINT so rosbag2 finalizes the file.

The gateway needs a ROS 2 environment for this; in simulation use `bash server/sim_ros_server.sh`,
which runs the gateway with the ROS bus. To record without a client:

```bash
python3 server/data_collection.py my_label        # records until Ctrl-C
```

### Viewing in Rerun

```bash
python3 server/bag_to_rerun.py recordings/<session>            # writes <session>/rerun.rrd
rerun recordings/<session>/rerun.rrd
python3 server/bag_to_rerun.py recordings/<session> --spawn    # stream into a live viewer
```

The converter rebuilds the TF tree as the Rerun entity hierarchy (resolving the complete tree
before logging, so no frame is stranded at the origin), draws the robot links from the
`mabel_description` URDF, and maps images, camera intrinsics, point clouds, the nvblox mesh, the
occupancy grid, laser scans, the planned path, goals and every joint, IMU and twist signal to
Rerun types.

| Flag | Effect |
|---|---|
| `--no-robot`, `--urdf PATH` | Omit the robot body, or use another URDF |
| `--no-depth` | Skip depth images (keep point clouds) |
| `--max-range M` | Clip depth points beyond M metres (default 8; 0 keeps all). Far-range outliers otherwise make the scene bounds, and the auto camera, jump between frames |
| `--point-size P` | Point size in screen-space UI points (default 1.5) |
| `--frustum M` | Camera frustum size (default 0.25 m) |
| `--exclude SUBSTR ...` | Skip matching topics |

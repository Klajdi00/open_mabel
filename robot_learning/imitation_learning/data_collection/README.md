# Whole-body teleoperation data collection

Episode loggers for whole-body MABEL imitation learning. The operator drives the robot
through the shared whole-body teleop controller (the same one the live simulator uses), and
every frame logs everything the platform exposes: all cameras, LiDAR, position, velocity and
torque of every joint, base twist, end-effector and head poses, and the target command for
every actuator. Simulated and real episodes share one HDF5 schema.

Real-robot collection through the teleop gateway uses `server/real_curation_collector.py`
instead; see [`../../PIPELINE.md`](../../PIPELINE.md). The tools here cover simulation,
ROS-recorded simulation, and a template for a standalone real-robot logger.

## Layout

| File | Role |
|---|---|
| `config.py` | recording configuration; joint, camera and sensor names come from the model and `controller/mabel/config.py` |
| `providers.py` | `SimStateProvider` / `RealStateProvider`: one per-frame observation/action dict for both |
| `recorder.py` | `EpisodeRecorder`: accumulate frames and save one HDF5 per episode on a background thread |
| `collect_sim.py` | interactive simulation collection with a Vision Pro (or synthetic) operator |
| `collect_fast.py` | unattended scripted simulation collection straight to HDF5 |
| `collect_ros.py` | unattended scripted simulation published on ROS 2 and recorded as MCAP rosbags |
| `collect_real.py` | real-robot template with the same schema and keys; hardware hooks marked `# TODO` |
| `inspect_episode.py` | print schema, shapes and achieved logging rate, or replay all cameras tiled |
| `visualize_dataset.py` | per-episode diagnostic PDFs (joints, end-effector paths, base) |
| `merge_dataset.py` | pool several dataset folders, renumbering episodes |
| `tools/bench_collection.py` | headless benchmark that regenerates the report's numbers and figures |
| `docs/data_collection.pdf` | data model, benchmark and sim-to-real notes |

## Collect in simulation

The Vision Pro hand/head stream is published over ZMQ by
`simulation/mabel_mujoco/scripts/vision_pro_stream.py` (requires the `avp_stream` package and
the Tracking Streamer app on the headset):

```bash
python3 simulation/mabel_mujoco/scripts/vision_pro_stream.py --ip <headset-ip>   # binds tcp://*:5556
```

Then, from this folder:

```bash
python3 collect_sim.py --name pick_cube_v1 --task "pick the blue cube"
python3 collect_sim.py --name pick_cube_v1 --mock                          # synthetic operator, no headset
python3 collect_sim.py --name pick_cube_v1 --addr tcp://<publisher-host>:5556
```

Episodes are written to `datasets/<name>/episode_XXXX.h5`; numbering resumes from what is
already in the folder. `--no-randomize` fixes the manipuland pose.

Operator keys (captured globally with `pynput`; the viewer need not have focus):

| Key | Action |
|---|---|
| `C` | (re)calibrate the teleop clutch at the current operator pose |
| `M` | toggle manipulation / navigation |
| `B` | toggle lazy base-follow |
| `Space` | start a new episode / stop and save the current one |
| `X` | discard the episode in progress |
| `R` | reset the robot to home and re-randomize the manipuland |
| `Q` / `Esc` | quit (an unsaved episode is discarded) |

### Unattended collection

Both scripts drive the home scene (`simulation/mabel_mujoco/scenes/mabel_home.xml`) with
scripted synthetic-operator motion: five task labels × ten episodes by default. The motion is
task-shaped but grasps are not guaranteed to succeed; these datasets exercise the logging,
curation and training pipeline end to end.

```bash
python3 collect_fast.py                            # -> robot_learning/processed_dataset/simulated/
python3 collect_fast.py --tasks 1 --episodes 2     # quick test
python3 collect_ros.py                             # -> recordings/sim_home/<task>/episode_XXXX (MCAP)
python3 collect_ros.py --no-publish                # bags only, no live ROS graph
```

`collect_fast.py` logs only the two wrist cameras (RGB) at a 4 ms physics step for throughput.
`collect_ros.py` needs a ROS 2 environment (`source ros/scripts/mabel_env.sh`); it publishes
`/joint_states`, `/scan`, `/odom`, `/tf`, the wrist cameras and `/action/joint_targets` while
recording.

## Collect on the real robot (template)

`collect_real.py` has the same schema, keys and flags (`--rate` sets the logging rate). Fill the
`# TODO` hooks in it and in `providers.RealStateProvider` (motor bus, cameras, LiDAR, IMU) to
use it. Keeping the keys identical to simulation lets a policy move between the two without
remapping its inputs and outputs.

## Episode schema (HDF5, axis 0 = frame)

Observations (`obs/…`):

- `cam/<name>/rgb`: JPEG blobs (variable-length `uint8`); `cam/<name>/depth`: `uint16`
  millimetres, clipped at 8 m. Cameras: `l_wrist_cam`, `r_wrist_cam`, `zed_left`, `zed_right`,
  `body_cam`, `body_front_cam`, `body_back_cam` (depth on a subset; see `config.CAMERAS`).
- `lidar`: `[360]` planar ranges (m), 1° resolution.
- `joint_pos`, `joint_vel`, `joint_torque`: `[59]`, aligned with the actuator list.
- `qpos`, `qvel`: full generalized state (simulation ground truth).
- `pose/{chassis,head,left_palm,right_palm}`: `[7]` world position + quaternion.
- `base_twist`: `[3]` measured `(vx, vy, wz)`.

Actions and operator intent:

- `action/ctrl`: `[59]` target command for every actuator (arm, wrist, finger, neck, lift and
  torso positions; base steer angles and drive velocities). This is the training label.
- `teleop/{left,right}_palm_target` `[7]`, `teleop/grasp` `[2]`, `teleop/base_cmd` `[3]`,
  `teleop/look_at` `[3]`, `teleop/mode` (0 = manipulation, 1 = navigation).

Timing: `time/sim`, `time/wall`. Actuator names and order, the camera list and units are stored
in the file attributes (`schema_json`) and in `dataset_meta.json`.

## Inspect, visualize, merge

```bash
python3 inspect_episode.py datasets/pick_cube_v1/episode_0000.h5          # schema + timing
python3 inspect_episode.py datasets/pick_cube_v1/episode_0000.h5 --play   # tiled camera replay
python3 visualize_dataset.py datasets/pick_cube_v1                         # PDFs -> docs/figures/
python3 merge_dataset.py --out datasets/pick_cube_all datasets/pick_cube_v1 datasets/pick_cube_v2
```

## Benchmark and figures

`tools/bench_collection.py` runs the whole stack headlessly on the synthetic operator, prints
achieved log rate, per-frame render time, episode size and per-modality byte breakdown, and
writes `docs/figures/bench_{joints,ee,base}.pdf`:

```bash
python3 tools/bench_collection.py --frames 60 --seed 0
python3 tools/bench_collection.py --frames 60 --cameras l_wrist_cam,r_wrist_cam   # camera subset
```

Rendering is the bottleneck: on a CPU-only laptop (software GL), seven cameras at 480×270
sustain about 2 Hz; a GPU offscreen context (EGL) or fewer cameras reaches the 15 Hz target.

## Tuning

`config.py`: `LOG_FPS` (15), `RGB_W`/`RGB_H` (480×270), per-camera `rgb`/`depth` flags,
`JPEG_QUALITY` (90), `DEPTH_MAX_M` (8.0).

## Requirements

`mujoco`, `numpy`, `opencv-python`, `h5py>=3.11`, `pyzmq`, `pynput`, `matplotlib`; the `mabel`
controller package from `controller/`.

# Simulation Control Center

The simulated twin of [`control_center/`](../control_center/README.md): the same studio (3-D
viewer, teleop client, server monitor), with everything on the page coming from the MuJoCo
simulation. It runs on macOS or Linux, needs no robot and, on macOS, no ROS. Port **8097**.

```bash
bash web_gui/simulation_control_center/run.sh              # simulation (home scene) + studio
bash web_gui/simulation_control_center/run.sh --viewer     # also open the native MuJoCo window
bash web_gui/simulation_control_center/run.sh -s kitchen   # start in another scene (name or .xml)
bash web_gui/simulation_control_center/run.sh --gui-only   # attach to a simulation already running
bash web_gui/simulation_control_center/run.sh --mock       # studio only, synthetic data
```

`server/run.sh` also starts it. One Ctrl-C stops the studio and the simulation it launched.

## Architecture

```text
                     simulation_bridge/  MuJoCo HAL, the only mj_step
                                ▲
browser ──ws :9090/teleop ──▶ gateway (server/sim_teleop_bridge.py)   control + robot_state @ 50 Hz
        ──ws :9090/monitor ─▶ gateway                                 clients, driver, e-stop
        ──http :8080 ───────▶ sim camera server                       MJPEG head, front and wrist renders
        ──:8097 ────────────▶ this backend
                               /ws/telemetry   all joints + targets + scan
                               /ws/map         point cloud, odometry, SLAM grid, nav path
                               /ws/simview     the MuJoCo viewer in the browser
                               /api/sim, /api/nav, /api/scenes, /api/episodes, /api/actions
```

The sim/real seam is `server/plant.py`: the simulation plant is the twin of the ROS plant
the real robot uses, and clients see only `kind: "sim"` in `hello`. See
[`simulation_bridge/README.md`](../../simulation_bridge/README.md).

The backend supervises the simulation stack (`server/start_teleop.sh`, the same stack
`server/run.sh` launches), so the scene switcher can restart it into another world. If a
gateway is already running on :9090, the backend attaches to it instead and scene switching
is disabled.

- **Telemetry** is a passive client of the gateway's own `robot_state`: every joint value is
  the actual MuJoCo state. Velocities are finite-differenced; efforts and temperatures are
  left blank because the simulation does not model them.
- **Map plane.** The backend re-poses the scene from the gateway's shared state and derives
  the LiDAR scan (the model's rangefinder ring), chassis odometry, and a colored head-depth
  point cloud, using the record format of `server/map_stream.py`, served at `/ws/map`.

## Navigation without ROS

macOS cannot run Nav2 or `slam_toolbox`, so the backend runs a native stack
(`server/macnav/`): particle-filter grid SLAM over swerve wheel odometry fused with the
simulated IMU gyro, A* on an inflated costmap, and an MPPI local controller, using the same
motion model as the teleop stack. Place a goal in the viewer and start it; the 2-D map
layer shows the live grid and the path layer the active plan.

On Ubuntu with a built `ros/` workspace, `run.sh` launches `server/sim_ros_server.sh`
instead (visual SLAM, nvblox, Nav2) and the ROS map plane on :9092.

## Teleop

- **Webcam teleop** imitates the Vision Pro path with a desktop webcam (MediaPipe pose and
  hands): arms, fingers and head-gaze neck always; torso lean when the torso is visible;
  whole body when the legs are visible. A chip shows the region being controlled.
- **Keyboard / mouse teleop**: sticks and WASD.
- **Dexterity panel** for both ORCA hands: per-joint sliders, hand poses, and two-hand
  webcam tracking.

## Simulation-only controls

- **Reset** returns the world to its initial state.
- **Reset elsewhere** picks a collision-free spawn (at least 0.8 m clearance and 1 m from
  the current pose), verifies it in MuJoCo, and respawns the robot there.
- **Scene library** from `simulation/mabel_mujoco/scenes/scenes.json`, with thumbnails;
  switching restarts the supervised simulation in that world.
- **Record / Stop / Save** shows length, frame count, rate, every modality (RGB cameras,
  stereo depth, `qpos`, `qvel`, `ctrl`, LiDAR, odometry, teleop commands) and size, and lets
  you rename or discard the episode before committing it.

## Ports and configuration

| Port | Service |
|---|---|
| 8097 | this backend and studio (`MABEL_SCC_PORT`) |
| 9090 | gateway, `/teleop` and `/monitor` (`MABEL_WS_PORT`) |
| 8080 | simulated cameras, MJPEG (`MABEL_HTTP_PORT`) |
| 9091 | UDP JPEG video for headsets |
| 9092 | ROS map plane (Ubuntu); with `MABEL_MAP_PORT=0` the map is served same-origin at `/ws/map` |

Other variables: `MABEL_SCC_SOURCE` (`auto | sim | ros | mock`), `MABEL_SCC_MANAGE_SIM`
(`auto | 1 | 0`), `MABEL_SCC_VIEWER=1`, `MABEL_SCENE`, `MABEL_EPISODES_DIR` (episode
folder, shared with the gateway).

Served over https from a single origin (e.g. `tailscale serve`), the page routes teleop and
video through the backend's same-origin `/ws/*` and `/api/cam` proxies (`?proxy=1/0` forces
the mode).

## Documentation

Technical reports in `docs/`: `simulation_control_center.pdf` (the studio and its sim/real
unification), `macnav.pdf` (native SLAM and MPPI navigation), `retargeting.pdf` (webcam
full-body retargeting), `choreography.pdf` (the choreography timeline and motion library).

# simulation_bridge — the MuJoCo HAL

The one process that owns the simulated robot (the only caller of `mj_step`), mirroring `hardware_bridge/` for the real robot. The gateway, ROS, cameras and studios are clients; switching the gateway plant between `real:ros` and `sim:bridge` changes nothing above it. See the root [Architecture](../README.md#architecture).

## Run

Usually started for you by `bash server/run.sh`. Standalone, from the repo root:

```bash
python3 -m simulation_bridge.simulation_bridge                  # default scene: simulation/mabel_mujoco/scenes/mabel_home.xml
python3 -m simulation_bridge.simulation_bridge --scene S.xml --instance sim1
./simulation_bridge/run_bridge.sh                               # same, interpreter from MABEL_PY
python3 -m pytest simulation_bridge/tests -q
```

Flags: `--scene`, `--instance` (default `sim0`; scopes lock, memmap, RPC port, ROS namespace), `--rpc-port`, `--ns` (default `/sim`), `--no-takeover` (refuse instead of replacing an existing owner).

## Interfaces

| Interface | Detail |
|---|---|
| State out | memmap `/tmp/mabel_teleop_state.f64` = `[time, qpos, qvel, ctrl]` (other instances `/tmp/mabel_sim_<instance>.f64`) |
| Commands in | newline JSON RPC on `127.0.0.1:8788`: `set_twist`, `set_estop`, `set_targets`, `set_effort`, `set_wrench`, `set_group_torque`, `set_speed_caps`, `reset`, `set_state`, `state` |
| ROS 2 (Linux) | the `hardware_bridge` DDS contract under `/sim` (full list in `ros_io.py`) |

Same safety semantics as the robot: base twist deadman 0.5 s, `set_estop` zeroes the base, disabled limbs go limp. Settings: `config.py` (`MABEL_SIM_*`).

| File | Role |
|---|---|
| `simulation_bridge.py` | entry point, instance lock |
| `engine.py` | physics loop, command latching, memmaps |
| `dynamics.py` | scripted scene actors (`.dynamics.json`) |
| `rpc.py`, `ros_io.py` | RPC server/client, `/sim` ROS mirror |

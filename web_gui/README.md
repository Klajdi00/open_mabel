# web_gui — browser studios

Every studio is a client of the same gateway contract (`server/wire_protocol.py`, WebSocket :9090) as the apps, so all motion is still gated by `controller/`.

| Studio | Port | Run | What it is |
|---|---|---|---|
| [`control_center/`](control_center/) | 8094 | `server/real_run.sh` | real-robot studio: 3-D view, teleop (keyboard, gamepad, webcam), cameras, map |
| [`simulation_control_center/`](simulation_control_center/) | 8097 | `bash server/run.sh` | same studio on the MuJoCo twin, with scenes and navigation |
| [`control_lab/`](control_lab/) | 8115 | `bash web_gui/control_lab/run.sh` | bench for the `mabel` control library |
| [`retarget_lab/`](retarget_lab/) | 8120 | `bash web_gui/retarget_lab/run.sh` | record Vision Pro motion, compare retargeters |
| [`setup_studio/`](setup_studio/) | 8300 | `bash web_gui/setup_studio/run.sh [--sim]` | hardware bring-up: per-module status, torque, calibration into `mabel.json` (talks to `mabel_hw` directly; stop the HAL first) |

`shared_assets/` holds the web robot rig generated from the MJCF (don't hand-edit it).

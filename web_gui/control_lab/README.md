# Control Lab

Test bench for the `mabel` control library. MuJoCo runs in-process around the same `MabelTeleop` facade the gateway uses, with a live 3-D view and measured plots (base twist vs. safe twist, tip usage, tracking error, torques, wrench estimates, IK timing).

## Run

```bash
bash web_gui/control_lab/run.sh                 # http://localhost:8115
bash web_gui/control_lab/run.sh -s <scene.xml>  # other scene
bash web_gui/control_lab/run.sh --no-viewer     # no native MuJoCo window (needs mjpython on macOS)
cd web_gui && python3 -m pytest control_lab/tests -q
```

The lab also binds the teleop port **:9090**, so the Vision Pro app connects to it directly. Stop any other gateway (`server/run.sh`) first.

Top bar: **Method** (Joint, Wrist, Drive through `safe_base_step`, Teleop), **Region** (arm, upper, whole body), **Stiff/Soft**, **Torque** (arms start limp; Enable Torque holds the current pose), **Input** (Off, Device, ZMQ Stream, Mock operator).

Config: `MABEL_CTS_PORT` (8115), `MABEL_CTS_VP_PORT` (9090), `MABEL_CTS_SCENE` (default `simulation/mabel_mujoco/scenes/mabel_bare.xml`), `MABEL_CTS_TIMESTEP` (0.004), `MABEL_CTS_AVP_ADDR`. `MABEL_NO_BROWSER=1` stops it opening a browser tab.

Experiments: `experiments/exp_lab_benches.py`, `exp_wrench_est.py`, `exp_wrist_edgecases.py`. Report: [`docs/control_lab.pdf`](docs/control_lab.pdf).

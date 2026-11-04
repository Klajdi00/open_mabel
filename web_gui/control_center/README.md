# Control Center

The real-robot web studio: 3-D visualizer, teleop client and server monitor in one page,
served from the robot computer on **:8094** and reachable over the LAN or Tailscale. Its
simulated twin is [`simulation_control_center/`](../simulation_control_center/README.md).

```bash
bash web_gui/control_center/run.sh                       # ROS tap if available, else mock plant
MABEL_CC_SOURCE=mock bash web_gui/control_center/run.sh  # no robot: synthetic plant
```

`server/real_run.sh` starts it; `install_service.sh` installs a systemd user service. Served
over https from one origin (e.g. `tailscale serve`) it uses same-origin proxies (`?proxy=1/0`).

**Architecture.** The page speaks the gateway's wire contract (`server/wire_protocol.py`,
ws `:9090/teleop` + `/monitor`), the map plane (`:9092`) and the camera server (`:8080`,
H.264 with MJPEG fallback). This backend (`server/`, FastAPI) serves `web/`, the rig from
`web_gui/shared_assets/` under `/website-assets`, and `/ws/telemetry` (every joint, temps,
e-stop, host stats) from ROS taps. The studio never publishes to a command topic.

**Studio.** Live rig with TF / LiDAR / cloud / map / path layers and camera frusta; drive or
manipulate (sticks, keyboard, gamepad, wrist-target IK); Stiff/Soft and Gravity Comp / Pure
PD toggles; a base-link fault banner when the base stops answering; per-module joint
sliders; ORCA hand panel (grasp, presets, shapes, animation library, 17 joints); Home
(no-op when already home, lift sent once); a 50 Hz choreography timeline with pose-blend and
lift-rate guards. A MAX SPEED slider appears only if the gateway advertises `speedCaps`.
`hello.robot` picks the hand mapping (`MABEL` ORCA v1, `NOVA` ORCA v2).

**Login.** On by default (`MABEL_CC_REQUIRE_LOGIN=1`) and fail-closed: credentials come only
from `MABEL_CC_USER` / `MABEL_CC_PASSKEY` (e.g. a local, untracked `~/.mabel_cc_env`).
Other env: `MABEL_CC_PORT`, `MABEL_CC_HOST`, `MABEL_CC_SOURCE` (`auto|ros|mock`),
`MABEL_CC_TELEMETRY_HZ`, `MABEL_CC_TEMP_WARN/HOT`, `MABEL_NO_BROWSER=1`.

```bash
cd web_gui && MABEL_CC_REQUIRE_LOGIN=0 python3 -m pytest control_center/tests -q
node --test control_center/tests/js     # three.js tests need `three` resolvable (npm i three)
```

`tests/js/frames.test.mjs` guards the frame rules (overlays in odom; render `qpos − ref`).
Reports in `docs/`; experiments in `tools/exp_*.py` and `experiments/`.

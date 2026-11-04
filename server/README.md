# server — the teleoperation gateway

The gateway between every operator device and the robot. It takes operator intent over one
WebSocket contract (`wire_protocol.py`), runs it through `controller/` (retargeting, whole-body
control, the base safety gate), sends the result to a pluggable *plant* (`plant.py`:
`sim:bridge`, `sim:mujoco` or `real:ros`), and streams `robot_state` back at 100 Hz. The gateway
is `sim_teleop_bridge.py`; despite the name it serves the real robot too. Cameras are served by
a separate process, never through the gateway.

Install: `pip install -r server/requirements.txt` (Python ≥ 3.10; set `MABEL_PY` to choose the
interpreter). The real robot also needs the built ROS 2 workspace in `ros/`.

## Run

```bash
bash server/run.sh          # simulation: MuJoCo HAL + cameras + gateway + studio on :8097
bash server/real_run.sh     # real robot (on the robot computer): HAL, ROS, cameras, SLAM, gateway, studio on :8094
```

`run.sh` flags: `-s <scene.xml>`, `--test`, `--viewer`, `--mock`, `--no-studio`, `--no-tui`,
`--real` (local USB cameras), `--quality N`. `real_run.sh -h` lists its per-subsystem switches
(`--no-arms`, `--no-hands`, `--no-slam`, …). Arms come up limp until an operator enables torque.

## Ports

| Port | Content |
|---|---|
| 9090 | WebSocket `/teleop` (commands up, `robot_state` down), `/monitor`, `/map` |
| 8080 | HTTP MJPEG `/camera/<name>/stream.mjpg`, WebSocket H.264 `/camera/<name>/h264`, `/console` |
| 9091 / 9092 | UDP JPEG / UDP H.264 video; 9092 also serves the WebSocket SLAM map `/map` |
| 9600–9623, 9610 | Read-only data plane; 9610 is duplex voice ([data_streams.md](docs/data_streams.md)) |
| 8097 / 8094 | Simulation / real-robot Control Center web studio |

### Camera streams

`main`, `head_left`, `head_right`, `head_stereo` (one side-by-side frame, for headsets),
`wrist_left`, `wrist_right`, `front`. Open `http://<host>:8080/` to see each stream's status.

## Connecting a client

All clients speak the same protocol to the same ports; only the address differs. Nothing is
exposed to the public internet.

- **Wi-Fi LAN.** Put the robot host and the client on the same network. The gateway advertises
  itself over Bonjour (`_mabel-teleop._tcp`), so the apps find it automatically; otherwise enter
  the host's LAN IP (e.g. `192.168.1.42`). Browsers open `http://<host>:8097` (sim) or
  `http://<host>:8094` (real). On macOS, accept the firewall prompt for Python.
- **Bluetooth LE.** For control without any network, run `/usr/bin/python3 server/ble_bridge.py`
  on the robot host (Linux, BlueZ, `python3-dbus`). It advertises as `MABEL` and forwards GATT
  writes to the gateway as ordinary wire messages. Control and telemetry only, no video.
- **Tailscale.** For a client on another network, install Tailscale on the robot host and the
  client, join both to your tailnet, and use the host's `100.x.y.z` address exactly like a LAN
  IP (Bonjour does not cross the tailnet, so add the host by hand).

**Access code (optional).** Start the server with `MABEL_TOKEN=<your-code>` to require
`?key=<your-code>` on every WebSocket; enter the same code in the client. Unset (the default)
means open, which is fine on a trusted LAN. No code ships in the repository.

## More

- [`docs/protocol.md`](docs/protocol.md) — the `:9090` wire contract and video transports
- [`docs/data_streams.md`](docs/data_streams.md) — the 96xx data plane and record formats
- [`docs/data_collection.md`](docs/data_collection.md) — recording demonstrations (HDF5, MCAP)
- `docs/*.pdf` — technical reports with measured results; `tools/` — experiments and probes

Tests: `python3 -m pytest server/tests -q`. The Command Monitor (`monitor_gui.py`) shows every
connected client and its command stream; `start_teleop.sh` opens it by default.

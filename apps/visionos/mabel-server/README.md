# mabel-server — minimal protocol stub

Two small standalone Python servers for checking the headset's wire traffic without the
MABEL stack. They implement only the basic envelope (`teleop_frame` in, `robot_state` and
`pong` out) and three fake camera streams; they do not retarget, simulate or serve the
newer message types. For end-to-end testing use the MuJoCo twin (`bash server/run.sh`),
which speaks the full contract.

| Script | What it does |
|---|---|
| `server.py` | WebSocket server on `ws://<host>:9090/teleop`: logs incoming `teleop_frame`s, replies to `ping`, sends a fake `robot_state` once a second |
| `fake_cameras.py` | MJPEG on `:8080` at `/camera/main/stream.mjpg`, `/camera/wrist_left/stream.mjpg`, `/camera/wrist_right/stream.mjpg`, with moving labels and timestamps |

```bash
cd apps/visionos/mabel-server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python server.py --host 0.0.0.0 --port 9090        # options: --path /teleop, --verbose
python fake_cameras.py --host 0.0.0.0 --port 8080  # prints this machine's LAN address
```

Add the printed address in the app (Settings → **Add robot**).

To feed frames into your own code, pass a handler to `TeleopServer(on_teleop_frame=...)`;
it receives each decoded `teleop_frame` payload as a dict. The module's
`unpack_transform` and `joint_world_transform` helpers return the 16-float row-major
matrices for the head and each hand joint.

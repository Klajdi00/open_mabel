# `mabel.sources` — device ingest

Every input device is converted once, at ingest, into `mabel.types.OperatorPose` (head, wrists, 21-point hand skeletons, pinches, timestamp). Everything downstream is device-agnostic.

| Module | Source |
|---|---|
| `wire.py` | gateway WebSocket `teleop_frame` (Vision Pro, iOS, browser, webcam) + `app_joystick` |
| `avp.py` | direct Vision Pro ZMQ stream (`AVPReceiver`) |
| `mock.py` | synthetic operator for tests |

```python
from mabel import OperatorPose, app_joystick
pose = OperatorPose.from_teleop_frame(msg)
joy = app_joystick(msg.get("navJoystick"))
```

Wire format: `server/wire_protocol.py`. Report: [`controller/docs/sources.pdf`](../../docs/sources.pdf).

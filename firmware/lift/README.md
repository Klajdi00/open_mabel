# lift — vertical column controller (Raspberry Pi Pico)

A DC standing-desk column (BTS7960 H-bridge, quadrature encoder) under cascaded position/velocity control in MicroPython on a Pico. The host talks to it over USB serial at 115200 baud with a line-based ASCII protocol (`help` lists the commands). On the robot the HAL owns the port (`hardware_bridge/mabel_hw/drivers/lift.py`; off by default, enable with `MABEL_HW_LIFT=true`). Stop the HAL before using these tools.

## Run

```bash
pip install mpremote
firmware/lift/tools/deploy.sh             # flash firmware/*.py (targets the Pico by USB id; LIFT_PICO_DEV=... to override)
firmware/lift/tools/deploy.sh repl

cd firmware/lift
./setup_gui_venv.sh && ./lift_gui.sh      # Lift Studio GUI (gui/README.md); -p /dev/ttyACM0
python3 host/console.py -p /dev/ttyACM0   # serial console with telemetry
```

```python
from lift import Lift                     # from firmware/lift/host; metres (lift_controller.py: millimetres)
lift = Lift("/dev/ttyACM0")
lift.lift_to_height(0.60)
lift.set_neutral()                        # torque off
```

| Folder | Contents |
|---|---|
| `firmware/` | Pico code; pins, geometry and gains in `config.py` |
| `host/` | `lift_controller.py` (mm), `lift.py` (m), `console.py` |
| `gui/`, `tools/` | Tk GUI; `deploy.sh` |

Run `calibrate` once to stall-seek floor and ceiling. The scale and last height persist in flash; re-run `home` if the column was back-driven while off.

## Safety

- `stop` / `torque off` / the GUI STOP halt the motor (`torque off` free-wheels).
- No encoder tick for 80 ms while driving cuts power. Soft limits keep moves inside the calibrated range (`limit off` disables them; use with care).
- The HAL never homes the lift at startup unless `MABEL_HW_LIFT_AUTO_HOME=true`.

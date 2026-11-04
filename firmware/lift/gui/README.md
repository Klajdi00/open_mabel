# Lift Studio (GUI)

Tk GUI for the Pico lift, built on `../host/lift_controller.py`.

```bash
cd firmware/lift
pip install -r host/requirements.txt   # pyserial
./lift_gui.sh                          # auto-detects the Pico; -p /dev/ttyACM0 to force
```

Only one program can hold the port, so close `mpremote repl`, `host/console.py` and the HAL first. Status dot: green = live, orange = port open but no telemetry (reset the Pico), red = disconnected.

Cards: position (slider + presets), velocity & jog, calibration & limits, home/zero/park. The global STOP sits in the toolbar. Files: `gui.py`, `lift_bus.py`, `style.py`.

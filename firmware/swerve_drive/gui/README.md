# Swerve Studio (GUI)

Tk GUI for the base's SPARK controllers over SocketCAN (`sparkcan_py`). It talks to the bus directly, so it's a bench tool: stop the HAL's base subsystem first.

```bash
cd firmware/swerve_drive
./setup_gui_venv.sh            # once: venv + builds sparkcan_py
./swerve_gui.sh                # can0; falls back to SIM mode if CAN is unavailable
./swerve_gui.sh --mock         # simulation only
./swerve_gui.sh --iface can1
```

Panels: **Drive** (per-module steer/drive controls, Enable/Disable All, E-STOP) and **CAN Bus** (up/restart/down/scan/doctor). Wiring (IDs, steer offsets, module positions) lives in `config.py`. Drive defaults to open-loop duty capped at `DRIVE_DUTY_MAX`. Only switch to closed-loop velocity after tuning the SPARK PID. Steer feedback needs status frame 5 enabled in REV Hardware Client.

# gui — Tk Control Center (bench GUI)

Desktop GUI for bench testing: calibration, telemetry and joint-level control of base, arms, body, hands and lift. It is only a client: DDS to the HAL, plus the hand RPC (`127.0.0.1:8779`) for the Hands tab. It never opens a device. The browser teleop UI is separate: [`web_gui/control_center/`](../../web_gui/control_center/).

## Run

```bash
cd hardware_bridge
./run_bridge.sh          # HAL first (or the systemd service)
./gui.sh                 # = gui/control_center.sh (same RMW / domain as the HAL)
```

On macOS (no `rclpy`) only the Hands tab works. It starts `python3 -m mabel_hw.rpc.hand_server` on demand. Close the standalone ORCA Hand Studio first, since only one program can hold the bus.

Tabs: Overview, Base, Arms, Body, Hands (control, setup, calibration, poses), Lift, Monitor (read-only view of the gateway at `ws://<host>:9090/monitor`).

## Safety

- **GLOBAL E-STOP** (or Escape) latches the base e-stop, zeroes the twist and frees both hands. It does **not** change arm/body torque; use those tabs' torque controls.
- Hands come up free-drive. Arm/body targets are sent only on Send or while Continuous is armed.

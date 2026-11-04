# sparkcan_py (vendored)

Python bindings for the third-party `sparkcan` C++ library (SocketCAN control of REV SPARK MAX / SPARK Flex). A modified fork of <https://github.com/grayson-arendt/sparkcan> via <https://github.com/vedAnts256/sparkcan_pybindings>. Used by the Swerve Studio GUI and `../scripts/`.

```bash
pip install -U scikit-build-core cmake ninja pybind11
pip install ./sparkcan_py          # from firmware/swerve_drive
```

Needs a USB-CAN adapter (e.g. CANable). Set each controller's device ID and motor type (`kBrushless` for MAXSwerve) in REV Hardware Client. The wrong motor type can damage the motor.

Not affiliated with or endorsed by REV Robotics. Product names are their owners' trademarks. License: [LICENSE](LICENSE).

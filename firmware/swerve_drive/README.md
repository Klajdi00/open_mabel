# swerve_drive — mobile base (3 × MAXSwerve, REV SPARK CAN)

Bench tooling for the base's CAN bus: three swerve modules (front-right, front-left, back-centre), each with a SPARK drive and a SPARK steer controller at 1 Mbit/s. On the robot the HAL owns the base (`hardware_bridge/mabel_hw/drivers/swerve.py`, UDP + protobuf on port 5800). The reference firmware in `src/` uses a simpler 12-byte twist packet on UDP 8888 and **does not speak the HAL's protocol**.

CAN IDs: odd = drive (velocity), even = steer (absolute position). 1/2 front-right, 3/4 front-left, 5/6 back-centre.

## Run

```bash
cd firmware/swerve_drive
pio run -e teensy41 -t upload            # reference firmware; settings in include/Config.h

./setup_can.sh                           # can0 at 1 Mbit/s (also: status, restart, doctor)
./setup_gui_venv.sh                      # venv + builds sparkcan_py (or: pip install ./sparkcan_py)
python3 scripts/verify_can_motors.py     # 1. confirm all 6 SPARKs answer
python3 scripts/set_drive_limits.py      # 2. burn a drive output cap into flash (once)
python3 scripts/swerve_teleop.py         # 3. keyboard swerve teleop (--dry-run)
./swerve_gui.sh                          # Swerve Studio GUI (gui/README.md); --mock for no hardware
```

Before the first real drive, set `STEER_OFFSET`, `DRIVE_SIGN`, `WHEEL_RADIUS` and `DRIVE_GEAR_RATIO` at the top of `swerve_teleop.py`. Protocol reference: [REV_CAN_PROTOCOL.md](REV_CAN_PROTOCOL.md). `swerve_drive_sim.py` plots the swerve kinematics.

## Safety

- Lift the wheels off the ground for first tests, and burn the output cap before driving on the floor.
- The SPARKs disable themselves when heartbeats stop, and the reference firmware stops after 100 ms without a twist.
- Only one program may command the bus. Stop the HAL (or its base subsystem) before using these tools on the robot.

# arm_calibration (`mabel_sysid`)

Identifies each arm link's mass and centre of mass (MuJoCo `<inertial>`) for gravity compensation and inverse dynamics. Validated in simulation by scrambling MJCF masses ±30 % and recovering them.

Report: [docs/ARM_CALIBRATION.pdf](docs/ARM_CALIBRATION.pdf) · notes: [THEORY.md](docs/THEORY.md), [LITERATURE.md](docs/LITERATURE.md).

## Run

```bash
pip install -r ../requirements.txt
./run.sh --no-video                        # extract model -> check -> identify -> gravity-comp demo

python tools/extract_arm_model.py          # build models/right_arm_sysid.xml from mabel_full.xml
python scripts/00_check_model.py           # analytic regressor vs MuJoCo
python scripts/01_run_sysid.py             # --seed, --noise, --n-static, ...
python scripts/02_gravity_comp_demo.py
python scripts/03_view_arm.py --excite     # needs a display
python experiments/validate.py             # experiments behind the report (--quick)
```

Outputs go to `results/`. For the real arm, `hardware/arm_bus.py` gives CAN access in MJCF joint coordinates; joint friction/damping come from [`../motor_calibration/`](../motor_calibration/).

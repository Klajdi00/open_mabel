# `mabel.motion_model` — tip-over envelope

Every base twist, from any source (joystick, apps, browser, Nav2, admittance), goes through `safe_base_step(...)` before `swerve_ik`. Nothing calls `swerve_ik` on a raw twist.

| Module | Contents |
|---|---|
| `tip_over.py` | lift-aware speed/decel caps + ZMP limiter `safe_base_step`; MJCF-measured CoM constants. NumPy-only |
| `com_support.py` | live CoM and support margin from MuJoCo (verifies the constants) |
| `base_response.py` | identified base twist response model (command odometry) |
| `swerve_modules.py` | per-module wheel-torque / steer-rate limits |
| `relative_follow.py` | `RelativeBaseFollow`: base follows the operator's relative motion |

```python
from mabel.config import safe_base_step     # NumPy-only
vx, vy, w = safe_base_step(vx, vy, w, target_vx, target_vy, target_w, dt, lift_z)
```

After mass, lift or link changes in the MJCF, run `python3 controller/experiments/wbc_stability.py`. If check E4 shows more than ~1 cm CoM error, paste the printed constants into `tip_over.py`.

Reports: [`motion_model.pdf`](../../docs/motion_model.pdf), [`wbc_stability.pdf`](../../docs/wbc_stability.pdf).

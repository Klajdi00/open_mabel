# `mabel.dynamic_models`

Region dynamics from the MJCF: one MuJoCo oracle (`model.py`, `MujocoDynamics`) supplies gravity, bias, mass matrix and inverse dynamics, in sim or on hardware (fed measured q, q̇). Regions (`regions.py`) only pick joints out of it.

| Region | Class | Joints |
|---|---|---|
| `arms` | `ArmsDynamicModel` | both arms + ORCA wrists |
| `upper_body` | `UpperBodyDynamicModel` | arms + torso, lift freed |
| `whole_body` | `WholeBodyDynamicModel` | + base wrench via `base_wrench()` |

Each region exposes `gravity_ff()`, `bias_ff()`, `mass_matrix()` and `external_torques(tau_cmd, dt)`, a momentum observer that needs no acceleration. `gravity.py` holds the one shared upper-body gravity-comp layer.

```python
from mabel.dynamic_models import ArmsDynamicModel
arms = ArmsDynamicModel(model, data)
pd.feedforward = arms.gravity_ff
tau_ext = arms.external_torques(tau_cmd, dt)
```

Report: [`controller/docs/dynamic_models.pdf`](../../docs/dynamic_models.pdf).

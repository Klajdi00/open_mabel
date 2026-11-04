# `mabel.hand_retargeting` — fingers to ORCA hands

Position-controlled finger retargeting, kept separate from the body. Input is a device-agnostic hand skeleton from `mabel.sources`.

| Module | Retargeter |
|---|---|
| `keypoint_ik.py` | `HandIKRetargeter`: keypoint IK, Gauss–Newton or SLSQP, gated pinch term (default) |
| `vector.py` | `HandVectorRetargeter`, `make_hand_retargeter` (config-selected factory) |
| `anatomical.py` | `HandRetargeter`: per-joint angle mapping baseline |
| `grasp.py` | `apply_grasp`: one [0, 1] grip curls all fingers |

Selection: `config.FINGER_RETARGET`, `config.HAND_SOLVER`; in the deployed engine `MABEL_FINGER_SOLVER=keypoint|anatomical`, `MABEL_HAND_SOLVER=gn|slsqp`.

```python
from mabel.hand_retargeting import make_hand_retargeter
rt = make_hand_retargeter(model, data)
rt.apply("Left", pose.left_joints)
```

Report: [`controller/docs/hand_retargeting.pdf`](../../docs/hand_retargeting.pdf).

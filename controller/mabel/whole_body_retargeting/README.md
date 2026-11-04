# `mabel.whole_body_retargeting`

Maps the operator's head and wrist poses onto the arms, neck, torso, lift and base. Each module stands alone. Fingers are in `mabel.hand_retargeting`, and the deployed composition is `mabel.teleop_engine.TeleopEngine`.

| Module | Role |
|---|---|
| `wrist_clutch.py` | `Retargeter`: clutch-relative or shoulder-anchored wrist mapping |
| `arm_cosine.py` | `ArmRetargeter`: cosine-directional arm IK, ORCA wrist as 8th DoF |
| `neck.py` | `NeckController`: 3-DOF gaze look-at |
| `followers.py`, `workspace_assist.py` | `LazyBase`/`LazyLift`/`LazyTorso` reach assists; base frees arm workspace |
| `twist_estimator.py`, `body_frame_estimator.py`, `heading.py` | torso frame / heading from head + wrists |
| `calibration.py`, `profile_capture.py` | per-operator profile (`BodyCalibrator`, `OperatorProfile`) and capture screening |

Usually used through the facade (`MabelTeleop.start_body_calibration`, `load_profile`, `apply_profile`). Offline profile fitting: `controller/scripts/tools/fit_operator_profile.py`.

Reports (`controller/docs/`): [`whole_body_retargeting.pdf`](../../docs/whole_body_retargeting.pdf), [`body_frame.pdf`](../../docs/body_frame.pdf), [`calibration.pdf`](../../docs/calibration.pdf).

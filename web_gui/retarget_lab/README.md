# Retarget Lab

Record operator motion from the Vision Pro and compare retargeting maps side by side, kinematically, with no robot. It poses as a teleop gateway on **:9090** (the unmodified app connects, found over Bonjour) and serves its studio on **:8120**.

## Run

```bash
bash web_gui/retarget_lab/run.sh               # stop any other :9090 gateway first
bash web_gui/retarget_lab/run.sh --no-browser
python3 web_gui/retarget_lab/tools/fake_vp.py --host 127.0.0.1 --clip <clip.json.gz> --seconds 20   # no headset
cd web_gui && python3 -m pytest retarget_lab/tests -q
```

## Workflow

1. Pick the lab in the app's robot list. Head and hands start streaming.
2. **Record** saves the raw `teleop_frame` stream to `data/episodes/<id>/` (`MABEL_RTL_DATA` overrides).
3. Enable retargeters and view them as a **Lineup** or **Overlay**. The first active one is mirrored back to the headset.
4. **Evaluate** replays an episode through every active retargeter deterministically. The **Metrics** tab explains each metric.

Controls: **Scope** (upper body = local; whole body = base and lift follow), per-joint **Ablations** (torso/neck/lift Free or Fixed), **End effector** (ORCA hand or OpenArm gripper).

## Retargeters

Each variant is a `TeleopEngine` configuration in [`server/variants.py`](server/variants.py), running in its own worker. The refresh button reloads that file; to pick up engine code changes, toggle the variant off and on.

| Variant | Arm map |
|---|---|
| `deployed` | alias for the variant the robot runs (`DEPLOYED` in `variants.py`) |
| `ours`, `shipped_v1`…`v5` | engine defaults and pinned snapshots |
| `rel`, `gmr`, `ext`, `abs`, `wb_ik`, `direct` | baselines (relative increments, scaled keypoints, extension fraction, head-anchored, whole-body IK, joint copy) |
| `ours_fast`, `ours_anat`, `ours_vec`, `ours_noatlas` | single-change ablations |

The scripts behind the measurements are in `experiments/` (e.g. `exp_retarget_lab.py`, `exp_arm_shape.py`, `exp_local_scope.py`), and results go to `experiments/results/`.

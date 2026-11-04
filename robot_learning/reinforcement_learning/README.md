# reinforcement_learning — MABEL on mjlab

GPU RL environments built on a vendored copy of [mjlab](https://github.com/mujocolab/mjlab)
(Isaac-Lab manager-based API on MuJoCo-Warp). Task models are derived at load time from the
canonical `simulation/mabel_mujoco/models/mabel_full.xml`, so model edits flow through.

| Task ID | Embodiment | DOF |
|---|---|---|
| `Mjlab-Lift-Cube-Mabel` | right arm + ORCA hand, fixed pedestal | 24 |
| `Mjlab-Lift-Box-Mabel-Bimanual` | both arms + both hands | 48 |
| `Mjlab-Lift-Cube-Mabel-WholeBody` | **whole-body**: right arm + hand + lift + torso | 26 |

```bash
cd robot_learning/reinforcement_learning/mjlab
uv sync --extra cu128                       # CUDA host (use --extra cpu on macOS: evaluation only)
uv run play  Mjlab-Lift-Cube-Mabel-WholeBody --agent zero                  # sanity check
uv run train Mjlab-Lift-Cube-Mabel-WholeBody --env.scene.num-envs 4096     # PPO (NVIDIA GPU)
uv run play  Mjlab-Lift-Cube-Mabel-WholeBody --wandb-run-path <entity>/mjlab/<run-id>
uv run pytest tests/test_mabel_manipulation.py -q
```

Task code: `mjlab/src/mjlab/tasks/manipulation/config/mabel/` · robot:
`mjlab/src/mjlab/asset_zoo/robots/mabel/` · report: `mjlab/mabel/docs/mabel_rl_env.pdf`.
The swerve base is not yet part of any task.

## wbc_rl — whole-body teleop controller (NOVA)

[`wbc_rl/`](wbc_rl/README.md): one policy tracks a Vision Pro operator's head + wrists and outputs every joint
(swerve twist, lift, torso, neck, 14 arm joints). Two tracks: `NOVA-WBC-v0` (Isaac Lab 3 + Newton, `env.py`) and a
SONIC-style teacher -> student on mjlab (`sonic.py`); `deploy.py` / `sim2sim.py` run the exported policy in numpy +
MuJoCo. Motion data (EgoDex, LAFAN1) and checkpoints are not included.

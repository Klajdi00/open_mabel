# MABEL tasks for mjlab

GPU reinforcement-learning environments for MABEL dexterous manipulation, built on
[mjlab](https://github.com/mujocolab/mjlab) (the Isaac Lab manager-based API on
MuJoCo-Warp). `robot_learning/reinforcement_learning/mjlab/` is a vendored copy of mjlab (upstream commit in
`UPSTREAM_MJLAB_COMMIT.txt`) with the MABEL robot and tasks added in-tree, so the standard
`train` / `play` entry points accept the MABEL task IDs directly.

## Layout

| Path (under `robot_learning/reinforcement_learning/mjlab/`) | Content |
|---|---|
| `src/mjlab/asset_zoo/robots/mabel/derive_model.py` | derives reduced task models from the canonical `mabel_full.xml` with `MjSpec`: prune subtrees, fix the base, add grasp and fingertip sites, name collision geoms |
| `src/mjlab/asset_zoo/robots/mabel/mabel_constants.py` | position actuators, PD gains, effort limits, IK-designed home pose, action scales |
| `src/mjlab/tasks/manipulation/config/mabel/` | task definitions: `env_cfgs.py` (robot + table + object), `bimanual_mdp.py` (two-hand rewards and observations), `rl_cfg.py` (PPO), `__init__.py` (task registration) |
| `tests/test_mabel_manipulation.py` | 15 validation tests (model counts, sites, home pose limits, registration, shapes, rollouts, resting contact) |
| `mabel/experiments/generate_figures.py` | seeded figure and metrics script → `mabel/docs/figures/` |
| `mabel/docs/mabel_rl_env.pdf` | technical report: method, experiments, limitations |

The task models are derived at load time from `simulation/mabel_mujoco/models/mabel_full.xml`,
located through `$MABEL_ROOT` or by walking up from the package. Edits to the canonical model
flow through with no derived file to regenerate.

## Tasks

| Task ID | Embodiment | Actuated DOF |
|---|---|---|
| `Mjlab-Lift-Cube-Mabel` | right 7-DOF arm + ORCA hand (1 wrist + 16 finger joints), fixed pedestal; lift a cube from a table | 24 |
| `Mjlab-Lift-Box-Mabel-Bimanual` | both arms + both hands; lift a box too wide for one hand | 48 |
| `Mjlab-Lift-Cube-Mabel-WholeBody` | right arm + hand + lift + torso; tall lift goals reward raising the whole body | 26 |

In the whole-body task the lift's two slides stay coupled by the retained `Sync Lift`
equality and are driven by one actuator on `Lift Lower`. A mobile-base task is not
implemented: `build_manip_spec(..., keep_base=True, fixed_base=False)` keeps the swerve base
and floating joint, but locomotion and manipulation managers have not been composed.

## Usage

```bash
cd robot_learning/reinforcement_learning/mjlab
uv sync --extra cpu            # macOS / CPU (evaluation only); use --extra cu128 on a CUDA host

uv run python -m mjlab.asset_zoo.robots.mabel.derive_model --sides right --view
uv run python mabel/experiments/generate_figures.py
uv run pytest tests/test_mabel_manipulation.py -q
uv run play Mjlab-Lift-Cube-Mabel --agent zero                 # zero-action sanity check

# training requires an NVIDIA GPU
uv run train Mjlab-Lift-Cube-Mabel --env.scene.num-envs 4096
uv run play  Mjlab-Lift-Cube-Mabel --wandb-run-path <entity>/mjlab/<run-id>
```

`derive_model` options: `--sides left right`, `--no-hand`, `--keep-lift`, `--keep-torso`,
`--keep-neck`, `--keep-base`, `--out <file.xml>`, `--view`.

## Design notes

- **No actuators, sensors or tendons in the derived model.** Derivation strips them (including
  the 360-ray LiDAR and the coupled-lift tendon); mjlab attaches position actuators from
  `mabel_constants.py`.
- **Fixed base means a mocap wrapper.** Removing `base_root` makes the robot fixed-base and
  mjlab wraps it in a mocap body. Every fixed-base entity, the table included, needs a reset
  event (`reset_root_state_uniform`); without the table's reset event, objects in
  environments other than the first miss their table and fall.
- **Contact budget.** The ORCA hand adds many collision geoms, so the tasks raise `nconmax`
  and `njmax` above the mjlab defaults.
- **Home pose.** IK-designed to hover the hand over the table with every joint at least
  0.24 rad from its limit; it holds under gravity with less than 3° drift.
- **Reachable workspace** (measured, right hand): x ∈ [−0.87, 0.65], y ∈ [−0.45, 0.95],
  z ∈ [0.14, 1.70] m. The table and cube sit at (−0.30, 0.38), table top z = 0.50 m.

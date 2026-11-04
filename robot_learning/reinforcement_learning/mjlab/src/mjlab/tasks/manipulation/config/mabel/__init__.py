from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  mabel_bimanual_lift_env_cfg,
  mabel_lift_cube_env_cfg,
  mabel_wholebody_lift_env_cfg,
)
from .rl_cfg import (
  mabel_bimanual_lift_ppo_runner_cfg,
  mabel_lift_cube_ppo_runner_cfg,
  mabel_wholebody_lift_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Mabel",
  env_cfg=mabel_lift_cube_env_cfg(),
  play_env_cfg=mabel_lift_cube_env_cfg(play=True),
  rl_cfg=mabel_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Box-Mabel-Bimanual",
  env_cfg=mabel_bimanual_lift_env_cfg(),
  play_env_cfg=mabel_bimanual_lift_env_cfg(play=True),
  rl_cfg=mabel_bimanual_lift_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Mabel-WholeBody",
  env_cfg=mabel_wholebody_lift_env_cfg(),
  play_env_cfg=mabel_wholebody_lift_env_cfg(play=True),
  rl_cfg=mabel_wholebody_lift_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

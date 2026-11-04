"""PPO runner config for the MABEL lift-cube task.

The MABEL action space (24 DOF: 7 arm + wrist + 16 finger) is larger than YAM's
(7), so we keep generous MLP capacity. Hyperparameters otherwise follow the YAM
lift-cube reference.
"""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def mabel_lift_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mabel_lift_cube",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def mabel_bimanual_lift_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the 48-DOF bimanual box lift (both arms + hands)."""
  cfg = mabel_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "mabel_bimanual_lift"
  cfg.max_iterations = 15_000
  return cfg


def mabel_wholebody_lift_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the 26-DOF whole-body lift (arm + hand + lift + torso)."""
  cfg = mabel_lift_cube_ppo_runner_cfg()
  cfg.experiment_name = "mabel_wholebody_lift"
  cfg.max_iterations = 12_000
  return cfg

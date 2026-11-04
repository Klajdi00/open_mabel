"""Two-hand (bimanual) reward + observation terms for MABEL.

The shared manipulation MDP terms assume a single end-effector site
(``site_pos_w[:, site_ids].squeeze(1)``). Bimanual lifting needs both grasp
sites, so these variants take a two-site ``asset_cfg`` and reward/observe both
hands. The object->goal ("bring") part is embodiment-agnostic and reuses
``bring_object_reward`` from the shared MDP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def both_ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base-frame vectors from each grasp site to the object, concatenated (B, 6)."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  obj_pos_w = obj.data.root_link_pos_w.unsqueeze(1)  # (B, 1, 3)
  vec_w = obj_pos_w - ee_pos_w  # (B, 2, 3)
  base_quat = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, vec_w.shape[1], -1)
  vec_b = quat_apply(quat_inv(base_quat), vec_w)
  return vec_b.reshape(vec_b.shape[0], -1)


def bimanual_staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward: both hands reach the object, gated bonus for bringing to goal.

  reaching = exp(-(d_left^2 + d_right^2) / reaching_std^2) rewards *both* grasp
  sites approaching the object; the (1 + bringing) factor adds the lift bonus
  only once the hands are engaged.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  obj_pos_w = obj.data.root_link_pos_w.unsqueeze(1)  # (B, 1, 3)
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=(-1, -2))  # (B,)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def hands_symmetry_penalty(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize uneven hand engagement (encourages a balanced two-hand grasp).

  Returns |d_left - d_right|^2, the squared difference of the two grasp-site
  distances to the object; zero when both hands are equidistant.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (B, 2, 3)
  obj_pos_w = obj.data.root_link_pos_w.unsqueeze(1)
  dists = torch.linalg.norm(ee_pos_w - obj_pos_w, dim=-1)  # (B, 2)
  return torch.square(dists[:, 0] - dists[:, 1])

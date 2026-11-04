"""Tests for the MABEL single-arm + ORCA-hand tabletop lift-cube task.

Validates the model derivation (Phase 0) and the registered env (Phase 1):
correct DOF counts, soft-limit-compliant home, task registration, clean dummy
rollouts, and that the cube rests on the table across environments.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.asset_zoo.robots.mabel.derive_model import build_manip_spec, hand_joints
from mjlab.asset_zoo.robots.mabel.mabel_constants import RIGHT_HOME_JOINT_POS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

TASK = "Mjlab-Lift-Cube-Mabel"
N_ACT = 24  # 7 arm + wrist + 16 finger


def test_right_arm_hand_spec_counts():
  spec = build_manip_spec(sides=("right",))
  m = spec.compile()
  assert m.nq == N_ACT and m.nv == N_ACT
  non_free = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
  assert len(non_free) == N_ACT
  # No leftover actuators/sensors/tendons/equalities (mjlab adds actuators).
  assert len(spec.actuators) == 0
  assert len(spec.sensors) == 0
  assert len(spec.tendons) == 0
  assert len(spec.equalities) == 0


def test_grasp_and_fingertip_sites_present():
  m = build_manip_spec(sides=("right",)).compile()
  names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)}
  assert "right_grasp_site" in names
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    assert f"right_{finger}_tip_site" in names
  geoms = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}
  for finger in ("thumb", "index", "middle", "ring", "pinky"):
    assert f"right_{finger}_tip_collision" in geoms


def test_home_within_soft_limits():
  m = build_manip_spec(sides=("right",)).compile()
  for name, val in RIGHT_HOME_JOINT_POS.items():
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = m.jnt_range[jid]
    c, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.9  # 0.9 soft-limit factor
    assert c - half <= val <= c + half, f"{name} home violates soft limit"


def test_task_registered():
  assert TASK in list_tasks()


def test_hand_joint_count():
  assert len(hand_joints("right")) == 17  # 1 wrist + 16 finger


@pytest.fixture(scope="module")
def env():
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield e
  e.close()


def test_env_action_and_obs_shape(env):
  assert env.action_manager.total_action_dim == N_ACT
  obs, _ = env.reset()
  assert obs["actor"].shape[-1] == 78  # 24 qpos + 24 qvel + 3 + 3 + 24 actions


def test_dummy_rollout_no_nan(env):
  env.reset()
  for policy in (torch.zeros, lambda s, **k: torch.rand(s, **k) * 2 - 1):
    for _ in range(20):
      a = policy((env.num_envs, N_ACT), device=env.device)
      obs, rew, term, trunc, info = env.step(a)
    assert not torch.isnan(rew).any()
    assert not any(torch.isnan(v).any() for v in obs.values() if torch.is_tensor(v))


def test_cube_rests_on_table(env):
  env.reset()
  for _ in range(120):
    env.step(torch.zeros((env.num_envs, N_ACT), device=env.device))
  cube_z = env.scene["cube"].data.root_link_pos_w[:, 2].cpu().numpy()
  assert np.all((cube_z > 0.50) & (cube_z < 0.56)), f"cube not on table: {cube_z}"


# --------------------------------------------------------------------------- #
# Phase 2: dual-arm bimanual.
# --------------------------------------------------------------------------- #

BIMANUAL_TASK = "Mjlab-Lift-Box-Mabel-Bimanual"
N_ACT_BI = 48  # both arms + hands


def test_dual_arm_spec_counts():
  m = build_manip_spec(sides=("left", "right")).compile()
  assert m.nq == N_ACT_BI and m.nv == N_ACT_BI
  names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)}
  assert {"left_grasp_site", "right_grasp_site"} <= names


def test_bimanual_task_registered():
  assert BIMANUAL_TASK in list_tasks()


@pytest.fixture(scope="module")
def bi_env():
  cfg = load_env_cfg(BIMANUAL_TASK, play=True)
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield e
  e.close()


def test_bimanual_shapes_and_rollout(bi_env):
  assert bi_env.action_manager.total_action_dim == N_ACT_BI
  obs, _ = bi_env.reset()
  assert obs["actor"].shape[-1] == 153  # 48 + 48 + 6 + 3 + 48
  for _ in range(20):
    a = torch.zeros((bi_env.num_envs, N_ACT_BI), device=bi_env.device)
    obs, rew, term, trunc, info = bi_env.step(a)
  assert not torch.isnan(rew).any()
  assert not any(torch.isnan(v).any() for v in obs.values() if torch.is_tensor(v))


def test_bimanual_box_rests_on_table(bi_env):
  bi_env.reset()
  for _ in range(120):
    bi_env.step(torch.zeros((bi_env.num_envs, N_ACT_BI), device=bi_env.device))
  z = bi_env.scene["cube"].data.root_link_pos_w[:, 2].cpu().numpy()
  assert np.all((z > 0.53) & (z < 0.60)), f"box not on table: {z}"


# --------------------------------------------------------------------------- #
# Phase 3: whole-body (arm + hand + lift + torso).
# --------------------------------------------------------------------------- #

WB_TASK = "Mjlab-Lift-Cube-Mabel-WholeBody"
N_ACT_WB = 26  # 24 arm+hand + lift + torso


def test_wholebody_spec_keeps_lift_coupling():
  spec = build_manip_spec(sides=("right",), keep_lift=True, keep_torso=True)
  m = spec.compile()
  assert m.nq == 27  # 24 + 2 lift slides + torso
  assert m.neq == 1  # "Sync Lift" equality retained (couples the 2 slides)
  joints = {j.name for j in spec.joints}
  assert {"Lift Lower", "Lift Upper", "Torso"} <= joints


def test_wholebody_task_registered():
  assert WB_TASK in list_tasks()


@pytest.fixture(scope="module")
def wb_env():
  cfg = load_env_cfg(WB_TASK, play=True)
  cfg.scene.num_envs = 2
  e = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield e
  e.close()


def test_wholebody_shapes_and_stability(wb_env):
  assert wb_env.action_manager.total_action_dim == N_ACT_WB
  wb_env.reset()
  for _ in range(120):
    wb_env.step(torch.zeros((wb_env.num_envs, N_ACT_WB), device=wb_env.device))
  jp = wb_env.scene["robot"].data.joint_pos
  jn = wb_env.scene["robot"].joint_names
  lift = jp[:, jn.index("Lift Lower")].abs().max().item()
  assert lift < 0.03, f"lift drifted from home under gravity: {lift}"
  z = wb_env.scene["cube"].data.root_link_pos_w[:, 2].cpu().numpy()
  assert np.all((z > 0.50) & (z < 0.56))

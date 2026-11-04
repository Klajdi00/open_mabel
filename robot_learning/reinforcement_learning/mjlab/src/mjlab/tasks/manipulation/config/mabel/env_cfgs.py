"""MABEL single-arm + ORCA-hand tabletop lift-cube task.

Phase 1 of the MABEL manipulation progression: the right 7-DOF arm + 16-DOF
ORCA hand (fixed-base pedestal) must reach a cube on a table and lift it to a
commanded height. Built on the shared ``make_lift_cube_env_cfg`` template, with
the per-robot placeholders filled for MABEL (grasp site, fingertip friction
geoms, workspace, viewer).

Coordinates are relative to the robot base at the environment origin. The right
arm's reachable workspace (measured in Phase 0) centres near
(x=-0.30, y=0.38, z=0.5-0.7); the table + cube live there.
"""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.mabel import (
  action_scale,
  get_mabel_robot_cfg,
)
from mjlab.asset_zoo.robots.mabel.mabel_constants import BIMANUAL_HOME_JOINT_POS
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.config.mabel import bimanual_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# --------------------------------------------------------------------------- #
# Scene props.
# --------------------------------------------------------------------------- #

TABLE_CENTER = (-0.30, 0.38)
TABLE_TOP_Z = 0.50
_TABLE_HALF = (0.30, 0.30, TABLE_TOP_Z / 2.0)
CUBE_SIZE = 0.02
FINGERTIP_GEOMS = r"right_(thumb|index|middle|ring|pinky)_tip_collision"


def get_cube_spec(
  cube_size: float = CUBE_SIZE,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
    condim=4,
    friction=[1.0, 5e-3, 5e-4],
  )
  return spec


def get_table_spec(
  center: tuple[float, float] = TABLE_CENTER,
  half: tuple[float, float, float] = _TABLE_HALF,
) -> mujoco.MjSpec:
  """Static table (fixed-base -> mjlab auto-wraps in mocap). Top at 2*half_z."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half,
    pos=(center[0], center[1], half[2]),
    rgba=(0.55, 0.45, 0.35, 1.0),
    friction=[1.0, 5e-3, 5e-4],
  )
  return spec


# --------------------------------------------------------------------------- #
# Environment.
# --------------------------------------------------------------------------- #


def mabel_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_mabel_robot_cfg(sides=("right",)),
    "table": EntityCfg(spec_fn=get_table_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # Per-actuator action scale (0.25 * effort / kp).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale(("right",))

  # Grasp site for reach observation + reward.
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "right_grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("right_grasp_site",)

  # Position the (fixed-base, mocap-wrapped) table at each env's origin. Without
  # this, all tables stay at the world origin and cubes in envs>0 miss the table.
  cfg.events["reset_table"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  # Fingertip friction domain randomization.
  for term in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[term].params["asset_cfg"].geom_names = FINGERTIP_GEOMS

  # End-effector / ground collision sensor: watch the right hand vs the plane.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "right_palm"

  # Cube spawn + lift-goal ranges, tuned to the MABEL tabletop workspace.
  cube_z = TABLE_TOP_Z + CUBE_SIZE + 0.005
  cfg.commands["lift_height"] = LiftingCommandCfg(
    entity_name="cube",
    resampling_time_range=(8.0, 12.0),
    debug_vis=True,
    difficulty="dynamic",
    object_pose_range=LiftingCommandCfg.ObjectPoseRangeCfg(
      x=(TABLE_CENTER[0] - 0.10, TABLE_CENTER[0] + 0.10),
      y=(TABLE_CENTER[1] - 0.10, TABLE_CENTER[1] + 0.10),
      z=(cube_z, cube_z),
      yaw=(-3.14, 3.14),
    ),
    target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
      x=(TABLE_CENTER[0] - 0.08, TABLE_CENTER[0] + 0.08),
      y=(TABLE_CENTER[1] - 0.08, TABLE_CENTER[1] + 0.08),
      z=(TABLE_TOP_Z + 0.10, TABLE_TOP_Z + 0.25),
    ),
  )

  cfg.viewer.body_name = "torso"

  # MABEL's ORCA hand has many collision geoms; raise the contact/constraint
  # budget above the YAM default so finger+table+cube contacts are not dropped.
  cfg.sim.nconmax = 200
  cfg.sim.njmax = 800

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


# --------------------------------------------------------------------------- #
# Phase 2: dual-arm bimanual box lift.
# --------------------------------------------------------------------------- #

BIMANUAL_TABLE_CENTER = (-0.35, 0.0)
BOX_HALF = (0.045, 0.09, 0.05)  # 9 x 18 x 10 cm: too wide/heavy for one hand
BOX_MASS = 0.4
BOTH_GRASP_SITES = ("left_grasp_site", "right_grasp_site")
BIMANUAL_FINGERTIP_GEOMS = r"(left|right)_(thumb|index|middle|ring|pinky)_tip_collision"


def get_box_spec() -> mujoco.MjSpec:
  """Free-jointed graspable box, sized to require two hands."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=BOX_HALF,
    mass=BOX_MASS,
    rgba=(0.2, 0.5, 0.85, 1.0),
    condim=4,
    friction=[1.0, 5e-3, 5e-4],
  )
  return spec


def mabel_bimanual_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Both arms + hands lift a wide box off a central table to a goal height."""
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_mabel_robot_cfg(sides=("left", "right"), home=BIMANUAL_HOME_JOINT_POS),
    "table": EntityCfg(spec_fn=lambda: get_table_spec(BIMANUAL_TABLE_CENTER)),
    "cube": get_box_spec_entity(),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale(("left", "right"))

  # Two-hand reach observation (replaces the single-site ee_to_cube in both
  # actor and critic groups; they share the term objects, but be explicit).
  for group in ("actor", "critic"):
    cfg.observations[group].terms["ee_to_cube"] = ObservationTermCfg(
      func=bimanual_mdp.both_ee_to_object_distance,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=BOTH_GRASP_SITES),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )

  # Two-hand staged lift reward + hand-balance penalty.
  cfg.rewards["lift"] = RewardTermCfg(
    func=bimanual_mdp.bimanual_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "object_name": "cube",
      "reaching_std": 0.25,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=BOTH_GRASP_SITES),
    },
  )
  cfg.rewards["hands_symmetry"] = RewardTermCfg(
    func=bimanual_mdp.hands_symmetry_penalty,
    weight=-0.5,
    params={
      "object_name": "cube",
      "asset_cfg": SceneEntityCfg("robot", site_names=BOTH_GRASP_SITES),
    },
  )

  cfg.events["reset_table"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )
  for term in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[term].params["asset_cfg"].geom_names = BIMANUAL_FINGERTIP_GEOMS

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "right_palm"

  cx, cy = BIMANUAL_TABLE_CENTER
  box_z = TABLE_TOP_Z + BOX_HALF[2] + 0.005
  cfg.commands["lift_height"] = LiftingCommandCfg(
    entity_name="cube",
    resampling_time_range=(8.0, 12.0),
    debug_vis=True,
    difficulty="dynamic",
    object_pose_range=LiftingCommandCfg.ObjectPoseRangeCfg(
      x=(cx - 0.05, cx + 0.05),
      y=(cy - 0.04, cy + 0.04),
      z=(box_z, box_z),
      yaw=(-0.2, 0.2),
    ),
    target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
      x=(cx - 0.05, cx + 0.05),
      y=(cy - 0.04, cy + 0.04),
      z=(TABLE_TOP_Z + 0.12, TABLE_TOP_Z + 0.25),
    ),
  )

  cfg.viewer.body_name = "torso"
  # Two ORCA hands -> many collision geoms; raise the contact/constraint budget.
  cfg.sim.nconmax = 400
  cfg.sim.njmax = 1600

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def get_box_spec_entity() -> EntityCfg:
  return EntityCfg(spec_fn=get_box_spec)


# --------------------------------------------------------------------------- #
# Phase 3: whole-body (upper-body) lift -- arm + hand + lift + torso.
# --------------------------------------------------------------------------- #


def mabel_wholebody_lift_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Phase-1 lift task, but the lift column and torso pitch are also actuated.

  Adds 2 DOF (a coupled lift + torso) to the 24-DOF arm+hand, and raises the
  lift-goal height range so tall goals reward raising the whole body rather than
  over-extending the arm. Same right-arm workspace / table as Phase 1.
  """
  cfg = mabel_lift_cube_env_cfg(play=play)

  cfg.scene.entities["robot"] = get_mabel_robot_cfg(
    sides=("right",), keep_lift=True, keep_torso=True
  )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = action_scale(("right",), keep_lift=True, keep_torso=True)

  # Tall goals (up to +45 cm above the table) reward using the lift/torso.
  assert cfg.commands is not None
  cmd = cfg.commands["lift_height"]
  assert isinstance(cmd, LiftingCommandCfg)
  cmd.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(TABLE_CENTER[0] - 0.08, TABLE_CENTER[0] + 0.08),
    y=(TABLE_CENTER[1] - 0.08, TABLE_CENTER[1] + 0.08),
    z=(TABLE_TOP_Z + 0.10, TABLE_TOP_Z + 0.45),
  )
  return cfg

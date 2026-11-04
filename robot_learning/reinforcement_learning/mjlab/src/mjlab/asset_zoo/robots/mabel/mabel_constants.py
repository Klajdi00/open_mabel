"""MABEL robot constants for mjlab (tabletop dexterous manipulation).

Mirrors the YAM reference robot: the MJCF (derived from the canonical
``mabel_full.xml`` by :mod:`derive_model`) carries only geometry, joints,
collision geoms and sites; the position actuators, PD gains, effort limits,
home pose and collision config are attached here.

Per-joint gains and effort limits are the ones authored in ``mabel_full.xml``:
  * arm joints: kp=100, kv=20, effort = 40/27/7 N.m by DAMIAO motor tier
  * wrist + ORCA hand joints: kp=2, effort = 1 N.m (``hand_base`` default)

Joint armature / passive damping / frictionloss are left untouched (``None``),
so the physically-tuned per-motor values baked into the MJCF are preserved.
"""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.mabel.derive_model import (
  ARM_JOINTS,
  build_manip_spec,
  hand_joints,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# --------------------------------------------------------------------------- #
# Per-joint gains / effort limits (from mabel_full.xml <actuator> + defaults).
# --------------------------------------------------------------------------- #

ARM_KP = 100.0
ARM_KV = 20.0
# effort limit (N.m) per arm joint index 1..7 (DAMIAO J8009/J4340/J4310 peaks).
ARM_EFFORT = (40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 7.0)

HAND_KP = 2.0
HAND_KV = 0.1  # small derivative gain for sim stability (MJCF joint damping=0.1).
HAND_EFFORT = 1.0

# Body DOFs (whole-body phase). Gains from the mabel_full.xml <actuator> block.
# The lift is a heavy vertical slide (armature=1); a single actuator on
# "Lift Lower" drives both stages via the kept "Sync Lift" equality.
LIFT_KP = 10000.0
LIFT_KV = 1000.0
LIFT_EFFORT = 5000.0  # N (prismatic)
TORSO_KP = 5000.0
TORSO_KV = 100.0
TORSO_EFFORT = 400.0  # N.m (DM-J10422P peak)
# The body joints have very high kp (heavy links), so the effort/kp action scale
# is tiny; use explicit, usable scales instead so the policy can move them.
LIFT_ACTION_SCALE = 0.2  # m target per unit action (range 0..0.3175, coupled x2)
TORSO_ACTION_SCALE = 0.2  # rad per unit action (range -1.05..0.26)


def arm_actuators(side: str) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """One position actuator per arm joint (kp/kv/effort by tier)."""
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,),
      stiffness=ARM_KP,
      damping=ARM_KV,
      effort_limit=eff,
      # armature/frictionloss left None -> preserve MJCF motor values.
    )
    for name, eff in zip(ARM_JOINTS[side], ARM_EFFORT, strict=True)
  )


def hand_actuators(side: str) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """One position actuator per wrist + finger joint (uniform ORCA gains)."""
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=(name,),
      stiffness=HAND_KP,
      damping=HAND_KV,
      effort_limit=HAND_EFFORT,
    )
    for name in hand_joints(side)
  )


def side_actuators(side: str) -> tuple[BuiltinPositionActuatorCfg, ...]:
  return (*arm_actuators(side), *hand_actuators(side))


def body_actuators(
  keep_lift: bool = False, keep_torso: bool = False
) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """Lift + torso actuators for the whole-body phase."""
  acts: tuple[BuiltinPositionActuatorCfg, ...] = ()
  if keep_lift:
    acts += (
      BuiltinPositionActuatorCfg(
        target_names_expr=("Lift Lower",),
        stiffness=LIFT_KP,
        damping=LIFT_KV,
        effort_limit=LIFT_EFFORT,
      ),
    )
  if keep_torso:
    acts += (
      BuiltinPositionActuatorCfg(
        target_names_expr=("Torso",),
        stiffness=TORSO_KP,
        damping=TORSO_KV,
        effort_limit=TORSO_EFFORT,
      ),
    )
  return acts


# --------------------------------------------------------------------------- #
# Home keyframe.
# --------------------------------------------------------------------------- #

# Single-arm (Phase 1): right hand over a table on the robot's right, elbow bent.
# Values in radians; joints not listed default to 0. IK-designed in Phase 0/1.
RIGHT_HOME_JOINT_POS: dict[str, float] = {
  "Right Arm 1": -1.030,
  "Right Arm 2": -0.169,
  "Right Arm 3": 0.232,
  "Right Arm 4": -0.377,
  "Right Arm 5": 0.203,
  "Right Arm 6": 0.121,
  "Right Arm 7": -1.335,
}

# Bimanual (Phase 2): both hands reach a shared object on the midline, grasping
# it from either side. Symmetric IK homes (Phase 2 design).
BIMANUAL_HOME_JOINT_POS: dict[str, float] = {
  "Right Arm 1": 0.335,
  "Right Arm 2": -1.658,
  "Right Arm 3": -0.695,
  "Right Arm 4": 0.386,
  "Right Arm 5": 0.003,
  "Right Arm 6": -0.668,
  "Right Arm 7": -0.593,
  "Left Arm 1": -0.336,
  "Left Arm 2": 1.644,
  "Left Arm 3": 0.696,
  "Left Arm 4": -0.388,
  "Left Arm 5": -0.006,
  "Left Arm 6": 0.668,
  "Left Arm 7": 0.585,
}


def default_home(sides: tuple[str, ...]) -> dict[str, float]:
  """Pick a sensible home for the given arm set."""
  if sides == ("right",):
    return dict(RIGHT_HOME_JOINT_POS)
  # Multi-arm: use the bimanual home, filtered to the requested sides.
  return {
    k: v
    for k, v in BIMANUAL_HOME_JOINT_POS.items()
    if any(k.startswith(s.capitalize()) for s in sides)
  }


def home_state(joint_pos: dict[str, float]) -> EntityCfg.InitialStateCfg:
  jp: dict[str, float] = {".*": 0.0}
  jp.update(joint_pos)
  return EntityCfg.InitialStateCfg(joint_pos=jp, joint_vel={".*": 0.0})


# --------------------------------------------------------------------------- #
# Entity config.
# --------------------------------------------------------------------------- #


def get_mabel_robot_cfg(
  sides: tuple[str, ...] = ("right",),
  *,
  include_hand: bool = True,
  home: dict[str, float] | None = None,
  keep_lift: bool = False,
  keep_torso: bool = False,
) -> EntityCfg:
  actuators: tuple[BuiltinPositionActuatorCfg, ...] = ()
  for side in sides:
    actuators += arm_actuators(side)
    if include_hand:
      actuators += hand_actuators(side)
  actuators += body_actuators(keep_lift=keep_lift, keep_torso=keep_torso)

  return EntityCfg(
    init_state=home_state(home if home is not None else default_home(sides)),
    spec_fn=lambda: build_manip_spec(
      sides=sides,
      include_hand=include_hand,
      keep_lift=keep_lift,
      keep_torso=keep_torso,
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=0.9,
    ),
    collisions=(),  # preserve mabel_full.xml collision classes.
  )


# --------------------------------------------------------------------------- #
# Action scale (per-actuator), following the YAM convention: 0.25 * effort / kp.
# --------------------------------------------------------------------------- #


def action_scale(
  sides: tuple[str, ...],
  *,
  include_hand: bool = True,
  keep_lift: bool = False,
  keep_torso: bool = False,
) -> dict[str, float]:
  scale: dict[str, float] = {}
  for side in sides:
    for name, eff in zip(ARM_JOINTS[side], ARM_EFFORT, strict=True):
      scale[name] = 0.25 * eff / ARM_KP
    if include_hand:
      for name in hand_joints(side):
        scale[name] = 0.25 * HAND_EFFORT / HAND_KP
  if keep_lift:
    scale["Lift Lower"] = LIFT_ACTION_SCALE
  if keep_torso:
    scale["Torso"] = TORSO_ACTION_SCALE
  return scale


MABEL_ACTION_SCALE: dict[str, float] = action_scale(("right",))


if __name__ == "__main__":
  cfg = get_mabel_robot_cfg()
  spec = cfg.spec_fn()
  m = spec.compile()
  print(f"compiled: nq={m.nq} nv={m.nv} nu(before mjlab)={m.nu}")
  assert cfg.articulation is not None
  print(f"actuator cfgs: {len(cfg.articulation.actuators)}")
  print(
    "action scale sample:",
    {k: round(v, 4) for k, v in list(MABEL_ACTION_SCALE.items())[:9]},
  )

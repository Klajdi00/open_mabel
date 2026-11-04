"""Derive reduced MuJoCo task models for MABEL from the canonical MJCF.

The single source of truth for the MABEL robot is
``simulation/mabel_mujoco/models/mabel_full.xml`` (a dual-arm humanoid on a
3-module swerve base, ~59 actuators, floating base). For tabletop RL we do not
want the whole robot: we want a fixed-base pedestal carrying one (or both) 7-DOF
arm + 16-DOF ORCA hand, with grasp / fingertip sites the reward and observation
terms need, and *no* actuators/sensors/tendons in the XML (mjlab attaches
position actuators programmatically from ``mabel_constants.py``, exactly like the
YAM reference robot).

This module builds those reduced ``mujoco.MjSpec`` objects at load time, so the
task models never drift from ``mabel_full.xml`` -- editing the canonical MJCF
automatically flows through here. It is also runnable as a CLI to export a
static XML and/or open the viewer for inspection.

Phase map (``keep_*`` flags select the progression):
  * Phase 1 single-arm:  build_manip_spec(sides=("right",))
  * Phase 2 dual-arm:    build_manip_spec(sides=("left", "right"))
  * Phase 3 whole-body:  build_manip_spec(..., keep_lift=True, keep_torso=True,
                                          keep_base=True, fixed_base=False)
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

# --------------------------------------------------------------------------- #
# Locating the canonical MJCF.
# --------------------------------------------------------------------------- #

_REL_MJCF = Path("simulation/mabel_mujoco/models/mabel_full.xml")


def find_mabel_root() -> Path:
  """Resolve the MABEL repo root (holding ``simulation/mabel_mujoco``).

  Honors ``$MABEL_ROOT`` (the convention used by ``simulation_bridge``), then
  falls back to walking up from this file until the canonical MJCF is found.
  """
  env = os.environ.get("MABEL_ROOT")
  if env:
    root = Path(env).expanduser()
    if (root / _REL_MJCF).exists():
      return root
  for parent in Path(__file__).resolve().parents:
    if (parent / _REL_MJCF).exists():
      return parent
  raise FileNotFoundError(
    "Could not locate mabel_full.xml. Set $MABEL_ROOT to the MABEL repo root."
  )


def mabel_full_xml() -> Path:
  return find_mabel_root() / _REL_MJCF


# --------------------------------------------------------------------------- #
# Joint / body inventory (names match mabel_full.xml exactly).
# --------------------------------------------------------------------------- #

# 7-DOF arm joints per side.
ARM_JOINTS = {
  "left": tuple(f"Left Arm {i}" for i in range(1, 8)),
  "right": tuple(f"Right Arm {i}" for i in range(1, 8)),
}
# 1 wrist + 16 finger joints per side (the ORCA hand, direct-drive).
_FINGERS = {
  "thumb": ("MCP", "Abd", "PIP", "DIP"),
  "index": ("Abd", "MCP", "PIP"),
  "middle": ("Abd", "MCP", "PIP"),
  "ring": ("Abd", "MCP", "PIP"),
  "pinky": ("Abd", "MCP", "PIP"),
}


def hand_joints(side: str) -> tuple[str, ...]:
  pfx = side.capitalize()
  out = [f"{pfx} Wrist"]
  for finger, dofs in _FINGERS.items():
    for dof in dofs:
      out.append(f"{pfx} {finger.capitalize()} {dof}")
  return tuple(out)


# Distal fingertip *bodies* (for fingertip sites + named tip collisions).
def fingertip_bodies(side: str) -> dict[str, str]:
  return {
    "thumb": f"{side}_thumb_dp",
    "index": f"{side}_index_ip",
    "middle": f"{side}_middle_ip",
    "ring": f"{side}_ring_ip",
    "pinky": f"{side}_pinky_ip",
  }


# Subtree roots that are always removed for tabletop manipulation.
_SWERVE_ROOTS = ("fl_swerve_housing", "fr_swerve_housing", "b_swerve_housing")
_NECK_ROOT = "head_mount"
_LIDAR_BODY = "lidar"
_ARM_MOUNT = {"left": "l_arm_mount", "right": "r_arm_mount"}
# Palm-local grasp centre (metres), from the wrist-cam annotation in the MJCF.
_GRASP_SITE_LOCAL = (0.0, 0.0, 0.102)
_PALM_BODY = {"left": "left_palm", "right": "right_palm"}


# --------------------------------------------------------------------------- #
# Spec surgery helpers.
# --------------------------------------------------------------------------- #


def _delete_all(spec: mujoco.MjSpec, collection: str) -> int:
  items = list(getattr(spec, collection))
  for it in items:
    spec.delete(it)
  return len(items)


def _try_delete_body(spec: mujoco.MjSpec, name: str) -> bool:
  body = spec.body(name)
  if body is None:
    return False
  spec.delete(body)  # cascades the whole subtree
  return True


def _delete_joint(spec: mujoco.MjSpec, name: str) -> bool:
  jnt = spec.joint(name)
  if jnt is None:
    return False
  spec.delete(jnt)
  return True


def _add_hand_sites(spec: mujoco.MjSpec, side: str) -> None:
  """Add a palm grasp site (TCP) + one site per fingertip."""
  palm = spec.body(_PALM_BODY[side])
  assert palm is not None, f"missing palm body {_PALM_BODY[side]}"
  palm.add_site(
    name=f"{side}_grasp_site",
    pos=_GRASP_SITE_LOCAL,
    size=[0.005],
    group=4,
    rgba=[0, 1, 0, 1],
  )
  for finger, body_name in fingertip_bodies(side).items():
    body = spec.body(body_name)
    if body is None:
      continue
    # Fingertip pads sit ~1.5 cm along the distal bone's local +z.
    body.add_site(
      name=f"{side}_{finger}_tip_site",
      pos=[0.0, 0.0, 0.015],
      size=[0.004],
      group=4,
      rgba=[1, 0.5, 0, 1],
    )


def _name_fingertip_collisions(spec: mujoco.MjSpec, side: str) -> None:
  """Give the distal fingertip collision geoms stable names.

  mabel_full.xml tags collision geoms by class (``hand_collision``) but leaves
  them unnamed. The task's friction domain-randomization and contact terms
  select geoms by name, so we name the fingertip tip geoms
  ``{side}_{finger}_tip_collision``.
  """
  for finger, body_name in fingertip_bodies(side).items():
    body = spec.body(body_name)
    if body is None:
      continue
    idx = 0
    geom = body.first_geom()
    while geom is not None:
      # Collision geoms in this model reference "*_c_*" collision meshes.
      is_collision = geom.contype != 0 or geom.conaffinity != 0
      if is_collision and not geom.name:
        geom.name = f"{side}_{finger}_tip_collision" + ("" if idx == 0 else f"_{idx}")
        idx += 1
      geom = body.next_geom(geom)


# --------------------------------------------------------------------------- #
# Public builder.
# --------------------------------------------------------------------------- #


def build_manip_spec(
  sides: tuple[str, ...] = ("right",),
  *,
  include_hand: bool = True,
  fixed_base: bool = True,
  keep_lift: bool = False,
  keep_torso: bool = False,
  keep_neck: bool = False,
  keep_base: bool = False,
) -> mujoco.MjSpec:
  """Build a reduced MABEL manipulation spec from the canonical MJCF."""
  for s in sides:
    if s not in ("left", "right"):
      raise ValueError(f"side must be 'left'/'right', got {s!r}")

  spec = mujoco.MjSpec.from_file(str(mabel_full_xml()))

  # 1. Strip everything that references actuation / sensing / coupling. mjlab
  #    re-adds position actuators from mabel_constants.py.
  _delete_all(spec, "sensors")  # 360-ray lidar + imu, all irrelevant here.
  _delete_all(spec, "actuators")
  _delete_all(spec, "tendons")  # "Total Lift" (unused; we actuate the joint)
  # Keep the "Sync Lift" equality when the lift stays active -- it couples the
  # two lift slides so a single actuator on "Lift Lower" drives both. Otherwise
  # strip all equalities (the lift joints are frozen anyway).
  for eq in list(spec.equalities):
    if keep_lift and eq.name == "Sync Lift":
      continue
    spec.delete(eq)

  # 2. Base: fixed pedestal for tabletop tasks.
  if fixed_base:
    free = spec.joint("base_root")
    if free is not None:
      spec.delete(free)

  # 3. Prune unused subtrees.
  _try_delete_body(spec, _LIDAR_BODY)
  if not keep_base:
    for root in _SWERVE_ROOTS:
      _try_delete_body(spec, root)
  if not keep_neck:
    _try_delete_body(spec, _NECK_ROOT)
  # Remove the arm(s) we are not using.
  for side in ("left", "right"):
    if side not in sides:
      _try_delete_body(spec, _ARM_MOUNT[side])

  # 4. Freeze pedestal DOFs we are not training (delete the joints -> the
  #    bodies stay rigidly at their zero pose).
  if not keep_lift:
    _delete_joint(spec, "Lift Lower")
    _delete_joint(spec, "Lift Upper")
  if not keep_torso:
    _delete_joint(spec, "Torso")

  # 5. Optionally drop the hand entirely (arm-only), else add sites + names.
  for side in sides:
    if not include_hand:
      tower = spec.body(f"{side}_tower")
      if tower is not None:
        spec.delete(tower)
      continue
    _add_hand_sites(spec, side)
    _name_fingertip_collisions(spec, side)

  return spec


def spec_stats(spec: mujoco.MjSpec) -> dict:
  m = spec.compile()
  return {
    "nbody": m.nbody,
    "nq": m.nq,
    "nv": m.nv,
    "njnt": m.njnt,
    "ngeom": m.ngeom,
    "nsite": m.nsite,
    "nu": m.nu,
  }


def actuated_joint_names(spec: mujoco.MjSpec) -> list[str]:
  """Non-free hinge/slide joints remaining in the spec (the action DOFs)."""
  names = []
  for j in spec.joints:
    if j.type == mujoco.mjtJoint.mjJNT_FREE:
      continue
    names.append(j.name)
  return names


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


def main() -> None:
  import argparse

  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--sides", nargs="+", default=["right"], choices=["left", "right"])
  p.add_argument("--no-hand", action="store_true")
  p.add_argument("--keep-lift", action="store_true")
  p.add_argument("--keep-torso", action="store_true")
  p.add_argument("--keep-neck", action="store_true")
  p.add_argument("--keep-base", action="store_true")
  p.add_argument("--out", type=str, default=None, help="write XML here")
  p.add_argument("--view", action="store_true", help="launch mujoco viewer")
  args = p.parse_args()

  spec = build_manip_spec(
    sides=tuple(args.sides),
    include_hand=not args.no_hand,
    keep_lift=args.keep_lift,
    keep_torso=args.keep_torso,
    keep_neck=args.keep_neck,
    keep_base=args.keep_base,
    fixed_base=not args.keep_base,
  )
  stats = spec_stats(spec)
  print("sides:", args.sides)
  print("stats:", stats)
  joints = actuated_joint_names(spec)
  print(f"actuated joints ({len(joints)}):")
  for j in joints:
    print("   ", j)

  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Absolute meshdir so the exported XML resolves meshes from anywhere.
    spec.meshdir = str((find_mabel_root() / "simulation/mabel_mujoco/assets").resolve())
    out.write_text(spec.to_xml())
    print("wrote", out)

  if args.view:
    import mujoco.viewer

    mujoco.viewer.launch(spec.compile())


if __name__ == "__main__":
  main()

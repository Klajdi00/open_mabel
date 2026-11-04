"""
Map a recorded episode's 59-actuator vector to the 60 named GLB rig joints that
``website/assets/robot-viewer.js`` animates, plus a base pose for the chassis.

The mapping is the same one the live teleop bridge uses (see
``server/sim_teleop_bridge.py`` _TELEMETRY_JOINTS): it is essentially a name
normalization ("Left Arm 1" -> "left_arm_1", "FL Steer" -> "fl_steer") with ONE
special case — the single ``Lift`` actuator drives the two prismatic GLB stages
``lift_lower`` + ``lift_upper`` (each = Lift/2). Actuator names come from each
episode's own ``schema_json`` so this never goes stale after an MJCF re-sync.

Quaternion note: ``obs/pose/*`` stores MuJoCo ``xquat`` order (w, x, y, z), NOT
xyzw — base yaw is derived accordingly.
"""
import math


def normalize(name: str) -> str:
    return name.lower().replace(" ", "_")


def glb_from_actuators(joint_pos, actuator_names):
    """joint_pos: length-59 sequence (one frame of obs/joint_pos OR action/ctrl).
    Returns {glb_joint_name: radians} with 60 entries (Lift -> two stages)."""
    out = {}
    for name, val in zip(actuator_names, joint_pos):
        v = float(val)
        if name == "Lift":
            out["lift_lower"] = 0.5 * v
            out["lift_upper"] = 0.5 * v
        else:
            out[normalize(name)] = v
    return out


def base_from_chassis(pose7):
    """obs/pose/chassis = [x, y, z, qw, qx, qy, qz] (MuJoCo xquat order).
    Returns {x, y, z, yaw, quat:[w,x,y,z]} for the viewer's base placement."""
    x, y, z, qw, qx, qy, qz = (float(c) for c in pose7[:7])
    yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    return {"x": x, "y": y, "z": z, "yaw": yaw, "quat": [qw, qx, qy, qz]}


def validate_against_manifest(actuator_names, manifest_joint_names):
    """Return (missing, extra) between the GLB keys we'd produce and the rig
    manifest — a startup sanity check that catches drift after a model re-sync."""
    produced = set()
    for name in actuator_names:
        if name == "Lift":
            produced.update(("lift_lower", "lift_upper"))
        else:
            produced.add(normalize(name))
    want = set(manifest_joint_names)
    return sorted(want - produced), sorted(produced - want)

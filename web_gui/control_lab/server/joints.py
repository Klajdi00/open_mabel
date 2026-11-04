"""
Joint registry for the Control Lab: the canonical wire joint names
(matching web_gui/shared_assets/mabel_joints.json, same as every other studio) and
the MuJoCo-name mapping built live from the loaded model.

Wire names derive from MuJoCo names by lowercasing + underscoring
("Left Arm 3" -> left_arm_3, "FL Steer" -> fl_steer, "Left Thumb MCP" ->
left_thumb_mcp, "Lift Lower" -> lift_lower) — the same convention the sim
bridge uses, so the shared GLB rig poses identically here.
"""

_FINGERS = ["thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip",
            "index_abd", "index_mcp", "index_pip",
            "middle_abd", "middle_mcp", "middle_pip",
            "ring_abd", "ring_mcp", "ring_pip",
            "pinky_abd", "pinky_mcp", "pinky_pip"]

MODULES = [
    ("base", "Base", ["fl_steer", "fl_drive", "fr_steer", "fr_drive",
                      "b_steer", "b_drive"]),
    ("lift_torso", "Lift/Torso", ["lift_lower", "lift_upper", "torso"]),
    ("neck", "Neck", ["neck_yaw", "neck_pitch", "neck_roll"]),
    ("left_arm", "L Arm", [f"left_arm_{i}" for i in range(1, 8)]),
    ("right_arm", "R Arm", [f"right_arm_{i}" for i in range(1, 8)]),
    ("left_hand", "L Hand", ["left_wrist"] + ["left_" + f for f in _FINGERS]),
    ("right_hand", "R Hand", ["right_wrist"] + ["right_" + f for f in _FINGERS]),
]
JOINT_NAMES = [n for _, _, names in MODULES for n in names]
JOINT_SET = set(JOINT_NAMES)


def wire_name(mj_name: str) -> str:
    return mj_name.strip().lower().replace(" ", "_")


def build_maps(model):
    """(wire->qposadr, wire->jointname) for every model joint whose wire name
    is in the canonical registry. The lift cascade maps both lift_lower and
    lift_upper; steer/drive wheels come from their joints directly."""
    import mujoco
    qadr, mjname = {}, {}
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not nm:
            continue
        w = wire_name(nm)
        if w in JOINT_SET:
            qadr[w] = int(model.jnt_qposadr[j])
            mjname[w] = nm
    return qadr, mjname


def actuator_ranges(model):
    """{wire_id: [lo, hi]} control range for every position-servo actuator, so the
    joint sliders can span exactly what the joint can be commanded to. Grips are
    synthetic [0,1] (a single curl value, no actuator)."""
    import mujoco
    out = {"left_grip": [0.0, 1.0], "right_grip": [0.0, 1.0]}
    for aid in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if not nm:
            continue
        lo, hi = model.actuator_ctrlrange[aid]
        out[wire_name(nm)] = [float(lo), float(hi)]
    return out


def schema(model=None):
    s = {"names": JOINT_NAMES,
         "modules": [{"key": k, "label": l, "joints": j} for k, l, j in MODULES]}
    if model is not None:
        s["ranges"] = actuator_ranges(model)
    return s

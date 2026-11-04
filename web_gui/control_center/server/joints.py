"""
Canonical MABEL joint registry: the 60 kinematic joints / 59 actuated DOF,
grouped into the 7 UI modules. Single source for the telemetry schema, the
mock plant, and the tests. Names are URDF/`/joint_states` names (see
web_gui/shared_assets/mabel_joints.json; lift_lower+lift_upper share one actuator).
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
JOINT_GROUP = {n: key for key, _, names in MODULES for n in names}
DOF_COUNT = len(JOINT_NAMES) - 1        # lift_lower+lift_upper = one actuator

assert len(JOINT_NAMES) == 60 and DOF_COUNT == 59


def schema():
    """The one-shot schema message body sent to each telemetry client."""
    return {
        "names": JOINT_NAMES,
        "modules": [{"key": k, "label": lab, "joints": names}
                    for k, lab, names in MODULES],
        "dof": DOF_COUNT,
    }

"""
MABEL embodiment registry for the trainer — joint groups, DOF, cameras, and the
module *scope presets* that drive modular input/output selection.

The group table is derived from the collection pipeline config (the single
source of truth, shared with the MJCF), in the canonical actuator order verified
against a real episode schema:

    base(6) | lift_torso(2) | neck(3) | left_arm(8) | left_hand(16) |
    right_arm(8) | right_hand(16)  = 59

so index maps are correct and generalize to any MABEL actuator set.
"""
import functools

from . import config as CFG

# Canonical group ORDER as it appears in obs/joint_pos and action/ctrl (per-side
# arm THEN hand), not the declaration order in the collection config.
GROUP_ORDER = ["base", "lift_torso", "neck",
               "left_arm", "left_hand", "right_arm", "right_hand"]

# Hardcoded fallback names (used only if the collection config can't be imported).
_FINGERS = ["Thumb MCP", "Thumb Abd", "Thumb PIP", "Thumb DIP",
            "Index Abd", "Index MCP", "Index PIP",
            "Middle Abd", "Middle MCP", "Middle PIP",
            "Ring Abd", "Ring MCP", "Ring PIP",
            "Pinky Abd", "Pinky MCP", "Pinky PIP"]
_FALLBACK = {
    "base": ["FL Steer", "FL Drive", "FR Steer", "FR Drive", "B Steer", "B Drive"],
    "lift_torso": ["Lift", "Torso"],
    "neck": ["Neck Yaw", "Neck Pitch", "Neck Roll"],
    "left_arm": [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"],
    "left_hand": [f"Left {f}" for f in _FINGERS],
    "right_arm": [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"],
    "right_hand": [f"Right {f}" for f in _FINGERS],
}

# Friendly module scopes → the groups they cover. `custom` is resolved from an
# explicit group multiselect. All generalize because they compose GROUP_ORDER.
SCOPES = {
    "whole_body": GROUP_ORDER,
    "upper_body": ["neck", "left_arm", "left_hand", "right_arm", "right_hand"],
    "arms":       ["left_arm", "right_arm"],
    "hands":      ["left_hand", "right_hand"],
    "arms_hands": ["left_arm", "left_hand", "right_arm", "right_hand"],
    "left_arm":   ["left_arm", "left_hand"],
    "right_arm":  ["right_arm", "right_hand"],
    "base":       ["base"],
    "mobile_base": ["base", "lift_torso"],
}

SCOPE_LABELS = {
    "whole_body": "Whole body", "upper_body": "Upper body", "arms": "Both arms",
    "hands": "Both hands", "arms_hands": "Arms + hands", "left_arm": "Left arm + hand",
    "right_arm": "Right arm + hand", "base": "Mobile base", "mobile_base": "Base + lift",
}


@functools.lru_cache(maxsize=1)
def _groups():
    """Ordered dict {group: [actuator names]} in canonical index order."""
    c = getattr(CFG, "CFG", None)
    out = {}
    if c is not None:
        src = {
            "base": list(getattr(c, "BASE_ACTUATORS", _FALLBACK["base"])),
            "lift_torso": list(getattr(c, "LIFT_TORSO", _FALLBACK["lift_torso"])),
            "neck": list(getattr(c, "NECK", _FALLBACK["neck"])),
            "left_arm": list(getattr(c, "LEFT_ARM", _FALLBACK["left_arm"])),
            "left_hand": list(getattr(c, "LEFT_HAND", _FALLBACK["left_hand"])),
            "right_arm": list(getattr(c, "RIGHT_ARM", _FALLBACK["right_arm"])),
            "right_hand": list(getattr(c, "RIGHT_HAND", _FALLBACK["right_hand"])),
        }
    else:
        src = dict(_FALLBACK)
    for g in GROUP_ORDER:
        out[g] = src.get(g, _FALLBACK[g])
    return out


@functools.lru_cache(maxsize=1)
def _index_map():
    """{group: [global indices]} and the flat ordered name list."""
    groups = _groups()
    idx, names, cur = {}, [], 0
    for g in GROUP_ORDER:
        n = len(groups[g])
        idx[g] = list(range(cur, cur + n))
        names.extend(groups[g])
        cur += n
    return idx, names


def group_indices():
    return _index_map()[0]


def actuator_names():
    return _index_map()[1]


def dof():
    return len(actuator_names())


# Camera → sensor-group classification for the frequency breakdown.
CAMERA_GROUPS = {
    "head_cam": ["zed_left", "zed_right"],
    "wrist_cam": ["l_wrist_cam", "r_wrist_cam"],
    "base_cam": ["body_cam", "body_front_cam", "body_back_cam"],
}


def camera_group(name):
    for g, members in CAMERA_GROUPS.items():
        if name in members:
            return g
    return "other_cam"


def cameras():
    """List of {name,w,h,rgb,depth} for the 7 on-board cameras."""
    cams = getattr(CFG, "CAMERAS", []) or []
    if cams:
        return [dict(c) for c in cams]
    # fallback: the known 7-camera rig
    names = ["l_wrist_cam", "r_wrist_cam", "zed_left", "zed_right",
             "body_cam", "body_front_cam", "body_back_cam"]
    return [{"name": n, "w": 480, "h": 270, "rgb": True, "depth": False} for n in names]


def registry():
    """One record describing the rig, for the UI Inputs/Info panels."""
    groups = _groups()
    idx = group_indices()
    return {
        "robot": "mabel",
        "dof": dof(),
        "group_order": GROUP_ORDER,
        "groups": {g: {"names": groups[g], "indices": idx[g], "n": len(groups[g])}
                   for g in GROUP_ORDER},
        "actuator_names": actuator_names(),
        "cameras": cameras(),
        "camera_groups": CAMERA_GROUPS,
        "scopes": {k: {"groups": v, "label": SCOPE_LABELS.get(k, k)}
                   for k, v in SCOPES.items()},
        "action_spec": {"space": "joint_pos", "frame": "base",
                        "gripper": "continuous", "is_delta": False},
    }


def resolve_scope(scope="whole_body", groups=None):
    """Return the ordered list of group names for a scope preset or a custom
    group multiselect (``groups`` overrides ``scope`` when scope == 'custom')."""
    if scope == "custom" and groups:
        return [g for g in GROUP_ORDER if g in set(groups)]
    return SCOPES.get(scope, GROUP_ORDER)

"""End effectors the arm retargeter can drive: MABEL's ORCA hand v1 (default) or the ORIGINAL OpenArm v1
parallel gripper (Enactic openarm_mujoco v1/openarm.xml, meshes in simulation/mabel_mujoco/models/meshes).

With the gripper, the arm's tracked point is the gripper's BASE FACE — TRACK_OFFSET out from the WRIST CENTRE
(where the J5, J6 and J7 axes meet) along the approach axis — instead of the ORCA wrist anchor (tracked_point),
and the gripper sits on J7 exactly as on an OpenArm: in OpenArm's link7 frame
(origin = the wrist centre, +y = the J7 axis, +z = the approach direction) the hand body is at
(0, -4.5, 100.1) mm, its bottom face 89.6 mm out, the fingertip TCP 180.1 mm out, and each finger slides
0-44 mm along +-y (88 mm total opening). The operator's hand closure sets that opening (GripperAperture).

    from mabel.teleop_engine.end_effector import wrist_centre, tracked_point, grasp_axes, gripper_frame, GripperAperture
"""
from __future__ import annotations

import os

import numpy as np
import mujoco

KINDS = ("orca", "gripper")

# OpenArm v1 gripper, from openarm.xml (link7 -> link8 -> hand -> fingers). Lengths in metres, frame = link7.
GRIPPER = dict(
    hand_pos=(0.0, -0.0045, 0.1001),               # link8 (1e-6, 0.0205, 0) + hand (0, -0.025, 0.1001)
    hand_geom_pos=(0.0, 0.005, -0.6585),           # hand_0/1.obj are in arm-global mm: scale 0.001, then this
    bottom=0.0896,                                 # wrist centre -> the hand's bottom face, along +z
    tcp=0.1801,                                    # wrist centre -> TCP (between the fingertips), along +z
    tip=0.19560089209,                             # distal mesh end; distinct from the grasp-centre TCP
    fingers=(                                      # (body pos in hand, slide axis, geom pos, mesh y-flip)
        dict(name="right_finger", pos=(0.0, 0.0, 0.015), axis=(0.0, -1.0, 0.0), geom_pos=(0.0, 0.05, -0.673001), flip=True),
        dict(name="left_finger", pos=(0.0, 0.012, 0.015), axis=(0.0, 1.0, 0.0), geom_pos=(0.0, -0.05, -0.673001), flip=False),
    ),
    finger_range=(0.0, 0.044),                     # per finger; the opening is twice this
    # OpenArm's link7: the wrist housing and the gripper's drive motor (link7_1: a 57 mm cylinder 49.5-88.5 mm
    # out, ending where the hand's bottom face starts). On MABEL it replaces the ORCA hand-mount adapter.
    housing_geom_pos=(0.0, 0.0, -0.5585),
    meshes=dict(housing=("visual/arm/link7_0.obj", "visual/arm/link7_1.obj"),
                hand=("visual/gripper/hand_0.obj", "visual/gripper/hand_1.obj"),
                finger=("visual/gripper/finger_0.obj", "visual/gripper/finger_1.obj")),
    colors=dict(link7_0="#cbcbcb", link7_1="#3f3f3f", hand_0="#cbcbcb", hand_1="#cbcbcb",
                finger_0="#d9d9d9", finger_1="#3f3f3f"),
)
OPEN_MAX = 2.0 * GRIPPER["finger_range"][1]       # 0.088 m
# Where the operator's wrist lands on the gripper, metres out from the wrist centre along the approach axis.
# The wrist centre itself (0) sits 21.6 cm inboard of the ORCA wrist anchor, and the arm map scales motion by
# the robot's reach to the tracked point, so tracking it shrank the reach 0.669 -> 0.459 m and every operator
# wrist motion by ~35 % (robot/operator path 1.39 -> 0.88 on the 2026-10-15 take): the "slow" gripper. The
# base face (bottom, 89.6 mm) gives reach 0.546 m and path 1.05 at the same tracking error. The TCP (180.1 mm)
# overshoots: rotations about it drive the arm, left p95 error 8.7 -> 30 cm. Env MABEL_GRIPPER_TRACK (m).
TRACK_OFFSET = float(os.environ.get("MABEL_GRIPPER_TRACK", str(GRIPPER["bottom"])))
# Which way the gripper POINTS. The arm's orientation target is the operator's metacarpal axis (wrist -> middle
# metacarpal) and palm normal, and the gripper's approach is rigid on it — but a grasping hand pinches PALMWARD of
# its metacarpals: thumb+index tip midpoint, while the tips are < 3 cm apart, sits 25-40 deg (median 37) palmward
# on the three 2026-10 Vision Pro takes (13-20 deg with the hand open). So a hand pointing down to grasp left the
# gripper ~35 deg flatter than the fingers. The target is pitched GRASP_TILT about the palm's index->pinky axis,
# toward the palm: a constant per-hand rotation offset, as the upstream GMR wrist keyframes carry
# (rot_offset = R_human^-1 R_robot, measured once). Env MABEL_GRIPPER_TILT_DEG; 0 = the metacarpal axis.
GRASP_TILT = np.radians(float(os.environ.get("MABEL_GRIPPER_TILT_DEG", "35")))


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def wrist_centre(model, side):
    """The arm's wrist centre as (J7 body id, point in that body's frame): the point on the J7 axis closest to
    the J5 and J6 axes (on MABEL/NOVA it is 20 mm along J7 from its anchor; J6 meets it exactly, J5 within
    2.5 mm). A point ON the J7 axis is fixed in both the J6 and J7 bodies, so it does not move with J7."""
    S = side.capitalize()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    lines = {i: (d.xanchor[_jid(model, f"{S} Arm {i}")].copy(), d.xaxis[_jid(model, f"{S} Arm {i}")].copy())
             for i in (5, 6, 7)}
    p0, u = lines[7]
    a = b = 0.0
    for i in (5, 6):
        q, v = lines[i]
        P = np.eye(3) - np.outer(v, v)
        a += u @ P @ u
        b += u @ P @ (q - p0)
    c = p0 + (b / a) * u
    body = int(model.jnt_bodyid[_jid(model, f"{S} Arm 7")])
    R = d.xmat[body].reshape(3, 3)
    return body, R.T @ (c - d.xpos[body])


def gripper_frame(model, side, palm_bid, R_rob):
    """4x4 pose of OpenArm's link7 frame (wrist centre; +y = J7 axis, +z = approach) in the J7 BODY frame.
    The approach is the ORCA palm's forward axis with the ORCA wrist at 0, made perpendicular to J7 — the
    direction the hand pointed, so the arm map's palm orientation carries over unchanged."""
    S = side.capitalize()
    body, c = wrist_centre(model, side)
    d = mujoco.MjData(model)
    d.qpos[model.jnt_qposadr[_jid(model, f"{S} Wrist")]] = 0.0
    mujoco.mj_forward(model, d)
    Rb = d.xmat[body].reshape(3, 3)
    y = Rb.T @ d.xaxis[_jid(model, f"{S} Arm 7")]
    z = Rb.T @ (d.xmat[palm_bid].reshape(3, 3) @ R_rob)[:, 0]
    z = z - (z @ y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = c
    return T


def tracked_point(model, side, palm_bid, R_rob):
    """The gripper's tracked point as (J7 body id, point in that body's frame): the wrist centre pushed
    TRACK_OFFSET out along the approach axis (gripper_frame's +z). Fixed on J7, like the wrist centre."""
    body, _ = wrist_centre(model, side)
    T = gripper_frame(model, side, palm_bid, R_rob)
    return body, T[:3, 3] + TRACK_OFFSET * T[:3, 2]


def distal_tip(model, side, palm_bid, R_rob):
    """Physical distal-tip reference (body, local point), rather than the grasp centre."""
    body, _ = wrist_centre(model, side)
    T = gripper_frame(model, side, palm_bid, R_rob)
    return body, T[:3, 3] + GRIPPER["tip"] * T[:3, 2]


def gripper_collision_shapes(opening=None):
    """Conservative capsules in the OpenArm link7 frame, metres.

    Fitted to the same transformed OBJ vertices as the app rig, including the
    housing motor and BOTH fingers over their complete 0–44 mm stroke. The
    distal envelope ends only 0.5 mm beyond the actual tip. ``opening`` is one
    finger's maximum measured/commanded slide position; the envelope includes
    every opening from closed through that value. Unknown feedback retains the
    full stroke rather than assuming a commanded closure already happened.
    """
    stroke = GRIPPER["finger_range"][1]
    opening = stroke if opening is None else float(np.clip(opening, 0., stroke))
    inset = stroke - opening
    return {
        "housing": ((0., 0., -.017500), (0., 0., .095501), .038684),
        "palm": ((.000001, -.083558, .105101), (.000001, .084558, .105101), .032443),
        "finger_mount": ((.000001, -.076000 + inset, .135349), (.000001, .079000 - inset, .135349), .039216),
        "finger_tip": ((0., -.065821 + inset, .17160089209), (0., .068821 - inset, .17160089209), .0245),
    }


def grasp_axes(fwd, palmar, tilt=None):
    """(fwd, palmar) target axes pitched `tilt` (default GRASP_TILT) toward the palm about their common normal."""
    t = GRASP_TILT if tilt is None else tilt
    f, n = np.asarray(fwd, float), np.asarray(palmar, float)
    return np.cos(t) * f + np.sin(t) * n, np.cos(t) * n - np.sin(t) * f


def _pos(m):
    M = np.asarray(m, float).reshape(4, 4)
    return M[:3, 3] if np.allclose(M[3], [0.0, 0.0, 0.0, 1.0]) else M[3, :3]


class GripperAperture:
    """Operator hand closure -> gripper opening [m, 0..0.088], one per side.

    The grasp aperture is the gap between the thumb tip and the nearer of the index / middle tips — what a
    pinch closes and what a power grasp wraps an object with — so the gripper opens to about the width of
    the thing in the hand: opening = gap - PAD, 1:1, so closing on an object closes the jaws on its width.
    PAD is the gap the Vision Pro reports with the fingertips touching: the tracked tip joints sit inside the
    finger pads, 17 / 26 mm apart at p1 / p5 of the gap on the 2026-10-15 take, so 22 mm. Clipped to the
    gripper's range, then a first-order low-pass (TAU) so hand-tracking jitter does not chatter the jaws. A side
    with no tracked hand holds its last opening. (A relaxed open hand, ~96 mm gap at p90, opens ~74 mm; full
    88 mm needs a spread hand.)"""

    PAD = 0.022
    TAU = 0.06                                     # s

    def __init__(self):
        self.open = OPEN_MAX
        self._t = None

    def update(self, joints, t=None):
        if joints:
            try:
                th = _pos(joints["thumb_tip"])
                gap = min(float(np.linalg.norm(th - _pos(joints[k]))) for k in ("index_tip", "middle_tip"))
            except (KeyError, ValueError):
                return self.open
            target = float(np.clip(gap - self.PAD, 0.0, OPEN_MAX))
            if t is None or self._t is None:
                a = 1.0
            else:
                a = 1.0 - np.exp(-max(float(t) - self._t, 0.0) / self.TAU)
            self.open += a * (target - self.open)
            self._t = t
        return self.open

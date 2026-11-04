"""
The operator, drawn next to the robots.

Turns one OperatorPose (ARKit world: X right, Y up, -Z forward) into a stick
figure in the ROBOT's frame (MuJoCo world, Z up, chassis front = -X), rotated
so the operator faces the same way the robots do. The figure is metric and
unscaled — the point is to see the human's real proportions beside MABEL's.
Heights stay absolute (the Vision Pro immersive origin is floor-level).
"""
from __future__ import annotations

import numpy as np

# Robot body axes in the MJ world at the home pose: (right, up, fwd) columns.
# W = chassis R @ R_BASE_ALIGN with an identity chassis = diag(-1,-1,1);
# right = -W[:,1], up = W[:,2], fwd = W[:,0]  (engine._body_map_M).
_R_ROB = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

FINGERS = ("thumb", "index", "middle", "ring", "little")
CHAIN = {f: ([f"{f}_knuckle"] if f == "thumb" else [f"{f}_metacarpal", f"{f}_knuckle"])
         + [f"{f}_intermediate_base", f"{f}_intermediate_tip", f"{f}_tip"]
         for f in FINGERS}


def skeleton(pose, anchor=None):
    """-> {head, fwd, hands:{side:{wrist, fingers:[[pts]], elbow}}} in MJ coords.

    anchor=None: relative to the point under the head, turned to the head's facing
    (the figure stands still and always faces the robots' way — a pose view).
    anchor=(h0, f0) (the head position and horizontal facing at calibration): in
    the OPERATOR'S WORLD, referred to that anchor — the figure walks and turns as
    the operator does, in the same frame the base follows (start = the robot's
    start), so it walks beside the robot instead of staying at the origin."""
    if pose is None or pose.head is None:
        return None
    H = np.asarray(pose.head, float)
    h = H[:3, 3]
    if anchor is not None:
        h, f = np.asarray(anchor[0], float), np.asarray(anchor[1], float)
    else:
        f = -H[:3, 2]
        f = np.array([f[0], 0.0, f[2]])
    n = np.linalg.norm(f)
    if n < 1e-6:
        return None
    f /= n
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up)

    def mj(p):
        p = np.asarray(p, float)[:3]
        d = p - h
        return (_R_ROB @ np.array([d @ right, p[1], d @ f])).round(4).tolist()

    h_now = H[:3, 3]
    out = {"head": mj(h_now), "gaze": (_R_ROB @ np.array([-H[:3, 2] @ right, -H[1, 2],
                                                       -H[:3, 2] @ f])).round(3).tolist(),
           "hands": {}, "world": anchor is not None}
    for side in ("left", "right"):
        joints = getattr(pose, f"{side}_joints", None)
        if not joints or "wrist" not in joints:
            continue
        P = {k: mj(np.asarray(v)[:3, 3]) for k, v in joints.items()}
        hand = {"wrist": P["wrist"],
                "fingers": [[P["wrist"]] + [P[k] for k in CHAIN[fg] if k in P]
                            for fg in FINGERS]}
        el = getattr(pose, f"{side}_elbow", None)
        if el is not None:
            hand["elbow"] = mj(np.asarray(el)[:3, 3])
        out["hands"][side] = hand
    return out


def anchor_of(pose):
    """(head position, horizontal facing) of a pose — the skeleton's world anchor."""
    if pose is None or pose.head is None:
        return None
    H = np.asarray(pose.head, float)
    f = -H[:3, 2]
    f = np.array([f[0], 0.0, f[2]])
    if np.linalg.norm(f) < 1e-6:
        return None
    return H[:3, 3].copy(), f / np.linalg.norm(f)

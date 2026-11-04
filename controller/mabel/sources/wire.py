"""Wire-format ingest for the MABEL apps (iOS / Vision Pro / web).

The three frontends all send the same `teleop_frame` payload over the WebSocket
bridge: a head transform plus, per hand, an `anchorTransform` and a list of named
joint `localTransform`s (all row-major 4x4 in the ARKit world frame, X right, Y up,
-Z forward), and a `navJoystick`. This module turns that raw payload dict into the
library's frontend-agnostic `OperatorPose` -- so the wire dialect lives WITH the
controller, not in the transport bridge.

It is the in-package successor of the bridge's old `teleopframe_adapter.py`: same
math (world transform of a joint = anchorTransform @ localTransform), but it also
derives the thumb-index pinch gap each hand (so the clutch works) and keeps the full
named skeleton (so the per-finger ORCA retargeter works).
"""
from __future__ import annotations

import numpy as np

# Joint name strings as sent on the wire (HandJoint rawValues; PoseModels.swift).
_WRIST = "wrist"
_THUMB_TIP = "thumb_tip"
_INDEX_TIP = "index_tip"
# ARKit's forearm joint at the ELBOW end of the forearm. The desktop streamer
# (vision_pro_stream.py) publishes explicit elbows from the same joint; deriving it
# here gives the headset/app path the elbow-direction arm-IK objective too.
_FOREARM_ARM = "forearm_arm"

# Named ARKit skeleton -> MediaPipe-21 landmark layout ([0]=wrist, [4]=thumb tip,
# [8]=index tip, [12]=middle tip). Same table as teleop/engine._NAMES_21 and the
# streamer's WIRE_NAME — the layout every retargeter indexes into.
_NAMES_21 = [
    "wrist",
    "thumb_knuckle", "thumb_intermediate_base", "thumb_intermediate_tip", "thumb_tip",
    "index_knuckle", "index_intermediate_base", "index_intermediate_tip", "index_tip",
    "middle_knuckle", "middle_intermediate_base", "middle_intermediate_tip", "middle_tip",
    "ring_knuckle", "ring_intermediate_base", "ring_intermediate_tip", "ring_tip",
    "little_knuckle", "little_intermediate_base", "little_intermediate_tip", "little_tip",
]

# MediaPipe-style indices the Retargeter reads from the per-hand list.
_IDX_WRIST = 0
_IDX_THUMB = 4
_IDX_INDEX = 8


def _mat(transform) -> np.ndarray | None:
    """Row-major 16-float `{matrix:[...]}` (or bare list) -> (4,4) ndarray."""
    if transform is None:
        return None
    m = transform.get("matrix") if isinstance(transform, dict) else transform
    if m is None or len(m) < 16:
        return None
    return np.array(m, dtype=float).reshape(4, 4)


def _hand_world(hand: dict | None):
    """Return (mats21, joints_world, pinch_gap, elbow) for one hand, or four Nones.

    mats21       : FULL 21-element MediaPipe-layout list of 4x4 ARKit-frame world
                   transforms ([0]=wrist, [4]=thumb tip, [8]=index tip, [12]=middle
                   tip). Missing joints fall back to the wrist transform, so the
                   list is always well-formed for every consumer (Retargeter,
                   HandOptimizer, operator_frame). This used to be a 9-element
                   wrist-padded stub that starved the retargeters of the real
                   skeleton the apps already send.
    joints_world : {wire_joint_name: 4x4} full skeleton (per-finger ORCA IK).
    pinch_gap    : thumb-tip <-> index-tip distance [m], or None (drives the clutch).
    elbow        : 4x4 world transform of the forearm's elbow end (ARKit
                   `forearm_arm`), or None — feeds the arm IK's elbow-direction
                   objective exactly like the desktop streamer's explicit elbows.
    """
    if not hand or not hand.get("isTracked", False):
        return None, None, None, None
    anchor = _mat(hand.get("anchorTransform"))
    if anchor is None:
        return None, None, None, None

    name_to_local = {}
    for jp in hand.get("joints", []) or []:
        local = _mat(jp.get("localTransform"))
        if local is not None:
            name_to_local[jp.get("joint")] = local

    def world(name):
        local = name_to_local.get(name)
        return anchor @ local if local is not None else None

    wrist = world(_WRIST)
    if wrist is None:
        return None, None, None, None

    mats = [world(n) if n != _WRIST else wrist for n in _NAMES_21]
    mats = [m if m is not None else wrist.copy() for m in mats]

    joints_world = {name: anchor @ local for name, local in name_to_local.items()}

    thumb, index = mats[_IDX_THUMB], mats[_IDX_INDEX]
    pinch = None
    if _THUMB_TIP in name_to_local and _INDEX_TIP in name_to_local:
        pinch = float(np.linalg.norm(thumb[:3, 3] - index[:3, 3]))

    return mats, joints_world, pinch, world(_FOREARM_ARM)


def operator_pose_from_teleop_frame(payload: dict):
    """Build an OperatorPose from a `teleop_frame` payload dict (the schema the
    iOS / Vision Pro / web apps send). Hands absent / untracked become None so the
    fail-safe and late-anchor logic behave correctly."""
    from ..types import OperatorPose

    head = None
    h = payload.get("head")
    # Only treat the head as a live pose when it is ACTUALLY tracked. The iPhone
    # streams an identity head with trackingState "notAvailable" (head-follow off);
    # taking that as a real pose made the engine perpetually (re)calibrate + retarget
    # the arms to a bogus head → the 3D model drifted/moved with no operator input.
    if isinstance(h, dict) and h.get("trackingState", "normal") == "normal":
        head = _mat(h.get("transform"))

    l_mats, l_joints, l_pinch, l_forearm = _hand_world(payload.get("leftHand"))
    r_mats, r_joints, r_pinch, r_forearm = _hand_world(payload.get("rightHand"))

    # Per-side elbow transforms (4x4, ARKit world) feed the arm IK's elbow-direction
    # objective (scripts/teleop/arm.py ARM_W_ELBOW) so the redundant arm bends like
    # the operator's (salute / arms-up match). An explicit payload elbow (the webcam
    # client's PoseLandmarker) wins; otherwise the headset's own `forearm_arm` joint
    # (the elbow end of the ARKit forearm) supplies it — the headset path used to
    # send None here, which silently switched the elbow objective off.
    l_elbow = _mat(payload.get("leftElbow"))
    r_elbow = _mat(payload.get("rightElbow"))
    l_elbow = l_elbow if l_elbow is not None else l_forearm
    r_elbow = r_elbow if r_elbow is not None else r_forearm

    return OperatorPose(
        head=head,
        right_hand=r_mats, left_hand=l_mats,
        right_joints=r_joints, left_joints=l_joints,
        right_pinch=r_pinch, left_pinch=l_pinch,
        right_elbow=r_elbow, left_elbow=l_elbow,
        timestamp=float(payload.get("timestamp", 0.0) or 0.0),
    )


# Canonical app navJoystick -> controller joystick (lx, ly, rx, ry).
#
# The apps' navJoystick is in the wire frame; the shared TeleopController maps the
# stick straight through (tvx=+ly, tvy=+lx), so the app forward/strafe axes are
# NEGATED here to match MABEL's body frame. Yaw (rx) and lift (ry) already agree.
# This is the SAME convention the bridge used to apply inline (-lx, -ly, rx, ry);
# it now lives with the controller so a frame change is a one-line library edit.
def app_joystick(nav: dict | None):
    """`{lx,ly,rx,ry}` (or None) -> (lx, ly, rx, ry) tuple in the controller frame."""
    if not nav:
        return None
    return (-float(nav.get("lx", 0.0)), -float(nav.get("ly", 0.0)),
            float(nav.get("rx", 0.0)), float(nav.get("ry", 0.0)))

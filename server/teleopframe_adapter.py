"""
teleopframe_adapter.py — convert a WebSocket TeleopFrame (from the MabelTeleop
visionOS app) into the AVPFrame the simulation's Retargeter expects.

The two pipelines speak different dialects:

  • The app sends TeleopFrame (wire_protocol.py): a head transform plus, per hand,
    an `anchorTransform` and a list of 27 named joint `localTransform`s, all in the
    ARKit world frame (X right, Y up, -Z forward), row-major 4x4.

  • The sim's Retargeter (mabel/retarget.py) consumes AVPFrame (mabel/avp_receiver.py):
    `head` 4x4 and per-hand lists of 4x4 where index 0 = wrist, 4 = thumb tip,
    8 = index tip (a MediaPipe-style ordering). It only ever reads indices 0/4/8,
    and AVP_TO_ROBOT in config.py maps the ARKit frame into MABEL's frame — so we
    pass the ARKit-frame world transforms through unchanged.

World transform of a joint = anchorTransform @ jointLocalTransform (both row-major
homogeneous 4x4 — identical to how mujoco_pose_viz.py reconstructs joint poses).
"""
from __future__ import annotations

import numpy as np

from wire_protocol import TeleopFrame, HandPose

# Joint name strings as sent on the wire (HandJoint rawValues, see PoseModels.swift
# / JOINT_NAMES in mujoco_pose_viz.py). Only the three the Retargeter needs.
_WRIST = "wrist"
_THUMB_TIP = "thumb_tip"
_INDEX_TIP = "index_tip"

# MediaPipe-style indices the Retargeter reads.
_IDX_WRIST = 0
_IDX_THUMB = 4
_IDX_INDEX = 8
_LIST_LEN = 9          # must be >= 9 so retarget._pinch_grasp doesn't bail out


def _mat(transform) -> np.ndarray:
    """Row-major 16-float Transform4x4 -> (4,4) ndarray."""
    return np.array(transform.matrix, dtype=float).reshape(4, 4)


def _hand_mats(hand: HandPose | None) -> list[np.ndarray] | None:
    """Return a 9-element list of 4x4 ARKit-frame world transforms with index
    0 = wrist, 4 = thumb tip, 8 = index tip, or None if the wrist is unavailable.

    Every joint is used regardless of its per-joint isTracked confidence — ARKit
    always reports an estimated transform, matching mujoco_pose_viz.py's policy."""
    if hand is None or not hand.isTracked:
        return None

    anchor = _mat(hand.anchorTransform)
    name_to_local = {jp.joint: _mat(jp.localTransform) for jp in hand.joints}

    def world(name: str) -> np.ndarray | None:
        local = name_to_local.get(name)
        return anchor @ local if local is not None else None

    wrist = world(_WRIST)
    if wrist is None:
        return None
    thumb = world(_THUMB_TIP)
    index = world(_INDEX_TIP)

    mats = [wrist.copy() for _ in range(_LIST_LEN)]
    if thumb is not None:
        mats[_IDX_THUMB] = thumb
    if index is not None:
        mats[_IDX_INDEX] = index
    return mats


def _hand_world_joints(hand: HandPose | None) -> dict[str, np.ndarray] | None:
    """Full per-joint world transforms {wire_joint_name: 4x4}, for the
    per-finger ORCA hand retargeting (mabel/hand_retarget.py). None when the
    hand is untracked."""
    if hand is None or not hand.isTracked or not hand.joints:
        return None
    anchor = _mat(hand.anchorTransform)
    return {jp.joint: anchor @ _mat(jp.localTransform) for jp in hand.joints}


class AdaptedFrame:
    """Duck-typed AVPFrame: exposes .head, .left, .right, .timestamp, plus
    .left_joints/.right_joints (full skeletons) for finger retargeting.

    Mirrors mabel.avp_receiver.AVPFrame so Retargeter.calibrate()/update() and
    mock_avp consumers treat it identically, without importing the sim package
    here (the bridge wires the sim path in)."""

    __slots__ = ("timestamp", "head", "left", "right", "left_joints", "right_joints")

    def __init__(self, frame: TeleopFrame):
        self.timestamp = frame.timestamp
        self.head = _mat(frame.head.transform)
        self.left = _hand_mats(frame.leftHand)
        self.right = _hand_mats(frame.rightHand)
        self.left_joints = _hand_world_joints(frame.leftHand)
        self.right_joints = _hand_world_joints(frame.rightHand)


def to_avp_frame(frame: TeleopFrame | None) -> AdaptedFrame | None:
    """Adapt a decoded TeleopFrame into an AVPFrame-compatible object, or None."""
    if frame is None:
        return None
    return AdaptedFrame(frame)

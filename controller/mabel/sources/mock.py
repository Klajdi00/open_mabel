"""
Synthetic Vision Pro source for developing without the headset.

Produces AVPFrame-compatible objects (numpy 4x4 matrices in the ARKit frame:
X right, Y up, -Z forward) whose wrists trace a slow circle and whose head
slowly pans, so the retargeting + look-at neck can be exercised end to end.
"""
import math
import numpy as np

from .avp import AVPFrame


def _T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def mock_frame(t):
    """Return an AVPFrame for sim time t (seconds)."""
    payload = {"timestamp": t}

    # Head slowly pans left/right about the ARKit up axis (+Y).
    payload["head_mat"] = _T(_rot_y(0.4 * math.sin(0.25 * t)),
                             np.array([0.0, 1.4, 0.0])).flatten().tolist()

    def hand(cx, sign, pinch_phase):
        wrist = np.array([cx + 0.10 * math.sin(0.5 * t),
                          1.0 + 0.10 * math.sin(0.7 * t),
                          -0.30 - 0.10 * math.cos(0.5 * t)])
        mats = [_T(np.eye(3), wrist) for _ in range(21)]   # 21 MediaPipe joints
        # thumb tip (joint 4) and index tip (joint 8) drive the pinch distance
        pinch = 0.05 + 0.03 * math.sin(pinch_phase + 0.8 * t)
        mats[4] = _T(np.eye(3), wrist)                       # thumb tip
        mats[8] = _T(np.eye(3), wrist + np.array([0.0, pinch, 0.0]))  # index tip
        return [m.flatten().tolist() for m in mats]

    payload["right_hand_mats"] = hand(0.18, 1.0, 0.0)
    payload["left_hand_mats"] = hand(-0.18, -1.0, math.pi)
    return AVPFrame(payload)

"""Frame algebra for the two-mode retargeter: the F_a -> F_d reduction that
makes local mode possible, plus the small SO(3) helpers the estimators share.

THE ONE IDEA WORTH READING
--------------------------
The apps stream every hand joint as an ARKit WORLD transform (anchorTransform @
localTransform, both in F_a) and the head as T_ad. Local mode wants the hands in
the DEVICE frame:

    T_dh = T_ad^-1 T_ah

Both factors are referenced to the SAME anchor at the SAME instant, so whatever
F_a is doing -- drifting in heading, tilting its gravity estimate under a
vehicle's lateral acceleration -- cancels EXACTLY in the product. The result is
a pure optical measurement of where the hand is relative to the headset.

That is why local mode needs no change on the app side, and it is worth being
precise about the one place it is not exact: the cancellation holds only when
the head sample and the hand sample were resolved against the same anchor pose.
ARKit resolves both from one snapshot, so the residual is the relocalisation
that may occur BETWEEN snapshots — bounded, and quantified in E1 of
experiments/exp_retarget_modes.py rather than assumed away.
"""
from __future__ import annotations

import numpy as np

# ARKit (Y-up, -Z forward) vector -> robot frame (x fwd, y left, z up). Same
# matrix as teleop_engine.hyperparams.AVP_TO_ROBOT / mabel.config.AVP_TO_ROBOT;
# duplicated as a literal here ONLY so this package imports nothing from the
# engine (it must stay usable headless, without MuJoCo). Guarded by
# tests/test_retarget_modes.py::test_avp_to_robot_matches_engine.
AVP_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])

# ARKit world up, and the head's gaze axis in the head's own frame.
UP_ARKIT = np.array([0.0, 1.0, 0.0])
GAZE_LOCAL = np.array([0.0, 0.0, -1.0])


def rigid_inverse(T):
    """Inverse of a 4x4 rigid transform, by transpose (exact, and ~10x cheaper
    than a general inverse — this runs twice per hand per frame)."""
    T = np.asarray(T, float)
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def to_device_frame(T_ad, T_ah):
    """T_dh = T_ad^-1 T_ah — one anchor-frame pose expressed in the device
    frame. Returns None if either input is missing."""
    if T_ad is None or T_ah is None:
        return None
    return rigid_inverse(T_ad) @ np.asarray(T_ah, float)


def device_measurements(T_ad, hands, skeletons=None, pinch=None, timestamp=0.0):
    """Build the LOCAL admitted set S_L from the wire's anchor-frame poses.

    hands      {side: (4,4) wrist in F_a}
    skeletons  {side: {joint: (4,4) in F_a}}
    pinch      {side: float|None} — already frame-free, passed through

    This is the ONLY function that consumes T_ad on the local path, and it does
    so purely to cancel it. Nothing downstream of here can see F_a.
    """
    from .types import DeviceMeasurements

    inv = rigid_inverse(T_ad) if T_ad is not None else None
    wrists, skel = {}, {}
    for side, T in (hands or {}).items():
        if T is None or inv is None:
            continue
        wrists[side] = inv @ np.asarray(T, float)
    for side, joints in (skeletons or {}).items():
        if not joints or inv is None:
            continue
        skel[side] = {n: inv @ np.asarray(T, float)
                      for n, T in joints.items() if T is not None}
    return DeviceMeasurements(wrists=wrists, skeleton=skel,
                              pinch=dict(pinch or {}), timestamp=float(timestamp))


# ── SO(3) helpers ────────────────────────────────────────────────────────────
def skew(v):
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w):
    """Rodrigues: so(3) tangent vector -> rotation matrix."""
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)          # 1st order is exact enough here
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    """Rotation matrix -> so(3) tangent vector (rotation-vector form)."""
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    s = np.sin(th)
    if abs(s) < 1e-9:                        # th ~ pi: use the symmetric part
        A = (R + np.eye(3)) * 0.5
        v = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(v))
        axis = A[:, i] / (v[i] + 1e-12)
        axis /= (np.linalg.norm(axis) + 1e-12)
        return axis * th
    return (th / (2.0 * s)) * np.array([R[2, 1] - R[1, 2],
                                        R[0, 2] - R[2, 0],
                                        R[1, 0] - R[0, 1]])


def orthonormalize(R):
    """Nearest rotation matrix (polar / SVD) — GN steps drift off SO(3)."""
    U, _, Vt = np.linalg.svd(np.asarray(R, float))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:                # reflect back to a proper rotation
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def basis_from_axes(right, up, fwd):
    """(right, up, forward) -> the column-basis B of the package convention."""
    return np.column_stack([np.asarray(right, float),
                            np.asarray(up, float),
                            np.asarray(fwd, float)])


def gravity_aligned_basis(R_head, up=UP_ARKIT):
    """Yaw-only body basis from a head/device rotation and a known up: project
    the gaze axis onto the horizontal plane. Used as the LOCAL fit's seed and as
    the global estimator's fallback before a body yaw exists."""
    up = np.asarray(up, float)
    up = up / (np.linalg.norm(up) + 1e-12)
    fwd = np.asarray(R_head, float) @ GAZE_LOCAL
    fwd = fwd - up * float(fwd @ up)
    n = float(np.linalg.norm(fwd))
    if n < 1e-6:                              # looking straight up / down
        fwd = np.asarray(R_head, float)[:, 1]  # head's own up as the fallback
        fwd = fwd - up * float(fwd @ up)
        n = float(np.linalg.norm(fwd))
    fwd = fwd / (n + 1e-12)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-12)
    return basis_from_axes(right, up, np.cross(up, right))


def smoothstep(x, lo, hi):
    """C1 Hermite ramp — the beta_i of eq. (17), so the shape/extension blend
    introduces no velocity discontinuity."""
    t = (float(x) - lo) / max(hi - lo, 1e-9)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

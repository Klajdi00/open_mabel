"""Wrist targets — eqs. (16)-(18). MODE-INVARIANT: the only thing this module
knows about the mode is the single rotation R_bo handed to it.

    rho_i = min(||p_i - s_i|| / L_o, 1)          sigma = L_b / L_o        (16)

    r_i^ext = r_b^{sigma_i b} + rho_i L_b R_bo u_i
    r_i^shp = mu_b + sigma R_bo (p_i - mu_o)                              (17)
    r_i^*   = (1 - beta_i) r_i^shp + beta_i r_i^ext,  beta_i = S(rho_i; 0.85, 1)

WHY TWO BRANCHES
----------------
Anchoring each target at its OWN robot shoulder (the extension branch) preserves
REACH exactly -- full human extension maps to full robot extension in every
direction -- but distorts bimanual SHAPE, because MABEL's shoulders are 0.148 m
apart against a human's ~0.45 m, so inward reaches cross the midline far sooner
on the robot than on the operator. A single similarity about the mid-shoulder
root (the shape branch, GMR's root-uniform scaled-keypoint form) preserves the
head-wrist-wrist triangle by construction but truncates reach, since a uniform
sigma cannot also make the arm straighten at rho = 1.

Neither is right everywhere, and which one matters depends on rho: shape
dominates the mid-workspace where bimanual geometry is the task, extension
dominates at the boundary where running out of arm is the task. beta_i is a
smoothstep, so the blend is C^1 and no velocity discontinuity is injected.

Deliberately NOT ported from the shipped engine path (measured in E4 of
experiments/exp_retarget_modes.py rather than assumed harmless):
  * the near-full-extension OVERDRIVE (x1.04) that nudges targets marginally
    past the reach boundary so the position term overrules posture regularizers;
  * the absolute anti-collapse separation FLOOR. The shape branch preserves the
    pose triangle, which is what the floor was compensating for; it is available
    here as an opt-in (`hand_clear`) rather than always-on.
"""
from __future__ import annotations

import numpy as np

from .frames import smoothstep

EXT_GATE = 0.85          # rho at which the extension branch starts to take over
RHO_MAX = 1.0            # hyperextension saturates rather than overshooting


def wrist_target(side, p_i, est, robot, model, R_bo, ext_gate=EXT_GATE):
    """One side's robot-world wrist target, or None when it cannot be formed.

    p_i    operator wrist position, in the ESTIMATE's source frame
    est    BodyFrameEstimate (shoulders + root in that same frame)
    robot  RobotFrames
    model  OperatorModel (arm length L_o)
    R_bo   (3,3) source frame -> robot world
    """
    s_i = est.shoulders.get(side)
    mu_o = est.root
    r_b = robot.shoulders.get(side)
    mu_b = robot.root
    L_b = robot.reach.get(side)
    if any(v is None for v in (p_i, s_i, mu_o, r_b, mu_b, L_b, R_bo)):
        return None, {}

    r = np.asarray(p_i, float) - np.asarray(s_i, float)
    d = float(np.linalg.norm(r))
    if d < 1e-6:
        return None, {}                    # wrist AT the shoulder: no direction
    L_o = max(model.arm_length(side), 1e-6)
    rho = min(d / L_o, RHO_MAX)
    sigma = float(L_b) / L_o
    u = r / d

    r_ext = np.asarray(r_b, float) + rho * float(L_b) * (R_bo @ u)
    r_shp = np.asarray(mu_b, float) + sigma * (R_bo @ (np.asarray(p_i, float)
                                                      - np.asarray(mu_o, float)))
    beta = smoothstep(rho, ext_gate, 1.0)
    target = (1.0 - beta) * r_shp + beta * r_ext
    return target, {"rho": rho, "beta": beta, "sigma": sigma,
                    "L_o": L_o, "L_b": float(L_b)}


def wrist_targets(points, est, robot, model, R_bo, ext_gate=EXT_GATE,
                  hand_clear=None):
    """All tracked sides.

    `points` is {side: (3,) operator wrist position IN THE ESTIMATE'S SOURCE
    FRAME} — F_d in local mode, F_a in global. Taking plain positions rather
    than a measurement object keeps this function frame-agnostic, which is the
    concrete form of "eqs. (16)-(18) are mode-invariant".

    `hand_clear` (metres) optionally re-imposes the legacy absolute separation
    floor; None (default) trusts the shape branch to preserve the pose triangle.
    """
    out, diag = {}, {}
    for side, p in points.items():
        if p is None:
            continue
        t, d = wrist_target(side, p, est, robot, model, R_bo, ext_gate)
        if t is not None:
            out[side] = t
            diag[side] = d
    if hand_clear and "left" in out and "right" in out:
        sep = out["right"] - out["left"]
        n = float(np.linalg.norm(sep))
        # Fade to the operator's OWN separation near contact so touching hands
        # still touch; never push apart what is already a hand-clearance apart.
        pl, pr = points.get("left"), points.get("right")
        d_h = float(np.linalg.norm(np.asarray(pr, float) - np.asarray(pl, float))) \
            if pl is not None and pr is not None else float(hand_clear)
        floor = min(float(hand_clear), d_h)
        if n < floor:
            axis = sep / n if n > 1e-6 else R_bo @ np.array([0.0, 0.0, 1.0])
            mid = 0.5 * (out["right"] + out["left"])
            out["right"] = mid + 0.5 * floor * axis
            out["left"] = mid - 0.5 * floor * axis
            diag["floor"] = floor
    return out, diag


def wrist_orientation(R_src, R_bo):
    """Map an operator wrist ORIENTATION through the same rotation, giving the
    R* half of T*_be_i in eq. (18). The caller forms the task-space error
    xi* = K_xi Log(T_be^-1 T*_be) itself — that is the controller's business,
    not the retargeter's."""
    if R_src is None or R_bo is None:
        return None
    return np.asarray(R_bo, float) @ np.asarray(R_src, float)

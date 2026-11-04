"""Damped least-squares IK from a wrist (palm) target to arm joint angles.

Position-only by design. The task is 3-DOF and the chain is 7-DOF, so the arm is
redundant and the null space is free — we resolve it with a weak pull toward a
rest posture rather than leaving it to drift, because an unresolved null space is
how a 7-DOF solver reaches the same palm pose by swinging the elbow through the
torso between two frames that look identical at the fingertip.

DAMPING IS NOT OPTIONAL. Near a singularity (arm straight, or the target outside
reach) J·Jᵀ becomes ill-conditioned and a plain pseudo-inverse asks for enormous
joint rates — the classic "the arm explodes when you drag past its reach". The
Levenberg-Marquardt damping `λ²` bounds that: the step degrades to something
small and stable instead of diverging. `lambda_` is therefore a SAFETY
parameter, not a tuning nicety.
"""
from __future__ import annotations

import numpy as np
import mujoco


class WristIK:
    """Solve for one arm's joint angles that put `palm` at a world target.

    Parameters
    ----------
    model, data : MuJoCo pair. Solving uses a SCRATCH copy of `data` so the live
        state is never disturbed — a solver that poses the real robot to search
        is a solver that renders garbage on the way to an answer.
    joints : the ordered chain, base → wrist.
    palm_body : the body whose origin tracks the target.
    lambda_ : DLS damping (rad·m⁻¹ scale). Larger = slower, safer near limits.
    rest : optional {joint: angle} the null space is pulled toward.
    """

    def __init__(self, model, data, joints, palm_body,
                 lambda_=0.08, rest=None, null_gain=0.02):
        self.model = model
        self.data = data
        self.joints = list(joints)
        self.palm_id = model.body(palm_body).id
        self.lambda_ = float(lambda_)
        self.null_gain = float(null_gain)
        self.rest = dict(rest or {})
        self._dof = np.array([int(model.joint(j).dofadr[0]) for j in self.joints])
        self._qadr = np.array([int(model.joint(j).qposadr[0]) for j in self.joints])
        self._lo = np.array([model.joint(j).range[0] for j in self.joints])
        self._hi = np.array([model.joint(j).range[1] for j in self.joints])
        self._scratch = mujoco.MjData(model)

    # ── forward kinematics on the scratch copy ───────────────────────────────
    def _fk(self, q):
        d = self._scratch
        d.qpos[:] = self.data.qpos
        d.qpos[self._qadr] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d

    def palm_pos(self, q=None):
        """World position of the palm, for `q` (default: the live pose)."""
        if q is None:
            return np.array(self.data.body(self.palm_id).xpos, dtype=float)
        return np.array(self._fk(q).body(self.palm_id).xpos, dtype=float)

    def q_now(self):
        return np.array([self.data.qpos[a] for a in self._qadr], dtype=float)

    def _jacobian(self, d):
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, d, jacp, None, self.palm_id)
        return jacp[:, self._dof]                      # 3 x n_chain

    # ── the solve ────────────────────────────────────────────────────────────
    def solve(self, target, q0=None, iters=12, step_clip=0.25, tol=1e-4):
        """Joint angles that put the palm at `target` (world xyz).

        Returns (q, err_m). `err_m` is the residual distance — REPORT IT. An
        out-of-reach target has no solution, and a solver that silently returns
        its best effort as though it succeeded is how "the arm went somewhere
        strange" becomes untraceable.
        """
        target = np.asarray(target, dtype=float)
        q = self.q_now() if q0 is None else np.asarray(q0, dtype=float).copy()
        q = np.clip(q, self._lo, self._hi)
        n = len(self.joints)
        rest = np.array([self.rest.get(j, 0.0) for j in self.joints], dtype=float)

        err = np.inf
        for _ in range(int(iters)):
            d = self._fk(q)
            e = target - np.array(d.body(self.palm_id).xpos, dtype=float)
            err = float(np.linalg.norm(e))
            if err < tol:
                break
            J = self._jacobian(d)
            # Levenberg-Marquardt: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
            JJt = J @ J.T + (self.lambda_ ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, e)
            if self.null_gain:
                # null-space posture pull: (I - J⁺J)(q_rest - q)
                Jpinv = J.T @ np.linalg.solve(JJt, np.eye(3))
                dq += (np.eye(n) - Jpinv @ J) @ (self.null_gain * (rest - q))
            m = float(np.max(np.abs(dq)))
            if m > step_clip:                 # never take a wild step
                dq *= step_clip / m
            q = np.clip(q + dq, self._lo, self._hi)

        return q, err

    def solve_dict(self, target, **kw):
        q, err = self.solve(target, **kw)
        return {j: float(v) for j, v in zip(self.joints, q)}, err

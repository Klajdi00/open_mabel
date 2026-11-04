"""
Active perception via a 3D look-at point (HoMMI-style).

Rather than copying the operator's 6-DOF head pose onto MABEL's neck — which is
kinematically incompatible (different height, joint limits) — we reduce the head
command to a single 3D point the robot should look at, then solve the 3-DOF neck
so the head's forward (optical) axis points at it. Roll is left free, exactly the
relaxation HoMMI uses to transfer active-perception intent across embodiments.

The look-at point is obtained by casting the gaze ray into the MuJoCo scene
(intersection with geometry), with a fixed-distance fallback when nothing is hit.
"""
import numpy as np
import mujoco

from .. import config as C


def look_at_point(origin, forward_dir, model, data, exclude_body=None):
    """Cast a ray from `origin` along `forward_dir`; return the 3D hit point
    (or origin + forward * fallback distance if the ray hits nothing)."""
    origin = np.asarray(origin, dtype=float)
    d = np.asarray(forward_dir, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return origin
    d = d / n

    bodyexclude = -1
    if exclude_body is not None:
        bodyexclude = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, exclude_body)

    # Start the ray a little ahead of the head so it clears the neck/torso.
    pnt = origin + d * C.NECK_RAY_OFFSET
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, d, None, 1, bodyexclude, geomid)
    if dist >= 0.0 and geomid[0] != -1 and dist >= C.NECK_MIN_HIT_DIST:
        return pnt + d * dist
    return origin + d * C.NECK_FALLBACK_DIST


class NeckController:
    """Differential-IK pointing of the 3-DOF neck toward a look-at point."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.neck_dofs = [model.joint(j).dofadr[0] for j in C.NECK_JOINTS]
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.HEAD_BODY)
        self.target_q = {n: data.actuator(n).ctrl[0] for n in C.NECK_ACTUATORS}
        self.look_at = data.body(C.HEAD_BODY).xpos.copy()

    def head_pose(self):
        R = self.data.body(C.HEAD_BODY).xmat.reshape(3, 3)
        p = self.data.body(C.HEAD_BODY).xpos
        return p, R

    def gaze_origin_dir(self):
        """Current head optical origin and forward direction in world frame."""
        p, R = self.head_pose()
        return p.copy(), R @ C.HEAD_FORWARD_LOCAL

    def solve_and_apply(self, target_point, dt):
        """Point the head's forward axis at `target_point` and command the neck."""
        m, d = self.model, self.data
        self.look_at = np.asarray(target_point, dtype=float)

        p, R = self.head_pose()
        f_cur = R @ C.HEAD_FORWARD_LOCAL
        f_cur /= (np.linalg.norm(f_cur) + 1e-9)
        d_des = self.look_at - p
        nd = np.linalg.norm(d_des)
        if nd < 1e-6:
            return
        d_des /= nd

        # Angular velocity that rotates the forward axis onto the desired one.
        w = C.NECK_KP * np.cross(f_cur, d_des)

        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, self.head_id)
        Jr = jacr[:, self.neck_dofs]                       # 3 x 3
        lam = 0.05
        dq = Jr.T @ np.linalg.inv(Jr @ Jr.T + (lam ** 2) * np.eye(3)) @ w

        for i, name in enumerate(C.NECK_ACTUATORS):
            act_id = m.actuator(name).id
            lo, hi = m.actuator_ctrlrange[act_id]
            self.target_q[name] = float(np.clip(self.target_q[name] + dq[i] * dt, lo, hi))
            d.actuator(name).ctrl[0] = self.target_q[name]

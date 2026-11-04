"""
Hardware-realistic external-wrench estimation from JOINT-TORQUE sensing.

The simulator can hand us the true external wrench (data.xfrc_applied), but a real
robot cannot. Instead we estimate the operator's interaction wrench from what the
real hardware actually provides: joint encoders (q, qd) and motor-torque feedback
(tau_cmd). Three stages:

  1) Generalized-momentum disturbance observer (De Luca & Mattone): recovers the
     external joint torque tau_ext WITHOUT joint acceleration, as a first-order
     low-pass of bandwidth K:
         r = K ( p - p0 - INT( tau_cmd - b(q,qd) + Mdot*qd + r ) dt ),   r -> tau_ext
     using p = M(q) qd, bias b = C qd + g (MuJoCo qfrc_bias), and Mdot from a
     finite difference of the mass matrix.

  2) Cartesian wrench via damped least squares (optimization): the external joint
     torque on an arm is tau_ext = J(q)^T F. We solve the (over-determined) inverse
     for the equivalent end-effector wrench,
         F_hat = argmin_F || J^T F - tau_ext ||^2 + lambda^2 ||F||^2
               = (J J^T + lambda^2 I)^{-1} J tau_ext,
     then reduce it to the chassis planar frame for the base admittance.

  3) Contact localization ON THE ROBOT SURFACE by hypothesis scoring: for each
     arm link, reconstruct the force's line of action from the proximal-joint
     wrench, raycast it into that link's geoms (mesh geoms via mj_rayMesh,
     primitives via mju_rayGeom) to get a candidate surface point, refit a PURE
     POINT FORCE there against the whole arm's residual, and keep the link whose
     hypothesis leaves the smallest unexplained residual. The reported point is
     ON the robot skin, and the pure-force refit keeps magnitudes honest even
     when only a few proximal joints are informative.

SCOPE: estimation is deliberately ARMS-ONLY (per-link on the two arm chains +
their hand subtrees). Contacts on the chassis, lift, or torso are NOT claimed:
on the real robot those need a system identification the platform doesn't have,
whereas the arms' dynamics come from the same MJCF model that drives control.

Only q, qd, tau_cmd, and the model (M, b, J) are used -- all available on the real
robot (torque from the DAMIAO drivers' current sensing; the same MuJoCo model used
as the dynamics model) -- so the pipeline transfers directly to hardware.
"""
import numpy as np
import mujoco

from ... import config as C
# The observer primitive lives with the dynamic models; re-exported here so the
# legacy `from mabel.force_estimation import MomentumObserver` shim still works.
from ...dynamic_models.observers import MomentumObserver  # noqa: F401

_LEFT_ARM  = [f"Left Arm {i}" for i in range(1, 8)] + ["Left Wrist"]
_RIGHT_ARM = [f"Right Arm {i}" for i in range(1, 8)] + ["Right Wrist"]


def _dofs(model, joints):
    return [model.joint(j).dofadr[0] for j in joints]


# ── surface geometry: where does the force line of action meet the robot? ────

def body_subtree(model, body):
    """All body ids in the kinematic subtree rooted at `body` (MuJoCo orders
    bodies topologically, so one forward pass suffices)."""
    sub = {int(body)}
    for b in range(int(body) + 1, model.nbody):
        if int(model.body_parentid[b]) in sub:
            sub.add(b)
    return sorted(sub)


_ARM_BODY_CACHE = {}


def _arm_bodies(model, arm_joints):
    """Bodies an arm contact can live on: the chain links of `arm_joints` plus
    the subtree hanging off the most distal link (wrist -> palm -> fingers)."""
    key = (id(model), tuple(arm_joints))
    got = _ARM_BODY_CACHE.get(key)
    if got is None:
        chain = []
        for j in arm_joints:
            b = int(model.jnt_bodyid[model.joint(j).id])
            if b not in chain:
                chain.append(b)
        got = sorted(set(chain) | set(body_subtree(model, chain[-1])))
        _ARM_BODY_CACHE[key] = got
    return got


def _ray_bodies(model, data, bodies, pnt, vec):
    """Nearest intersection of ray (pnt, vec) with the geoms of `bodies`
    (visual meshes included — they ARE the surface the user sees). Returns
    the distance along vec, or -1 on no hit."""
    best = -1.0
    pnt = np.ascontiguousarray(pnt, dtype=np.float64)
    vec = np.ascontiguousarray(vec, dtype=np.float64)
    for b in bodies:
        adr, num = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        for g in range(adr, adr + num):
            t = int(model.geom_type[g])
            if t == mujoco.mjtGeom.mjGEOM_MESH:
                dist = mujoco.mj_rayMesh(model, data, g, pnt, vec)
            else:
                dist = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g],
                                          model.geom_size[g], pnt, vec, t)
            if dist >= 0 and (best < 0 or dist < best):
                best = float(dist)
    return best


def surface_point(model, data, bodies, p_line, force, backoff=0.6):
    """Entry point of a force's line of action on the given bodies' surface:
    cast from outside the robot (p_line - backoff*Fhat) along the push
    direction Fhat. Returns (point, hit) — on a miss, p_line comes back
    unchanged with hit=False so callers can fall back."""
    p_line = np.asarray(p_line, dtype=np.float64)
    f = np.asarray(force, dtype=np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return p_line, False
    d = f / n
    dist = _ray_bodies(model, data, bodies, p_line - backoff * d, d)
    if 0.0 <= dist <= 2.0 * backoff:
        return p_line - backoff * d + dist * d, True
    return p_line, False


def friction_compensate(model, r, qvel, v_eps=0.02, deadband=1.0, ramp=0.4):
    """Undo joint FRICTION LOSS from the observer residual (model knowledge:
    dof_frictionloss; MuJoCo puts it in qfrc_constraint, which the momentum
    observer cannot see — it silently absorbs ~fl of any external torque).
      moving joint:  friction = -fl*sign(qd)      -> tau_ext = r + fl*sign(qd)
                     (v_eps must sit BELOW slow recovery creep ~0.02 rad/s:
                     a joint crawling back after a shove drags kinetic
                     friction, and the static branch mis-reads that drag as
                     a standing ~2 N contact)
      static joint:  friction opposes tau_ext     -> tau_ext = r + fl*sign(r),
                     RAMPED in over |r| in [deadband*fl, (deadband+ramp)*fl]:
                     full +fl on a joint that clearly BROKE friction, zero
                     below. The deadband must sit at ~1.0*fl: after any arm
                     motion, static friction freezes up to +-fl of torque into
                     every joint (a standing, spatially smooth residual of
                     ||r|| ~ 2 Nm arm-wide that fits a fake ~4 N contact as
                     coherently as a real push, ratio 0.28 vs 0.28 measured) —
                     the ONLY thing separating it from a real contact is that
                     per-joint it never exceeds fl. Compensating below that
                     line inflates it straight into a phantom.
    Exact for saturated static friction; the zone |tau_ext| <~ fl is
    structurally unobservable — reported as-is rather than guessed."""
    fl = model.dof_frictionloss
    r = np.asarray(r, dtype=float)
    gain = np.clip((np.abs(r) - deadband * fl) / np.maximum(ramp * fl, 1e-9),
                   0.0, 1.0)
    s = np.where(np.abs(qvel) > v_eps, np.sign(qvel), np.sign(r) * gain)
    return r + fl * s


def limit_attribute(model, data, r, margin=0.03):
    """Zero residual components explainable by a JOINT-LIMIT reaction (model
    knowledge, like friction loss): a joint within `margin` of a limit can
    carry an arbitrarily large constraint torque pushing AWAY from the limit
    — to the momentum observer it looks exactly like a grab (measured: a
    trajectory commanded 0.14 rad past a limit produced a +13.7 Nm residual
    => a fake 60 N contact). Attribution by position+sign: at the lower limit
    only positive reaction is possible, at the upper only negative; residual
    of that sign on a joint parked at the limit is discarded. Conservative:
    a real push that jams a joint onto its stop loses that joint's info."""
    r = np.asarray(r, dtype=float).copy()
    for j in range(model.njnt):
        if not model.jnt_limited[j]:
            continue
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE,
                                     mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        d = int(model.jnt_dofadr[j])
        q = float(data.qpos[model.jnt_qposadr[j]])
        lo, hi = model.jnt_range[j]
        if q <= lo + margin and r[d] > 0:
            r[d] = 0.0
        elif q >= hi - margin and r[d] < 0:
            r[d] = 0.0
    return r


def estimate_arm_wrench(model, data, body, dofs, r, lam=3e-2):
    """Damped-least-squares end-effector wrench F (6-vec, world) explaining the
    external joint torque r on the arm DOFs: F = (J J^T + lam^2 I)^{-1} J r."""
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, model.body(body).id)
    J = np.vstack([jacp, jacr])[:, dofs]          # 6 x ndof
    r_sub = np.asarray(r)[dofs]
    F = np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), J @ r_sub)
    return F                                       # [fx,fy,fz, tx,ty,tz] world


_LINK_BODY_CACHE = {}


def _link_bodies(model, arm_joints, k):
    """Bodies belonging to link k of the arm: the subtree of joint k's body
    MINUS the subtree of the next arm joint's body (so mounts and fixed
    fittings stay with their link; the last link keeps the whole hand)."""
    key = (id(model), tuple(arm_joints), k)
    got = _LINK_BODY_CACHE.get(key)
    if got is None:
        b_k = int(model.jnt_bodyid[model.joint(arm_joints[k]).id])
        sub = set(body_subtree(model, b_k))
        if k + 1 < len(arm_joints):
            b_n = int(model.jnt_bodyid[model.joint(arm_joints[k + 1]).id])
            sub -= set(body_subtree(model, b_n))
        got = sorted(sub)
        _LINK_BODY_CACHE[key] = got
    return got


def localize_contact(model, data, arm_joints, r, tau_thresh=0.3, lam=3e-2,
                     f_min=1.5, prefer_body=None, prefer_margin=1.25):
    """Estimate WHERE an external force is applied on an arm, from the
    joint-torque residual r, by HYPOTHESIS SCORING over the arm links
    (residual pattern matching, cf. Haddadin's collision identification):

    for each candidate link k (distal -> proximal):
      1. DLS wrench (F0, M0) at the link origin from the joints PROXIMAL to k
         -- only used to reconstruct the force's LINE OF ACTION
         p(s) = o + F0 x M0/|F0|^2 + s*F0hat;
      2. raycast the line into link k's own geoms -> candidate surface point;
      3. refit a PURE POINT FORCE at that point against the residual of the
         WHOLE arm:  F_k = argmin_F ||Jp(p_k)^T F - r_arm||^2 + mu||F||^2.
         (The pure-force refit is what makes magnitudes honest: a 6-DOF wrench
         fit on few proximal joints soaks the residual into the couple and
         reports |F| ~ 0.)
      4. score by the unexplained residual ||Jp^T F_k - r_arm||; a hypothesis
         on the wrong link cannot reproduce the loaded/unloaded joint pattern.
         Adjacent links can score within noise of each other, so an incumbent
         (`prefer_body`, the currently tracked link) keeps the contact unless
         a challenger scores `prefer_margin` times better — hysteresis in
         hypothesis space, which is what stops the reported contact from
         strobing between neighbouring links.

    Returns dict(body, joint, point[3], force[3], surface, resid) in world
    frame, or None if no joint carries more than tau_thresh of residual (or no
    hypothesis reaches f_min newtons — sub-threshold noise fits)."""
    dofs = [model.joint(j).dofadr[0] for j in arm_joints]   # shoulder -> wrist
    r = np.asarray(r)
    r_arm = r[dofs]
    if float(np.max(np.abs(r_arm))) < tau_thresh:
        return None
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    best = None; incumbent = None
    for k in range(len(dofs) - 1, -1, -1):
        jid = model.joint(arm_joints[k]).id
        body = int(model.jnt_bodyid[jid])       # link distal to joint k
        prox = dofs[:k + 1]
        # 1) line of action from the proximal-joint wrench at the link origin
        mujoco.mj_jacBody(model, data, jacp, jacr, body)
        J = np.vstack([jacp, jacr])[:, prox]
        W = np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), J @ r[prox])
        F0, M0 = W[:3], W[3:]
        o = data.body(body).xpos
        nf0 = float(F0 @ F0)
        p_line = o + np.cross(F0, M0) / nf0 if nf0 > 1.0 else o.copy()
        # 2) entry point on THIS link's surface (fallback: through the origin,
        #    then the origin itself)
        lb = _link_bodies(model, arm_joints, k)
        d_hint = F0 if nf0 > 1e-6 else np.array([0.0, 1.0, 0.0])
        p_k, hit = surface_point(model, data, lb, p_line, d_hint)
        if not hit:
            p_k, hit = surface_point(model, data, lb, o, d_hint)
            if not hit:
                p_k = o.copy()
        # 3) pure point force at p_k refit against the WHOLE arm residual
        mujoco.mj_jac(model, data, jacp, None, p_k, body)
        A = jacp[:, dofs]                        # 3 x n_arm
        Fk = np.linalg.solve(A @ A.T + 1e-3 * np.eye(3), A @ r_arm)
        # 4) score: unexplained arm residual
        resid = float(np.linalg.norm(A.T @ Fk - r_arm))
        if float(np.linalg.norm(Fk)) < f_min:
            continue
        hyp = dict(body=model.body(body).name, joint=arm_joints[k],
                   point=p_k, force=Fk, surface=bool(hit), resid=resid)
        if best is None or resid < best["resid"]:
            best = hyp
        if hyp["body"] == prefer_body:
            incumbent = hyp
    if (best is not None and incumbent is not None
            and best["body"] != prefer_body
            and incumbent["resid"] <= best["resid"] * prefer_margin):
        return incumbent
    return best


class WrenchEstimator:
    """Full pipeline: momentum observer -> per-arm DLS wrench -> net wrench in the
    chassis planar frame (fx, fy, Mz), the drop-in replacement for the
    simulator-only external_base_wrench()."""

    def __init__(self, model, data, gain=25.0, lam=3e-2):
        self.m, self.d = model, data
        self.obs = MomentumObserver(model, data, gain=gain)
        self.lam = lam
        self.l_dofs = _dofs(model, _LEFT_ARM)
        self.r_dofs = _dofs(model, _RIGHT_ARM)
        self.r = np.zeros(model.nv)
        self.F_l = np.zeros(6); self.F_r = np.zeros(6)

    def update(self, tau_cmd, dt):
        """tau_cmd: full nv commanded joint torque (gravity comp + actuators).
        Returns (fx, fy, Mz) in the chassis planar frame."""
        self.r = self.obs.update(tau_cmd, dt)
        r_c = friction_compensate(self.m, self.r, self.d.qvel)
        self.F_l = estimate_arm_wrench(self.m, self.d, C.LEFT_PALM,  self.l_dofs, r_c, self.lam)
        self.F_r = estimate_arm_wrench(self.m, self.d, C.RIGHT_PALM, self.r_dofs, r_c, self.lam)
        base = self.d.body(C.CHASSIS_BODY)
        pb = base.xpos; Rb = base.xmat.reshape(3, 3)
        F = np.zeros(3); Mz = 0.0
        for body, Fw in ((C.LEFT_PALM, self.F_l), (C.RIGHT_PALM, self.F_r)):
            f = Fw[:3]
            F += f
            rvec = self.d.body(body).xpos - pb
            Mz += np.cross(rvec, f)[2] + Fw[5]
        f_body = Rb.T @ F
        return float(f_body[0]), float(f_body[1]), float(Mz)


class _ArmTrack:
    """Temporal stabilizer for one arm's contact: hysteresis on the contact
    link (switching links needs `confirm` consecutive detections), exponential
    smoothing of the reported force/point while the link is unchanged, and a
    short hold on dropout so a marginal residual doesn't strobe the contact."""

    def __init__(self, blend=0.5, confirm=4, hold=30):
        self.blend, self.confirm, self.hold = blend, confirm, hold
        self.c = None
        self._miss = 0
        self._cand = None
        self._cand_n = 0

    def step(self, c):
        if c is None:
            self._cand = None; self._cand_n = 0
            if self.c is not None:
                self._miss += 1
                if self._miss > self.hold:
                    self.c = None
            return self.c
        self._miss = 0
        if self.c is not None and c["body"] == self.c["body"]:
            b = self.blend
            self.c["point"] = (1 - b) * self.c["point"] + b * np.asarray(c["point"], float)
            self.c["force"] = (1 - b) * self.c["force"] + b * np.asarray(c["force"], float)
            self.c["joint"] = c["joint"]
            self.c["surface"] = c.get("surface", False) or self.c.get("surface", False)
            self._cand = None; self._cand_n = 0
        else:
            if self._cand == c["body"]:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = c["body"], 1
            # first appearance needs the same confirmation as a link switch —
            # friction handoffs at trajectory reversals flash 1–2-sample
            # detections that would otherwise latch and then linger on hold
            if self._cand_n >= self.confirm:
                self.c = dict(body=c["body"], joint=c["joint"],
                              point=np.asarray(c["point"], float),
                              force=np.asarray(c["force"], float),
                              surface=c.get("surface", False))
                self._cand = None; self._cand_n = 0
        return self.c


class ContactEstimator:
    """Momentum observer + per-arm CONTACT LOCALIZATION. Unlike WrenchEstimator
    (which assumes the force is at the palm), this finds which link is touched and
    where ON ITS SURFACE, then reduces the wrench to the base from the TRUE
    contact point -- so the base responds correctly whether you push the hand or
    the elbow. ARMS-ONLY by design (see module docstring).

    Stability: the observer residual is low-pass filtered (res_tau) before the
    DLS inversion — the DLS amplifies per-joint noise near the detection
    threshold — and the reported contact goes through _ArmTrack (link
    hysteresis + output smoothing + dropout hold). res_tau/blend trade noise
    for lag: at 60 ms/0.35 a compliant arm free-swinging under a sustained
    push (~1 rad/s) outran the filter chain and the estimate read 3 N of a
    15 N truth by the end of the swing; 30 ms/0.5 tracks it at 6-8 N with
    stiff-bench jitter still ≤0.4 N.

    After update(): .contacts is a list of dicts
    {body, joint, point, force, surface}."""

    def __init__(self, model, data, gain=35.0, lam=3e-2, tau_thresh=0.6,
                 res_tau=0.03, blend=0.5, confirm=4, hold=30):
        self.m, self.d = model, data
        self.obs = MomentumObserver(model, data, gain=gain)
        self.lam, self.tau_thresh = lam, tau_thresh
        self.res_tau = float(res_tau)
        self.r = np.zeros(model.nv)
        self.r_f = np.zeros(model.nv)               # filtered residual
        self._tracks = {id(_LEFT_ARM): _ArmTrack(blend, confirm, hold),
                        id(_RIGHT_ARM): _ArmTrack(blend, confirm, hold)}
        self.contacts = []

    def update(self, tau_cmd, dt):
        """Returns (fx, fy, Mz) in the chassis planar frame, reduced from the
        localized contact point(s). Populates self.contacts with the where/what."""
        self.r = self.obs.update(tau_cmd, dt)
        a = dt / (self.res_tau + dt)                 # one-pole low-pass
        self.r_f += a * (self.r - self.r_f)
        r_c = friction_compensate(self.m, self.r_f, self.d.qvel)
        r_c = limit_attribute(self.m, self.d, r_c)
        self.contacts = []
        for arm in (_LEFT_ARM, _RIGHT_ARM):
            tr = self._tracks[id(arm)]
            raw = localize_contact(self.m, self.d, arm, r_c,
                                   self.tau_thresh, self.lam,
                                   prefer_body=tr.c["body"] if tr.c else None)
            c = tr.step(raw)
            if c is not None:
                self.contacts.append(c)
        base = self.d.body(C.CHASSIS_BODY)
        pb = base.xpos; Rb = base.xmat.reshape(3, 3)
        F = np.zeros(3); Mz = 0.0
        for c in self.contacts:
            f = c["force"]
            F += f
            Mz += np.cross(c["point"] - pb, f)[2]   # moment about base from true point
        f_body = Rb.T @ F
        return float(f_body[0]), float(f_body[1]), float(Mz)

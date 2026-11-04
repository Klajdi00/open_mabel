"""LOCAL mode's body-frame estimator — eq. (14), solved in the DEVICE frame.

    T_do = argmin_T  sum_i rho( ||r_d^{h_i,d} - T r^{sigma_i,o}|| - L_o )^2
                     + mu ||Log((T^-)^-1 T)||^2      s.t.  T in T_neck

The operator's shoulders are rigid on the torso at calibrated offsets and each
wrist lies at arm's length from its own shoulder, so BOTH WRISTS ARE EVIDENCE
ABOUT WHERE THE TORSO IS -- without ever consulting the headset's world
estimate. That is the whole point: in a moving vehicle the cabin is visually
static while the IMU integrates the vehicle's yaw and tilts its gravity vector,
so any F_a-referenced body frame is wrong, while this one is not.

WHY IT IS A 3-DOF PROBLEM, NOT A 6-DOF ONE
------------------------------------------
T in T_neck says the torso is reachable from the device through the cervical
range. Since the device's own torso coordinates p_d^o are part of the operator
model, the torso origin in device coordinates is pinned at t_do = -R_do p_d^o
and a shoulder lands at s_i^d = R_do c_i with c_i = r^{sigma_i,o} - p_d^o
constant (see operator_model.py). The translation cancels: only the ROTATION is
free, and the cervical box is imposed on it directly rather than as a penalty.
So (14) becomes a 3-parameter robust fit, warm-started every frame and solved in
a few Gauss-Newton steps.

Residual, Jacobian (analytic, from R(delta) = R exp(delta)):

    u_i = p_i - R c_i,  d_i = ||u_i||,  e_i = d_i - L_i
    J_i = (u_i/d_i)^T R [c_i]_x                                    (1x3)

    (J^T W J + mu I) delta = -(J^T W e + mu Delta)

with W the robust weights and Delta the accumulated tangent step from the
previous frame's estimate, which is what makes the mu term the paper's
temporal-continuity prior rather than a plain damper.

WHICH POINTS p_i, AND WHY NOT ONLY THE WRISTS
---------------------------------------------
Read literally, (14) constrains each WRIST to lie at arm's length L_o from its
shoulder. That is only true when the arm is STRAIGHT. Measured on a synthetic
1.75 m operator holding a comfortable rho = 0.6 reach, the literal residual sits
at -0.14 m and no rotation can null it, so a two-sided quadratic would rotate
the torso purely to chase an infeasible target. Two corrections, both of which
the paper's own wording anticipates:

  * ROBUST + ONE-SIDED. The real geometric fact is an INEQUALITY, d_i <= L_o:
    a wrist can be nearer than arm's length but never further. Violations
    (d_i > L_o) are hard evidence that the torso estimate is wrong and get full
    weight; the feasible side gets W_SLACK, a weak pull expressing that during
    manipulation the arms tend toward extension. That is what the paper's robust
    loss rho is for, and it removes the bent-arm bias instead of bounding it.
  * THE ELBOW, WHEN THE SKELETON HAS ONE. ||elbow_i - shoulder_i|| = upper-arm
    length is a RIGID-SEGMENT equality, exactly satisfiable at every arm
    configuration, so it is informative when the wrist term is not. ARKit's
    `forearm_arm` is extrapolated rather than tracked (which is why the arm IK
    ignores it as a POSE objective), but a segment LENGTH tolerates that far
    better than a pose does. With both sides tracked this gives four residuals
    on three unknowns — a genuinely determined fit rather than a rank-2 one.

Set `use_elbow=False` to recover the paper-literal wrist-only form; E2 in
experiments/exp_retarget_modes.py ablates exactly that.

CONDITIONING -- STATED, NOT ASSUMED
-----------------------------------
Two range residuals cannot determine three angles at one instant: the data
Hessian has rank <= 2, so a null direction always exists and the estimate is
completed by continuity across frames. The useful per-frame confidence is
therefore NOT the smallest eigenvalue (identically ~0) but the smallest singular
value INSIDE the row space -- sigma_2 of the weighted 2x3 Jacobian, which
collapses when the two residuals become redundant (symmetric arm poses, hands
converging on the midline) and is healthy when the arms are extended and
asymmetric. Below `conf_floor` the data weight is smoothly faded out and the fit
HOLDS its previous estimate instead of chasing a rank-deficient update. E2 in
experiments/exp_retarget_modes.py maps where that happens.

The seed is identity: at engage the torso is assumed aligned with the device
(operator looking ahead, head level). Any error in that assumption is a constant
offset, and the mode-entry alignment latch in mode.py absorbs it exactly.

THE TRADE-OFF THIS ESTIMATOR CANNOT ESCAPE  (measured; see E7)
-------------------------------------------------------------
The reach frame is only as steady as this fit, and the fit's ONLY evidence about
head-versus-torso rotation is the hands. So when the operator moves their head
with their hands held still on the torso, the estimated frame wobbles and the
robot's targets move although the operator commanded nothing. Measured target
wobble under pure neck motion, hands fixed relative to the torso (E7):

                                       paper-literal    dead_reckon=True
    head still                              0.0 mm            0.0 mm
    look-around (+-60 deg yaw,   0.2 Hz)    2.9 mm            0.0 mm
    nod         (+-30 deg pitch, 0.15 Hz)   174  mm           0.0 mm
    both at once                            176  mm           0.0 mm
    ---------------------------------------------------------------------
    error under a 20 deg/s + 6 deg vehicle  0.0000 mm       203.6 mm

The nod row is the honest weak spot and it is NOT lag — it is invariant to mu.
It is the cost of `TILT_REF = "device"`, which buys the excellent look-around
row; `"previous"` trades them the other way (nod 2 mm, look-around 136 mm).
Glancing is the head motion operators make constantly and a nod reads as a slow
vertical drift rather than a jump, so the default takes that side of the trade.

WHY NOT JUST DEAD-RECKON THE HEAD. The right column above is the whole answer:
propagating R_do through the device's frame-to-frame rotation zeroes every
wobble row and destroys the property local mode exists for, turning an exact
0.0000 mm into 203.6 mm of uncommanded target error in a turning vehicle.
No filter splits the difference, because the corrupting motion
(vehicle yaw ~20 deg/s) and the legitimate motion (head yaw ~75 deg/s) are the
SAME ORDER on the SAME sensor — there is no frequency separation to exploit.
Breaking the tie needs more information about the torso than two hands carry: a
torso-mounted tracker, or a calibration pinning the shoulders far more tightly
than anthropometric ratios do.

Default is therefore paper-literal (`dead_reckon=False`): exact drift immunity
with a bounded, gracefully-degrading wobble, rather than an unbounded
correctness failure.
"""
from __future__ import annotations

import numpy as np

from .frames import exp_so3, log_so3, orthonormalize, skew, smoothstep
from .types import BodyFrameEstimate

# Cervical range of motion — the T_neck constraint, applied to the HEAD-relative-
# to-torso rotation R_od = R_do^T. Conservative adult ranges (Youdas et al.,
# Phys Ther 1992): axial rotation ~+-80 deg, flexion/extension ~+-60 deg,
# lateral bend ~+-45 deg.
ROM_YAW = np.deg2rad(80.0)
ROM_PITCH = np.deg2rad(60.0)
ROM_ROLL = np.deg2rad(45.0)

# Continuity weight mu, quoted AT 60 Hz and rate-normalised inside `update`
# (mu_eff = mu * REF_DT / dt) so the estimator's time constant is in SECONDS,
# not frames — the apps stream at 30 Hz, the sim substeps faster, and a
# frame-counted prior would behave differently in each.
#
# Its scale is not arbitrary. The damped Gauss-Newton step is a fraction
# k = H/(H + mu) of the full step, so tau ~ dt/k. The measured data curvature of
# eq. (14) at a normal reach is H ~ 0.1 m^2/rad^2 (E2 maps it across the
# workspace), and tau = 0.15 s at 60 Hz then wants mu = H(tau/dt - 1) ~ 0.8.
# The first draft of this file used mu = 40 — 400x too stiff, which froze the
# fit at its seed no matter what the operator did. Balance the units, not the
# vibes: residuals are metres, the prior is radians.
MU_CONTINUITY = 0.01     # weight of ||Log((T^-)^-1 T)||^2 at 60 Hz
MU_SEED = 0.01           # ...on the cold-start frame, where T^- does not exist
REF_DT = 1.0 / 60.0
HUBER_K = 0.05           # m; residual beyond this is down-weighted linearly
W_SLACK = 0.002          # weight on the FEASIBLE side of d_i <= L_o
W_TILT = 0.02            # weight of the tilt prior on the torso's up axis
# What the tilt prior pulls TOWARD. Measured head-to-head (E7):
#                        static yaw err      look-around    nod      combined
#   "device"  (default)   2 deg mean / 5 max    3 mm         157 mm    171 mm
#   "previous"           10 deg mean / 60 max  136 mm          2 mm     99 mm
# "device" says the head is roughly upright ON THE TORSO (a claim about the
# neck, NOT about gravity, so it stays legal in a vehicle); "previous" is pure
# tilt continuity. "device" wins on static accuracy and on the head motion
# operators actually make most — glancing around — and loses on a sustained
# nod, which reads as a slow vertical drift rather than a jump.
TILT_REF = "device"
W_ELBOW = 1.0            # weight of the rigid upper-arm equality
# Confidence is sigma_min of the weighted Jacobian NORMALISED by the shoulder
# lever |c| (~0.32 m), so it reads as "what fraction of a rigid lever arm is
# observable" and is comparable across operators. mu is the PRIMARY regulariser;
# this gate is the backstop for genuinely degenerate poses only, so the floor
# sits well below the 0.06-0.21 a normal reach produces.
CONF_FLOOR = 0.03
GN_ITERS = 4             # warm-started steps per frame
GN_ITERS_SEED = 24       # steps per seed on a cold start / basin escape
STEP_MAX = np.deg2rad(12.0)   # per-iteration tangent step cap [rad]
# Multi-start grid [deg]. Yaw is the ambiguous DOF and pitch is what carries the
# ROM corners; ROLL IS DELIBERATELY NOT SWEPT. Measured over 13 neck poses:
#   yaw only            mean 4.4 deg, max 28.0
#   yaw x pitch (3)     mean 3.1 deg, max  9.0     <- chosen
#   yaw x pitch x roll  mean 6.0 deg, max 50.7
# The last line is the interesting one: MORE search made it WORSE, because at a
# ROM corner the objective's global minimum is not the truth. Adding seeds finds
# lower cost and a worse answer. That is a property of eq. (14) at extreme neck
# poses, not of the solver, and it is why the fix is a bounded sweep plus an
# honest confidence output rather than a better optimiser.
SEED_YAWS = (-75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0)
SEED_PITCHES = (-40.0, 0.0, 40.0)
BASIN_PERIOD = 30        # frames between basin-escape re-solves (~0.5 s at 60 Hz)
BASIN_MARGIN = 0.7       # escape only if the alternative beats 70% of the cost


def _ypr_from_R(R):
    """Intrinsic (yaw about up, pitch about right, roll about forward) angles of
    R = Ry(yaw) Rx(pitch) Rz(roll), the natural cervical decomposition in the
    torso's (right, up, forward) = (x, y, z) axes."""
    pitch = float(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0)))
    roll = float(np.arctan2(R[1, 0], R[1, 1]))
    yaw = float(np.arctan2(R[0, 2], R[2, 2]))
    return yaw, pitch, roll


def _R_from_ypr(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    Rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def project_cervical(R_do):
    """Project a torso->device rotation onto T_neck. Returns (R_do, clamped)."""
    yaw, pitch, roll = _ypr_from_R(np.asarray(R_do, float).T)   # R_od = R_do^T
    y = float(np.clip(yaw, -ROM_YAW, ROM_YAW))
    p = float(np.clip(pitch, -ROM_PITCH, ROM_PITCH))
    r = float(np.clip(roll, -ROM_ROLL, ROM_ROLL))
    clamped = (y != yaw) or (p != pitch) or (r != roll)
    return _R_from_ypr(y, p, r).T, clamped


class LocalTorsoFit:
    """Streaming solver for eq. (14).

    Its `update` takes a DeviceMeasurements and NOTHING else — there is no
    parameter through which the F_a-referenced device pose could reach it. That
    signature is the enforcement of the admitted measurement set S_L.
    """

    def __init__(self, model, mu=MU_CONTINUITY, huber_k=HUBER_K,
                 conf_floor=CONF_FLOOR, use_elbow=True, w_slack=W_SLACK,
                 w_tilt=W_TILT, tilt_ref=TILT_REF, dead_reckon=False):
        self.model = model
        self.mu = float(mu)
        self.huber_k = float(huber_k)
        self.conf_floor = float(conf_floor)
        self.use_elbow = bool(use_elbow)
        self.w_slack = float(w_slack)
        # The tilt prior and dead-reckoning are ALTERNATIVE ways to pin the two
        # weakly-observed DOFs, so running both leaves the prior fighting the
        # propagation every frame — which silently turns the documented
        # dead-reckoning trade-off into something in between. Pair them here
        # unless the caller has deliberately chosen a weight.
        if dead_reckon and float(w_tilt) == W_TILT:
            w_tilt = 0.0
        self.w_tilt = float(w_tilt)
        self.tilt_ref = str(tilt_ref)
        # See the module docstring's TRADE-OFF section before flipping this.
        self.dead_reckon = bool(dead_reckon)
        self.reset()

    def reset(self):
        self.R_do = np.eye(3)      # torso -> device rotation (the estimate)
        self.ready = False
        self.confidence = 0.0
        self._since_escape = 0
        self._up_prev = None       # torso up axis last frame (tilt-continuity)
        self._R_ad_prev = None     # device attitude last frame (dead-reckoning)
        self.diag = {}
        return self

    # ── the fit ───────────────────────────────────────────────────────────────
    def _residuals(self, R, obs):
        """(e, J, W) for the current iterate.

        `obs` is a list of (offset, point, length, base_weight, one_sided) rows
        — the wrist reach inequalities and, when available, the rigid upper-arm
        equalities (see the module docstring).
        """
        rows, errs, wts = [], [], []
        for c, p, L, w0, one_sided in obs:
            u = p - R @ c
            d = float(np.linalg.norm(u))
            if d < 1e-6:
                continue                       # point AT the shoulder: no info
            e = d - L
            w = float(w0)
            if one_sided and e < 0.0:
                w *= self.w_slack              # feasible side: weak prior only
            if abs(e) > self.huber_k:          # Huber: linear beyond k
                w *= self.huber_k / abs(e)
            rows.append((u / d) @ (R @ skew(c)))
            errs.append(e)
            wts.append(w)
        if not rows:
            return None, None, None
        if self.w_tilt > 0.0:
            # TILT-CONTINUITY PRIOR. Two elbow spheres are only two constraints
            # on three unknowns, so one direction is always unobserved; the hard
            # ROM box bounds it but does not shape it, and the fit wanders into
            # large spurious pitch/roll (measured: a 36 deg pitch solution
            # beating the truth on cost at rho = 0.5).
            #
            # Penalising the torso's UP axis against ITS OWN PREVIOUS VALUE
            # regularises exactly the two weak DOFs (tilt) and is EXACTLY
            # yaw-invariant — a yaw rotation leaves the up axis fixed — so the
            # DOF that matters most for reach direction stays purely
            # data-driven. Pulling toward the DEVICE's up instead (an "the head
            # is upright on the torso" prior) was tried and measured worse: it
            # fixes a head glance but fights a genuine nod, because during a nod
            # the assumption is wrong by exactly the nod angle.
            e_y = np.array([0.0, 1.0, 0.0])
            ref = (e_y if self.tilt_ref == "device"
                   else (self._up_prev if self._up_prev is not None else R @ e_y))
            res = R @ e_y - ref
            Jn = -R @ skew(e_y)
            for k in range(3):
                rows.append(Jn[k])
                errs.append(float(res[k]))
                wts.append(self.w_tilt)
        return (np.asarray(errs, float), np.asarray(rows, float),
                np.asarray(wts, float))

    def _observations(self, dev, offsets):
        """Build the residual rows for this frame from the admitted set S_L."""
        obs = []
        for side in dev.sides():
            c = offsets[side]
            w = dev.wrist_pos(side)
            if w is not None:
                obs.append((c, w, self.model.arm_length(side), 1.0, True))
            if not self.use_elbow:
                continue
            e = dev.elbow_pos(side)
            if e is not None:
                obs.append((c, e, self.model.upper_arm(side), W_ELBOW, False))
        return obs

    def _confidence(self, J, W):
        """Smallest singular value INSIDE the row space of the weighted
        Jacobian (see the module docstring on why not the smallest eigenvalue),
        normalised by the shoulder lever so it is a dimensionless fraction."""
        if J is None or len(J) == 0:
            return 0.0
        sv = np.linalg.svd(np.sqrt(W)[:, None] * J, compute_uv=False)
        lever = float(np.linalg.norm(self.model.shoulder_offsets_device()["right"]))
        return float(sv[-1]) / max(lever, 1e-6)   # sigma_n for n residuals

    def _cost(self, R, obs):
        """The robust objective of eq. (14) WITHOUT the continuity prior — the
        quantity competing seeds are compared on. Huber, with the one-sided
        slack on the wrist inequalities, so it matches what the weights in
        `_residuals` are the IRLS form of."""
        total = 0.0
        for c, p, L, w0, one_sided in obs:
            e = float(np.linalg.norm(p - R @ c)) - L
            w = float(w0) * (self.w_slack if (one_sided and e < 0.0) else 1.0)
            a = abs(e)
            total += w * (e * e if a <= self.huber_k
                          else self.huber_k * (2.0 * a - self.huber_k))
        if self.w_tilt > 0.0:
            e_y = np.array([0.0, 1.0, 0.0])
            ref = (e_y if self.tilt_ref == "device"
                   else (self._up_prev if self._up_prev is not None else R @ e_y))
            total += self.w_tilt * float(np.sum((R @ e_y - ref) ** 2))
        return total

    def _gn(self, R0, obs, mu, iters):
        """Damped Gauss-Newton from one seed. Returns (R, conf, rms, clamped)."""
        R = np.asarray(R0, float).copy()
        R_prev = np.asarray(R0, float).copy()
        conf, rms, clamped = 0.0, float("nan"), False
        for _ in range(iters):
            e, J, W = self._residuals(R, obs)
            if e is None:
                break
            conf = self._confidence(J, W)
            # Fade the DATA term out where the geometry cannot orient the torso,
            # so a rank-deficient frame holds instead of chasing its null space.
            # mu is the primary regulariser; this is the degenerate-pose backstop.
            Wd = W * smoothstep(conf, 0.0, self.conf_floor)
            H = (J.T * Wd) @ J + mu * np.eye(3)
            g = J.T @ (Wd * e) + mu * log_so3(R_prev.T @ R)
            try:
                step = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            n = float(np.linalg.norm(step))
            if n > STEP_MAX:
                step *= STEP_MAX / n
            R = orthonormalize(R @ exp_so3(step))
            R, clamped = project_cervical(R)
            rms = float(np.sqrt(np.mean(np.asarray(e, float) ** 2)))
            if n < 1e-5:
                break
        return R, conf, rms, clamped

    def _seeds(self):
        """Cold-start seeds. Gauss-Newton is local and eq. (14) is NOT convex in
        the torso rotation: from the identity seed alone, a genuine +30 deg body
        turn converged to a local minimum 25 deg away and the continuity prior
        then pinned it there for the rest of the session (measured while
        building this). Sweeping yaw and pitch costs ~21 3x3 solves once, and is
        the difference between the estimator working and quietly not."""
        return [_R_from_ypr(y, p, 0.0).T
                for y in np.deg2rad(SEED_YAWS)
                for p in np.deg2rad(SEED_PITCHES)]

    def update(self, dev, dt=REF_DT):
        """Ingest one device-frame frame; return the BodyFrameEstimate.

        `dt` enters ONLY through the rate normalisation of the continuity prior,
        which is what keeps the estimator's time constant fixed in seconds as
        the client's frame rate changes.
        """
        # The continuity prior is a statement about the PREVIOUS estimate, so on
        # the very first frame there is nothing for it to say: hold it near zero
        # and let the data alone place the torso. Keeping the running weight
        # here would converge the cold start to only H/(H+mu) ~ 11% of the way
        # to the data's optimum, and no number of iterations would fix it —
        # within one frame T^- is fixed, so the prior is a steady-state offset,
        # not a transient.
        warm = self.ready
        mu = (self.mu * REF_DT / max(float(dt or REF_DT), 1e-4)
              if warm else MU_SEED)
        self.model.observe(dev)
        obs = self._observations(dev, self.model.shoulder_offsets_device())

        if not obs:                          # no hands: hold everything
            self.confidence = 0.0
            self.diag = {"n": 0, "held": True}
            self._since_escape += 1
            return self._estimate(valid=self.ready)

        if warm:
            R, conf, rms, clamped = self._gn(self.R_do, obs, mu, GN_ITERS)
            escaped = False
            # BASIN ESCAPE. A warm solve can only descend, so an estimate that
            # fell into the wrong basin (a fast turn outrunning the tracker, a
            # dropout resumed in a new pose) would never come back. Periodically
            # re-run the multi-start and take it only if it is CLEARLY better,
            # so an ordinary flat frame never causes a jump.
            self._since_escape += 1
            if self._since_escape >= BASIN_PERIOD:
                self._since_escape = 0
                best = min((self._gn(S, obs, MU_SEED, GN_ITERS_SEED)
                            for S in self._seeds()),
                           key=lambda r: self._cost(r[0], obs))
                if self._cost(best[0], obs) < BASIN_MARGIN * self._cost(R, obs):
                    R, conf, rms, clamped = best
                    escaped = True
        else:
            R, conf, rms, clamped = min(
                (self._gn(S, obs, mu, GN_ITERS_SEED) for S in self._seeds()),
                key=lambda r: self._cost(r[0], obs))
            escaped = False
            self._since_escape = 0

        self.R_do = R
        self.ready = True
        self._up_prev = (R @ np.array([0.0, 1.0, 0.0])).copy()
        self.confidence = conf
        self.diag = {"n": len(obs), "n_sides": len(dev.sides()), "conf": conf,
                     "rms_m": rms, "cost": self._cost(R, obs),
                     "rom_clamped": bool(clamped), "escaped": escaped,
                     "elbow": any(o[4] is False for o in obs),
                     "held": conf < self.conf_floor}
        return self._estimate(valid=True)

    def _estimate(self, valid):
        R = self.R_do
        offsets = self.model.shoulder_offsets_device()
        shoulders = {side: R @ c for side, c in offsets.items()}
        root = -R @ self.model.device_body()      # torso origin in device coords
        return BodyFrameEstimate(basis=R.copy(), shoulders=shoulders, root=root,
                                 frame="device", device_origin=np.zeros(3),
                                 confidence=self.confidence,
                                 valid=bool(valid), diag=dict(self.diag))

    # ── queries ───────────────────────────────────────────────────────────────
    def propagate(self, dR_device):
        """OPT-IN dead-reckoning: rotate the estimate by the device's own
        frame-to-frame rotation (the torso is assumed fixed, so R_do turns by
        the inverse of however the head turned).

        Deliberately a SEPARATE method rather than a parameter of `update`, so
        that the default path keeps its structural guarantee — `update` still
        has nowhere for an F_a-referenced quantity to enter — and switching this
        on is a visible act at the call site rather than a hidden flag deep in a
        solver. Read the module docstring's TRADE-OFF section first: this buys a
        perfectly steady reach frame and costs the drift immunity that is the
        entire reason local mode exists.
        """
        if dR_device is None or not self.ready:
            return self
        self.R_do = orthonormalize(np.asarray(dR_device, float).T @ self.R_do)
        self.R_do, _ = project_cervical(self.R_do)
        self._up_prev = (self.R_do @ np.array([0.0, 1.0, 0.0])).copy()
        return self

    def neck_rotation(self):
        """R_od = R_do^-1 — the head's orientation RELATIVE TO THE TORSO, i.e.
        exactly what the robot's neck should reproduce. Unlike the world-anchored
        head reference the engine uses today, this is drift-free by construction:
        gaze survives in local mode (paper §V-B.1)."""
        return self.R_do.T.copy()

    def cervical_angles(self):
        """(yaw, pitch, roll) of the head on the torso [rad] — the neck command
        in the torso's own axes, already inside T_neck."""
        return _ypr_from_R(self.R_do.T)

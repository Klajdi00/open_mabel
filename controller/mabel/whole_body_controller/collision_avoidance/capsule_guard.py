"""
self_collision.py — capsule self-collision avoidance for MABEL's wrist-tracking IK.

The teleop "Hold" controller (wbc.WholeBodyIK) tracks operator wrist targets with a
differential IK. Left unconstrained, an operator can drive a palm straight into the
robot's own chest, head, or the other arm. This module turns the bare damped-least-
squares step into a small, hard-constrained quadratic program (QP) — a one-step
receding-horizon controller — whose inequality constraints keep every monitored
body pair outside a safety margin while still tracking the targets as closely as the
constraints allow.

Geometry (the "bounding volumes")
---------------------------------
Each protected link is wrapped in a capsule (a swept sphere: segment + radius), the
tightest smooth bounding volume for an elongated link and the one with a closed-form
signed distance. We wrap the two forearms and two hands (the parts that move under
the arm DOFs) and the torso + head (the parts they crash into). Capsule endpoints are
captured once in each anchor body's local frame at the rest pose, so they ride with
the body for free as the configuration changes.

Avoidance law (control barrier function)
----------------------------------------
For a capsule pair the surface gap is  d(q) = ||c_a − c_b|| − r_a − r_b, with c_a, c_b
the closest points on the two core segments. Its time derivative along the unit
separation n = (c_a − c_b)/||c_a − c_b|| is

    ḋ = nᵀ (ṗ_a − ṗ_b) = nᵀ (J_a − J_b) q̇ ,           (1)

where J_a, J_b are the translational point Jacobians at c_a, c_b. We impose the
discrete control-barrier-function (CBF) constraint

    ḋ ≥ −α (d − d_s)        for d < d_i  (influence distance),     (2)

i.e. the gap may shrink only ever more slowly as it approaches the safety distance
d_s, and must grow if already inside it. Rearranged into the QP's G q̇ ≤ h form,

    −nᵀ (J_a − J_b) q̇ ≤ α (d − d_s).                               (3)

QP
--
    minimise   ½ q̇ᵀ H q̇ + gᵀ q̇          (tracking + damping + posture)
    subject to  G q̇ ≤ h                   (self-collision CBFs)
                q̇_lo ≤ q̇ ≤ q̇_hi          (velocity + joint-limit dampers)

with H = JᵀJ + λ²W + μI and g = −(Jᵀ v_des + μ q̇_post) reproducing the original
weighted-DLS + null-space-posture behaviour when no constraint is active. The
strictly-convex dense QP is solved with quadprog (Goldfarb–Idnani dual active-set,
tens of microseconds at this size); osqp and an unconstrained-clamp fallback cover
environments where quadprog is unavailable.
"""
import numpy as np
import mujoco

from ... import config as C

try:
    import quadprog
    _HAVE_QUADPROG = True
except Exception:
    _HAVE_QUADPROG = False
try:
    import osqp
    import scipy.sparse as _sp
    _HAVE_OSQP = True
except Exception:
    _HAVE_OSQP = False


def _seg_seg(p1, q1, p2, q2):
    """Closest points between segments p1q1 and p2q2 (Ericson, RTCD §5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e <= eps:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return p1 + d1 * s, p2 + d2 * t


class Capsule:
    """Swept-sphere bounding volume anchored to a body (endpoints in body frame)."""
    __slots__ = ("body", "p0", "p1", "radius", "name")

    def __init__(self, body, p0, p1, radius, name):
        self.body = body
        self.p0 = np.asarray(p0, float)
        self.p1 = np.asarray(p1, float)
        self.radius = float(radius)
        self.name = name

    def world(self, data):
        R = data.xmat[self.body].reshape(3, 3)
        o = data.xpos[self.body]
        return o + R @ self.p0, o + R @ self.p1


class SelfCollisionGuard:
    """Builds the capsule model + monitored pairs and emits CBF rows each tick."""

    def __init__(self, model, data,
                 influence=None, margin=None, alpha=None, stop_decel=None):
        self.model = model
        self.data = data
        self.d_inf = C.SC_INFLUENCE if influence is None else influence
        self.d_safe = C.SC_MARGIN if margin is None else margin
        self.alpha = C.SC_ALPHA if alpha is None else alpha
        self.stop_decel = C.SC_STOP_DECEL if stop_decel is None else stop_decel
        self._jp = np.zeros((3, model.nv))      # scratch for mj_jac
        self._jr = np.zeros((3, model.nv))
        self.min_dist = np.inf                  # telemetry: tightest gap last tick
        #: capsules whose pose is carried by a low-authority joint, and which
        #: therefore receive `rows(margin_pad=...)`. Empty = the pad is global.
        self.lag_capsules = frozenset()
        # Optional contact-specific pad multipliers; absent pairs keep the
        # original lag margin. Consumers must retain their physical margin.
        self.pair_lag_scale = {}
        # Optional redundant endpoint barriers for extended contact patches.
        # Endpoint spheres are subsets of their existing capsules; the exact
        # closest-pair barrier is retained alongside these additional rows.
        self.endpoint_pairs = frozenset()

        mujoco.mj_forward(model, data)
        self.caps = self._build_capsules()
        self.pairs = self._build_pairs()
        # NOTE (2026-07-30): J1/J2 "cannot collide" (operator) — implemented by
        # REMOVING the elbow~torso pairs (their gradients were ~pure J1/J2 and
        # walked J1 onto its range limit). Full G-column masking was tried and
        # rejected: it broke the hard no-interpenetration guarantee (the
        # remaining hand/wrist pairs sometimes need a little shoulder motion).

    # ── model construction ───────────────────────────────────────────────────
    def _bid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _cap_from_bodies(self, anchor, a_world_name, b_world_name, radius, name,
                         frac=0.0):
        """Capsule on `anchor` spanning two bodies' current origins. ``frac``
        moves the START endpoint toward the end (a distal sub-segment): the
        upper-arm guard needs it — the FULL mount→elbow segment is only
        ≈ 0.08 m from the torso centreline at REST (arms hanging), inside the
        combined radii, so guarding it would violate from boot; the distal 60 %
        rests 0.16--0.19 m clear while still covering the elbow-side half that
        actually crosses the chest."""
        d = self.data
        a = d.xpos[self._bid(a_world_name)].copy()
        b = d.xpos[self._bid(b_world_name)]
        if frac > 0.0:
            a = a + frac * (b - a)
        ba = self._bid(anchor)
        R = d.xmat[ba].reshape(3, 3); o = d.xpos[ba]
        return Capsule(ba, R.T @ (a - o), R.T @ (b - o), radius, name)

    def _fit_to_geometry(self, cap, ref_body):
        """Sit a capsule on the REAL geometry it is meant to wrap.

        Capsule endpoints are BODY ORIGINS, and a body origin is wherever the
        MJCF author put its frame — not necessarily inside the mesh. The torso
        capsule was built `body_mount` -> `head_mount`, and those two origins are
        at y = -43 mm and y = +2 mm: the capsule was both OFFSET 43 mm toward the
        right arm and TILTED, leaning right at the bottom, exactly where the
        elbows are. Measured at the identical rest pose, the right forearm read
        13 mm of clearance and the left read 100 mm.

        That is not cosmetic. It is almost certainly why guarding the upper arm
        against the torso "spring-loaded the natural rest pose" and was reverted
        on 2026-07-30: on the right, the rest pose was already inside a
        mis-placed capsule, so any margin at all pushed against it.

        The fit keeps the capsule's LENGTH (it is the right length for a torso)
        and replaces its placement: centred on the geometry's volume-weighted
        centroid, and aligned with the body's own vertical rather than with the
        line between two unrelated frame origins.
        """
        m, d = self.model, self.data
        b = self._bid(ref_body)
        gs = list(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
        if not gs:
            return cap
        R = d.xmat[b].reshape(3, 3)
        o = d.xpos[b]
        # VOLUME-WEIGHTED, not a plain mean: the torso body also carries two tiny
        # decorative camera geoms 9-12 cm forward of the mesh, and an unweighted
        # centroid let them drag the capsule 52 mm out of the chest — which would
        # make every forward reach trip a guard meant to protect the BODY.
        # Weighting by the largest half-extent cubed makes the torso mesh (0.157)
        # outweigh a 12 mm marker by ~2000x.
        w = np.array([max(float(np.max(m.geom_size[g])), 1e-6) ** 3 for g in gs])
        pts = np.array([R.T @ (d.geom_xpos[g] - o) for g in gs])
        cen = (w[:, None] * pts).sum(axis=0) / w.sum()          # body-local
        half = 0.5 * float(np.linalg.norm(cap.p1 - cap.p0))
        up = R.T @ np.array([0.0, 0.0, 1.0])                    # world up, body-local
        n = np.linalg.norm(up)
        if n < 1e-9:
            return cap
        up = up / n
        return Capsule(cap.body, cen - half * up, cen + half * up,
                       cap.radius, cap.name)

    def _sphere(self, body, radius, name):
        c = Capsule(self._bid(body), np.zeros(3), np.zeros(3), radius, name)
        return c

    def _build_capsules(self):
        R = C.SC_RADII
        return {
            "l_fore": self._cap_from_bodies("l_forearm", "l_elbow", "l_wrist", R["forearm"], "l_fore"),
            "r_fore": self._cap_from_bodies("r_forearm", "r_elbow", "r_wrist", R["forearm"], "r_fore"),
            "l_hand": self._cap_from_bodies("left_palm", "l_wrist", "left_palm", R["hand"], "l_hand"),
            "r_hand": self._cap_from_bodies("right_palm", "r_wrist", "right_palm", R["hand"], "r_hand"),
            # RECENTRED on the torso's own geometry — see _recentre. Without it
            # the capsule leans 43 mm right and the two arms are guarded
            # asymmetrically at the same pose.
            "torso":  self._fit_to_geometry(
                self._cap_from_bodies("torso", "body_mount", "head_mount",
                                      R["torso"], "torso"), "torso"),
            "head":   self._sphere("head", R["head"], "head"),
            # ELBOWS (2026-07-22): the upper arm was previously entirely
            # unguarded — a cross-body reach with the elbow swung inward
            # pressed it into the chest ("weird behaviours" as the servos and
            # geometry fought). Guarded as ELBOW SPHERES with a ZERO per-pair
            # margin (see _build_pairs): the SETTLED right arm rests its elbow
            # at +0.6 cm of the combined radii and its upper-arm SEGMENT 6 cm
            # INSIDE the torso capsule — any segment guard or global margin
            # pushes the natural rest pose (measured: 15° of fake "sag" in
            # impedance holds). The sphere + zero margin brakes/blocks a
            # chest-crossing elbow without ever fighting a hanging arm.
            "l_upper": self._sphere("l_elbow", R["upper"], "l_upper"),
            "r_upper": self._sphere("r_elbow", R["upper"], "r_upper"),
            # WRIST/TOWER (2026-07-30, operator: "the link from J7 to the bottom
            # of the ORCA hand wrist has no collision avoidance"): the ORCA
            # wrist assembly BENDS — the tower sits ~2 cm off the straight
            # wrist→palm axis and extends beyond the hand capsule's radius, so
            # the J7→wrist segment was uncovered. One capsule per side spanning
            # wrist→tower closes it.
            "l_wristmt": self._cap_from_bodies("l_hand_mount", "l_wrist",
                                               "left_tower", R["hand"], "l_wristmt"),
            "r_wristmt": self._cap_from_bodies("r_hand_mount", "r_wrist",
                                               "right_tower", R["hand"], "r_wristmt"),
        }

    def _hops(self, b0, b1):
        """Number of parent links between two bodies in the kinematic tree
        (0 = same body, 1 = parent/child). Used to skip adjacent links."""
        anc = {b0: 0}
        b, n = b0, 0
        while self.model.body_parentid[b] != b:
            b = self.model.body_parentid[b]; n += 1; anc[b] = n
        b, n = b1, 0
        while b not in anc:
            if self.model.body_parentid[b] == b:
                return 10 ** 6                 # disjoint subtrees -> far apart
            b = self.model.body_parentid[b]; n += 1
        return anc[b] + n

    def _build_pairs(self):
        c = self.caps
        # Distal links vs the body / the opposite arm. forearm↔head is dropped
        # (measured min gap 44 mm over the whole front workspace — it never comes
        # close) to save compute.
        names = [
            ("l_hand", "torso"), ("l_hand", "head"), ("l_hand", "r_hand"), ("l_hand", "r_fore"),
            ("r_hand", "torso"), ("r_hand", "head"), ("r_hand", "l_fore"),
            ("l_fore", "r_fore"),
            ("l_wristmt", "torso"), ("l_wristmt", "head"),
            ("r_wristmt", "torso"), ("r_wristmt", "head"),
            # ARM ~ BODY, restored 2026-08-23 (operator: "the arms should not
            # collide with the body ... right now the robot arms will collide
            # with the robot body"). MEASURED over the front-workspace posture
            # table before the fix, with these pairs absent: the RIGHT FOREARM
            # was inside the torso capsule on 17.0 % of commanded postures, up
            # to 122 mm deep, and the upper arm on 9.2 %. Only the hands and
            # wrist mounts were ever guarded against the body.
            #
            # These pairs existed once and were removed on 2026-07-30 because
            # they "pinned the settled right arm". That was true, and it was a
            # symptom rather than a cause: the torso capsule was built from two
            # body ORIGINS that put it 43 mm right of the chest and tilted it
            # into the right elbow, so the right arm's rest pose sat inside a
            # capsule that was in the wrong place (13 mm clearance right vs
            # 100 mm left at the SAME pose). With the capsule fitted to the
            # torso's own geometry (_fit_to_geometry) the rest pose clears it by
            # 49/52 mm on both sides, and the pairs cost nothing at rest.
            ("l_fore", "torso"), ("r_fore", "torso"),
            ("l_upper", "torso"), ("r_upper", "torso"),
        ]
        pairs = []
        # PER-PAIR safety margins are no longer needed. The elbow~torso pairs
        # used to run at 3 mm because the settled elbow rested at +6 mm of the
        # combined radii — an artefact of the mis-placed torso capsule, not of
        # the robot. Fitted to the geometry the rest clearance is 49/52 mm, so
        # every pair can share one thin margin (C.SC_MARGIN).
        self._pair_margin = {}
        for a, b in names:
            ca, cb = c[a], c[b]
            if self._hops(ca.body, cb.body) <= C.SC_ADJ_HOPS:
                continue                       # adjacent links can't self-collide
            pairs.append((ca, cb))
        return pairs

    def _margin_for(self, ca, cb):
        m = self._pair_margin.get((ca.name, cb.name))
        if m is None:
            m = self._pair_margin.get(tuple(sorted((ca.name, cb.name))))
        return self.d_safe if m is None else m

    # ── per-tick constraint assembly ─────────────────────────────────────────
    def _point_jac(self, point, body):
        mujoco.mj_jac(self.model, self.data, self._jp, self._jr, point, body)
        return self._jp.copy()

    def rows(self, dofs, braking=False, margin_pad=0.0):
        """Return (G, h) for the active pairs: G q̇ ≤ h over the given DOF columns.
        Only pairs within the influence distance contribute a row.

        ``braking=True`` additionally caps each row's allowed closing speed by the
        braking-distance law √(2·a·slack) (the fastest approach from which the link
        can still stop within the remaining gap at decel ``stop_decel``) — the same
        robustification speed_scale() carries. Inside the margin the row keeps the
        plain α(d−d_s) < 0, which MANDATES opening.

        ``margin_pad`` inflates a pair's margin — used by consumers whose
        commanded velocity reaches the metal through a lagging position servo:
        the physical link overshoots the commanded stop by its following error,
        so the MEASURED-pose rows must brake that much earlier. Capped per pair
        at the pair's OWN margin: the elbow~torso pairs run a deliberate 3 mm
        margin because the SETTLED rest pose sits just outside it — a full
        12 mm pad there would spring-load the natural rest pose (the exact
        failure the per-pair margins were introduced to fix)."""
        G, h = [], []
        self.min_dist = np.inf
        for ca, cb in self.pairs:
            pa0, pa1 = ca.world(self.data)
            pb0, pb1 = cb.world(self.data)
            wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
            sep = wa - wb
            L = np.linalg.norm(sep)
            d = L - ca.radius - cb.radius            # surface gap
            self.min_dist = min(self.min_dist, d)
            if d >= self.d_inf:
                continue
            n = sep / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
            Ja = self._point_jac(wa, ca.body)[:, dofs]
            Jb = self._point_jac(wb, cb.body)[:, dofs]
            G.append(-(n @ (Ja - Jb)))               # −nᵀ(J_a−J_b)
            m = self._margin_for(ca, cb)
            # The lag pad applies ONLY to pairs running the GLOBAL margin. The
            # tight-margin pairs (elbow~torso, 3 mm) are margined so the SETTLED
            # rest pose sits just outside — any pad re-enters the margin at
            # rest, and an inside-margin row MANDATES an opening correction
            # every tick with zero commanded input: measured live as the right
            # arm oscillating from the moment torque came on (2026-07-30).
            # The pad covers a SERVO FOLLOWING ERROR, so it belongs only to pairs
            # whose geometry is carried by a joint that actually lags. Applied
            # uniformly it taxes links that track well: measured, a 20 mm blanket
            # pad pushed the FOREARM standoff from 10 to 28 mm to buy a margin
            # only the HAND needed (the hand capsule turns with the ~2.9 Hz ORCA
            # wrist). `lag_capsules` names the ones that need it.
            lag = (not self.lag_capsules) or (ca.name in self.lag_capsules
                                              or cb.name in self.lag_capsules)
            key = tuple(sorted((ca.name, cb.name)))
            lag_scale = self.pair_lag_scale.get(key, 1.0)
            pad = (margin_pad * lag_scale
                   if lag and (m >= self.d_safe or key in self.pair_lag_scale) else 0.0)
            slack = d - m - pad                      # allowed: α (d − d_s − pad)
            cap = self.alpha * slack
            if braking and slack > 0.0:
                cap = min(cap, np.sqrt(2.0 * self.stop_decel * slack))
            h.append(cap)
        if not G:
            return np.zeros((0, len(dofs))), np.zeros(0)
        G, h = np.asarray(G), np.asarray(h)
        if self.endpoint_pairs:
            Ge, he = self._endpoint_rows(dofs, braking, margin_pad)
            if len(Ge):
                G, h = np.vstack([G, Ge]), np.r_[h, he]
        return G, h

    def _endpoint_rows(self, dofs, braking=False, margin_pad=0.0):
        """Stabilize extended, nearly parallel contact without dropping any barrier.

        A capsule's closest points become non-unique when the cores are parallel.
        Using only one such point lets tiny pose roundoff select very different
        contact Jacobians. Its two endpoint spheres provide stable additional
        witnesses for the same protected volume and keep both ends from closing.
        """
        G, h = [], []
        for ca, cb in self.pairs:
            key = tuple(sorted((ca.name, cb.name)))
            if key not in self.endpoint_pairs:
                continue
            a0, a1 = ca.world(self.data)
            b0, b1 = cb.world(self.data)
            margin = self._margin_for(ca, cb)
            lag = (not self.lag_capsules) or ca.name in self.lag_capsules or cb.name in self.lag_capsules
            pad = (margin_pad * self.pair_lag_scale.get(key, 1.0)
                   if lag and (margin >= self.d_safe or key in self.pair_lag_scale) else 0.0)
            for c0, c1, p0, p1, first in ((a0, a1, b0, b1, True), (b0, b1, a0, a1, False)):
                axis = p1 - p0
                denom = axis @ axis
                for point in (c0, c1):
                    other = p0 + np.clip((point - p0) @ axis / max(denom, 1e-15), 0., 1.) * axis
                    wa, wb = (point, other) if first else (other, point)
                    separation = wa - wb
                    length = np.linalg.norm(separation)
                    gap = length - ca.radius - cb.radius
                    if gap >= self.d_inf:
                        continue
                    normal = separation / length if length > 1e-9 else np.array([0., 0., 1.])
                    Ja = self._point_jac(wa, ca.body)[:, dofs]
                    Jb = self._point_jac(wb, cb.body)[:, dofs]
                    slack = gap - margin - pad
                    cap = self.alpha * slack
                    if braking and slack > 0.0:
                        cap = min(cap, np.sqrt(2.0 * self.stop_decel * slack))
                    G.append(-(normal @ (Ja - Jb)))
                    h.append(cap)
        return np.asarray(G).reshape(-1, len(dofs)), np.asarray(h)

    def project(self, dq, dofs, braking=True, rows=None):
        """CBF VELOCITY PROJECTION — the "slide, don't stop" law.

        Returns the closest joint velocity to ``dq`` (Euclidean) satisfying every
        active barrier row  G q̇ ≤ h : the minimal-norm correction removes ONLY the
        normal (gap-closing) excess along each constraint's gradient and leaves the
        TANGENTIAL component untouched, so a hand commanded into the torso slides
        along it toward the target instead of stopping dead (speed_scale) or being
        re-optimised sideways by the tracking QP (rows+solve_qp).

        With one active constraint this is exactly
            q̇' = q̇ − a·(aᵀq̇ − h)/(aᵀa)      [only when aᵀq̇ > h]
        i.e. classic velocity projection onto the CBF half-space; with several it
        is the least-distance QP over their intersection. Free motion (no row
        violated) is returned bit-identical.

        ``rows`` (optional) supplies pre-built (G, h) — e.g. rows STACKED from
        two poses. A servo-driven consumer must constrain against BOTH the
        commanded target pose (where the servos are told to go) and the MEASURED
        pose (where the metal actually is): sliding tangentially around a curved
        obstacle at the target pose is a CLOSING motion at the lagging physical
        pose (the corner-cut), measured as 4 cm of hand-hand penetration on the
        arms-crossing scenario when only the target rows were enforced."""
        dq = np.asarray(dq, float)
        G, h = rows if rows is not None else self.rows(dofs, braking=braking)
        if not len(G) or np.all(G @ dq <= h + 1e-12):
            return dq                              # free motion untouched
        if len(G) == 1:                            # closed form: one half-space
            a = G[0]
            aa = float(a @ a)
            if aa <= 1e-12:
                return dq
            return dq - a * ((float(a @ dq) - float(h[0])) / aa)
        # Several simultaneous rows: Dykstra's alternating projection computes the
        # EXACT Euclidean projection onto the intersection of the half-spaces —
        # deterministic and dependency-free (solve_qp's last-resort fallback would
        # silently drop the constraints when no QP solver is installed).
        x = dq.copy()
        p = np.zeros((len(G), len(dq)))
        for _ in range(50):
            for i, a in enumerate(G):
                aa = float(a @ a)
                if aa <= 1e-12:
                    continue
                y = x + p[i]
                viol = (float(a @ y) - float(h[i])) / aa
                xn = y - a * max(0.0, viol)
                p[i] = y - xn
                x = xn
            r = G @ x - h
            worst = float(np.max(r)) if len(r) else 0.0
            if worst <= 1e-9:
                break
        return x

    def speed_scale(self, dq, dofs):
        """Uniform slow-down factor s ∈ [0,1] for the velocity-tracking ("stop")
        law: scaling the tracking solution by s keeps every monitored gap's closing
        rate within the barrier limit  −ḋ ≤ α(d − d_s).

        Pure scaling preserves the motion DIRECTION, so the hand decelerates
        straight to a stop along the operator's intended path — no sideways push or
        redirection (the "force field" the tracking-QP produces). It only ever
        SLOWS a closing motion: tangential sliding and retraction are never
        impeded (s = 1 unless something is approaching faster than allowed), and at
        the safety margin the allowed approach is 0 so a head-on motion fully stops.

        Returns 1.0 when nothing is closing too fast (free motion untouched)."""
        s = 1.0
        self.min_dist = np.inf
        for ca, cb in self.pairs:
            pa0, pa1 = ca.world(self.data)
            pb0, pb1 = cb.world(self.data)
            wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
            sep = wa - wb
            L = np.linalg.norm(sep)
            d = L - ca.radius - cb.radius
            self.min_dist = min(self.min_dist, d)
            if d >= self.d_inf:
                continue
            n = sep / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
            a = (n @ (self._point_jac(wa, ca.body) - self._point_jac(wb, cb.body)))[dofs]
            v_close = -float(a @ dq)                 # >0 when the gap is shrinking
            if v_close > 1e-9:
                slack = max(0.0, d - self._margin_for(ca, cb))
                # Allowed closing speed: the smooth CBF ramp α·slack AND a braking-
                # distance cap √(2·a·slack) — the fastest approach from which the
                # link can still decelerate to rest within the remaining gap at
                # decel `stop_decel`. The √ term dominates at speed, so a fast
                # incoming motion is slowed early enough to stop before the margin
                # (robust to the velocity-level linearization error); the linear
                # term keeps the final approach gentle.
                cap = min(self.alpha * slack,
                          np.sqrt(2.0 * self.stop_decel * slack))
                if v_close > cap:
                    s = min(s, cap / v_close)
        return max(0.0, s)

    def barrier_force(self, qd, kv=None, kp=None):
        """Generalized "soft wall" force (length nv) for the TORQUE-driven path
        (compliance / gravity-comp), where there is no QP to clamp velocities.

        For every monitored pair within the influence distance it brakes ONLY the
        gap-closing motion and springs back out of the safety margin — it never
        pushes the link in a free direction, so a dragged arm decelerates to a
        stop at the boundary instead of being deflected sideways:

            ḋ      = a·q̇          with a = nᵀ(J_a − J_b)   (gap rate)
            ḋ_min  = −α (d − d_s)                            (allowed closing rate)
            f      = kv·max(0, ḋ_min − ḋ)  +  kp·max(0, d_s − d)   (≥ 0)
            τ     += aᵀ f

        Adding τ to the joint efforts opposes the approach. Returns a zero vector
        when nothing is within influence (no effect on free motion)."""
        kv = C.SC_BARRIER_KV if kv is None else kv
        kp = C.SC_BARRIER_KP if kp is None else kp
        tau = np.zeros(self.model.nv)
        self.min_dist = np.inf
        for ca, cb in self.pairs:
            pa0, pa1 = ca.world(self.data)
            pb0, pb1 = cb.world(self.data)
            wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
            sep = wa - wb
            L = np.linalg.norm(sep)
            d = L - ca.radius - cb.radius
            self.min_dist = min(self.min_dist, d)
            if d >= self.d_inf:
                continue
            n = sep / L if L > 1e-9 else np.array([0.0, 0.0, 1.0])
            a = n @ (self._point_jac(wa, ca.body) - self._point_jac(wb, cb.body))
            ddot = float(a @ qd)
            ds = self._margin_for(ca, cb)
            ddot_min = -self.alpha * (d - ds)
            f = kv * max(0.0, ddot_min - ddot) + kp * max(0.0, ds - d)
            if f > 0.0:
                tau += a * f
        return tau


def solve_qp(H, g, dq_lo, dq_hi, G, h):
    """min ½ q̇ᵀH q̇ + gᵀq̇  s.t.  G q̇ ≤ h, dq_lo ≤ q̇ ≤ dq_hi.

    Strictly-convex dense QP. Uses quadprog (fast dual active-set); falls back to
    osqp, then to a box-clamped unconstrained solve if no solver is present."""
    n = H.shape[0]
    # Stack box limits as general inequalities so every backend sees one form.
    Gbox = np.vstack([np.eye(n), -np.eye(n)])
    hbox = np.hstack([dq_hi, -dq_lo])
    Gall = np.vstack([G, Gbox]) if G.size else Gbox
    hall = np.hstack([h, hbox]) if G.size else hbox

    if _HAVE_QUADPROG:
        # quadprog: min ½xᵀG₀x − a₀ᵀx s.t. C₀ᵀx ≥ b₀.  Our G q̇ ≤ h ⇒ −G q̇ ≥ −h.
        G0 = 0.5 * (H + H.T) + 1e-9 * np.eye(n)      # symmetrise + tiny SPD jitter
        a0 = -g
        C0 = -Gall.T
        b0 = -hall
        try:
            return quadprog.solve_qp(G0, a0, C0, b0, 0)[0]
        except ValueError:
            pass                                     # infeasible/degenerate → fall through
    if _HAVE_OSQP:
        prob = osqp.OSQP()
        prob.setup(P=_sp.csc_matrix(0.5 * (H + H.T)), q=g,
                   A=_sp.csc_matrix(Gall), l=-np.inf * np.ones(len(hall)), u=hall,
                   verbose=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=200)
        res = prob.solve()
        if res.info.status_val in (1, 2):            # solved / solved-inaccurate
            return res.x
    # Last resort: unconstrained optimum, clamped into the box (no collision guard).
    dq = np.linalg.solve(0.5 * (H + H.T) + 1e-9 * np.eye(n), -g)
    return np.clip(dq, dq_lo, dq_hi)

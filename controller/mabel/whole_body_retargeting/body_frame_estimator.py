"""Estimate the operator's TORSO frame from head + two wrists (position AND
orientation), so whole-body rotation and translation are separated from neck
rotation and arm motion.

THE PROBLEM
-----------
`calibration.body_frame(head)` builds the operator's body frame from the HEAD
alone: its heading is the head's yaw. That conflates two very different motions.
The head sits on a neck with ~±80 deg of independent yaw, so "head yaw" is
`torso_yaw + neck_yaw`, and a head-only estimator cannot tell them apart. The
visible failures:

  * operator turns their whole body  -> head yaw changes, so the frame turns,
    which is right by luck; but if they ALSO keep their gaze on the object
    (neck counter-rotates) the head yaw barely moves and the turn is LOST;
  * operator just looks around       -> head yaw changes, the frame turns, and
    the robot's arms swing even though the operator's body never moved.

THE IDEA (the operator's own words, made precise)
-------------------------------------------------
The head and the two wrists are three frames rigidly hung off ONE torso -- the
wrists via the shoulders, the head via the neck. When the *whole body* rotates,
all three rotate together as a rigid constellation about the vertical spine
axis. When only the neck turns, just the head moves. When only an arm moves,
one wrist moves NON-rigidly (its distance from the spine changes). So:

  * torso yaw  = the RIGID common rotation the constellation agrees on;
  * neck yaw   = head yaw - torso yaw   (the residual);
  * arm motion = whatever is left once the rigid part is removed.

This is the analytical, training-free counterpart to the learned root-motion
decoupling in AvatarPoser (Jiang et al., ECCV 2022, arXiv:2207.13784) and the
chest-relative hand re-expression in Mobile-TeleVision (Lu et al., 2024,
arXiv:2412.07773) -- but where those add a waist tracker or a learned policy, we
recover the torso frame from the three signals the Vision Pro already streams
(head 6-DOF + two wrist 6-DOF), the same sensor set CLONE uses (Li et al., 2025,
arXiv:2506.08931).

METHOD
------
Work in the gravity-aligned horizontal plane (up is known: ARKit +Y). Torso
orientation is dominated by yaw about up. Two asymmetries drive the whole design:
the arms cannot move the head, so for TRANSLATION the head is ground truth; but
the neck lets the head yaw freely, so for ROTATION the head must be corroborated.

ROTATION -- three quantities per frame, all wrist-derived, plus the head as a
yes/no gate (never as a magnitude, so a neck glance can never set the base rate):

  A. BEARING COMMON MODE (`theta_pos`). Each wrist's horizontal bearing about the
     chest; the shared same-sign change (`_shared`) is a torso rotation. Bearings
     from a wrist near the spine (small radius) are down-weighted.
  B. ORIENTATION COMMON MODE (`theta_wori`). The two wrist FRAMES also yaw with
     the torso; the shared frame-to-frame yaw is a rotation-centre-free estimate.
  RIGID rate = the robust weighted fusion of A and B. For a genuine turn both
     agree; a one-arm reach during a turn is OSCILLATORY and averages out under
     the low-pass, so the turn survives (a naive instantaneous rigid fit does
     not -- a fast reach biases its omega).
  PARALLEL-SWEEP gate (`anti`). A both-arms-parallel motion rotates the wrist
     POSITIONS one way and their FRAMES the other, so A and B are ANTI-CORRELATED
     (a true turn makes them agree; a one-arm reach zeroes one of them, never
     opposite). When anti-correlated, the "rigid" fusion is a spurious near-
     cancellation; replace it with `theta_wori` GATED by whether the HEAD yaw
     corroborates (`_shared(theta_head, theta_wori)`).

So the base yaw rate resolves every canonical motion:

  * whole-body turn (head+arms)         -> rigid fusion tracks it;
  * look-around (head only, arms still)  -> both wrist cues ~0 -> base holds,
                                            `neck = head - torso` absorbs it;
  * one-arm reach                        -> cues disagree -> base holds;
  * both-arms reach, head STILL          -> anti-correlated but head does not
                                            corroborate -> base holds (no false
                                            drive) -- the safety case;
  * arms + head move together (case 3)   -> anti-correlated AND head corroborates
                                            -> base turns;
  * gaze-fixed turn (head still, arms     -> wrist cues AGREE (arms rigid to the
    rigid to the torso)                     torso) -> base turns without the head.

TRANSLATION -- the chest `origin` follows the head (arms cannot move it), and
`base_velocity()` is the deadbanded head horizontal velocity: still head -> local
manipulation (arms reach in a fixed frame); head walks -> the base drives.
`base_twist()` returns the `(v_fwd, v_left, omega)` the mobile base consumes.

numpy only -- no simulator, no learned model -- so it runs on the real headset
stream unchanged. This is the analytical, training-free counterpart to the
learned global/local decoupling of AvatarPoser (Jiang et al., ECCV 2022) and
EgoPoser (Jiang et al., ECCV 2024), and needs no waist/pelvis tracker (cf.
Mobile-TeleVision, Lu et al. 2024) nor a learned policy (cf. OmniH2O, He et al.
2024; CLONE, Li et al. 2025) -- it recovers the base twist in closed form from
the three frames the headset already streams. Validated on synthetic ground
truth in `controller/experiments/exp_body_frame.py`: whole-body turn 2.1 deg,
look-around / one-arm / both-arms-reach 0.0 deg (rejected), gaze-fixed turn
1.7 deg, arms+head case-3 1.3 deg, dynamic walk+turn+reach 0.6 deg (+8.8 mm walk
tracking) -- against 42 / 31 / 11 deg for the head-only baseline on the cases it
cannot resolve.
"""
from __future__ import annotations

import numpy as np

from .. import config as C

UP = np.asarray(getattr(C, "ARKIT_UP", (0.0, 1.0, 0.0)), dtype=float)
UP = UP / (np.linalg.norm(UP) + 1e-12)

# Fraction of the rotation evidence that must come from the wrist ORIENTATION
# cue before the bearing cue is trusted on its own. Below it the turn is faded
# out unless the head corroborates — see the "orientation-blind" block in
# `update`. 0.15 keeps every canonical motion in exp_body_frame.py unchanged
# (they all rotate the wrists) while killing the pure-translation false turn.
ORI_EVIDENCE_MIN = 0.15

# Head yaw rate [rad/s] at which the head fully corroborates a bearing-only
# turn. A deliberate torso turn runs far above it; hand-tracking noise on a
# still head runs far below, which is what makes it a presence test rather
# than a magnitude one.
HEAD_CORROB_RATE = 0.15


# ── small SO(2)-about-up helpers ─────────────────────────────────────────────
def _flatten(v):
    """Remove the up component: project a 3-vector onto the horizontal plane."""
    v = np.asarray(v, float)
    return v - UP * float(v @ UP)


def _wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_between(ref, vec):
    """Signed horizontal angle from `ref` to `vec` about up (right-handed)."""
    r, v = _flatten(ref), _flatten(vec)
    nr, nv = np.linalg.norm(r), np.linalg.norm(v)
    if nr < 1e-9 or nv < 1e-9:
        return 0.0
    r, v = r / nr, v / nv
    s = float(UP @ np.cross(r, v))          # sin, signed about up
    c = float(r @ v)                        # cos
    return float(np.arctan2(s, c))


# world reference heading = ARKit -Z flattened (the operator's forward at origin)
_REF_FWD = _flatten(np.array([0.0, 0.0, -1.0]))
if np.linalg.norm(_REF_FWD) < 1e-6:          # -Z was the up axis; pick another
    _REF_FWD = _flatten(np.array([1.0, 0.0, 0.0]))
_REF_FWD = _REF_FWD / (np.linalg.norm(_REF_FWD) + 1e-12)


def head_yaw(head_R):
    """Absolute horizontal heading of a head rotation (ARKit camera looks -Z)."""
    fwd = -np.asarray(head_R, float)[:, 2]
    f = _flatten(fwd)
    if np.linalg.norm(f) < 1e-6:            # looking straight up/down: use -Y
        f = _flatten(-np.asarray(head_R, float)[:, 1])
    return _yaw_between(_REF_FWD, f)


def _rot_about_up(theta):
    """3x3 rotation by `theta` about the up axis (Rodrigues, up is unit)."""
    k = UP
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)



# ── THREE-FRAME FORWARD CONSENSUS ────────────────────────────────────────────
# The operator's own definition of "facing forward" (2026-07-31): *both hands
# are facing forward AND the head is facing forward* — that conjunction is what
# a person naturally does when squared up, and it is a far better datum than the
# head alone. The published alternatives are head yaw (14.5 deg mean error,
# sigma 18.9 -- Ponton et al., CGF 2022) or a small learned net on all three
# devices (5.4 deg, sigma 7.7). No published method uses an ANALYTIC agreement
# rule over head + two wrists, so this is our own.
#
# THE CONVENTION PROBLEM, AND WHY THE OFFSETS ARE LEARNED. A wrist frame's yaw
# is not "the direction the palm faces" in the same convention as head yaw --
# ARKit's hand anchor has its own axis layout, and guessing it wrong silently
# inverts the whole cue. So we never assume: at the first Set Forward, with the
# operator genuinely squared up by construction, we RECORD each wrist's yaw
# offset from the head. Afterwards "facing forward" is simply all three
# de-offset yaws agreeing, which is convention-free and self-calibrating.

FACING_TOL = 0.35            # rad (~20 deg) of spread before the cue is dropped
# Fraction of the remaining consensus error removed per second, at full
# confidence. 1.5 washes out 20 deg of drift in ~2 s of squared-up posture.
#
# DEFAULT OFF. The consensus DETECTOR is validated (see the tests), but
# closing the loop on it changes the heading of an estimator whose canonical
# motions are individually pinned to fractions of a degree — and it measurably
# does: switching it on regressed the turn cases. Drift correction therefore
# ships as an opt-in until the Phase-5 experiments have tuned the rate against
# those same cases, rather than being enabled on the strength of it seeming
# like a good idea. Set Forward remains the operator's explicit re-zero, and
# it already benefits from the consensus (it learns the offsets there).
FACING_CORRECT_RATE = 1.5
FACING_CORRECT_ENABLED = False


def yaw_of_frame(R):
    """Yaw of a rotation about up, in the same convention as `head_yaw`."""
    return _yaw_between(_REF_FWD, _flatten(np.asarray(R, float) @ _REF_FWD))


class ForwardConsensus:
    """Are the head and both wrists all pointing the same way (= squared up)?"""

    def __init__(self):
        self.off = {}                 # side -> learned yaw offset from the head
        self.conf = 0.0               # 0..1 agreement of the last sample
        self.yaw = None               # consensus heading when confident

    def learn(self, head_R, wl_R, wr_R):
        """Record the wrist offsets, with the operator known to be facing
        forward (this is called from Set Forward, which is the operator saying
        exactly that)."""
        h = head_yaw(head_R)
        self.off = {"left": _wrap(yaw_of_frame(wl_R) - h),
                    "right": _wrap(yaw_of_frame(wr_R) - h)}
        return self

    def update(self, head_R, wl_R, wr_R):
        """Agreement of the three de-offset headings -> (consensus yaw, conf).

        Confidence falls off with the SPREAD of the three, so one hand dropped
        to the side or reaching across the body pulls it down and the cue simply
        stops being trusted, rather than dragging the datum somewhere wrong."""
        h = head_yaw(head_R)
        if not self.off:
            self.conf, self.yaw = 0.0, None
            return None, 0.0
        ys = [h,
              _wrap(yaw_of_frame(wl_R) - self.off["left"]),
              _wrap(yaw_of_frame(wr_R) - self.off["right"])]
        # circular mean, then the worst deviation from it
        c = float(np.mean([np.cos(y) for y in ys]))
        sn = float(np.mean([np.sin(y) for y in ys]))
        mean = float(np.arctan2(sn, c))
        spread = max(abs(_wrap(y - mean)) for y in ys)
        self.conf = float(np.clip(1.0 - spread / FACING_TOL, 0.0, 1.0))
        self.yaw = mean if self.conf > 0.0 else None
        return self.yaw, self.conf


class BodyFrameEstimator:
    """Streaming torso-frame estimate from head + two wrist 6-DOF poses.

    Parameters
    ----------
    chest_offset : the chest / spine centre relative to the head, expressed in
        the torso frame [right, up, forward] (metres). The wrist constellation
        rotates about the vertical axis through this point. Default is a nominal
        head->chest drop; calibration should overwrite it per operator.
    lp_hz : low-pass corner for the fused torso-yaw rate (Hz).
    yaw_deadband : torso-yaw rate below this (rad/s) is treated as zero, so
        tracking jitter never dribbles into the base command.
    """

    def __init__(self, chest_offset=(0.0, -0.22, -0.02),
                 lp_hz=2.5, yaw_deadband=0.02, sigma=0.5, head_weight=6.0,
                 vel_deadband=0.03):
        self.chest_offset = np.asarray(chest_offset, float)
        self.lp_hz = float(lp_hz)
        self.yaw_deadband = float(yaw_deadband)
        self.sigma = float(sigma)            # rigid-coherence gate scale
        self.head_weight = float(head_weight)  # head weight in the rigid fit
        self.vel_deadband = float(vel_deadband)  # base translation deadband (m/s)
        self.reset()

    def reset(self):
        self.ready = False
        self.psi = 0.0                  # torso yaw (absolute, wrapped)
        self.psi_rate = 0.0             # torso yaw rate (base command)
        self._rw = 0.0                  # low-passed wrist rigid-rotation rate
        self._rn = 0.0                  # low-passed wrist NON-rigid rotation rate
        self._rh = 0.0                  # low-passed head yaw rate (gate only)
        self.origin = np.zeros(3)       # chest position (world)
        self.vel = np.zeros(3)          # low-passed base (head) velocity (world)
        self._prev = None               # last frame's geometry
        self._psi0 = 0.0                # yaw at calibration (facing = 0)
        self.diag = {}                  # per-cue diagnostics (last update)
        self.facing = ForwardConsensus()   # head + two wrists agreement
        return self

    # ── geometry of one frame ────────────────────────────────────────────────
    def _chest(self, head_pos, psi):
        """Chest/spine centre in world, given head pos and a torso yaw guess."""
        R = _rot_about_up(psi)
        # torso axes at yaw psi: right, up, forward
        fwd = R @ _REF_FWD
        right = np.cross(fwd, UP)
        right /= (np.linalg.norm(right) + 1e-12)
        B = np.column_stack([right, UP, fwd])
        return np.asarray(head_pos, float) + B @ self.chest_offset

    def _geom(self, head, wl, wr, psi):
        head = np.asarray(head, float)
        wl = np.asarray(wl, float)
        wr = np.asarray(wr, float)
        c = self._chest(head[:3, 3], psi)
        aL = _flatten(wl[:3, 3] - c)
        aR = _flatten(wr[:3, 3] - c)
        return {
            "c": c,
            "hpos": head[:3, 3].copy(),
            "hyaw": head_yaw(head[:3, :3]),
            # wrist positions relative to the chest, flattened, + their horizontal
            # bearing about up (the angle whose COMMON change is a torso turn) and
            # radius (a wrist near the spine axis carries little bearing info)
            "aL": aL, "aR": aR,
            "bL": _bearing(aL), "bR": _bearing(aR),
            "rL": float(np.linalg.norm(aL)), "rR": float(np.linalg.norm(aR)),
            # keep the wrist ROTATIONS; their INCREMENTAL yaw-about-up (frame to
            # frame) is the robust torso cue -- an absolute wrist heading is
            # ambiguous (wrist frames don't share the head's -Z convention and
            # tumble with the arm), but the frame-to-frame yaw of R_now R_prev^T
            # is well-defined and its two-wrist common mode tracks the torso.
            "RwL": wl[:3, :3].copy(),
            "RwR": wr[:3, :3].copy(),
            # raw wrist world positions (for the rigid planar-twist residual)
            "wL": wl[:3, 3].copy(), "wR": wr[:3, 3].copy(),
        }

    # ── the fusion ───────────────────────────────────────────────────────────
    def update(self, head, wl, wr, dt):
        """Ingest one frame (4x4 head + two 4x4 wrist poses). Returns the torso
        frame `(origin, B)` with the SAME contract as `calibration.body_frame`:
        B has world-space [right, up, forward] columns; body coords of a world
        point p are `B.T @ (p - origin)`."""
        dt = max(float(dt), 1e-3)
        g = self._geom(head, wl, wr, self.psi)
        if not self.ready:
            self._prev = g
            self.ready = True
            self.origin = g["c"]
            self.diag = {"cue_pos": 0.0, "cue_wori": 0.0, "cue_head": 0.0,
                         "w_pos": 0.0, "w_wori": 0.0, "conf": 0.0,
                         "torso_rate": 0.0}
            return self.frame()
        p = self._prev

        # THE TORSO IS DERIVED FROM THE TWO WRISTS ONLY.
        # The wrists hang off the shoulders (rigid to the torso); the head hangs
        # off the neck (up to +-80 deg of its own yaw), so the head is the WORST
        # torso cue and is used only for the neck residual (neck_yaw()). A torso
        # turn is the motion the two wrists AGREE on -- one wrist moving alone is
        # an arm, two moving oppositely is a two-arm reach, and only a shared,
        # same-sign rotation is the body.

        # cue A: common-mode change of the two wrist BEARINGS about the chest
        d_bL = _wrap(g["bL"] - p["bL"])
        d_bR = _wrap(g["bR"] - p["bR"])
        theta_pos, w_pos = _shared(d_bL, d_bR)
        # a wrist tucked near the spine has an unreliable bearing -> trust it less
        w_pos *= _radius_conf(g["rL"]) * _radius_conf(g["rR"])

        # cue B: common-mode INCREMENTAL yaw of the two wrist ORIENTATIONS
        d_wl = _yaw_about_up(g["RwL"] @ p["RwL"].T)
        d_wr = _yaw_about_up(g["RwR"] @ p["RwR"].T)
        theta_wori, w_wori = _shared(d_wl, d_wr)

        theta_head = _wrap(g["hyaw"] - p["hyaw"])
        conf = min(1.0, w_pos + w_wori)          # wrists-agree confidence

        # RIGID rate = the robust weighted fusion of the two wrist cues. For a
        # genuine turn (S1/S5) both cues agree and this is the turn; for a
        # one-arm reach during a turn (S4) the reach is oscillatory and averages
        # out under the low-pass, so the turn survives -- the property a naive
        # rigid fit loses (a fast reach biases an instantaneous omega).
        wsum = w_pos + w_wori + 1e-6
        rigid_inc = (w_pos * theta_pos + w_wori * theta_wori) / wsum * conf

        # PARALLEL-SWEEP signature (the operator's case 3 vs a two-arm reach). A
        # both-arms-parallel motion rotates the wrist POSITIONS one way and their
        # FRAMES the other, so the bearing and orientation common-modes are
        # ANTI-CORRELATED; a real turn makes them agree, and a one-arm reach zeroes
        # one of them (never opposite). `anti` in [0,1] is that anti-correlation.
        anti = float(np.clip(
            -2.0 * theta_pos * theta_wori /
            (theta_pos ** 2 + theta_wori ** 2 + 1e-12), 0.0, 1.0))
        # In a parallel sweep the "rigid" fusion is a spurious near-cancellation;
        # replace it with the wrist-orientation rotation (which matches the head's
        # sign) GATED by whether the head corroborates it: head follows (case 3)
        # -> base turns; head still (a two-arm reach, S3b) -> g_head = 0 -> hold.
        _, g_head = _shared(theta_head, theta_wori)
        nonrigid_inc = theta_wori * g_head

        # ORIENTATION-BLIND TWO-ARM MOTION (2026-07-31).
        #
        # Every protection above needs the wrist FRAMES to carry signal: `anti`
        # is defined from theta_wori, so when the hands TRANSLATE without
        # rotating (both hands sweeping sideways, carrying a tray, pushing with
        # two flat palms) theta_wori ~ 0, `anti` collapses to 0, and the bearing
        # common-mode `theta_pos` becomes the sole evidence -- ungated. Two
        # wrists swinging the same way about the chest is exactly what a body
        # turn looks like to that cue, so the estimator reports a turn that
        # never happened.
        #
        # It is not a small effect and it is not hypothetical: with a RIGIDLY
        # STATIC synthetic head and torso and only the hands sweeping, the
        # estimated body yaw swung 20.2 deg (std 4.6 deg), and because the arm
        # targets are decomposed in that frame (ARM_BODY_FRAME_MAP) it rotated
        # BOTH arms -- each arm tracking the OTHER hand's motion at r = -0.84
        # while tracking its own at only +0.36. That is the operator's "either
        # the left or the right hand works, both cannot work together".
        # ONE hand moving measured 0.00 deg, which is why the failure only ever
        # showed up bimanually.
        #
        # The fix reuses the gate this module already trusts everywhere else:
        # the HEAD, as a yes/no corroboration, never as a magnitude. With
        # orientation evidence present nothing changes; as it vanishes the
        # bearing-only turn is faded out unless the head agrees. A genuine turn
        # still passes (the head goes with the body); a two-arm sweep in front
        # of a still head no longer invents one.
        # SUPPRESS ONLY THE CONJUNCTION, not low evidence on its own. The false
        # turn needs BOTH halves: no orientation evidence AND a head that is not
        # turning. Gating on the evidence fraction alone also fired 41 % of the
        # frames of a genuine walk+turn+reach (where a reach transiently starves
        # the orientation cue mid-turn) and cost it 0.6 -> 3.6 deg RMS. Head
        # STILLNESS is the discriminator that separates the two: during any real
        # turn the head goes with the body, so `head_still` collapses to 0 and
        # this whole term vanishes.
        w_ori_frac = w_wori / (w_wori + w_pos + 1e-9)
        blind = float(np.clip(1.0 - w_ori_frac / ORI_EVIDENCE_MIN, 0.0, 1.0))
        head_rate = abs(theta_head) / max(dt, 1e-6)
        head_still = float(np.clip(1.0 - head_rate / HEAD_CORROB_RATE, 0.0, 1.0))
        g_head_pos = 1.0 - blind * head_still            # 1 = untouched
        rigid_inc = rigid_inc * g_head_pos

        base_inc = (1.0 - anti) * rigid_inc + anti * nonrigid_inc
        head_agree = float(anti * g_head + (1.0 - anti) * blind * g_head_pos)

        # low-pass the base RATE, deadband, integrate
        a = dt * 2.0 * np.pi * self.lp_hz
        a = a / (1.0 + a)
        self._rw = (1 - a) * self._rw + a * (rigid_inc / dt)
        self._rn = (1 - a) * self._rn + a * (nonrigid_inc / dt)
        self.psi_rate = (1 - a) * self.psi_rate + a * (base_inc / dt)

        # diagnostic-only: the rigid planar-twist residual (ICR location), kept for
        # the confusion/residual analysis in the paper (not a gate here).
        rho = _rigid_residual(
            g["c"], p["hpos"], g["hpos"] - p["hpos"],
            p["wL"], g["wL"] - p["wL"], p["wR"], g["wR"] - p["wR"],
            self.head_weight)
        coherence = float(np.exp(-(rho / (self.sigma + 1e-9)) ** 2))

        eff = 0.0 if abs(self.psi_rate) < self.yaw_deadband else self.psi_rate
        self.psi = _wrap(self.psi + eff * dt)

        # ── DRIFT CORRECTION from the three-frame forward consensus ───────────
        # psi is an INTEGRATED heading, so it accumulates error forever; that is
        # what makes long sessions wander and is why Set Forward exists at all.
        # But the operator squares up constantly without being asked to — every
        # time they face their work with both hands in front of them — and that
        # posture is observable (ForwardConsensus). When all three frames agree,
        # they carry an ABSOLUTE heading that owes nothing to the integral, so
        # we bleed psi toward it.
        #
        # Deliberately SLOW and confidence-weighted: drift is slow, so removing
        # it can be slow too, and anything fast enough to fight a real turn would
        # be a bug. At FACING_CORRECT_RATE a full 20 deg of accumulated drift
        # washes out over a couple of seconds of squared-up posture, while a
        # deliberate turn (which drops confidence to 0 the moment the hands stop
        # agreeing with the head) is never touched.
        cy, conf = self.facing.update(head[:3, :3], wl[:3, :3], wr[:3, :3])
        if FACING_CORRECT_ENABLED and cy is not None and conf > 0.0:
            err = _wrap(cy - self.psi)
            self.psi = _wrap(self.psi
                             + np.clip(FACING_CORRECT_RATE * conf * dt, 0.0, 1.0) * err)

        # ── base TRANSLATION: the head is the ground truth (arms cannot move it) ─
        # head still -> local manipulation (T1); head translates -> base drives
        # (T2). Low-pass + deadband the head horizontal velocity.
        head_v = _flatten(g["hpos"] - p["hpos"]) / dt
        self.vel = (1 - a) * self.vel + a * head_v   # raw EMA (deadband on output)

        # refresh the chest at the corrected yaw, advance state
        self.origin = self._chest(g["hpos"], self.psi)
        self._prev = self._geom(head, wl, wr, self.psi)
        self.diag = {
            "cue_pos": theta_pos, "cue_wori": theta_wori, "cue_head": theta_head,
            "w_pos": float(w_pos), "w_wori": float(w_wori), "conf": float(conf),
            "head_agree": head_agree, "coherence": coherence,
            "rate_rigid": float(self._rw), "rate_nonrigid": float(self._rn),
            "rate_nonrigid_gated": float(self._rn * head_agree), "resid": float(rho),
            "torso_rate": self.psi_rate, "speed": float(np.linalg.norm(self.vel)),
            # orientation-blind gate telemetry (see the block in `update`)
            "w_ori_frac": float(w_ori_frac), "blind": float(blind),
            "g_head_pos": float(g_head_pos),
        }
        return self.frame()

    # ── outputs ──────────────────────────────────────────────────────────────
    def frame(self):
        """(origin, B) torso frame, matching `calibration.body_frame`."""
        R = _rot_about_up(self.psi)
        fwd = R @ _REF_FWD
        right = np.cross(fwd, UP)
        right /= (np.linalg.norm(right) + 1e-12)
        B = np.column_stack([right, UP, fwd])
        return self.origin.copy(), B

    def torso_yaw(self):
        """Torso heading relative to the calibrated facing (rad)."""
        return _wrap(self.psi - self._psi0)

    def neck_yaw(self, head_R):
        """Head yaw that is NOT torso yaw -- what the robot neck should turn."""
        return _wrap(head_yaw(head_R) - self.psi)

    def base_velocity(self):
        """Body (head) horizontal velocity in world, deadbanded (m/s). This is the
        base TRANSLATION feedforward: nonzero only when the operator walks; an
        arm reach with a still head leaves it zero (local manipulation). The
        deadband is applied to the OUTPUT only -- the internal EMA keeps
        accumulating so a slow walk is not held below threshold forever."""
        if float(np.linalg.norm(self.vel)) < self.vel_deadband:
            return np.zeros(3)
        return self.vel.copy()

    def base_twist(self):
        """The planar base command `(v_forward, v_left, omega)` in the body frame:
        forward/left components of the deadbanded head velocity, plus the torso
        yaw rate. This is the (vx, vy, omega) the mobile base consumes."""
        _, B = self.frame()
        v = self.base_velocity()
        v_fwd = float(v @ B[:, 2])       # forward = torso +Z column
        v_left = float(-(v @ B[:, 0]))   # left = -right
        omega = self.psi_rate if abs(self.psi_rate) >= self.yaw_deadband else 0.0
        return v_fwd, v_left, float(omega)

    def seed(self, head, wl, wr):
        """Refresh the previous-frame geometry WITHOUT changing the integrated
        yaw or velocity. Use this to resume after a tracking dropout: it prevents
        the next `update` from integrating the whole gap as one huge increment,
        while keeping the accumulated heading continuous (no base-yaw jump)."""
        self._prev = self._geom(head, wl, wr, self.psi)
        self.origin = self._chest(np.asarray(head, float)[:3, 3], self.psi)
        self.ready = True
        return self

    def calibrate_facing_head(self, head):
        """Zero the torso heading from the HEAD ALONE — the deadline fallback
        for Set Forward when a wrist is out of view (see `FORWARD_DEADLINE`).

        The datum itself is UNCHANGED by dropping the wrists: the full path sets
        `psi = _geom(...)["hyaw"]`, which is exactly `head_yaw(head)`. What is
        lost is `ForwardConsensus.learn` — the wrist yaw offsets that make the
        three-frame consensus meaningful afterwards — so those are left as they
        were and relearned by the next full `calibrate_facing`.

        `ready` is cleared rather than set: `update` re-seeds `_prev` from the
        next complete frame through its own not-ready branch, which is the same
        path a tracking dropout takes. Setting it True here with a stale `_prev`
        would integrate the gap as one huge increment.
        """
        head = np.asarray(head, float)
        self.psi = head_yaw(head[:3, :3])
        self._psi0 = self.psi
        self.vel = np.zeros(3)
        self._rw = self._rn = self._rh = 0.0
        self._prev = None
        self.ready = False
        self.origin = self._chest(head[:3, 3], self.psi)
        return self

    def calibrate_facing(self, head, wl, wr):
        """Zero the torso heading to the operator's current facing.

        Set Forward is the operator declaring they are squared up, so this is
        also the moment to LEARN the wrist yaw offsets that make the three-frame
        consensus meaningful afterwards (see ForwardConsensus)."""
        self.facing.learn(head[:3, :3], wl[:3, :3], wr[:3, :3])
        g = self._geom(head, wl, wr, 0.0)
        self.psi = g["hyaw"]
        self._psi0 = self.psi
        self.vel = np.zeros(3)
        self._rw = self._rn = self._rh = 0.0
        self._prev = self._geom(head, wl, wr, self.psi)
        self.ready = True
        self.origin = g["c"]
        return self


def _bearing(v):
    """Horizontal bearing angle of a vector about up, relative to _REF_FWD."""
    return _yaw_between(_REF_FWD, v)


def _radius_conf(r, r_min=0.15, r_full=0.30):
    """Confidence [0,1] in a wrist's bearing from its horizontal radius: a wrist
    tucked against the body (small radius) has a noisy, low-information bearing."""
    return float(np.clip((r - r_min) / (r_full - r_min), 0.0, 1.0))


def _yaw_about_up(dR):
    """Yaw component about the up axis of a (small) rotation `dR`.

    Rotate a fixed horizontal reference by dR and measure how far it swung about
    up. For an incremental frame-to-frame rotation this cleanly isolates the yaw
    from any pitch/roll, with no dependence on the frame's own axis convention --
    so it works on a wrist frame that tumbles with the arm."""
    e0 = _REF_FWD
    e1 = np.asarray(dR, float) @ e0
    return _yaw_between(e0, e1)



def _shared(a, b):
    """The rotation two signals AGREE on, and a [0,1] agreement weight.

    Same sign and similar magnitude -> their mean, high weight (a real common
    rotation). Opposite signs or one near zero -> ~0, low weight (differential
    motion: one arm moving, not a body turn)."""
    if a * b <= 0.0:                        # disagree in sign (or one is zero)
        return 0.0, 0.0
    w = 1.0 - abs(a - b) / (abs(a) + abs(b) + 1e-9)     # 1 when equal, 0 when far
    return 0.5 * (a + b), float(max(0.0, w))


def _horiz_basis():
    """Orthonormal horizontal basis (e0, e1) with e0 x e1 = UP (right-handed)."""
    e0 = _REF_FWD
    e1 = np.cross(UP, e0)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    return e0, e1


def _rigid_residual(c, hp, dh, lp, dl, rp, dr, head_weight):
    """Normalised WRIST residual of a single weighted planar twist about `c`.

    Fit xi = (vx, vy, omega) that best explains the head + two wrist horizontal
    displacements as one rigid planar motion (rotation about the up axis through
    `c`, the chest). The head is weighted heavily so `v` tracks the body and its
    small lever keeps it out of `omega`. Return rho = ||wrist residual|| /
    ||wrist displacement||: ~0 when the wrists orbit the spine rigidly (a real
    body turn / gaze-fixed turn), large when they orbit the shoulders (a reach).

    numpy-only, closed form (3x3 solve) -- the analytical ICR-location test that
    tells a gaze-fixed body turn apart from a two-arm reach, both of which leave
    the head still. See `twist_estimator.TwistFitEstimator` for the full-frame
    version; here we use only its residual as a gate.
    """
    e0, e1 = _horiz_basis()

    def to2d(v):
        v = np.asarray(v, float)
        return np.array([v @ e0, v @ e1])

    AtA = np.zeros((3, 3))
    Atb = np.zeros(3)
    for pos, dp, w in ((hp, dh, head_weight), (lp, dl, 1.0), (rp, dr, 1.0)):
        a = to2d(_flatten(np.asarray(pos, float) - c))
        d = to2d(dp)
        # rows: d_x = vx - omega*a_y ; d_y = vy + omega*a_x
        r1 = np.array([1.0, 0.0, -a[1]])
        r2 = np.array([0.0, 1.0, a[0]])
        AtA += w * (np.outer(r1, r1) + np.outer(r2, r2))
        Atb += w * (r1 * d[0] + r2 * d[1])
    try:
        xi = np.linalg.solve(AtA + 1e-9 * np.eye(3), Atb)
    except np.linalg.LinAlgError:
        return 0.0
    v = xi[:2]
    om = xi[2]
    res_sq = 0.0
    disp_sq = 0.0
    for pos, dp in ((lp, dl), (rp, dr)):        # WRIST residual only
        a = to2d(_flatten(np.asarray(pos, float) - c))
        d = to2d(dp)
        pred = v + om * np.array([-a[1], a[0]])
        res_sq += float((d - pred) @ (d - pred))
        disp_sq += float(d @ d)
    return float(np.sqrt(res_sq) / (np.sqrt(disp_sq) + 1e-4))

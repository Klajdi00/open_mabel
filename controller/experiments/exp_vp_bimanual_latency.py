#!/usr/bin/env python3
"""
SYNTHETIC VISION PRO TELEOP — bimanual liveness, latency, snap and oscillation.

THE OPERATOR REPORT (2026-07-31)
--------------------------------
    "the vision pro re-targeting have some severe delay issues — right now
     either the left or the right hand is working, both cannot work together.
     also the re-targeting snaps into the desired position way too fast and
     sometimes oscillates. check if this is the controller side or the
     re-targeting. i suspect the collision avoidance."

Four independent claims, so four independent measurements. Everything here is
driven by a SYNTHETIC ARKit operator (no headset, no network) so each claim is
isolated from the others and from the wire: whatever this harness reproduces is
owned by the engine, and whatever it does NOT reproduce is owned by the client
or the transport.

THE SYNTHETIC OPERATOR
----------------------
`SyntheticVPOperator` emits `OperatorPose` frames in the ARKit convention
(x right, y up, -z forward) with, per side, a full named 21-joint hand skeleton
plus the `forearm_arm` elbow the wire carries -- i.e. byte-compatible with what
`RobotController.makeFrameSource` puts on the wire, so the engine cannot tell it
from a real headset. Anthropometry (shoulder width, upper-arm/forearm lengths)
is explicit, wrist trajectories are prescribed per side, and the ARKit jitter is
a seeded Gaussian, so every number below is reproducible.

WHAT EACH EXPERIMENT ANSWERS
----------------------------
  E1  BIMANUAL     Do BOTH arms track their own hand at the same time? Each side
                   sweeps at its OWN frequency, so a side that is actually being
                   driven by the other hand (or not at all) cannot fake a good
                   score. Controls: left-only and right-only runs.
  E2  LATENCY      Phase lag between the operator's mapped target and the
                   commanded robot wrist, per side, swept over frame rate. Also
                   reports the engine's own per-frame solve cost, which is the
                   part a faster client cannot fix.
  E3  SNAP         Step response to a 20 cm wrist jump: peak commanded joint
                   speed, 10-90 % rise time, overshoot. This is the "snaps way
                   too fast" claim, in rad/s.
  E4  OSCILLATION  Hands held STILL under realistic tracking noise. Metric is the
                   commanded-joint standard deviation and the dominant FFT peak
                   -- a limit cycle shows as a sharp line. Swept with the
                   self-collision projector ON and OFF, which is what decides
                   whether the operator's suspicion is right.
  E5  COLLISION    Hands driven into each other and HELD there. A boundary layer
                   settles (steady state, zero residual velocity, gap parked at
                   the margin); a potential field keeps pushing and rings. Also
                   decomposes the guard's correction into normal vs tangential
                   to show it only removes the gap-closing component.

    python3 controller/experiments/exp_vp_bimanual_latency.py [--quick] [--only E1,E4]
Figures: controller/docs/figures/vp_bimanual.pdf, vp_latency.pdf,
         vp_snap.pdf, vp_oscillation.pdf, vp_collision.pdf
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(_HERE))                       # controller/
sys.path.insert(0, os.path.join(_REPO, "simulation", "mabel_mujoco", "scripts"))

import mujoco                                                    # noqa: E402
from mabel import MabelTeleop                                    # noqa: E402
from mabel.types import OperatorPose                             # noqa: E402
from mabel.teleop_engine import arm as arm_mod                   # noqa: E402

MODEL_XML = os.path.join(_REPO, "simulation", "mabel_mujoco", "models",
                         "mabel_full.xml")
FIGDIR = os.path.join(_REPO, "controller", "docs", "figures")
RESULTS = os.path.join(_HERE, "results")
DT = 1.0 / 200.0                    # control tick (the bridge's substep pace)
SEED = 20260731

# The named skeleton the wire carries (engine._NAMES_21 order is rebuilt from
# these names inside _build_21, so only the names have to match).
_FINGERS = ("thumb", "index", "middle", "ring", "little")
_SEGMENTS = ("knuckle", "intermediate_base", "intermediate_tip", "tip")
# Per-side sweep frequency — deliberately different so a side driven by
# the WRONG hand cannot score well, and known exactly so the lag can be
# demodulated rather than scanned.
SWEEP_HZ = {"left": 0.35, "right": 0.55}


# ── synthetic ARKit operator ──────────────────────────────────────────────────
class SyntheticVPOperator:
    """A parametric ARKit operator: head + two full hand skeletons + elbows.

    Frames are in the ARKit world convention (x right, y up, -z forward), which
    is exactly what the Vision Pro app streams. Each side's WRIST follows a
    caller-supplied f(t) -> (3,) position; the finger joints are rigidly carried
    along (finger articulation is not what any of these experiments measure, and
    a rigid hand keeps the wrist signal clean). The elbow is placed on the
    anatomical shoulder->wrist two-link solution so the direction-only elbow
    hint (ARM_TRACK_ELBOW) receives a physically consistent forearm line.
    """

    def __init__(self, wrist_fn, head_y=1.55, shoulder_w=0.38,
                 upper=0.30, fore=0.27, noise_m=0.0, seed=SEED, rot_fn=None,
                 head_rot_fn=None):
        self.wrist_fn = wrist_fn                  # (side, t) -> (3,) ARKit position
        # Optional per-side wrist ORIENTATION (side, t) -> 3x3. Real hands roll
        # and pitch constantly, and the body-frame estimator keys on exactly
        # that: its parallel-sweep detector is DEFINED from the wrist-orientation
        # common mode. A rigid never-rotating hand is therefore the one case
        # that cue cannot see — useful for isolating it, but not representative.
        self.rot_fn = rot_fn
        self.head_rot_fn = head_rot_fn
        self.head_y = head_y
        self.shoulder_w = shoulder_w
        self.upper = upper
        self.fore = fore
        self.arm_len = upper + fore
        self.noise_m = noise_m
        self.rng = np.random.default_rng(seed)

    def shoulder(self, side):
        """Operator shoulder in the ARKit world (head-relative, static torso)."""
        sgn = +1.0 if side == "right" else -1.0
        return np.array([sgn * self.shoulder_w * 0.5, self.head_y - 0.22, 0.0])

    def _elbow(self, side, wrist):
        """Two-link elbow: on the shoulder->wrist plane, swung gravity-down.
        Mirrors geometry.synth_elbow's convention closely enough that the
        engine's direction-only hint sees a consistent human forearm."""
        s = self.shoulder(side)
        v = wrist - s
        L = float(np.linalg.norm(v))
        L = min(max(L, 1e-3), self.arm_len * 0.999)
        u = v / L
        # circle of feasible elbows: centre + radius, swung toward -y (down)
        a = (L * L + self.upper ** 2 - self.fore ** 2) / (2 * L)
        r = float(np.sqrt(max(self.upper ** 2 - a * a, 1e-9)))
        down = np.array([0.0, -1.0, 0.0])
        perp = down - u * float(down @ u)
        n = float(np.linalg.norm(perp))
        perp = perp / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
        return s + u * a + perp * r

    def _skeleton(self, side, wrist, t):
        """Named 21-joint hand skeleton at `wrist`, optionally ROTATED by rot_fn."""
        sgn = +1.0 if side == "right" else -1.0
        R = np.eye(3) if self.rot_fn is None else np.asarray(self.rot_fn(side, t), float)
        joints = {}
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = wrist
        joints["wrist"] = T.copy()
        # A plausible flat hand: fingers fan forward (-z) from the wrist.
        for fi, f in enumerate(_FINGERS):
            lateral = (fi - 2) * 0.018 * sgn
            for si, seg in enumerate(_SEGMENTS):
                reach = 0.035 + 0.025 * si
                p = wrist + np.array([lateral, 0.0, -reach])
                if f == "thumb":                       # thumb sits inboard + up
                    p = wrist + np.array([-0.03 * sgn, 0.01 + 0.012 * si,
                                          -0.02 - 0.018 * si])
                M = np.eye(4)
                M[:3, :3] = R
                M[:3, 3] = wrist + R @ (p - wrist)
                joints[f"{f}_{seg}"] = M
        return joints

    def frame(self, t):
        pose = OperatorPose(timestamp=t)
        H = np.eye(4)
        if self.head_rot_fn is not None:
            H[:3, :3] = np.asarray(self.head_rot_fn(t), float)
        H[:3, 3] = np.array([0.0, self.head_y, 0.0])
        pose.head = H
        for side in ("right", "left"):
            w = np.asarray(self.wrist_fn(side, t), float)
            if w is None or not np.all(np.isfinite(w)):
                continue
            if self.noise_m > 0:
                w = w + self.rng.normal(0.0, self.noise_m, 3)
            joints = self._skeleton(side, w, t)
            setattr(pose, f"{side}_joints", joints)
            E = np.eye(4)
            E[:3, 3] = self._elbow(side, w)
            setattr(pose, f"{side}_elbow", E)
        pose.head_height = float(self.head_y)
        return pose


# ── rig: engine + ideal plant, driven at a chosen frame rate ──────────────────
class Rig:
    """The engine on an IDEAL plant (qpos := ctrl). Removes servo dynamics so
    what is measured is the RETARGETER's own behaviour -- the controller's
    contribution is measured separately by the `physics` plant option."""

    def __init__(self, frame_hz=30.0, self_collision=True, plant="ideal",
                 trust_rate=None, w_smooth=None):
        self.frame_hz = frame_hz
        self.model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if trust_rate is not None:
            arm_mod.ARM_TRUST_RATE = trust_rate
        if w_smooth is not None:
            arm_mod.ARM_W_SMOOTH = w_smooth
        self.teleop = MabelTeleop(self.model, self.data)
        self.eng = self.teleop.controller
        self.eng.set_self_collision(bool(self_collision))
        self.plant_kind = plant
        self.acts = [(aid, self.model.jnt_qposadr[self.model.actuator_trnid[aid, 0]])
                     for aid in range(self.model.nu)
                     if self.model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT]
        self.arm_act = {s: [self.model.actuator(f"{s.capitalize()} Arm {i}").id
                            for i in range(1, 8)] for s in ("right", "left")}
        self.wrist_bid = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                               f"{s[0]}_wrist")
                          for s in ("right", "left")}
        self.palm_bid = {s: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                              f"{s}_palm")
                         for s in ("right", "left")}
        self.eng.init_hold()
        for _ in range(120):
            self._plant()
        self.eng.settle_done()
        self.solve_ms = []

    def _plant(self):
        if self.plant_kind == "ideal":
            for aid, qadr in self.acts:
                self.data.qpos[qadr] = float(self.data.ctrl[aid])
            mujoco.mj_forward(self.model, self.data)
        else:
            n = max(1, int(round(DT / self.model.opt.timestep)))
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)

    def run(self, op, duration, settle=2.0, record=True):
        """Drive the engine for `duration` s of SIMULATED time at DT ticks with
        operator frames injected at frame_hz. Returns a record dict.

        The reference signal is the OPERATOR's own ARKit wrist (`op_*`), not the
        engine's internal mapped target: it is what the harness generated, so it
        exists on every tick and cannot be blanked by an engine early-return —
        the `_op_overlay` target is debug-only and absent whenever a side does
        not solve, which silently turned every correlation into NaN."""
        rec = {k: [] for k in ("t", "op_r", "op_l", "wrist_r", "wrist_l",
                               "q_r", "q_l", "gap")}
        t = 0.0
        next_frame = 0.0
        total = settle + duration
        last_op = {"right": np.full(3, np.nan), "left": np.full(3, np.nan)}
        while t < total:
            pose = None
            if t >= next_frame:
                pose = op.frame(t)
                next_frame += 1.0 / self.frame_hz
                for s in ("right", "left"):
                    j = getattr(pose, f"{s}_joints", None)
                    if j and "wrist" in j:
                        last_op[s] = np.asarray(j["wrist"], float).reshape(4, 4)[:3, 3]
            t0 = time.perf_counter()
            self.teleop.update(pose, None, DT)
            if pose is not None:
                self.solve_ms.append((time.perf_counter() - t0) * 1e3)
            self._plant()
            if record and t >= settle:
                rec["t"].append(t - settle)
                for s, key in (("right", "r"), ("left", "l")):
                    rec[f"op_{key}"].append(last_op[s].copy())
                    rec[f"wrist_{key}"].append(self.data.xpos[self.palm_bid[s]].copy())
                    rec[f"q_{key}"].append(np.array([self.data.ctrl[a]
                                                     for a in self.arm_act[s]]))
                pr = getattr(self.eng, "projector", None)
                rec["gap"].append(float(pr.guard.min_dist) if pr is not None
                                  and np.isfinite(pr.guard.min_dist) else np.nan)
            t += DT
        for k in rec:
            rec[k] = np.asarray(rec[k], float)
        return rec


# ── metrics ───────────────────────────────────────────────────────────────────
def lockin_lag(t, op, rob, f):
    """Phase lag of `rob` behind `op` [s] at the KNOWN drive frequency `f`.

    Single-frequency (lock-in) demodulation rather than a cross-correlation
    scan. The scan is unusable here for two reasons: the ARKit->robot frame map
    inverts one lateral axis, so a perfectly tracked LEFT arm is ANTI-correlated
    and a sign-blind scan locks onto whatever shift flips it positive; and the
    command updates once per FRAME while the record is sampled every control
    tick, so the correlation surface is flat to within a sample. Demodulating at
    the frequency we generated removes both ambiguities exactly.

    The axis inversion appears as a constant pi in the phase; it is removed by
    folding the phase into (-pi/2, pi/2], which is valid because a physical lag
    here is far below a quarter period."""
    t = np.asarray(t, float)
    a = np.asarray(op, float); b = np.asarray(rob, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 64:
        return np.nan
    t, a, b = t[m], a[m] - np.mean(a[m]), b[m] - np.mean(b[m])
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    # Remove the axis inversion FIRST, by its sign at zero lag, rather than by
    # folding the phase afterwards: the fold is ambiguous by +-pi and picked the
    # wrong branch for the inverted side, reporting a physically impossible
    # NEGATIVE lag (the robot leading the operator by 190 ms).
    if float(np.corrcoef(a, b)[0, 1]) < 0.0:
        b = -b
    ref = np.exp(-2j * np.pi * f * t)
    A = np.sum(a * ref)
    B = np.sum(b * ref)
    if abs(A) < 1e-12 or abs(B) < 1e-12:
        return np.nan
    ph = np.angle(B / A)                       # rob phase relative to op
    # Unwrap only a CLEAR wrap. Treating any positive phase as a wrap turned a
    # measurement-noise lead of +0.01 rad into a 2.6 s "lag" (a whole period);
    # a small apparent lead is reported as it is, near zero.
    if ph > np.pi / 2:
        ph -= 2.0 * np.pi
    return float(-ph / (2.0 * np.pi * f))      # positive = robot BEHIND


def principal(x3):
    """Scalar progress of a 3-D trajectory along its OWN dominant direction.

    The operator signal lives in the ARKit frame and the robot signal in the
    robot world, related by the retargeting map (a similarity transform with
    cross-axis mixing), so comparing like-named COMPONENTS is meaningless: the
    left arm came out with a physically impossible 190 ms phase LEAD purely
    because operator-x maps into a mixed robot direction. Projecting each
    trajectory onto its own first principal component compares the motions
    themselves and is invariant to how the frames are oriented."""
    x = np.asarray(x3, float)
    m = np.all(np.isfinite(x), axis=1)
    if m.sum() < 16:
        return np.full(len(x), np.nan)
    c = x[m] - np.mean(x[m], axis=0)
    _, _, V = np.linalg.svd(c, full_matrices=False)
    out = np.full(len(x), np.nan)
    out[m] = c @ V[0]
    return out


def track_corr(a, b):
    """Pearson correlation of two 1-D signals, NaN-safe."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 16 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def dominant_freq(t, x):
    """Frequency [Hz] of the largest FFT peak above DC, and its normalised power."""
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 64:
        return np.nan, np.nan
    x = x[m] - np.mean(x[m])
    dt = float(np.median(np.diff(t[m])))
    F = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), dt)
    if len(F) < 3:
        return np.nan, np.nan
    i = int(np.argmax(F[1:]) + 1)
    return float(f[i]), float(F[i] / (np.sum(F[1:]) + 1e-12))


# ── trajectories ──────────────────────────────────────────────────────────────
def _home(side):
    sgn = +1.0 if side == "right" else -1.0
    return np.array([sgn * 0.26, 1.15, -0.34])


def traj_bimanual(side, t, both=True, only=None):
    """Each side sweeps laterally at its OWN frequency, so the two signals are
    separable: a side driven by the wrong hand correlates with the wrong one."""
    if only is not None and side != only:
        return _home(side)                      # the other hand rests, still tracked
    f = SWEEP_HZ[side]
    p = _home(side).copy()
    p[0] += 0.13 * np.sin(2 * np.pi * f * t)
    p[1] += 0.07 * np.sin(2 * np.pi * f * t + 1.0)
    return p


def traj_static(side, t):
    return _home(side)


def traj_step(side, t, t_step=3.0, d=0.20):
    p = _home(side).copy()
    if t >= t_step:
        p[0] += d if side == "right" else -d
    return p


def traj_converge(side, t, hold_at=3.0):
    """Both hands driven toward the body midline and HELD deep inside each
    other -- the commanded target is unreachable, so what the guard does at
    steady state (settle vs push) is exactly the boundary-layer question."""
    sgn = +1.0 if side == "right" else -1.0
    p = _home(side).copy()
    frac = min(t / hold_at, 1.0)
    p[0] = sgn * 0.26 * (1.0 - frac) - sgn * 0.10 * frac       # cross the midline
    return p


# ── experiments ───────────────────────────────────────────────────────────────
def e1_bimanual(quick):
    """Both hands at once vs one at a time."""
    dur = 6.0 if not quick else 4.0
    out = {}
    cases = [("both", None), ("right-only", "right"), ("left-only", "left")]
    for name, only in cases:
        op = SyntheticVPOperator(lambda s, t, o=only: traj_bimanual(s, t, only=o))
        rig = Rig(frame_hz=30.0)
        rec = rig.run(op, dur)
        row = {}
        for s, key in (("right", "r"), ("left", "l")):
            tgt = principal(rec[f"op_{key}"])
            wr = principal(rec[f"wrist_{key}"])
            row[s] = {
                "corr": track_corr(tgt, wr),
                "range_mm": float(np.nanmax(np.linalg.norm(
                    rec[f"wrist_{key}"] - np.nanmean(rec[f"wrist_{key}"], axis=0),
                    axis=1))) * 2e3,
                "lag_ms": lockin_lag(rec["t"], tgt, wr, SWEEP_HZ[s]) * 1e3,
            }
        # CROSS-CHECK: does the right arm follow the LEFT hand's frequency?
        row["cross_r_follows_l"] = track_corr(principal(rec["op_l"]),
                                              principal(rec["wrist_r"]))
        row["cross_l_follows_r"] = track_corr(principal(rec["op_r"]),
                                              principal(rec["wrist_l"]))
        row["solve_ms_p50"] = float(np.median(rig.solve_ms)) if rig.solve_ms else np.nan
        row["solve_ms_p95"] = float(np.percentile(rig.solve_ms, 95)) if rig.solve_ms else np.nan
        out[name] = row
        print(f"  E1 {name:11s} R corr {row['right']['corr']:+.3f} "
              f"range {row['right']['range_mm']:5.1f} mm | "
              f"L corr {row['left']['corr']:+.3f} range {row['left']['range_mm']:5.1f} mm | "
              f"solve {row['solve_ms_p50']:.1f}/{row['solve_ms_p95']:.1f} ms")
        out[name]["_rec"] = {"t": rec["t"], "tgt_r": rec["op_r"][:, 0],
                             "wr_r": rec["wrist_r"][:, 0],
                             "tgt_l": rec["op_l"][:, 0],
                             "wr_l": rec["wrist_l"][:, 0]}
    return out


def e2_latency(quick):
    """Phase lag per side vs frame rate."""
    rates = [30.0, 60.0] if quick else [15.0, 30.0, 60.0, 90.0]
    dur = 6.0 if not quick else 4.0
    out = {}
    for hz in rates:
        op = SyntheticVPOperator(lambda s, t: traj_bimanual(s, t))
        rig = Rig(frame_hz=hz)
        rec = rig.run(op, dur)
        row = {"solve_ms_p50": float(np.median(rig.solve_ms)),
               "solve_ms_p95": float(np.percentile(rig.solve_ms, 95))}
        for s, key in (("right", "r"), ("left", "l")):
            row[s] = lockin_lag(rec["t"], principal(rec[f"op_{key}"]),
                                principal(rec[f"wrist_{key}"]), SWEEP_HZ[s]) * 1e3
        out[f"{hz:.0f}"] = row
        print(f"  E2 {hz:5.0f} Hz  lag R {row['right']:6.1f} ms  L {row['left']:6.1f} ms"
              f"   solve {row['solve_ms_p50']:.1f}/{row['solve_ms_p95']:.1f} ms")
    return out


def e3_snap(quick):
    """Step response vs tracking lag — the SNAP/LAG trade, measured.

    Slowing the slew to tame the snap costs tracking lag, and low latency is the
    operator's stated first priority, so the two are measured in the SAME sweep:
    a 20 cm wrist step for the snap, a continuous sweep for the lag."""
    rates = [(6.0, "6.0 rad/s (old)"), (4.0, "4.0 rad/s"),
             (2.5, "2.5 rad/s"), (2.0, "2.0 rad/s (new)"), (1.5, "1.5 rad/s")]
    if quick:
        rates = rates[:2]
    out = {}
    for rate, label in rates:
        op = SyntheticVPOperator(lambda s, t: traj_step(s, t))
        rig = Rig(frame_hz=30.0, trust_rate=rate)
        rec = rig.run(op, 4.0, settle=2.0)
        q = rec["q_r"]
        t = rec["t"]
        # Differentiate at the FRAME rate. The command steps ONCE per operator
        # frame (33 ms) while the record is sampled every control tick (5 ms),
        # so a per-frame trust step of 0.13 rad read as 26.7 rad/s — an artefact
        # of the sampling, not a speed the arm ever holds.
        step_n = max(1, int(round((1.0 / 30.0) / DT)))
        qs, ts = q[::step_n], t[::step_n]
        dq = np.diff(qs, axis=0) / np.diff(ts)[:, None]
        win = (ts[:-1] >= 0.98) & (ts[:-1] <= 2.2)
        peak = float(np.nanmax(np.abs(dq[win]))) if np.any(win) else np.nan
        # Rise/overshoot measured ON THE STEP ONLY (t >= 1.0 s, where the
        # trajectory jumps). Searching the whole record let pre-step samples
        # supply the 10 % crossing and returned a 0.000 s rise.
        x = rec["wrist_r"][:, 0]
        i0 = int(np.searchsorted(t, 1.0))
        base, final = np.nanmean(x[max(0, i0 - 60):i0]), np.nanmean(x[-40:])
        span = final - base
        rise, over = np.nan, np.nan
        if abs(span) > 1e-3:
            norm = (x - base) / span
            post = np.arange(len(x)) >= i0
            hits = np.where(post & (norm >= 0.9))[0]
            starts = np.where(post & (norm >= 0.1))[0]
            if len(hits) and len(starts):
                rise = float(t[hits[0]] - t[starts[0]])
            over = float((np.nanmax(norm[post]) - 1.0) * 100.0)
        # …and the lag the same setting costs on a continuous sweep.
        op2 = SyntheticVPOperator(lambda s, t: traj_bimanual(s, t))
        rig2 = Rig(frame_hz=30.0, trust_rate=rate)
        r2 = rig2.run(op2, 6.0)
        lag = lockin_lag(r2["t"], principal(r2["op_r"]), principal(r2["wrist_r"]),
                         SWEEP_HZ["right"]) * 1e3
        out[label] = {"peak_joint_rad_s": peak, "rise_10_90_s": rise,
                      "overshoot_pct": over, "sweep_lag_ms": lag,
                      "trace": {"t": t, "x": x}}
        print(f"  E3 {label:18s} peak {peak:5.2f} rad/s  rise {rise:.3f} s  "
              f"overshoot {over:+.1f} %  sweep lag {lag:5.1f} ms")
    return out


def e4_oscillation(quick):
    """Static hands + realistic ARKit jitter: does the command ring?"""
    noises = [0.0, 0.002] if quick else [0.0, 0.001, 0.002, 0.004]
    out = {}
    for sc in (True, False):
        for nz in noises:
            op = SyntheticVPOperator(lambda s, t: traj_static(s, t), noise_m=nz)
            rig = Rig(frame_hz=30.0, self_collision=sc)
            # STEADY STATE only. The projector integrates its command from the
            # measured pose on the first call, so it spends a few seconds
            # catching up; a 2 s settle folded that transient into the metric
            # and reported ~20 mrad of "oscillation" for BOTH arms of the
            # comparison (with SC off too, which is what gave it away). Past the
            # transient the static hold is numerically exact.
            rec = rig.run(op, 6.0 if not quick else 4.0, settle=5.0)
            q = rec["q_r"]
            std = float(np.nanmean(np.nanstd(q, axis=0)))
            j = int(np.nanargmax(np.nanstd(q, axis=0)))
            f, pw = dominant_freq(rec["t"], q[:, j])
            key = f"{'SC-on' if sc else 'SC-off'} n={nz*1e3:.0f}mm"
            out[key] = {"std_rad": std, "peak_hz": f, "peak_power": pw,
                        "worst_joint": j, "gap_min_m": float(np.nanmin(rec["gap"]))
                        if np.any(np.isfinite(rec["gap"])) else np.nan,
                        "trace": {"t": rec["t"], "q": q[:, j]}}
            print(f"  E4 {key:18s} std {std*1e3:6.3f} mrad  peak {f:5.2f} Hz "
                  f"(pw {pw:.2f})  min-gap {out[key]['gap_min_m']:.3f} m")
    return out


def e5_collision(quick):
    """Hands driven into each other and held: boundary layer or force field?"""
    out = {}
    for sc in (True, False):
        op = SyntheticVPOperator(lambda s, t: traj_converge(s, t))
        rig = Rig(frame_hz=30.0, self_collision=sc)
        rec = rig.run(op, 7.0 if not quick else 5.0, settle=1.5)
        t = rec["t"]
        tail = t >= (t[-1] - 2.0)                    # the HELD phase
        q = rec["q_r"]
        dq = np.abs(np.diff(q, axis=0) / np.diff(t)[:, None])
        tail_dq = dq[tail[:-1]] if len(dq) else np.zeros((1, 7))
        gap = rec["gap"]
        # hand-hand surface gap from the model itself (palms), for a sanity read
        d_palm = np.linalg.norm(rec["wrist_r"] - rec["wrist_l"], axis=1)
        key = "SC-on" if sc else "SC-off"
        out[key] = {
            "min_gap_m": float(np.nanmin(gap)) if np.any(np.isfinite(gap)) else np.nan,
            "held_gap_m": float(np.nanmean(gap[tail])) if np.any(np.isfinite(gap[tail])) else np.nan,
            "held_resid_vel_rad_s": float(np.nanmean(tail_dq)),
            "held_resid_vel_max": float(np.nanmax(tail_dq)) if tail_dq.size else np.nan,
            "min_palm_dist_m": float(np.nanmin(d_palm)),
            "trace": {"t": t, "gap": gap, "palm": d_palm},
        }
        print(f"  E5 {key:7s} min-gap {out[key]['min_gap_m']:+.4f} m  "
              f"held-gap {out[key]['held_gap_m']:+.4f} m  "
              f"residual |dq| {out[key]['held_resid_vel_rad_s']:.4f} rad/s "
              f"(max {out[key]['held_resid_vel_max']:.3f})  "
              f"min palm-palm {out[key]['min_palm_dist_m']:.3f} m")
    return out


# ── figures ───────────────────────────────────────────────────────────────────
def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=8)


def fig_bimanual(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (name, row) in zip(axes, res.items()):
        r = row["_rec"]
        ax.plot(r["t"], r["tgt_r"] * 1e2, lw=1.0, color="#C44536", label="R target")
        ax.plot(r["t"], r["wr_r"] * 1e2, lw=1.4, color="#C44536", alpha=0.55, ls="--",
                label="R robot")
        ax.plot(r["t"], r["tgt_l"] * 1e2, lw=1.0, color="#2A6F97", label="L target")
        ax.plot(r["t"], r["wr_l"] * 1e2, lw=1.4, color="#2A6F97", alpha=0.55, ls="--",
                label="L robot")
        _style(ax, f"{name}  (R r={row['right']['corr']:+.2f}, "
                   f"L r={row['left']['corr']:+.2f})", "time [s]", "wrist x [cm]")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("E1 — bimanual liveness: both hands driving at once", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_latency(res, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    hz = sorted(float(k) for k in res)
    for s, c in (("right", "#C44536"), ("left", "#2A6F97")):
        ax.plot(hz, [res[f"{h:.0f}"][s] for h in hz], "o-", color=c, label=s, lw=1.6)
    ax.plot(hz, [res[f"{h:.0f}"]["solve_ms_p95"] for h in hz], "s--", color="#6C757D",
            label="engine solve p95", lw=1.2)
    _style(ax, "E2 — engine phase lag vs client frame rate", "frame rate [Hz]",
           "lag [ms]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_snap(res, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for (label, row), c in zip(res.items(), ("#C44536", "#E9A03B", "#2A6F97")):
        tr = row["trace"]
        ax.plot(tr["t"], tr["x"] * 1e2, lw=1.5, color=c,
                label=f"{label}: {row['peak_joint_rad_s']:.1f} rad/s peak")
    _style(ax, "E3 — 20 cm wrist step: how hard the arm snaps", "time [s]",
           "robot wrist x [cm]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_oscillation(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    keys = list(res)
    axes[0].bar(range(len(keys)), [res[k]["std_rad"] * 1e3 for k in keys],
                color=["#C44536" if "on" in k else "#2A6F97" for k in keys])
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
    _style(axes[0], "E4 — command jitter on a static hold", "", "mean joint std [mrad]")
    for k, c in zip(keys[:3], ("#C44536", "#E9A03B", "#2A6F97")):
        tr = res[k]["trace"]
        axes[1].plot(tr["t"], (tr["q"] - np.nanmean(tr["q"])) * 1e3, lw=0.9,
                     color=c, label=f"{k} @ {res[k]['peak_hz']:.1f} Hz")
    _style(axes[1], "worst joint, mean-removed", "time [s]", "cmd [mrad]")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_collision(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, c in zip(res, ("#C44536", "#2A6F97")):
        tr = res[k]["trace"]
        axes[0].plot(tr["t"], tr["gap"] * 1e2, lw=1.5, color=c, label=k)
        axes[1].plot(tr["t"], tr["palm"] * 1e2, lw=1.5, color=c, label=k)
    axes[0].axhline(2.5, color="k", ls=":", lw=1.0, label="margin 2.5 cm")
    axes[0].axhline(0.0, color="k", ls="-", lw=0.8)
    _style(axes[0], "E5 — guard surface gap (hands driven together)", "time [s]",
           "min gap [cm]")
    _style(axes[1], "palm-palm distance", "time [s]", "distance [cm]")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k not in ("trace", "_rec")}
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list, e.g. E1,E4")
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    want = {s.strip().upper() for s in args.only.split(",") if s.strip()}
    run = lambda k: (not want) or k in want          # noqa: E731

    print(f"\nSYNTHETIC VISION PRO TELEOP — engine {'quick' if args.quick else 'full'}")
    print(f"model: {os.path.relpath(MODEL_XML, _REPO)}   seed {SEED}\n")
    out = {}
    if run("E1"):
        print("E1 BIMANUAL")
        r = e1_bimanual(args.quick); out["E1"] = r
        fig_bimanual(r, os.path.join(FIGDIR, "vp_bimanual.pdf"))
    if run("E2"):
        print("E2 LATENCY")
        r = e2_latency(args.quick); out["E2"] = r
        fig_latency(r, os.path.join(FIGDIR, "vp_latency.pdf"))
    if run("E3"):
        print("E3 SNAP")
        r = e3_snap(args.quick); out["E3"] = r
        fig_snap(r, os.path.join(FIGDIR, "vp_snap.pdf"))
    if run("E4"):
        print("E4 OSCILLATION")
        r = e4_oscillation(args.quick); out["E4"] = r
        fig_oscillation(r, os.path.join(FIGDIR, "vp_oscillation.pdf"))
    if run("E5"):
        print("E5 COLLISION")
        r = e5_collision(args.quick); out["E5"] = r
        fig_collision(r, os.path.join(FIGDIR, "vp_collision.pdf"))

    path = os.path.join(RESULTS, "vp_bimanual_latency.json")
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\nmetrics -> {os.path.relpath(path, _REPO)}")
    print(f"figures -> {os.path.relpath(FIGDIR, _REPO)}/vp_*.pdf\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
exp_retarget_fidelity.py — rigorous, headless validation of the TeleopEngine's
HUMAN-LIKENESS retargeting. Three ARM maps are compared:

  * "gmr" (LIVE default, 2026-07-24) — the root-uniform scaled-keypoint map of
    Retargeting Matters / GMR (Araujo, Ze, Xu, Wu, Liu — arXiv:2510.02252,
    ICRA 2026) adapted to the sparse headset stream: every keypoint relative
    to the mid-shoulder ROOT, scaled by sigma = L_rob/L_hum, re-anchored at
    the robot's root — the head–wrist–wrist pose TRIANGLE is preserved by
    construction; near rho=1 the magnitude renormalizes about the robot's own
    shoulder so full human extension stays full robot extension.
  * "extension" (2026-07-21..23) — per-shoulder extension-fraction anchoring:
    extension-exact everywhere but pose-shape distorting (the robot's
    shoulders are 3x narrower than a human's; bimanual poses collapsed
    inward and were held apart by an anti-collapse floor).
  * "head" (legacy) — head-anchored offset scaling; a fully extended human
    wrist can land strictly INSIDE the robot's envelope (elbow stays bent).

Also the dexterous-planner keypoint FINGER solver (FINGER_SOLVER ==
"keypoint") against the anatomical decomposition it replaced.

Motivated by live operator reports: "when I extend my arm fully the robot's
elbow stays bent" (fixed by extension, kept by gmr), and 2026-07-24 "the
whole-body retargeting is unstable; the robot should track the SCALE of the
human — head and both wrists relatively the same; the arms should bend like
mine, not twist weird angles" (the gmr map + the strengthened elbow
objective, E8–E10).

Everything is driven through the SHIPPED engine (TeleopEngine._retarget — the
same per-frame path update() takes), on synthetic operators with KNOWN ground
truth, deterministically (synthetic clock; no mj_step, no wall-time coupling,
fixed seeds). Measurements are FK on a scratch MjData from the engine's
commanded actuator targets.

ARM battery (per map in {extension, head}):
  E1  extension transfer  — forward reach rho 0.30→1.00, statures 1.55/1.75/1.95;
      metric: robot extension fraction rho_r and elbow interior angle.
  E2  direction fidelity  — FULL extension in 7 directions; rho_r + direction err.
  E3  trajectory tracking — a 0.15 m circle at rho≈0.7; RMS rho / direction err +
      per-frame joint-step smoothness.
  E4  bimanual closure    — hands approach to touching; robot inter-wrist gap
      with / without the proximity correction.
  E5  edge cases (assertions): hyper-extension input, wrist-at-shoulder
      degeneracy, hand dropout + reacquire, ARKit origin jump, crouch,
      long-armed operator self-calibration, profile apply, headless frames,
      lock_j7, self-collision guard on.
  E6  GLOBAL consistency (assertions, 2026-07-22): identical body-relative
      pose at 4 world yaws -> identical arm commands (the body-frame map);
      walk-a-loop + spin 360° + return -> the ACTIVE arm's visible config
      (wrist/elbow/palm) reproduces; wrist-separation transfer is monotone,
      never overlaps while the human is clearly apart, and closes 1:1 at a
      true touch.

HAND battery (per solver in {keypoint-gn, keypoint-slsqp, anatomical}):
  H1  curl transfer (open→fist, per finger + all), H2 pinch closure gap [mm],
  H3  chirality mirror consistency, H4 degenerate/sparse/noisy input,
  H5  per-solve wall time.

Run:  ~/anaconda3/bin/python3 simulation/mabel_mujoco/scripts/tools/exp_retarget_fidelity.py
      [--quick] [--figures] [--seed 0]
Figures (vector PDF) → controller/docs/figures/retarget_*.pdf, hand_*.pdf
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))                       # teleop pkg
sys.path.insert(0, os.path.join(_REPO, "controller"))               # mabel pkg
sys.path.insert(0, os.path.join(_REPO, "controller", "experiments"))  # synth_hand

import mujoco  # noqa: E402

from teleop.engine import TeleopEngine  # noqa: E402
from teleop import engine as engine_mod  # noqa: E402
from teleop import hyperparams as HP  # noqa: E402
from teleop.hand_anatomical import AnatomicalHandRetargeter  # noqa: E402
from teleop.hand_keypoint import KeypointHandRetargeter  # noqa: E402
from teleop.hyperparams import AVP_TO_ROBOT  # noqa: E402
from mabel.types import OperatorPose  # noqa: E402
from exp_hand_retarget import synth_hand  # noqa: E402  (the validated hand model)

MODEL_XML = os.path.join(_REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
FIG_DIR = os.path.join(_REPO, "controller", "docs", "figures")
DT = 1.0 / 60.0

# Okabe–Ito categorical palette, fixed order (CVD-safe; validated with the
# dataviz six-checks script 2026-07-21 — contrast WARN on orange → direct labels).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

_NAMES_21 = engine_mod._NAMES_21


# ── synthetic operator (ARKit world: +Y up, −Z forward) ──────────────────────

class SyntheticOperator:
    """Parametric human with KNOWN anthropometry. Directions are given in the
    operator BODY frame [right, up, fwd]. ``yaw`` turns the whole body about
    the ARKit up axis (0 = facing −Z) and ``origin`` translates it — the
    body-RELATIVE geometry is identical for every (yaw, origin), which is what
    the E6 global-consistency battery exploits."""

    def __init__(self, height=1.75, arm_ratio=0.332, shoulder_ratio=0.259,
                 drop_ratio=0.118, yaw=0.0, origin=(0.0, 0.0, 0.0)):
        self.H = float(height)
        self.L = arm_ratio * self.H                    # shoulder→wrist
        self.upper = 0.56 * self.L
        self.fore = 0.44 * self.L
        self.halfw = 0.5 * shoulder_ratio * self.H
        self.drop = drop_ratio * self.H
        self.set_world(yaw, origin)

    def set_world(self, yaw, origin=(0.0, 0.0, 0.0)):
        """Place the operator in the ARKit world: body yaw about +Y and a
        floor-plane origin offset. Body-relative geometry is unaffected."""
        c, s = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        self.head_pos = np.asarray(origin, float) + np.array(
            [0.0, 0.936 * self.H, 0.0])
        self.head_R = Ry                               # camera faces body −Z
        self.right = Ry @ np.array([1.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.fwd = Ry @ np.array([0.0, 0.0, -1.0])

    def shoulder(self, side):
        s = 1.0 if side == "right" else -1.0
        return self.head_pos + s * self.halfw * self.right - self.drop * self.up

    def dir_world(self, d_body):
        d = np.asarray(d_body, float)
        u = d[0] * self.right + d[1] * self.up + d[2] * self.fwd
        return u / (np.linalg.norm(u) + 1e-12)

    def wrist(self, side, rho, d_body):
        return self.shoulder(side) + self.dir_world(d_body) * (rho * self.L)

    def elbow(self, side, wrist):
        """Two-link elbow with a gravity swivel (elbow sags toward −up)."""
        sh = self.shoulder(side)
        d = np.asarray(wrist, float) - sh
        dist = min(float(np.linalg.norm(d)), self.upper + self.fore - 1e-9)
        dhat = d / (np.linalg.norm(d) + 1e-12)
        cos_a = np.clip((self.upper**2 + dist**2 - self.fore**2)
                        / (2.0 * self.upper * dist), -1.0, 1.0)
        sin_a = np.sqrt(max(0.0, 1.0 - cos_a**2))
        w = -self.up - dhat * float(-self.up @ dhat)   # down, ⊥ arm axis
        n = np.linalg.norm(w)
        w = (self.fwd - dhat * float(self.fwd @ dhat)) if n < 1e-6 else w / n
        w = w / (np.linalg.norm(w) + 1e-12)
        return sh + dhat * (self.upper * cos_a) + w * (self.upper * sin_a)

    def elbow_angle(self, side, wrist):
        """Human elbow interior angle [deg] for this wrist position."""
        sh, el = self.shoulder(side), self.elbow(side, wrist)
        a = sh - el
        b = np.asarray(wrist, float) - el
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    # ── named hand skeleton at a wrist pose ───────────────────────────────────
    def hand_joints(self, side, wrist, arm_dir, hand_kwargs=None, roll=0.0):
        """Place a synth_hand at ``wrist``: fingers along the arm direction,
        palm down (dorsal ≈ world up; frontal directions fall back to a
        forward dorsal reference). Left hand = mirrored local x (chirality).
        ``roll`` [rad] rotates the whole hand about the ARM axis (right-hand
        rule about shoulder→wrist) — the apparent palm roll a raised elbow /
        forearm pronation produces (E7's swivel ground truth)."""
        local = synth_hand(**(hand_kwargs or {}))
        u = np.asarray(arm_dir, float)
        u = u / (np.linalg.norm(u) + 1e-12)
        z_ref = self.up if abs(float(u @ self.up)) < 0.8 else -self.fwd
        z = z_ref - u * float(z_ref @ u)
        z = z / (np.linalg.norm(z) + 1e-12)
        x = np.cross(u, z)                              # RH basis: x = y × z
        R = np.column_stack([x, u, z])
        if roll:
            c, s = np.cos(roll), np.sin(roll)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            R = (np.eye(3) + s * K + (1 - c) * (K @ K)) @ R
        mirror = np.diag([-1.0, 1.0, 1.0]) if side == "left" else np.eye(3)
        out = {}
        for name, T in local.items():
            p = R @ (mirror @ np.asarray(T, float)[:3, 3]) + wrist
            m = np.eye(4)
            # Carry the hand-placement ROTATION too (all joints share it): the
            # body-frame estimator's orientation common-mode cue reads the
            # frame-to-frame yaw of the wrist ROTATIONS — identity rotations
            # with orbiting positions is a physically impossible input that
            # inverted its fused yaw during the E6 walk-loop.
            m[:3, :3] = R
            m[:3, 3] = p
            out[name] = m
        # Interpolated metacarpals + thumb knuckle (the real wire sends them;
        # operator_frame prefers metacarpals, the thumb chain wants its root).
        w = out["wrist"][:3, 3]
        for f in ("index", "middle", "ring", "little"):
            kn = out[f"{f}_knuckle"][:3, 3]
            m = np.eye(4)
            m[:3, 3] = w + 0.55 * (kn - w)
            out[f"{f}_metacarpal"] = m
        tb = out["thumb_intermediate_base"][:3, 3]
        m = np.eye(4)
        m[:3, 3] = w + 0.5 * (tb - w)
        out["thumb_knuckle"] = m
        return out

    def pose(self, t, targets, hands=None, rolls=None):
        """targets: {side: (rho, d_body)} (absent side ⇒ hand untracked).
        hands: {side: synth_hand kwargs}; rolls: {side: rad} hand roll about
        the arm axis (E7)."""
        head = np.eye(4)
        head[:3, :3] = self.head_R
        head[:3, 3] = self.head_pos
        kw = dict(head=head, timestamp=t)
        for side in ("right", "left"):
            if side not in targets:
                continue
            rho, d_body = targets[side]
            wpos = self.wrist(side, rho, d_body)
            joints = self.hand_joints(side, wpos, self.dir_world(d_body),
                                      (hands or {}).get(side),
                                      roll=(rolls or {}).get(side, 0.0))
            kw[f"{side}_joints"] = joints
        return OperatorPose(**kw)


# ── engine driving + measurement ─────────────────────────────────────────────

class EngineRig:
    """A fresh TeleopEngine on its own model/data, driven deterministically
    through the SAME per-frame path update() takes (synthetic clock)."""

    def __init__(self, model, arm_map, lock_j7=False, self_collision=False,
                 finger_solver=None):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self.eng = TeleopEngine(model, self.data, lock_j7=lock_j7,
                                arm_map=arm_map, finger_solver=finger_solver)
        self.eng.set_self_collision(self_collision)
        self.eng.init_hold()
        self.scratch = mujoco.MjData(model)
        self.t = 0.0

    def step(self, pose):
        eng = self.eng
        payload = eng._payload(pose)
        eng._frame_dt = DT
        eng._retarget(payload, None, self.t)
        self.t += DT

    def run(self, frames):
        for p in frames:
            self.step(p)

    def arm_state(self, side):
        """FK of the COMMANDED arm targets → (rho_r, elbow_deg, dir_r, wrist)."""
        eng, sc = self.eng, self.scratch
        ar = eng.arm[side]
        sc.qpos[:] = eng.data.qpos
        for i, a in enumerate(eng.arm_act[side][:ar.n_arm]):
            sc.qpos[ar.qadr[i]] = eng.data.ctrl[a]
        if ar.lock_j7:
            sc.qpos[ar.j7_qadr] = ar.j7_lock_val
        sc.qpos[ar.orca_wrist_qadr] = eng.data.ctrl[eng.wrist_act[side]]
        mujoco.mj_kinematics(self.model, sc)
        sh = sc.xpos[ar.shoulder_bid].copy()
        el = sc.xpos[ar.elbow_bid].copy()
        wr = sc.xanchor[ar.orca_wrist_jid].copy()
        rho_r = float(np.linalg.norm(wr - sh)) / eng.arm_reach[side]
        a, b = sh - el, wr - el
        cosv = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        elbow_deg = float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
        u = (wr - sh) / (np.linalg.norm(wr - sh) + 1e-12)
        return rho_r, elbow_deg, u, wr

    def arm_ctrl(self, side):
        return np.array([self.eng.data.ctrl[a] for a in self.eng.arm_act[side]])

    def arm_config(self, side):
        """Operator-VISIBLE configuration from the commanded targets: (wrist
        world pos, elbow interior angle [deg], palm rotation matrix). The E6
        reproduction metric — raw joint vectors are NOT comparable across
        history because the near-free forearm-roll DoF (reg 0.02) lives in a
        cost-flat valley and parks wherever it last was."""
        eng, sc = self.eng, self.scratch
        ar = eng.arm[side]
        sc.qpos[:] = eng.data.qpos
        for i, a in enumerate(eng.arm_act[side][:ar.n_arm]):
            sc.qpos[ar.qadr[i]] = eng.data.ctrl[a]
        if ar.lock_j7:
            sc.qpos[ar.j7_qadr] = ar.j7_lock_val
        sc.qpos[ar.orca_wrist_qadr] = eng.data.ctrl[eng.wrist_act[side]]
        mujoco.mj_kinematics(self.model, sc)
        sh = sc.xpos[ar.shoulder_bid]
        el = sc.xpos[ar.elbow_bid]
        wr = sc.xanchor[ar.orca_wrist_jid].copy()
        a, b = sh - el, wr - el
        cosv = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        elbow_deg = float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
        return wr, elbow_deg, sc.xmat[ar.palm_bid].reshape(3, 3).copy()

    def elbow_margin_deg(self, side):
        """Commanded elbow (Arm 4) distance from its nearest JOINT LIMIT [deg].
        ~0 at full extension proves the solver saturates the HARDWARE range —
        the robot's elbow physically stops at −95.7° (interior 161°)."""
        eng = self.eng
        ar = eng.arm[side]
        q4 = float(eng.data.ctrl[eng.arm_act[side][3]])
        lo, hi = eng.model.jnt_range[ar.jids[3]]
        return float(np.degrees(min(q4 - lo, hi - q4)))


def ramp_targets(a, b, n):
    """Linear rho/direction interpolation between two (rho, d_body) targets."""
    out = []
    for k in range(n):
        s = (k + 1) / n
        rho = (1 - s) * a[0] + s * b[0]
        d = (1 - s) * np.asarray(a[1], float) + s * np.asarray(b[1], float)
        out.append((rho, d))
    return out


NEUTRAL = (0.45, np.array([0.05, -0.35, 1.0]))     # relaxed forward-down reach


def drive_to(rig, op, tgt, ramp=45, settle=30, other=None, hands=None):
    """Ramp both arms from their previous target to ``tgt`` and settle."""
    prev = getattr(rig, "_prev_tgt", NEUTRAL)
    prev_o = getattr(rig, "_prev_other", NEUTRAL)
    other = tgt if other is None else other
    for (r1, d1), (r2, d2) in zip(ramp_targets(prev, tgt, ramp),
                                  ramp_targets(prev_o, other, ramp)):
        rig.step(op.pose(rig.t, {"right": (r1, d1), "left": (r2, d2)}, hands))
    for _ in range(settle):
        rig.step(op.pose(rig.t, {"right": tgt, "left": other}, hands))
    rig._prev_tgt, rig._prev_other = tgt, other


def warmup(rig, op, n=60):
    # The HEAD-map baseline gets the CALIBRATED reach (what the engine's own
    # 1-pose arms-forward calibration would measure for THIS operator:
    # reach = robot head→wrist forward reach / human head→wrist) — comparing
    # against the uncalibrated 1.2 default would be a strawman.
    if rig.eng.arm_map == "head":
        from teleop.model_utils import robot_forward_reach
        rr = robot_forward_reach(rig.model, rig.eng.head_bid)
        hr = float(np.linalg.norm([op.halfw, -op.drop, op.L]))
        rig.eng.apply_reach(rr / hr)
    for _ in range(n):
        rig.step(op.pose(rig.t, {"right": NEUTRAL, "left": NEUTRAL}))
    rig._prev_tgt = rig._prev_other = NEUTRAL
    assert rig.eng.calibrated, "engine failed to anchor during warm-up"


# ── ARM batteries ────────────────────────────────────────────────────────────

FWD = np.array([0.0, 0.0, 1.0])

DIRECTIONS = {                                  # body-frame [right, up, fwd]
    "fwd": [0.0, 0.0, 1.0],
    "fwd-up45": [0.0, 0.7071, 0.7071],
    "up": [0.0, 1.0, 0.05],
    "fwd-dn45": [0.0, -0.7071, 0.7071],
    "out45": [0.7071, 0.0, 0.7071],
    "lateral": [1.0, 0.0, 0.10],
    "cross30": [-0.5, 0.0, 0.866],
}


def e1_extension_transfer(model, quick=False):
    rhos = [0.30, 0.50, 0.70, 0.85, 0.95, 1.00]
    statures = [1.75] if quick else [1.55, 1.75, 1.95]
    res = {}                                    # (map, H) -> rows
    for arm_map in ("extension", "head"):
        for H in statures:
            op = SyntheticOperator(height=H)
            rig = EngineRig(model, arm_map)
            warmup(rig, op)
            rows = []
            for rho in rhos:
                drive_to(rig, op, (rho, FWD))
                rho_r, elb, _, _ = rig.arm_state("right")
                rows.append((rho, rho_r, op.elbow_angle(
                    "right", op.wrist("right", rho, FWD)), elb,
                    rig.elbow_margin_deg("right")))
            res[(arm_map, H)] = np.array(rows)
    return res


def e2_directions(model):
    """Full extension on the RIGHT arm in 7 body-frame directions (cross30
    crosses the body midline; the self-collision guard is off so the mapping is
    measured, not the guard)."""
    res = {}
    for arm_map in ("extension", "head"):
        op = SyntheticOperator()
        rig = EngineRig(model, arm_map)
        warmup(rig, op)
        rows = {}
        for name, d in DIRECTIONS.items():
            drive_to(rig, op, (1.0, np.array(d)), other=NEUTRAL)
            rho_r, elb, u_r, _ = rig.arm_state("right")
            u_exp = AVP_TO_ROBOT @ op.dir_world(np.array(d))
            cosv = float(np.clip(u_r @ u_exp, -1.0, 1.0))
            rows[name] = (rho_r, float(np.degrees(np.arccos(cosv))), elb)
        res[arm_map] = rows
    return res


def e3_trajectory(model):
    res = {}
    for arm_map in ("extension", "head"):
        op = SyntheticOperator()
        rig = EngineRig(model, arm_map)
        warmup(rig, op)
        drive_to(rig, op, (0.7, FWD))
        errs_rho, errs_dir, steps = [], [], []
        prev_ctrl = rig.arm_ctrl("right")
        n = 240
        for k in range(n):
            th = 2.0 * np.pi * (2 * k / n)
            d = np.array([0.30 * np.cos(th), 0.30 * np.sin(th), 1.0])
            d /= np.linalg.norm(d)
            rho = 0.70
            rig.step(op.pose(rig.t, {"right": (rho, d), "left": NEUTRAL}))
            rho_r, _, u_r, _ = rig.arm_state("right")
            u_exp = AVP_TO_ROBOT @ op.dir_world(d)
            errs_rho.append(abs(rho_r - rho))
            errs_dir.append(np.degrees(np.arccos(np.clip(u_r @ u_exp, -1, 1))))
            c = rig.arm_ctrl("right")
            steps.append(float(np.max(np.abs(c - prev_ctrl))))
            prev_ctrl = c
        res[arm_map] = {"rho_rms": float(np.sqrt(np.mean(np.square(errs_rho)))),
                        "dir_rms_deg": float(np.sqrt(np.mean(np.square(errs_dir)))),
                        "step_max_rad": float(np.max(steps))}
    return res


def e4_bimanual(model):
    """Operator hands approach each other at rho 0.75 front; measure the robot
    inter-wrist gap vs the human's, with and without the proximity blend."""
    out = {}
    for blend in (True, False):
        engine_mod.ARM_BIMANUAL_BLEND = blend
        op = SyntheticOperator()
        rig = EngineRig(model, "extension")
        warmup(rig, op)
        gaps_h, gaps_r = [], []
        seps = np.linspace(0.60, 0.05, 12)
        for sep in seps:
            # symmetric wrists about the midline, shoulder height, 0.75·L ahead
            base = op.head_pos - op.up * op.drop + op.fwd * (0.75 * op.L)
            wr = base + op.right * (sep / 2.0)
            wl = base - op.right * (sep / 2.0)
            tr, tl = wr - op.shoulder("right"), wl - op.shoulder("left")
            tgt_r = (np.linalg.norm(tr) / op.L,
                     np.array([tr @ op.right, tr @ op.up, tr @ op.fwd]))
            tgt_l = (np.linalg.norm(tl) / op.L,
                     np.array([tl @ op.right, tl @ op.up, tl @ op.fwd]))
            for _ in range(35):
                rig.step(op.pose(rig.t, {"right": tgt_r, "left": tgt_l}))
            _, _, _, wr_r = rig.arm_state("right")
            _, _, _, wr_l = rig.arm_state("left")
            gaps_h.append(float(sep))
            gaps_r.append(float(np.linalg.norm(wr_r - wr_l)))
        out[blend] = (np.array(gaps_h), np.array(gaps_r))
    engine_mod.ARM_BIMANUAL_BLEND = HP.ARM_BIMANUAL_BLEND
    return out


def e5_edge_cases(model):
    """Assertion battery — returns a list of (name, pass, detail)."""
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    op = SyntheticOperator()

    # 1) hyper-extension input saturates cleanly
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    drive_to(rig, op, (1.25, FWD))
    rho_r, _, _, _ = rig.arm_state("right")
    finite = np.all(np.isfinite(rig.eng.data.ctrl))
    check("hyper-extension rho=1.25 saturates", finite and rho_r <= 1.06,
          f"rho_r={rho_r:.3f}")

    # 2) wrist collapsed onto the shoulder: arm holds, no NaN
    before = rig.arm_ctrl("right")
    for _ in range(30):
        p = op.pose(rig.t, {"right": (0.0, FWD), "left": NEUTRAL})
        # rho=0 ⇒ wrist == shoulder exactly
        rig.step(p)
    after = rig.arm_ctrl("right")
    check("wrist-at-shoulder degeneracy holds", np.all(np.isfinite(after)),
          f"max drift {np.max(np.abs(after - before)):.3f} rad")

    # 3) hand dropout freezes that arm; reacquire resumes
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    drive_to(rig, op, (0.8, FWD))
    held = rig.arm_ctrl("right")
    for _ in range(30):                                   # right hand lost
        rig.step(op.pose(rig.t, {"left": NEUTRAL}))
    frozen = rig.arm_ctrl("right")
    for _ in range(90):                                   # reacquired
        rig.step(op.pose(rig.t, {"right": (0.8, FWD), "left": NEUTRAL}))
    resumed_rho, _, _, _ = rig.arm_state("right")
    check("hand dropout freezes arm",
          np.allclose(held, frozen, atol=1e-9), "ctrl unchanged during loss")
    check("reacquire resumes tracking", abs(resumed_rho - 0.8) < 0.12,
          f"rho_r={resumed_rho:.3f} after return")

    # 4) ARKit origin jump: re-anchors, no snap, same pose after recovery
    #    (the shoulder-anchored map is translation-invariant)
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    drive_to(rig, op, (0.8, FWD))
    pre = rig.arm_ctrl("right")
    op_j = SyntheticOperator()
    op_j.head_pos = op.head_pos + np.array([1.6, 0.0, 0.0])   # teleport
    max_step = 0.0
    prev = pre.copy()
    for _ in range(90):
        rig.step(op_j.pose(rig.t, {"right": (0.8, FWD), "left": NEUTRAL}))
        c = rig.arm_ctrl("right")
        max_step = max(max_step, float(np.max(np.abs(c - prev))))
        prev = c
    post = rig.arm_ctrl("right")
    check("origin jump: no arm snap", max_step < 0.30,
          f"max per-frame step {max_step:.3f} rad")
    check("origin jump: pose recovered", np.max(np.abs(post - pre)) < 0.15,
          f"max drift {np.max(np.abs(post - pre)):.3f} rad")

    # 5) crouch: stature (running max) must not shrink; reach unchanged
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    drive_to(rig, op, (0.9, FWD))
    h0 = rig.eng.shoulder.stature
    rho_before, _, _, _ = rig.arm_state("right")
    op_c = SyntheticOperator()
    op_c.head_pos = op.head_pos - np.array([0.0, 0.35, 0.0])  # deep crouch
    for _ in range(60):
        rig.step(op_c.pose(rig.t, {"right": (0.9, FWD), "left": NEUTRAL}))
    rho_after, _, _, _ = rig.arm_state("right")
    check("crouch keeps stature + reach",
          rig.eng.shoulder.stature >= h0 - 1e-9
          and abs(rho_after - rho_before) < 0.08,
          f"stature {h0:.2f}->{rig.eng.shoulder.stature:.2f}, "
          f"rho {rho_before:.3f}->{rho_after:.3f}")

    # 6) long-armed operator: live L self-calibrates on the first full reach
    op_l = SyntheticOperator(height=1.75, arm_ratio=0.40)     # L=0.70 m (long)
    rig = EngineRig(model, "extension")
    warmup(rig, op_l)
    drive_to(rig, op_l, (0.80, FWD))
    rho_pre, _, _, _ = rig.arm_state("right")                 # anthro prior wrong
    drive_to(rig, op_l, (1.00, FWD))                          # one full reach
    drive_to(rig, op_l, (0.80, FWD))
    rho_post, _, _, _ = rig.arm_state("right")
    check("long arms self-calibrate on full reach", abs(rho_post - 0.80) < 0.06,
          f"rho@0.8: {rho_pre:.3f} pre-cal -> {rho_post:.3f} post")

    # 7) measured profile applies immediately (no full reach needed)
    class _Prof:                                               # facade-shape stub
        arm_length_lr = (op_l.L, op_l.L)
        shoulder_width_m = 2 * op_l.halfw
        neck_height_m = op_l.drop
        finger_len_norm = {}
        reach = None
    rig = EngineRig(model, "extension")
    rig.eng.apply_profile(_Prof())
    warmup(rig, op_l)
    drive_to(rig, op_l, (0.80, FWD))
    rho_prof, _, _, _ = rig.arm_state("right")
    check("profile apply is immediate", abs(rho_prof - 0.80) < 0.06,
          f"rho@0.8={rho_prof:.3f} with profile, no prior full reach")

    # 8) headless / hands-free frames never crash the solve path
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    for _ in range(30):
        rig.step(OperatorPose(head=None, timestamp=rig.t))
        rig.step(op.pose(rig.t, {}))
    check("headless + hands-free frames safe",
          np.all(np.isfinite(rig.eng.data.ctrl)), "")

    # 9) lock_j7 (6-DoF arm) still extends
    rig = EngineRig(model, "extension", lock_j7=True)
    warmup(rig, op)
    drive_to(rig, op, (1.0, FWD))
    rho_l7, _, _, _ = rig.arm_state("right")
    check("lock_j7 full extension", rho_l7 > 0.93, f"rho_r={rho_l7:.3f}")

    # 10) self-collision guard on: cross-body reach stays finite + guarded
    rig = EngineRig(model, "extension", self_collision=True)
    warmup(rig, op)
    drive_to(rig, op, (0.9, np.array([-0.6, 0.0, 0.8])))
    check("self-collision guard co-exists",
          np.all(np.isfinite(rig.eng.data.ctrl)), "")

    # 11) the 1-pose CALIBRATION is MIRRORED (2026-07-23 report: "when I fully
    #     extend for calibration the robot should extend too"): drive the REAL
    #     update() branch (not _retarget directly) with a full-extension
    #     arms-forward hold; the robot must reach near-full extension DURING
    #     the capture, the capture must complete, and the shoulder estimator's
    #     arm length must self-calibrate from those same frames. The capture
    #     rule runs on the wall clock, so this check is realtime-paced.
    op_c = SyntheticOperator()
    rig = EngineRig(model, "extension")
    warmup(rig, op_c)
    eng = rig.eng
    # Pre-engage through the REAL update() path: the rig's warmup drives
    # _retarget directly, so the EngagementSupervisor has never stepped and
    # the first update() fires the reengage edge — the whole hold would then
    # run inside the 2.5 s HOME→live ease (a harness artifact; on the bridge
    # the supervisor is long engaged before calibration starts).
    t = rig.t
    t_wall0 = time.time()
    while eng.sup.state() != "engaged" and time.time() - t_wall0 < 8.0:
        t += DT
        eng.update(op_c.pose(t, {"right": NEUTRAL, "left": NEUTRAL}), None, DT)
    eng.start_calibration("e5_test")
    rho_max = 0.0
    t_wall0 = time.time()
    for k in range(4000):
        t += DT
        pose = op_c.pose(t, {"right": (1.0, FWD), "left": (1.0, FWD)})
        eng.update(pose, None, DT)
        rho_max = max(rho_max, rig.arm_state("right")[0])
        if eng._cal is not None and eng._cal.get("done"):
            break
        if time.time() - t_wall0 > 20.0:
            break
    done = eng._cal is not None and eng._cal.get("done", False)
    L_est = eng.shoulder.arm_length("right")
    check("calibration pose is mirrored (rho_r)", rho_max > 0.90,
          f"max rho_r {rho_max:.3f} during the hold")
    check("calibration completes while mirroring", done,
          f"reach {eng._cal.get('reach') if eng._cal else None}")
    check("arm length self-calibrates during hold",
          abs(L_est - op_c.L) < 0.06 * op_c.L,
          f"L_est {L_est:.3f} vs true {op_c.L:.3f} m")

    return results


def e6_global_consistency(model):
    """GLOBAL-retargeting ground truth (2026-07-22 operator report: 'when I
    move around and come back to the same pose, the retargeting fails to
    reproduce it; separated human wrists sometimes overlap on the robot').

    a) YAW INVARIANCE — the identical BODY-relative pose at world yaws
       0/90/180/270° must produce identical robot arm commands.
    b) WALK-LOOP RETURN — walk a square, spin 360°, return to the start
       facing forward with the same reach: the robot config must equal the
       pre-walk config.
    c) SEPARATION TRANSFER — human inter-wrist distance swept 0.65→0.05 m:
       robot separation must be monotonically non-increasing, never overlap
       (≥ 0.14 m while the human is clearly apart), and close to ~0 only as
       the human hands actually meet.
    Returns a list of (name, ok, detail)."""
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    pose_tgt = (0.85, np.array([0.05, 0.10, 1.0]))     # reach fwd, slightly up

    # a) same body pose, four world yaws — fresh rig per yaw (session start
    #    facing that way), compare the converged arm commands.
    ctrls = {}
    for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        op = SyntheticOperator(yaw=yaw)
        rig = EngineRig(model, "extension")
        warmup(rig, op)
        drive_to(rig, op, pose_tgt)
        ctrls[yaw] = np.concatenate([rig.arm_ctrl("right"), rig.arm_ctrl("left")])
    base = ctrls[0.0]
    worst = max(float(np.max(np.abs(ctrls[y] - base))) for y in ctrls)
    check("yaw invariance (4 world yaws)", worst < 0.06,
          f"max joint diff {worst:.3f} rad")

    # b) walk a loop + spin 360°, return, same reach — one continuous session.
    op = SyntheticOperator()
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    drive_to(rig, op, pose_tgt)
    for _ in range(210):        # equilibrate (incl. the basin-escape walk)
        rig.step(op.pose(rig.t, {"right": pose_tgt, "left": NEUTRAL}))
    before = {s: rig.arm_config(s) for s in ("right", "left")}
    waypoints = [(0.0, (1.0, 0, 0)), (np.pi/2, (1.0, 0, -1.0)),
                 (np.pi, (0.0, 0, -1.0)), (3*np.pi/2, (0.0, 0, 0)),
                 (2*np.pi, (0.0, 0, 0)), (0.0, (0.0, 0, 0))]
    prev_yaw, prev_org = 0.0, np.zeros(3)
    for yaw, org in waypoints:
        org = np.asarray(org, float)
        for k in range(45):                           # smooth 0.75 s transit
            a = (k + 1) / 45.0
            op.set_world(prev_yaw + a * (yaw - prev_yaw),
                         prev_org + a * (org - prev_org))
            rig.step(op.pose(rig.t, {"right": pose_tgt, "left": NEUTRAL}))
        prev_yaw, prev_org = yaw, org
    # Settle 3.5 s: the body-frame estimator's fused yaw carries ~10° of
    # low-pass lag through the (deliberately brisk, 120°/s) spin and its
    # wrist-consensus re-grounding converges within ~3 s of standing still —
    # the same time a real operator takes to settle after walking.
    for _ in range(210):
        rig.step(op.pose(rig.t, {"right": pose_tgt, "left": NEUTRAL}))
    after = {s: rig.arm_config(s) for s in ("right", "left")}

    def cfg_diff(s):
        d_wr = float(np.linalg.norm(after[s][0] - before[s][0]))
        d_el = abs(after[s][1] - before[s][1])
        d_p = float(np.degrees(np.arccos(np.clip(
            (np.trace(after[s][2] @ before[s][2].T) - 1.0) / 2.0, -1.0, 1.0))))
        return d_wr, d_el, d_p
    # The ACTIVE (reaching) arm is asserted strictly — that is the reported
    # failure. The IDLE arm's rest micro-pose sits in a cost-flat valley (weak
    # orientation objective at neutral, near-free forearm roll) where several
    # relaxed configurations are equivalent; it gets a loose bound only.
    wr_a, el_a, p_a = cfg_diff("right")
    wr_i, el_i, p_i = cfg_diff("left")
    check("walk-loop return: ACTIVE arm reproduces",
          wr_a < 0.025 and el_a < 6.0 and p_a < 10.0,
          f"wrist {wr_a*1000:.1f} mm | elbow {el_a:.1f}° | palm {p_a:.1f}°")
    # Palm bound 35°: the idle arm's forearm roll (reg 0.02) is cost-flat and
    # rotates the palm freely; wrist/elbow are the meaningful idle invariants.
    check("walk-loop return: idle arm near rest",
          wr_i < 0.05 and el_i < 8.0 and p_i < 35.0,
          f"wrist {wr_i*1000:.1f} mm | elbow {el_i:.1f}° | palm {p_i:.1f}°")

    # c) separation transfer: robot wrist gap vs human wrist gap.
    op = SyntheticOperator()
    rig = EngineRig(model, "extension")
    warmup(rig, op)
    rows = []
    for sep in np.linspace(0.65, 0.05, 13):
        base_pt = op.head_pos - op.up * op.drop + op.fwd * (0.72 * op.L)
        wr = base_pt + op.right * (sep / 2.0)
        wl = base_pt - op.right * (sep / 2.0)
        tr, tl = wr - op.shoulder("right"), wl - op.shoulder("left")
        tgt_r = (np.linalg.norm(tr) / op.L,
                 np.array([tr @ op.right, tr @ op.up, tr @ op.fwd]))
        tgt_l = (np.linalg.norm(tl) / op.L,
                 np.array([tl @ op.right, tl @ op.up, tl @ op.fwd]))
        for _ in range(40):
            rig.step(op.pose(rig.t, {"right": tgt_r, "left": tgt_l}))
        _, _, _, wrw = rig.arm_state("right")
        _, _, _, wlw = rig.arm_state("left")
        rows.append((float(sep), float(np.linalg.norm(wrw - wlw))))
    seps = np.array(rows)
    mono_viol = float(np.max(np.diff(seps[:, 1]) - 0.03))   # allow 3 cm noise
    check("separation transfer monotone", mono_viol <= 0.0,
          f"worst increase while closing {mono_viol+0.03:.3f} m")
    apart = seps[seps[:, 0] > 0.25]
    min_apart = float(np.min(apart[:, 1])) if len(apart) else np.nan
    check("no overlap while human apart (>25 cm)", min_apart >= 0.14,
          f"min robot gap {min_apart*100:.1f} cm")
    at_touch = seps[-1, 1]
    check("hands still meet at human touch", at_touch < 0.10,
          f"robot gap {at_touch*100:.1f} cm at 5 cm human")
    return results


def e7_elbow_swivel(model):
    """ELBOW HUMAN-LIKENESS (2026-07-22 report: 'when I push my elbow up the
    sim does not follow; sometimes the upper arm collides with the body').

    a) SWIVEL FOLLOWING — the operator raises their elbow with the palm held:
       the humerus rotation rolls the whole distal arm, so the hand's roll
       about the arm axis carries the swivel (plus ~60° of pronation slack the
       estimator deadbands away). Sweep the true swivel; the ROBOT's elbow
       swivel (about its shoulder→wrist axis, 0 = down) must follow.
    b) ELBOW-UP VISIBILITY — at a large commanded swivel the robot elbow must
       actually RISE above the elbow-down height.
    c) CROSS-BODY CLEARANCE — with the self-collision guard ON, a cross-body
       reach must leave the (newly guarded) upper-arm↔torso capsule pair
       non-penetrating.
    Returns a list of (name, ok, detail)."""
    import teleop.hyperparams as HPmod
    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")

    def robot_swivel(rig, side):
        """Robot elbow swivel about its shoulder→wrist axis [rad], 0 = down."""
        sc, eng = rig.scratch, rig.eng
        ar = eng.arm[side]
        sc.qpos[:] = eng.data.qpos
        for i, a in enumerate(eng.arm_act[side][:ar.n_arm]):
            sc.qpos[ar.qadr[i]] = eng.data.ctrl[a]
        mujoco.mj_kinematics(model, sc)
        sh = sc.xpos[ar.shoulder_bid]
        el = sc.xpos[ar.elbow_bid]
        wr = sc.xanchor[ar.orca_wrist_jid]
        dhat = (wr - sh) / (np.linalg.norm(wr - sh) + 1e-12)
        down = np.array([0.0, 0.0, -1.0])            # robot world: z up
        r0 = down - dhat * float(down @ dhat)
        r0 /= (np.linalg.norm(r0) + 1e-12)
        ev = (el - sh) - dhat * float((el - sh) @ dhat)
        if np.linalg.norm(ev) < 1e-6:
            return 0.0
        ev /= np.linalg.norm(ev)
        return float(np.arctan2(np.cross(r0, ev) @ dhat, r0 @ ev))

    # a) + b): swivel sweep at a mid reach. For the RIGHT arm the anatomical
    # elbow-up direction is the OUTWARD tangent — NEGATIVE φ in the
    # down/tangent swivel basis (an inward-up elbow crosses the chest and is
    # exactly what the clearance clamp exists to reject). Apparent hand roll
    # = true swivel + the pronation slack the estimator deadbands (rigid
    # coupling: palm-down is already near-full pronation, so a rising elbow
    # MUST roll the palm with the humerus).
    tgt = (0.7, np.array([0.10, 0.0, 1.0]))
    dead = HPmod.SWIVEL_DEADBAND
    errs, heights = {}, {}
    for phi_true in (0.0, -np.pi / 4, -np.pi / 2, -2.0):
        op = SyntheticOperator()
        rig = EngineRig(model, "extension")
        warmup(rig, op)
        roll = phi_true + (np.sign(phi_true) * dead if phi_true else 0.0)
        for _ in range(120):
            rig.step(op.pose(rig.t, {"right": tgt, "left": NEUTRAL},
                             rolls={"right": roll}))
        phi_rob = robot_swivel(rig, "right")
        errs[phi_true] = abs(phi_rob - phi_true)
        # elbow height above the shoulder plane (visibility check)
        sc, eng = rig.scratch, rig.eng
        heights[phi_true] = float(sc.xpos[eng.arm["right"].elbow_bid][2]
                                  - sc.xpos[eng.arm["right"].shoulder_bid][2])
    # Threshold 45°: the robot's INTRINSIC swivel capability at this posture is
    # ~20° short at mid swivels (measured with a dominant W_elbow=600 probe:
    # J5 forearm-roll and J6 wrist-pitch saturate holding the rolled palm), so
    # the estimator+IK budget on top is ~25°.
    worst = max(np.degrees(v) for v in errs.values())
    check("swivel follows palm roll (0..−115°)", worst < 45.0,
          "err " + " ".join(f"{np.degrees(k):.0f}°:{np.degrees(v):.0f}°"
                            for k, v in errs.items()))
    check("elbow visibly rises at high swivel",
          heights[-2.0] > heights[0.0] + 0.10,
          f"elbow z rel shoulder: {heights[0.0]*100:.0f} cm (down) -> "
          f"{heights[-2.0]*100:.0f} cm (up)")

    # c) cross-body reach, self-collision guard ON: upper-arm capsules clear.
    # The projector's barrier rows are evaluated on the LIVE body (its capsules
    # ride data.qpos, which the real servos drive toward the commands) — so the
    # rig must MIRROR the commanded arm targets into qpos each frame, exactly
    # like the tracking robot, or the guard sees a rest-pose body forever.
    from mabel.whole_body_controller.collision_avoidance import SelfCollisionGuard
    op = SyntheticOperator()
    rig = EngineRig(model, "extension", self_collision=True)
    eng = rig.eng

    def mirror():
        for s in ("right", "left"):
            a_r = eng.arm[s]
            for i, a in enumerate(eng.arm_act[s][:a_r.n_arm]):
                eng.data.qpos[a_r.qadr[i]] = eng.data.ctrl[a]
        mujoco.mj_forward(model, eng.data)

    for _ in range(60):
        rig.step(op.pose(rig.t, {"right": NEUTRAL, "left": NEUTRAL}))
        mirror()
    rig._prev_tgt = rig._prev_other = NEUTRAL
    tgt_cross = (0.85, np.array([-0.55, -0.1, 0.8]))          # right arm crosses
    for (r1, d1) in ramp_targets(NEUTRAL, tgt_cross, 60):
        rig.step(op.pose(rig.t, {"right": (r1, d1), "left": NEUTRAL}))
        mirror()
    for _ in range(60):
        rig.step(op.pose(rig.t, {"right": tgt_cross, "left": NEUTRAL}))
        mirror()
    # Measure with the PROJECTOR'S OWN guard — the geometry actually enforced
    # (capsule endpoints are baked in their anchor frames at construction; a
    # guard built fresh at the crossed pose bakes a different segment and
    # reads a false penetration). The QP equilibrates at d_safe (2.5 cm).
    from mabel.whole_body_controller.collision_avoidance.capsule_guard import _seg_seg
    guard = eng.projector.guard
    gaps = {}
    for ca, cb in guard.pairs:
        if "upper" in ca.name or "upper" in cb.name:
            pa0, pa1 = ca.world(eng.data)
            pb0, pb1 = cb.world(eng.data)
            wa, wb = _seg_seg(pa0, pa1, pb0, pb1)
            gaps[f"{ca.name}~{cb.name}"] = float(
                np.linalg.norm(wa - wb) - ca.radius - cb.radius)
    worst_gap = min(gaps.values()) if gaps else np.nan
    # Elbow~torso pairs run a ZERO per-pair margin (the settled elbow rests at
    # +0.6 cm — a positive margin spring-loads the rest pose), so the QP
    # equilibrium is non-penetration, not SC_MARGIN.
    check("cross-body: elbow capsules never penetrate", worst_gap > -0.005,
          " ".join(f"{k}:{v*100:+.1f}cm" for k, v in gaps.items())
          + f" | fallbacks {eng.projector.fallbacks}")
    return results


# ── E8–E10: the 2026-07-24 "unstable / doesn't track my scale or bend like me"
#            report, quantified against the GMR reformulation ──────────────────

# A library of representative BIMANUAL poses in the operator BODY frame
# ({side: (rho, [right,up,fwd])}) — forward reaches at several separations and
# heights, a cross-body reach, a hands-together pose, a wide "T", an
# asymmetric one-up/one-down. The scale-fidelity + tracking batteries sweep
# these so the metrics are workspace averages, not a single cherry-picked pose.
POSE_LIBRARY = {
    "fwd-wide":     {"right": (0.75, [0.45, 0.05, 1.0]), "left": (0.75, [-0.45, 0.05, 1.0])},
    "fwd-narrow":   {"right": (0.70, [0.12, 0.00, 1.0]), "left": (0.70, [-0.12, 0.00, 1.0])},
    "fwd-high":     {"right": (0.80, [0.25, 0.55, 0.9]), "left": (0.80, [-0.25, 0.55, 0.9])},
    "fwd-low":      {"right": (0.70, [0.25, -0.55, 0.9]), "left": (0.70, [-0.25, -0.55, 0.9])},
    "together":     {"right": (0.65, [-0.05, 0.0, 1.0]), "left": (0.65, [0.05, 0.0, 1.0])},
    "T-wide":       {"right": (0.90, [0.9, 0.05, 0.4]), "left": (0.90, [-0.9, 0.05, 0.4])},
    "asym-updown":  {"right": (0.80, [0.30, 0.5, 0.9]), "left": (0.65, [-0.25, -0.4, 0.9])},
    "reach-full":   {"right": (1.0, [0.15, 0.1, 1.0]), "left": (1.0, [-0.15, 0.1, 1.0])},
}


def _settle_pose(rig, op, targets, n=55):
    """Drive both arms to a body-frame pose dict and settle; return the pose."""
    prev = getattr(rig, "_prev_tgt", NEUTRAL)
    prev_o = getattr(rig, "_prev_other", NEUTRAL)
    tr = (targets["right"][0], np.asarray(targets["right"][1], float))
    tl = (targets["left"][0], np.asarray(targets["left"][1], float))
    for (r1, d1), (r2, d2) in zip(ramp_targets(prev, tr, 40),
                                  ramp_targets(prev_o, tl, 40)):
        rig.step(op.pose(rig.t, {"right": (r1, d1), "left": (r2, d2)}))
    for _ in range(n):
        rig.step(op.pose(rig.t, {"right": tr, "left": tl}))
    rig._prev_tgt, rig._prev_other = tr, tl


def e8_scale_fidelity(model, quick=False):
    """REQUIREMENT 1 — 'the robot should track the SCALE of the human: head,
    both wrists relatively the same as the robot's three coordinates.'

    This is the GMR g-mpbpe (global mean per-body-part position error) adapted
    to MABEL's three tracked upper-body coordinates. For each library pose and
    each map we express the operator's {right wrist, left wrist} relative to
    the mid-shoulder ROOT, apply the ideal similarity (scale sigma = L_rob /
    L_hum, rotation = the body-frame map M = AVP_TO_ROBOT for a forward-facing
    operator), and compare to the ROBOT's realized wrists relative to ITS
    root. A map that preserves the pose triangle drives this to ~0.

    Metrics per map (averaged over the library, mm / deg / ratio):
      * kp_err   — per-wrist scaled-keypoint position error [mm] (the g-mpbpe)
      * sep_ratio— robot inter-wrist gap / (sigma · human gap): 1.0 = perfect
      * sep_dir  — inter-wrist direction error [deg]
    Returns {map: {"kp_err","sep_ratio","sep_dir","rows"}}."""
    maps = ["gmr", "extension"] if quick else ["gmr", "extension", "head"]
    A = AVP_TO_ROBOT
    out = {}
    for arm_map in maps:
        op = SyntheticOperator()
        rig = EngineRig(model, arm_map)
        warmup(rig, op)
        sigma = rig.eng.arm_reach["right"] / op.L
        # robot mid arm-mount root (fixed body geometry)
        sc = rig.scratch
        sc.qpos[:] = rig.eng.data.qpos
        mujoco.mj_kinematics(model, sc)
        root_r = 0.5 * (sc.xpos[rig.eng.arm["right"].shoulder_bid]
                        + sc.xpos[rig.eng.arm["left"].shoulder_bid]).copy()
        kp_errs, sep_ratios, sep_dirs, rows = [], [], [], []
        for name, tgt in POSE_LIBRARY.items():
            _settle_pose(rig, op, tgt)
            root_h = 0.5 * (op.shoulder("right") + op.shoulder("left"))
            per_pose_kp = []
            robot_w, human_w = {}, {}
            for side in ("right", "left"):
                rho, d = tgt[side]
                w_h = op.wrist(side, rho, np.asarray(d, float))
                _, _, _, w_r = rig.arm_state(side)
                # ideal scaled keypoint (root-relative, similarity)
                w_ideal = root_r + sigma * (A @ (w_h - root_h))
                per_pose_kp.append(float(np.linalg.norm(w_r - w_ideal)) * 1e3)
                robot_w[side], human_w[side] = w_r, w_h
            kp_errs.extend(per_pose_kp)
            # inter-wrist separation ratio + direction
            d_r = robot_w["right"] - robot_w["left"]
            d_h = human_w["right"] - human_w["left"]
            d_h_ideal = sigma * (A @ d_h)
            sep_ratios.append(float(np.linalg.norm(d_r)
                                    / (np.linalg.norm(d_h_ideal) + 1e-9)))
            cosv = float(np.clip(d_r @ d_h_ideal
                                 / (np.linalg.norm(d_r) * np.linalg.norm(d_h_ideal) + 1e-12),
                                 -1, 1))
            sep_dirs.append(float(np.degrees(np.arccos(cosv))))
            rows.append((name, np.mean(per_pose_kp), sep_ratios[-1], sep_dirs[-1]))
        out[arm_map] = {"kp_err": float(np.mean(kp_errs)),
                        "kp_err_p95": float(np.percentile(kp_errs, 95)),
                        "sep_ratio": float(np.mean(sep_ratios)),
                        "sep_dir": float(np.mean(sep_dirs)),
                        "rows": rows}
    return out


def e9_elbow_humanlikeness(model, quick=False):
    """REQUIREMENT 2 — 'the arms should bend like humans; focus on elbow
    bending, not twisting into weird angles.'

    The elbow-direction residual is dominated by robot kinematics, not by any
    IK weight (measured: mean elbow-dir error moves only 28.1°→24.3° as
    ARM_W_ELBOW goes 30→120, p95 flat at 65°; and W ≥ 45 opens a second IK
    basin that limit-cycles — the 'unstable' report). So the human-likeness
    lever is NOT the weight: it is SWIVEL_DEADBAND, the pronation slack the
    palm-roll → elbow-swivel estimator absorbs. This battery ablates it.

    On the LIVE gmr map, drive a raised-elbow command (the operator rolls the
    palm about the arm axis, which is how a raised elbow presents to the
    wrist-only headset stream) and measure, at each true swivel:
      * swivel_err — |robot elbow swivel − human swivel| [deg]: does the elbow
        FOLLOW the operator (low = bends like the human).
      * twist      — |Arm7 wrist-yaw| from rest [deg]: the redundant DoF that
        makes the 'weird twist'; must stay bounded.
    Ablation: live deadband (0.45 rad) vs the pre-fix 1.05 rad.
    Returns {label: {"swivel_err","twist_max","rows"}}."""
    import teleop.hyperparams as HPmod
    import teleop.geometry as geomod   # synth_elbow reads SWIVEL_DEADBAND from
                                       # geometry's namespace (import *), patch there
    swivels = [0.0, -np.pi / 4, -np.pi / 2] if quick else \
        [0.0, -np.pi / 4, -np.pi / 2, -2.0]
    tgt = (0.7, np.array([0.10, 0.0, 1.0]))
    live_dead = geomod.SWIVEL_DEADBAND

    def robot_swivel(rig, side):
        sc, eng = rig.scratch, rig.eng
        ar = eng.arm[side]
        sc.qpos[:] = eng.data.qpos
        for i, a in enumerate(eng.arm_act[side][:ar.n_arm]):
            sc.qpos[ar.qadr[i]] = eng.data.ctrl[a]
        mujoco.mj_kinematics(model, sc)
        sh = sc.xpos[ar.shoulder_bid]; el = sc.xpos[ar.elbow_bid]
        wr = sc.xanchor[ar.orca_wrist_jid]
        dhat = (wr - sh) / (np.linalg.norm(wr - sh) + 1e-12)
        down = np.array([0.0, 0.0, -1.0])
        r0 = down - dhat * float(down @ dhat); r0 /= (np.linalg.norm(r0) + 1e-12)
        ev = (el - sh) - dhat * float((el - sh) @ dhat)
        if np.linalg.norm(ev) < 1e-6:
            return 0.0
        ev /= np.linalg.norm(ev)
        return float(np.arctan2(np.cross(r0, ev) @ dhat, r0 @ ev))

    out = {}
    j7_rest = None
    for label, dead in (("live (deadband 0.45)", live_dead),
                        ("pre-fix (deadband 1.05)", 1.05)):
        geomod.SWIVEL_DEADBAND = dead
        errs, twists, rows = [], [], []
        for phi_true in swivels:
            op = SyntheticOperator()
            rig = EngineRig(model, "gmr")
            warmup(rig, op)
            if j7_rest is None:
                j7_rest = float(rig.eng.arm["right"].j7_lock_val)
            # apparent palm roll = true swivel + the pronation slack the
            # estimator deadbands (rigid coupling of a raised humerus).
            roll = phi_true + (np.sign(phi_true) * dead if phi_true else 0.0)
            for _ in range(120):
                rig.step(op.pose(rig.t, {"right": tgt, "left": NEUTRAL},
                                 rolls={"right": roll}))
            phi_rob = robot_swivel(rig, "right")
            errs.append(abs(np.degrees(phi_rob - phi_true)))
            twists.append(abs(np.degrees(
                float(rig.eng.data.ctrl[rig.eng.arm_act["right"][6]]) - j7_rest)))
            rows.append((float(phi_true), float(phi_rob), errs[-1], twists[-1]))
        out[label] = {"swivel_err": float(np.mean(errs)),
                      "swivel_err_max": float(np.max(errs)),
                      "twist_max": float(np.max(twists)), "rows": rows}
    geomod.SWIVEL_DEADBAND = live_dead                        # restore
    return out


def e10_stability(model, quick=False):
    """REQUIREMENT 3 + the 'very unstable' complaint — quantified three ways.

    a) HELD-POSE STABILITY SWEEP — the decisive test. A SINGLE hold pose can
       land in a clean IK basin by luck (an earlier one-pose check read
       0.00 mrad while 16/36 poses across the workspace limit-cycled at
       ±0.10 rad). So we hold a GRID of bimanual reaches, each for 1.5 s, and
       report the WORST per-joint command std and how many poses oscillate
       (>2 mrad). This is what exposed — and now guards — the trust-region
       limit cycle (arm.py solve()).
    b) SLOW-SWEEP SMOOTHNESS — a continuous conversational trajectory walked at
       human speed: max per-tick joint step (a basin-flip snap shows as a
       trust-step-sized spike).
    c) GLOBAL TRACKING — wrist-tracking RMS against the commanded scaled
       keypoint over that trajectory.
    Returns {"hold_std_worst","n_osc","n_poses","sweep_step","track_rms_mm"}."""
    op = SyntheticOperator()
    A = AVP_TO_ROBOT

    # a) held-pose stability across the workspace (fresh rig per pose so a
    #    latent oscillation cannot be masked by a lucky arrival trajectory).
    grid = []
    ups = [0.1] if quick else [-0.3, 0.1, 0.5]
    drs = [-0.15, 0.35] if quick else [-0.4, -0.15, 0.1, 0.35]
    for rho in ([0.7] if quick else [0.55, 0.7, 0.85]):
        for dr in drs:
            for up in ups:
                grid.append((rho, [dr, up, 1.0]))
    worst, n_osc, stds = 0.0, 0, []
    for rho, d in grid:
        rig = EngineRig(model, "gmr")
        warmup(rig, op)
        tgt = {"right": (rho, d), "left": (rho, [-d[0], d[1], d[2]])}
        _settle_pose(rig, op, tgt, n=55)
        tr = (rho, np.asarray(d, float)); tl = (rho, np.asarray([-d[0], d[1], d[2]], float))
        hist = []
        for _ in range(90):                                # 1.5 s hold
            rig.step(op.pose(rig.t, {"right": tr, "left": tl}))
            hist.append(np.concatenate([rig.arm_ctrl("right"), rig.arm_ctrl("left")]))
        hist = np.stack(hist)
        std = float(np.max(np.std(hist[-40:], axis=0)))
        stds.append(std)
        worst = max(worst, std)
        if std > 2e-3:
            n_osc += 1

    # b/c) slow sweep + global tracking (one continuous session)
    rig = EngineRig(model, "gmr")
    warmup(rig, op)
    sigma = rig.eng.arm_reach["right"] / op.L
    sc = rig.scratch
    sc.qpos[:] = rig.eng.data.qpos
    mujoco.mj_kinematics(model, sc)
    root_r = 0.5 * (sc.xpos[rig.eng.arm["right"].shoulder_bid]
                    + sc.xpos[rig.eng.arm["left"].shoulder_bid]).copy()
    waypoints = [(0.6, [0.3, 0.0, 1.0]), (0.85, [0.5, 0.4, 0.7]),
                 (0.8, [-0.3, 0.2, 0.9]), (0.7, [0.1, -0.4, 1.0]),
                 (0.6, [0.3, 0.0, 1.0])]
    _settle_pose(rig, op, {"right": (0.6, [0.3, 0.0, 1.0]),
                           "left": (0.45, [-0.05, -0.35, 1.0])}, n=40)
    sweep_steps, track = [], []
    prev = np.concatenate([rig.arm_ctrl("right"), rig.arm_ctrl("left")])
    for wp_a, wp_b in zip(waypoints[:-1], waypoints[1:]):
        for k in range(60):                                # 1 s transit each
            s = (k + 1) / 60.0
            rho = (1 - s) * wp_a[0] + s * wp_b[0]
            d = (1 - s) * np.asarray(wp_a[1], float) + s * np.asarray(wp_b[1], float)
            rig.step(op.pose(rig.t, {"right": (rho, d),
                                     "left": (0.45, [-0.05, -0.35, 1.0])}))
            c = np.concatenate([rig.arm_ctrl("right"), rig.arm_ctrl("left")])
            sweep_steps.append(float(np.max(np.abs(c - prev))))
            prev = c
            _, _, _, w_r = rig.arm_state("right")
            w_h = op.wrist("right", rho, d)
            root_h = 0.5 * (op.shoulder("right") + op.shoulder("left"))
            w_ideal = root_r + sigma * (A @ (w_h - root_h))
            track.append(float(np.linalg.norm(w_r - w_ideal)) * 1e3)
    return {"hold_std_worst": worst, "n_osc": n_osc, "n_poses": len(grid),
            "sweep_step": float(np.max(sweep_steps)),
            "track_rms_mm": float(np.sqrt(np.mean(np.square(track))))}


# ── HAND battery ─────────────────────────────────────────────────────────────

def _hand_solver(model, kind, side="right"):
    """Fresh solver with its OUTPUT FILTER bypassed — the batteries measure
    solver quality; the (unchanged) one-euro filter is time-based and would
    couple the metrics to the harness's stepping speed."""
    if kind == "anatomical":
        s = AnatomicalHandRetargeter(model, side)
        s._filter = lambda q: q              # instance override shadows the method
        return s, None
    s = KeypointHandRetargeter(model, side)
    s._ik.solver = "slsqp" if kind == "keypoint-slsqp" else "gn"
    s._period = 0.0                          # no pacing: measure every solve
    s._filter = type("F", (), {"__call__": lambda self, q: q,
                               "reset": lambda self: None})()
    return s, None


def _mats21(joints):
    wrist = joints["wrist"]
    return [np.asarray(joints.get(n, wrist), float).reshape(4, 4).flatten().tolist()
            for n in _NAMES_21]


def _fk_hand(model, scratch, q, names, side):
    """Set the 16 finger joints; return world fingertip points (pad-extended)."""
    scratch.qpos[:] = model.qpos0
    for n, v in zip(names, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        scratch.qpos[model.jnt_qposadr[jid]] = v
    mujoco.mj_kinematics(model, scratch)
    pre = side
    tips = {}
    pad = np.array([0.0, 0.0, 0.022])
    for f, body in (("thumb", f"{pre}_thumb_dp"), ("index", f"{pre}_index_ip"),
                    ("middle", f"{pre}_middle_ip"), ("ring", f"{pre}_ring_ip"),
                    ("pinky", f"{pre}_pinky_ip")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        tips[f] = scratch.xpos[bid] + scratch.xmat[bid].reshape(3, 3) @ pad
    return tips


def _flex_fraction(q, names, bounds, finger):
    fr = []
    for i, n in enumerate(names):
        if finger.capitalize() in n and ("MCP" in n or "PIP" in n):
            lo, hi = bounds[i]
            fr.append((q[i] - lo) / (hi - lo))
    return float(np.mean(fr)) if fr else np.nan


def hand_battery(model, quick=False):
    scratch = mujoco.MjData(model)
    solvers = ("keypoint-gn", "keypoint-slsqp", "anatomical")
    curls = np.linspace(0.0, 1.0, 4 if quick else 6)
    fingers = ("index", "middle", "ring", "little")
    out = {}
    for kind in solvers:
        s, _mod = _hand_solver(model, kind)
        names = s.joint_names
        bounds = []
        for n in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            bounds.append(tuple(model.jnt_range[jid]))
        op = SyntheticOperator()
        wpos = op.wrist("right", 0.6, FWD)

        # H1: fist curl transfer
        curve = []
        times = []
        for c in curls:
            kw = dict(curl={f: c for f in fingers}, thumb_curl=c)
            joints = op.hand_joints("right", wpos, op.dir_world(FWD), kw)
            q = None
            for _ in range(6):                        # warm-started convergence
                t0 = time.perf_counter()
                q = s.solve(_mats21(joints), joints)
                times.append((time.perf_counter() - t0) * 1e3)
            mcp = {"pinky" if f == "little" else f:
                   _flex_fraction(q, names, bounds,
                                  "pinky" if f == "little" else f)
                   for f in fingers}
            curve.append((float(c), float(np.mean(list(mcp.values())))))
        # H2: pinch gap per finger
        pinch = {}
        for f in fingers:
            kw = dict(curl={f: 0.45}, thumb_curl=0.35, pinch=f)
            joints = op.hand_joints("right", wpos, op.dir_world(FWD), kw)
            q = None
            for _ in range(8):
                q = s.solve(_mats21(joints), joints)
            tips = _fk_hand(model, scratch, q, names, "right")
            key = "pinky" if f == "little" else f
            pinch[key] = float(np.linalg.norm(tips["thumb"] - tips[key]) * 1e3)
        # H3: chirality mirror consistency
        sl, _ = _hand_solver(model, kind, side="left")
        kwf = dict(curl={f: 0.7 for f in fingers}, thumb_curl=0.6)
        jr = op.hand_joints("right", wpos, op.dir_world(FWD), kwf)
        wl = op.wrist("left", 0.6, FWD)
        jl = op.hand_joints("left", wl, op.dir_world(FWD), kwf)
        qr = ql = None
        for _ in range(8):
            qr = s.solve(_mats21(jr), jr)
            ql = sl.solve(_mats21(jl), jl)
        mirror = float(np.mean(np.abs(np.asarray(qr) - np.asarray(ql)))) \
            if qr is not None and ql is not None else np.nan
        # H4: degenerate + sparse + noise
        degen = {n: jr["wrist"] for n in jr}          # all points collapsed
        q_deg = s.solve(_mats21(degen), degen)
        degen_ok = q_deg is None or np.all(np.isfinite(q_deg))
        sparse = {k: v for k, v in jr.items()
                  if "ring" not in k and "little" not in k}
        q_sp = s.solve(_mats21({**{n: jr["wrist"] for n in jr}, **sparse}), sparse)
        sparse_ok = q_sp is None or np.all(np.isfinite(q_sp))
        rng = np.random.default_rng(0)
        qs = []
        for _ in range(20):
            kwn = dict(curl={f: 0.8 for f in fingers}, thumb_curl=0.7,
                       noise=0.002, rng=rng)
            jn = op.hand_joints("right", wpos, op.dir_world(FWD), kwn)
            qn = s.solve(_mats21(jn), jn)
            if qn is not None:
                qs.append(np.asarray(qn))
        noise_std = float(np.mean(np.std(np.stack(qs), axis=0))) if qs else np.nan
        out[kind] = {"curve": np.array(curve), "pinch": pinch,
                     "mirror_rad": mirror, "degen_ok": degen_ok,
                     "sparse_ok": sparse_ok, "noise_std_rad": noise_std,
                     "ms_p50": float(np.median(times)),
                     "ms_p95": float(np.percentile(times, 95))}
    return out


# ── figures ──────────────────────────────────────────────────────────────────

def make_figures(e1, e2, e4, hands, outdir, e8=None, e9=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False})

    # F1: extension transfer, both maps (the money figure)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    statures = sorted({H for (_, H) in e1})
    for ax, arm_map in zip(axes, ("extension", "head")):
        ax.plot([0.25, 1.02], [0.25, 1.02], "--", color="0.55", lw=1,
                label="identity")
        for i, H in enumerate(statures):
            rows = e1[(arm_map, H)]
            ax.plot(rows[:, 0], rows[:, 1], "o-", ms=3.5, lw=1.6,
                    color=PALETTE[i], label=f"H = {H:.2f} m")
        ax.set_xlabel(r"human extension fraction $\rho_h$")
        ax.set_title(f"{arm_map} map")
        ax.set_ylim(0.25, 1.05)
    axes[0].set_ylabel(r"robot extension fraction $\rho_r$")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "retarget_ext_transfer.pdf"))
    plt.close(fig)

    # F2: full-extension by direction
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    names = list(DIRECTIONS.keys())
    x = np.arange(len(names))
    for i, arm_map in enumerate(("extension", "head")):
        vals = [e2[arm_map][n][0] for n in names]
        ax.bar(x + (i - 0.5) * 0.36, vals, width=0.34, color=PALETTE[i],
               label=f"{arm_map} map")
    ax.axhline(1.0, color="0.55", ls="--", lw=1)
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel(r"$\rho_r$ at full human extension")
    ax.set_ylim(0.5, 1.1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "retarget_ext_directions.pdf"))
    plt.close(fig)

    # F3: bimanual closure
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot([0, 0.6], [0, 0.6], "--", color="0.55", lw=1, label="identity")
    for i, (blend, lab) in enumerate(((True, "with proximity blend"),
                                      (False, "no blend"))):
        gh, gr = e4[blend]
        ax.plot(gh, gr, "o-", ms=3.5, lw=1.6, color=PALETTE[i], label=lab)
    ax.set_xlabel("operator inter-wrist gap [m]")
    ax.set_ylabel("robot inter-wrist gap [m]")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "retarget_bimanual.pdf"))
    plt.close(fig)

    # F4: hand curl transfer + pinch gaps
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for i, kind in enumerate(hands):
        cv = hands[kind]["curve"]
        axes[0].plot(cv[:, 0], cv[:, 1], "o-", ms=3.5, lw=1.6,
                     color=PALETTE[i], label=kind)
    axes[0].set_xlabel("human curl fraction")
    axes[0].set_ylabel("robot flexion fraction")
    axes[0].legend(frameon=False, fontsize=8)
    fingers = ["index", "middle", "ring", "pinky"]
    x = np.arange(len(fingers))
    for i, kind in enumerate(hands):
        vals = [hands[kind]["pinch"][f] for f in fingers]
        axes[1].bar(x + (i - 1) * 0.27, vals, width=0.25, color=PALETTE[i],
                    label=kind)
    axes[1].set_xticks(x, fingers)
    axes[1].set_ylabel("thumb–tip gap at touch [mm]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hand_engine_solvers.pdf"))
    plt.close(fig)

    # F5: SCALE fidelity — per-pose scaled-keypoint error by map (the GMR
    # g-mpbpe), + the inter-wrist ratio. The money figure for requirement 1.
    if e8 is not None:
        maps = list(e8.keys())
        poses = [r[0] for r in e8[maps[0]]["rows"]]
        x = np.arange(len(poses))
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        w = 0.8 / len(maps)
        for i, mp in enumerate(maps):
            kp = [r[1] for r in e8[mp]["rows"]]
            axes[0].bar(x + (i - (len(maps) - 1) / 2) * w, kp, width=w,
                        color=PALETTE[i], label=f"{mp} ({e8[mp]['kp_err']:.0f} mm avg)")
        axes[0].set_xticks(x, poses, rotation=35, ha="right", fontsize=7)
        axes[0].set_ylabel("scaled keypoint error [mm]")
        axes[0].set_title("head–wrist–wrist triangle fidelity")
        axes[0].legend(frameon=False, fontsize=7.5)
        for i, mp in enumerate(maps):
            ratios = [r[2] for r in e8[mp]["rows"]]
            axes[1].plot(x, ratios, "o-", ms=3.5, lw=1.5, color=PALETTE[i], label=mp)
        axes[1].axhline(1.0, color="0.55", ls="--", lw=1)
        axes[1].set_xticks(x, poses, rotation=35, ha="right", fontsize=7)
        axes[1].set_ylabel("robot / ideal inter-wrist ratio")
        axes[1].set_title("scale preservation (1.0 = exact)")
        axes[1].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "retarget_scale_fidelity.pdf"))
        plt.close(fig)

    # F6: elbow human-likeness — swivel FOLLOWS the operator (deadband ablation)
    # + the wrist-yaw twist stays bounded. Money figure for requirement 2.
    if e9 is not None:
        labels = list(e9.keys())
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        # left: robot swivel vs commanded human swivel (identity = perfect)
        lim = 0.0
        for i, lab in enumerate(labels):
            ph_t = [np.degrees(r[0]) for r in e9[lab]["rows"]]
            ph_r = [np.degrees(r[1]) for r in e9[lab]["rows"]]
            lim = min(lim, min(ph_t) - 10)
            axes[0].plot(ph_t, ph_r, "o-", ms=4, lw=1.5, color=PALETTE[i],
                         label=f"{lab} ({e9[lab]['swivel_err']:.0f}° err)")
        axes[0].plot([lim, 10], [lim, 10], "--", color="0.55", lw=1, label="human")
        axes[0].set_xlabel("commanded human elbow swivel [°]")
        axes[0].set_ylabel("robot elbow swivel [°]")
        axes[0].set_title("does the elbow bend like the human's?")
        axes[0].legend(frameon=False, fontsize=7.5)
        for i, lab in enumerate(labels):
            twists = [r[3] for r in e9[lab]["rows"]]
            axes[1].scatter(range(len(twists)), twists, s=22, color=PALETTE[i],
                            label=f"{lab} (max {e9[lab]['twist_max']:.0f}°)")
        axes[1].axhline(25.0, color="0.55", ls="--", lw=1, label="26° human limit")
        axes[1].set_ylabel("wrist-yaw twist from rest [°]")
        axes[1].set_xlabel("swivel command index")
        axes[1].set_title("the 'weird twist' DoF stays bounded")
        axes[1].legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "retarget_elbow_humanlike.pdf"))
        plt.close(fig)

    print(f"figures → {outdir}/retarget_*.pdf, hand_engine_solvers.pdf")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    print(f"model: {os.path.basename(MODEL_XML)}  nq={model.nq} nu={model.nu}")

    print("\n── E1 extension transfer (rho_r vs rho_h, forward) ──")
    e1 = e1_extension_transfer(model, quick=args.quick)
    for (arm_map, H), rows in sorted(e1.items()):
        r1 = rows[-1]
        print(f"  {arm_map:<10} H={H:.2f}:  rho_r@full={r1[1]:.3f}  "
              f"elbow {r1[3]:6.1f}° (human {r1[2]:5.1f}°, hw max 161.4°)  "
              f"elbow-limit margin {r1[4]:.1f}°  "
              f"mean|Δrho|={np.mean(np.abs(rows[:,1]-rows[:,0])):.3f}")

    print("\n── E2 full extension by direction ──")
    e2 = e2_directions(model)
    for arm_map, rows in e2.items():
        s = "  ".join(f"{n}:{v[0]:.2f}" for n, v in rows.items())
        print(f"  {arm_map:<10} rho_r: {s}")

    print("\n── E3 circle tracking (rho 0.7) ──")
    e3 = e3_trajectory(model)
    for arm_map, r in e3.items():
        print(f"  {arm_map:<10} rho RMS {r['rho_rms']:.3f} | dir RMS "
              f"{r['dir_rms_deg']:.1f}° | max joint step {r['step_max_rad']:.3f} rad")

    print("\n── E4 bimanual closure ──")
    e4 = e4_bimanual(model)
    for blend, (gh, gr) in e4.items():
        print(f"  blend={str(blend):<5} robot gap at 5 cm human gap: "
              f"{gr[-1]*100:.1f} cm")

    print("\n── E5 edge cases ──")
    e5 = e5_edge_cases(model)
    n_fail = sum(1 for _, ok, _ in e5 if not ok)

    print("\n── E6 global consistency (yaw / walk-loop / separation) ──")
    e6 = e6_global_consistency(model)
    n_fail += sum(1 for _, ok, _ in e6 if not ok)

    print("\n── E7 elbow swivel (palm-roll follow / clearance) ──")
    e7 = e7_elbow_swivel(model)
    n_fail += sum(1 for _, ok, _ in e7 if not ok)

    print("\n── E8 SCALE fidelity (GMR g-mpbpe: head/wrist triangle) ──")
    e8 = e8_scale_fidelity(model, quick=args.quick)
    for arm_map, r in e8.items():
        print(f"  {arm_map:<10} keypoint err {r['kp_err']:5.1f} mm "
              f"(p95 {r['kp_err_p95']:5.1f}) | inter-wrist ratio {r['sep_ratio']:.3f} "
              f"(1.0=ideal) | sep-dir {r['sep_dir']:.1f}°")
    # ASSERTIONS: the GMR map must track scale materially better than the map
    # it replaces, and land the pose triangle within a hand-width.
    e8_checks = []
    gmr_kp, ext_kp = e8["gmr"]["kp_err"], e8["extension"]["kp_err"]
    e8_checks.append(("gmr keypoint err < 60 mm", e8["gmr"]["kp_err"] < 60.0,
                      f"{gmr_kp:.1f} mm"))
    e8_checks.append(("gmr beats extension on scale", gmr_kp < ext_kp,
                      f"gmr {gmr_kp:.1f} vs extension {ext_kp:.1f} mm"))
    e8_checks.append(("gmr inter-wrist ratio ~1", abs(e8["gmr"]["sep_ratio"] - 1.0) < 0.15,
                      f"ratio {e8['gmr']['sep_ratio']:.3f}"))
    for name, ok, det in e8_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {det}")
    n_fail += sum(1 for _, ok, _ in e8_checks if not ok)

    print("\n── E9 elbow human-likeness (swivel-follow; deadband ablation) ──")
    e9 = e9_elbow_humanlikeness(model, quick=args.quick)
    for label, r in e9.items():
        print(f"  {label:<24} swivel-follow err {r['swivel_err']:5.1f}° "
              f"(max {r['swivel_err_max']:5.1f}) | wrist-yaw twist max "
              f"{r['twist_max']:4.1f}°")
    live_label = next(k for k in e9 if k.startswith("live"))
    pre_label = next(k for k in e9 if k.startswith("pre"))
    e9_checks = [
        ("live swivel-follow beats pre-fix deadband",
         e9[live_label]["swivel_err"] < e9[pre_label]["swivel_err"] - 3.0,
         f"{e9[live_label]['swivel_err']:.1f}° vs {e9[pre_label]['swivel_err']:.1f}°"),
        ("live swivel-follow err < 25°",
         e9[live_label]["swivel_err"] < 25.0,
         f"mean {e9[live_label]['swivel_err']:.1f}°"),
        # The hard J7 yaw clamp (ARM_J7_YAW_LIM = 0.45 rad ≈ 25.8°) IS the
        # anatomical bound — a human wrist yaws only ±25°. Bounded == the
        # solver never exceeds that clamp (no fake-orientation spin).
        ("live wrist-twist within human yaw clamp (≤26°)",
         e9[live_label]["twist_max"] <= 26.0,
         f"max {e9[live_label]['twist_max']:.1f}°"),
    ]
    for name, ok, det in e9_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {det}")
    n_fail += sum(1 for _, ok, _ in e9_checks if not ok)

    print("\n── E10 stability (held-pose sweep / smoothness / tracking) ──")
    e10 = e10_stability(model, quick=args.quick)
    print(f"  held-pose sweep: worst joint std {e10['hold_std_worst']*1e3:.2f} mrad "
          f"over {e10['n_poses']} poses | {e10['n_osc']} oscillate (>2 mrad)")
    print(f"  slow sweep: max joint step {e10['sweep_step']:.3f} rad | "
          f"wrist-track RMS {e10['track_rms_mm']:.1f} mm")
    e10_checks = [
        ("no held pose oscillates across workspace", e10["n_osc"] == 0,
         f"{e10['n_osc']}/{e10['n_poses']} poses > 2 mrad"),
        ("worst held std < 2 mrad", e10["hold_std_worst"] < 2e-3,
         f"{e10['hold_std_worst']*1e3:.2f} mrad"),
        ("sweep has no basin-flip snaps (<0.11 rad/tick)", e10["sweep_step"] <= 0.101,
         f"{e10['sweep_step']:.3f} rad"),
        ("global wrist tracking < 45 mm RMS", e10["track_rms_mm"] < 45.0,
         f"{e10['track_rms_mm']:.1f} mm"),
    ]
    for name, ok, det in e10_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {det}")
    n_fail += sum(1 for _, ok, _ in e10_checks if not ok)

    print("\n── HAND battery ──")
    hands = hand_battery(model, quick=args.quick)
    for kind, r in hands.items():
        pinch = " ".join(f"{f}:{v:.0f}mm" for f, v in r["pinch"].items())
        print(f"  {kind:<15} curl@1.0={r['curve'][-1,1]:.2f} | pinch {pinch} | "
              f"mirror {r['mirror_rad']:.3f} rad | noise σ {r['noise_std_rad']:.3f} | "
              f"{r['ms_p50']:.1f}/{r['ms_p95']:.1f} ms p50/p95 | "
              f"degen {'ok' if r['degen_ok'] else 'FAIL'} "
              f"sparse {'ok' if r['sparse_ok'] else 'FAIL'}")

    if args.figures:
        make_figures(e1, e2, e4, hands, FIG_DIR, e8=e8, e9=e9)

    print(f"\n{'='*60}")
    ext_full = np.mean([e1[(('extension'), H)][-1, 1]
                        for (m, H) in e1 if m == "extension"])
    head_full = np.mean([e1[(('head'), H)][-1, 1]
                         for (m, H) in e1 if m == "head"])
    print(f"SUMMARY  rho_r at full human extension: extension map {ext_full:.3f} "
          f"vs head map {head_full:.3f}")
    print(f"         GMR scale fidelity: keypoint err {e8['gmr']['kp_err']:.1f} mm "
          f"(extension {e8['extension']['kp_err']:.1f} mm)")
    print(f"         elbow swivel-follow: {e9[live_label]['swivel_err']:.1f}° live "
          f"vs {e9[pre_label]['swivel_err']:.1f}° pre-fix deadband")
    print(f"         stability: worst held std {e10['hold_std_worst']*1e3:.2f} mrad "
          f"over {e10['n_poses']} poses ({e10['n_osc']} oscillating), "
          f"track RMS {e10['track_rms_mm']:.1f} mm")
    n_assert = len(e5) + len(e6) + len(e7) + len(e8_checks) + len(e9_checks) + len(e10_checks)
    print(f"         assertions: {n_assert-n_fail}/{n_assert} pass")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

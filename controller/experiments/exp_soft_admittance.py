#!/usr/bin/env python3
"""exp_soft_admittance.py — rigorous sim verification of FLOATING soft mode.

Emulates the REAL robot loop faithfully (no robot, no hands):
  · the 14 arm actuators are NEUTRALIZED and driven instead by the DaMiao MIT law
    τ = kp(q*−q) + kd(v*−q̇) + τ_ff at the DEPLOYED per-joint compliant gains
    (control_param.json arm.gains) + the DEPLOYED gravity FF (nominal MJCF,
    hands=NONE, sysid trims) — applied via qfrc_applied at physics rate;
  · joint stiction is emulated with per-dof frictionloss;
  · the PLANT is deliberately WRONG vs the controller's model — random mass/CoM
    deltas on every arm link, AUTO-SCALED so the plant-vs-model gravity error RMS
    lands ON the robot's measured post-trim sysid residual (0.4–1.0 N·m; the
    operator's directive: "the real robot system ID is not perfect … make sure
    floating still works") — plus τ-feedback noise (σ 0.15 N·m), per-joint
    sensor bias (±0.3), and the plant hands stripped (none are mounted);
  · SoftArmAdmittance runs at the 50 Hz control tick from the emulated measured
    signals, exactly as server/plant.py hosts it.

Scenarios (each judged PASS/FAIL):
  S1 HOLD      60 s, zero external force → the arm must not drift.
  S2 DRAG      15 N at the left wrist — backward at the hanging pose, sideways +
               a 10 N guided lift at a bent working pose → the palm must float,
               then settle ballistically (0.5 s) and PARK (no spring-back).
  S3 FLOATING  the same backward drag with admittance OFF (frozen-target soft):
               the admittance palm travel must be ≥ 2× the rubber-band travel.
  S4 LIGHT     3 N (sub-deadband-scale) for 5 s → bounded elastic yield only,
               and no drift after release.
  S5 MISMATCH  hold + a guided lift repeated across 3 mismatch seeds.

    python3 experiments/exp_soft_admittance.py            # full suite
    python3 experiments/exp_soft_admittance.py --quick    # shorter holds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(HERE))
SCENE = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
NOMINAL = os.path.join(REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")

from mabel.arm_control import ArmGravityCompensator, HandsConfig, SoftArmAdmittance
from mabel.control_params import arm_gains, arm_soft_float

FW = [f"{s}_j{k}" for s in ("l", "r") for k in range(1, 8)]
MJ = {f"{s}_j{k}": f"{'Left' if s == 'l' else 'Right'} Arm {k}"
      for s in ("l", "r") for k in range(1, 8)}
CTRL_DT = 0.02                     # 50 Hz admittance/host tick
FF_DT = 0.1                        # gravity-FF refresh cadence (gateway ~4-10 Hz)
DRAG_BODY = "l_wrist"
# emulated stiction (N·m) per joint class — the real 1.1-1.7 N·m band
FRICTION = {"j1": 1.2, "j2": 0.9, "j3": 0.6, "j4": 0.6, "j5": 0.3, "j6": 0.3, "j7": 0.25}


def _cls(fw):
    return fw.split("_", 1)[1]




class EmulatedRealArm:
    """MuJoCo plant + MIT-law arm emulation + noisy 'measured' signals."""

    def __init__(self, seed=0, mismatch=True, delay_ticks=2, tau_quant=0.1):
        # delay_ticks: FEEDBACK TRANSPORT DELAY in control ticks (2 = 40 ms —
        # CAN + HAL cache + DDS + gateway ingest). The first harness had ZERO
        # feedback delay and certified a loop the real arm found ERRATIC
        # (2026-07-30 live test) — the delay is not optional realism, it is the
        # destabilizer. tau_quant: DaMiao τ telemetry quantization (N·m).
        self.delay_ticks = int(delay_ticks)
        self.tau_quant = float(tau_quant)
        self._fb_hist = []
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._strip_hands()                       # the REAL robot has no hands (2026-07)
        self._perturb(mismatch)
        # neutralize the 14 arm actuators: the MIT emulation is the only driver
        self.aid = {}
        for fw, mj in MJ.items():
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mj)
            self.aid[fw] = a
            self.model.actuator_gainprm[a][:] = 0.0
            self.model.actuator_biasprm[a][:] = 0.0
        self.jadr = {fw: int(self.model.joint(mj).qposadr[0]) for fw, mj in MJ.items()}
        self.dof = {fw: int(self.model.joint(mj).dofadr[0]) for fw, mj in MJ.items()}
        for fw, d in self.dof.items():
            self.model.dof_frictionloss[d] = FRICTION[_cls(fw)]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(600):                      # settle on the servos + friction
            mujoco.mj_step(self.model, self.data)

        # DEPLOYED gains + the NOMINAL controller-side gravity model (trimmed)
        g = arm_gains()
        self.kp = {fw: float(g.get(fw, {}).get("kp", 15.0)) for fw in FW}
        self.kd = {fw: float(g.get(fw, {}).get("kd", 1.5)) for fw in FW}
        # mount_tilt_deg=-10: the deployed compensator re-tilts the MJCF's drawn
        # -10 deg shoulder mounts to 0 because the REAL metal measures 0. In this
        # harness the PLANT (drawn MJCF) is the metal, so the comp must keep the
        # drawn tilt — otherwise every run carries a constant ~2.4 N.m fake
        # "model error" that drowned the calibrated corruption below.
        self.comp = ArmGravityCompensator(NOMINAL, {}, {}, hands=HandsConfig.NONE,
                                          mount_tilt_deg=-10.0,
                                          log=lambda *a, **k: None)
        assert self.comp.available
        # calibrate the corruption AFTER the comp exists (it measures against it)
        self._calibrate_mismatch()
        # per-joint τ-sensor bias — part of the "imperfect real robot"
        self.tau_bias = {fw: float(self.rng.uniform(-0.3, 0.3)) for fw in FW}

        self.targets = {fw: float(self.data.qpos[self.jadr[fw]]) for fw in FW}
        self._vff = {fw: 0.0 for fw in FW}
        self._prev_tgt = dict(self.targets)
        self._tau_m = {fw: 0.0 for fw in FW}      # last applied MIT torque
        self._ff = {}
        self._ff_t = -1e9
        self.t = 0.0

    def _perturb(self, on):
        """Make the PLANT differ from the nominal model like a bad sysid would:
        arm-chain link masses ×(1±15%), CoMs shifted ≤2 cm along each axis."""
        if not on:
            self._pert = None
            return
        # Draw UNIT deltas once; _calibrate_mismatch() scales them so the model
        # error RMS matches the robot's measured sysid residual band.
        m = self.model
        bids = []
        for mj in MJ.values():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mj)
            bids.append(int(m.jnt_bodyid[jid]))
        self._pert = {
            "bids": bids,
            "mass0": {b: float(m.body_mass[b]) for b in bids},
            "ipos0": {b: m.body_ipos[b].copy() for b in bids},
            "dm": {b: float(self.rng.uniform(-0.05, 0.05)) for b in bids},
            "dc": {b: self.rng.uniform(-0.008, 0.008, 3) for b in bids},
        }
        self._apply_perturb(1.0)

    def _apply_perturb(self, s):
        m, p = self.model, self._pert
        for b in p["bids"]:
            m.body_mass[b] = p["mass0"][b] * (1.0 + s * p["dm"][b])
            m.body_ipos[b] = p["ipos0"][b] + s * p["dc"][b]
        mujoco.mj_setConst(m, mujoco.MjData(m))

    def _strip_hands(self):
        """Zero the ORCA hand subtree mass on the PLANT — the physical robot has
        no hands mounted, and the controller model (HandsConfig.NONE) strips
        them; a plant that keeps the 1.12 kg/side was a constant ~2.4 N·m fake
        'model error' that drowned the calibrated corruption."""
        m = self.model
        roots = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                 for n in ("left_tower", "right_tower")]
        drop = set(b for b in roots if b >= 0)
        changed = True
        while changed:
            changed = False
            for b in range(m.nbody):
                if int(m.body_parentid[b]) in drop and b not in drop:
                    drop.add(b)
                    changed = True
        for b in drop:
            m.body_mass[b] = 0.0
            m.body_inertia[b][:] = 0.0
        mujoco.mj_setConst(m, mujoco.MjData(m))

    def _grav_err_rms(self, n_poses=6):
        """RMS plant-vs-model gravity error over a fixed pose grid (all joints)."""
        m = self.model
        s = mujoco.MjData(m)
        rng = np.random.default_rng(99)          # fixed grid, not per-seed
        errs = []
        for _ in range(n_poses):
            s.qpos[:] = self.data.qpos
            pose = {}
            for fw in FW:
                lo, hi = m.joint(MJ[fw]).range
                v = float(rng.uniform(lo + 0.1, hi - 0.1))
                s.qpos[self.jadr[fw]] = v
                pose[fw] = v
            s.qvel[:] = 0.0
            mujoco.mj_forward(m, s)
            nom = self.comp.tff(pose, frame="model") or {}
            errs += [float(s.qfrc_bias[self.dof[fw]]) - float(nom.get(fw, 0.0))
                     for fw in FW]
        return float(np.sqrt(np.mean(np.square(errs))))

    def _calibrate_mismatch(self):
        """Scale the drawn mass/CoM deltas so the model error RMS lands ON the
        robot's measured post-trim sysid residual (0.4–1.0 N·m; target 0.7).
        The corruption is then exactly as bad as the real robot's — neither a
        strawman (first cut: 2.5–3.3 N·m hid every drag) nor a gift."""
        if self._pert is None:
            self.mismatch_rms = 0.0
            return
        r1 = self._grav_err_rms()
        s = min(3.0, max(0.1, 0.7 / max(r1, 1e-6)))
        self._apply_perturb(s)
        self.mismatch_rms = self._grav_err_rms()

    def pose_arm(self, targets_fw, seconds=3.0):
        """Drive the emulated MIT loop to a working pose (targets in rad) —
        like the operator posing the arm in Stiff before flipping to Soft."""
        for fw, v in targets_fw.items():
            self.targets[fw] = float(v)
        for _ in range(int(seconds / CTRL_DT)):
            self.tick()

    def gravity_error(self):
        """Per-joint |true - nominal| gravity hold torque at the CURRENT pose —
        the honest size of the model error this run must float through."""
        m, d = self.model, self.data
        s = mujoco.MjData(m)
        s.qpos[:] = d.qpos
        s.qvel[:] = 0.0
        mujoco.mj_forward(m, s)
        pose = {fw: float(d.qpos[self.jadr[fw]]) for fw in FW}
        nom = self.comp.tff(pose, frame="model") or {}
        return {fw: abs(float(s.qfrc_bias[self.dof[fw]]) - float(nom.get(fw, 0.0)))
                for fw in FW}

    # ── one control tick (CTRL_DT), stepping physics inside ──────────────────
    def tick(self, xfrc=None):
        m, d = self.model, self.data
        # gravity FF at the measured pose, refreshed at the gateway cadence
        if self.t - self._ff_t >= FF_DT:
            pose = {fw: float(d.qpos[self.jadr[fw]]) for fw in FW}
            self._ff = self.comp.tff(pose, frame="model") or {}
            self._ff_t = self.t
        # velocity FF = d/dt of the streamed target (EMA), like ros_plant
        for fw in FW:
            raw = (self.targets[fw] - self._prev_tgt[fw]) / CTRL_DT
            self._vff[fw] = 0.4 * raw + 0.6 * self._vff[fw]
        self._prev_tgt = dict(self.targets)

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, DRAG_BODY)
        steps = int(round(CTRL_DT / m.opt.timestep))
        for _ in range(steps):
            d.xfrc_applied[:] = 0.0
            if xfrc is not None:
                d.xfrc_applied[bid, :3] = xfrc
            for fw in FW:
                dof = self.dof[fw]
                q = d.qpos[self.jadr[fw]]
                qd = d.qvel[dof]
                tau = (self.kp[fw] * (self.targets[fw] - q)
                       + self.kd[fw] * (self._vff[fw] - qd)
                       + float(self._ff.get(fw, 0.0)))
                self._tau_m[fw] = tau
                d.qfrc_applied[dof] = tau
            mujoco.mj_step(m, d)
        self.t += CTRL_DT

    # ── the noisy, DELAYED 'measured' wire the gateway would see ─────────────
    def measured(self):
        d = self.data
        qq = self.tau_quant
        snap = (
            {fw: float(d.qpos[self.jadr[fw]] + self.rng.normal(0, 5e-4)) for fw in FW},
            {fw: float(d.qvel[self.dof[fw]] + self.rng.normal(0, 0.01)) for fw in FW},
            {fw: float(round((self._tau_m[fw] + self.tau_bias[fw]
                              + self.rng.normal(0, 0.15)) / qq) * qq) if qq > 0
             else float(self._tau_m[fw] + self.tau_bias[fw]
                        + self.rng.normal(0, 0.15)) for fw in FW},
        )
        self._fb_hist.append(snap)
        if len(self._fb_hist) > self.delay_ticks + 1:
            self._fb_hist.pop(0)
        return self._fb_hist[0]

    def palm(self):
        return self.data.body("left_palm").xpos.copy()

    def qvec(self):
        return np.array([self.data.qpos[self.jadr[fw]] for fw in FW])


def build_admittance(arm):
    limits = {fw: tuple(arm.model.joint(MJ[fw]).range) for fw in FW}
    adm = SoftArmAdmittance(arm.comp, FW, params=arm_soft_float())
    return adm, limits


def run_hold(arm, adm, limits, seconds, engage=True):
    if engage:
        q, qd, tau = arm.measured()
        arm.tick()                                # one tick so τ_m is real
        q, qd, tau = arm.measured()
        adm.engage(q, arm.t)
    q0 = arm.qvec()
    p0 = arm.palm()
    for _ in range(int(seconds / CTRL_DT)):
        arm.tick()
        q, qd, tau = arm.measured()
        tgt = adm.update(q, tau, CTRL_DT, arm.t, limits=limits, qd_meas=qd)
        arm.targets.update(tgt)
    drift_j = float(np.max(np.abs(arm.qvec() - q0)))
    drift_p = float(np.linalg.norm(arm.palm() - p0))
    return np.degrees(drift_j), drift_p * 100.0   # deg, cm


def run_drag(arm, adm, limits, force, seconds, coast=2.0):
    p0 = arm.palm()
    for _ in range(int(seconds / CTRL_DT)):
        arm.tick(xfrc=force)
        q, qd, tau = arm.measured()
        tgt = adm.update(q, tau, CTRL_DT, arm.t, limits=limits, qd_meas=qd)
        arm.targets.update(tgt)
    p_rel = arm.palm()
    # 0.5 s ballistic settle: at force-off the arm is still MOVING (a 15 N swing
    # carries momentum) — "where the operator left it" is where it comes to
    # rest, not the instant the hand let go. Springback = drift AFTER that.
    for _ in range(int(0.5 / CTRL_DT)):
        arm.tick()
        q, qd, tau = arm.measured()
        tgt = adm.update(q, tau, CTRL_DT, arm.t, limits=limits, qd_meas=qd)
        arm.targets.update(tgt)
    p_settled = arm.palm()
    for _ in range(int(coast / CTRL_DT)):         # park watch
        arm.tick()
        q, qd, tau = arm.measured()
        tgt = adm.update(q, tau, CTRL_DT, arm.t, limits=limits, qd_meas=qd)
        arm.targets.update(tgt)
    p_end = arm.palm()
    travel = float(np.linalg.norm(p_rel - p0)) * 100.0
    springback = float(np.linalg.norm(p_end - p_settled)) * 100.0
    return travel, springback                     # cm, cm


def run_drag_frozen(arm, seconds, force):
    """Baseline = today's soft: targets FROZEN, only kp·error yields."""
    p0 = arm.palm()
    for _ in range(int(seconds / CTRL_DT)):
        arm.tick(xfrc=force)
    trav = float(np.linalg.norm(arm.palm() - p0)) * 100.0
    for _ in range(int(2.0 / CTRL_DT)):
        arm.tick()
    back = float(np.linalg.norm(arm.palm() - p0)) * 100.0
    return trav, back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    hold_s = 20.0 if args.quick else 60.0
    results, ok_all = {}, True

    def judge(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        results[name] = {"pass": bool(ok), **detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("── S1 HOLD (mismatched plant, zero force) ─────────────────────────")
    arm = EmulatedRealArm(seed=1)
    print(f"  plant-vs-model gravity error rms (pose grid): {arm.mismatch_rms:.2f} N·m "
          "— calibrated onto the real sysid residual band (0.4–1.0)")
    adm, limits = build_admittance(arm)
    dj, dp = run_hold(arm, adm, limits, hold_s)
    judge("S1_hold", dj < 2.0 and dp < 2.0,
          {"drift_deg": round(dj, 3), "palm_drift_cm": round(dp, 2), "s": hold_s})

    print("── S2 DRAG: back at hanging, side/up at a bent working pose ───────")
    def bend(a):
        # a real session pose: elbow bent, shoulder slightly raised (a hanging
        # arm is torque-degenerate for vertical drags; a bent one clamps
        # backward drags into the elbow's range stop — each drag runs at the
        # pose where it is physically meaningful)
        tgt = {}
        for fw in ("l_j2", "l_j4", "r_j2", "r_j4"):
            lo, hi = a.model.joint(MJ[fw]).range
            frac = 0.55 if fw.endswith("j4") else 0.3
            tgt[fw] = lo + frac * (hi - lo) if abs(lo) > abs(hi) else hi - frac * (hi - lo)
        a.pose_arm(tgt, seconds=4.0)
    # 15 N ~ a firm one-hand drag. down(-z) on a HANGING arm is geometrically
    # degenerate (force along the links, near-zero joint torque) — up(+z) is the
    # honest vertical test: it unloads gravity and the arm must float UP.
    dirs = {"back(-x)": np.array([-15.0, 0, 0]), "side(+y)": np.array([0, 15.0, 0]),
            "up(+z)": np.array([0, 0, 10.0])}    # vertical = a guided lift, not a hoist
    for name, f in dirs.items():
        if name.startswith("side"):              # bend once, before the bent-pose drags
            bend(arm)
            run_hold(arm, adm, limits, 1.0)      # re-engage at the bent pose
        trav, back = run_drag(arm, adm, limits, f, seconds=2.5 if not name.startswith("up") else 1.5)
        # springback tolerance: the target may trail the hand by up to err_max,
        # so some settle-back on release is physical; it must stay a FRACTION of
        # the travel (vertical gets more: gravity pulls the trailing gap closed).
        frac = 0.5 if name.startswith("up") else 0.35
        judge(f"S2_drag_{name}", trav > 8.0 and back <= frac * trav + 3.0,
              {"travel_cm": round(trav, 1), "springback_cm": round(back, 2)})

    print("── S3 FLOATING vs frozen-target baseline ──────────────────────────")
    arm_b = EmulatedRealArm(seed=1)                # identical mismatched plant
    trav_b, back_b = run_drag_frozen(arm_b, 3.0, dirs["back(-x)"])
    trav_f = results["S2_drag_back(-x)"]["travel_cm"]
    judge("S3_floating_gain", trav_f >= 2.0 * max(trav_b, 1e-3),
          {"admittance_cm": trav_f, "rubberband_cm": round(trav_b, 1),
           "rubberband_springback_cm": round(back_b, 1)})

    print("── S4 LIGHT touch 3 N (sub-deadband scale) ────────────────────────")
    arm4 = EmulatedRealArm(seed=1)
    adm4, lim4 = build_admittance(arm4)
    run_hold(arm4, adm4, lim4, 2.0)
    trav, back = run_drag(arm4, adm4, lim4, np.array([-3.0, 0, 0]), seconds=5.0)
    dj, dp = run_hold(arm4, adm4, lim4, 10.0, engage=False)
    judge("S4_light_touch", trav < 12.0 and dj < 2.0,
          {"travel_cm": round(trav, 1), "post_drift_deg": round(dj, 2)})

    print("── S5 MISMATCH sweep (3 seeds: mass ±7%, CoM ±1 cm, re-fit trims) ─")
    for seed in (2, 3, 4):
        arm5 = EmulatedRealArm(seed=seed)
        adm5, lim5 = build_admittance(arm5)
        dj, dp = run_hold(arm5, adm5, lim5, 20.0 if args.quick else 30.0)
        bend(arm5)
        run_hold(arm5, adm5, lim5, 1.0)
        trav, back = run_drag(arm5, adm5, lim5, dirs["up(+z)"], seconds=1.5)
        judge(f"S5_seed{seed}", dj < 2.0 and trav > 8.0 and back <= 0.5 * trav + 3.0,
              {"mismatch_rms": round(arm5.mismatch_rms, 2), "drift_deg": round(dj, 2),
               "travel_cm": round(trav, 1), "springback_cm": round(back, 2)})

    out = os.path.join(HERE, "results", "soft_admittance_verify.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"when": time.strftime("%Y-%m-%d %H:%M:%S"),
               "quick": args.quick, "results": results}, open(out, "w"), indent=1)
    print(f"\n{'ALL PASS' if ok_all else 'FAILURES PRESENT'} — wrote {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

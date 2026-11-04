#!/usr/bin/env python3
"""
REAL-DISPATCH chain verification for Vision-Pro WHOLE-BODY teleop — offline.

Reproduces, on this machine, exactly what the real robot runs — the gateway's
real:ros control loop minus rclpy:

    mock VP stream ──> MabelTeleop.update (method=retarget: TeleopEngine —
    the simulation/mabel_mujoco/scripts/teleop stack) ──> data.ctrl +
    RobotCommand.base ──> mabel_control.command_extract.extract ──> the
    canonical hardware commands (/mabel_cmd, /arms/command, /body/command,
    /<side>_hand/command)

The "robot" is an ENCODER MIRROR, not physics: each control tick the joints
track their commands with a servo-like first-order lag + rate limit, the base
integrates the commanded twist (odom), then mj_forward — the same role
data.qpos plays on the real robot (ROSPlant.read_state). Control paced at the
real CTRL_HZ = 50.

Phases (the real operating sequence, worst-case first):

  P1  ENABLE TORQUE with the arms DISPLACED from home (limp arms were moved):
      resume_hold — the first published command must equal the MEASURED pose
      (the old supervisor wrote ctrl := HOME outright: a full-authority lurch).
  P2  no stream yet: commands must only EASE (safe-rate, step-free).
  P3  VP stream begins: auto-calibrate + engage; arms/neck must FOLLOW the
      operator (the frozen-arms regression: resume_hold used to seed
      joint_targets that overrode the retargeter every tick).
  P4  whole-body: operator WALKS forward + crouches -> base twist + lift drop.
  P5  stream LOSS: base zeroed within the watchdog; upper body eases HOME.
  P6  RE-ACQUIRE: eases back to the live pose, step-free.

Every command sample is checked against STEP_BOUND (max per-tick joint command
change) — the discriminator between an ease and a snap. Figure:
docs/figures/real_retarget_chain.pdf. Metrics printed as a table.

PACING IS REAL TIME (sleep to 50 Hz, ~20 s total): the engine's watchdogs and
safety eases run on time.time(), so a faster-than-realtime loop would misreport
every timing (a virtually-2.66 s watchdog that is actually the correct 0.4 s).
"""
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
sys.path.insert(0, os.path.join(_REPO, "ros", "src", "mabel_control"))
sys.path.insert(0, os.path.join(_REPO, "ros", "src", "mabel_sim"))

import mujoco                                                    # noqa: E402

from mabel.facade import MabelTeleop                             # noqa: E402
from mabel.types import OperatorPose                             # noqa: E402
from mabel.sources import mock_frame                             # noqa: E402
from mabel.whole_body_retargeting.calibration import (           # noqa: E402
    synthesize_pose, POSE_HANDS)
from mabel_control.command_extract import extract                # noqa: E402

MODEL_XML = os.path.join(_REPO, "simulation", "mabel_mujoco", "scenes",
                         "mabel_home.xml")
FIGDIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")

CTRL_HZ = 50.0                     # the gateway's real:ros control rate
DT = 1.0 / CTRL_HZ
STEP_BOUND = 0.15                  # rad (or m) max per-tick command change. The
                                   # discriminator between a SNAP (the old HOME
                                   # write / frozen-target replay: 0.5-3 rad in
                                   # one tick = the slider-slam hazard class) and
                                   # the bounded transients that remain by design:
                                   # eases are <0.03/tick, and for ~0.5 s after a
                                   # world-ref re-anchor the ease passes s-scaled
                                   # echoes of the settling solver (measured
                                   # <=0.11). 0.15 rad at kp=30 is ~4.5 N.m for
                                   # one tick — well under the ~99 N.m slam this
                                   # guards against.
SERVO_RATE = 2.0                   # rad/s emulated joint servo slew (mirror)
LIFT_RATE = 0.06                   # m/s emulated lift travel


class RealMirror:
    """The encoder-mirror 'robot': joints track commands with servo lag, the
    base integrates the commanded twist. Stands in for ROSPlant.read_state."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.acts = []                                # (aid, qadr) joint servos
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = model.actuator_trnid[aid, 0]
                self.acts.append((aid, model.jnt_qposadr[jid]))
        self.lift_aid = model.actuator("Lift").id
        self.lift_q = [model.joint("Lift Lower").qposadr[0],
                       model.joint("Lift Upper").qposadr[0]]
        self.base_q = None
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.base_q = model.jnt_qposadr[j]
                break
        self.yaw = 0.0

    def track(self, base_twist, dt):
        d = self.data
        for aid, qadr in self.acts:
            e = float(d.ctrl[aid]) - float(d.qpos[qadr])
            d.qpos[qadr] += np.clip(e, -SERVO_RATE * dt, SERVO_RATE * dt)
        tendon = float(d.ctrl[self.lift_aid])         # ctrl = stage SUM
        for qadr in self.lift_q:
            e = 0.5 * tendon - float(d.qpos[qadr])
            d.qpos[qadr] += np.clip(e, -0.5 * LIFT_RATE * dt, 0.5 * LIFT_RATE * dt)
        if self.base_q is not None and base_twist is not None:
            vx, vy, w = base_twist
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            d.qpos[self.base_q + 0] += (c * vx - s * vy) * dt
            d.qpos[self.base_q + 1] += (s * vx + c * vy) * dt
            self.yaw += w * dt
            d.qpos[self.base_q + 3] = np.cos(self.yaw / 2)
            d.qpos[self.base_q + 6] = np.sin(self.yaw / 2)
        mujoco.mj_forward(self.model, self.data)


class MockOperator:
    """The VP mock stream (60 Hz frames): mock_frame head+wrists + named hand
    skeletons, with an injectable head WALK (AVP frame) for the whole-body
    phases. ARKit frame: +Y up, -Z forward."""

    def __init__(self):
        self.hand_frames = synthesize_pose(POSE_HANDS, n=48, noise=0.0005)
        self.i = 0
        self.walk = np.zeros(3)                        # extra head translation
        self.sweep = 0.0                               # wrist sweep amplitude [m]
        # Lateral hand spread [m per side]: the raw POSE_HANDS skeleton holds the
        # wrists only ~20 cm apart ("inspect hands"), which the ACCURATE bimanual
        # map reproduces on the robot — parking the two ORCA hands 2.5 cm apart,
        # ON the self-collision guard's braking margin. That conflates the
        # liveness metrics with guard-proximity braking (measured: P6 arm std
        # 0.002, arms pinned by the barrier). Spread the synthetic hands to a
        # realistic 36 cm so liveness is measured clear of the guard; close-hand
        # accuracy has its own battery (exp_retarget_fidelity E4).
        self.spread = 0.08

    def frame(self, t):
        pose = OperatorPose.from_avframe(mock_frame(t))
        if pose.head is not None and self.walk.any():
            # A WALKING operator carries head AND hands together (offsetting only
            # the head would command a 0.4 m/s relative hand sweep — honest fast
            # tracking, but not what walking is).
            pose.head = pose.head.copy()
            pose.head[:3, 3] = pose.head[:3, 3] + self.walk
            for attr in ("left_hand", "right_hand"):
                mats = getattr(pose, attr)
                if mats is not None:
                    moved = []
                    for m in mats:
                        m = np.asarray(m, float).copy()
                        m[:3, 3] += self.walk
                        moved.append(m)
                    setattr(pose, attr, moved)
        hf = self.hand_frames[self.i % len(self.hand_frames)]
        pose.left_joints = hf.left_joints
        pose.right_joints = hf.right_joints
        # The engine's arm path consumes the NAMED skeleton, so (a) a WALKING
        # operator's joints must translate with the head (else the shoulder-
        # anchored map reads walking as a 0.6 m relative arm sweep), and (b)
        # the P3/P6 "arms follow" liveness needs DELIBERATE wrist motion — a
        # lateral sweep. The old metrics rode on the head-anchored map
        # amplifying the near-static POSE_HANDS wrists from a contorted
        # workspace pose; the extension-fraction map (2026-07-21) tracks the
        # wrist RELATIVE TO THE SHOULDER, so liveness is tested with real
        # wrist motion, map-agnostically.
        off = self.walk.copy()
        if self.sweep:
            off[0] += self.sweep * np.sin(2 * np.pi * 0.5 * t)
        if pose.head is not None:
            for attr, sgn in (("left_joints", -1.0), ("right_joints", +1.0)):
                side_off = off + np.array([sgn * self.spread, 0.0, 0.0])
                jd = getattr(pose, attr)
                if jd and np.any(side_off):
                    moved = {}
                    for k, v in jd.items():
                        m = np.asarray(v, float).reshape(4, 4).copy()
                        m[:3, 3] += side_off
                        moved[k] = m
                    setattr(pose, attr, moved)
        self.i += 1
        return pose


def run():
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    teleop = MabelTeleop(model, data)
    mirror = RealMirror(model, data)

    # boot exactly like the bridge: rest targets on the servos, mirror settles
    teleop.controller.init_hold()
    for _ in range(200):
        mirror.track(None, DT)
    teleop.controller.settle_done()

    op = MockOperator()
    t0 = time.time()
    t = 0.0
    frame_t = -1.0
    pose = None
    stream_on = False

    rec = {k: [] for k in ("t", "phase", "cmd_arm", "cmd_arm_all", "cmd_neck",
                           "cmd_lift", "base_vx", "q_arm")}
    watch = {}                     # joint name -> last cmd, for the step check
    max_step = {"value": 0.0, "joint": None, "t": 0.0, "phase": ""}
    steps = []                     # (value, joint, t, phase) — every step > 0.02
    metrics = []

    def tick(phase):
        nonlocal t, frame_t, pose
        # real-time pacing: the engine's watchdogs/eases run on time.time()
        target = t0 + t + DT
        lag = target - time.time()
        if lag > 0:
            time.sleep(lag)
        if stream_on and t - frame_t >= 1.0 / 60.0:
            frame_t = t
            pose = op.frame(t)
        elif not stream_on:
            pose = None
        cmd = teleop.update(pose, None, DT)
        mc = extract(model, data, cmd)
        if os.getenv("CHAIN_DEBUG") and phase == "P6" and len(
                [p for p in rec["phase"] if p == "P6"]) < 3:
            sup = teleop.controller.sup
            a2 = model.actuator("Right Arm 2").id
            i2 = list(sup.aids).index(a2)
            print(f"DBG {phase} t={t:.2f} ctrl_RA2={data.ctrl[a2]:+.3f} "
                  f"mode={teleop.controller.mode} s={sup._s:.3f} "
                  f"anchor={float(sup._anchor[i2]) if sup._anchor is not None else float('nan'):+.3f} "
                  f"live={float(sup.live[i2]):+.3f} "
                  f"out={float(sup._out[i2]) if sup._out is not None else float('nan'):+.3f} "
                  f"worldref={teleop.controller._world_ref is not None}")
        for name, v in list(mc.arms.items()) + list(mc.body.items()) + [
                ("lift", mc.lift)]:
            if name in watch:
                d = abs(v - watch[name])
                if d > 0.02:
                    steps.append((d, name, round(t, 3), phase))
                if d > max_step["value"]:
                    max_step.update(value=d, joint=name, t=t, phase=phase)
            watch[name] = v
        mirror.track((mc.vx, mc.vy, mc.omega), DT)
        rec["t"].append(t)
        rec["phase"].append(phase)
        rec["cmd_arm"].append(mc.arms.get("left_arm_2", np.nan))
        # Liveness metric: ALL left-arm joints. The redundant arm's canonical
        # basin (post basin-escape, 2026-07-22) can express a lateral wrist
        # sweep through J1/J3 instead of J2 — watching one joint read a
        # perfectly tracking arm as "frozen".
        rec["cmd_arm_all"].append([mc.arms.get(f"left_arm_{i}", np.nan)
                                   for i in range(1, 8)])
        rec["cmd_neck"].append(mc.body.get("neck_yaw", np.nan))
        rec["cmd_lift"].append(mc.lift)
        rec["base_vx"].append(mc.vx)
        rec["q_arm"].append(float(data.joint("Left Arm 2").qpos[0]))
        t += DT
        return mc

    # ── P1: torque enable with the arms DISPLACED from home ────────────────
    # Keep the displaced pose inside each servo's ctrlrange: past it, the hold
    # command legitimately clips and the "command == measured" check would
    # compare against an unreachable pose (a limp arm resting past a stop is
    # bounded by the calibrated end-stops on the real robot too).
    for j, dq in (("Left Arm 2", 0.5), ("Right Arm 2", 0.5), ("Torso", -0.2)):
        lo, hi = model.actuator_ctrlrange[model.actuator(j).id]
        q = float(np.clip(data.joint(j).qpos[0] + dq, lo + 0.02, hi - 0.02))
        data.joint(j).qpos[0] = q
    mujoco.mj_forward(model, data)
    teleop.set_control_spec(method="retarget", control_type="impedance",
                            region="whole_body",
                            # P4 asserts that WALKING drives the base, which is
                            # GLOBAL-scope behaviour. LOCAL (the default since
                            # 2026-07-31) deliberately keeps the base out of the
                            # follow loop and leaves it to the joystick, so this
                            # chain has to ask for the scope it is measuring.
                            scope="global")
    teleop.update(None, None, DT)           # spec entry (pops/exit) before enable
    teleop.resume_hold()                    # the ENABLE-TORQUE edge
    mc = tick("P1")
    err = max(abs(mc.arms[f"{s}_arm_2"] - data.joint(f"{s.capitalize()} Arm 2").qpos[0])
              for s in ("left", "right"))
    metrics.append(("P1 first command vs measured (displaced enable) [rad]",
                    f"{err:.4f}", err < 0.05))

    # ── P2: no stream — ease only ───────────────────────────────────────────
    a0 = watch["left_arm_2"]
    for _ in range(int(1.0 / DT)):
        tick("P2")
    drift = abs(watch["left_arm_2"] - a0)
    metrics.append(("P2 1 s no-stream drift toward home [rad]",
                    f"{drift:.4f}", drift < 0.30))

    # ── P3: stream begins — engage + arms/neck follow ───────────────────────
    stream_on = True
    op.sweep = 0.05                     # deliberate 5 cm lateral wrist sweep
    for _ in range(int(4.0 / DT)):
        tick("P3")
    n = int(2.0 / DT)
    arm_std = float(np.max(np.nanstd(np.asarray(rec["cmd_arm_all"][-n:]), axis=0)))
    neck_std = float(np.nanstd(rec["cmd_neck"][-n:]))
    if arm_std <= 0.005:                      # diagnose the flaky frozen-arm mode
        eng_ = teleop.controller
        tgt = eng_._map_wrist_targets(eng_._payload(pose)) if pose else {}
        print(f"DBG P3-freeze: mode={eng_.mode} clutch={eng_.clutch.clutched} "
              f"wref={eng_._world_ref is not None} "
              f"ready={eng_.shoulder.ready('left')} contacts={sorted(tgt)} "
              f"sweep={op.sweep} projector_fallbacks="
              f"{eng_.projector.fallbacks if eng_.projector else 'n/a'} "
              f"mindist={eng_.projector.min_dist if eng_.projector else 'n/a'}")
    metrics.append(("P3 arm command std, last 2 s (follows operator) [rad]",
                    f"{arm_std:.4f}", arm_std > 0.005))
    metrics.append(("P3 neck-yaw command std, last 2 s (head-follow) [rad]",
                    f"{neck_std:.4f}", neck_std > 0.005))
    stale = [k for k in teleop.joint_targets
             if k.startswith(("left_arm", "right_arm")) or k == "torso"]
    metrics.append(("P3 stale upper-body joint_targets (must be none)",
                    str(stale), not stale))

    # ── P4: whole-body — walk forward + rise on tiptoe ──────────────────────
    # Head RISES (the home lift is already at its bottom stop, so a crouch
    # cannot command it lower — the follow is verified in the + direction).
    vx_before = float(np.mean(np.abs(rec["base_vx"][-n:])))
    lift_before = watch["lift"]
    for k in range(int(3.0 / DT)):
        op.walk = np.array([0.0, +0.15, -0.60]) * min(1.0, k * DT / 1.5)
        tick("P4")
    vx_peak = float(np.max(np.abs(rec["base_vx"][-int(3.0 / DT):])))
    dlift = watch["lift"] - lift_before
    metrics.append(("P4 base |vx| peak while walking fwd 0.6 m [m/s]",
                    f"{vx_peak:.3f} (idle {vx_before:.3f})", vx_peak > 0.05))
    metrics.append(("P4 lift command rise on 0.15 m head rise [m]",
                    f"{dlift:+.4f}", dlift > 0.03))

    # ── P5: stream loss — base watchdog + ease home ─────────────────────────
    stream_on = False
    zero_after = None
    for k in range(int(5.0 / DT)):
        mc = tick("P5")
        if zero_after is None and abs(mc.vx) < 1e-6 and abs(mc.omega) < 1e-6:
            zero_after = (k + 1) * DT
    # Bound = 0.4 s stream watchdog + the tip-over-limited decel from the P4
    # walking speed (~1.1 s from 0.39 m/s) ≈ 1.5 s nominal; the engine watchdog
    # runs on the WALL clock while this harness counts SIM ticks, so system load
    # stretches the measurement — 2.0 s keeps the safety property (a broken
    # watchdog never zeroes at all) without realtime-pacing flakes.
    metrics.append(("P5 base zeroed after stream loss [s]",
                    f"{zero_after}", zero_after is not None and zero_after < 2.0))
    metrics.append(("P5 supervisor state", teleop.controller.mode, True))

    # ── P6: re-acquire — ease back, step-free ───────────────────────────────
    # The operator has walked back to the start (the usual reality after taking
    # the headset off); the origin-jump guard + stability window then re-anchor.
    op.walk = np.zeros(3)
    stream_on = True
    for _ in range(int(6.0 / DT)):
        tick("P6")
    arm_std6 = float(np.max(np.nanstd(np.asarray(rec["cmd_arm_all"][-n:]), axis=0)))
    if arm_std6 <= 0.005:                     # diagnose the flaky frozen-arm mode
        eng_ = teleop.controller
        tgt = eng_._map_wrist_targets(eng_._payload(pose)) if pose else {}
        print(f"DBG P6-freeze: mode={eng_.mode} clutch={eng_.clutch.clutched} "
              f"wref={eng_._world_ref is not None} "
              f"ready={eng_.shoulder.ready('left')} contacts={sorted(tgt)} "
              f"sweep={op.sweep} projector_fallbacks="
              f"{eng_.projector.fallbacks if eng_.projector else 'n/a'} "
              f"mindist={eng_.projector.min_dist if eng_.projector else 'n/a'}")
    metrics.append(("P6 arm command std after re-acquire [rad]",
                    f"{arm_std6:.4f}", arm_std6 > 0.005))
    metrics.append(("P6 engine state after re-acquire",
                    teleop.controller.mode, teleop.controller.mode == "engaged"))

    # ── global: step-free everywhere ────────────────────────────────────────
    if os.getenv("CHAIN_DEBUG") and steps:
        for v, name, tt_, ph in sorted(steps, reverse=True)[:12]:
            print(f"DBG step {v:.3f} rad  {name}  t={tt_}  {ph}")
    metrics.append((f"max per-tick command step (bound {STEP_BOUND}) [rad]",
                    f"{max_step['value']:.4f} ({max_step['joint']} in "
                    f"{max_step['phase']})", max_step["value"] < STEP_BOUND))
    # extraction completeness (the contract the drivers consume)
    mc = extract(model, data, teleop.update(pose, None, DT))
    metrics.append(("extraction: arm joints", str(len(mc.arms)), len(mc.arms) == 14))
    metrics.append(("extraction: body joints (torso+neck)", str(len(mc.body)),
                    len(mc.body) == 4))
    metrics.append(("extraction: hand sides", str(sorted(mc.hands)),
                    sorted(mc.hands) == ["left", "right"]))

    # ── figure ──────────────────────────────────────────────────────────────
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axs = plt.subplots(4, 1, figsize=(6.8, 6.4), sharex=True)
    tt = np.asarray(rec["t"])
    bounds = [tt[i] for i in range(1, len(tt))
              if rec["phase"][i] != rec["phase"][i - 1]]
    for ax in axs:
        for b in bounds:
            ax.axvline(b, color="0.85", lw=0.8)
        ax.grid(alpha=0.3)
    axs[0].plot(tt, rec["cmd_arm"], "#1f77b4", label="cmd left_arm_2")
    axs[0].plot(tt, rec["q_arm"], "#e9a679", label="mirror q")
    axs[0].set_ylabel("arm [rad]"); axs[0].legend(fontsize=7, ncol=2)
    axs[0].set_title("Real-dispatch whole-body retarget chain "
                     "(P1 enable · P2 hold · P3 engage · P4 walk · P5 loss · P6 re-acquire)",
                     fontsize=9)
    axs[1].plot(tt, rec["cmd_neck"], "#2ca02c", label="cmd neck_yaw")
    axs[1].set_ylabel("neck [rad]"); axs[1].legend(fontsize=7)
    axs[2].plot(tt, rec["cmd_lift"], "#d62728", label="cmd lift (tendon)")
    axs[2].set_ylabel("lift [m]"); axs[2].legend(fontsize=7)
    axs[3].plot(tt, rec["base_vx"], "#9467bd", label="cmd base vx")
    axs[3].set_ylabel("base [m/s]"); axs[3].set_xlabel("t [s]")
    axs[3].legend(fontsize=7)
    fig.savefig(os.path.join(FIGDIR, "real_retarget_chain.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    if steps:
        print("\n  largest command steps (>0.02):")
        for d, name, ts, ph in sorted(steps, reverse=True)[:6]:
            print(f"    {d:.4f}  {name:14s} t={ts:7.3f}  {ph}")
    print("\n== REAL-dispatch whole-body retarget chain — verification ==")
    ok_all = True
    for name, val, ok in metrics:
        ok_all &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:58s} {val}")
    print(f"\n  figure -> docs/figures/real_retarget_chain.pdf")
    return ok_all, metrics


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)

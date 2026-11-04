#!/usr/bin/env python3
"""
RETARGET LATENCY — why the arm lags the operator, and which knob owns it.

THE OPERATOR REPORT (2026-07-28)
--------------------------------
    "the teleop whole body re-targeting is way too slow — I need it to be fast
     and more real time for me."

Measured first over the live wire (server/tools/exp_retarget_latency.py): a
0.4 Hz hand sweep came back 245 ms behind, and a 25 cm hand step took 467 ms to
reach 63 % of its travel. This harness isolates how much of that the ENGINE owns
(the wire number also contains the sim's servo and the state stream) and sweeps
the two knobs that can plausibly own it.

THE HYPOTHESIS
--------------
Both of the arm solver's anti-jitter mechanisms are defined PER FRAME, not per
second, so their effect in seconds scales with the operator's frame interval:

  * ARM_TRUST_STEP  bounds the solve to +/- 0.10 rad per SOLVE. Its own comment
    reads "0.10 rad @ 60 Hz frames = a 6 rad/s per-joint slew ... far above
    human teleop speed, so tracking feel is unchanged."
  * ARM_W_SMOOTH    penalises |q - q_prev|^2 against the previous SOLVE. For a
    first-order response the settling time in SECONDS is ~ dt * W / P, i.e.
    directly proportional to the frame interval.

The Vision Pro app streamed at 60 Hz when both were tuned. On 2026-07-19 it
dropped to 30 Hz to kill a SwiftUI invalidation storm (visionOS_v2 the root README
gotcha #21: ~90 Hz @Published writes starved the send timer, so "steady beats
fast" won). Nothing in the engine noticed. At 30 Hz the trust region is a
3 rad/s slew, not 6, and the smoothness lag is twice as long in seconds — the
sluggishness is the exact cost of that trade, paid on the server.

If the hypothesis holds, lag at a FIXED knob setting scales with the frame
interval, and rate-defined knobs (rad/SECOND, and a W scaled by dt) make it flat.

WHAT IT MEASURES
----------------
The engine runs against a PERFECT plant (qpos := ctrl), so what comes out is
the commanded wrist — engine lag alone, with no servo dynamics folded in.
Cells sweep frame rate x knob definition; the metric is the cross-correlation
delay between the operator's hand and the robot's commanded wrist.

    python3 controller/experiments/exp_retarget_latency.py [--quick]
Figure: controller/docs/figures/retarget_latency.pdf
"""
import argparse
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
from mabel.teleop_engine import arm as arm_mod                   # noqa: E402
from exp_real_retarget_chain import MockOperator, MODEL_XML      # noqa: E402

FIGDIR = os.path.join(os.path.dirname(_HERE), "docs", "figures")

CTRL_HZ = 50.0
DT = 1.0 / CTRL_HZ
SWEEP_HZ = 0.5              # MockOperator's built-in lateral sweep frequency
SWEEP_AMP = 0.18            # m per side
SETTLE = 4.0                # s of streaming before the measured window
MEASURE = 10.0              # s of measured sweep


def phase_lag(t, op, rob, max_lag=0.8, grid_hz=200.0):
    """Delay in seconds that best aligns `rob` with `op` (cross-correlation).

    Both signals are resampled onto a uniform grid first: the control loop is
    not perfectly isochronous, and correlating unevenly-sampled arrays folds
    the loop's own jitter into the answer.
    """
    grid = np.arange(t[0], t[-1], 1.0 / grid_hz)
    a = np.interp(grid, t, op)
    b = np.interp(grid, t, rob)
    a = a - a.mean()
    b = b - b.mean()
    best, best_c = 0, -np.inf
    for k in range(int(max_lag * grid_hz)):
        sa, sb = a[: len(a) - k], b[k:]
        d = np.linalg.norm(sa) * np.linalg.norm(sb)
        if d <= 0:
            continue
        c = float(np.dot(sa, sb) / d)
        if c > best_c:
            best_c, best = c, k
    return best / grid_hz, best_c


class Cell:
    """One run of the engine at a given frame rate against a given plant.

    `plant` is the whole point of the decomposition:
      "ideal"   every joint lands on its command instantly -> ENGINE lag alone.
      "physics" mj_step at the bridge's own timestep -> engine + SERVO lag,
                i.e. what the operator actually sees streamed back.
    The difference between the two IS the servo's contribution.
    """

    def __init__(self, frame_hz, trust, w_smooth, rate_based, plant="ideal",
                 kp_scale=1.0):
        self.frame_hz = frame_hz
        self.trust = trust
        self.w_smooth = w_smooth
        self.rate_based = rate_based
        self.plant = plant
        self.kp_scale = kp_scale

    @property
    def label(self):
        mode = "rate" if self.rate_based else "frame"
        k = f" kp x{self.kp_scale:g}" if self.kp_scale != 1.0 else ""
        return f"{self.frame_hz:.0f} Hz/{mode}/{self.plant}{k}"

    def run(self):
        # Knobs live as module globals copied in by `from .hyperparams import *`,
        # so the sweep patches the ARM MODULE's copy — that is what solve() reads.
        if self.rate_based:
            # Rate-defined: the per-solve step is the design slew (6 rad/s at the
            # 60 Hz the constant was tuned at) times this cell's frame interval,
            # and W is scaled so the smoothing time constant stays fixed in
            # seconds. Both become frame-rate invariant by construction.
            arm_mod.ARM_TRUST_STEP = self.trust * (60.0 / self.frame_hz)
            arm_mod.ARM_W_SMOOTH = self.w_smooth * (self.frame_hz / 60.0)
        else:
            arm_mod.ARM_TRUST_STEP = self.trust
            arm_mod.ARM_W_SMOOTH = self.w_smooth

        model = mujoco.MjModel.from_xml_path(MODEL_XML)
        arm_acts = [aid for aid in range(model.nu)
                    if (model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                        and " Arm " in (model.actuator(aid).name or ""))]
        if self.kp_scale != 1.0:
            # A MuJoCo position servo is gainprm[0] = kp and biasprm[1] = -kp;
            # scaling only one of them turns the servo into an open-loop torque
            # source, so both move together. kv (biasprm[2]) scales as sqrt(kp)
            # to hold the damping ratio — bumping kp alone makes it ring.
            for aid in arm_acts:
                model.actuator_gainprm[aid, 0] *= self.kp_scale
                model.actuator_biasprm[aid, 1] *= self.kp_scale
                model.actuator_biasprm[aid, 2] *= np.sqrt(self.kp_scale)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        teleop = MabelTeleop(model, data)
        teleop.controller.init_hold()

        acts = [(aid, model.jnt_qposadr[model.actuator_trnid[aid, 0]])
                for aid in range(model.nu)
                if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT]
        sat = {"n": 0, "tot": 0}

        def plant():
            if self.plant == "ideal":
                for aid, qadr in acts:
                    data.qpos[qadr] = float(data.ctrl[aid])
                mujoco.mj_forward(model, data)
                return
            # Physics: advance real time by one control tick, exactly as the
            # bridge's accumulator does.
            n = max(1, int(round(DT / model.opt.timestep)))
            for _ in range(n):
                mujoco.mj_step(model, data)
            for aid in arm_acts:
                lo, hi = model.actuator_forcerange[aid]
                sat["tot"] += 1
                if hi > 0 and abs(float(data.actuator_force[aid])) > 0.98 * hi:
                    sat["n"] += 1

        for _ in range(100):
            plant()
        teleop.controller.settle_done()

        op = MockOperator()
        op.sweep = SWEEP_AMP
        wrist_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Right Wrist")

        t0 = time.time()
        t = 0.0
        frame_t = -1e9
        pose = None
        rec_t, rec_op, rec_rob = [], [], []
        total = SETTLE + MEASURE
        while t < total:
            target = t0 + t + DT
            lag = target - time.time()
            if lag > 0:
                time.sleep(lag)          # the engine's watchdogs run on time.time()
            if t - frame_t >= 1.0 / self.frame_hz:
                frame_t = t
                pose = op.frame(t)
            teleop.update(pose, None, DT)
            plant()
            if t >= SETTLE:
                rec_t.append(t)
                # Operator hand lateral offset — MockOperator's own sweep law.
                rec_op.append(SWEEP_AMP * np.sin(2 * np.pi * SWEEP_HZ * t))
                rec_rob.append(float(data.xanchor[wrist_jid][1]))
            t += DT

        t_a = np.asarray(rec_t)
        rob_a = np.asarray(rec_rob)
        # Ringing shows up as energy well above the 0.5 Hz sweep. Measure it as
        # the RMS of the second difference (discrete acceleration) normalised by
        # what the sweep itself demands — a clean servo sits near 1.
        d2 = np.diff(rob_a, 2)
        expect = (2 * np.pi * SWEEP_HZ * DT) ** 2 * float(np.ptp(rob_a)) / 2
        ring = float(np.sqrt(np.mean(d2 ** 2))) / max(expect, 1e-9)
        lag_s, corr = phase_lag(t_a, np.asarray(rec_op), np.asarray(rec_rob))
        amp_op = float(np.ptp(rec_op))
        amp_rob = float(np.ptp(rec_rob))
        return {"lag": lag_s, "corr": corr, "amp": amp_rob / max(amp_op, 1e-6),
                "sat": sat["n"] / max(sat["tot"], 1), "ring": ring,
                "t": t_a, "op": np.asarray(rec_op), "rob": np.asarray(rec_rob)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="30 Hz only — the app's actual rate")
    args = ap.parse_args()

    TRUST = 0.10             # the shipped ARM_TRUST_STEP
    W = 2.0                  # the shipped ARM_W_SMOOTH
    # DECOMPOSITION FIRST, knob sweep second. The wire says 245 ms; the ideal
    # plant says how much of that the ENGINE owns and the physics plant says how
    # much the SERVO owns. Only then is it worth asking which knob to turn.
    cells = [Cell(30.0, TRUST, W, False, "ideal"),
             Cell(30.0, TRUST, W, False, "physics"),
             # x5 is not a tuning guess: it lands on kp=500 / kv=45, which is
             # what the STANDALONE arm model (models/right_arm.xml) already
             # runs on the same joints with the same force limits. The full
             # model's 100/20 is the outlier of the two.
             Cell(30.0, TRUST, W, False, "physics", kp_scale=5.0),
             Cell(30.0, TRUST, W, False, "physics", kp_scale=9.0)]
    if not args.quick:
        cells += [Cell(r, TRUST, W, rb, "physics")
                  for r in (15.0, 60.0) for rb in (False, True)]
        cells += [Cell(30.0, TRUST, W, True, "physics")]

    print(f"{'cell':<28s} {'lag (ms)':>10s} {'corr':>7s} {'amp':>7s} "
          f"{'sat':>7s} {'ring':>7s}")
    print("-" * 72)
    out = []
    for c in cells:
        r = c.run()
        out.append((c, r))
        print(f"{c.label:<28s} {r['lag'] * 1000:>10.0f} {r['corr']:>7.3f} "
              f"{r['amp']:>7.2f} {r['sat'] * 100:>6.0f}% {r['ring']:>7.2f}")

    ideal = next((r for c, r in out if c.plant == "ideal"), None)
    phys = next((r for c, r in out if c.plant == "physics" and c.kp_scale == 1.0
                 and c.frame_hz == 30.0 and not c.rate_based), None)
    if ideal and phys:
        print(f"\nDECOMPOSITION at the app's 30 Hz:")
        print(f"  engine            {ideal['lag'] * 1000:>5.0f} ms")
        print(f"  + arm servo       {(phys['lag'] - ideal['lag']) * 1000:>5.0f} ms")
        print(f"  = commanded->real {phys['lag'] * 1000:>5.0f} ms  "
              f"(wire measured 245 ms incl. transport + state stream)")

    # The hypothesis in one number: does lag track the frame interval?
    frame_cells = [(c, r) for c, r in out
                   if not c.rate_based and c.plant == "physics" and c.kp_scale == 1.0]
    rate_cells = [(c, r) for c, r in out
                  if c.rate_based and c.plant == "physics" and c.kp_scale == 1.0]
    if len(frame_cells) > 1 and len(rate_cells) > 1:
        sp_f = (max(r["lag"] for _, r in frame_cells)
                - min(r["lag"] for _, r in frame_cells))
        sp_r = (max(r["lag"] for _, r in rate_cells)
                - min(r["lag"] for _, r in rate_cells))
        print(f"\nlag spread across 15-60 Hz (physics plant):")
        print(f"  per-FRAME knobs : {sp_f * 1000:.0f} ms  "
              f"(rate-dependent -> the app's 60->30 Hz drop cost this)")
        print(f"  per-RATE  knobs : {sp_r * 1000:.0f} ms  (invariant)")
    at30 = {c.rate_based: r for c, r in out
            if c.frame_hz == 30.0 and c.plant == "physics" and c.kp_scale == 1.0}
    if len(at30) == 2 and at30[False]["lag"] > 0:
        print(f"\nAT THE APP'S 30 Hz: {at30[False]['lag'] * 1000:.0f} ms -> "
              f"{at30[True]['lag'] * 1000:.0f} ms "
              f"({(1 - at30[True]['lag'] / at30[False]['lag']) * 100:+.0f} %)")

    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    kp_cells = [(c, r) for c, r in out
                if c.plant == "physics" and c.frame_hz == 30.0 and not c.rate_based]
    if ideal:
        ax[0].axhline(ideal["lag"] * 1000, color="0.5", ls=":", lw=1.0,
                      label="engine alone (ideal plant)")
    ax[0].plot([c.kp_scale for c, _ in kp_cells],
               [r["lag"] * 1000 for _, r in kp_cells], "o-", label="engine + servo")
    ax[0].set_xlabel("arm servo stiffness (x nominal kp)")
    ax[0].set_ylabel("following lag (ms)")
    ax[0].set_title("Where the lag lives")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for c, r in kp_cells:
        ax[1].plot(r["t"], (r["rob"] - r["rob"].mean()) * 100, lw=1.2,
                   label=f"kp x{c.kp_scale:g} ({r['lag'] * 1000:.0f} ms)")
    if kp_cells:
        r0 = kp_cells[0][1]
        ax[1].plot(r0["t"], (r0["op"] - r0["op"].mean()) * 100, "k:", lw=1.0,
                   label="operator hand")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("lateral offset (cm)")
    ax[1].set_title("Realised wrist at the app's 30 Hz")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "retarget_latency.pdf")
    fig.savefig(path)
    print(f"\nfigure -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

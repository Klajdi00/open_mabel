#!/usr/bin/env python3
"""
exp_wbc_ref_smoothing.py — does the WBC impedance REFERENCE smoother tame the
jittery task-space wrist tracking the operator reported ("very jittery vs the
smooth joint slider")?

BACKGROUND (measured, see the reconnaissance in the WBC doc). The task-space
wrist target reaches the robot as a MULTI-JOINT IK solution over the wire
(`joint_command`); its null-space/redundancy resolution flips frame to frame, so
the commanded q_des JUMPS between 60 Hz frames. Fed straight into the impedance
law  τ = bias + K(q_des − q) − D q̇  (engine.py) with the modest joint damping
D = 0.6, every jump steps the spring and the arm BUZZES. The joint slider is
smooth only because ITS q_des moves slowly and one joint at a time.

The fix (controller/mabel/whole_body_controller/engine.py `_smooth_reference`,
tuned by `MABEL_WBC_REF_TAU` / `MABEL_WBC_REF_SLEW` in config.py) filters the
REFERENCE the spring tracks: a first-order low-pass rejects the HF wiggle and a
slew cap turns a large jump into a deliberate ramp — "slower but surely".

This harness drives ONE arm joint through the FULL facade (real physics, mj_step)
with three stimuli, each replayed with the smoother OFF (verbatim, the old
behaviour) and ON (the shipped default), and quantifies:

  E1  NOISY TRACK — a slow intended sweep + injected per-frame IK jitter. Metric:
      realized joint-ACCELERATION RMS (the physical buzz the operator sees and the
      motor feels) and tracking error to the INTENDED (noise-free) sweep. The
      smoother must cut the acceleration RMS hard while tracking the intent at
      least as well (it ignores the noise the raw law chases).
  E2  STEP — a sudden wrist jump (+0.4 rad). Metric: overshoot and the settling
      shape. The smoother must remove the spring-impulse overshoot and move the
      arm in a bounded, monotone ramp.
  E3  CLEAN — a slow smooth sweep with NO noise (the joint-slider case). Metric:
      the lag the smoother ADDS. It must be a few tens of ms — i.e. the smoother
      does not degrade the already-smooth path.

Run:  ~/anaconda3/bin/python3 controller/experiments/exp_wbc_ref_smoothing.py
      (figure -> controller/docs/figures/wbc_ref_smoothing.pdf)

A summary table + PASS/FAIL is printed; the process exits non-zero on any FAIL.
All numbers come from the sim — nothing is hand-entered.
"""
import os
import sys
import time as _time_mod

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))     # -> mabel package
sys.path.insert(0, _HERE)                          # -> exp_wbc_validation (WBCRig, SimClock)

# exp_wbc_validation patches time.time() to a deterministic sim clock at import and
# defines the settled WBCRig we reuse verbatim.
import exp_wbc_validation as V                      # noqa: E402
from exp_wbc_validation import WBCRig, CLOCK, PAL, _save, check, FAILURES  # noqa: E402
import mujoco                                       # noqa: E402
from mabel import config as C                       # noqa: E402

SEED = 0
JOINT, IOS = "Right Arm 4", "right_arm_4"
STIFFNESS = 30.0        # ~ the app default (slider ~0.6 × maxStiffness 45)
FRAME_HZ = 60.0         # the apps stream joint_command at 60 Hz
FIGDIR = os.path.join(_HERE, "..", "docs", "figures")

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.6,
                     "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})


def _set_smoothing(rig, on):
    """Toggle the WHOLE stabilization fix (reference smoother + stiffness-scaled
    damping) on the live engine. ON = the shipped new law (config defaults); OFF =
    the OLD law: verbatim q_des AND base damping only (no √K term). The toggle drives
    both parts together so the before/after is a faithful old-vs-new comparison."""
    eng = rig.teleop.wbc
    if on:
        eng._ref_tau = float(C.WBC_REF_TAU)
        eng._ref_slew = float(C.WBC_REF_SLEW)
        eng._damp_ratio = float(C.WBC_DAMP_RATIO)
    else:
        eng._ref_tau = 0.0          # no low-pass
        eng._ref_slew = 1.0e9       # no slew cap  -> q_des passes through verbatim
        eng._damp_ratio = 0.0       # base damping only (the old underdamped law)
    eng._q_ref.clear()              # re-seed to the current pose on the next tick


def drive(rig, cmd_frames):
    """Feed a 60 Hz joint-target stream through the facade; step physics at the model
    timestep. Sample the realized joint qpos at EVERY physics substep (to resolve the
    buzz) plus the command held over that frame. Returns fine-time arrays."""
    dt = rig.dt
    n_sub = max(1, int(round((1.0 / FRAME_HZ) / dt)))
    t, q, cmd = [], [], []
    for c in cmd_frames:
        rig.teleop.set_joint_targets({IOS: float(c)})
        for _ in range(n_sub):
            CLOCK.advance(dt)
            rig.d.xfrc_applied[:] = 0.0
            rig.teleop.update(None, dt=dt)
            mujoco.mj_step(rig.m, rig.d)
            t.append(CLOCK.t)
            q.append(float(rig.d.joint(JOINT).qpos[0]))
            cmd.append(float(c))
    t = np.asarray(t); t -= t[0]
    return t, np.asarray(cmd), np.asarray(q)


def _accel_rms(t, q):
    """RMS of the realized joint acceleration (rad/s²) — the physical 'buzz'. Uniform
    dt from the deterministic clock, so a plain second difference is exact."""
    dt = float(np.median(np.diff(t)))
    a = np.diff(q, 2) / (dt * dt)
    return float(np.sqrt(np.mean(a * a)))


def _new_rig():
    rig = WBCRig()
    rig.teleop.set_control_spec(method="wrist", control_type="impedance",
                               region="arm", stiffness=STIFFNESS)
    rig.run(0.5)            # enter the torque engine + settle at the seeded target
    return rig


def _prime_command(rig):
    """The joint's home value — every stimulus starts from where the arm rests so the
    first frame is not itself a step."""
    return float(rig.d.joint(JOINT).qpos[0])


# =====================================================================
# E1 — noisy track: slow intent + per-frame IK jitter
# =====================================================================
def e1_noisy():
    print("\n=== E1  NOISY TRACK: slow wrist sweep + injected per-frame IK jitter ===")
    T = 8.0
    n = int(T * FRAME_HZ)
    tf = np.arange(n) / FRAME_HZ
    rng = np.random.default_rng(SEED)
    out = {}
    for on in (False, True):
        rig = _new_rig()
        q0 = _prime_command(rig)
        intent = q0 + 0.35 * np.sin(2 * np.pi * 0.25 * tf)          # the TRUE wrist sweep
        noise = 0.025 * rng.standard_normal(n)                       # frame-to-frame IK flip
        cmd = intent + noise
        _set_smoothing(rig, on)
        t, c, q = drive(rig, cmd)
        # tracking error to the INTENDED (noise-free) sweep, resampled to fine time
        intent_fine = np.interp(t, tf, intent)
        err = float(np.sqrt(np.mean((q - intent_fine) ** 2)))
        out[on] = {"t": t, "cmd": c, "q": q, "intent_fine": intent_fine,
                   "accel_rms": _accel_rms(t, q), "track_rms": err,
                   "cmd_accel_rms": _accel_rms(tf, cmd)}
        tag = "ON " if on else "OFF"
        print(f"  smoother {tag}: joint accel RMS {out[on]['accel_rms']:7.2f} rad/s²  |  "
              f"tracking err to intent {np.degrees(err):5.2f}°")
    print(f"  (command accel RMS, injected: {out[False]['cmd_accel_rms']:.1f} rad/s²)")
    red = out[True]["accel_rms"] / max(out[False]["accel_rms"], 1e-9)
    check("E1 smoother cuts realized joint-accel (buzz) RMS by >2x",
          red < 0.5, f"ON/OFF = {red:.2f} ({out[False]['accel_rms']:.1f} -> "
          f"{out[True]['accel_rms']:.1f} rad/s²)")
    check("E1 smoother tracks the INTENT at least as well as raw (ignores noise)",
          out[True]["track_rms"] <= out[False]["track_rms"] + np.radians(0.5),
          f"ON {np.degrees(out[True]['track_rms']):.2f}° vs OFF "
          f"{np.degrees(out[False]['track_rms']):.2f}°")
    return out


# =====================================================================
# E2 — step: a sudden wrist jump
# =====================================================================
def e2_step():
    print("\n=== E2  STEP: sudden +0.4 rad wrist jump (slower-but-surely) ===")
    T = 4.0
    n = int(T * FRAME_HZ)
    out = {}
    for on in (False, True):
        rig = _new_rig()
        q0 = _prime_command(rig)
        q1 = q0 + 0.40
        cmd = np.concatenate([np.full(int(0.5 * FRAME_HZ), q0),
                              np.full(n - int(0.5 * FRAME_HZ), q1)])
        _set_smoothing(rig, on)
        t, c, q = drive(rig, cmd)
        span = q1 - q0
        x = (q - q0) / span
        overshoot = float(max(0.0, np.max(x) - 1.0) * 100.0)
        # 10-90% rise
        try:
            t10 = t[np.argmax(x >= 0.1)]
            t90 = t[np.argmax(x >= 0.9)]
            rise = float(t90 - t10)
        except Exception:
            rise = float("nan")
        out[on] = {"t": t, "q": q, "cmd": c, "overshoot": overshoot, "rise": rise, "q0": q0}
        tag = "ON " if on else "OFF"
        print(f"  smoother {tag}: overshoot {overshoot:5.1f} %  |  10–90% rise {rise:4.2f} s")
    check("E2 fix settles cleanly on a step (overshoot < 8%)",
          out[True]["overshoot"] < 8.0, f"{out[True]['overshoot']:.1f} % "
          f"(old law {out[False]['overshoot']:.1f} %)")
    check("E2 fix at least halves the step overshoot vs the old law",
          out[True]["overshoot"] < 0.6 * out[False]["overshoot"],
          f"ON {out[True]['overshoot']:.1f} % vs OFF {out[False]['overshoot']:.1f} %")
    return out


# =====================================================================
# E3 — clean: the joint-slider case (no noise) — added lag must be small
# =====================================================================
def e3_clean():
    print("\n=== E3  CLEAN sweep (joint-slider case): added lag must stay small ===")
    T = 6.0
    n = int(T * FRAME_HZ)
    tf = np.arange(n) / FRAME_HZ
    out = {}
    for on in (False, True):
        rig = _new_rig()
        q0 = _prime_command(rig)
        cmd = q0 + 0.30 * np.sin(2 * np.pi * 0.2 * tf)      # smooth, slow, no noise
        _set_smoothing(rig, on)
        t, c, q = drive(rig, cmd)
        out[on] = {"t": t, "q": q, "cmd": c}
    # lag = argmax cross-correlation between the two realized responses
    a = out[False]["q"] - np.mean(out[False]["q"])
    b = out[True]["q"] - np.mean(out[True]["q"])
    dt = float(np.median(np.diff(out[True]["t"])))
    corr = np.correlate(b, a, mode="full")
    lag = (np.argmax(corr) - (len(a) - 1)) * dt
    rms_diff = float(np.sqrt(np.mean((out[True]["q"] - out[False]["q"]) ** 2)))
    print(f"  added lag (ON vs OFF) {lag*1000:5.1f} ms  |  RMS pose diff "
          f"{np.degrees(rms_diff):.2f}°")
    check("E3 smoother adds < 120 ms lag on a clean slow sweep",
          abs(lag) < 0.120, f"{lag*1000:.0f} ms")
    out["lag"] = lag
    return out


def _figure(e1, e2, e3):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # E1: noisy command vs realized, OFF vs ON
    ax = axes[0]
    o0, o1 = e1[False], e1[True]
    ax.plot(o0["t"], np.degrees(o0["cmd"] - o0["cmd"][0]), color="0.7", lw=0.7,
            label="commanded (jittery IK)")
    ax.plot(o0["t"], np.degrees(o0["q"] - o0["q"][0]), color=PAL["red"],
            label=f"realized, smoother OFF ({o0['accel_rms']:.0f} rad/s²)")
    ax.plot(o1["t"], np.degrees(o1["q"] - o1["q"][0]), color=PAL["blue"],
            label=f"realized, smoother ON ({o1['accel_rms']:.0f} rad/s²)")
    ax.plot(o1["t"], np.degrees(o1["intent_fine"] - o0["cmd"][0]), color="k", ls=":",
            lw=1.0, label="intended sweep")
    ax.set_title("E1  noisy wrist track"); ax.set_xlabel("time [s]")
    ax.set_ylabel(f"{JOINT} [deg from start]"); ax.legend(loc="lower center", fontsize=7)

    # E2: step response
    ax = axes[1]
    s0, s1 = e2[False], e2[True]
    ax.plot(s0["t"], np.degrees(s0["cmd"] - s0["q0"]), color="0.7", lw=0.8, label="commanded step")
    ax.plot(s0["t"], np.degrees(s0["q"] - s0["q0"]), color=PAL["red"],
            label=f"OFF (overshoot {s0['overshoot']:.0f}%)")
    ax.plot(s1["t"], np.degrees(s1["q"] - s1["q0"]), color=PAL["blue"],
            label=f"ON (overshoot {s1['overshoot']:.0f}%)")
    ax.axhline(np.degrees(0.40), ls=":", c="k", lw=0.8)
    ax.set_title("E2  sudden wrist jump"); ax.set_xlabel("time [s]")
    ax.set_ylabel("step response [deg]"); ax.legend(loc="lower right", fontsize=7)

    # E3: buzz-RMS bar + clean lag annotation
    ax = axes[2]
    labels = ["noisy track", "step jump"]
    off_vals = [e1[False]["accel_rms"], _accel_rms(e2[False]["t"], e2[False]["q"])]
    on_vals = [e1[True]["accel_rms"], _accel_rms(e2[True]["t"], e2[True]["q"])]
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, off_vals, w, color=PAL["red"], label="smoother OFF")
    ax.bar(x + w / 2, on_vals, w, color=PAL["blue"], label="smoother ON")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("realized joint accel RMS [rad/s²]")
    ax.set_title(f"E3  buzz reduction\n(clean-sweep added lag {e3['lag']*1000:.0f} ms)")
    ax.legend(fontsize=8)

    fig.suptitle("WBC wrist-tracking stabilization: reference smoothing + stiffness-scaled damping "
                 f"(τ={C.WBC_REF_TAU:.2f} s, slew={C.WBC_REF_SLEW:.1f} rad/s, "
                 f"D=0.6+{C.WBC_DAMP_RATIO:.2f}√K, K={STIFFNESS:.0f})")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save("wbc_ref_smoothing.pdf")


def main():
    print(f"MABEL WBC reference-smoothing validation (scene mabel_home.xml, seed {SEED}, "
          f"deterministic sim clock)")
    t0 = _time_mod.perf_counter()
    e1 = e1_noisy()
    e2 = e2_step()
    e3 = e3_clean()
    _figure(e1, e2, e3)

    print("\n  SUMMARY")
    print(f"    E1 buzz RMS   : OFF {e1[False]['accel_rms']:6.1f}  ON {e1[True]['accel_rms']:6.1f} "
          f"rad/s²   ({e1[True]['accel_rms']/max(e1[False]['accel_rms'],1e-9)*100:.0f}% of raw)")
    print(f"    E1 track err  : OFF {np.degrees(e1[False]['track_rms']):5.2f}  "
          f"ON {np.degrees(e1[True]['track_rms']):5.2f} deg (to the intended sweep)")
    print(f"    E2 overshoot  : OFF {e2[False]['overshoot']:5.1f}  ON {e2[True]['overshoot']:5.1f} %")
    print(f"    E3 added lag  : {e3['lag']*1000:5.1f} ms on a clean slow sweep")

    print(f"\n{'='*70}")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print(f"RESULT: ALL CHECKS PASSED   ({_time_mod.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()

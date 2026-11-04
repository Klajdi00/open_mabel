#!/usr/bin/env python3
"""ARM PD TUNING — one gain per joint, chosen from the joint's own inertia.

THE OPERATOR REPORT (2026-08-23)
  "for the controller, default to just pure PD control — tune the PD gains
   nicely, not too stiff to avoid oscillation but also quick to respond."

WHAT IS ACTUALLY WRONG TODAY. Every one of MABEL's fourteen arm joints runs the
same position servo: kp = 500, kv = 45. They are not the same joint. Measured
off the MJCF at the home pose, the reflected inertia spans

    Arm 1  0.536      Arm 2/3/4  0.31      Arm 5  0.053   kg m^2

a factor of ten, and the actuators behind them span 40 / 27 / 7 N.m. One gain
across that spread cannot be right anywhere: with kp = 500, kv = 45 the damping
ratio zeta = kv / (2 sqrt(kp M)) comes out

    Arm 1  1.37     Arm 2  1.80     Arm 3/4  1.82     Arm 5  4.38   Arm 6/7  3.40

so every joint is overdamped and the distal ones are overdamped by more than
4x. Overdamping does not oscillate — it is SLOW, which is the other half of what
the operator asked for. The repo already knows this in principle: the
whole_body_controller the root README says "PD gains AND clamps MUST be per-joint",
and the position controller inherits per-joint gains for exactly this reason.
The MJCF servos the kinematic teleop path actually uses never got the same
treatment.

THE DESIGN. A joint-space PD is a second-order system, so pick the two numbers
that describe one — a bandwidth and a damping ratio — and let each joint's own
inertia turn them into gains:

    kp_j = M_j * w_n^2          kd_j = 2 * zeta * M_j * w_n

Then every joint answers a step with the SAME rise time, and none of them rings.
zeta and w_n are chosen by sweep (P2), not asserted, and the sweep also reports
the actuator headroom so a gain that only works because it saturates is visible.

    python3 controller/experiments/exp_pd_tuning.py --all
    python3 controller/experiments/exp_pd_tuning.py --phase P2 --quick
    python3 controller/experiments/exp_pd_tuning.py --apply     # write the MJCF

Figures: controller/docs/figures/pdtune_*.pdf
"""
import argparse
import json
import os
import re
import time

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
FIGDIR = os.path.join(_HERE, "..", "docs", "figures")
RESDIR = os.path.join(_HERE, "results")
MJCF = os.path.join(_REPO, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
SCENE = os.path.join(_REPO, "simulation", "mabel_mujoco", "scenes", "mabel_sim.xml")

ARM_JOINTS = [f"{s} Arm {i}" for s in ("Left", "Right") for i in range(1, 8)]
WRIST_JOINTS = ["Left Wrist", "Right Wrist"]
PROBE = [f"Right Arm {i}" for i in range(1, 8)]     # one arm is enough; they mirror

#: shipped uniform servo, for the before/after
SHIPPED = dict(kp=500.0, kv=45.0)


def load(scene=True, fast=True):
    """The model the step tests run on.

    `fast` disables CONTACT and the mocap/equality machinery: an arm step
    response involves no contact, and MuJoCo spends most of a step on the
    chassis/wheel contact set. It makes the sweep ~8x faster and changes no
    reported number (checked: the shipped-gain row is identical either way).
    """
    m = mujoco.MjModel.from_xml_path(SCENE if scene else MJCF)
    if fast:
        m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                               | mujoco.mjtDisableBit.mjDSBL_EQUALITY)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.nkey - 1)
    mujoco.mj_forward(m, d)
    return m, d


# NOTE ON GRAVITY COMPENSATION, and why this harness does not add one.
#
# The first version of this sweep applied a gravity feedforward to the fourteen
# arm joints, on the reasoning that MABEL's gravity compensation is a hard
# invariant so the sweep should include it. MEASURED, that made the droop 100x
# WORSE (Arm 1: 2.4 mrad without it, 241 mrad with it), and the reason is worth
# writing down because it is easy to get wrong twice:
#
#   The equation of motion is M(q) qacc = tau - c(q, qd), and M is NOT diagonal.
#   Cancelling the c-row of a SUBSET of joints does not cancel gravity for that
#   subset — it injects a net wrench that the rest of the tree has to absorb. A
#   compensator is only exact over the whole coupled system.
#
# It is also unnecessary here: measured with weak servos (kp = 60) at poses with
# the arm out to the side and the elbow bent, the steady-state error is 2.4 mrad
# — because every arm joint carries frictionloss = 0.5 N.m, and stiction, not
# the servo, is what holds this arm up. So the steady-state error this harness
# reports is the STICTION floor 0.5/kp, and that is the honest quantity: it is
# what actually sets how precisely a commanded pose is held.


def _jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def _dof(m, n):
    return m.jnt_dofadr[_jid(m, n)]


def _qadr(m, n):
    return m.jnt_qposadr[_jid(m, n)]


def save_fig(name):
    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, name), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  figure -> controller/docs/figures/{name}")


def dump(name, obj):
    os.makedirs(RESDIR, exist_ok=True)
    with open(os.path.join(RESDIR, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ── P1: what each joint actually weighs ──────────────────────────────────────
def p1_inertia(n_samples=400, seed=3):
    """Per-joint reflected inertia and stiction over the arm's own workspace.

    The design basis. kp and kd follow from these two numbers and a choice of
    bandwidth/damping; nothing here is a preference.
    """
    print("\n=== P1  INERTIA — the design basis, measured off the MJCF ===")
    m, d = load()
    rs = np.random.default_rng(seed)
    jids = [_jid(m, n) for n in PROBE]
    qadr = np.array([m.jnt_qposadr[j] for j in jids])
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    acc = {n: [] for n in PROBE}
    M = np.zeros((m.nv, m.nv))
    for s_i in range(n_samples):
        d.qpos[qadr] = m.qpos0[qadr] if s_i == 0 else lo + rs.random(len(jids)) * (hi - lo)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, M, d.qM)
        for n in PROBE:
            acc[n].append(M[_dof(m, n), _dof(m, n)])
    out = {}
    print(f"  {'joint':14s}{'M min':>8s}{'M med':>8s}{'M max':>8s}   "
          f"{'armature':>9s}{'friction':>9s}{'F limit':>9s}")
    for n in PROBE:
        a = np.array(acc[n])
        aid = m.actuator(n).id
        fr = m.actuator_forcerange[aid]
        out[n] = dict(m_min=float(a.min()), m_med=float(np.median(a)), m_max=float(a.max()),
                      armature=float(m.dof_armature[_dof(m, n)]),
                      friction=float(m.dof_frictionloss[_dof(m, n)]),
                      f_lim=float(max(abs(fr[0]), abs(fr[1]))))
        o = out[n]
        print(f"  {n:14s}{o['m_min']:8.4f}{o['m_med']:8.4f}{o['m_max']:8.4f}   "
              f"{o['armature']:9.4f}{o['friction']:9.2f}{o['f_lim']:9.1f}")
    print("\n  the shipped uniform servo kp=500 kv=45, at the MEDIAN inertia:")
    print(f"    every joint shares kd/kp = {SHIPPED['kv']/SHIPPED['kp']*1000:.0f} ms of "
          f"tracking lag — inertia does not enter that ratio, which is why one gain\n"
          f"    for fourteen different joints produces one uniform lag and fourteen "
          f"different damping ratios:")
    for n in PROBE:
        Mm = out[n]["m_med"]
        wn = np.sqrt(SHIPPED["kp"] / Mm)
        z = SHIPPED["kv"] / (2.0 * np.sqrt(SHIPPED["kp"] * Mm))
        out[n].update(wn_shipped=float(wn), zeta_shipped=float(z))
        print(f"    {n:14s} w_n {wn:6.1f} rad/s ({wn/2/np.pi:5.2f} Hz)   zeta {z:5.2f}"
              f"{'   <-- overdamped' if z > 1.3 else ''}")
    dump("pdtune_p1.json", out)
    return out


def step_response(m, d0, joint, gains, amp=0.15, t_total=1.2, hold=0.35):
    """One joint stepped by `amp` rad with `gains` applied to the whole arm.

    Returns rise / overshoot / settle / steady-state / peak-force. The whole arm
    is held by its own servos throughout, so this is the response the KINEMATIC
    teleop path actually gets — not an idealised single-link model.

    amp is 0.15 rad on purpose. A PD saturates its actuator on ANY large step —
    the shipped kp=500 demands 175 N.m for a 0.35 rad error against a 40 N.m
    joint — so a big-step test measures the force limit and calls it a gain
    choice. 0.15 rad is the size of error teleop actually produces: the retarget
    trust region caps the command at 2 rad/s, which at a 30 Hz stream is
    0.067 rad per frame, and a re-anchor is a few of those.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = d0.ctrl
    aid = m.actuator(joint).id
    for n, (kp, kv) in gains.items():
        a = m.actuator(n).id
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kv
    qa = _qadr(m, joint)
    lo, hi = m.jnt_range[_jid(m, joint)]

    def run(n_steps, rec=False):
        q = np.zeros(n_steps) if rec else None
        f = np.zeros(n_steps) if rec else None
        for i in range(n_steps):
            mujoco.mj_step(m, d)
            if rec:
                q[i] = d.qpos[qa]
                f[i] = abs(d.actuator_force[aid])
        return q, f

    run(int(hold / m.opt.timestep))                    # settle at the start pose
    q0 = float(d.qpos[qa])
    tgt = float(np.clip(q0 + amp, lo + 0.02, hi - 0.02))
    span = tgt - q0
    if abs(span) < 1e-3:
        return None
    d.ctrl[aid] = tgt
    n_steps = int(t_total / m.opt.timestep)
    t = np.arange(n_steps) * m.opt.timestep
    q, f = run(n_steps, rec=True)
    frac = (q - q0) / span

    def cross(v):
        idx = np.where(frac >= v)[0]
        return float(t[idx[0]]) if len(idx) else float("nan")

    tail = frac[int(0.8 * n_steps):]
    band = max(0.02, 0.002 / abs(span))                # 2 % of the step, floor 2 mrad
    outside = np.where(np.abs(frac - 1.0) > band)[0]
    return dict(rise=float(cross(0.9) - cross(0.1)),
                overshoot=float(max(0.0, (np.max(frac) - 1.0) * 100.0)),
                settle=float(t[outside[-1]]) if len(outside) else 0.0,
                sse=float(abs(1.0 - np.mean(tail)) * abs(span)),
                jitter=float(np.ptp(tail) * abs(span)),
                fpeak=float(np.max(f)),
                f_lim=float(max(abs(m.actuator_forcerange[aid, 0]),
                                abs(m.actuator_forcerange[aid, 1]))),
                t=t, frac=frac)


def gains_for(inertia, wn, zeta, joints=None):
    """kp / kv per joint from a shared bandwidth and damping ratio.

    kd/kp = 2*zeta/w_n comes out the same for every joint by construction, which
    is the point: one tracking lag across the arm, and a damping ratio that is
    right for each joint's own inertia instead of right for none of them.
    """
    g = {}
    for n in (joints or ARM_JOINTS):
        base = n.replace("Left", "Right")
        Mm = inertia[base]["m_med"]
        g[n] = (Mm * wn ** 2, 2.0 * zeta * Mm * wn)
    return g


def wrist_gains(wn, zeta):
    """The ORCA wrist servos, which never got gains at all.

    NOT APPLIED BY DEFAULT (--wrists to opt in), and the reason is a measured
    coupling rather than caution. Stiffening this joint BREAKS external-contact
    force estimation: `test_dynamic_models.py::test_whole_body_base_wrench_and_
    contacts` pushes the palm with 15 N and checks the estimated direction, and
    the estimate degrades from passing at the shipped kp=1 to cos 0.17-0.22 at
    ANY stiffening — including kp=2.0, the value the MJCF's own `hand_base`
    default class intends for these actuators. The palm hangs off this joint, so
    its compliance is what makes a contact observable in the arm's torques.
    Raising it is therefore a coupled decision about the admittance/contact
    features, not a free correction of an oversight.

    Measured: both "<Side> Wrist" position actuators run kp = 1, kv = 0 — MuJoCo's
    bare defaults. The `hand_base` default class in the MJCF specifies kp = 2.0
    and forcerange +/-1, but these actuators are declared outside that class, so
    they inherited nothing. A 1 N.m/rad servo on the joint that carries a 1.74 kg
    hand is held up mostly by the joint's own 0.1 damping, and its tracking lag
    is that damping over that gain = 100 ms — the same lag as the arm, arrived at
    by accident. This is the joint the palm-flip complaint is about.
    """
    m, d = load()
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, M, d.qM)
    out = {}
    for n in WRIST_JOINTS:
        dof = _dof(m, n)
        Mm = float(M[dof, dof])
        # the joint's own viscous damping already contributes to zeta; ask the
        # servo only for the remainder, never for a negative kv
        kd_total = 2.0 * zeta * Mm * wn
        out[n] = (Mm * wn ** 2, max(0.0, kd_total - float(m.dof_damping[dof])))
    return out


# ── P2: sweep the two design numbers ─────────────────────────────────────────
#
# WHAT THE FIRST SWEEP TAUGHT US, and why this one is parameterised differently.
#
#   1. TRACKING LAG IS kd/kp, and nothing else.  On a ramp of rate v the servo
#      settles at an error e = tau_0/kp + (kd/kp) v.  The shipped servo has
#      kd/kp = 45/500 = 90 ms on EVERY joint — inertia does not enter — which is
#      exactly the "the servo contributed 170 ms" measured on the retarget path
#      in July, and exactly what the operator feels as lag.  In the (w_n, zeta)
#      parameterisation that number is 2*zeta/w_n, so "quick to respond" means a
#      HIGH bandwidth and a LOW damping ratio, and the two knobs are not
#      independent of the thing being asked for.
#   2. STEADY-STATE ERROR IS STICTION, not gravity.  Every arm joint carries
#      frictionloss = 0.5 N.m, so the servo parks anywhere inside +/- 0.5/kp.
#      At the low-bandwidth end of the first sweep that was 17 mrad on Arm 5 and
#      it read as "droop" — it is not; gravity at the home pose is already
#      balanced to 0.04 mrad.  It sets a FLOOR on kp: 0.5/0.005 = 100 for 5 mrad.
#   3. "TOO STIFF" HAS TO BE MEASURED, not asserted.  Saturating on a step is
#      not the cost (a big error SHOULD slew at full torque); the cost is that a
#      stiff servo turns command jitter into torque.  So the sweep injects the
#      retarget jitter the arm actually receives and reports the torque it
#      produces.
def p2_sweep(inertia, quick=False, jitter_mrad=2.0):
    print("\n=== P2  SWEEP — bandwidth x damping, measured on the model ===")
    wns = (20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0) if not quick else (30.0, 60.0, 90.0)
    zetas = (0.7, 0.85, 1.0, 1.3) if not quick else (0.85, 1.0)
    mm, dd0 = load()
    dd0.ctrl[:] = 0.0
    dd0.ctrl[mm.actuator("Lift").id] = 0.3175
    rows = []

    def evaluate(g, label):
        per = [r for r in (step_response(mm, dd0, j, g) for j in PROBE) if r]
        lag = ramp_lag(mm, dd0, g)
        noise = hold_noise(mm, dd0, g, jitter=jitter_mrad * 1e-3)
        row = dict(label=label,
                   lag_ms=lag * 1000.0,
                   rise=float(np.median([p["rise"] for p in per])) * 1000.0,
                   settle=float(np.median([p["settle"] for p in per])) * 1000.0,
                   overshoot=float(np.max([p["overshoot"] for p in per])),
                   sse=float(np.max([p["sse"] for p in per])) * 1000.0,
                   tau_rms=noise,
                   headroom=float(np.min([1.0 - p["fpeak"] / p["f_lim"] for p in per])) * 100.0)
        return row

    for wn in wns:
        for z in zetas:
            row = evaluate(gains_for(inertia, wn, z), f"wn{wn:.0f}_z{z:.2f}")
            row.update(wn=wn, zeta=z)
            rows.append(row)
            print(f"  w_n {wn:5.1f} ({wn/2/np.pi:4.1f} Hz) zeta {z:4.2f} | "
                  f"LAG {row['lag_ms']:6.1f} ms | rise {row['rise']:5.0f} ms  "
                  f"settle {row['settle']:5.0f} ms  overshoot {row['overshoot']:5.1f}%  "
                  f"sse {row['sse']:5.1f} mrad | jitter-torque {row['tau_rms']:5.1f}% of F_lim")

    g_ship = {n: (SHIPPED["kp"], SHIPPED["kv"]) for n in ARM_JOINTS}
    ship = evaluate(g_ship, "shipped")
    print(f"\n  SHIPPED kp=500 kv=45 (uniform)          | "
          f"LAG {ship['lag_ms']:6.1f} ms | rise {ship['rise']:5.0f} ms  "
          f"settle {ship['settle']:5.0f} ms  overshoot {ship['overshoot']:5.1f}%  "
          f"sse {ship['sse']:5.1f} mrad | jitter-torque {ship['tau_rms']:5.1f}% of F_lim")

    # PICK: the lowest tracking lag that does not ring, parks accurately, and
    # does not pump torque into the retarget jitter. The jitter-torque ceiling is
    # the shipped servo's own value: the arm is already known to be quiet at it,
    # so a gain set that stays under it cannot be "too stiff" in the sense that
    # matters, and one that exceeds it is buying speed with vibration.
    ok = [r for r in rows if r["overshoot"] <= 5.0 and r["sse"] <= 5.0
          and r["tau_rms"] <= ship["tau_rms"]]
    pick = min(ok, key=lambda r: r["lag_ms"]) if ok else min(rows, key=lambda r: r["lag_ms"])
    print(f"\n  PICK  w_n {pick['wn']:.1f} rad/s ({pick['wn']/2/np.pi:.1f} Hz)  "
          f"zeta {pick['zeta']:.2f}   ->  lag {pick['lag_ms']:.1f} ms "
          f"(shipped {ship['lag_ms']:.1f} ms), overshoot {pick['overshoot']:.1f}%, "
          f"sse {pick['sse']:.1f} mrad, jitter-torque {pick['tau_rms']:.1f}% "
          f"of F_lim (shipped {ship['tau_rms']:.1f}%)")

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.4))
    for z in zetas:
        sel = [r for r in rows if r["zeta"] == z]
        x = [r["wn"] for r in sel]
        axes[0].plot(x, [r["lag_ms"] for r in sel], "o-", label=f"zeta={z}")
        axes[1].plot(x, [r["overshoot"] for r in sel], "o-", label=f"zeta={z}")
        axes[2].plot(x, [r["sse"] for r in sel], "o-", label=f"zeta={z}")
        axes[3].plot(x, [r["tau_rms"] for r in sel], "o-", label=f"zeta={z}")
    for ax, key, ylab, ref, title in (
            (axes[0], "lag_ms", "tracking lag  [ms]", ship["lag_ms"],
             "P2a — lag = kd/kp (what teleop feels)"),
            (axes[1], "overshoot", "overshoot  [%]", 5.0, "P2b — ringing"),
            (axes[2], "sse", "steady-state error  [mrad]", 5.0, "P2c — stiction floor"),
            (axes[3], "tau_rms", "torque RMS on 2 mrad jitter  [% of F_lim]", ship["tau_rms"],
             "P2d — the cost of stiffness")):
        ax.axhline(ref, color="0.4", ls="--", lw=1)
        ax.set_xlabel("bandwidth w_n  [rad/s]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].scatter([pick["wn"]], [pick["lag_ms"]], s=90, facecolors="none",
                    edgecolors="k", zorder=5)
    fig.tight_layout()
    save_fig("pdtune_sweep.pdf")
    dump("pdtune_p2.json", dict(rows=rows, shipped=ship, pick=pick))
    return pick, ship


def ramp_lag(m, d0, gains, rate=2.0, t_total=1.4, joint="Right Arm 4"):
    """Measured tracking lag on a constant-rate ramp — the teleop quantity.

    The retarget trust region hands the servo a ramp, not a step, so this is the
    error the operator actually sees. It converges to (kd/kp)*rate plus the
    stiction term, which is why kd/kp is the number to design against.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = d0.ctrl
    for n, (kp, kv) in gains.items():
        a = m.actuator(n).id
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kv
    aid = m.actuator(joint).id
    qa = _qadr(m, joint)
    lo, hi = m.jnt_range[_jid(m, joint)]
    # start at the bottom of the range and ramp up, sized so the command never
    # reaches the stop inside the window — a clipped ramp stops being a ramp and
    # the "lag" then measures the settled stiction offset instead (it even comes
    # out negative, which is how this bug announced itself).
    span = min(rate * t_total, (hi - lo) - 0.10)
    t_total = span / rate
    q0 = float(lo + 0.05)
    d.ctrl[aid] = q0
    for _ in range(int(0.30 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    n_steps = int(t_total / m.opt.timestep)
    err = np.zeros(n_steps)
    for i in range(n_steps):
        u = q0 + rate * i * m.opt.timestep
        d.ctrl[aid] = u
        mujoco.mj_step(m, d)
        err[i] = u - d.qpos[qa]
    return float(np.median(err[int(0.5 * n_steps):]) / rate)      # seconds of lag


def hold_noise(m, d0, gains, jitter=2e-3, t_total=1.0, seed=5):
    """Torque RMS produced by holding a pose against a noisy command.

    This is the honest price of stiffness. The retargeting output is not
    perfectly still — a few mrad of command jitter is normal — and a servo's
    stiffness is exactly the gain from that jitter to motor torque. A gain set
    that is "snappy" only because it turns 2 mrad of noise into a large fraction
    of the joint's torque budget is not snappy, it is loud.

    Returned as the worst joint's torque RMS as a PERCENTAGE of its own
    actuator's force limit.
    """
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = d0.ctrl
    for n, (kp, kv) in gains.items():
        a = m.actuator(n).id
        m.actuator_gainprm[a, 0] = kp
        m.actuator_biasprm[a, 1] = -kp
        m.actuator_biasprm[a, 2] = -kv
    rs = np.random.default_rng(seed)
    aids = [m.actuator(j).id for j in PROBE]
    lim = np.array([max(abs(m.actuator_forcerange[a, 0]), abs(m.actuator_forcerange[a, 1]))
                    for a in aids])
    base = np.array([d.ctrl[a] for a in aids])
    n_steps = int(t_total / m.opt.timestep)
    tau = np.zeros((n_steps, len(aids)))
    for i in range(n_steps):
        for c, a in enumerate(aids):
            d.ctrl[a] = base[c] + jitter * rs.standard_normal()
        mujoco.mj_step(m, d)
        tau[i] = [abs(d.actuator_force[a]) for a in aids]
    rms = np.sqrt(np.mean(tau[int(0.3 * n_steps):] ** 2, axis=0))
    # As a FRACTION of each joint's own actuator: 1 N.m of noise torque is
    # nothing on the 40 N.m shoulder and 14 % of full scale on a 7 N.m wrist.
    # Reporting newtons picks the heaviest joint every time and hides the joint
    # that is actually being driven hard — which, for the shipped uniform gain,
    # is the distal one.
    return float(np.max(rms / lim) * 100.0)


# ── P3: the chosen gains, against the shipped ones ───────────────────────────
def p3_verify(inertia, wn, zeta):
    print(f"\n=== P3  VERIFY — per-joint w_n={wn:.0f} rad/s zeta={zeta:.2f} "
          f"vs the shipped uniform 500/45 ===")
    g_new = gains_for(inertia, wn, zeta)
    g_old = {n: (SHIPPED["kp"], SHIPPED["kv"]) for n in ARM_JOINTS}
    print(f"  {'joint':14s}{'kp new':>9s}{'kv new':>9s}   {'zeta old':>9s}{'zeta new':>9s}")
    for n in PROBE:
        Mm = inertia[n]["m_med"]
        z_old = SHIPPED["kv"] / (2.0 * np.sqrt(SHIPPED["kp"] * Mm))
        print(f"  {n:14s}{g_new[n][0]:9.1f}{g_new[n][1]:9.2f}   {z_old:9.2f}{zeta:9.2f}")

    curves, res = {}, {}
    mm, dd0 = load()
    dd0.ctrl[:] = 0.0
    dd0.ctrl[mm.actuator("Lift").id] = 0.3175
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for tag, g, ls in (("shipped", g_old, "--"), ("tuned", g_new, "-")):
        rr = [(j, r) for j, r in ((j, step_response(mm, dd0, j, g)) for j in PROBE) if r]
        res[tag] = dict(
            rise=float(np.median([r["rise"] for _, r in rr])) * 1000,
            settle=float(np.median([r["settle"] for _, r in rr])) * 1000,
            overshoot=float(np.max([r["overshoot"] for _, r in rr])),
            droop=float(np.max([r["sse"] for _, r in rr])) * 1000,
            headroom=float(np.min([1.0 - r["fpeak"] / r["f_lim"] for _, r in rr])) * 100)
        curves[tag] = rr
        o = res[tag]
        print(f"  {tag:8s} rise {o['rise']:6.1f} ms   settle {o['settle']:6.0f} ms   "
              f"overshoot {o['overshoot']:5.1f}%   droop {o['droop']:5.1f} mrad   "
              f"headroom {o['headroom']:5.1f}%")
        for ax, jname in ((axes[0], "Right Arm 1"), (axes[1], "Right Arm 5")):
            for j, r in rr:
                if j == jname:
                    ax.plot(r["t"] * 1000, r["frac"], ls, label=f"{tag}")
    for ax, jname in ((axes[0], "Right Arm 1  (shoulder, M=0.54)"),
                      (axes[1], "Right Arm 5  (forearm roll, M=0.053)")):
        ax.axhline(1.0, color="0.6", lw=0.8)
        ax.axhspan(0.98, 1.02, color="0.9", zorder=0)
        ax.set_xlabel("time  [ms]")
        ax.set_ylabel("step response  [fraction]")
        ax.set_title(jname, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("P3 — same step, per-joint gains vs one uniform gain", fontsize=11)
    fig.tight_layout()
    save_fig("pdtune_step.pdf")
    dump("pdtune_p3.json", res)
    return res, g_new


# ── P4: does the arm still TRACK? (the reason the gains were raised in July) ──
def p4_tracking(inertia, wn, zeta, hz=0.5, t_total=6.0):
    print("\n=== P4  TRACKING — a moving target, not a step: lag and RMS ===")
    g_new = gains_for(inertia, wn, zeta)
    g_old = {n: (SHIPPED["kp"], SHIPPED["kv"]) for n in ARM_JOINTS}
    out = {}
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    js = ["Right Arm 2", "Right Arm 4", "Right Arm 5"]
    for tag, g, ls in (("shipped", g_old, "--"), ("tuned", g_new, "-")):
        mm, dd = load()
        dd.ctrl[:] = 0.0
        dd.ctrl[mm.actuator("Lift").id] = 0.3175
        for n, (kp, kv) in g.items():
            a = mm.actuator(n).id
            mm.actuator_gainprm[a, 0] = kp
            mm.actuator_biasprm[a, 1] = -kp
            mm.actuator_biasprm[a, 2] = -kv
            aid = [mm.actuator(j).id for j in js]
        qad = [_qadr(mm, j) for j in js]
        n_steps = int(t_total / mm.opt.timestep)
        t = np.arange(n_steps) * mm.opt.timestep
        cmd = np.zeros((n_steps, len(js)))
        act = np.zeros((n_steps, len(js)))
        for i in range(n_steps):
            u = 0.45 * np.sin(2 * np.pi * hz * t[i])
            for c, a in enumerate(aid):
                dd.ctrl[a] = u
                cmd[i, c] = u
            mujoco.mj_step(mm, dd)
            act[i] = dd.qpos[qad]
        skip = int(1.0 / mm.opt.timestep)
        err = act[skip:] - cmd[skip:]
        a0 = act[skip:, 1] - act[skip:, 1].mean()
        c0 = cmd[skip:, 1] - cmd[skip:, 1].mean()
        xc = np.correlate(a0, c0, mode="full")
        lag = (np.argmax(xc) - (len(c0) - 1)) * mm.opt.timestep
        out[tag] = dict(rms_mrad=float(np.sqrt(np.mean(err ** 2)) * 1000),
                        peak_mrad=float(np.max(np.abs(err)) * 1000),
                        lag_ms=float(lag * 1000))
        print(f"  {tag:8s} tracking RMS {out[tag]['rms_mrad']:5.1f} mrad   "
              f"peak {out[tag]['peak_mrad']:5.1f} mrad   lag {out[tag]['lag_ms']:+6.1f} ms")
        ax.plot(t[skip:skip + 1500], act[skip:skip + 1500, 1], ls, label=f"{tag}")
        _cmd = cmd
    ax.plot(t[skip:skip + 1500], _cmd[skip:skip + 1500, 1], ":", color="0.3", label="command")
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("Right Arm 4  [rad]")
    ax.set_title(f"P4 — {hz} Hz joint sweep: the tuned servo tracks at least as well",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig("pdtune_tracking.pdf")
    dump("pdtune_p4.json", out)
    return out


# ── apply: rewrite the MJCF servo gains ──────────────────────────────────────
def apply_gains(inertia, wn, zeta, dry=False, wrists=False):
    print(f"\n=== APPLY — writing per-joint gains into the MJCF "
          f"(w_n={wn:.0f}, zeta={zeta:.2f}) ===")
    g = dict(gains_for(inertia, wn, zeta))
    if wrists:
        g.update(wrist_gains(wn, zeta))
    src = open(MJCF).read()
    n = 0
    for name, (kp, kv) in g.items():
        have = re.compile(rf'(<position name="{re.escape(name)}"[^>]*?)kp="[\d.]+" kv="[\d.]+"')
        new_src, k = have.subn(
            lambda mo, _kp=kp, _kv=kv: f'{mo.group(1)}kp="{_kp:.1f}" kv="{_kv:.2f}"', src)
        if k == 0:                       # the ORCA wrists carry no gains at all
            bare = re.compile(rf'(<position name="{re.escape(name)}"\s+joint="[^"]+")')
            new_src, k = bare.subn(
                lambda mo, _kp=kp, _kv=kv: f'{mo.group(1)} kp="{_kp:.1f}" kv="{_kv:.2f}"', src)
        if k != 1:
            print(f"  !! {name}: {k} matches — NOT written")
            continue
        src = new_src
        n += 1
        print(f"     {name:16s} kp {kp:8.1f}  kv {kv:7.2f}")
    print(f"  rewrote {n}/{len(g)} servos")
    if dry:
        print("  (dry run: nothing written)")
        return
    open(MJCF, "w").write(src)
    print(f"  wrote {MJCF}")
    print("  NOTE: actuator gains are not geometry — the MJCF -> URDF -> GLB -> rig\n"
          "  chain carries no gains, so no derived artefact needs regenerating.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write the chosen gains to the MJCF")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--wrists", action="store_true",
                    help="also tune the ORCA wrist servos — BREAKS contact-force "
                         "estimation, see wrist_gains()")
    ap.add_argument("--wn", type=float, default=None, help="override the chosen bandwidth")
    ap.add_argument("--zeta", type=float, default=None)
    args = ap.parse_args()
    todo = ["P1", "P2", "P3", "P4"] if (args.all or not args.phase) else \
        [p.upper() for p in args.phase]
    t0 = time.perf_counter()
    inertia = p1_inertia(n_samples=120 if args.quick else 400)
    wn, zeta = args.wn, args.zeta
    needs_pick = any(p in todo for p in ("P2", "P3", "P4")) or args.apply
    if "P2" in todo or (needs_pick and wn is None):
        pick, _ = p2_sweep(inertia, quick=args.quick)
        wn = wn if wn is not None else pick["wn"]
        zeta = zeta if zeta is not None else pick["zeta"]
    if "P3" in todo and wn is not None:
        p3_verify(inertia, wn, zeta)
    if "P4" in todo and wn is not None:
        p4_tracking(inertia, wn, zeta)
    if args.apply and wn is not None:
        apply_gains(inertia, wn, zeta, dry=args.dry, wrists=args.wrists)
    print(f"\nDone in {time.perf_counter() - t0:.0f}s.")


if __name__ == "__main__":
    main()

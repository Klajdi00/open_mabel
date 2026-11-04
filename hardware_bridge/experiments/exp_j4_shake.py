#!/usr/bin/env python3
"""exp_j4_shake.py — staged, watchdogged hardware diagnosis of the l_j4 torque-on shake.

The report: "J4 shakes crazily when I torque on via real_run.sh". This experiment owns
the arms Teensy EXCLUSIVELY (stop the HAL and the setup studio first — last-Hello-wins
firmware) and walks the real_run command stack one layer at a time, so the trace shows
WHERE the shake enters:

  1  BASELINE     torque off — telemetry noise floor, wrap proximity, latched DM faults
  2  MATH AUDIT   no energize — the calibrated fold round-trip + the gravity-FF value
                  and sign for the live pose (what real_run would command)
  3  HOLD (PD)    l_j4 ONLY: engage at the measured pose with the deployed per-joint
                  gains — pure kp/kd, no feedforward. 8 s trace.
  4  HOLD + FF    same + the model-frame gravity t_ff mapped exactly as the HAL maps it
                  (dir × model). 8 s trace.
  5  WHOLE ARM    the actual real_run condition: all seven left joints held at their
                  measured pose (per-joint gains + gravity FF). 8 s trace.

Every energized stage runs a WATCHDOG per sample: |pos − pos₀| > ABORT_POS or
|vel| > ABORT_VEL on ANY held joint → immediate damp + abort. Metrics per stage:
peak-to-peak position (deg), RMS velocity, mean/peak |τ|, dominant oscillation
frequency (velocity zero-crossing rate).

Usage (mabel env, robot powered, HAL + studio stopped):
    python3 experiments/exp_j4_shake.py            # stages 1-2 (never energizes)
    python3 experiments/exp_j4_shake.py --energize # stages 1-5 (holds at measured pose)
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

HB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HB))
sys.path.insert(0, str(HB.parent / "controller"))

from mabel_hw import config                                    # noqa: E402
from mabel_hw.calibration import CalibrationStore              # noqa: E402
from mabel_hw.calibration.schema import JointCalib             # noqa: E402
from mabel_hw.drivers.openarm_link import OpenArmLink          # noqa: E402

LEFT = [f"l_j{i}" for i in range(1, 8)]
J = "l_j4"
ABORT_POS = 0.35          # rad from the engage pose → abort (≈20°)
ABORT_VEL = 3.0           # rad/s → abort
STAGE_S = 8.0
HZ = 200.0                # sampling (the link streams commands on its own thread)


def wait(pred, timeout=6.0, dt=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


def snap(link, idx):
    st = link.state()
    f = st.joints[idx]
    return f.pos, f.vel, f.tau


def record(link, held, idx_of, seconds, label, extra_note=""):
    """Sample all HELD joints for `seconds`; watchdog aborts by damping the slice.
    Returns (aborted, {joint: trace}) where trace = list of (t, pos, vel, tau)."""
    traces = {j: [] for j in held}
    start = {j: snap(link, idx_of[j])[0] for j in held}
    t0 = time.monotonic()
    aborted = None
    while time.monotonic() - t0 < seconds:
        t = time.monotonic() - t0
        for j in held:
            p, v, tau = snap(link, idx_of[j])
            traces[j].append((t, p, v, tau))
            if abs(p - start[j]) > ABORT_POS or abs(v) > ABORT_VEL:
                aborted = (j, p, v)
        if aborted:
            link.damp_slice(list(idx_of.values()), kd=1.0)   # NOW, not next frame
            print(f"  ✗ WATCHDOG ABORT [{label}]: {aborted[0]} pos={aborted[1]:+.3f} "
                  f"vel={aborted[2]:+.3f} — slice damped")
            break
        time.sleep(1.0 / HZ)
    return aborted, traces


def metrics(trace):
    ts = [s[0] for s in trace]
    ps = [s[1] for s in trace]
    vs = [s[2] for s in trace]
    taus = [s[3] for s in trace]
    pp = (max(ps) - min(ps)) if ps else 0.0
    rms_v = math.sqrt(sum(v * v for v in vs) / len(vs)) if vs else 0.0
    # dominant frequency ~ velocity zero-crossing rate / 2
    zc = sum(1 for a, b in zip(vs, vs[1:]) if a * b < 0)
    dur = ts[-1] - ts[0] if len(ts) > 1 else 1.0
    freq = zc / (2.0 * dur) if dur > 0 else 0.0
    return dict(pp_deg=math.degrees(pp), rms_vel=rms_v, f_hz=freq,
                tau_mean=statistics.fmean(abs(t) for t in taus) if taus else 0.0,
                tau_peak=max((abs(t) for t in taus), default=0.0))


def report(label, m, verdict=""):
    print(f"  {label:16s} p-p {m['pp_deg']:6.2f}°  vRMS {m['rms_vel']:6.3f} rad/s  "
          f"f≈{m['f_hz']:5.1f} Hz  |τ| mean {m['tau_mean']:5.2f} / peak {m['tau_peak']:5.2f} N·m"
          + (f"   {verdict}" if verdict else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energize", action="store_true",
                    help="run the energized stages 3-5 (holds at the measured pose)")
    args = ap.parse_args()

    print("── stage 0: exclusive link ─────────────────────────────────────────")
    link = OpenArmLink.get(config.LINKS["arms_openarm"]["ip"],
                           config.LINKS["arms_openarm"]["port"],
                           config.LINKS["arms_openarm"]["stream_hz"],
                           logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    specs = link.specs()
    idx_of = {s.name: i for i, s in enumerate(specs)}
    assert J in idx_of, f"{J} not in manifest"
    time.sleep(2.5)                                     # let health/temps land

    print("── stage 1: baseline (torque off) ──────────────────────────────────")
    temps = link.temps()
    for j in LEFT:
        i = idx_of[j]
        p, v, tau = snap(link, i)
        t = temps[i] if i < len(temps) else (None, None)
        print(f"  {j}: pos {p:+8.4f} rad ({math.degrees(p):+7.1f}°)  vel {v:+.3f}  "
              f"tau {tau:+.3f}  t_mos {t[0]}")
    _, base = record(link, [J], idx_of, 3.0, "baseline")
    bm = metrics(base[J])
    report("baseline", bm)
    p0 = base[J][-1][1]
    print(f"  wrap headroom: p_max 12.5 − pos {p0:.3f} = {12.5 - p0:.3f} rad")

    print("── stage 2: math audit (never energizes) ───────────────────────────")
    store = CalibrationStore().load()
    cal = store.slice("arm/openarm-dual/calib") or {}
    jc = JointCalib.from_dict(cal[J])
    model = jc.to_model(p0)
    back = jc.to_raw(model, cur=p0)
    print(f"  {J}: measured raw {p0:+.4f} → model {model:+.4f} → to_raw(cur) {back:+.4f}"
          f"   round-trip err {abs(back - p0):.5f} rad "
          + ("✓" if abs(back - p0) < 0.02 else "✗ FOLD BUG"))
    clamped = jc.clamp_raw(p0, cur=p0)
    print(f"  clamp_raw(measured) = {clamped:+.4f}  (Δ {abs(clamped - p0):.5f} rad "
          + ("✓ hold would not move)" if abs(clamped - p0) < 0.02 else "✗ WOULD YANK)"))
    # gravity FF exactly as real_run computes it (model frame, HAL applies dir)
    tau_ff = None
    try:
        from mabel.arm_control import ArmGravityCompensator
        mjcf = str(HB.parent / "simulation/mabel_mujoco/models/mabel_full.xml")
        g = ArmGravityCompensator(mjcf, {}, {}, log=lambda *a, **k: None)
        pose = {}
        for j in LEFT + [f"r_j{i}" for i in range(1, 8)]:
            e = cal.get(j)
            if e:
                pose[j] = JointCalib.from_dict(e).to_model(snap(link, idx_of[j])[0])
        tff = g.tff(pose, frame="model")
        tau_ff = float(tff.get(J, 0.0))
        d = int(cal[J].get("dir", 1))
        print(f"  gravity FF @pose: model {tau_ff:+.3f} N·m → raw (×dir {d:+d}) "
              f"{d * tau_ff:+.3f} N·m   (measured passive tau {bm['tau_mean']:.3f})")
    except Exception as e:
        print(f"  gravity FF audit skipped: {e}")

    if not args.energize:
        print("── done (add --energize for stages 3-5) ────────────────────────")
        link.close()
        return

    kp_map = dict(config.LEFT_ARM["kp"])
    kd_map = dict(config.LEFT_ARM["kd"])
    print(f"── stage 3: {J} PD hold at measured (kp {kp_map.get(J)} kd {kd_map.get(J)}) ─")
    g1 = link.group([J], kp=kp_map, kd=kd_map)
    assert wait(lambda: g1.resolved), "group unresolved"
    g1.set_torque(True)
    ab, tr = record(link, [J], idx_of, STAGE_S, "PD hold")
    m3 = metrics(tr[J])
    report("PD hold", m3, "SHAKE" if m3["pp_deg"] > 2.0 or m3["rms_vel"] > 0.3 else "stable")
    g1.set_torque(False)
    time.sleep(1.0)

    print(f"── stage 4: {J} PD hold + gravity feedforward ──────────────────────")
    if tau_ff is not None and not ab:
        d = int(cal[J].get("dir", 1))
        p_now = snap(link, idx_of[J])[0]
        g1.set_torque(True)
        g1.command({J: p_now}, effort={J: d * tau_ff})     # dir-mapped, like the HAL
        ab4, tr4 = record(link, [J], idx_of, STAGE_S, "PD+FF hold")
        m4 = metrics(tr4[J])
        report("PD+FF hold", m4,
               "SHAKE" if m4["pp_deg"] > 2.0 or m4["rms_vel"] > 0.3 else "stable")
        g1.set_torque(False)
        time.sleep(1.0)
    else:
        print("  skipped (no FF value or prior abort)")

    print("── stage 5: WHOLE LEFT ARM hold (the real_run torque-on condition) ─")
    if not ab:
        g7 = link.group(LEFT, kp=kp_map, kd=kd_map)
        assert wait(lambda: g7.resolved)
        g7.set_torque(True)
        if tau_ff is not None:
            eff = {}
            for j in LEFT:
                e = cal.get(j)
                if e and j in pose:
                    eff[j] = int(e.get("dir", 1)) * float(tff.get(j, 0.0))
            pos_now = {j: snap(link, idx_of[j])[0] for j in LEFT}
            g7.command(pos_now, effort=eff)
        ab5, tr5 = record(link, LEFT, idx_of, STAGE_S, "whole-arm")
        for j in LEFT:
            mj = metrics(tr5[j])
            report(f"arm hold {j}", mj,
                   "SHAKE" if mj["pp_deg"] > 2.0 or mj["rms_vel"] > 0.3 else "")
        g7.set_torque(False)
    else:
        print("  skipped (prior abort)")

    time.sleep(0.5)
    temps = link.temps()
    i = idx_of[J]
    print(f"  post-run {J} t_mos: {temps[i][0] if i < len(temps) else '?'} °C")
    link.close()
    print("── done — all slices damped ────────────────────────────────────────")


if __name__ == "__main__" and not any(f in sys.argv for f in ("--run2", "--run3", "--run4", "--run5")):
    main()


# ── deep probes (run 2, post power-cycle): err nibbles + signed τ + step response ──
class HealthLink(OpenArmLink):
    """OpenArmLink + the DM error nibble from Health (f1 err — the base link only
    keeps temps). 1 = enabled-OK, 0x8..0xE = real faults (0x9 = undervoltage)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._errs = []

    def errs(self):
        with self._lock:
            return list(self._errs)

    def _on_health(self, body):
        super()._on_health(body)
        errs = []
        from mabel_hw.drivers.openarm_link import _fields
        for f, wt, v in _fields(body):
            if f == 3 and wt == 2:
                e = 0
                for sf, _swt, sv in _fields(v):
                    if sf == 1:
                        e = sv
                errs.append(e)
        if errs:
            with self._lock:
                self._errs = errs


def signed_stats(trace):
    taus = [s[3] for s in trace]
    return (statistics.fmean(taus), min(taus), max(taus)) if taus else (0, 0, 0)


def step_response(link, group, idx_of, j, step_rad, label):
    """Small position step from the CURRENT pose; record 2.5 s; return + report.
    The direct probe for shake-under-motion: overshoot + ring count = damping health."""
    p0 = snap(link, idx_of[j])[0]
    tgt = p0 + step_rad
    group.set_targets({j: tgt})
    _, tr = record(link, [j], idx_of, 2.5, label)
    ps = [s[1] for s in tr[j]]
    vs = [s[2] for s in tr[j]]
    final = ps[-1]
    err0 = tgt - p0
    overshoot = (max(ps) - tgt) / abs(err0) * 100 if step_rad > 0 else \
                (tgt - min(ps)) / abs(err0) * 100
    rings = sum(1 for a, b in zip(vs, vs[1:]) if a * b < 0)
    tau_m, tau_lo, tau_hi = signed_stats(tr[j])
    print(f"  {label:22s} step {math.degrees(step_rad):+5.1f}° → reached "
          f"{math.degrees(final - p0):+5.2f}°  overshoot {overshoot:5.1f}%  "
          f"vel zero-x {rings:3d}  τ [{tau_lo:+.2f}..{tau_hi:+.2f}] mean {tau_m:+.2f}")
    group.set_targets({j: final})     # park where it settled — no snap-back
    time.sleep(0.3)
    return overshoot, rings


def main2():
    print("══ RUN 2: post power-cycle deep probes ═════════════════════════════")
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    specs = link.specs()
    idx_of = {s.name: i for i, s in enumerate(specs)}
    time.sleep(3.0)

    print("── manifest after the SD update ────────────────────────────────────")
    for s in specs:
        if s.name in ("torso", "neck_yaw", "l_j4", "l_j3"):
            print(f"  {s.name:9s} bus {s.bus} id {s.mid}  t_max {s.t_max:g}  motor {s.motor}")
    errs = link.errs()
    bad = {specs[i].name: e for i, e in enumerate(errs) if e >= 8}
    print(f"  DM error nibbles: {'ALL CLEAR' if not bad else bad}")

    store = CalibrationStore().load()
    cal = store.slice("arm/openarm-dual/calib") or {}
    jc = JointCalib.from_dict(cal[J])
    p0 = snap(link, idx_of[J])[0]
    print(f"  {J} boots at raw {p0:+.4f} ({math.degrees(p0):+.1f}°); window "
          f"[{cal[J]['raw_min']}, {cal[J]['raw_max']}]  in-window: "
          f"{cal[J]['raw_min'] <= p0 <= cal[J]['raw_max']}")

    kp_map = dict(config.LEFT_ARM["kp"])
    kd_map = dict(config.LEFT_ARM["kd"])

    print(f"── {J}: PD hold — SIGNED τ + kp-from-sag identification ────────────")
    g = link.group([J], kp=kp_map, kd=kd_map)
    assert wait(lambda: g.resolved)
    g.set_torque(True)
    time.sleep(1.5)                                   # settle to equilibrium
    _, tr = record(link, [J], idx_of, 4.0, "PD hold")
    m = metrics(tr[J])
    tau_m, tau_lo, tau_hi = signed_stats(tr[J])
    p_eq = statistics.fmean(s[1] for s in tr[J])
    sag = p0 - p_eq
    report("PD hold", m)
    print(f"  signed τ [{tau_lo:+.2f} .. {tau_hi:+.2f}] mean {tau_m:+.2f} N·m")
    print(f"  sag from engage {math.degrees(sag):+.2f}° → kp×sag = "
          f"{kp_map.get(J, 30) * sag:+.2f} N·m of real PD torque")
    if abs(tau_m) > 1e-3 and abs(sag) > 1e-4:
        k = tau_m / (kp_map.get(J, 30) * sag) if kp_map.get(J, 30) * sag != 0 else float("nan")
        print(f"  ⇒ reported/real torque ratio ≈ {k:+.1f} (≈1 = telemetry true; "
              "≫1 = t_max register < SD json)")
    errs = link.errs()
    e4 = errs[idx_of[J]] if idx_of[J] < len(errs) else None
    print(f"  {J} err nibble during hold: 0x{e4:X}" if e4 is not None else "  (no health)")

    print(f"── step responses (±3°): {J} vs healthy twin l_j3 ──────────────────")
    step_response(link, g, idx_of, J, +0.05, f"{J} up")
    step_response(link, g, idx_of, J, -0.05, f"{J} down")
    g.set_torque(False)
    time.sleep(0.8)
    g3 = link.group(["l_j3"], kp=kp_map, kd=kd_map)
    assert wait(lambda: g3.resolved)
    g3.set_torque(True)
    time.sleep(1.0)
    step_response(link, g3, idx_of, "l_j3", +0.05, "l_j3 up (reference)")
    step_response(link, g3, idx_of, "l_j3", -0.05, "l_j3 down (reference)")
    g3.set_torque(False)
    link.close()
    print("══ run 2 done — all slices damped ══════════════════════════════════")


if __name__ == "__main__" and "--run2" in sys.argv:
    sys.argv.remove("--run2")
    main2()
    sys.exit(0)


def torque_ramp(link, idx_of, j, kd, direction, cap_nm=8.0, step_nm=0.2, dwell=0.4):
    """Break-loose identification: kp=0, small kd, ramp t_ff until the joint MOVES.
    Watchdog cuts torque the moment it breaks loose (|Δpos| > 0.10 rad or |vel| > 0.8).
    Returns the commanded t_ff at break-loose (None = never moved below cap)."""
    g = link.group([j], kp=0.0, kd=kd)
    assert wait(lambda: g.resolved)
    p0 = snap(link, idx_of[j])[0]
    g.set_torque(True)                       # latches hold at measured — kp=0, no spring
    broke = None
    tff = 0.0
    try:
        while tff < cap_nm:
            tff += step_nm
            g.command({j: p0}, effort={j: direction * tff})
            t0 = time.monotonic()
            while time.monotonic() - t0 < dwell:
                p, v, _tau = snap(link, idx_of[j])
                if abs(p - p0) > 0.10 or abs(v) > 0.8:
                    broke = tff
                    break
                time.sleep(0.01)
            if broke:
                break
    finally:
        g.command({j: snap(link, idx_of[j])[0]}, effort={j: 0.0})
        g.set_torque(False)
    print(f"  {j}: break-loose at commanded t_ff = "
          + (f"{direction * broke:+.1f} N·m" if broke else f">±{cap_nm:g} N·m (never moved)"))
    time.sleep(0.8)
    return broke


def main3():
    print("══ RUN 3: break-loose torque identification (kp=0 ramp) ════════════")
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    specs = link.specs()
    idx_of = {s.name: i for i, s in enumerate(specs)}
    time.sleep(2.0)
    for j in ("l_j4", "l_j3"):
        p = snap(link, idx_of[j])[0]
        print(f"  {j} at raw {p:+.4f} ({math.degrees(p):+.1f}°)")
    # +raw on l_j4 is the ANTI-GRAVITY direction at this pose (dir=-1; FF audit showed
    # +1.55 N·m raw holds it) and has ~2 rad of window headroom. l_j3: +raw toward its
    # upper window edge (~1.9 rad of room).
    b4 = torque_ramp(link, idx_of, "l_j4", kd=0.5, direction=+1)
    b3 = torque_ramp(link, idx_of, "l_j3", kd=0.5, direction=+1)
    if b4 and b3:
        print(f"  ⇒ break-loose ratio l_j4/l_j3 = {b4 / b3:.2f} "
              "(≈1 = same effective torque scale; ≫1 = J4's torque path weak)")
    link.close()
    print("══ run 3 done — all slices damped ══════════════════════════════════")


if __name__ == "__main__" and "--run3" in sys.argv:
    main3()
    sys.exit(0)


def main4():
    """Gain sweep on l_j4: stick-slip says dead-zone = stiction/kp (±3.4° at kp 30.5,
    measured break-loose 1.8 N·m). Sweep kp/kd with gravity FF on; measure the hold
    limit cycle AND a ±3° step's tracking. The winner ships to control_param.json."""
    print("══ RUN 4: l_j4 gain sweep (hold limit-cycle + step tracking) ═══════")
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    specs = link.specs()
    idx_of = {s.name: i for i, s in enumerate(specs)}
    time.sleep(2.0)

    store = CalibrationStore().load()
    cal = store.slice("arm/openarm-dual/calib") or {}
    d4 = int(cal[J].get("dir", 1))
    # gravity FF at the live pose (same as run 1's audit — reuse, ~constant over ±3°)
    tau_ff_raw = 0.0
    try:
        from mabel.arm_control import ArmGravityCompensator
        mjcf = str(HB.parent / "simulation/mabel_mujoco/models/mabel_full.xml")
        g = ArmGravityCompensator(mjcf, {}, {}, log=lambda *a, **k: None)
        pose = {j: JointCalib.from_dict(cal[j]).to_model(snap(link, idx_of[j])[0])
                for j in LEFT + [f"r_j{i}" for i in range(1, 8)] if j in cal}
        tau_ff_raw = d4 * float(g.tff(pose, frame="model").get(J, 0.0))
        print(f"  gravity FF (raw frame): {tau_ff_raw:+.2f} N·m — applied in every condition")
    except Exception as e:
        print(f"  (no gravity FF: {e})")

    CONDS = [("deployed  kp30.5 kd4.27", 30.5, 4.27),
             ("stiffer   kp45   kd5.0 ", 45.0, 5.00),
             ("stiff     kp60   kd5.0 ", 60.0, 5.00),
             ("soft-kd   kp30.5 kd2.0 ", 30.5, 2.00)]
    results = []
    for label, kp, kd in CONDS:
        grp = link.group([J], kp=kp, kd=kd)
        assert wait(lambda: grp.resolved)
        p_start = snap(link, idx_of[J])[0]
        grp.set_torque(True)
        grp.command({J: p_start}, effort={J: tau_ff_raw})
        time.sleep(1.0)
        _, tr = record(link, [J], idx_of, 5.0, label)
        m = metrics(tr[J])
        # step: +3° then back, measure how much of it the joint actually tracks
        p1 = snap(link, idx_of[J])[0]
        grp.command({J: p1 + 0.05}, effort={J: tau_ff_raw})
        time.sleep(1.2)
        reached = snap(link, idx_of[J])[0] - p1
        grp.command({J: p1}, effort={J: tau_ff_raw})
        time.sleep(1.2)
        back = snap(link, idx_of[J])[0] - p1
        grp.set_torque(False)
        time.sleep(0.8)
        pct = reached / 0.05 * 100
        results.append((label, m, pct))
        print(f"  {label}  hold p-p {m['pp_deg']:5.2f}° vRMS {m['rms_vel']:.3f}  "
              f"step tracked {pct:5.1f}%  return residual {math.degrees(back):+5.2f}°")
    print("  ⇒ dead-zone theory: tracked% should rise with kp; hold p-p should not grow")
    link.close()
    print("══ run 4 done — all slices damped ══════════════════════════════════")


if __name__ == "__main__" and "--run4" in sys.argv:
    main4()
    sys.exit(0)


def main5():
    """kd discriminator: identical +2.2 N·m t_ff pulses at kd 0.5 vs kd 5.0. If the
    motor honors kd, the stiff-damped pulse peaks much slower. Unchanged peak velocity
    = every gain field is dead on this motor and only the torque field acts."""
    print("══ RUN 5: l_j4 kd discriminator (pulse at kd 0.5 vs 5.0) ═══════════")
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: print("   " + str(m)))
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    idx_of = {s.name: i for i, s in enumerate(link.specs())}
    time.sleep(2.0)
    for j, kd in (("l_j4", 0.5), ("l_j4", 5.0), ("l_j3", 0.5), ("l_j3", 5.0)):
        g = link.group([j], kp=0.0, kd=kd)
        assert wait(lambda: g.resolved)
        p0 = snap(link, idx_of[j])[0]
        g.set_torque(True)
        g.command({j: p0}, effort={j: +2.2})       # fixed pulse, anti-gravity direction
        vmax_seen, moved = 0.0, 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            p, v, _ = snap(link, idx_of[j])
            vmax_seen = max(vmax_seen, abs(v))
            moved = max(moved, abs(p - p0))
            if moved > 0.25:                        # far enough — cut early
                break
            time.sleep(0.01)
        g.command({j: snap(link, idx_of[j])[0]}, effort={j: 0.0})
        g.set_torque(False)
        print(f"  {j} kd={kd:3.1f}: pulse +2.2 N·m → moved {math.degrees(moved):5.2f}°  "
              f"peak vel {vmax_seen:.3f} rad/s")
        time.sleep(1.0)
    link.close()
    print("══ run 5 done — all slices damped ══════════════════════════════════")


if __name__ == "__main__" and "--run5" in sys.argv:
    main5()
    sys.exit(0)

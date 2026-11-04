#!/usr/bin/env python3
"""exp_j4_pd_tune.py — PD gain tuning for the elbow joints (l_j4 + r_j4), NO hands.

The deployed derived gains (kp 30.5 / kd 4.27) were computed for the arm WITH the ORCA
hand's inertia; without it the elbows oscillate under real_run. Grid-test step responses
per joint on the real hardware and score smooth+precise+quick:

  score per (kp, kd): 3-deg step → settle time (|err| < 0.6 deg sustained), overshoot %,
  ring count (velocity zero-crossings after first goal-cross), quiescent-hold vRMS.

Gravity FF is held STATIC at the pose's model value so PD is isolated. Watchdogged;
returns each joint to start. Exclusive Teensy access required.
"""
import math
import statistics
import sys
import time
from pathlib import Path

HB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HB))
sys.path.insert(0, str(HB.parent / "controller"))
sys.path.insert(0, str(HB / "experiments"))

from exp_j4_shake import HealthLink, wait  # noqa: E402
from mabel_hw import config                # noqa: E402
from mabel_hw.calibration import CalibrationStore  # noqa: E402

STEP = 0.052                    # ~3.0°
SETTLE_BAND = math.radians(0.6)


def step_metrics(samples, t_goal):
    """samples: (t, pos, vel); t_goal = commanded target (abs)."""
    if not samples:
        return None
    p0 = samples[0][1]
    err0 = t_goal - p0
    # settle: last time |err| exceeded the band
    t_settle = 0.0
    for t, p, v in samples:
        if abs(p - t_goal) > SETTLE_BAND:
            t_settle = t
    peak = max((p for _, p, _ in samples), default=p0) if err0 > 0 else \
           min((p for _, p, _ in samples), default=p0)
    overshoot = max(0.0, (peak - t_goal) / err0 * 100) if err0 > 0 else \
                max(0.0, (t_goal - peak) / (-err0) * 100)
    crossed = False
    rings = 0
    prev_v = None
    for t, p, v in samples:
        if not crossed and ((err0 > 0 and p >= t_goal) or (err0 < 0 and p <= t_goal)):
            crossed = True
        if crossed and prev_v is not None and prev_v * v < 0 and abs(v) > 0.05:
            rings += 1
        prev_v = v
    final_err = abs(samples[-1][1] - t_goal)
    return dict(settle=t_settle, overshoot=overshoot, rings=rings,
                final_deg=math.degrees(final_err))


def main():
    cal = CalibrationStore().load().slice("arm/openarm-dual/calib") or {}
    link = HealthLink(config.LINKS["arms_openarm"]["ip"],
                      config.LINKS["arms_openarm"]["port"], 100,
                      logger=lambda m: None)
    link.connect()
    assert wait(lambda: link.specs() and link.state()), "Teensy not answering"
    idx_of = {s.name: i for i, s in enumerate(link.specs())}
    time.sleep(2.5)

    # static gravity FF per joint at the current pose (isolates PD)
    ff = {}
    try:
        from mabel.arm_control import ArmGravityCompensator
        from mabel.arm_control.gravity import HandsConfig
        g = ArmGravityCompensator(str(HB.parent / "simulation/mabel_mujoco/models/mabel_full.xml"),
                                  {}, {}, hands=HandsConfig.NONE, pose_pad=10.0,
                                  log=lambda *a, **k: None)
        pose = {}
        for j, e in cal.items():
            if isinstance(e, dict) and "raw_min" in e and j in idx_of:
                rmid = 0.5 * (e["raw_min"] + e["raw_max"])
                pose[j] = float(e.get("model_mid", 0)) + int(e.get("dir", 1)) * (
                    link.state().joints[idx_of[j]].pos - rmid)
        tm = g.tff(pose, frame="model")
        for j in ("l_j4", "r_j4"):
            ff[j] = int(cal[j].get("dir", 1)) * float(tm.get(j, 0.0))
        print(f"static FF: l_j4 {ff.get('l_j4', 0):+.2f}  r_j4 {ff.get('r_j4', 0):+.2f} N·m")
    except Exception as e:
        print(f"(FF unavailable: {e})")

    GRID = [(20.0, 1.5), (20.0, 2.5), (30.5, 1.5), (30.5, 2.5), (30.5, 3.5),
            (30.5, 4.27), (45.0, 2.5), (45.0, 3.5)]
    results = {}
    for J in ("l_j4", "r_j4"):
        print(f"── {J} ──   (step ±3.0°, settle band ±0.6°)")
        results[J] = []
        for kp, kd in GRID:
            grp = link.group([J], kp=kp, kd=kd)
            assert wait(lambda: grp.resolved)
            p0 = link.state().joints[idx_of[J]].pos
            grp.set_torque(True)
            grp.command({J: p0}, effort={J: ff.get(J, 0.0)})
            time.sleep(0.6)
            # quiescent hold noise
            vs = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                vs.append(link.state().joints[idx_of[J]].vel)
                time.sleep(0.005)
            hold_vrms = math.sqrt(sum(v * v for v in vs) / len(vs))
            # step up, record 1.8 s
            tgt = p0 + STEP
            grp.command({J: tgt}, effort={J: ff.get(J, 0.0)})
            samples = []
            t0 = time.monotonic()
            aborted = False
            while time.monotonic() - t0 < 1.8:
                f = link.state().joints[idx_of[J]]
                samples.append((time.monotonic() - t0, f.pos, f.vel))
                if abs(f.pos - p0) > 0.35 or abs(f.vel) > 3.0:
                    aborted = True
                    break
                time.sleep(0.004)
            m = step_metrics(samples, tgt)
            # back to start
            grp.command({J: p0}, effort={J: ff.get(J, 0.0)})
            time.sleep(1.0)
            grp.set_torque(False)
            time.sleep(0.4)
            tag = "ABORT" if aborted else (
                f"settle {m['settle']:4.2f}s  overshoot {m['overshoot']:5.1f}%  "
                f"rings {m['rings']:2d}  final {m['final_deg']:4.2f}°  "
                f"holdvRMS {hold_vrms:.3f}")
            print(f"  kp {kp:4.1f} kd {kd:4.2f}:  {tag}")
            if not aborted:
                results[J].append((kp, kd, m, hold_vrms))
    link.close()
    # score: settle + rings + overshoot penalty + hold noise
    print("\n── recommendation (min settle + 0.15·rings + 0.01·overshoot + 2·holdvRMS) ──")
    for J, rows in results.items():
        if not rows:
            continue
        best = min(rows, key=lambda r: r[2]["settle"] + 0.15 * r[2]["rings"]
                   + 0.01 * r[2]["overshoot"] + 2.0 * r[3])
        print(f"  {J}: kp {best[0]:g} kd {best[1]:g}  "
              f"(settle {best[2]['settle']:.2f}s, overshoot {best[2]['overshoot']:.0f}%, "
              f"rings {best[2]['rings']}, holdvRMS {best[3]:.3f})")


if __name__ == "__main__":
    main()

"""
Rigorous tip-over stability sweep for the MABEL base motion model.

Drives the base through a large grid of input combinations using the SAME control
stack as the teleop (config.safe_base_step -> swerve_ik -> mj_step) and measures
the ACTUAL chassis tilt in simulation -- the ground truth, independent of the
tip model's assumed CoM. Reports the worst-case tilt per category and flags any
combination whose peak tilt exceeds a threshold (an "edge case").

Varied inputs:
  - lift height z            (CoM height)
  - linear speed magnitude   (how fast)
  - linear direction         (8 headings)
  - angular velocity         (sign + magnitude)
  - maneuver                 (step-in-then-stop; hard reversal)

Run:  python scripts/tools/tip_stability_sweep.py
"""
import os, sys, math, itertools
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mabel import config as C
from mabel.swerve import swerve_ik

TILT_WARN = math.radians(5.0)    # peak tilt above this = flagged edge case
DT = None


def tilt_of(data, cid):
    """Angle [rad] of the chassis z-axis from world vertical."""
    r22 = data.body(cid).xmat.reshape(3, 3)[2, 2]
    return math.acos(max(-1.0, min(1.0, r22)))


def settle(model, data, z, n=400):
    mujoco.mj_resetData(model, data)
    try: data.actuator("Lift").ctrl[0] = z
    except KeyError: pass
    for _ in range(n):
        mujoco.mj_step(model, data)


def run_scenario(model, data, z, tw_drive, dur_drive, dur_stop):
    """tw_drive = (tvx,tvy,tw) target during the drive phase, then stop. Returns
    (peak_tilt, stopped_speed)."""
    cid = model.body("mabel_chassis").id
    settle(model, data, z)
    vx = vy = w = 0.0
    peak = 0.0
    n_drive = int(dur_drive / DT); n_stop = int(dur_stop / DT)
    for k in range(n_drive + n_stop):
        tvx, tvy, tw = tw_drive if k < n_drive else (0.0, 0.0, 0.0)
        vx, vy, w = C.safe_base_step(vx, vy, w, tvx, tvy, tw, DT, z)
        swerve_ik(vx, vy, w, data)
        mujoco.mj_step(model, data)
        peak = max(peak, tilt_of(data, cid))
    return peak, math.hypot(vx, vy) + abs(w)


def run_reversal(model, data, z, speed, ang):
    """Drive to (forward speed, +ang), then snap to (back speed, -ang)."""
    cid = model.body("mabel_chassis").id
    settle(model, data, z)
    vx = vy = w = 0.0; peak = 0.0
    phases = [((-speed, 0.0, ang), 1.0), ((speed, 0.0, -ang), 1.2), ((0, 0, 0), 0.8)]
    for tw_t, dur in phases:
        for _ in range(int(dur / DT)):
            vx, vy, w = C.safe_base_step(vx, vy, w, *tw_t, DT, z)
            swerve_ik(vx, vy, w, data)
            mujoco.mj_step(model, data)
            peak = max(peak, tilt_of(data, cid))
    return peak, math.hypot(vx, vy) + abs(w)


def main():
    global DT
    xml = os.path.join(HERE, "..", C.SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    DT = model.opt.timestep

    lifts  = [0.0, 0.635]
    speeds = [1.0, 1.5]
    dirs   = [i * math.pi / 4 for i in range(8)]
    omegas = [0.0, 2.0, 3.5, -2.0, -3.5]

    worst = {}          # category -> (tilt, descr)
    flagged = []
    n = 0
    for z, speed, th, om in itertools.product(lifts, speeds, dirs, omegas):
        tvx = -speed * math.cos(th)   # forward = -X
        tvy =  speed * math.sin(th)
        peak, stopped = run_scenario(model, data, z, (tvx, tvy, om), 1.2, 0.8)
        n += 1
        key = f"z={z}"
        if key not in worst or peak > worst[key][0]:
            worst[key] = (peak, f"speed={speed} dir={int(math.degrees(th))}deg om={om}")
        if peak > TILT_WARN or stopped > 0.05:
            flagged.append((peak, stopped, z, speed, int(math.degrees(th)), om))

    # hard reversals (worst combined transient)
    rev_worst = 0.0; rev_desc = ""
    for z, speed, ang in itertools.product(lifts, [1.2], [2.0, 3.5]):
        peak, stopped = run_reversal(model, data, z, speed, ang)
        if peak > rev_worst: rev_worst, rev_desc = peak, f"z={z} speed={speed} ang={ang}"
        if peak > TILT_WARN or stopped > 0.05:
            flagged.append((peak, stopped, z, speed, "REV", ang))

    print(f"=== Tip-stability sweep: {n} step scenarios + reversals ===")
    print(f"(flag if peak tilt > {math.degrees(TILT_WARN):.0f} deg or failed to stop)\n")
    for k in sorted(worst):
        t, d = worst[k]
        print(f"  worst {k}: peak tilt {math.degrees(t):.2f} deg   [{d}]")
    print(f"  worst reversal: peak tilt {math.degrees(rev_worst):.2f} deg   [{rev_desc}]")
    print()
    if not flagged:
        print(f"PASS: all scenarios stable (<{math.degrees(TILT_WARN):.0f} deg) and stopped.")
    else:
        print(f"FAILED {len(flagged)} edge case(s):")
        for peak, stopped, z, speed, d, om in sorted(flagged, reverse=True)[:20]:
            print(f"  peak {math.degrees(peak):5.1f} deg  end-speed {stopped:.2f}  "
                  f"z={z} speed={speed} dir={d} om={om}")


if __name__ == "__main__":
    main()

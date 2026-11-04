"""
Rigorous sim test for MABEL whole-body force compliance (drag-to-move).

Applies a simulated external force to a hand (data.xfrc_applied, as if a human
were dragging it) over a grid of directions and magnitudes, runs the FULL
compliant stack (gravity-comp arms -> wrench sensing -> base admittance ->
tip-safe smoother -> swerve), and measures whether the base:
  - engages quickly                          (engage time, s)
  - moves in the intended (force) direction  (dir error, deg)
  - offloads the arm before it over-extends  (peak hand excursion, m)
  - stays upright                            (peak tilt, deg)
  - stops when released                      (stop time, s)

Run:  python scripts/tools/compliance_drag_test.py
"""
import os, sys, math
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mabel import config as C
from mabel.dynamics import UpperBodyDynamics, SimPlant, ComplianceController, COMPLIANT_ARM_JOINTS, LAZY_JOINTS
from mabel.admittance import BaseAdmittance, external_base_wrench, boundary_gate, hand_centroid_base
from mabel.swerve import swerve_ik


def arm_bodies(model):
    keys = ("arm", "wrist", "palm", "forearm", "hand", "tower",
            "thumb", "index", "middle", "ring", "pinky")
    return [model.body(i).name for i in range(model.nbody)
            if model.body(i).name and any(k in model.body(i).name.lower() for k in keys)]


def lift_z(data):
    q = data.joint(C.LIFT_JOINT).qpos[0]
    return min(C.LIFT_RANGE[1], max(C.LIFT_RANGE[0], q / 0.303 * C.LIFT_RANGE[1]))


def run(model, data, dyn, plant, bodies, pid, F_world, push_s=2.0, rel_s=1.5):
    # fresh compliant controller + admittance each run
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    comp = ComplianceController(dyn, plant, compliant=COMPLIANT_ARM_JOINTS, lazy=LAZY_JOINTS)
    base = BaseAdmittance()
    dt = model.opt.timestep
    cid = model.body("mabel_chassis").id
    # settle a few ticks, capture neutral hand centroid (base frame)
    for _ in range(50):
        comp.update(); mujoco.mj_step(model, data)
    c0 = hand_centroid_base(data, data.body(C.LEFT_PALM).xpos, data.body(C.RIGHT_PALM).xpos)
    p0 = data.body(cid).xpos.copy()

    vx = vy = w = 0.0
    engage_t = None; peak_tilt = 0.0; peak_exc = 0.0; stop_t = None
    n_push = int(push_s / dt); n_rel = int(rel_s / dt)
    prev = data.body(cid).xpos.copy()
    push_start = data.body(cid).xpos.copy(); push_end = push_start.copy()
    for k in range(n_push + n_rel):
        data.xfrc_applied[pid, :3] = F_world if k < n_push else 0.0
        comp.update()
        fx, fy, mz = external_base_wrench(model, data, bodies)
        cb = hand_centroid_base(data, data.body(C.LEFT_PALM).xpos, data.body(C.RIGHT_PALM).xpos)
        gate = boundary_gate(cb, c0)
        tvx, tvy, tw = base.step(fx, fy, mz, dt, gate=gate)
        vx, vy, w = C.safe_base_step(vx, vy, w, tvx, tvy, tw, dt, lift_z(data))
        swerve_ik(vx, vy, w, data)
        mujoco.mj_step(model, data)

        cur = data.body(cid).xpos
        vw = (cur - prev) / dt; prev = cur.copy()
        sp = math.hypot(vw[0], vw[1])
        r = data.body(cid).xmat.reshape(3, 3)[2, 2]
        peak_tilt = max(peak_tilt, math.degrees(math.acos(max(-1, min(1, r)))))
        peak_exc = max(peak_exc, float(np.linalg.norm((cb - c0)[:2])))
        if 30 < k < n_push and sp > 0.05 and engage_t is None:
            engage_t = (k - 30) * dt
        if k == n_push - 1:
            push_end = data.body(cid).xpos.copy()
        if k >= n_push and stop_t is None and math.hypot(vx, vy) + abs(w) < 0.03:
            stop_t = (k - n_push) * dt
    # direction from NET displacement over the push (robust to per-tick noise)
    disp = (push_end - push_start)[:2]
    f_dir = math.atan2(F_world[1], F_world[0])
    if np.linalg.norm(disp) > 0.02:
        dir_err = abs((math.degrees(math.atan2(disp[1], disp[0]) - f_dir) + 180) % 360 - 180)
    else:
        dir_err = float('nan')
    return dict(engage=engage_t, dir_err=dir_err, peak_exc=peak_exc,
                peak_tilt=peak_tilt, stop=stop_t, final_speed=math.hypot(vx, vy) + abs(w))


def main():
    xml = os.path.join(HERE, "..", C.SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    dyn = UpperBodyDynamics(model, data)
    plant = SimPlant(model, data)
    bodies = arm_bodies(model)
    pid = model.body(C.LEFT_PALM).id

    print(f"box={C.DRAG_BOX} ramp={C.DRAG_BOX_RAMP} mass={C.ADMIT_MASS} "
          f"b_lin={C.ADMIT_LIN_DAMP} f_static={C.ADMIT_F_STATIC}\n")
    print("force         engage  dir_err  hand_exc  tilt   stop   end_spd")
    dirs = {"fwd(-x)": (-1, 0), "back(+x)": (1, 0), "left(-y)": (0, -1), "right(+y)": (0, 1)}
    for mag in (15.0, 30.0):
        for name, (ux, uy) in dirs.items():
            r = run(model, data, dyn, plant, bodies, pid,
                    np.array([ux * mag, uy * mag, 0.0]))
            eng = f"{r['engage']:.2f}s" if r['engage'] is not None else " none"
            stp = f"{r['stop']:.2f}s" if r['stop'] is not None else " >1.5"
            print(f"{mag:3.0f}N {name:9s} {eng:>7}  {r['dir_err']:5.1f}   "
                  f"{r['peak_exc']:.3f}   {r['peak_tilt']:4.1f}  {stp:>6}  {r['final_speed']:.3f}")


if __name__ == "__main__":
    main()

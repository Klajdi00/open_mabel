#!/usr/bin/env python3
"""Base motion-model system identification for MABEL (bare floor scene).

Drives the 3-module swerve base with the controller's own IK (mabel.swerve.swerve_ik)
in a robot+floor-only scene and measures the TRUE steady-state body twist produced by
each commanded twist. The commanded->actual relationship is the calibrated forward
motion model that the no-slip kinematics gets wrong (it assumes wheel_speed*r = ground
speed; the real base slips and the drive actuators saturate).

Run:  python3 motion_id.py            # sweep + fit + print model
"""
import os, sys, types, math
import numpy as np
import mujoco

# --- import the controller's swerve IK + config WITHOUT the heavy package __init__ ---
_CTRL = os.path.expanduser(os.environ.get("MABEL_CONTROLLER", "~/open_mabel/controller"))
if "mabel" not in sys.modules:
    pkg = types.ModuleType("mabel"); pkg.__path__ = [os.path.join(_CTRL, "mabel")]
    sys.modules["mabel"] = pkg
from mabel import config as C          # noqa: E402
from mabel import swerve               # noqa: E402
from mabel import base_motion as BM    # noqa: E402

SCENE = os.path.expanduser("~/open_mabel/simulation/mabel_mujoco/scenes/mabel_bare.xml")
model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

# base free-joint addresses
bj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_root")
QADR = model.jnt_qposadr[bj]
# non-swerve position actuators -> hold the upper body static during the test
SWERVE_ACTS = {"FL Steer", "FL Drive", "FR Steer", "FR Drive", "B Steer", "B Drive"}
hold_acts = []
for i in range(model.nu):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if n in SWERVE_ACTS:
        continue
    trn = model.actuator_trnid[i, 0]
    jq = model.jnt_qposadr[trn] if trn >= 0 else -1
    hold_acts.append((i, jq))


def settle(steps=1500):
    mujoco.mj_resetData(model, data)
    for _ in range(steps):
        for ai, jq in hold_acts:          # hold arms/body/hands at rest qpos
            if jq >= 0:
                data.ctrl[ai] = data.qpos[jq]
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


QSET, VSET = settle()


def yaw_of():
    qw, qx, qy, qz = data.qpos[QADR+3:QADR+7]
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


def measure(vx, vy, omega, T=4.0, warm=2.0):
    """Apply (vx,vy,omega) via swerve IK; return mean BODY-frame (vx,vy,omega) over
    the steady window [warm, T]."""
    data.qpos[:] = QSET; data.qvel[:] = VSET
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    n_total = int(T/dt); n_warm = int(warm/dt)
    acc = np.zeros(3); cnt = 0
    px, py, pth = data.qpos[QADR], data.qpos[QADR+1], yaw_of()
    for k in range(n_total):
        for ai, jq in hold_acts:
            if jq >= 0:
                data.ctrl[ai] = data.qpos[jq]
        swerve.swerve_ik(vx, vy, omega, data)
        mujoco.mj_step(model, data)
        nx, ny, nth = data.qpos[QADR], data.qpos[QADR+1], yaw_of()
        if k >= n_warm:
            # world displacement -> body frame (rotate by -yaw)
            dx, dy = (nx-px)/dt, (ny-py)/dt
            c, s = math.cos(-pth), math.sin(-pth)
            acc += [c*dx - s*dy, s*dx + c*dy,
                    ((nth-pth+math.pi) % (2*math.pi)-math.pi)/dt]
            cnt += 1
        px, py, pth = nx, ny, nth
    return acc/max(cnt, 1)


def sweep_axis(idx, vals):
    """Sweep one command axis (0=vx,1=vy,2=omega); return list of (cmd, actual3)."""
    out = []
    for v in vals:
        cmd = [0.0, 0.0, 0.0]; cmd[idx] = v
        out.append((v, measure(*cmd)))
    return out


def fit_axis(rows, idx):
    """Fit actual_idx = gain*cmd with saturation at vmax. Returns (gain, vmax)."""
    c = np.array([r[0] for r in rows])
    a = np.array([r[1][idx] for r in rows])
    # gain from the small-command (unsaturated, |cmd|<=lin_lim) linear region
    lin = np.abs(c) <= (0.5 if idx < 2 else 1.2)
    gain = float(np.polyfit(c[lin], a[lin], 1)[0]) if lin.sum() >= 2 else 1.0
    vmax = float(np.max(np.abs(a)))
    return gain, vmax


def verify_trajectory():
    """Drive a held-out, varied command trajectory; integrate the calibrated model
    AND the old no-slip model (gain=1.0) in lockstep with the physics, and compare
    both predicted paths to the actual ground-truth path."""
    data.qpos[:] = QSET; data.qvel[:] = VSET
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    # (vx, vy, omega, seconds) -- all inside the reliable envelope, varied 2D motion
    segs = [(-0.5, 0.0, 0.0, 2.5), (-0.4, 0.0, 0.6, 3.0), (0.0, 0.4, 0.0, 2.0),
            (-0.3, -0.3, 0.0, 2.5), (0.0, 0.0, 1.2, 2.0), (-0.5, 0.0, -0.5, 3.0),
            (0.3, 0.3, 0.4, 2.5)]
    # initial pose = actual start (so all three share an origin)
    ax = data.qpos[QADR]; ay = data.qpos[QADR+1]; ath = yaw_of()
    mm = BM.BaseMotionModel(ax, ay, ath)   # calibrated stateful model (gain + ramp)
    nx, ny, nth = ax, ay, ath              # no-slip model (gain 1.0, no ramp)
    mx, my, mth = ax, ay, ath
    err_m = []; err_n = []
    for vx, vy, w, dur in segs:
        for _ in range(int(dur/dt)):
            for ai, jq in hold_acts:
                if jq >= 0:
                    data.ctrl[ai] = data.qpos[jq]
            swerve.swerve_ik(vx, vy, w, data)
            mujoco.mj_step(model, data)
            mx, my, mth = mm.step(vx, vy, w, dt)
            # no-slip: same integrator but gain 1.0, no envelope clamp, no ramp
            thm = nth + 0.5*w*dt
            nx += (vx*math.cos(thm) - vy*math.sin(thm))*dt
            ny += (vx*math.sin(thm) + vy*math.cos(thm))*dt
            nth = math.atan2(math.sin(nth+w*dt), math.cos(nth+w*dt))
            gx, gy = data.qpos[QADR], data.qpos[QADR+1]
            err_m.append(math.hypot(mx-gx, my-gy))
            err_n.append(math.hypot(nx-gx, ny-gy))
    gx, gy, gth = data.qpos[QADR], data.qpos[QADR+1], yaw_of()
    path = 0.0  # rough GT path
    print("== held-out trajectory verification (predicted vs ground truth) ==")
    print(f"  GT final pose      : x={gx:+.3f} y={gy:+.3f} yaw={math.degrees(gth):+.1f}deg")
    print(f"  calibrated model   : x={mx:+.3f} y={my:+.3f} yaw={math.degrees(mth):+.1f}deg  "
          f"| final err {math.hypot(mx-gx,my-gy)*100:.1f} cm | ATE {1e2*np.mean(err_m):.1f} cm")
    print(f"  no-slip model (g=1): x={nx:+.3f} y={ny:+.3f} yaw={math.degrees(nth):+.1f}deg  "
          f"| final err {math.hypot(nx-gx,ny-gy)*100:.1f} cm | ATE {1e2*np.mean(err_n):.1f} cm")
    tot = math.hypot(gx-ax, gy-ay)
    print(f"  (net GT displacement {tot:.2f} m; calibrated model error "
          f"{100*math.hypot(mx-gx,my-gy)/max(tot,1e-6):.1f}% of distance travelled)")


if __name__ == "__main__":
    if "verify" in sys.argv:
        verify_trajectory(); sys.exit(0)
    print("== per-axis sweep: commanded -> actual steady-state body twist ==")
    res = {}
    vx_vals = [-1.5, -1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
    w_vals = [-3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
    res['vx'] = sweep_axis(0, vx_vals)
    res['vy'] = sweep_axis(1, vx_vals)
    res['w'] = sweep_axis(2, w_vals)
    for name, idx in [('vx', 0), ('vy', 1), ('w', 2)]:
        print(f"\n  -- {name} axis --")
        for v, a in res[name]:
            print(f"    cmd {v:+.2f} -> vx{a[0]:+.3f} vy{a[1]:+.3f} w{a[2]:+.3f}")
        g, vm = fit_axis(res[name], idx)
        print(f"    => gain={g:.3f}  saturation|max|={vm:.3f}")

    print("\n== coupling check (combined commands) ==")
    for cmd in [(-0.4, 0.3, 0), (-0.4, 0, 1.0), (0, 0.4, 1.0), (-0.5, 0.3, 0.8)]:
        a = measure(*cmd)
        print(f"  cmd vx{cmd[0]:+.2f} vy{cmd[1]:+.2f} w{cmd[2]:+.2f} -> "
              f"vx{a[0]:+.3f} vy{a[1]:+.3f} w{a[2]:+.3f}")

    np.savez("/tmp/motion_id.npz",
             vx=np.array([(v, *a) for v, a in res['vx']]),
             vy=np.array([(v, *a) for v, a in res['vy']]),
             w=np.array([(v, *a) for v, a in res['w']]))
    print("\nsaved /tmp/motion_id.npz")

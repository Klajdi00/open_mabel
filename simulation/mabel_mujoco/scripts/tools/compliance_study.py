"""
Experimental study of MABEL's whole-body compliant controller.

Runs a suite of rigorous experiments (grounded in the admittance-control / pHRI
literature) over the gravity-compensation + base-admittance stack, saves
IEEE-column-width figures to docs/figures/, and prints the measured metrics and a
recommended ("natural feel") parameter set. See docs/compliance_study.pdf for the
write-up.

    python scripts/tools/compliance_study.py
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from mabel import config as C
from mabel.dynamics import UpperBodyDynamics, SimPlant, ComplianceController, COMPLIANT_ARM_JOINTS, LAZY_JOINTS
from mabel.admittance import BaseAdmittance, external_base_wrench, boundary_gate, hand_centroid_base
from mabel.swerve import swerve_ik

FIG = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "figures"))
os.makedirs(FIG, exist_ok=True)
XML = os.path.join(HERE, "..", C.SCENE_XML)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "lines.linewidth": 1.3})
COL = (3.4, 2.5)   # IEEE single column


def _model():
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


# ── Study 1: gravity-comp back-drivability ───────────────────────────────────
def study_backdrivability():
    m, d = _model()
    dyn = UpperBodyDynamics(m, d)
    arms = COMPLIANT_ARM_JOINTS
    # (a) residual felt torque vs mass-model error
    scales = np.linspace(0.6, 1.4, 17)
    g_true = dyn.gravity().copy()
    dofs = [m.joint(j).dofadr[0] for j in arms]
    resid = []
    base_mass = {b: m.body(b).mass[0] for b in
                 ["Torso"] + [f"Left Arm {i}" for i in range(1, 8)] + [f"Right Arm {i}" for i in range(1, 8)]
                 if _has(m, b)}
    for s in scales:
        for b, m0 in base_mass.items():
            m.body(b).mass[0] = m0 * s
        mujoco.mj_setConst(m, d); mujoco.mj_forward(m, d)
        g_hat = dyn.gravity()
        resid.append(np.linalg.norm((g_true - g_hat)[dofs]))
    for b, m0 in base_mass.items():   # restore
        m.body(b).mass[0] = m0
    mujoco.mj_setConst(m, d)

    # (b) joint drift over time vs compensation damping
    damps = [0.0, 0.2, 0.5, 1.0, 2.0]
    drift_curves = {}
    for D in damps:
        m2, d2 = _model(); dyn2 = UpperBodyDynamics(m2, d2); plant2 = SimPlant(m2, d2)
        ctrl = ComplianceController(dyn2, plant2, compliant=arms, lazy=LAZY_JOINTS, compliant_damping=D)
        q0 = np.array([d2.joint(j).qpos[0] for j in arms])
        ts, dr = [], []
        for k in range(2500):
            ctrl.update(); mujoco.mj_step(m2, d2)
            if k % 25 == 0:
                ts.append(k * m2.opt.timestep)
                dr.append(np.max(np.abs([d2.joint(j).qpos[0] for j in arms] - q0)))
        drift_curves[D] = (ts, dr)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax[0].plot((scales - 1) * 100, resid, "o-")
    ax[0].set_xlabel("upper-body mass error [%]"); ax[0].set_ylabel(r"residual $\|g-\hat g\|$ [N·m]")
    ax[0].set_title("(a) felt resistance vs model error")
    for D, (ts, dr) in drift_curves.items():
        ax[1].plot(ts, np.degrees(dr), label=f"D={D}")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("max joint drift [deg]")
    ax[1].set_title("(b) drift vs comp. damping"); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/backdrivability.pdf"); plt.close(fig)
    # pick smallest damping with <2 deg drift at 5 s
    rec_D = next((D for D in damps if np.degrees(drift_curves[D][1][-1]) < 2.0), damps[-1])
    return dict(resid_at_20pct=float(np.interp(20, (scales-1)*100, resid)), rec_damping=rec_D)


# ── Study 2: admittance step response (virtual mass sweep) ────────────────────
def study_step_response():
    dt = 0.002; F = 20.0; T = 1.5
    masses = [2.0, 4.5, 8.0, 16.0]
    fig, ax = plt.subplots(figsize=COL)
    metrics = {}
    for mvm in masses:
        adm = BaseAdmittance(mass=mvm)
        ts, vs = [], []; v = 0.0
        for k in range(int(T / dt)):
            vx, _, _ = adm.step(F, 0, 0, dt, gate=1.0)
            ts.append(k * dt); vs.append(abs(vx))
        vs = np.array(vs); v_ss = vs[-1]
        tau = ts[np.argmax(vs >= 0.632 * v_ss)] if v_ss > 0 else float("nan")
        t10 = ts[np.argmax(vs >= 0.1 * v_ss)]; t90 = ts[np.argmax(vs >= 0.9 * v_ss)]
        metrics[mvm] = dict(v_ss=v_ss, tau=tau, rise=t90 - t10)
        ax.plot(ts, vs, label=f"m={mvm:g} kg ($\\tau$={tau:.2f}s)")
    ax.axhline(F / C.ADMIT_LIN_DAMP, ls="--", c="k", lw=0.8, label="$v_{ss}=F/b$")
    ax.set_xlabel("time [s]"); ax.set_ylabel("base speed [m/s]")
    ax.set_title(f"Admittance step response (F={F:g} N, b={C.ADMIT_LIN_DAMP:g})")
    ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/step_response.pdf"); plt.close(fig)
    return metrics


# ── Study 3: effort–speed law ─────────────────────────────────────────────────
def study_effort_speed():
    dt = 0.002; forces = np.arange(0, 51, 2.5)
    vss = []
    for F in forces:
        adm = BaseAdmittance()
        v = 0.0
        for _ in range(1500):
            v = adm.step(F, 0, 0, dt, gate=1.0)[0]
        vss.append(abs(v))
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(forces, vss, "o-")
    ax.axvline(C.ADMIT_F_STATIC, ls=":", c="r", label=f"deadband {C.ADMIT_F_STATIC:g} N")
    ax.set_xlabel("applied force [N]"); ax.set_ylabel("steady base speed [m/s]")
    ax.set_title("Effort–speed law (lightness = 1/b)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/effort_speed.pdf"); plt.close(fig)
    return dict(slope=float((vss[-1]-vss[4])/(forces[-1]-forces[4])))


# ── Study 4: Coulomb deadband vs drift under noise ────────────────────────────
def study_deadband():
    dt = 0.002; rng = np.random.default_rng(0)
    fstatics = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
    drift = []
    for fs in fstatics:
        adm = BaseAdmittance(f_static=fs)
        x = 0.0
        for _ in range(5000):
            n = rng.normal(0, 2.0)   # 2 N RMS sensor/estimation noise
            v = adm.step(n, 0, 0, dt, gate=1.0)[0]; x += v * dt
        drift.append(abs(x))
    fig, ax = plt.subplots(figsize=COL)
    ax.plot(fstatics, np.array(drift) * 100, "o-")
    ax.set_xlabel("static deadband $f_s$ [N]"); ax.set_ylabel("drift over 10 s [cm]")
    ax.set_title("No-creep: drift vs deadband (2 N noise)")
    fig.tight_layout(); fig.savefig(f"{FIG}/deadband.pdf"); plt.close(fig)
    rec = next((fs for fs, dr in zip(fstatics, drift) if dr * 100 < 1.0), fstatics[-1])
    return dict(rec_fstatic=rec, drift=dict(zip(fstatics, [float(x) for x in drift])))


# ── Study 5: directional isotropy (full stack) ────────────────────────────────
def study_isotropy():
    m, d = _model(); dyn = UpperBodyDynamics(m, d); plant = SimPlant(m, d)
    bodies = [m.body(i).name for i in range(m.nbody) if m.body(i).name and any(
        k in m.body(i).name.lower() for k in ("arm","wrist","palm","forearm","hand","tower","thumb","index","middle","ring","pinky"))]
    pid = m.body(C.LEFT_PALM).id; cid = m.body("mabel_chassis").id; dt = m.opt.timestep
    dirs = np.linspace(0, 2*math.pi, 8, endpoint=False)
    eng, derr = [], []
    for th in dirs:
        mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
        comp = ComplianceController(dyn, plant); base = BaseAdmittance()
        for _ in range(50): comp.update(); mujoco.mj_step(m, d)
        c0 = hand_centroid_base(d, d.body(C.LEFT_PALM).xpos, d.body(C.RIGHT_PALM).xpos)
        F = 25.0 * np.array([math.cos(th), math.sin(th), 0.0])
        vx=vy=w=0.0; p0=d.body(cid).xpos.copy(); et=None
        for k in range(900):
            d.xfrc_applied[pid,:3]=F
            comp.update(); fx,fy,mz=external_base_wrench(m,d,bodies)
            cb=hand_centroid_base(d,d.body(C.LEFT_PALM).xpos,d.body(C.RIGHT_PALM).xpos)
            g=boundary_gate(cb,c0); tvx,tvy,tw=base.step(fx,fy,mz,dt,gate=g)
            vx,vy,w=C.safe_base_step(vx,vy,w,tvx,tvy,tw,dt,0.02); swerve_ik(vx,vy,w,d); mujoco.mj_step(m,d)
            if et is None and k>30 and math.hypot(vx,vy)>0.05: et=(k-30)*dt
        disp=(d.body(cid).xpos-p0)[:2]
        eng.append(et if et else np.nan)
        derr.append(abs((math.degrees(math.atan2(disp[1],disp[0])-th)+180)%360-180))
    fig = plt.figure(figsize=COL); ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.append(dirs, dirs[0]), np.append(eng, eng[0]), "o-", label="engage [s]")
    ax.set_title("Directional engage time [s]", pad=12); ax.set_rlabel_position(135)
    fig.tight_layout(); fig.savefig(f"{FIG}/isotropy.pdf"); plt.close(fig)
    return dict(engage=[float(x) for x in eng], dir_err=[float(x) for x in derr])


# ── Study 6: tip-stability envelope (speed x yaw) at raised lift ──────────────
def study_tip_envelope():
    m, d = _model(); dt = m.opt.timestep; cid = m.body("mabel_chassis").id
    speeds = np.linspace(0.0, 1.6, 6); omegas = np.linspace(0.0, 4.0, 6)
    Z = np.zeros((len(omegas), len(speeds)))
    for i, om in enumerate(omegas):
        for j, sp in enumerate(speeds):
            mujoco.mj_resetData(m, d); d.actuator("Lift").ctrl[0] = 0.635
            for _ in range(300): mujoco.mj_step(m, d)
            vx=vy=w=0.0; peak=0.0
            for k in range(700):
                tvx,tvy,tw = (-sp,0.0,om) if k<450 else (0,0,0)
                vx,vy,w=C.safe_base_step(vx,vy,w,tvx,tvy,tw,dt,0.635)
                swerve_ik(vx,vy,w,d); mujoco.mj_step(m,d)
                r=d.body(cid).xmat.reshape(3,3)[2,2]
                peak=max(peak, math.degrees(math.acos(max(-1,min(1,r)))))
            Z[i, j] = peak
    fig, ax = plt.subplots(figsize=COL)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="inferno",
                   extent=[speeds[0], speeds[-1], omegas[0], omegas[-1]], vmax=15)
    cs = ax.contour(speeds, omegas, Z, levels=[5], colors="cyan")
    ax.clabel(cs, fmt="5°")
    ax.set_xlabel("linear speed [m/s]"); ax.set_ylabel("yaw rate [rad/s]")
    ax.set_title("Peak tilt [deg], lift up (combined motion)")
    fig.colorbar(im, ax=ax, label="peak tilt [deg]")
    fig.tight_layout(); fig.savefig(f"{FIG}/tip_envelope.pdf"); plt.close(fig)
    return dict(max_tilt=float(Z.max()), frac_over5=float((Z > 5).mean()))


# ── Study 7: smoothness / jerk under a drag ───────────────────────────────────
def study_smoothness():
    dt = 0.002; F = 25.0; T = 2.0
    fig, ax = plt.subplots(figsize=COL)
    out = {}
    for label, mvm in [("light m=2", 2.0), ("tuned m=4.5", 4.5), ("heavy m=16", 16.0)]:
        adm = BaseAdmittance(mass=mvm)
        vs = []
        for k in range(int(T/dt)):
            vs.append(adm.step(F if k < int(T/dt*0.6) else 0.0, 0, 0, dt, gate=1.0)[0])
        vs = np.array(vs); acc = np.gradient(vs, dt); jerk = np.gradient(acc, dt)
        ax.plot(np.arange(len(vs))*dt, np.abs(jerk), label=label)
        out[label] = float(np.sqrt(np.mean(jerk**2)))
    ax.set_xlabel("time [s]"); ax.set_ylabel("|jerk| [m/s³]")
    ax.set_title("Drag smoothness (jerk) vs virtual mass"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/smoothness.pdf"); plt.close(fig)
    return out


def study_contact_localization():
    """Push several arm links in several directions in CLOSED LOOP (the base
    admittance keeps the arm near its well-conditioned nominal pose) and report
    the force-direction and contact-point error of the joint-torque localizer,
    averaged over a steady window."""
    from mabel.dynamics import UpperBodyDynamics, SimPlant, ComplianceController
    from mabel.force_estimation import ContactEstimator
    m, d = _model()
    links = [b for b in ["left_palm", "l_wrist", "l_forearm", "l_elbow"] if _has(m, b)]
    Fs = {"-y": np.array([0, -15., 0]), "-x": np.array([-15., 0, 0])}
    pos_err = {k: [] for k in Fs}; dir_err = {k: [] for k in Fs}
    dt = m.opt.timestep
    # Quasi-static characterization: base held, light force, read in the window
    # just after the observer (bandwidth K=40 -> ~3/K = 0.075 s) converges and
    # before the free arm drifts out of its well-conditioned nominal pose.
    for body in links:
        pid = m.body(body).id
        for dlabel, F in Fs.items():
            dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
            comp = ComplianceController(UpperBodyDynamics(m, dd), SimPlant(m, dd))
            est = ContactEstimator(m, dd, gain=40.0, tau_thresh=0.3)
            for _ in range(50): comp.update(); mujoco.mj_step(m, dd)
            pes = []; des = []
            for k in range(90):
                comp.update(); tau = (dd.qfrc_applied + dd.qfrc_actuator).copy()
                dd.xfrc_applied[pid, :3] = F; mujoco.mj_step(m, dd); est.update(tau, dt)
                if 45 <= k <= 80:   # observer converged, arm still near nominal
                    cs = [c for c in est.contacts if c["body"].startswith("l_") or c["body"] == "left_palm"]
                    if cs:
                        c = cs[0]
                        pes.append(np.linalg.norm(c["point"] - dd.body(pid).xpos))
                        fn = c["force"][:2]; tn = F[:2]
                        des.append(math.degrees(math.acos(np.clip(
                            fn @ tn / (np.linalg.norm(fn)*np.linalg.norm(tn)+1e-9), -1, 1))))
            pos_err[dlabel].append(float(np.median(pes)) if pes else np.nan)
            dir_err[dlabel].append(float(np.median(des)) if des else np.nan)
    x = np.arange(len(links)); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for i, dl in enumerate(Fs):
        ax[0].bar(x + (i-0.5)*w, pos_err[dl], w, label=f"push {dl}")
        ax[1].bar(x + (i-0.5)*w, dir_err[dl], w, label=f"push {dl}")
    for a, t, yl in ((ax[0], "(a) contact-point error", "position error [m]"),
                     (ax[1], "(b) force-direction error", "direction error [deg]")):
        a.set_xticks(x); a.set_xticklabels([l.replace("l_", "").replace("left_", "") for l in links], rotation=20)
        a.set_title(t); a.set_ylabel(yl); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{FIG}/contact_localization.pdf"); plt.close(fig)
    return dict(links=links,
                pos_err={k: [round(x,3) for x in v] for k,v in pos_err.items()},
                dir_err={k: [round(x,1) for x in v] for k,v in dir_err.items()})


def _has(m, name):
    try: m.body(name); return True
    except KeyError: return False


def main():
    print("Running compliance study (figures -> docs/figures/) ...\n")
    r1 = study_backdrivability();  print("1 back-drivability:", r1)
    r2 = study_step_response();    print("2 step response:", {k: {kk: round(vv,3) for kk,vv in v.items()} for k,v in r2.items()})
    r3 = study_effort_speed();     print("3 effort-speed:", r3)
    r4 = study_deadband();         print("4 deadband:", r4["rec_fstatic"], "N rec")
    r5 = study_isotropy();         print("5 isotropy engage(s):", [round(x,2) for x in r5["engage"]],
                                         "dir_err(deg):", [round(x,1) for x in r5["dir_err"]])
    r6 = study_tip_envelope();     print("6 tip envelope:", r6)
    r7 = study_smoothness();       print("7 smoothness RMS jerk:", r7)
    r8 = study_contact_localization(); print("8 contact localization:", r8)
    print("\nFigures written to", FIG)


if __name__ == "__main__":
    main()

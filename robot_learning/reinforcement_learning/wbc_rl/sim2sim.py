"""sim2sim.py — the deploy path (deploy.py + policy.npz) driving NOVA in plain MuJoCo: no Isaac, no GPU, runs on a Mac.

A second physics engine around exactly what NOVA runs: deploy.Adapter (retarget, policy, action filter, lift ramp) ->
NOVA's control chain re-implemented from env.py (nova_hw shaper wn 40, wrist 80 + MIT PD with v_ff, gravity feed-forward
clipped at ff_max, neck pitch/roll servo trapezoid, base twist low-pass + accel limit + tip governor + swerve
steering, drive force at the floor, frictionless wheel spheres) -> MuJoCo. Operator: the clips in
data/clips.npz (5 EgoDex tasks, 2 LAFAN1 walks), fed as operator frames (head, wrist + palm axes, Z-up). The clip
file is derived from EgoDex / LAFAN1 and is not distributed: build it from your own motion library with
--make-clips (after motion_lib.py, see README).
Prints tracking, smoothness and every physical limit (SAFETY x datasheet) per clip.

    pip install mujoco numpy
    python sim2sim.py --make-clips [--motions data/motions_v4.npz]   # data/clips.npz: 5 EgoDex + 2 LAFAN1 clips, <= 20 s
    python sim2sim.py checkpoints/v25_policy.npz                      # all clips, headless, numbers
    python sim2sim.py checkpoints/v25_policy.npz --clip 5 --mode upper
    mjpython sim2sim.py checkpoints/v25_policy.npz --view --clip 0    # macOS viewer (mjpython ships with mujoco)
Viewer: green = operator targets (head, wrists), red = NOVA's tracked frames.
"""
import argparse, math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deploy as D                            # noqa: E402
import robot as R                             # noqa: E402

G, H, DT = 9.81, 0.005, D.DT                  # physics 5 ms, policy / gateway 20 ms (env.py)
WHEELS_B = np.array([(0.17, 0.17), (0.17, -0.17), (-0.15, 0.0)])
ARMATURE = {"arm_1": 0.05, "arm_2": 0.05, "arm_3": 0.03, "arm_4": 0.03, "torso": 0.2}   # robot.articulation_cfg
TWIST_TAU, STEER_RATE, KV, KW = 0.08, (math.pi / 2) / 0.17, 10.0, 8.0                  # env.py base model


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle(Ra, Rb):
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Nova:
    """MuJoCo NOVA with the real control chain between the adapter's commands and the joints."""
    def __init__(self):
        import mujoco
        self.mj = mujoco
        if not os.path.exists(R.TRAIN_MJCF): R.build_mjcf()          # shipped in data/; rebuilding needs NOVA_DESC (robot.py)
        spec = mujoco.MjSpec.from_file(R.TRAIN_MJCF)
        spec.option.timestep = H; spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; spec.option.impratio = 1.0
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], friction=[0, 0, 0], rgba=[0.85, 0.85, 0.82, 1])
        spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
        # NOVA's PD as implicit actuators (Isaac Lab ImplicitActuatorCfg: the solver integrates kp / kd), force capped
        # at SAFETY x peak; v_ff and gravity feed-forward enter as a setpoint offset so the cap covers the whole MIT torque
        for n in R.ACTUATED + R.HAND_WRIST:
            g = R.joint_group(n) if n not in R.LIFT else None
            kp, kd = (40000.0, 4000.0) if g is None else R.HW_GAINS[R.gain_group(g)][:2]
            fr = 3000.0 if g is None else R.SAFETY * R.JOINT_SPEC[g][1]
            a = spec.add_actuator(name=n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                                  biastype=mujoco.mjtBias.mjBIAS_AFFINE, forcelimited=True, forcerange=[-fr, fr])
            a.gainprm[0] = kp; a.biasprm[1] = -kp; a.biasprm[2] = -kd
        self.m = m = spec.compile(); self.d = mujoco.MjData(m)
        m.jnt_actfrclimited[:] = False                                 # URDF placeholder 100 N m (Isaac overrides it too)
        self.qa = [m.joint(n).qposadr[0] for n in R.ACTUATED]; self.va = [m.joint(n).dofadr[0] for n in R.ACTUATED]
        self.qh = [m.joint(n).qposadr[0] for n in R.HAND_WRIST]; self.vh = [m.joint(n).dofadr[0] for n in R.HAND_WRIST]
        for v in self.vh: m.dof_armature[v] = 0.01
        for n, v in zip(R.ACTUATED, self.va):
            m.dof_armature[v] = 1.0 if n in R.LIFT else ARMATURE.get(R.joint_group(n), 0.01)
        self.ip = [R.ACTUATED.index(n) for n in D.POS]; self.il = [0, 1]
        grp = [R.joint_group(n) for n in D.POS]
        self.kp = np.array([R.HW_GAINS[R.gain_group(g)][0] for g in grp]); self.kd = np.array([R.HW_GAINS[R.gain_group(g)][1] for g in grp])
        self.t_pk = np.array([R.SAFETY * R.JOINT_SPEC[g][1] for g in grp])
        self.v_lim = np.array([R.SAFETY * R.JOINT_SPEC[g][0] for g in grp]); self.t_rated = np.array([R.SAFETY * R.JOINT_SPEC[g][2] for g in grp])
        self.ff_max = np.array([R.FF_MAX_NM[g] for g in grp])
        self.servo = [D.POS.index(n) for n in ("neck_pitch", "neck_roll")]
        self.wn = np.full(len(D.POS), 40.0)                               # nova_hw shaper wn; wrist arm_5-7: WRIST_WN
        self.wn[[D.POS.index(f"{s}_arm_{j}") for s in ("left", "right") for j in (5, 6, 7)]] = WRIST_WN
        self.chassis = m.body("mabel_chassis").id
        # gravity feed-forward: each joint holds its subtree (env.gravity_ff, nominal masses)
        par = m.body_parentid
        def under(b, top):
            while b > 0:
                if b == top: return True
                b = par[b]
            return False
        self.jb = [m.jnt_bodyid[m.joint(n).id] for n in R.ACTUATED]
        self.sub = np.array([[under(b, jb) for b in range(m.nbody)] for jb in self.jb], float)
        self.trk = R.tracked_offsets(); self.trk_b = [m.body(self.trk[k][0]).id for k in ("head", "left_wrist", "right_wrist")]

    def reset(self):
        d = self.d; self.mj.mj_resetData(self.m, d)
        d.qpos[:3] = [0, 0, R.CHASSIS_Z + 0.002]; d.qpos[3:7] = [0, 0, 0, 1]          # URDF base faces -x: yaw pi
        for n, a in zip(R.ACTUATED, self.qa): d.qpos[a] = R.READY.get(n, 0.0)
        d.qpos[self.qa[0]] = d.qpos[self.qa[1]] = R.LIFT_STROKE / 2
        self.mj.mj_forward(self.m, d)
        q = self.q()
        self.sp, self.sp_v, self.v_ff, self.q_f = q[self.ip].copy(), np.zeros(18), np.zeros(18), q[self.ip].copy()
        self.lift = float(np.mean(q[:2])); self.tw_f = np.zeros(3); self.v_cmd = np.zeros(3); self.v_ref = np.zeros(3)
        self.steer = np.zeros(3); self.steer_cmd = np.zeros(3); self.gov = np.array([R.BASE_ACCEL, R.BASE_VMAX, R.BASE_WMAX, R.BASE_ACCEL * R.BASE_DECEL]); self.n = 0
        self.tau2 = np.zeros(18); self.tau = np.zeros(18); self.hw_g = np.zeros(2); self.hw_sp = np.zeros(2); self.hw_spv = np.zeros(2)

    def q(self): return self.d.qpos[self.qa].copy()
    def hq(self): return self.d.qpos[self.qh].copy()
    def qd(self): return self.d.qvel[self.va].copy()

    def odom(self):
        Rw = quat_mat(self.d.qpos[3:7]); yw = math.atan2(-Rw[1, 0], -Rw[0, 0])       # REP yaw = root yaw + pi
        return float(self.d.qpos[0]), float(self.d.qpos[1]), yw

    def twist(self):
        _, _, yw = self.odom(); v = self.d.qvel[:2]; c, s = math.cos(yw), math.sin(yw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], self.d.qvel[5]])

    def tracked_world(self):
        out = []
        for k, b in zip(("head", "left_wrist", "right_wrist"), self.trk_b):
            Rb = self.d.xmat[b].reshape(3, 3); out.append((self.d.xpos[b] + Rb @ self.trk[k][1], Rb @ self.trk[k][2]))
        return out

    def gravity(self):
        d, m = self.d, self.m
        mg = m.body_mass[:, None] * np.array([0, 0, -G])
        tau = []
        for i, jb in enumerate(self.jb):
            j = m.body_jntadr[jb]; ax = d.xaxis[j]
            if m.jnt_type[j] == self.mj.mjtJoint.mjJNT_SLIDE:
                tau.append(-(self.sub[i] * mg[:, 2]).sum() * ax[2])                  # lift: the screw carries it all
            else:
                r = d.xipos - d.xanchor[j]
                tau.append(-(np.cross(r, mg) * self.sub[i][:, None]).sum(0) @ ax)
        return np.array(tau)

    def governor(self, yw):
        """stability.py limits on the live COM (env._update_state, 10 Hz; drives coasting: speed cap lifted)."""
        com = self.d.subtree_com[self.chassis]; c, s = math.cos(yw), math.sin(yw); d = com[:2] - self.d.qpos[:2]
        cb = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]); mg = []
        for i in range(3):
            a, b, o = WHEELS_B[i], WHEELS_B[(i + 1) % 3], WHEELS_B[(i + 2) % 3]
            e = b - a; nn = np.array([e[1], -e[0]]) / np.linalg.norm(e); nn = nn if nn @ (o - a) > 0 else -nn
            mg.append((cb - a) @ nn)
        h = max(com[2], 0.1); acc = max(G * max(min(mg) - 0.03, 0.0) / h, 0.15)
        rc = max(np.linalg.norm(cb - 0.5 * (WHEELS_B[0] + WHEELS_B[1])), 0.02)
        self.gov = np.array([min(acc, R.BASE_ACCEL), R.BASE_VMAX, min(max(math.sqrt(acc / rc), 0.3), R.BASE_WMAX), min(min(acc, R.BASE_ACCEL) * R.BASE_DECEL, acc)])

    def swerve(self, tw):
        u = tw[0] - tw[2] * WHEELS_B[:, 1], tw[1] + tw[2] * WHEELS_B[:, 0]
        sp = np.hypot(*u); head = np.where(sp > 0.02, np.arctan2(u[1], u[0]), self.steer_cmd)
        turn = wrap(head - self.steer_cmd); flip = np.abs(turn) > np.pi / 2           # against the LAST COMMANDED heading
        turn = np.where(flip, wrap(turn + np.pi), turn); sp = np.where(flip, -sp, sp)
        self.steer_cmd = wrap(self.steer_cmd + turn)
        st = STEER_RATE * DT; self.steer = wrap(self.steer + np.clip(wrap(self.steer_cmd - self.steer), -st, st))
        v = sp * np.where(np.abs(turn) > R.STEER_COS_GATE, 0.0, np.cos(turn))          # base.py cos rule, no measured gate
        A = np.zeros((6, 3))
        for i, (x, y) in enumerate(WHEELS_B): A[2 * i] = [1, 0, -y]; A[2 * i + 1] = [0, 1, x]
        return np.linalg.pinv(A) @ np.stack([v * np.cos(self.steer), v * np.sin(self.steer)], -1).reshape(6)

    def command(self, cmd):
        """One 50 Hz gateway tick: joint goals, lift stage target, base twist (env._pre_physics_step + _apply_action)."""
        x, y, yw = self.odom()
        if self.n % 5 == 0: self.governor(yw)
        self.n += 1
        tw = np.array(cmd["twist"], float); sp = np.hypot(*tw[:2])
        tw[:2] *= min(self.gov[1] / max(sp, 1e-6), 1.0); tw[2] = np.clip(tw[2], -self.gov[2], self.gov[2])
        self.tw_f += (1 - math.exp(-DT / TWIST_TAU)) * (tw - self.tw_f)
        vxy = R.base_ramp(self.v_cmd[:2], self.tw_f[:2], self.gov[0], self.gov[3], DT, np)
        self.v_cmd = np.r_[vxy, self.v_cmd[2] + np.clip(self.tw_f[2] - self.v_cmd[2], -R.BASE_ALPHA * DT, R.BASE_ALPHA * DT)]
        self.v_ref = self.swerve(self.v_cmd)
        self.lift = cmd["lift"]; self.q_f = np.array([cmd["joints"][n] for n in D.POS])
        self.hw_g = np.array([cmd["joints"].get(n, 0.0) for n in R.HAND_WRIST])
        wn, sv = self.wn, self.servo                                    # nova_hw shaper (wrist 80, rest 40); neck pitch/roll: servo profile
        for _ in range(int(round(DT / H))):
            spv = self.sp_v + H * (wn * wn * (self.q_f - self.sp) - 2.0 * wn * self.sp_v)
            e = self.q_f[sv] - self.sp[sv]; v0 = self.sp_v[sv]
            vs = np.sign(e) * np.minimum(np.sqrt(2 * R.NECK_SERVO["acc"] * np.abs(e)), R.NECK_SERVO["speed"])
            spv[sv] = v0 + np.clip(vs - v0, -R.NECK_SERVO["acc"] * H, R.NECK_SERVO["acc"] * H); self.sp_v = spv
            self.sp = self.sp + H * self.sp_v
            e = self.hw_g - self.hw_sp; v0 = self.hw_spv                         # ORCA hinge: HLS3930 profile
            vs = np.sign(e) * np.minimum(np.sqrt(2 * R.HAND_SERVO["acc"] * np.abs(e)), R.HAND_SERVO["speed"])
            self.hw_spv = v0 + np.clip(vs - v0, -R.HAND_SERVO["acc"] * H, R.HAND_SERVO["acc"] * H)
            self.hw_sp = self.hw_sp + H * self.hw_spv
        self.v_ff = self.sp_v.copy(); self.v_ff[sv] = 0.0
        tau_g = self.gravity()                                          # once per tick, held (env: per policy step)
        for _ in range(int(round(DT / H))):
            self.substep(tau_g)
        self.tau2 += (DT / 5.0) * (self.tau ** 2 - self.tau2)

    def substep(self, tau_g):
        d, ip = self.d, self.ip
        q, qd = self.q(), self.qd()
        ff = self.kd * self.v_ff + np.clip(tau_g[ip], -self.ff_max, self.ff_max)
        tau = self.kp * (self.sp - q[ip]) + self.kd * (self.v_ff - qd[ip]) + np.clip(tau_g[ip], -self.ff_max, self.ff_max)
        self.tau = np.clip(tau, -self.t_pk, self.t_pk)                  # what the motors deliver (logged)
        d.ctrl[ip] = self.sp + ff / self.kp
        d.ctrl[self.il] = self.lift + tau_g[self.il] / 40000.0
        d.ctrl[len(R.ACTUATED):] = self.hw_sp
        # swerve drive: force at the floor under the chassis tracking the base twist, traction-limited
        _, _, yw = self.odom(); c, s = math.cos(yw), math.sin(yw); vb = self.v_ref
        v_ref = np.array([c * vb[0] - s * vb[1], s * vb[0] + c * vb[1]])
        b = self.chassis; com = d.xipos[b]; w = d.qvel[3:6]
        v = d.qvel[:2] + np.cross(w, com - d.qpos[:3])[:2]
        mt = self.m.body_subtreemass[b]
        F = mt * KV * (v_ref - v); Fn = np.linalg.norm(F); F *= min(0.8 * G * mt / max(Fn, 1e-6), 1.0)
        F3 = np.r_[F, 0.0]; tq = np.cross(np.r_[com[:2], 0.0] - com, F3); tq[2] += 4.0 * KW * (vb[2] - w[2])
        d.xfrc_applied[b] = np.r_[F3, tq]
        self.mj.mj_step(self.m, d)


CLIPS = os.path.join(HERE, "data", "clips.npz")
CLIP_NAMES = ("egodex:arrange_topple_dominoes", "egodex:assemble_disassemble_soft_legos", "egodex:insert_remove_usb",
              "egodex:basic fold", "egodex:wipe screen", "lafan:walk3_subject3", "lafan:walk3_subject1")


def make_clips(motions, out=CLIPS, seconds=20.0):
    """motion_lib's library -> the sim2sim clip set: for each CLIP_NAMES prefix the first matching clip (else the next
    unused clip of the same source), at most `seconds` long; arm_k / sh_k = robot / operator reach and shoulder width
    (per-clip medians, as env.py's load_motions)."""
    import robot as R
    import motion_lib as ML
    z = np.load(motions); meta = [str(m) for m in z["meta"]]
    fk = R.FK(); P = lambda j: fk.pose(fk.joint_child[j])[1]
    width = np.linalg.norm(P("left_arm_2") - P("right_arm_2"))
    n = lambda a, b, s, l: np.linalg.norm(z[a][s:s + l] - z[b][s:s + l], axis=1)
    used, arr, names = set(), {}, []
    for want in CLIP_NAMES:
        c = next((i for i, m in enumerate(meta) if m.startswith(want) and i not in used), None)
        if c is None: c = next((i for i, m in enumerate(meta) if m.split(":")[0] == want.split(":")[0] and i not in used), None)
        if c is None: continue
        used.add(c); k = len(names); names.append(meta[c])
        s0, l = int(z["starts"][c]), int(min(z["lens"][c], seconds * ML.FPS))
        for key in ("head_p", "head_q", "lw_p", "lw_q", "rw_p", "rw_q"):
            arr[f"{k}_{key}"] = z[key][s0:s0 + l].astype(np.float32)
        reach = 0.5 * (n("le", "ls", s0, l) + n("lw_p", "le", s0, l) + n("re", "rs", s0, l) + n("rw_p", "re", s0, l))
        arr[f"{k}_arm_k"] = np.float32(ML.ROBOT_REACH / np.median(reach))
        arr[f"{k}_sh_k"] = np.float32(width / np.median(n("ls", "rs", s0, l)))
    assert names, f"no clips in {motions}"
    np.savez_compressed(out, names=np.array(names), **arr)
    print(f"wrote {out}: {len(names)} clips"); [print("  ", m) for m in names]


def load_clip(z, c):
    g = lambda k: z[f"{c}_{k}"]
    Rm = lambda k: np.array([quat_mat(x) for x in g(k)])
    return dict(head_p=g("head_p").astype(float), head_R=Rm("head_q"), lw_p=g("lw_p").astype(float), lw_R=Rm("lw_q"),
                rw_p=g("rw_p").astype(float), rw_R=Rm("rw_q")), float(g("arm_k"))


WRIST_WN = 80.0                                                       # nova_hw NOVA_HW_WRIST_WN


def run(policy, clip, mode, delay, view, wrist=True):
    z = np.load(CLIPS)
    op_all, arm_k = load_clip(z, clip)
    sim = Nova(); sim.reset()
    ad = D.Adapter(D.Policy(policy), upper=(mode == "upper"), arm_k=arm_k)
    frame = lambda t: {k: v[t] for k, v in op_all.items()}
    ad.engage(frame(0), sim.q(), sim.odom())
    buf = [(np.zeros(22), None)] * (delay + 1)
    T = len(op_all["head_p"]); log = {k: [] for k in ("e_head", "e_hr", "e_lw", "e_rw", "e_wr", "acc", "qd", "tau", "lift_v", "base", "rms")}
    viewer = None
    if view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)
    hv_prev, hv_pp = None, None; lift_prev = sim.q()[:2]
    for t in range(T):
        q, qd, od, vb = sim.q(), sim.qd(), sim.odom(), sim.twist()
        ad.hq, ad.yaw = sim.hq(), od[2]
        tg = ad.targets(frame(t)); ad.blend_t += DT
        ad.euro(np.stack([tg["lw_R"], tg["rw_R"]]))
        obs = ad.observe(ad.target_obs(tg, od), q, qd, vb, np.zeros(3), tg)
        buf = [(ad.pi(obs), ad.solve_wrist(q, od) if wrist else None)] + buf[:-1]   # gateway + CAN + shaper latency (env: 1-4 ticks)
        cmd = ad.act(*buf[delay])
        sim.command(cmd)
        cur = sim.tracked_world()
        if t > 1.5 / DT:                                               # after the 1 s glide-in
            log["e_head"].append(np.linalg.norm(cur[0][0] - tg["head_p"]))
            log["e_hr"].append(angle(cur[0][1], tg["head_R"]))
            log["e_lw"].append(np.linalg.norm(cur[1][0] - tg["lw_p"])); log["e_rw"].append(np.linalg.norm(cur[2][0] - tg["rw_p"]))
            log["e_wr"].append([angle(cur[i][1], tg[k]) for i, k in ((1, "lw_R"), (2, "rw_R"))])
        hp = np.array([cur[1][0], cur[2][0]])
        if hv_pp is not None: log["acc"].append(np.linalg.norm((hp - 2 * hv_prev + hv_pp) / DT ** 2, axis=-1).mean())
        hv_pp, hv_prev = hv_prev, hp
        qd = sim.qd(); log["qd"].append(np.abs(qd[sim.ip]) / sim.v_lim); log["tau"].append(np.abs(sim.tau) / sim.t_pk); log["rms"].append(np.sqrt(sim.tau2) / sim.t_rated)
        lq = sim.q()[:2]; log["lift_v"].append(np.abs(lq - lift_prev).max() / DT); lift_prev = lq
        log["base"].append(np.abs(vb))
        if viewer is not None:
            if not viewer.is_running(): break
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for (p, _), col in [((tg[k + "_p"], None), (0.1, 0.8, 0.2, 0.8)) for k in ("head", "lw", "rw")] + [(c_, (0.9, 0.15, 0.1, 0.8)) for c_ in cur]:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    sim.mj.mjv_initGeom(g, sim.mj.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], p, np.eye(3).ravel(), np.array(col, np.float32))
                    viewer.user_scn.ngeom += 1
            viewer.sync()
            import time; time.sleep(max(0.0, DT - 0.002))
    a = {k: np.array(v) for k, v in log.items()}
    k0 = int(1.0 / DT)                    # skip the first second (settling)
    worst = lambda x: f"{float(x.max()):.2f} {D.POS[int(np.unravel_index(x.argmax(), x.shape)[-1])]}"
    soft = [D.POS.index(n) for n in R.OVERLOAD_SOFT]; hard = [i for i in range(18) if i not in soft]
    res = dict(clip=str(z["names"][clip]).split(",")[0], mode=mode, s=round(T * DT, 1),
               head_cm=round(100 * float(a["e_head"].mean()), 2), head_deg=round(float(a["e_hr"].mean()), 2),
               wrist_cm=round(50 * float(a["e_lw"].mean() + a["e_rw"].mean()), 2), wrist_deg_l_r=a["e_wr"].mean(0).round(1).tolist(), hand_acc=round(float(a["acc"][k0:].mean()), 2),
               speed=worst(a["qd"][k0:]), peak_torque=worst(a["tau"][k0:]), rated_rms=worst(np.where(np.isin(np.arange(18), hard), a["rms"], 0.0)),
               neck_servo_rms=round(float(a["rms"][:, soft].max()), 2), lift_mm_s=round(1000 * float(a["lift_v"][k0:].max()), 1),
               base_max=[round(float(x), 2) for x in a["base"].max(0)], fell=bool(abs(quat_mat(sim.d.qpos[3:7])[2, 2]) < 0.97))
    if viewer is not None: viewer.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=os.path.join(HERE, "checkpoints", "v25_policy.npz"))
    ap.add_argument("--clip", type=int, default=-1, help="index in data/clips.npz (-1 = all)")
    ap.add_argument("--clips", default=CLIPS, help="clip file (built by --make-clips)")
    ap.add_argument("--make-clips", action="store_true", help="build the clip file from --motions and exit")
    ap.add_argument("--motions", default=os.path.join(HERE, "data", "motions_v4.npz"))
    ap.add_argument("--mode", default="whole", choices=("whole", "upper"))
    ap.add_argument("--delay", type=int, default=2, help="policy ticks of output latency (training: 1-4)")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-wrist-ik", action="store_true", help="policies before v28 (no wrist solver)")
    ap.add_argument("--wrist-wn", type=float, default=80.0, help="nova_hw shaper wn on arm_5-7 (40 for policies before v35)")
    a = ap.parse_args()
    WRIST_WN = a.wrist_wn
    CLIPS = a.clips
    if a.make_clips:
        make_clips(a.motions, CLIPS); sys.exit(0)
    n = len(np.load(CLIPS)["names"])
    print("speed / peak_torque / rated_rms: worst joint as a share of SAFETY x datasheet (<= 1); neck servos: known soft overload; "
          "lift: per stage, cap 15 mm/s (30 mm/s column)")
    for c in (range(n) if a.clip < 0 else [a.clip]):
        print(run(a.policy, c, a.mode, a.delay, a.view, not a.no_wrist_ik), flush=True)

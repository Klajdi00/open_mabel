"""deploy.py — the whole-body RL policy on NOVA: Vision Pro teleop_frame + robot_state -> joint / base / lift commands.

numpy only (runs on the robot's CPU, no torch / Isaac). Everything that shapes the policy's input or output is the
SAME code or the same numbers as training (env.py); `parity` proves it against the sim, step by step:
  motion_lib.shoulders_from_head / frame_xy / YUP_TO_ZUP, robot.FK / tracked_offsets / READY / limits,
  motion_lib.wire_positions (the AVP hand joints -> world points), env.py's action filter, slews and lift ramp.

    $ISAACLAB_PY deploy.py export <model.pt> [out.npz]   # policy weights -> numpy (policy.npz)
    $ISAACLAB_PY deploy.py parity <model.pt> [--mode upper]  # adapter vs env, numbers printed
    $ISAACLAB_PY deploy.py kparity <model.pt>                # KinAdapter (kinematic, k3 on) vs kin.py
    $ISAACLAB_PY deploy.py sexport <student.pt> [out.npz]    # sonic.py student (token model) -> numpy
    $MJLAB_PY deploy.py sparity <student.pt>                  # SonicAdapter (OpenArm grippers) vs sonic.py (mjlab)

What the gateway must do with Adapter.step() output (not wired yet; the first motion on the robot is the operator's to run):
  joints  -> joint_command {name: rad} (torso, neck, 14 arm joints; nova_hw shapes and clamps as in training)
  lift    -> joint_command lift_lower / lift_upper (stage m; ramped here at <= 30 mm/s column, held 2 cm off the stops)
  twist   -> navJoystick via nav_from_twist (the base layer low-passes, accel-limits and governs as in training)
The built-in arm solver must be off (NOVA_ARM_RETARGET=0), and nothing else may command these joints.
"""
import math, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import motion_lib as ML                       # noqa: E402
import robot as R                             # noqa: E402

DT = 0.02                                     # policy / gateway tick (env decimation 4 x 5 ms)
ACTION_SCALE, LPF_HZ = 0.5, 3.0               # env.NovaWbcEnvCfg.action_scale / action_lpf_hz
WRIST_RES, WRIST_HZ = 0.0, 20.0               # env cfg.wrist_res / wrist_lpf_hz (wrist joints: the solver; policy residual off)
HAND_GUARD = (0.015, 0.035)                   # env cfg.hand_guard: m of hand clearance where the wrist command stops / runs free
EURO = dict(min_hz=1.5, beta=2.0, d_hz=1.0)   # env cfg.euro: one-euro filter on the received palms
ARM_K = 1.10                                  # ROBOT_REACH / operator reach (training median); Adapter.calibrate() per operator
EYE_Z0, EYE_FWD = 1.374, 0.096                # robot eye height at lift 0 (+2 x stage), torso line ahead of the base
GLIDE_S = 1.0                                 # engage: blend from the robot's own frames into the operator's
UPPER_DRIFT_TAU = 5.0                         # s, upper-body mode: slow walking drift removed (env.compute_targets)
BASE_CAP = np.array([R.BASE_VMAX, R.BASE_VMAX, R.BASE_WMAX])
POS = ["torso"] + R.NECK + R.ARM              # 18 position joints, action order (env.i_pos)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw(Rm):
    return math.atan2(Rm[1, 0], Rm[0, 0])


def rotlog(E):
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / 2
    sn = np.linalg.norm(v); th = math.atan2(sn, (np.trace(E) - 1) / 2)
    return v * th / (sn + 1e-9)


def six_d(Rm):
    return Rm[:, :2].reshape(6)


def orthonorm(M):
    x = M[:, 0] / np.linalg.norm(M[:, 0]); y = M[:, 1] - (M[:, 1] @ x) * x; y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], -1)


# ---------------------------------------------------------------- the Vision Pro frame -> operator frames (Z-up)
A = ML.YUP_TO_ZUP


def head_from_avp(M):
    """teleop_frame.head.transform (4x4, ARKit: x right, y up, z back) -> eye position, frame (x fwd, y left, z up).
    EgoDex's camera is OpenCV (x right, y down, z fwd): motion_lib builds the same frame from it."""
    Rd = A @ M[:3, :3]
    return A @ M[:3, 3], np.stack([-Rd[:, 2], -Rd[:, 0], Rd[:, 1]], -1)


def palm_from_avp(hand):
    """A HandPose dict -> wrist position + palm frame (x wrist -> middle knuckle, y little -> index), like EgoDex."""
    P = {k: A @ v for k, v in ML.wire_positions(hand).items()}
    if not all(k in P for k in ("wrist", "middle_knuckle", "index_knuckle", "little_knuckle")):
        return None
    w = P["wrist"]
    return w, ML.frame_xy(P["middle_knuckle"] - w, P["index_knuckle"] - P["little_knuckle"])


# ---------------------------------------------------------------- policy
class Policy:
    """rsl-rl actor (ELU MLP, empirical normaliser, eps 1e-2), deterministic mean."""
    def __init__(self, npz):
        z = np.load(npz); self.mean, self.std = z["mean"], z["std"]
        self.W = [(z[f"w{i}"], z[f"b{i}"]) for i in range(4)]

    def __call__(self, obs):
        obs = obs[:len(self.mean)]                    # a policy trained before an input block was added sees its own prefix
        x = (obs - self.mean) / (self.std + 1e-2)
        for i, (w, b) in enumerate(self.W):
            x = w @ x + b
            if i < 3: x = np.where(x > 0, x, np.expm1(x))
        return x


def export(pt, out):
    import torch
    sd = torch.load(pt, map_location="cpu", weights_only=False)["actor_state_dict"]
    np.savez(out, mean=sd["obs_normalizer._mean"][0].numpy(), std=sd["obs_normalizer._std"][0].numpy(),
             **{f"w{i}": sd[f"mlp.{2 * i}.weight"].numpy() for i in range(4)}, **{f"b{i}": sd[f"mlp.{2 * i}.bias"].numpy() for i in range(4)})
    print("wrote", out)


# ---------------------------------------------------------------- the adapter
def rotvec(u):
    t = np.linalg.norm(u)
    if t < 1e-12: return np.eye(3)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]) / t
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def trilinear(g, u):
    """g (nx, ny, nz) sampled at fractional grid indices u (n, 3), clamped at the border (env: grid_sample)."""
    n = np.array(g.shape) - 1; u = np.clip(u, 0, n); i = np.minimum(np.floor(u).astype(int), np.maximum(n - 1, 0)); f = u - i
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * g[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return out


class HandClearance:
    """env.clearance, the hands' share: each hand's smallest gap (m) to every part it may not touch (robot.collision_model
    spheres at robot.FK poses against the parts' distance fields). Only fingers may touch fingers."""
    def __init__(self, fk):
        parts, sph, pairs = R.collision_model(); self.fk = fk
        self.sl = [str(x[0]) for x in sph]; self.c = np.array([x[1] for x in sph], float); self.r = np.array([x[2] for x in sph], float)
        self.hand = np.array([[str(x[4]) in R.HAND and str(x[3]) == s for s in "lr"] for x in sph])
        self.links = sorted(set(self.sl) | {p["link"] for p in parts}); self.fields = []
        for j, p in enumerate(parts):
            ids = np.nonzero(pairs[:, j])[0]
            if len(ids) == 0: continue
            fh = "lr".index(p["side"]) if p["side"] in ("l", "r") and p["name"] in R.HAND else -1
            self.fields.append((p["link"], np.asarray(p["origin"], float), float(p["res"]), np.asarray(p["sdf"], float), ids, fh))

    def __call__(self, qd):
        pose = {l: self.fk.pose(l, qd) for l in self.links}
        P = np.array([pose[l][1] + pose[l][0] @ c for l, c in zip(self.sl, self.c)])
        ch = np.ones(2); self.total = 1.0                  # total: the smallest gap of any pair (kin.py clearance)
        for link, org, res, g, ids, fh in self.fields:
            Rl, pl = pose[link]
            dd = trilinear(g, ((P[ids] - pl) @ Rl - org) / res) - self.r[ids]
            self.total = min(self.total, float(dd.min()))
            for h in (0, 1):
                m = self.hand[ids, h]
                if m.any(): ch[h] = min(ch[h], dd[m].min())
            if fh >= 0: ch[fh] = min(ch[fh], dd.min())
        return ch


class Adapter:
    def __init__(self, policy, upper=False, arm_k=ARM_K):
        self.pi, self.upper, self.arm_k = policy, float(upper), arm_k
        self.fk = R.FK(); self.trk = R.tracked_offsets(self.fk)
        self.q_ready = np.array([R.READY.get(n, 0.0) for n in R.ACTUATED]); self.q_ready[:2] = R.LIFT_STROKE / 2
        lo = np.array([self.fk.J[self.fk.joint_child[n]]["lim"][0] for n in R.ACTUATED])
        hi = np.array([self.fk.J[self.fk.joint_child[n]]["lim"][1] for n in R.ACTUATED])
        ip = [R.ACTUATED.index(n) for n in POS]; self.ip = ip
        span = np.maximum((hi - lo)[ip] * 0.04, 0.08)
        self.q_lo, self.q_hi = lo[ip] + span, hi[ip] - span
        self.slew = np.array([min(R.HW_GAINS[R.gain_group(R.joint_group(n))][2], R.HEADROOM * R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0])
                              for n in POS])
        self.i_wr = [[POS.index(f"{s}_arm_{j}") for j in (5, 6, 7)] for s in ("left", "right")]
        self.i_wr_all = self.i_wr[0] + self.i_wr[1]
        hz = np.full(len(POS), LPF_HZ); hz[self.i_wr_all] = WRIST_HZ
        self.beta = 1.0 - np.exp(-2 * math.pi * hz * DT)
        self.wc = [R.wrist_chain(self.fk, s) for s in ("left", "right")]
        lh = np.array([self.fk.J[self.fk.joint_child[n]]["lim"] for n in R.HAND_WRIST])
        self.hw_lo, self.hw_hi = lh[:, 0] + 0.08, lh[:, 1] - 0.08                      # ORCA hinges (env.hw_lo / hw_hi)
        self.hq = np.zeros(2); self.hw_f1 = np.zeros(2); self.hw_f = np.zeros(2); self.wres = np.zeros((2, 3))
        self.hc = HandClearance(self.fk); self.clear = np.ones(2); self.w_safe = np.zeros((2, 4))
        self.engaged = False

    # -- robot side
    def tracked(self, q):
        """FK: head / left / right tracked frames in the REP base frame (URDF base faces -x)."""
        qd = dict(zip(R.ACTUATED, q)); qd.update(zip(R.HAND_WRIST, self.hq)); Rf = rot_z(math.pi); out_p, out_R = [], []
        for k in ("head", "left_wrist", "right_wrist"):
            body, p, Rb = self.trk[k]
            Rl, pl = self.fk.pose(body, qd)
            out_p.append(Rf @ (pl + Rl @ p)); out_R.append(Rf @ Rl @ Rb)
        return np.array(out_p), np.array(out_R)

    # -- engage: the operator's gaze heading and head -> the robot's (env._reset_idx anchor, deploy-computable)
    def engage(self, op, q, odom):
        """op: dict head_p/head_R (operator, Z-up); q: 20 joint positions (ACTUATED); odom: (x, y, yaw)."""
        x, y, yaw0 = odom
        h_yaw = yaw(op["head_R"])
        self.anc_yaw, self.anc_h = yaw0 - h_yaw, op["head_p"].copy()
        sl, sr = ML.shoulders_from_head(op["head_p"], h_yaw); sm = 0.5 * (sl + sr)
        off = rot_z(self.anc_yaw) @ (op["head_p"] - sm)
        self.anc_r = np.array([x + math.cos(yaw0) * EYE_FWD + off[0], y + math.sin(yaw0) * EYE_FWD + off[1],
                               EYE_Z0 + 2 * float(np.mean(q[:2]))])
        self.sm0 = self.sm_lp = sm[:2].copy()
        cp, cR = self.tracked(q); Rb = rot_z(yaw0)
        self.blend_p = Rb @ cp.T + np.array([x, y, R.CHASSIS_Z])[:, None]; self.blend_p = self.blend_p.T
        self.blend_R = Rb[None] @ cR; self.blend_t = 0.0
        self.q_f = self.q_f1 = q[self.ip].copy(); self.lift = float(np.mean(q[:2]))
        self.actions = np.zeros(22); self.hist = None; self.vb = np.zeros(3); self.engaged = True
        self.wik = np.concatenate([q[self.ip][self.i_wr_all].reshape(2, 3), self.hq[:, None]], -1)
        self.hw_f1 = self.hq.copy(); self.hw_f = self.hq.copy(); self.w_safe = self.wik.copy()
        self.oe_R = self.blend_R[1:].copy(); self.oe_s = np.zeros(2)

    def calibrate(self, ops, robot_reach=ML.ROBOT_REACH):
        """Operator reach from a short 'both arms straight out' hold (a list of operator frame dicts):
        the 90th percentile wrist distance from the head-estimated shoulders -> arm_k."""
        d = []
        for op in ops:
            sl, sr = ML.shoulders_from_head(op["head_p"], yaw(op["head_R"]))
            d += [np.linalg.norm(op["lw_p"] - sl), np.linalg.norm(op["rw_p"] - sr)]
        self.arm_k = robot_reach / float(np.percentile(d, 90)); return self.arm_k

    def targets(self, op, step=True):
        """Operator frames (Z-up human world) -> NOVA world targets (env.frame + env.compute_targets, no augmentation)."""
        hp, hR = op["head_p"], op["head_R"]
        sl, sr = ML.shoulders_from_head(hp, yaw(hR)); sm = 0.5 * (sl + sr)
        out = {"head_p": hp, "head_R": hR}
        for s in "lr":
            out[s + "w_p"] = ML.retarget_hand(hp, yaw(hR), op[s + "w_p"], self.arm_k, s); out[s + "w_R"] = op[s + "w_R"]
        if step: self.sm_lp = self.sm_lp + (1 - math.exp(-DT / UPPER_DRIFT_TAU)) * (sm[:2] - self.sm_lp)
        d = np.r_[-self.upper * (self.sm_lp - self.sm0), 0.0]
        Rn = rot_z(self.anc_yaw); W = lambda p: Rn @ (p + d - self.anc_h) + self.anc_r
        dz = np.array([0.0, 0.0, ML.WRIST_DZ])
        t = {"head_p": W(out["head_p"]), "head_R": Rn @ out["head_R"],
             "lw_p": W(out["lw_p"]) + dz, "lw_R": Rn @ out["lw_R"], "rw_p": W(out["rw_p"]) + dz, "rw_R": Rn @ out["rw_R"]}
        al = min(self.blend_t / getattr(self, "blend_T", GLIDE_S), 1.0); al = al * al * (3 - 2 * al)
        if not getattr(self, "blend_on", 1.0): al = 1.0
        for j, k in enumerate(("head", "lw", "rw")):
            t[k + "_p"] = self.blend_p[j] + al * (t[k + "_p"] - self.blend_p[j])
            t[k + "_R"] = orthonorm(self.blend_R[j] + al * (t[k + "_R"] - self.blend_R[j]))
        return t

    def target_obs(self, t, odom):
        x, y, yw = odom; Rb = rot_z(yw).T; pos = np.array([x, y, R.CHASSIS_Z])
        return np.concatenate([np.r_[Rb @ (t[k + "_p"] - pos), six_d(Rb @ t[k + "_R"])] for k in ("head", "lw", "rw")])

    def wrist_obs(self, t, q):
        """env.wrist_obs: ORCA hinge angles, solver residual, palm error vs the received target (base yaw frame)."""
        _, cR = self.tracked(q); Rb = rot_z(self.yaw).T
        pe = [rotlog(Rb @ t[k] @ cR[i].T) for i, k in ((1, "lw_R"), (2, "rw_R"))]
        return np.concatenate([self.hq, (Rb @ self.wres.T).T.reshape(6), np.concatenate(pe)])

    def observe(self, tob, q, qd, vb, age, t=None):
        """The 191 actor inputs, env._get_observations order (noise-free)."""
        self.hist = np.repeat(tob[None], 10, 0) if self.hist is None else np.concatenate([tob[None], self.hist[:-1]])
        cp, cR = self.tracked(q)
        cur = np.concatenate([cp.reshape(9), np.concatenate([six_d(r) for r in cR])])
        act = np.concatenate([self.actions[:4], (self.q_f - self.q_ready[self.ip]) / ACTION_SCALE])
        return np.concatenate([self.hist[0], self.hist[2], self.hist[5], cur, q - self.q_ready, 0.1 * qd, vb, act,
                               [self.upper], np.minimum(age, 1.0), self.wrist_obs(t, q)])

    def euro(self, Rx):
        """One-euro filter on the two received palm orientations (env.euro_step)."""
        al = lambda hz: 1.0 / (1.0 + 1.0 / (2 * math.pi * hz * DT))
        for j in range(2):
            Dm = self.oe_R[j].T @ Rx[j]
            v = np.array([Dm[2, 1] - Dm[1, 2], Dm[0, 2] - Dm[2, 0], Dm[1, 0] - Dm[0, 1]]) / 2
            sn = np.linalg.norm(v); th = math.atan2(sn, (np.trace(Dm) - 1) / 2)
            dv = v * th / (sn + 1e-9)
            self.oe_s[j] += al(EURO["d_hz"]) * (th / DT - self.oe_s[j])
            a = al(EURO["min_hz"] + EURO["beta"] * self.oe_s[j]); u = dv * a
            self.oe_R[j] = self.oe_R[j] @ rotvec(u)

    def solve_wrist(self, q, odom):
        """robot.wrist_ik for the measured forearm (env.solve_wrist)."""
        qd = dict(zip(R.ACTUATED, q)); Rb = rot_z(odom[2]) @ rot_z(math.pi)
        for j, wc in enumerate(self.wc):
            A = Rb @ R.chain_rot(wc["chain"], qd, np); ix = self.i_wr[j]
            L = np.r_[self.q_lo[ix], self.hw_lo[j]]; H = np.r_[self.q_hi[ix], self.hw_hi[j]]
            self.wik[j] = R.wrist_ik(A, self.oe_R[j], wc, self.wik[j], L, H, np, iters=8)
            self.wres[j] = R.wrist_err(A, self.oe_R[j], wc, self.wik[j], np)
        return self.wik.reshape(8)

    def act(self, a, ik=None):
        """Policy output (22) [+ the wrist solution] -> commands, env._pre_physics_step: twist, lift stage target,
        18 joint targets + the two ORCA wrist hinges (routed to the hand subsystems)."""
        a = np.clip(a, -5.0, 5.0); self.actions = a
        twist = np.tanh(a[:3]) * BASE_CAP * (1 - self.upper)
        rate = math.tanh(a[3]) * R.HEADROOM * R.LIFT_VLIM * (1 - self.upper)
        self.lift = min(max(self.lift + 0.5 * rate * DT, R.LIFT_MARGIN), R.LIFT_STROKE - R.LIFT_MARGIN)
        q = self.q_ready[self.ip] + ACTION_SCALE * a[4:]
        if ik is not None:
            ik = R.hand_guard(ik.reshape(2, 4), self.w_safe, self.clear, HAND_GUARD, np).reshape(8)
            q[self.i_wr_all] = ik[[0, 1, 2, 4, 5, 6]] + WRIST_RES * a[4:][self.i_wr_all]
            g = np.clip(ik[[3, 7]], self.hw_lo, self.hw_hi); b = self.beta[self.i_wr_all[0]]
            self.hw_f1 = self.hw_f1 + b * (g - self.hw_f1); self.hw_f = self.hw_f + b * (self.hw_f1 - self.hw_f)
        q = np.clip(q, self.q_lo, self.q_hi)
        self.q_f1 = self.q_f1 + self.beta * (q - self.q_f1)
        self.q_f = self.q_f + np.clip(self.beta * (self.q_f1 - self.q_f), -self.slew * DT, self.slew * DT)
        return dict(twist=twist, lift=self.lift, joints={**dict(zip(POS, self.q_f.tolist())), **dict(zip(R.HAND_WRIST, self.hw_f.tolist()))})

    def step(self, op, q, qd, odom, vb, age, hq=None):
        """One 50 Hz tick. op: latest received operator frames (held while a part is lost), q/qd: 20 joints,
        odom (x, y, yaw), vb measured twist (vx, vy, wz), age: s since a tracked head / left / right frame,
        hq: the two ORCA wrist hinges as measured (hand servos; None: assume they follow their command)."""
        self.hq = np.asarray(hq, float) if hq is not None else self.hw_f.copy(); self.yaw = odom[2]
        t = self.targets(op); self.blend_t += DT
        self.euro(np.stack([t["lw_R"], t["rw_R"]]))
        obs = self.observe(self.target_obs(t, odom), q, qd, vb, age, t)      # the residual of the last solve (as env.py)
        a = self.pi(obs)
        qh = dict(zip(R.ACTUATED, q)); qh.update(zip(R.HAND_WRIST, self.hq)); self.clear = self.hc(qh)
        return self.act(a, self.solve_wrist(q, odom)), obs


POSE_GUARD = 0.02                             # m: kin.py cfg.pose_guard


class KinAdapter(Adapter):
    """The kinematic retargeter (kin.py, k3 on) for a whole-body controller of your own: the reference is its own
    state. q, qd, base pose and twist are its previous output, never the encoders, so it sees what it saw in
    training. Joint windows, datasheet slews, lift 3 cm/s and the base accel ramp hold by construction, and a step
    that would bring any two parts closer than POSE_GUARD (and closer than they are) is cut to 1/2, 1/4 or held.
    step() -> dict(q: 20 ACTUATED + 2 ORCA hinges (rad / m), base: (x, y, yaw) in odom, twist: (vx, vy, wz) body)."""
    def engage(self, op, q, odom, hq=None):
        """Start from the robot as measured once (q: 20 ACTUATED, odom: (x, y, yaw), hq: the 2 ORCA hinges)."""
        self.hq = np.zeros(2) if hq is None else np.asarray(hq, float); self.yaw = odom[2]
        super().engage(op, q, odom)
        self.q = np.r_[np.asarray(q, float), self.hq]; self.qd = np.zeros(22)
        self.pose = np.array(odom, float); self.vb = np.zeros(3)
        self.i_lift = [R.ACTUATED.index(n) for n in ("lift_lower", "lift_upper")]
        self.hw_vlim = R.SAFETY * R.JOINT_SPEC["hand_wrist"][0]
        self.clear = self.hc(self._qd(self.q)); self.c_all = self.hc.total; self.guard_s = 1.0

    def _qd(self, q):
        d = dict(zip(R.ACTUATED, q[:20])); d.update(zip(R.HAND_WRIST, q[20:22])); return d

    def _pose_of(self, a_twist):
        """kin.apply's base: twist command -> nova_hw accel ramp -> integrate."""
        cmd = np.tanh(a_twist) * BASE_CAP * (1 - self.upper)
        acc = np.array([R.BASE_ACCEL])
        vxy = R.base_ramp(self.vb[None, :2], cmd[None, :2], acc, acc * R.BASE_DECEL, DT, np)[0]
        w = self.vb[2] + float(np.clip(cmd[2] - self.vb[2], -R.BASE_ALPHA * DT, R.BASE_ALPHA * DT))
        self.vb = np.r_[vxy, w]
        x, y, yw = self.pose; ym = yw + 0.5 * w * DT; c, s_ = math.cos(ym), math.sin(ym)
        self.pose = np.array([x + (c * vxy[0] - s_ * vxy[1]) * DT, y + (s_ * vxy[0] + c * vxy[1]) * DT,
                              math.remainder(yw + w * DT, 2 * math.pi)])

    def step(self, op, age):
        """One 50 Hz tick. op: latest received operator frames (held while lost), age: s since head / left / right."""
        self.hq = self.q[20:22].copy(); self.yaw = self.pose[2]; od = tuple(self.pose); q = self.q[:20]
        t = self.targets(op); self.blend_t += DT
        self.euro(np.stack([t["lw_R"], t["rw_R"]]))
        obs = self.observe(self.target_obs(t, od), q, self.qd[:20], self.vb, age, t)
        return self.advance(self.pi(obs), od), obs

    def advance(self, a, od):
        """kin.apply + kin.pose_guard for one action."""
        ik = self.solve_wrist(self.q[:20], od)            # self.clear: the hands' gaps of the current reference
        hw0 = self.hw_f.copy()
        self.act(a, ik)
        self.hw_f = hw0 + np.clip(self.hw_f - hw0, -R.HEADROOM * self.hw_vlim * DT, R.HEADROOM * self.hw_vlim * DT)
        self._pose_of(a[:3])
        q_old = self.q.copy(); q_new = q_old.copy()
        q_new[self.ip] = self.q_f; q_new[self.i_lift] = self.lift; q_new[20:22] = self.hw_f
        need = min(POSE_GUARD, self.c_all); self.guard_s = 1.0
        for f in (1.0, 0.5, 0.25, 0.0):
            qc = q_old + f * (q_new - q_old)
            ch = self.hc(self._qd(qc)); self.guard_s = f
            if f == 0.0 or self.hc.total >= need: break
        if self.guard_s < 1.0:                            # the filters continue from the pose sent
            self.q_f = qc[self.ip].copy(); self.hw_f = qc[20:22].copy(); self.lift = float(qc[self.i_lift[0]])
        self.clear, self.c_all = ch, self.hc.total
        self.qd = (qc - q_old) / DT; self.q = qc
        return dict(q=qc.copy(), base=self.pose.copy(), twist=self.vb.copy(), guard=self.guard_s)


def kparity(pt):
    """KinAdapter vs kin.py, step by step (faults, noise, augmentation off): env state in, compare what comes out."""
    import torch, kin
    cfg = kin.make_cfg(); cfg.faults = False; cfg.neck_aug = 0.0; cfg.twist_aug = 0.0
    cfg.noise = {k: 0.0 for k in cfg.noise}
    global POSE_GUARD
    POSE_GUARD = cfg.pose_guard = float(os.environ.get("KPARITY_GUARD", POSE_GUARD))   # e.g. 0.08: exercise the guard
    env = kin.KinEnv(cfg, num_envs=4, device="cuda:0", seed=7)
    from isaaclab_rl.rsl_rl.utils import create_rsl_rl_runner
    runner = create_rsl_rl_runner(env, kin.agent_cfg(1, "cuda:0"), log_dir=None); runner.load(pt, map_location="cuda:0")
    pol = runner.get_inference_policy(device="cuda:0")
    out = os.path.join(HERE, "runs", "policy_parity.npz"); export(pt, out)
    ad = KinAdapter(Policy(out)); e = 0
    T = lambda x: x[e].detach().cpu().numpy().astype(np.float64)
    worst = {k: 0.0 for k in ("q", "base_cm", "twist", "obs", "action", "guard_steps_env", "guard_steps_ad")}
    obs = env.get_observations()
    for i in range(500):
        od = (float(env.pos[e, 0]), float(env.pos[e, 1]), float(env.yaw[e]))
        if i == 0 or env.episode_length_buf[e] <= 1:
            h = env.frame(env.phase)
            ad.engage({"head_p": T(h["head_p"]), "head_R": T(h["head_R"])}, T(env.q)[:20], od, T(env.q)[20:22])
        # env state -> adapter (what the adapter would hold had it run alongside)
        ad.q, ad.qd, ad.pose, ad.vb = T(env.q).copy(), T(env.qd).copy(), np.array(od), T(env.v_b).copy()
        ad.q_f, ad.q_f1, ad.lift = T(env.q_f).copy(), T(env.q_f1).copy(), float(env.lift_tgt[e])
        ad.wik, ad.oe_R, ad.oe_s = T(env.wik).copy(), T(env.oe_R).copy(), T(env.oe_s).copy()
        ad.hw_f1, ad.hw_f, ad.w_safe = T(env.hw_f1).copy(), T(env.hw_f).copy(), T(env.w_safe).copy()
        ad.actions = T(env.actions).copy(); ad.yaw = od[2]
        ad.clear = ad.hc(ad._qd(ad.q)); ad.c_all = ad.hc.total
        with torch.inference_mode():
            a = pol(obs)
        worst["action"] = max(worst["action"], float(np.abs(ad.pi(T(obs["policy"])) - T(a)).max()))
        # the actor inputs the adapter builds from the env's targets and state
        if env.episode_length_buf[e] > 2:
            t = {k: T(env.tgt[k]) for k in ("head_p", "head_R", "lw_p", "lw_R", "rw_p", "rw_R")}
            ad.hist = None; ad.hq = ad.q[20:22].copy(); ad.wres = T(env.wres).copy()
            mine = ad.observe(ad.target_obs(t, od), ad.q[:20], ad.qd[:20], ad.vb, T(env.age), t)
            ref = T(obs["policy"])
            dd = np.abs(np.r_[mine[:27], mine[81:]] - np.r_[ref[:27], ref[81:]])   # history slots need the past
            worst["obs"] = max(worst["obs"], float(dd.max()))
        cmd = ad.advance(T(a), od)
        obs, _, done, _ = env.step(a)
        if done[e]: continue
        dq = np.abs(cmd["q"] - T(env.q))
        if dq.max() > worst["q"]: worst["q"], worst["q_at"] = float(dq.max()), (R.ACTUATED + R.HAND_WRIST)[int(dq.argmax())]
        worst["base_cm"] = max(worst["base_cm"], 100 * float(np.abs(cmd["base"][:2] - T(env.pos)[:2]).max()))
        worst["twist"] = max(worst["twist"], float(np.abs(cmd["twist"] - T(env.v_b)).max()))
        worst["guard_steps_env"] += int(float(env.guard_s[e]) < 1); worst["guard_steps_ad"] += int(cmd["guard"] < 1)
    print({k: round(v, 6) if isinstance(v, float) else v for k, v in worst.items()})


# ---------------------------------------------------------------- parity: adapter vs the training env
# ---------------------------------------------------------------- sonic.py (OpenArm v1 grippers, SONIC-style token model)
S_REACH = 0.63 / 0.6573 * 0.4426              # sonic.REACH: shoulder -> wrist centre of the gripper arm
S_ARM_K = 0.74                                # sonic: S_REACH / operator reach, training median; calibrate() per operator
S_WRIST_HZ = 6.0                              # sonic cfg.wrist_lpf_hz (arm_5-7)
TOKEN_LEVELS = 8                              # sonic.TokenNet.LEVELS


class SonicPolicy:
    """sonic.TokenModel (student) in numpy: normalise [vr, prop] -> encoder -> tanh -> FSQ -> decoder(token, prop)."""
    def __init__(self, npz):
        z = np.load(npz); self.mean, self.std, self.n_prop = z["mean"], z["std"], int(z["n_prop"])
        self.E = [(z[f"e{i}w"], z[f"e{i}b"]) for i in range(int(z["n_enc"]))]
        self.D = [(z[f"d{i}w"], z[f"d{i}b"]) for i in range(int(z["n_dec"]))]

    @staticmethod
    def _mlp(L, x):
        for i, (w, b) in enumerate(L):
            x = w @ x + b
            if i < len(L) - 1: x = np.where(x > 0, x, np.expm1(x))
        return x

    def __call__(self, obs):
        x = (obs - self.mean) / (self.std + 1e-2)
        h = (TOKEN_LEVELS - 1) / 2
        zq = np.round(np.tanh(self._mlp(self.E, x)) * h) / h
        return self._mlp(self.D, np.concatenate([zq, x[-self.n_prop:]]))


def sexport(pt, out):
    import torch
    sd = torch.load(pt, map_location="cpu", weights_only=False)["student"]
    lin = lambda pre: sorted({int(k.split(".")[2]) for k in sd if k.startswith(pre) and k.endswith(".weight")})
    E, D = lin("mlp.enc."), lin("mlp.dec.")
    np.savez(out, mean=sd["obs_normalizer._mean"][0].numpy(), std=sd["obs_normalizer._std"][0].numpy(),
             n_prop=sd["mlp.dec.0.weight"].shape[1] - sd["mlp.enc." + str(E[-1]) + ".weight"].shape[0], n_enc=len(E), n_dec=len(D),
             **{f"e{i}w": sd[f"mlp.enc.{j}.weight"].numpy() for i, j in enumerate(E)}, **{f"e{i}b": sd[f"mlp.enc.{j}.bias"].numpy() for i, j in enumerate(E)},
             **{f"d{i}w": sd[f"mlp.dec.{j}.weight"].numpy() for i, j in enumerate(D)}, **{f"d{i}b": sd[f"mlp.dec.{j}.bias"].numpy() for i, j in enumerate(D)})
    print("wrote", out)


class SonicAdapter(Adapter):
    """sonic.py's student on NOVA with the OpenArm v1 grippers: the policy places all 18 position joints (no wrist
    solver); the tracked wrist is the J7 wrist centre. Inputs are env.py / sonic.py's: received head + palms
    (history 0, 2, 5 ticks), stream ages, mode bit, then FK frames, joints, base twist, commands."""
    def __init__(self, policy, upper=False, arm_k=S_ARM_K):
        self.pi, self.upper, self.arm_k = policy, float(upper), arm_k
        self.fk = R.FK(R.GRIP_URDF); self.trk = R.tracked_offsets(self.fk, "gripper")
        self.q_ready = np.array([R.READY.get(n, 0.0) for n in R.ACTUATED]); self.q_ready[:2] = R.LIFT_STROKE / 2
        lo = np.array([self.fk.J[self.fk.joint_child[n]]["lim"][0] for n in R.ACTUATED])
        hi = np.array([self.fk.J[self.fk.joint_child[n]]["lim"][1] for n in R.ACTUATED])
        ip = [R.ACTUATED.index(n) for n in POS]; self.ip = ip
        span = np.maximum((hi - lo)[ip] * 0.04, 0.08)
        self.q_lo, self.q_hi = lo[ip] + span, hi[ip] - span
        self.slew = np.array([min(R.HW_GAINS[R.gain_group(R.joint_group(n))][2], R.HEADROOM * R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0])
                              for n in POS])
        self.i_wr_all = [POS.index(f"{s}_arm_{j}") for s in ("left", "right") for j in (5, 6, 7)]
        hz = np.full(len(POS), LPF_HZ); hz[self.i_wr_all] = S_WRIST_HZ
        self.beta = 1.0 - np.exp(-2 * math.pi * hz * DT)
        self.engaged = False

    def tracked(self, q):
        qd = dict(zip(R.ACTUATED, q)); Rf = rot_z(math.pi); out_p, out_R = [], []
        for k in ("head", "left_wrist", "right_wrist"):
            body, p, Rb = self.trk[k]
            Rl, pl = self.fk.pose(body, qd)
            out_p.append(Rf @ (pl + Rl @ p)); out_R.append(Rf @ Rl @ Rb)
        return np.array(out_p), np.array(out_R)

    def engage(self, op, q, odom):
        x, y, yaw0 = odom
        h_yaw = yaw(op["head_R"])
        self.anc_yaw, self.anc_h = yaw0 - h_yaw, op["head_p"].copy()
        sl, sr = ML.shoulders_from_head(op["head_p"], h_yaw); sm = 0.5 * (sl + sr)
        off = rot_z(self.anc_yaw) @ (op["head_p"] - sm)
        self.anc_r = np.array([x + math.cos(yaw0) * EYE_FWD + off[0], y + math.sin(yaw0) * EYE_FWD + off[1], EYE_Z0 + 2 * float(np.mean(q[:2]))])
        self.sm0 = self.sm_lp = sm[:2].copy()
        cp, cR = self.tracked(q); Rb = rot_z(yaw0)
        self.blend_p = (Rb @ cp.T + np.array([x, y, R.CHASSIS_Z])[:, None]).T
        self.blend_R = Rb[None] @ cR; self.blend_t = 0.0
        self.q_f = self.q_f1 = q[self.ip].copy(); self.lift = float(np.mean(q[:2]))
        self.actions = np.zeros(22); self.hist = None; self.engaged = True

    def calibrate(self, ops, robot_reach=S_REACH):
        return Adapter.calibrate(self, ops, robot_reach)

    def observe(self, tob, q, qd, vb, age):
        """The student's inputs [vr 85, prop 92] in sonic._get_observations order (noise-free)."""
        self.hist = np.repeat(tob[None], 10, 0) if self.hist is None else np.concatenate([tob[None], self.hist[:-1]])
        cp, cR = self.tracked(q)
        cur = np.concatenate([cp.reshape(9), np.concatenate([six_d(r) for r in cR])])
        act = np.concatenate([self.actions[:4], (self.q_f - self.q_ready[self.ip]) / ACTION_SCALE])
        vr = np.concatenate([self.hist[0], self.hist[2], self.hist[5], np.minimum(age, 1.0), [self.upper]])
        return np.concatenate([vr, cur, q - self.q_ready, 0.1 * qd, vb, act])

    def act(self, a):
        """Policy output (22) -> commands (sonic._pre_physics_step): twist, lift stage target, 18 joint targets."""
        a = np.clip(a, -5.0, 5.0); self.actions = a
        twist = np.tanh(a[:3]) * BASE_CAP * (1 - self.upper)
        rate = math.tanh(a[3]) * R.HEADROOM * R.LIFT_VLIM * (1 - self.upper)
        self.lift = min(max(self.lift + 0.5 * rate * DT, R.LIFT_MARGIN), R.LIFT_STROKE - R.LIFT_MARGIN)
        q = np.clip(self.q_ready[self.ip] + ACTION_SCALE * a[4:], self.q_lo, self.q_hi)
        self.q_f1 = self.q_f1 + self.beta * (q - self.q_f1)
        self.q_f = self.q_f + np.clip(self.beta * (self.q_f1 - self.q_f), -self.slew * DT, self.slew * DT)
        return dict(twist=twist, lift=self.lift, joints=dict(zip(POS, self.q_f.tolist())))

    def step(self, op, q, qd, odom, vb, age):
        """One 50 Hz tick: op = the latest received operator frames (held while a part is lost), q / qd: 20 joints,
        odom (x, y, yaw), vb measured twist, age: s since a tracked head / left / right frame."""
        self.yaw = odom[2]
        t = self.targets(op); self.blend_t += DT
        obs = self.observe(self.target_obs(t, odom), q, qd, vb, age)
        return self.act(self.pi(obs)), obs


def sparity(pt):
    """sonic.py (mjlab) with faults, noise, DR, delay and augmentation off; at every step the adapter is given what NOVA
    would give it (operator frames, joints, odometry, the sim's anchor) and its targets, inputs, policy and commands
    compared. Run with mjlab's venv ($MJLAB_PY)."""
    import torch
    import sonic as S
    sc = S.SonicCfg(faults=False, noise=dict(q=0.0, qd=0.0, base=0.0, cur=0.0), neck_aug=0.0, twist_aug=0.0, action_delay=(0, 0),
                    dr_physics=False, push_force=(0.0, 0.0), upper_share=0.0, egodex_share=1.0, avp_share=0.0)
    c = S.make_env_cfg(4, sc); c.seed = 7
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    wenv = RslRlVecEnvWrapper(ManagerBasedRlEnv(c, device="cuda:0")); env = wenv.unwrapped.nova
    obs = wenv.get_observations()
    ck = torch.load(pt, map_location=env.device, weights_only=False)
    pol = S.TokenModel(obs, {"actor": ["vr", "prop"]}, "actor", 22, obs_normalization=True).to(env.device)
    pol.load_state_dict(ck["student"]); pol.eval()
    out = os.path.join(HERE, "runs", "sonic_parity.npz"); sexport(pt, out)
    ad = SonicAdapter(SonicPolicy(out)); e = 0
    T = lambda x: x[e].detach().cpu().numpy().astype(np.float64)
    worst = {k: 0.0 for k in ("targets_cm", "obs", "action", "q_f", "lift")}
    for i in range(400):
        q = T(env.robot.data.joint_pos.torch[:, env.jid])
        pos, yw, _ = env.base_pose(); od = (float(pos[e, 0]), float(pos[e, 1]), float(yw[e]))
        if i == 0 or env.episode_length_buf[e] <= 1:
            h = env.frame(env.phase); ad.engage({"head_p": T(h["head_p"]), "head_R": T(h["head_R"])}, q, od)
        ad.anc_yaw, ad.anc_h, ad.anc_r = float(env.anc_yaw[e]), T(env.anc_h), T(env.anc_r)
        ad.sm0, ad.sm_lp = T(env.sm0), T(env.sm_lp)
        ad.blend_p, ad.blend_R, ad.blend_t, ad.blend_T = T(env.blend_p), T(env.blend_R), float(env.blend_t[e]) + DT, float(env.blend_T[e])
        ad.blend_on = float(env.blend_on[e])
        with torch.inference_mode():
            a = pol(obs)
        ad.q_f, ad.q_f1, ad.lift = T(env.q_f), T(env.q_f1), float(env.lift_tgt[e]); ad.actions = T(env.actions)
        cmd = ad.act(T(a))
        ph = torch.minimum((env.phase + env.rate).clamp_min(0), env.c_len[env.c_id].float() - 1.001)
        k = ph.floor(); i0 = env.c_start[env.c_id] + k.long(); al = (ph - k)[:, None]
        ad.arm_k = float(env.m["arm_k"][i0][e, 0])
        hh = env.frame(env.phase + env.rate)
        Hm = np.eye(4); Rz_ = T(hh["head_R"]); Hm[:3, :3] = A.T @ np.stack([-Rz_[:, 1], Rz_[:, 2], -Rz_[:, 0]], -1); Hm[:3, 3] = A.T @ T(hh["head_p"])
        op = dict(zip(("head_p", "head_R"), head_from_avp(Hm)))
        for s in "lr":
            op[s + "w_p"] = T(env.m[s + "w_p"][i0] * (1 - al) + env.m[s + "w_p"][i0 + 1] * al)
            op[s + "w_R"] = T(E_orthonorm(env.m[s + "w_R"][i0] * (1 - al[..., None]) + env.m[s + "w_R"][i0 + 1] * al[..., None]))
        obs, _, done, _ = wenv.step(a)
        if done[e]: continue
        worst["q_f"] = max(worst["q_f"], float(np.abs(ad.q_f - T(env.q_f)).max()))
        worst["lift"] = max(worst["lift"], abs(cmd["lift"] - float(env.lift_tgt[e])))
        if env.episode_length_buf[e] <= 2 or (float(env.blend_t[e]) < 0.5 * DT and env.episode_length_buf[e] > 3): continue
        t = ad.targets(op)
        worst["targets_cm"] = max(worst["targets_cm"], max(100 * float(np.abs(t[k_ + "_p"] - T(env.tgt[k_ + "_p"])).max()) for k_ in ("head", "lw", "rw")))
        q = T(env.robot.data.joint_pos.torch[:, env.jid]); qd = T(env.robot.data.joint_vel.torch[:, env.jid])
        pos, yw, _ = env.base_pose(); od = (float(pos[e, 0]), float(pos[e, 1]), float(yw[e]))
        ad.actions = T(env.actions); ad.q_f = T(env.q_f)
        ad.hist = np.concatenate([T(env.hist)[1:], T(env.hist)[-1:]])     # rows 1.. become rows 0.. after observe's shift
        mine = ad.observe(ad.target_obs(t, od), q, qd, T(env.vb_obs), T(env.age))
        dd = np.abs(mine - T(obs["policy"]))
        if os.environ.get("PARITY_DEBUG") and dd.max() > 0.005: print(f"obs step {i}: [{int(dd.argmax())}] {dd.max():.4f}")
        worst["obs"] = max(worst["obs"], float(dd.max()))
        worst["action"] = max(worst["action"], float(np.abs(ad.pi(T(obs["policy"])) - T(pol(obs))).max()))
    print({k: round(v, 6) for k, v in worst.items()})


def E_orthonorm(M):
    from wbc_core import orthonorm
    return orthonorm(M)


def parity(pt, mode):
    """Run env.py with faults, noise and augmentation off; at every step feed the adapter what NOVA would give it
    (AVP-format head / hand matrices rebuilt from the operator frames, joint states, odometry) and compare."""
    import torch
    sys.argv = [sys.argv[0], "--checkpoint", pt, "--source", "egodex", "--seed", "7", "--mode", mode]
    import play, env as E
    cfgs = {}
    def tweak(cfg):
        cfg.faults = False; cfg.noise = dict(q=0.0, qd=0.0, base=0.0); cfg.neck_aug = 0.0; cfg.twist_aug = 0.0
        cfg.action_delay = (0, 0); cfg.dr_physics = False; cfg.push_force = (0.0, 0.0); cfgs["c"] = cfg
    E.NovaWbcEnvCfg.__post_init__orig = E.NovaWbcEnvCfg.__post_init__
    def post(self): E.NovaWbcEnvCfg.__post_init__orig(self); tweak(self)
    E.NovaWbcEnvCfg.__post_init__ = post
    env = play.make_env(4, render=False); pol = play.load_policy(env)
    out = os.path.join(HERE, "runs", "policy_parity.npz"); export(pt, out)
    ad = Adapter(Policy(out), upper=(mode == "upper"))
    obs, _ = env.reset(); e = 0
    T = lambda x: x[e].detach().cpu().numpy().astype(np.float64)
    worst = {k: 0.0 for k in ("targets_cm", "obs", "action", "q_f", "lift")}
    for i in range(400):
        q = T(env.robot.data.joint_pos.torch[:, env.jid]); qd = T(env.robot.data.joint_vel.torch[:, env.jid])
        pos, yw, _ = env.base_pose(); od = (float(pos[e, 0]), float(pos[e, 1]), float(yw[e]))
        if i == 0 or env.episode_length_buf[e] <= 1:                   # (re)engage on the env's own reset
            h = env.frame(env.phase)
            ad.hq, ad.yaw = T(env.robot.data.joint_pos.torch[:, env.jid_hw]), od[2]
            ad.engage({"head_p": T(h["head_p"]), "head_R": T(h["head_R"])}, q, od)
            worst["engage_yaw"] = max(worst.get("engage_yaw", 0.0), abs(math.remainder(ad.anc_yaw - float(env.anc_yaw[e]), 2 * math.pi)))
            worst["engage_cm"] = max(worst.get("engage_cm", 0.0), 100 * float(np.abs(ad.anc_r[:2] - T(env.anc_r)[:2]).max()))
        # the sim re-anchors when it chains clips (a live operator never does): follow its anchor, test the mapping
        ad.anc_yaw, ad.anc_h, ad.anc_r = float(env.anc_yaw[e]), T(env.anc_h), T(env.anc_r)
        ad.sm0, ad.sm_lp = T(env.sm0), T(env.sm_lp)
        ad.blend_p, ad.blend_R, ad.blend_t, ad.blend_T = T(env.blend_p), T(env.blend_R), float(env.blend_t[e]) + DT, float(env.blend_T[e])   # env advances it first
        ad.blend_on = float(env.blend_on[e])
        with torch.inference_mode():
            a = pol(obs)
        ph = torch.minimum((env.phase + env.rate).clamp_min(0), env.c_len[env.c_id].float() - 1.001)
        i0 = env.c_start[env.c_id] + ph.floor().long()
        ad.arm_k = float(env.m["arm_k"][i0][e, 0])
        ad.q_f, ad.q_f1, ad.lift = T(env.q_f), T(env.q_f1), float(env.lift_tgt[e])
        ad.wik, ad.oe_R, ad.oe_s = T(env.wik).copy(), T(env.oe_R).copy(), T(env.oe_s).copy()
        ad.hw_f1, ad.hw_f = T(env.hw_f1).copy(), T(env.hw_f).copy(); ad.w_safe = T(env.w_safe).copy()
        qh = dict(zip(R.ACTUATED, T(env.robot.data.joint_pos.torch[:, env.jid]))); qh.update(zip(R.HAND_WRIST, T(env.robot.data.joint_pos.torch[:, env.jid_hw])))
        ad.clear = ad.hc(qh)
        if env.episode_length_buf[e] > 0: worst["clear_hand_mm"] = max(worst.get("clear_hand_mm", 0.0), 1000 * float(np.abs(np.minimum(ad.clear, 0.1) - np.minimum(T(env.clear_hand), 0.1)).max()))
        worst["guard_active"] = worst.get("guard_active", 0) + int((ad.clear < HAND_GUARD[1]).any())
        ik = ad.solve_wrist(T(env.robot.data.joint_pos.torch[:, env.jid]), od) if env.cfg.wrist_ik else None
        cmd = ad.act(T(a), ik)                                          # the adapter's commands for this action
        oe_s0 = ad.oe_s.copy()
        # what the Vision Pro would send for the operator at this phase (before the env advances it)
        h = env.frame(env.phase + env.rate)
        Hm = np.eye(4); Rz_ = T(h["head_R"]); Hm[:3, :3] = A.T @ np.stack([-Rz_[:, 1], Rz_[:, 2], -Rz_[:, 0]], -1); Hm[:3, 3] = A.T @ T(h["head_p"])
        op = dict(zip(("head_p", "head_R"), head_from_avp(Hm)))
        raw = env.m
        for s in "lr":                                                  # unscaled operator wrist + palm frame
            ph = env.phase + env.rate; ph = torch.minimum(ph.clamp_min(0), env.c_len[env.c_id].float() - 1.001)
            k = ph.floor(); i0 = env.c_start[env.c_id] + k.long(); al = (ph - k)[:, None]
            op[s + "w_p"] = T(raw[s + "w_p"][i0] * (1 - al) + raw[s + "w_p"][i0 + 1] * al)
            op[s + "w_R"] = T(E.orthonorm(raw[s + "w_R"][i0] * (1 - al[..., None]) + raw[s + "w_R"][i0 + 1] * al[..., None]))
        obs, _, term, tout, _ = env.step(a)
        if term[e] or tout[e]: continue
        worst["q_f"] = max(worst["q_f"], float(np.abs(ad.q_f - T(env.q_f)).max()))
        worst["wrist_ik"] = max(worst.get("wrist_ik", 0.0), float(np.abs(ad.wik - T(env.wik)).max()))
        worst["hinge_cmd"] = max(worst.get("hinge_cmd", 0.0), float(np.abs(ad.hw_f - T(env.hw_f)).max()))
        ad.oe_s = oe_s0; ad.euro(np.stack([T(env.tgt["lw_R"]), T(env.tgt["rw_R"])]))              # one-euro: same input
        worst["euro_deg"] = max(worst.get("euro_deg", 0.0), max(math.degrees(math.acos(np.clip((np.trace(ad.oe_R[j].T @ T(env.oe_R)[j]) - 1) / 2, -1, 1))) for j in range(2)))
        worst["lift"] = max(worst["lift"], abs(cmd["lift"] - float(env.lift_tgt[e])))
        if env.episode_length_buf[e] <= 2: continue                     # the env starts its glide one tick after a reset
        if float(env.blend_t[e]) < 0.5 * DT and env.episode_length_buf[e] > 3: continue   # the sim chained clips this tick
        t = ad.targets(op)
        for k in ("head", "lw", "rw"):
            err = 100 * float(np.abs(t[k + "_p"] - T(env.tgt[k + "_p"])).max())
            if os.environ.get("PARITY_DEBUG") and err > 0.15:
                print(f"step {i} ep {int(env.episode_length_buf[e])} {k} {err:.2f} cm blend_t {ad.blend_t:.2f}/{float(env.blend_t[e]):.2f} on {ad.blend_on}")
            worst["targets_cm"] = max(worst["targets_cm"], err)
        crit = T(env._get_observations()["critic"])
        q = T(env.robot.data.joint_pos.torch[:, env.jid]); qd = T(env.robot.data.joint_vel.torch[:, env.jid])
        pos, yw, _ = env.base_pose(); od = (float(pos[e, 0]), float(pos[e, 1]), float(yw[e]))
        lin = T(env.robot.data.root_link_lin_vel_w.torch); ang = T(env.robot.data.root_link_ang_vel_w.torch)
        vb = np.array([math.cos(od[2]) * lin[0] + math.sin(od[2]) * lin[1], -math.sin(od[2]) * lin[0] + math.cos(od[2]) * lin[1], ang[2]])
        ad.actions = T(env.actions); ad.q_f = T(env.q_f)
        tob = ad.target_obs(t, od); ad.hist = None
        ad.hq, ad.yaw, ad.wres = T(env.robot.data.joint_pos.torch[:, env.jid_hw]), od[2], T(env.wres)
        mine = ad.observe(tob, q, qd, vb, np.zeros(3), t)
        clean = np.concatenate([mine[0:27], mine[81:108], mine[108:128], mine[128:148], mine[148:151], mine[151:173], mine[177:191]])
        dd = np.abs(clean - np.concatenate([crit[:119], crit[164:178]]))
        if os.environ.get("PARITY_DEBUG") and dd.max() > 0.005:
            names = ["tgt"] * 27 + ["cur"] * 27 + ["q"] * 20 + ["qd"] * 20 + ["vb"] * 3 + ["act"] * 22 + ["wrist"] * 14
            print(f"obs step {i}: {names[int(dd.argmax())]}[{int(dd.argmax())}] {dd.max():.4f}")
        worst["obs"] = max(worst["obs"], float(dd.max()))
        worst["action"] = max(worst["action"], float(np.abs(ad.pi(T(obs["policy"])) - T(pol(obs))).max()))
    print({k: round(v, 6) for k, v in worst.items()})


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "policy.npz"))
    elif sys.argv[1] == "kparity":
        kparity(sys.argv[2])
    elif sys.argv[1] == "sexport":
        sexport(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "sonic_policy.npz"))
    elif sys.argv[1] == "sparity":
        sparity(sys.argv[2])
    elif sys.argv[1] == "parity":
        parity(sys.argv[2], "upper" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "upper" else "whole")

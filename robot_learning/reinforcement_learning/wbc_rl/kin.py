"""kin.py — kinematics-only whole-body retargeter: Vision Pro head + wrists in, a feasible NOVA reference out.

An external whole-body controller handles dynamics (design rule: "the retargeting should just be kinematics, no
dynamics"). So this env has no physics engine, motors, gravity or tipping: the policy's command IS the robot's pose.
  state     = the reference itself: 20 joints + 2 ORCA wrist hinges, base pose (x, y, yaw), all integrated here
  limits    = by construction: joint windows (env.py's clamp), datasheet speeds x HEADROOM (NOVA's slews), lift 3 cm/s
              column, base 0.6 m/s / 1.2 rad/s with nova_hw's accel ramp (0.8 m/s^2, brake 1.5x, yaw 2.5 rad/s^2)
  safety    = env.py's collision model on batched FK (torch): a gap under 1 cm ends the episode, 4 cm band penalised
  inputs    = env.py's 191 actor inputs, fed from the reference (exact on the robot too: no encoders, no odometry),
              Vision Pro stream faults / tremor / latency as in env.py
  outputs   = env.py's 22 actions: base twist, lift rate, torso, neck, 14 arms; wrist joints from the exact solver
Targets, retarget, stream faults, mirror maps and the reward shape are env.py's own code (imported). rsl_rl trains it
directly (no Isaac sim); a run warm-started from an env.py checkpoint keeps the same network shapes.

    PY=$ISAACLAB_PY
    $PY kin.py train --run k1 --iters 3000 [--checkpoint checkpoints/v38_73749.pt]   # -> logs/rsl_rl/nova_kin/<date>_k1
    $PY kin.py eval <model.pt> [--envs 512 --seconds 60 --no-faults --out m.json]    # tracking, palm by turn speed, safety
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from datetime import datetime

import numpy as np
import torch
from tensordict import TensorDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env as E                     # noqa: E402  (helpers, targets, stream faults, mirror maps)
import robot as R                   # noqa: E402
from env import rot_z, rot_y, rot_x, rot_angle, wrap, rotlog, six_d, orthonorm   # noqa: E402

DT = 0.02


def make_cfg():
    c = E.NovaWbcEnvCfg()
    c.action_delay = (0, 0)                           # the controller's lag is not the retargeter's
    c.action_lpf_hz = 3.0                             # 2nd-order reference filter, as env.py / deploy.py
    c.stream = dict(c.stream, odom=(0.0, 0.0, 0.0))   # the base pose is the reference's own: exact
    c.w = dict(head_p=3.0, head_r=1.5, wrist_p=4.0, wrist_p_mid=4.0, wrist_p_fine=5.0, wrist_r=8.0, track_lin=-1.0,
               hand_vel=1.0, body_yaw=0.6, body_xy=0.4, lean=1.0, lean_excess=-5.0, neck=0.4, elbow=1.5,
               base_use=-0.1, base_acc=-0.1, lift_use=-2.5, act_sat=-0.5, action_rate=-0.06, action_acc=-0.01,
               hand_acc=-0.03, palm_acc=-1e-4, arm_acc=-4e-5, qf_acc=-3e-5, wrist_unreach=-3.0, joint_vel=-1e-3,
               joint_acc=-2e-7, joint_limit=-10.0, limits=-6.0, twist=-0.05, clearance=-40.0,
               terminated=-500.0, collide=-4500.0, alive=4.0, guard=-2.0,
               head_far=2.0, body_far=1.0)                    # k6: wide kernels, still pull when 30-60 cm behind
    c.drift_stop = False                              # k4 (k1-k3: True)
    c.pose_guard = 0.02                               # m: a step may not take the clearance below this (or below
                                                      # where it already is); it is cut back instead (k3)
    return c


class _V:                                              # stands in for Isaac's robot.data.<field>.torch (stream())
    def __init__(self, t): self.torch = t


class KinEnv:
    """rsl_rl VecEnv. Reuses NovaWbcEnv's robot-independent methods (bound below)."""
    frame = E.NovaWbcEnv.frame
    compute_targets = E.NovaWbcEnv.compute_targets
    load_motions = E.NovaWbcEnv.load_motions
    chain = E.NovaWbcEnv.chain
    stream = E.NovaWbcEnv.stream
    stream_reset = E.NovaWbcEnv.stream_reset
    euro_step = E.NovaWbcEnv.euro_step
    target_obs = E.NovaWbcEnv.target_obs
    act_obs = E.NovaWbcEnv.act_obs

    def __init__(self, cfg=None, num_envs=4096, device="cuda:0", seed=0):
        torch.manual_seed(seed)
        self.cfg = cfg or make_cfg(); self.num_envs = N = num_envs; self.device = dev = device
        self.num_actions = 22; self.step_dt = DT
        self.max_episode_length = int(round(self.cfg.episode_length_s / DT))
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=dev)
        fk = R.FK(); self.fk = fk
        tt = lambda x: torch.tensor(np.array(x, dtype=np.float32), device=dev)
        # ---- joints: ACTUATED (20) + ORCA hinges (2)
        self.jn = R.ACTUATED + R.HAND_WRIST
        names = R.ACTUATED
        self.i_lift = [names.index(n) for n in R.LIFT]; self.i_torso = names.index("torso")
        self.i_neck = [names.index(n) for n in R.NECK]; self.i_arm = [names.index(n) for n in R.ARM]
        self.i_pos = [self.i_torso] + self.i_neck + self.i_arm
        lim = tt([fk.J[fk.joint_child[n]]["lim"] for n in names])
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.q_ready = tt([R.READY.get(n, 0.0) for n in names]); self.q_ready[self.i_lift] = R.LIFT_STROKE / 2
        # ---- batched FK over the links anything here measures (and their ancestors)
        T = R.tracked_offsets(fk)
        parts, sph, pairs = R.collision_model()
        need = {T[k][0] for k in T} | {"l_elbow", "r_elbow", "l_shoulder_3", "r_shoulder_3", "torso", "head"}
        need |= {str(x[0]) for x in sph} | {p["link"] for p in parts}
        keep = set()
        for l in need:
            while l in fk.J: keep.add(l); l = fk.J[l]["parent"]
        order, root = [], "mabel_chassis"
        def visit(l):
            order.append(l)
            for c, d in fk.J.items():
                if d["parent"] == l and c in keep: visit(c)
        visit(root)
        self.links = order; self.L = {l: i for i, l in enumerate(order)}
        self.fk_par = [self.L[fk.J[l]["parent"]] if l in fk.J else -1 for l in order]
        self.fk_xyz = tt([fk.J[l]["xyz"] if l in fk.J else np.zeros(3) for l in order])
        self.fk_R = tt([fk.J[l]["R"] if l in fk.J else np.eye(3) for l in order])
        self.fk_j = []                                   # (type, q column, K, K^2, axis)
        for l in order:
            d = fk.J.get(l)
            if d is None or d["type"] == "fixed" or d["name"] not in self.jn: self.fk_j.append(None); continue
            k = d["axis"] / np.linalg.norm(d["axis"]); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            self.fk_j.append((d["type"], self.jn.index(d["name"]), tt(K), tt(K @ K), tt(d["axis"])))
        self.trk_body = [self.L[T[k][0]] for k in ("head", "left_wrist", "right_wrist")]
        self.trk_p = tt(np.stack([T[k][1] for k in ("head", "left_wrist", "right_wrist")]))
        self.trk_R = tt(np.stack([T[k][2] for k in ("head", "left_wrist", "right_wrist")]))
        self.elbow_body = [self.L["l_elbow"], self.L["r_elbow"]]; self.shoulder_body = [self.L["l_shoulder_3"], self.L["r_shoulder_3"]]
        self.torso_body = self.L["torso"]
        self.torso_R = tt([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        # ---- collision model (env.py's)
        self.sph_body = torch.tensor([self.L[str(x[0])] for x in sph], device=dev)
        self.sph_c, self.sph_r = tt([x[1] for x in sph]), tt([x[2] for x in sph])
        self.sph_sh = torch.tensor([str(x[4]) in ("shoulder_1", "shoulder_2") for x in sph], device=dev)
        self.sph_hand = torch.tensor(np.array([[str(x[4]) in R.HAND and str(x[3]) == s for s in "lr"] for x in sph]), device=dev)
        self.sdf = []
        for j, p in enumerate(parts):
            ids = torch.tensor(np.nonzero(pairs[:, j])[0], device=dev)
            if len(ids) == 0: continue
            g = tt(p["sdf"]).permute(2, 1, 0)[None, None].contiguous()
            fh = "lr".index(p["side"]) if p["side"] in ("l", "r") and p["name"] in R.HAND else -1
            self.sdf.append((self.L[p["link"]], tt(p["origin"]), tt((np.array(p["sdf"].shape) - 1) * p["res"]), g, ids, fh))
        # ---- limits, filters, wrist solver
        names_pos = [names[i] for i in self.i_pos]
        grp = lambda n: R.gain_group(R.joint_group(n))
        sp = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)
        self.span = sp
        lo = (self.q_lo[self.i_pos] + sp - self.q_ready[self.i_pos]) / self.cfg.action_scale
        hi = (self.q_hi[self.i_pos] - sp - self.q_ready[self.i_pos]) / self.cfg.action_scale
        self.a_lo = torch.cat([torch.full((4,), -2.5, device=dev), lo.clamp(min=-5.0)])
        self.a_hi = torch.cat([torch.full((4,), 2.5, device=dev), hi.clamp(max=5.0)])
        self.slew = tt([min(R.HW_GAINS[grp(n)][2], R.HEADROOM * R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0]) for n in names_pos])
        self.v_lim = tt([R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0] for n in names_pos])
        self.wc = [R.wrist_chain(fk, s) for s in ("left", "right")]
        self.i_wr = [[self.i_pos.index(names.index(f"{s}_arm_{j}")) for j in (5, 6, 7)] for s in ("left", "right")]
        self.i_wr_all = self.i_wr[0] + self.i_wr[1]
        lh = tt([fk.J[fk.joint_child[n]]["lim"] for n in R.HAND_WRIST])
        self.hw_lo, self.hw_hi = lh[:, 0] + 0.08, lh[:, 1] - 0.08
        self.hw_vlim = R.SAFETY * R.JOINT_SPEC["hand_wrist"][0]
        hz = torch.full((len(self.i_pos),), self.cfg.action_lpf_hz, device=dev); hz[self.i_wr_all] = self.cfg.wrist_lpf_hz
        self.beta = 1.0 - torch.exp(-2 * math.pi * hz * DT)
        self.load_motions()
        self.alloc()
        self.extras = {"log": {}}
        self.reset_idx(torch.arange(N, device=dev))
        self.obs = self.observations()

    # ------------------------------------------------------------ buffers
    def alloc(self):
        N, dev = self.num_envs, self.device
        z = lambda *s: torch.zeros(N, *s, device=dev)
        eye = lambda *s: torch.eye(3, device=dev).repeat(N, *s, 1, 1)
        self.phase, self.rate, self.c_id = z(), torch.ones(N, device=dev), torch.zeros(N, dtype=torch.long, device=dev)
        self.anc_yaw, self.anc_h, self.anc_r = z(), z(3), z(3)
        self.aug_on, self.aug_yaw, self.aug_xy, self.aug_w, self.aug_v = z(), z(), z(2), z(), z(2)
        self.neck_on, self.neck_off, self.neck_tgt = z(), z(2), z(2)
        self.tgt = None; self.hist = z(10, 27)
        self.rx_p, self.rx_R, self.rx_v = z(3, 3), eye(3), z(2, 3)
        self.buf_p, self.buf_R = z(8, 3, 3), eye(8, 3)
        self.age = z(3)
        self.lat = torch.zeros(N, dtype=torch.long, device=dev); self.loss, self.npos, self.nrot = z(), z(), z()
        self.trem_w, self.trem_a = z(3), z(3, 2); self.trem_z = torch.zeros(N, 3, 6, dtype=torch.complex64, device=dev)
        self.slow_k = torch.ones(N, dtype=torch.long, device=dev); self.slow_t, self.out_t = z(), z()
        self.occ_t, self.occ_pred = z(2), z(2)
        self.spk_n = torch.zeros(N, 2, dtype=torch.long, device=dev); self.spk_p, self.spk_R = z(2, 3), eye(2)
        self.bias_p, self.bias_R = z(3, 3), eye(3)
        self.hold_p, self.hold_R = z(3, 3), eye(3)
        self.lost_prev = torch.zeros(N, 3, dtype=torch.bool, device=dev)
        self.odo_xy, self.odo_yaw = z(2), z()
        self.rec_t = torch.full((N,), 10.0, device=dev); self.n_step = 0
        self.actions, self.prev_actions, self.prev2_actions, self.raw_act = z(22), z(22), z(22), z(22)
        P = len(self.i_pos)
        self.q_f, self.q_f_prev, self.q_f1, self.qf_acc = z(P), z(P), z(P), z(P)
        self.q, self.q_prev, self.qd, self.qd_prev = z(22), z(22), z(22), z(22)        # the reference: ACTUATED + hinges
        self.pos, self.yaw, self.v_b, self.v_b_prev = z(3), z(), z(3), z(3)            # base pose (REP), body twist
        self.lift_tgt = z()
        self.upper, self.sm0, self.sm_lp = z(), z(2), z(2)
        self.hand_v_prev, self.hand_acc, self.tgt_v, self.palm_w_prev, self.palm_acc = z(2, 3), z(2, 3), z(2, 3), z(2, 3), z(2, 3)
        self.hand_p_prev, self.palm_R_prev = z(2, 3), eye(2)
        self.clear, self.clear_mv, self.clear_sh = z(), z(), z()
        self.clear_hand = torch.ones(N, 2, device=dev); self.w_safe = z(2, 4)
        self.wik, self.wres = z(2, 4), z(2, 3)
        self.hw_f1, self.hw_f = z(2), z(2)
        self.oe_R, self.oe_s = eye(2), z(2)
        self.blend_on, self.blend_t, self.blend_T = z(), z(), torch.ones(N, device=dev)
        self.blend_p, self.blend_R = z(3, 3), eye(3)
        self.need_glide = torch.zeros(N, dtype=torch.bool, device=dev)
        self.reset_terminated = torch.zeros(N, dtype=torch.bool, device=dev)
        self.last_term = torch.zeros(N, 2, dtype=torch.bool, device=dev)          # collide, drift
        self.v_world = z(3); self.w_world = z(3)
        self.robot = type("Rb", (), {})(); self.robot.data = type("D", (), {})()      # stream() reads the base velocity
        self.robot.data.root_link_lin_vel_w = _V(self.v_world); self.robot.data.root_link_ang_vel_w = _V(self.w_world)

    # ------------------------------------------------------------ kinematics
    def fk_update(self):
        """Link poses in the world (N, B, 3), (N, B, 3, 3): base at (x, y, CHASSIS_Z), URDF base facing -x."""
        N = self.num_envs
        Rr = rot_z(self.yaw + math.pi); pr = self.pos
        Ps, Rs = [pr], [Rr]
        I = torch.eye(3, device=self.device)
        for i in range(1, len(self.links)):
            Rp, pp = Rs[self.fk_par[i]], Ps[self.fk_par[i]]
            p = pp + (Rp @ self.fk_xyz[i][:, None])[..., 0]
            Rm = Rp @ self.fk_R[i]
            j = self.fk_j[i]
            if j is not None:
                a = self.q[:, j[1]]
                if j[0] == "prismatic":
                    p = p + (Rm @ j[4][:, None])[..., 0] * a[:, None]
                else:
                    Rm = Rm @ (I + torch.sin(a)[:, None, None] * j[2] + (1 - torch.cos(a))[:, None, None] * j[3])
            Ps.append(p); Rs.append(Rm)
        self.Lp = torch.stack(Ps, 1); self.LR = torch.stack(Rs, 1)

    def base_pose(self):
        return self.pos, self.yaw, rot_z(self.yaw)

    def tracked_now(self):
        bp, bR = self.Lp[:, self.trk_body], self.LR[:, self.trk_body]
        return bp + (bR @ self.trk_p[None, :, :, None])[..., 0], bR @ self.trk_R[None]

    def clearance(self):
        """env.clearance on the FK poses."""
        bp, bq = self.Lp, self.LR
        P = bp[:, self.sph_body] + (bq[:, self.sph_body] @ self.sph_c[None, :, :, None])[..., 0]
        d = torch.full((self.num_envs,), 1.0, device=self.device); d_sh = d.clone(); ch = torch.ones(self.num_envs, 2, device=self.device)
        for body, org, ext, g, ids, fh in self.sdf:
            loc = (P[:, ids] - bp[:, body, None]) @ bq[:, body]
            u = (2 * (loc - org) / ext - 1).reshape(1, 1, 1, -1, 3)
            f = torch.nn.functional.grid_sample(g, u, mode="bilinear", padding_mode="border", align_corners=True)
            dd = f.view(self.num_envs, -1) - self.sph_r[ids]; sh = self.sph_sh[ids]
            d = torch.minimum(d, dd.masked_fill(sh, 1.0).amin(-1)); d_sh = torch.minimum(d_sh, dd.masked_fill(~sh, 1.0).amin(-1))
            ch = torch.minimum(ch, dd[:, :, None].masked_fill(~self.sph_hand[ids][None], 1.0).amin(1))
            if fh >= 0: ch[:, fh] = torch.minimum(ch[:, fh], dd.amin(-1))
        self.clear_sh = d_sh; self.clear_hand = ch
        return torch.minimum(d, d_sh), d

    def solve_wrist(self):
        """robot.wrist_ik for the reference forearm (env.solve_wrist with the reference as the 'measurement')."""
        qd = {n: self.q[:, i] for i, n in enumerate(R.ACTUATED)}
        Rb = rot_z(self.yaw) @ rot_z(torch.full_like(self.yaw, math.pi))
        lo, hi = self.q_lo[self.i_pos] + self.span, self.q_hi[self.i_pos] - self.span
        out = []
        for j, wc in enumerate(self.wc):
            A = Rb @ R.chain_rot(wc["chain"], qd, torch)
            ix = self.i_wr[j]
            L = torch.cat([lo[ix], self.hw_lo[j:j + 1]]); H = torch.cat([hi[ix], self.hw_hi[j:j + 1]])
            self.wik[:, j] = R.wrist_ik(A, self.oe_R[:, j], wc, self.wik[:, j], L, H, torch, iters=8)
            self.wres[:, j] = R.wrist_err(A, self.oe_R[:, j], wc, self.wik[:, j], torch)
            out.append(self.wik[:, j])
        return torch.cat(out, -1)

    def wrist_obs(self, yaw, cR, noisy):
        Ryt = rot_z(yaw).transpose(-1, -2)[:, None]
        Rt = self.rx_R[:, 1:] if noisy else torch.stack([self.tgt["lw_R"], self.tgt["rw_R"]], 1)
        pe = rotlog(Rt @ cR[:, 1:].transpose(-1, -2))
        res = (Ryt @ self.wres[..., None])[..., 0]; pe = (Ryt @ pe[..., None])[..., 0]
        return torch.cat([self.q[:, 20:22], res.reshape(-1, 6), pe.reshape(-1, 6)], -1)

    # ------------------------------------------------------------ step
    def apply(self, actions):
        """Policy output -> the next reference (env._pre_physics_step without the plant)."""
        self.raw_act = actions
        a = actions.clamp(-5.0, 5.0)
        self.prev2_actions = self.prev_actions; self.prev_actions = self.actions; self.actions = a
        dt = DT; dev = self.device
        # base: twist command -> nova_hw accel ramp -> integrate
        cmd = torch.tanh(a[:, 0:3]) * torch.tensor([R.BASE_VMAX, R.BASE_VMAX, R.BASE_WMAX], device=dev) * (1 - self.upper[:, None])
        acc = torch.full((self.num_envs, 1), R.BASE_ACCEL, device=dev)
        vxy = R.base_ramp(self.v_b[:, :2], cmd[:, :2], acc, acc * R.BASE_DECEL, dt, torch)
        dw = (cmd[:, 2] - self.v_b[:, 2]).clamp(-R.BASE_ALPHA * dt, R.BASE_ALPHA * dt)
        self.v_b_prev = self.v_b; self.v_b = torch.cat([vxy, (self.v_b[:, 2] + dw)[:, None]], -1)
        ym = self.yaw + 0.5 * self.v_b[:, 2] * dt; c, s = torch.cos(ym), torch.sin(ym)
        vw = torch.stack([c * self.v_b[:, 0] - s * self.v_b[:, 1], s * self.v_b[:, 0] + c * self.v_b[:, 1]], -1)
        self.pos = self.pos + torch.cat([vw, torch.zeros_like(vw[:, :1])], -1) * dt
        self.yaw = wrap(self.yaw + self.v_b[:, 2] * dt)
        self.v_world = torch.cat([vw, torch.zeros_like(vw[:, :1])], -1); self.w_world = torch.zeros_like(self.v_world); self.w_world[:, 2] = self.v_b[:, 2]
        self.robot.data.root_link_lin_vel_w.torch = self.v_world; self.robot.data.root_link_ang_vel_w.torch = self.w_world
        # lift
        rate = torch.tanh(a[:, 3]) * R.HEADROOM * R.LIFT_VLIM * (1 - self.upper)
        self.lift_tgt = (self.lift_tgt + 0.5 * rate * dt).clamp(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)
        # position joints: command (+ the wrist solver) -> window clamp -> 2nd-order filter -> slew
        q = self.q_ready[self.i_pos][None] + self.cfg.action_scale * a[:, 4:]
        ik = self.solve_wrist()
        ik = R.hand_guard(ik.view(-1, 2, 4), self.w_safe, self.clear_hand, self.cfg.hand_guard, torch).reshape(-1, 8)
        q[:, self.i_wr_all] = ik[:, [0, 1, 2, 4, 5, 6]] + self.cfg.wrist_res * a[:, 4:][:, self.i_wr_all]
        g = torch.max(torch.min(ik[:, [3, 7]], self.hw_hi), self.hw_lo)
        b = self.beta[self.i_wr_all[0]]
        self.hw_f1 = self.hw_f1 + b * (g - self.hw_f1)
        self.hw_f = self.hw_f + (b * (self.hw_f1 - self.hw_f)).clamp(-R.HEADROOM * self.hw_vlim * dt, R.HEADROOM * self.hw_vlim * dt)
        q = torch.max(torch.min(q, self.q_hi[self.i_pos] - self.span), self.q_lo[self.i_pos] + self.span)
        qf_pp = self.q_f_prev; self.q_f_prev = self.q_f
        self.q_f1 = self.q_f1 + self.beta * (q - self.q_f1)
        self.q_f = self.q_f + torch.max(torch.min(self.beta * (self.q_f1 - self.q_f), self.slew * dt), -self.slew * dt)
        self.qf_acc = (self.q_f - 2 * self.q_f_prev + qf_pp) / (dt * dt)
        self.q_prev = self.q; self.qd_prev = self.qd
        qn = self.q.clone(); qn[:, self.i_pos] = self.q_f; qn[:, self.i_lift[0]] = self.lift_tgt; qn[:, self.i_lift[1]] = self.lift_tgt
        qn[:, 20:22] = self.hw_f
        self.q = qn; self.qd = (self.q - self.q_prev) / dt

    def update_state(self):
        dt = DT
        g = self.need_glide
        if g.any():
            cp, cR = self.tracked_now()
            self.blend_p[g], self.blend_R[g] = cp[g], cR[g]
            self.blend_on[g], self.blend_t[g], self.blend_T[g] = 1.0, 0.0, 1.0
            self.need_glide[:] = False
        self.blend_t += dt
        self.phase += self.rate
        done = self.phase >= self.c_len[self.c_id].float() - 1.001
        if done.any(): self.chain(done.nonzero()[:, 0])
        old = torch.stack([self.tgt["lw_p"], self.tgt["rw_p"]], 1)
        self.tgt = self.compute_targets(self.phase)
        self.stream()
        v = (torch.stack([self.tgt["lw_p"], self.tgt["rw_p"]], 1) - old) / dt
        self.tgt_v = v * (3.0 / v.norm(dim=-1, keepdim=True).clamp_min(3.0))
        self.clear, self.clear_mv = self.clearance()

    def pose_guard(self, q_old, c_old):
        """Collision guard on the whole pose: if the new reference would bring any part closer than pose_guard (and
        closer than it already is), take the largest of 1/2, 1/4, 0 of the step that does not; 0 = hold the last
        pose, which never adds a collision (self-clearance does not depend on the base pose)."""
        need = lambda c: torch.minimum(torch.full_like(c_old, self.cfg.pose_guard), c_old)
        c, _ = self.clearance(); bad = c < need(c)
        self.guard_s = torch.ones(self.num_envs, device=self.device)
        if not bad.any(): return
        q_new = self.q.clone()
        for f in (0.5, 0.25, 0.0):
            fix = bad.clone()
            self.q = torch.where(fix[:, None], q_old + f * (q_new - q_old), self.q); self.guard_s[fix] = f
            if f == 0.0: break
            self.fk_update(); c, _ = self.clearance(); bad = fix & (c < need(c))
            if not bad.any(): break
        held = self.guard_s < 1
        self.qd = torch.where(held[:, None], (self.q - self.q_prev) / DT, self.qd)
        # the filters continue from the pose actually sent
        self.q_f = torch.where(held[:, None], self.q[:, self.i_pos], self.q_f)
        self.hw_f = torch.where(held[:, None], self.q[:, 20:22], self.hw_f)
        self.lift_tgt = torch.where(held, self.q[:, self.i_lift[0]], self.lift_tgt)
        self.fk_update()

    def step(self, actions):
        q_old, c_old = self.q.clone(), self.clear.clone()
        self.apply(actions)
        self.episode_length_buf += 1
        self.fk_update()
        if self.cfg.pose_guard > 0: self.pose_guard(q_old, c_old)
        else: self.guard_s = torch.ones(self.num_envs, device=self.device)
        self.extras = {"log": {}}
        term, tout = self.dones()
        self.reset_terminated = term
        rew = self.rewards()
        ids = (term | tout).nonzero()[:, 0]
        if len(ids):
            self.reset_idx(ids); self.fk_update()
            self.clear, self.clear_mv = self.clearance()
        self.obs = self.observations()
        self.extras["time_outs"] = tout
        return self.obs, rew, (term | tout).long(), self.extras

    def get_observations(self):
        return self.obs

    # ------------------------------------------------------------ observations (env.py layout)
    def observations(self):
        pos, yaw = self.pos, self.yaw
        tob = self.target_obs(pos, yaw, noisy=True)
        self.hist = torch.cat([tob[:, None], self.hist[:, :-1]], 1)
        cp, cR = self.tracked_now()
        Ryt = rot_z(yaw).transpose(-1, -2)[:, None]
        cur = torch.cat([((Ryt @ (cp - pos[:, None])[..., None])[..., 0]).reshape(self.num_envs, 9),
                         six_d(Ryt @ cR).reshape(self.num_envs, 18)], -1)
        q, qd = self.q[:, :20], self.qd[:, :20]
        actor = torch.cat([self.hist[:, 0], self.hist[:, 2], self.hist[:, 5], cur, q - self.q_ready, 0.1 * qd,
                           self.v_b, self.act_obs(), self.upper[:, None], self.age.clamp(max=1.0), self.wrist_obs(yaw, cR, True)], -1)
        t = self.tgt; N = self.num_envs
        clean = torch.cat([self.target_obs(pos, yaw, noisy=False), cur, q - self.q_ready, 0.1 * qd, self.v_b, self.act_obs()], -1)
        fut = self.compute_targets(self.phase + 25 * self.rate, step_aug=False)
        Rb = rot_z(yaw).transpose(-1, -2)
        loc = lambda p: (Rb @ (p - pos)[..., None])[..., 0]
        by = wrap(t["body_yaw"] - yaw)
        el = self.Lp[:, self.elbow_body]
        zer = torch.zeros(N, 1, device=self.device)
        grav = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(N, 1)
        priv = torch.cat([grav, torch.zeros(N, 3, device=self.device), zer, zer + 1.0, self.clear[:, None] * 10,
                          loc(fut["head_p"]), loc(fut["lw_p"]), loc(fut["rw_p"]),
                          torch.stack([torch.cos(by), torch.sin(by)], -1), loc(t["sm"])[:, :2], t["lean"][:, None], t["neck"],
                          loc(t["le"]), loc(t["re"]), loc(el[:, 0]), loc(el[:, 1]),
                          self.lift_tgt[:, None] * 5, zer + 1.0, zer, self.upper[:, None],
                          self.age.clamp(max=1.0), self.wrist_obs(yaw, cR, False)], -1)
        return TensorDict({"policy": actor, "critic": torch.cat([clean, priv], -1)}, batch_size=[N], device=self.device)

    # ------------------------------------------------------------ rewards (env.py's, the dynamics terms dropped)
    def rewards(self):
        w, t = self.cfg.w, self.tgt
        pos, yaw = self.pos, self.yaw
        cp, cR = self.tracked_now()
        e_h = (cp[:, 0] - t["head_p"]).norm(dim=-1)
        e_l = (cp[:, 1] - t["lw_p"]).norm(dim=-1); e_r = (cp[:, 2] - t["rw_p"]).norm(dim=-1)
        a_h = rot_angle(cR[:, 0], t["head_R"]); a_l = rot_angle(cR[:, 1], t["lw_R"]); a_r = rot_angle(cR[:, 2], t["rw_R"])
        K = lambda e, sg: torch.exp(-(e / sg) ** 2)
        r = {}
        r["head_p"] = K(e_h, 0.12); r["head_r"] = K(a_h, 0.35)
        r["wrist_p"] = 0.5 * (K(e_l, 0.10) + K(e_r, 0.10)); r["wrist_p_mid"] = 0.5 * (K(e_l, 0.05) + K(e_r, 0.05))
        r["wrist_p_fine"] = 0.5 * (K(e_l, 0.02) + K(e_r, 0.02))
        rot_ok = self.c_ego[self.c_id].float()
        r["wrist_r"] = (K(a_l, 0.087) + K(a_r, 0.087) + K(a_l, 0.2) + K(a_r, 0.2) + K(a_l, 0.5) + K(a_r, 0.5)
                        + K(a_l, 1.0) + K(a_r, 1.0)) / 8 * rot_ok
        hp = cp[:, 1:]
        hv = (hp - self.hand_p_prev) / DT
        fresh = (self.episode_length_buf <= 1)[:, None, None]
        hv = torch.where(fresh, torch.zeros_like(hv), hv)
        r["hand_vel"] = K((hv - self.tgt_v).norm(dim=-1), 0.3).mean(-1)
        r["track_lin"] = e_h + 2.0 * (e_l + e_r) + 0.1 * a_h + 0.6 * (a_l + a_r) * rot_ok
        r["wrist_unreach"] = self.wres.norm(dim=-1).sum(-1) * rot_ok
        r["body_yaw"] = K(wrap(t["body_yaw"] - yaw), 0.35)
        base_tgt = t["sm"][:, :2] - torch.stack([torch.cos(t["body_yaw"]), torch.sin(t["body_yaw"])], -1) * 0.096
        r["body_xy"] = K((pos[:, :2] - base_tgt).norm(dim=-1), 0.2)
        up = self.upper > 0
        r["head_far"] = K(e_h, 0.5); r["body_far"] = K((pos[:, :2] - base_tgt).norm(dim=-1), 0.6)
        r["body_yaw"] = torch.where(up, torch.ones_like(e_h), r["body_yaw"]); r["body_xy"] = torch.where(up, torch.ones_like(e_h), r["body_xy"])
        r["body_far"] = torch.where(up, torch.ones_like(e_h), r["body_far"])
        q = self.q[:, :20]
        lean_t = (-t["lean"]).clamp(-1.0, 0.15)
        r["lean"] = K(q[:, self.i_torso] - lean_t, 0.15)
        r["lean_excess"] = (lean_t - q[:, self.i_torso] - 0.2).clamp_min(0).square()
        tR = self.LR[:, self.torso_body] @ self.torso_R
        neck_lbl = rot_z(t["neck"][:, 0]) @ rot_y(t["neck"][:, 1]) @ rot_x(t["neck"][:, 2])
        r["neck"] = K(rot_angle(tR.transpose(-1, -2) @ cR[:, 0], neck_lbl), 0.35)
        bp = self.Lp
        nrm = torch.nn.functional.normalize
        ua_l = nrm(bp[:, self.elbow_body[0]] - bp[:, self.shoulder_body[0]], dim=-1)
        ua_r = nrm(bp[:, self.elbow_body[1]] - bp[:, self.shoulder_body[1]], dim=-1)
        ang = lambda u, v: torch.acos((u * v).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        r["elbow"] = 0.5 * (K(ang(ua_l, t["l_arm"]), 0.35) + K(ang(ua_r, t["r_arm"]), 0.35))
        r["base_use"] = self.v_b[:, :2].square().sum(-1) + 0.3 * self.v_b[:, 2].square()
        r["base_acc"] = ((self.v_b - self.v_b_prev) / DT).square().sum(-1)
        lr = torch.tanh(self.actions[:, 3]).abs()
        r["lift_use"] = lr + lr.square()
        r["act_sat"] = ((self.raw_act - self.a_hi).clamp_min(0) + (self.a_lo - self.raw_act).clamp_min(0)).sum(-1)
        r["action_rate"] = (self.actions - self.prev_actions).square().sum(-1)
        r["action_acc"] = (self.actions - 2 * self.prev_actions + self.prev2_actions).square().sum(-1)
        r["qf_acc"] = self.qf_acc.square().sum(-1).clamp(max=2e5)
        self.hand_acc = torch.where(fresh, torch.zeros_like(hv), (hv - self.hand_v_prev) / DT)
        self.hand_v_prev = hv; self.hand_p_prev = hp
        r["hand_acc"] = self.hand_acc.square().sum((-1, -2)).clamp(max=5000.0)
        pR = cR[:, 1:]
        pw = rotlog(pR @ self.palm_R_prev.transpose(-1, -2)) / DT
        pw = torch.where(fresh, torch.zeros_like(pw), pw)
        self.palm_acc = torch.where(fresh, torch.zeros_like(pw), (pw - self.palm_w_prev) / DT)
        self.palm_w_prev = pw; self.palm_R_prev = pR
        r["palm_acc"] = self.palm_acc.square().sum((-1, -2)).clamp(max=2e4)
        qd = self.qd[:, :20]
        r["joint_vel"] = qd.square().sum(-1)
        qdd = (qd - self.qd_prev[:, :20]) / DT
        qdd = torch.where(fresh[:, :, 0], torch.zeros_like(qdd), qdd)
        r["joint_acc"] = qdd.square().sum(-1)
        r["arm_acc"] = qdd[:, self.i_arm].square().sum(-1)
        span = self.q_hi - self.q_lo
        r["joint_limit"] = ((self.q_lo + 0.08 * span - q).clamp_min(0) + (q - self.q_hi + 0.08 * span).clamp_min(0)).sum(-1)
        bar = lambda x: ((x - 0.85) / 0.15).clamp(0, 1.5).square().sum(-1)
        r["limits"] = bar(qd[:, self.i_pos].abs() / self.v_lim) + bar(self.qd[:, 20:22].abs() / self.hw_vlim)
        twist_j = [self.i_arm[k] for k in (2, 4, 6, 9, 11, 13)]
        r["twist"] = (q[:, twist_j] - self.q_ready[twist_j]).square().sum(-1)
        band = lambda d, b: ((b - d) / (b - self.cfg.collide_margin)).clamp(0, 1.5)
        r["clearance"] = torch.maximum(band(self.clear_mv, 0.04), band(self.clear_sh, 0.02))
        r["terminated"] = self.reset_terminated.float()
        r["collide"] = self.last_term[:, 0].float()
        r["alive"] = torch.ones_like(e_h)
        r["guard"] = 1 - self.guard_s
        total = sum(w[k] * r[k] for k in w) * DT
        log = self.extras["log"]
        for k in w: log[f"RewTerm/{k}"] = (w[k] * r[k]).mean()
        log["Err/head_cm"] = e_h.mean() * 100; log["Err/wrist_cm"] = 0.5 * (e_l + e_r).mean() * 100
        log["Err/head_deg"] = torch.rad2deg(a_h).mean(); log["Err/wrist_deg"] = torch.rad2deg(0.5 * (a_l + a_r)).mean()
        log["Err/wrist_deg_ego"] = (torch.rad2deg(0.5 * (a_l + a_r)) * rot_ok).sum() / rot_ok.sum().clamp_min(1)
        log["Err/wrist_unreach_deg"] = torch.rad2deg(self.wres.norm(dim=-1)).mean()
        log["Safety/clear_cm"] = self.clear.mean() * 100; log["Safety/guard"] = (self.guard_s < 1).float().mean()
        log["Base/speed"] = self.v_b[:, :2].norm(dim=-1).mean(); log["Base/lift_mm_s"] = (self.qd[:, self.i_lift[0]].abs() * 2000).mean()
        log["Smooth/hand_acc"] = self.hand_acc.norm(dim=-1).mean(); log["Smooth/palm_acc"] = self.palm_acc.norm(dim=-1).mean()
        log["Smooth/action_rate"] = r["action_rate"].mean(); log["Smooth/qf_acc"] = r["qf_acc"].mean()
        log["Limits/speed_ratio_max"] = (qd[:, self.i_pos].abs() / self.v_lim).max(-1).values.mean()
        self.r_last = dict(e_h=e_h, e_l=e_l, e_r=e_r, a_h=a_h, a_l=a_l, a_r=a_r, rot_ok=rot_ok)
        return total

    def dones(self):
        self.update_state()
        collide = self.clear < self.cfg.collide_margin
        cp, cR = self.tracked_now(); t = self.tgt
        grace = (self.episode_length_buf * DT > self.cfg.grace_s) & (self.rec_t > 1.0)
        self.drift_kind = grace[:, None] & torch.stack([
            (cp[:, 0] - t["head_p"]).norm(dim=-1) > self.cfg.drift_head,
            torch.maximum((cp[:, 1] - t["lw_p"]).norm(dim=-1), (cp[:, 2] - t["rw_p"]).norm(dim=-1)) > self.cfg.drift_wrist,
            rot_angle(cR[:, 0], t["head_R"]) > self.cfg.drift_head_rot,
            (wrap(t["body_yaw"] - self.yaw).abs() > self.cfg.drift_body_yaw) & (self.upper == 0)], -1)
        drift = self.drift_kind.any(-1)
        tout = self.episode_length_buf >= self.max_episode_length - 1
        log = self.extras["log"]
        log["Term/collide"] = collide.float().mean(); log["Term/drift"] = drift.float().mean()
        self.last_term = torch.stack([collide, drift], -1)
        stop = collide | (drift & self.cfg.drift_stop)           # k4: falling behind is not a stop (env.py v40)
        return stop, tout & ~stop

    # ------------------------------------------------------------ reset (env._reset_idx, the robot part kinematic)
    def reset_idx(self, env_ids):
        n, dev = len(env_ids), self.device
        self.episode_length_buf[env_ids] = 0
        self.upper[env_ids] = (torch.rand(n, device=dev) < self.cfg.upper_share).float()
        up = self.upper[env_ids] > 0
        c = torch.multinomial(self.c_p, n, replacement=True)
        if up.any(): c[up] = torch.multinomial(self.c_p_up, int(up.sum()), replacement=True)
        L = self.c_len[c].float()
        start = torch.rand(n, device=dev) * (L - 2.0 * 50).clamp_min(0) * 0.8
        self.c_id[env_ids] = c; self.rate[env_ids] = torch.empty(n, device=dev).uniform_(*self.cfg.speed_range)
        self.phase[env_ids] = start
        self.aug_on[env_ids] = ((torch.rand(n, device=dev) < self.cfg.twist_aug) & ~up).float()
        self.neck_on[env_ids] = (torch.rand(n, device=dev) < self.cfg.neck_aug).float()
        for b in (self.aug_yaw, self.aug_w, self.aug_xy, self.aug_v, self.neck_off, self.neck_tgt): b[env_ids] = 0
        lift = torch.empty(n, device=dev).uniform_(0.25, 0.75) * R.LIFT_STROKE
        lift[up] = torch.empty(int(up.sum()), device=dev).uniform_(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)
        q = self.q_ready[None].repeat(n, 1)
        q[:, self.i_lift[0]] = lift; q[:, self.i_lift[1]] = lift
        q[:, self.i_arm] += torch.randn(n, len(self.i_arm), device=dev) * 0.05
        lo, hi = self.q_lo[self.i_pos] + self.span, self.q_hi[self.i_pos] - self.span
        q[:, self.i_pos] = torch.max(torch.min(q[:, self.i_pos], hi), lo)
        yaw0 = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        self.q[env_ids] = torch.cat([q, torch.zeros(n, 2, device=dev)], -1); self.q_prev[env_ids] = self.q[env_ids]
        self.qd[env_ids] = 0; self.qd_prev[env_ids] = 0
        self.pos[env_ids] = torch.tensor([0.0, 0.0, R.CHASSIS_Z], device=dev); self.yaw[env_ids] = yaw0
        self.v_b[env_ids] = 0; self.v_b_prev[env_ids] = 0
        i0 = (self.c_start[c].float() + start).long()
        hR = self.m["head_R"][i0]; h_yaw = torch.atan2(hR[:, 1, 0], hR[:, 0, 0])
        self.anc_yaw[env_ids] = yaw0 - h_yaw
        self.anc_h[env_ids] = self.m["head_p"][i0]
        sl, sr = E.ML.shoulders_from_head(self.m["head_p"][i0], h_yaw, torch); sm = 0.5 * (sl + sr)
        self.sm0[env_ids] = sm[:, :2]; self.sm_lp[env_ids] = sm[:, :2]
        off = (rot_z(yaw0 - h_yaw) @ (self.m["head_p"][i0] - sm)[..., None])[..., 0]
        fwd = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], -1)
        anc_r = torch.zeros(n, 3, device=dev)
        anc_r[:, :2] = fwd * 0.096 + off[:, :2]
        anc_r[:, 2] = 1.374 + 2 * lift + torch.randn(n, device=dev) * 0.03
        self.anc_r[env_ids] = anc_r
        self.lift_tgt[env_ids] = lift
        for b in (self.actions, self.prev_actions, self.prev2_actions, self.raw_act): b[env_ids] = 0
        qp = q[:, self.i_pos]
        self.q_f[env_ids] = qp; self.q_f_prev[env_ids] = qp; self.q_f1[env_ids] = qp; self.qf_acc[env_ids] = 0
        qw = torch.cat([qp[:, self.i_wr_all].reshape(-1, 2, 3), torch.zeros(n, 2, 1, device=dev)], -1)
        self.wik[env_ids] = qw; self.w_safe[env_ids] = qw; self.clear_hand[env_ids] = 1.0; self.wres[env_ids] = 0
        self.hw_f1[env_ids] = 0; self.hw_f[env_ids] = 0
        for b in (self.hand_v_prev, self.hand_acc, self.tgt_v, self.palm_w_prev, self.palm_acc, self.hand_p_prev): b[env_ids] = 0
        self.need_glide[env_ids] = True; self.blend_on[env_ids] = 0.0
        self.tgt = self.compute_targets(self.phase, step_aug=False)
        self.stream_reset(env_ids)
        tob = self.target_obs(self.pos, self.yaw, noisy=False)
        self.hist[env_ids] = tob[env_ids, None].repeat(1, self.hist.shape[1], 1)
        self.fk_update()
        cp, cR = self.tracked_now()
        self.hand_p_prev[env_ids] = cp[env_ids, 1:]; self.palm_R_prev[env_ids] = cR[env_ids, 1:]


# ---------------------------------------------------------------- train / eval
def agent_cfg(iters, device):
    import importlib.metadata
    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    import tasks
    a = tasks.NovaWbcPPOCfg(); a.experiment_name = "nova_kin"; a.max_iterations = iters; a.device = device
    return handle_deprecated_rsl_rl_cfg(a, importlib.metadata.version("rsl-rl-lib"))


def train(args):
    from isaaclab_rl.rsl_rl.utils import create_rsl_rl_runner
    env = KinEnv(num_envs=args.envs, device=args.device, seed=args.seed)
    a = agent_cfg(args.iters, args.device)
    log_dir = os.path.join(HERE, "logs", "rsl_rl", "nova_kin", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run}")
    runner = create_rsl_rl_runner(env, a, log_dir=log_dir)
    if args.checkpoint:
        print("[kin] warm start from", args.checkpoint)
        runner.load(args.checkpoint, map_location=args.device)
        if args.reset_iter: runner.current_learning_iteration = 0
    if args.std:                                       # warm start from a dynamics run: no motors filter the exploration
        for nm, p in runner.alg.actor.named_parameters():
            if nm.endswith("log_std_param"): p.data.clamp_(max=math.log(args.std))
            elif nm.endswith("std_param"): p.data.clamp_(max=args.std)
    runner.learn(num_learning_iterations=a.max_iterations, init_at_random_ep_len=True)


@torch.inference_mode()
def evaluate(args):
    from isaaclab_rl.rsl_rl.utils import create_rsl_rl_runner
    cfg = make_cfg()
    if args.no_faults: cfg.faults = False
    if args.no_guard: cfg.pose_guard = 0.0
    if args.drift_stop: cfg.drift_stop = True             # k1-k3 were trained with it
    if args.mode == "upper": cfg.upper_share = 1.0
    elif args.mode == "whole": cfg.upper_share = 0.0
    cfg.twist_aug = 0.0 if args.no_aug else cfg.twist_aug
    env = KinEnv(cfg, num_envs=args.envs, device=args.device, seed=args.seed)
    runner = create_rsl_rl_runner(env, agent_cfg(1, args.device), log_dir=None)
    runner.load(args.checkpoint, map_location=args.device)
    pi = runner.get_inference_policy(device=args.device)
    obs = env.get_observations(); N = env.num_envs
    acc = {k: [] for k in ("src", "e_h", "a_h", "e_w", "a_w", "turn", "hacc", "pacc", "spd", "base", "lift", "clear", "jerk", "guard")}
    ev = dict(collide=torch.zeros(2, device=args.device), drift=torch.zeros(2, device=args.device), behind=torch.zeros(2, device=args.device), t=torch.zeros(2, device=args.device))
    tR_prev = None; ha_prev = torch.zeros(N, 2, 3, device=args.device)
    steps = int(args.seconds / DT)
    for k in range(steps):
        obs, _, done, _ = env.step(pi(obs))
        src = 1 - env.c_ego[env.c_id].long()                     # 0 EgoDex, 1 LAFAN
        live = env.episode_length_buf > 75                       # past the 1.5 s glide-in
        tR = torch.stack([env.tgt["lw_R"], env.tgt["rw_R"]], 1)
        turn = torch.zeros(N, 2, device=args.device) if tR_prev is None else torch.rad2deg(rot_angle(tR, tR_prev)) / DT
        tR_prev = tR
        for j in range(2):
            m = src == j
            ev["collide"][j] += (env.last_term[:, 0] & m).sum(); ev["drift"][j] += (env.last_term[:, 1] & m).sum() * cfg.drift_stop; ev["behind"][j] += (env.last_term[:, 1] & m).sum() * DT; ev["t"][j] += m.sum() * DT
        if k < 10: continue
        rl = env.r_last; L = live & ~done.bool()
        jerk = ((env.hand_acc - ha_prev) / DT).norm(dim=-1); ha_prev = env.hand_acc.clone()
        acc["src"].append(src[L]); acc["e_h"].append(rl["e_h"][L]); acc["a_h"].append(rl["a_h"][L])
        acc["e_w"].append(torch.stack([rl["e_l"], rl["e_r"]], -1)[L]); acc["a_w"].append(torch.stack([rl["a_l"], rl["a_r"]], -1)[L])
        acc["turn"].append(turn[L]); acc["hacc"].append(env.hand_acc.norm(dim=-1)[L]); acc["jerk"].append(jerk[L])
        acc["pacc"].append(env.palm_acc.norm(dim=-1)[L])
        acc["spd"].append((env.qd[:, env.i_pos].abs() / env.v_lim)[L]); acc["base"].append(env.v_b[L])
        acc["lift"].append(env.qd[L][:, env.i_lift[0]].abs() * 2); acc["clear"].append(env.clear[L]); acc["guard"].append((env.guard_s < 1)[L].float())
    A = {k: torch.cat(v).float() for k, v in acc.items()}
    out = {"checkpoint": args.checkpoint, "faults": not args.no_faults, "mode": args.mode, "envs": N, "seconds": args.seconds}
    for j, nm in enumerate(("egodex", "lafan")):
        m = A["src"] == j
        if m.sum() == 0: continue
        d = dict(head_cm=A["e_h"][m].mean().item() * 100, head_deg=math.degrees(A["a_h"][m].mean().item()),
                 wrist_cm=A["e_w"][m].mean().item() * 100, wrist_cm_p95=A["e_w"][m].flatten().quantile(0.95).item() * 100,
                 hand_acc=A["hacc"][m].mean().item(), hand_jerk=A["jerk"][m].mean().item(), palm_acc=A["pacc"][m].mean().item(),
                 base_speed=A["base"][m][:, :2].norm(dim=-1).mean().item(), base_yaw_rate=A["base"][m][:, 2].abs().mean().item(),
                 lift_mm_s=A["lift"][m].mean().item() * 1000, clear_cm_min=A["clear"][m].min().item() * 100,
                 speed_ratio_max=A["spd"][m].max().item(), guard_share=A["guard"][m].mean().item(),
                 collide_per_min=(ev["collide"][j] / ev["t"][j] * 60).item(), drift_per_min=(ev["drift"][j] / ev["t"][j] * 60).item(), behind_s_per_min=(ev["behind"][j] / ev["t"][j] * 60).item())
        if j == 0:                                            # palm scored on EgoDex only (LAFAN has no finger markers)
            aw, tw = A["a_w"][m].flatten(), A["turn"][m].flatten()
            d["palm_deg"] = math.degrees(aw.mean().item()); d["palm_deg_p95"] = math.degrees(aw.quantile(0.95).item())
            d["palm_deg_by_turn"] = {}
            for lo, hi in ((0, 10), (10, 45), (45, 90), (90, 1e9)):
                b = (tw >= lo) & (tw < hi)
                if b.sum(): d["palm_deg_by_turn"][f"{lo}-{hi if hi < 1e9 else 'inf'}"] = round(math.degrees(aw[b].mean().item()), 2)
        out[nm] = d
    js = json.dumps(out, indent=1)
    print(js)
    if args.out:
        with open(args.out, "w") as f: f.write(js)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--run", default="k1"); t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--checkpoint"); t.add_argument("--reset-iter", action="store_true"); t.add_argument("--std", type=float); t.add_argument("--envs", type=int, default=4096)
    t.add_argument("--seed", type=int, default=42); t.add_argument("--device", default="cuda:0")
    e = sub.add_parser("eval"); e.add_argument("checkpoint"); e.add_argument("--envs", type=int, default=512)
    e.add_argument("--seconds", type=float, default=60.0); e.add_argument("--no-faults", action="store_true"); e.add_argument("--no-guard", action="store_true"); e.add_argument("--drift-stop", action="store_true")
    e.add_argument("--no-aug", action="store_true"); e.add_argument("--mode", default="mixed", choices=("mixed", "whole", "upper"))
    e.add_argument("--seed", type=int, default=7); e.add_argument("--device", default="cuda:0"); e.add_argument("--out")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)

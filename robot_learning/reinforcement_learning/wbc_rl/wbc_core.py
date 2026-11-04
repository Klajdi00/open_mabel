"""wbc_core.py — the parts of NOVA's whole-body RL env that do not depend on the simulator: the operator (clips,
retarget, anchor, Vision Pro stream faults), NOVA's base / lift / gravity feed-forward plant model, collision clearance.

Shared by env.py (Isaac Lab 3 + Newton) and sonic.py (mjlab + MuJoCo Warp) as a mixin. The host class provides
self.robot with Isaac Lab's data API (robot.data.<field>.torch; quaternions xyzw), self.cfg, self.num_envs,
self.device, self.step_dt, self.episode_length_buf, self.extras, self.scene.env_origins, and the buffers its alloc()
makes. Nothing here imports a simulator.
"""
import math

import numpy as np
import torch

import motion_lib as ML
import robot as R

G = 9.81
WHEELS_B = ((0.17, 0.17), (0.17, -0.17), (-0.15, 0.0))     # wheel contacts, REP base frame (FK of nova.urdf)


# ---------------------------------------------------------------- small rotation helpers (torch, matrices)
def quat_xyzw_to_mat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def quat_wxyz_to_mat(q):
    return quat_xyzw_to_mat(torch.cat([q[..., 1:], q[..., :1]], -1))


def rot_z(a):
    c, s, z, o = torch.cos(a), torch.sin(a), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(a.shape + (3, 3))


def rot_y(a):
    c, s, z, o = torch.cos(a), torch.sin(a), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(a.shape + (3, 3))


def orthonorm(M):
    x = torch.nn.functional.normalize(M[..., 0], dim=-1)
    y = M[..., 1] - (M[..., 1] * x).sum(-1, keepdim=True) * x
    y = torch.nn.functional.normalize(y, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def rot_angle(Ra, Rb):
    """Geodesic angle between rotation matrices."""
    tr = (Ra * Rb).sum((-1, -2))
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def rot_x(a):
    c, s, z, o = torch.cos(a), torch.sin(a), torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(a.shape + (3, 3))


def rotvec(v):
    """Rotation vectors (..., 3) -> matrices (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True).clamp_min(1e-9); k = v / th; th = th[..., None]
    K = torch.zeros(v.shape[:-1] + (3, 3), device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    return torch.eye(3, device=v.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def six_d(Rm):
    return Rm[..., :, :2].reshape(Rm.shape[:-2] + (6,))


def rotlog(E):
    """Rotation matrices (...,3,3) -> rotation vectors (...,3)."""
    v = torch.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1) / 2
    sn = v.norm(dim=-1); th = torch.atan2(sn, (E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2] - 1) / 2)
    return v * (th / (sn + 1e-9))[..., None]


def yaw_of(Rm):
    return torch.atan2(Rm[..., 1, 0], Rm[..., 0, 0])


# ---------------------------------------------------------------- the simulator-free env methods
class WbcCore:
    def base_pose(self):
        """REP base frame: position (N,3), yaw (N,), rotation (N,3,3) of the root incl. tilt."""
        q = self.robot.data.root_link_quat_w.torch
        Rw = quat_xyzw_to_mat(q) @ rot_z(torch.full((self.num_envs,), math.pi, device=self.device))
        return self.robot.data.root_link_pos_w.torch, yaw_of(Rw), Rw

    def tracked_now(self):
        bp = self.robot.data.body_link_pos_w.torch[:, self.trk_body]          # (N,3,3)
        bR = quat_xyzw_to_mat(self.robot.data.body_link_quat_w.torch[:, self.trk_body])
        p = bp + (bR @ self.trk_p[None, :, :, None])[..., 0]
        return p, bR @ self.trk_R[None]

    def frame(self, ph):
        """Interpolated human frame at in-clip phase ph (N,) of each env's clip -> dict of tensors.
        Clip index (int) and phase (small float) stay separate: a float32 global index loses half a frame at 5 M."""
        ph = torch.minimum(ph.clamp_min(0), self.c_len[self.c_id].float() - 1.001)
        k = ph.floor()
        i0 = self.c_start[self.c_id] + k.long(); a = (ph - k)[:, None]; i1 = i0 + 1
        L = lambda k: self.m[k][i0] * (1 - a) + self.m[k][i1] * a
        Rr = lambda k: orthonorm(self.m[k][i0] * (1 - a[..., None]) + self.m[k][i1] * a[..., None])
        out = {k: L(k) for k in ("head_p", "lw_p", "rw_p", "le", "re", "ls", "rs", "sm", "neck", "arm_k")}
        # shoulders as the robot will know them: from the head pose (motion_lib.shoulders_from_head), not ARKit's
        hR = Rr("head_R"); hy = torch.atan2(hR[:, 1, 0], hR[:, 0, 0])
        out["ls"], out["rs"] = ML.shoulders_from_head(out["head_p"], hy, torch)
        for s in "lr":                                                 # retarget: NOVA's landmarks (motion_lib.retarget_hand)
            out[s + "w_p"] = ML.retarget_hand(out["head_p"], hy, out[s + "w_p"], out["arm_k"], s, torch)
            out[s + "e"] = ML.retarget_hand(out["head_p"], hy, out[s + "e"], out["arm_k"], s, torch)
            sg = 1.0 if s == "l" else -1.0; x, y, zz = ML.ROBOT_SHOULDER_IN_EYE
            out[s + "s"] = out["head_p"] + torch.stack([torch.cos(hy) * x - torch.sin(hy) * sg * y, torch.sin(hy) * x + torch.cos(hy) * sg * y, 0 * hy + zz], -1)
        out.update({k: Rr(k) for k in ("head_R", "lw_R", "rw_R")})
        out["torso_yaw"] = self.m["torso_yaw"][i0] * (1 - a[:, 0]) + self.m["torso_yaw"][i1] * a[:, 0]
        out["lean"] = self.m["lean"][i0] * (1 - a[:, 0]) + self.m["lean"][i1] * a[:, 0]
        return out

    def compute_targets(self, ph, step_aug=True):
        h = self.frame(ph)
        N = self.num_envs
        # neck-only augmentation: rotate the head about its own neck pivot (torso labels unchanged)
        if step_aug and not self.cfg.scripted:
            self.neck_off += (self.neck_tgt - self.neck_off) * (1 - math.exp(-self.step_dt / 0.5))
            chg = torch.rand(N, device=self.device) < self.step_dt / 2.0
            self.neck_tgt[chg] = (torch.rand(int(chg.sum()), 2, device=self.device) * 2 - 1) * torch.tensor([0.8, 0.35], device=self.device)
        off = self.neck_off * self.neck_on[:, None]
        pivot = h["head_p"] - (h["head_R"] @ torch.tensor([0.10, 0.0, 0.08], device=self.device)[:, None])[..., 0]
        dR = rot_z(off[:, 0])
        head_R = dR @ h["head_R"] @ rot_y(off[:, 1])
        head_p = pivot + (head_R @ torch.tensor([0.10, 0.0, 0.08], device=self.device)[:, None])[..., 0]
        neck = h["neck"] + torch.stack([off[:, 0], off[:, 1], torch.zeros_like(off[:, 0])], -1)
        # synthetic walk / turn: rigid motion of the whole person about their torso axis
        if step_aug and not self.cfg.scripted:
            dt = self.step_dt
            chg = torch.rand(N, device=self.device) < dt / 1.5
            n = int(chg.sum())
            if n:
                self.aug_w[chg] = (torch.rand(n, device=self.device) * 2 - 1) * 0.7 * (torch.rand(n, device=self.device) < 0.7)
                sp = torch.rand(n, device=self.device) * self.cfg.aug_walk_max * (torch.rand(n, device=self.device) < 0.6)
                sp = sp * self.c_ego[self.c_id[chg]].float()
                ang = torch.rand(n, device=self.device) * 2 * math.pi
                self.aug_v[chg] = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang)], -1)
            self.aug_yaw += self.aug_w * self.aug_on * dt
            self.aug_xy += self.aug_v * self.aug_on[:, None] * dt
        c = torch.cat([h["sm"][:, :2], torch.zeros(N, 1, device=self.device)], -1)
        Ra = rot_z(self.aug_yaw)
        # upper mode: remove the operator's SLOW walking drift (5 s low-pass of the head-estimated shoulders; the
        # adapter runs the same), quick leans stay
        sm_e = 0.5 * (h["ls"] + h["rs"])[:, :2]
        if step_aug: self.sm_lp += (1 - math.exp(-self.step_dt / 5.0)) * (sm_e - self.sm_lp)
        d = torch.cat([self.aug_xy - self.upper[:, None] * (self.sm_lp - self.sm0), torch.zeros(N, 1, device=self.device)], -1)
        A = lambda p: c + (Ra @ (p - c)[..., None])[..., 0] + d
        # anchor into the robot world
        Rn = rot_z(self.anc_yaw)
        W = lambda p: (Rn @ (A(p) - self.anc_h)[..., None])[..., 0] + self.anc_r
        RR = Rn @ Ra
        dz = torch.tensor([0.0, 0.0, self.wrist_dz], device=self.device)
        t = dict(head_p=W(head_p), head_R=RR @ head_R,
                 lw_p=W(h["lw_p"]) + dz, lw_R=RR @ h["lw_R"], rw_p=W(h["rw_p"]) + dz, rw_R=RR @ h["rw_R"],
                 l_arm=torch.nn.functional.normalize(RR @ (h["le"] - h["ls"])[..., None], dim=-2)[..., 0],
                 r_arm=torch.nn.functional.normalize(RR @ (h["re"] - h["rs"])[..., None], dim=-2)[..., 0],
                 le=W(h["le"]) + dz, re=W(h["re"]) + dz, ls=W(h["ls"]) + dz, rs=W(h["rs"]) + dz,   # operator skeleton (site)
                 sm=W(h["sm"]), body_yaw=h["torso_yaw"] + self.aug_yaw + self.anc_yaw, lean=h["lean"], neck=neck)
        # glide / chain blend: from a frozen pose into the clip (smoothstep), only for the three tracked frames
        al = (self.blend_t / self.blend_T).clamp(0, 1); al = al * al * (3 - 2 * al)
        al = torch.where(self.blend_on > 0, al, torch.ones_like(al))
        for j, k in enumerate(("head", "lw", "rw")):
            t[k + "_p"] = self.blend_p[:, j] + al[:, None] * (t[k + "_p"] - self.blend_p[:, j])
            t[k + "_R"] = orthonorm(self.blend_R[:, j] + al[:, None, None] * (t[k + "_R"] - self.blend_R[:, j]))
        return t

    def swerve(self, tw, dt):
        """Body twist (N,3) -> what three steered modules deliver (nova_hw subsystems/base.py): each aims at atan2 of its
        velocity (holding below 0.02 m/s), decided against its LAST COMMANDED heading (flip rather than turn > 90 deg),
        drives at speed x cos(that turn), 0 past 30 deg, along wherever the wheel actually points; steers at a finite rate."""
        vx, vy, wz = tw[:, 0:1], tw[:, 1:2], tw[:, 2:3]
        mx, my = self.mods[:, 0][None], self.mods[:, 1][None]
        ux, uy = vx - wz * my, vy + wz * mx                                     # (N,3)
        sp = torch.hypot(ux, uy)
        head = torch.where(sp > 0.02, torch.atan2(uy, ux), self.steer_cmd)
        turn = wrap(head - self.steer_cmd)
        flip = turn.abs() > math.pi / 2
        turn = torch.where(flip, wrap(turn + math.pi), turn); sp = torch.where(flip, -sp, sp)
        self.steer_cmd = wrap(self.steer_cmd + turn)
        cosf = torch.where(turn.abs() > R.STEER_COS_GATE, torch.zeros_like(turn), torch.cos(turn))
        step = (self.cfg.steer_rate * self.steer_k * dt)[:, None]
        self.steer = wrap(self.steer + wrap(self.steer_cmd - self.steer).clamp(-step, step))
        v = sp * cosf * self.drive_k
        u = torch.stack([v * torch.cos(self.steer), v * torch.sin(self.steer)], -1).reshape(-1, 6)
        return u @ self.mods_pinv.T

    def gravity_ff(self):
        """Torque each position joint needs to hold its subtree against gravity, from the NOMINAL masses (the robot's
        model, no payload) times a per-env calibration scale: NOVA's gravity feed-forward, imperfect like the real one."""
        com = self.robot.data.body_com_pos_w.torch                                   # (N,B,3)
        jp = self.robot.data.body_link_pos_w.torch[:, self.gc_body]                  # (N,J,3) joint origin = child frame
        jR = quat_xyzw_to_mat(self.robot.data.body_link_quat_w.torch[:, self.gc_body])
        ax = (jR @ self.gc_axis[None, :, :, None])[..., 0]                           # (N,J,3)
        f = torch.zeros(3, device=self.device); f[2] = -G
        mg = self.mass0[None, :, None] * f                                           # (1,B,3)
        r = com[:, None] - jp[:, :, None]                                            # (N,J,B,3)
        tau_g = (torch.cross(r, mg[:, None].expand_as(r), dim=-1) * self.gc_sub[None, :, :, None]).sum(2)
        tau = -(tau_g * ax).sum(-1) * self.gc_scale
        return torch.max(torch.min(tau, self.ff_max), -self.ff_max)

    def lift_ff(self):
        R_ = quat_xyzw_to_mat(self.robot.data.body_link_quat_w.torch[:, self.lf_body])
        ax = (R_ @ self.lf_axis[None, :, :, None])[..., 0]                            # (N,2,3)
        m_up = (self.masses[:, None] * self.lf_sub[None]).sum(-1)                    # (N,2) true masses incl. payload
        return m_up * G * ax[..., 2]

    def _update_state(self):
        dt = self.step_dt
        # glide-in after a reset: blend from where the robot's frames actually are
        g = self.need_glide
        if g.any():
            cp, cR = self.tracked_now()
            self.blend_p[g], self.blend_R[g] = cp[g], cR[g]
            self.blend_on[g], self.blend_t[g], self.blend_T[g] = 1.0, 0.0, 1.0
            self.need_glide[:] = False
        self.blend_t += dt
        # advance the clip clock; a finished clip chains into a new one where the operator is now
        self.phase += self.rate
        done = self.phase >= self.c_len[self.c_id].float() - 1.001
        if done.any():
            self.chain(done.nonzero()[:, 0])
        old = torch.stack([self.tgt["lw_p"], self.tgt["rw_p"]], 1)
        self.tgt = self.compute_targets(self.phase)
        self.stream()
        v = (torch.stack([self.tgt["lw_p"], self.tgt["rw_p"]], 1) - old) / dt          # the operator's hand velocity
        self.tgt_v = v * (3.0 / v.norm(dim=-1, keepdim=True).clamp_min(3.0))
        # COM, ZMP and stability margin
        com = (self.robot.data.body_com_pos_w.torch * self.masses[..., None]).sum(1) / self.m_tot[:, None]
        com_v = (self.robot.data.body_com_lin_vel_w.torch * self.masses[..., None]).sum(1) / self.m_tot[:, None]
        # random pushes on the torso (someone bumps the robot)
        self.push_t -= dt
        start = (torch.rand(self.num_envs, device=self.device) < dt / self.cfg.push_every) & (self.push_t <= 0)
        if start.any():
            n = int(start.sum()); ang = torch.rand(n, device=self.device) * 2 * math.pi
            mag = torch.empty(n, device=self.device).uniform_(*self.cfg.push_force)
            self.push_f[start] = torch.stack([mag * torch.cos(ang), mag * torch.sin(ang), torch.zeros_like(mag)], -1)
            self.push_t[start] = 0.2
        self.com_acc = 0.7 * self.com_acc + 0.3 * (com_v - self.com_v_prev) / dt
        self.com_v_prev = com_v
        pos, yaw, Rb = self.base_pose()
        zmp = com[:, :2] - com[:, 2:3] / G * self.com_acc[:, :2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        d = zmp - pos[:, :2]
        zb = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
        m = []
        for i in range(3):
            a_, b_, o_ = self.wheels[i], self.wheels[(i + 1) % 3], self.wheels[(i + 2) % 3]
            e = b_ - a_; n = torch.stack([e[1], -e[0]]) / e.norm()
            n = n if (n * (o_ - a_)).sum() > 0 else -n
            m.append(((zb - a_) * n).sum(-1))
        self.zmp_margin = torch.stack(m, -1).min(-1).values
        if self.n_gov % 5 == 0:                                       # GOV_HZ 10: stability.limits on the live COM
            d = com[:, :2] - pos[:, :2]
            cb = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)
            mg = []
            for i in range(3):
                a_, b_, o_ = self.wheels[i], self.wheels[(i + 1) % 3], self.wheels[(i + 2) % 3]
                e = b_ - a_; n = torch.stack([e[1], -e[0]]) / e.norm()
                n = n if (n * (o_ - a_)).sum() > 0 else -n
                mg.append(((cb - a_) * n).sum(-1))
            margin = torch.stack(mg, -1).min(-1).values; h = com[:, 2].clamp_min(0.1)
            acc = (G * (margin - 0.03).clamp_min(0.0) / h).clamp_min(0.15)
            dm = (margin - 0.01).clamp_min(0.0); r = torch.hypot(h, dm)
            spd = torch.where(dm > 0, (torch.sqrt(2 * G * (r - h)) * r / h).clamp_min(0.10), torch.full_like(h, 0.10))
            rc = (cb - 0.5 * (self.wheels[0] + self.wheels[1])).norm(dim=-1).clamp_min(0.02)
            yaw = torch.sqrt(acc / rc).clamp_min(0.3)
            spd = torch.where(self.coast, torch.full_like(spd, R.BASE_VMAX), spd)   # cap lifted while drives coast
            self.gov = torch.stack([acc.clamp(max=R.BASE_ACCEL), spd.clamp(max=R.BASE_VMAX), yaw.clamp(max=R.BASE_WMAX),
                                    torch.min(acc.clamp(max=R.BASE_ACCEL) * R.BASE_DECEL, acc)], -1)   # base.py: decel = min(1.5 a, accel_max)
        self.n_gov += 1
        self.clear, self.clear_mv = self.clearance()       # every pair (terminates) / all but the shoulder motors (4 cm band)
        # actuator load: NOVA's PD + feed-forward, clipped like the solver; RMS over torque_tau for the thermal limit
        q = self.robot.data.joint_pos.torch[:, self.jid]; qd = self.robot.data.joint_vel.torch[:, self.jid]
        ip = self.i_pos
        tau = self.kp_env * (self.q_tgt[:, ip] - q[:, ip]) + self.kd_env * (self.v_tgt[:, ip] - qd[:, ip]) + self.tau_g
        self.tau = torch.max(torch.min(tau, self.t_pk), -self.t_pk)
        self.tau2 = self.tau2 + (dt / self.cfg.torque_tau) * (self.tau.square() - self.tau2)
        tol = self.cfg.limit_tol
        lift_v = qd[:, self.i_lift].abs().max(-1).values
        # per joint (ACTUATED order, lift first) and kind: speed, overload, position — what play.py blames a stop on
        z = torch.zeros_like(q, dtype=torch.bool)
        sp = z.clone(); sp[:, ip] = qd[:, ip].abs() > self.v_lim * tol; sp[:, self.i_lift] = qd[:, self.i_lift].abs() > 0.5 * R.LIFT_VLIM * tol
        ov = z.clone(); ov[:, ip] = (self.tau2.sqrt() > self.t_rated) & self.ovl_hard
        self.lim_joint = torch.stack([sp, ov, (q < self.q_lo - 0.01) | (q > self.q_hi + 0.01)], 1)       # (N, 3, J)
        self.lim_kind = torch.stack([sp.any(-1), (tau.abs() > self.t_pk * tol).any(-1), ov.any(-1), self.lim_joint[:, 2].any(-1)], -1)

    def chain(self, ids):
        """Continue envs `ids` with a new clip, anchored so the operator's heading and head stay put; 0.5 s blend."""
        old = self.tgt
        n, dev = len(ids), self.device
        c = torch.multinomial(self.c_p, n, replacement=True)
        up = self.upper[ids] > 0
        if up.any(): c[up] = torch.multinomial(self.c_p_up, int(up.sum()), replacement=True)
        self.c_id[ids] = c; self.phase[ids] = 0.0
        self.rate[ids] = torch.empty(n, device=dev).uniform_(*self.cfg.speed_range)
        i0 = self.c_start[c]
        self.anc_yaw[ids] = old["body_yaw"][ids] - self.m["torso_yaw"][i0]
        self.aug_yaw[ids] = 0.0; self.aug_xy[ids] = 0.0
        self.anc_h[ids] = self.m["head_p"][i0]
        hRc = self.m["head_R"][i0]; e0, e1 = ML.shoulders_from_head(self.m["head_p"][i0], torch.atan2(hRc[:, 1, 0], hRc[:, 0, 0]), torch)
        self.sm0[ids] = 0.5 * (e0 + e1)[:, :2]; self.sm_lp[ids] = self.sm0[ids]
        pivot_old = old["head_p"][ids]
        self.anc_r[ids] = pivot_old
        for j, k in enumerate(("head", "lw", "rw")):
            self.blend_p[ids, j] = old[k + "_p"][ids]; self.blend_R[ids, j] = old[k + "_R"][ids]
        self.blend_on[ids], self.blend_t[ids], self.blend_T[ids] = 1.0, 0.0, 0.5

    def clearance(self):
        """Smallest modelled gap (m) over every forbidden pair (robot.collision_model): sphere vs the other part's field."""
        bp = self.robot.data.body_link_pos_w.torch; bq = quat_xyzw_to_mat(self.robot.data.body_link_quat_w.torch)
        P = bp[:, self.sph_body] + (bq[:, self.sph_body] @ self.sph_c[None, :, :, None])[..., 0]       # (N,S,3)
        d = torch.full((self.num_envs,), 1.0, device=self.device); d_sh = d.clone(); ch = torch.ones(self.num_envs, 2, device=self.device)
        for body, org, ext, g, ids, fh in self.sdf:
            loc = (P[:, ids] - bp[:, body, None]) @ bq[:, body]                                       # into the part's frame
            u = (2 * (loc - org) / ext - 1).reshape(1, 1, 1, -1, 3)
            f = torch.nn.functional.grid_sample(g, u, mode="bilinear", padding_mode="border", align_corners=True)
            dd = f.view(self.num_envs, -1) - self.sph_r[ids]; sh = self.sph_sh[ids]
            d = torch.minimum(d, dd.masked_fill(sh, 1.0).amin(-1)); d_sh = torch.minimum(d_sh, dd.masked_fill(~sh, 1.0).amin(-1))
            ch = torch.minimum(ch, dd[:, :, None].masked_fill(~self.sph_hand[ids][None], 1.0).amin(1))     # a hand's spheres
            if fh >= 0: ch[:, fh] = torch.minimum(ch[:, fh], dd.amin(-1))                               # or a hand's field
        self.clear_sh = d_sh; self.clear_hand = ch
        return torch.minimum(d, d_sh), d

    def stream(self):
        """Vision Pro faults: the clean operator frames (self.tgt) -> what the gateway receives (self.rx_*, self.age).
        A lost part (stall, or a hand out of tracking) is scored against where it was last seen."""
        S, N, dev, dt = self.cfg.stream, self.num_envs, self.device, self.step_dt
        ks = ("head", "lw", "rw")
        P = torch.stack([self.tgt[k + "_p"] for k in ks], 1); Rm = torch.stack([self.tgt[k + "_R"] for k in ks], 1)
        self.buf_p = torch.cat([P[:, None], self.buf_p[:, :-1]], 1); self.buf_R = torch.cat([Rm[:, None], self.buf_R[:, :-1]], 1)
        a, b, c = S["odom"]
        lin = self.robot.data.root_link_lin_vel_w.torch; wz = self.robot.data.root_link_ang_vel_w.torch[:, 2]
        self.odo_xy = (self.odo_xy + torch.randn(N, 2, device=dev) * (a + b * lin[:, :2].norm(dim=-1, keepdim=True)) * dt ** 0.5).clamp(-0.2, 0.2)
        self.odo_yaw = (self.odo_yaw + torch.randn(N, device=dev) * (0.1 * a + c * wz.abs()) * dt ** 0.5).clamp(-0.2, 0.2)
        if not self.cfg.faults:
            self.rx_p, self.rx_R = P, Rm; self.age.zero_(); self.odo_xy.zero_(); self.odo_yaw.zero_(); self.euro_step(self.rx_R[:, 1:]); return
        u = lambda *s: torch.rand(N, *s, device=dev)
        # connection: stalls, frame-rate drops, packet loss
        self.out_t = torch.where((self.out_t <= 0) & (u() < S["outage_rate"] * dt), 0.1 + 1.4 * u(), self.out_t - dt)
        new_slow = (self.slow_t <= 0) & (u() < S["slow_rate"] * dt)
        k = torch.tensor([2, 3, 4, 6], device=dev)[torch.randint(0, 4, (N,), device=dev)]
        self.slow_k = torch.where(new_slow, k, torch.where(self.slow_t > 0, self.slow_k, torch.ones_like(self.slow_k)))
        self.slow_t = torch.where(new_slow, 0.5 + 2.5 * u(), self.slow_t - dt)
        self.n_step += 1
        arrive = (self.out_t <= 0) & (self.n_step % self.slow_k == 0) & (u() >= self.loss)
        # hand tracking loss: the hand freezes, or keeps coming as a drifting prediction
        new_occ = (self.occ_t <= 0) & (u(2) < S["occl_rate"] * dt)
        self.occ_pred = torch.where(new_occ, (u(2) < S["predict"]).float(), self.occ_pred)
        self.occ_t = torch.where(new_occ, 0.1 + 1.9 * u(2), self.occ_t - dt)
        occ = self.occ_t > 0
        # a fresh frame: the clean one from lat (+ jitter) steps ago, calibration offset, white noise
        lag = (self.lat + (u() < S["lat_jit"]).long()).clamp(max=self.buf_p.shape[1] - 1)
        ar = torch.arange(N, device=dev)
        R_ = self.buf_R[ar, lag]
        p = self.buf_p[ar, lag] + (R_ @ self.bias_p[..., None])[..., 0] + torch.randn(N, 3, 3, device=dev) * self.npos[:, None, None]
        # tremor: a rotating phasor per part and axis (6), decorrelating over trem_tau, RMS exactly trem_p / trem_r
        rho = math.exp(-dt / S["trem_tau"])
        self.trem_z = self.trem_z * torch.polar(torch.full_like(self.trem_w, rho), self.trem_w * dt)[..., None] + \
            math.sqrt((1 - rho * rho) / 2) * torch.complex(torch.randn(N, 3, 6, device=dev), torch.randn(N, 3, 6, device=dev))
        tr = self.trem_z.real * math.sqrt(2)
        p = p + tr[..., :3] * self.trem_a[..., 0:1]
        R_ = R_ @ self.bias_R @ rotvec(torch.randn(N, 3, 3, device=dev) * self.nrot[:, None, None] + tr[..., 3:] * self.trem_a[..., 1:2])
        # glitches: a hand frame jumps for 1-3 frames
        new_spk = arrive[:, None] & (self.spk_n <= 0) & (u(2) < S["spike"])
        m = int(new_spk.sum())
        if m:
            nrm = lambda x: torch.nn.functional.normalize(x, dim=-1)
            self.spk_p[new_spk] = nrm(torch.randn(m, 3, device=dev)) * (0.03 + 0.12 * torch.rand(m, 1, device=dev))
            self.spk_R[new_spk] = rotvec(nrm(torch.randn(m, 3, device=dev)) * (0.17 + 0.52 * torch.rand(m, 1, device=dev)))
            self.spk_n[new_spk] = torch.randint(1, 4, (m,), device=dev)
        sp = (self.spk_n > 0) & arrive[:, None]
        p = torch.cat([p[:, :1], torch.where(sp[..., None], p[:, 1:] + self.spk_p, p[:, 1:])], 1)
        R_ = torch.cat([R_[:, :1], torch.where(sp[..., None, None], self.spk_R @ R_[:, 1:], R_[:, 1:])], 1)
        self.spk_n = self.spk_n - sp.long()
        fresh = torch.cat([arrive[:, None], arrive[:, None] & ~occ], 1)
        prev = self.rx_p[:, 1:].clone()
        self.rx_p = torch.where(fresh[..., None], p, self.rx_p); self.rx_R = torch.where(fresh[..., None, None], R_, self.rx_R)
        hv = ((self.rx_p[:, 1:] - prev) / (self.age[:, 1:, None] + dt)).clamp(-1.5, 1.5)
        self.rx_v = torch.where(fresh[:, 1:, None], 0.7 * self.rx_v + 0.3 * hv, self.rx_v)
        pred = arrive[:, None] & occ & (self.occ_pred > 0)
        self.rx_v = torch.where(pred[..., None], self.rx_v * math.exp(-dt / 0.4), self.rx_v)
        self.rx_p = torch.cat([self.rx_p[:, :1], torch.where(pred[..., None], self.rx_p[:, 1:] + self.rx_v * dt, self.rx_p[:, 1:])], 1)
        self.age = torch.where(fresh, torch.zeros_like(self.age), self.age + dt)
        # reward reference: a stalled stream or an untracked hand holds where it was last seen
        lost = torch.cat([(self.out_t > 0)[:, None], (self.out_t > 0)[:, None] | occ], 1)
        start = lost & ~self.lost_prev
        self.hold_p = torch.where(start[..., None], P, self.hold_p); self.hold_R = torch.where(start[..., None, None], Rm, self.hold_R)
        self.rec_t = torch.where((self.lost_prev & ~lost).any(-1), torch.zeros_like(self.rec_t), self.rec_t + dt)
        self.lost_prev = lost
        for j, k in enumerate(ks):
            self.tgt[k + "_p"] = torch.where(lost[:, j, None], self.hold_p[:, j], P[:, j])
            self.tgt[k + "_R"] = torch.where(lost[:, j, None, None], self.hold_R[:, j], Rm[:, j])
        self.euro_step(self.rx_R[:, 1:])

    def stream_reset(self, env_ids):
        S, n, dev = self.cfg.stream, len(env_ids), self.device
        U = lambda lo, hi, *s: torch.empty(n, *s, device=dev).uniform_(lo, hi)
        self.lat[env_ids] = torch.randint(S["lat"][0], S["lat"][1] + 1, (n,), device=dev)
        self.loss[env_ids] = U(*S["loss"]); self.npos[env_ids] = U(*S["noise_p"]); self.nrot[env_ids] = U(*S["noise_r"])
        sc3 = torch.tensor([0.3, 1.0, 1.0], device=dev)[None, :]
        self.trem_w[env_ids] = 2 * math.pi * U(*S["trem_hz"], 3)
        self.trem_a[env_ids] = torch.stack([U(*S["trem_p"], 3), U(*S["trem_r"], 3)], -1) * sc3[..., None]
        self.trem_z[env_ids] = 0
        sc = torch.tensor([0.5, 1.0, 1.0], device=dev)[None, :, None]
        self.bias_p[env_ids] = U(-1, 1, 3, 3) * S["bias_p"] * sc
        self.bias_R[env_ids] = rotvec(U(-1, 1, 3, 3) * S["bias_r"] * sc)
        for x in (self.slow_t, self.out_t, self.occ_t, self.occ_pred, self.spk_n, self.rx_v, self.age, self.odo_xy, self.odo_yaw):
            x[env_ids] = 0
        self.slow_k[env_ids] = 1; self.rec_t[env_ids] = 10.0; self.lost_prev[env_ids] = False
        P = torch.stack([self.tgt[k + "_p"] for k in ("head", "lw", "rw")], 1)
        Rm = torch.stack([self.tgt[k + "_R"] for k in ("head", "lw", "rw")], 1)
        self.rx_p[env_ids] = P[env_ids]; self.rx_R[env_ids] = Rm[env_ids]
        self.hold_p[env_ids] = P[env_ids]; self.hold_R[env_ids] = Rm[env_ids]
        self.buf_p[env_ids] = P[env_ids, None]; self.buf_R[env_ids] = Rm[env_ids, None]
        self.oe_R[env_ids] = Rm[env_ids, 1:]; self.oe_s[env_ids] = 0

    def target_obs(self, pos, yaw, noisy):
        """Target frames in the base yaw frame (27,): clean, or as received (stream faults, odometry error)."""
        if noisy:
            pos = pos + torch.cat([self.odo_xy, torch.zeros_like(self.odo_yaw)[:, None]], -1); yaw = yaw + self.odo_yaw
        Rb = rot_z(yaw).transpose(-1, -2)
        out = []
        for j, k in enumerate(("head", "lw", "rw")):
            P, Rm = (self.rx_p[:, j], self.rx_R[:, j]) if noisy else (self.tgt[k + "_p"], self.tgt[k + "_R"])
            out += [(Rb @ (P - pos)[..., None])[..., 0], six_d(Rb @ Rm)]
        return torch.cat(out, -1)

    def act_obs(self):
        """Base + lift commands as sent, position joints as the filter state (action units): what the motors are heading to."""
        return torch.cat([self.actions[:, :4], (self.q_f - self.q_ready[self.i_pos]) / self.cfg.action_scale], -1)

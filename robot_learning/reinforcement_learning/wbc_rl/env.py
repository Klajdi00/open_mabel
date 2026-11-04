"""env.py — NOVA whole-body teleop tracking: one RL policy, Vision Pro head + two wrists in, every joint out.

Isaac Lab 3 DirectRLEnv on Newton (MuJoCo-Warp); the simulator-free parts (operator, stream faults, base / lift /
gravity plant, collision clearance) are wbc_core.WbcCore. Registered by tasks.py as `NOVA-WBC-v0`; train with Isaac Lab's
scripts/reinforcement_learning/train.py (--external_callback tasks.register), roll out / score with play.py.

POLICY (50 Hz) sees only what the robot has at teleop time:
  the three target frames (head eye, ORCA-wrist palms) now, 40 ms and 100 ms ago, in the base's yaw frame;
  where its own three frames are (FK of the encoders); joint positions / velocities; base twist (odometry);
  its last action, and the mode bit (1 = upper body: base and lift locked, 0 = whole body). No tilt, no human body.
ACTIONS (22): base twist vx, vy, wz (the swerve IK on the robot turns it into wheel commands) · lift rate (stages held
  robot.LIFT_MARGIN off both stops) ·
  torso pitch · neck yaw/pitch/roll · 14 arm joints (position targets around the ready pose, low-passed at
  action_lpf_hz and ramped across physics substeps: run the same filter on NOVA).
CRITIC additionally sees privileged state: tilt, ZMP margin, collision clearance, future targets, the
  human's torso heading / position / lean / neck angles / elbows (motion_lib.py), randomised parameters.

Hard terminations (penalised): tilt > TILT_MAX · any arm touching the body/head/lift/base or the other arm ·
head / wrist / body-heading drift past DRIFT_* is logged, not a stop (v40). Timeouts at clip end bootstrap.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import motion_lib as ML
import robot as R
from wbc_core import (G, WHEELS_B, WbcCore, orthonorm, quat_wxyz_to_mat, quat_xyzw_to_mat, rot_angle,  # noqa: F401
                      rot_x, rot_y, rot_z, rotlog, rotvec, six_d, wrap, yaw_of)

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- config
@configclass
class NovaSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1,
                                 physics_material=sim_utils.RigidBodyMaterialCfg(
                                     friction_combine_mode="multiply", static_friction=0.0, dynamic_friction=0.0))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))
    robot: ArticulationCfg = None


@configclass
class NovaWbcEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 22
    observation_space = 191
    state_space = 178
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4, physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=80, nconmax=16, cone="elliptic", impratio=1.0, integrator="implicitfast"),
        num_substeps=1))
    scene: NovaSceneCfg = NovaSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    motions = os.path.join(HERE, "data", "motions_v4.npz")   # EgoDex torso heading re-centred on the gaze (motion_lib.py)
    egodex_share = 0.5                  # probability a reset draws an EgoDex clip (else LAFAN1 walking)
    speed_range = (0.7, 1.4)            # playback-rate randomisation: how fast the human moves
    twist_aug = 0.6                     # fraction of envs that get synthetic walk/turn on top of the clip
    aug_walk_max = 0.35                 # m/s synthetic walking, EgoDex clips only (LAFAN1 already walks; both together
                                        # outran NOVA's 0.6 m/s cap and made drift unavoidable)
    upper_share = 0.3                   # upper-body mode (the deploy toggle): base and lift locked, EgoDex clips
                                        # only, the operator's walking drift removed; the policy sees the mode bit
    neck_aug = 0.5                      # fraction of envs that get extra neck-only head rotation
    scripted = False                    # play.py scenarios set aug_yaw / aug_xy / neck_off themselves
    action_scale = 0.5
    wrist_ik = True                     # v28: palm orientation solved exactly by the wrist joints (robot.wrist_ik) for the
                                        # measured forearm; the policy adds a residual of at most wrist_res rad
    wrist_res = 0.0                     # v30: 0 — the policy's residual added 4.5 deg (MuJoCo, v29): the solver alone is exact
    hand_guard = (0.015, 0.035)         # m: the solver's wrist command stops at this hand clearance and goes back to its
                                        # last pose with room (robot.hand_guard; v33: the solver alone put hands into the
                                        # other hand / the lift top, 1.4 stops/min on walking clips)
    wrist_lpf_hz = 20.0                 # the wrist joints' command filter (v30: 5 -> 10 Hz: 6.1 -> 4.8 deg under tremor;
                                        # v33: 20 Hz, MuJoCo 4.9 -> 4.4 deg, 4.97 -> 4.51 with tremor, palm jitter +17 %)
    # one-euro filter on the received palm orientation (the solver's input): cutoff = min + beta x speed (rad/s)
    euro = dict(min_hz=1.5, beta=2.0, d_hz=1.0)
    action_lpf_hz = 3.0                 # two cascaded first-order low-passes (critically damped, 2nd order) on the 18
                                        # position targets (run the same filter on NOVA): v19's 1st-order 4 Hz let the
                                        # policy's step-to-step chatter through as ~5 Hz arm oscillation,
                                        # then a linear ramp across the physics substeps: no 50 Hz steps reach the motors
    hw_smooth_wn = 40.0                 # rad/s, nova_hw NOVA_HW_SMOOTH_WN: the setpoint follower on every DaMiao joint
    hw_wrist_wn = 80.0                  # rad/s, nova_hw NOVA_HW_WRIST_WN: arm_5-7 follow faster (v35)
    action_delay = (1, 4)               # policy steps of output latency (20-80 ms): gateway tick hand-off, CAN, nova_hw
                                        # shaper ramp (~50 ms at wn 40)
    # Vision Pro stream faults, applied to what the gateway receives ("dropped predicted frames,
    # occlusions, connection issues, frame rate drop"). Rewards score the clean operator, except a lost part is
    # scored against where it was last seen: hold still, do not chase a stale or predicted hand.
    faults = True
    stream = dict(lat=(0, 5),           # policy steps of input latency per episode (0-100 ms) ...
                  lat_jit=0.25,         # ... +1 step on this share of frames (network jitter)
                  loss=(0.0, 0.15),     # per-episode packet loss: the frame does not arrive, the last one stays
                  slow_rate=1 / 6.0,    # per s: frame rate drops to 25 / 17 / 12 / 8 Hz for 0.5-3 s
                  outage_rate=1 / 15.0, # per s: the connection stalls 0.1-1.5 s, then the stream jumps to now
                  occl_rate=1 / 8.0,    # per s per hand: hand tracking lost 0.1-2 s (out of view, occluded) ...
                  predict=0.5,          # ... the hand then freezes, or (this share) keeps coming as a drifting prediction
                  spike=0.004,          # per received hand frame: a 3-15 cm / 10-40 deg glitch for 1-3 frames
                  noise_p=(0.001, 0.004), noise_r=(0.005, 0.04),   # white per frame: m, rad per axis (level per episode)
                  # hand tremor / unsteady operator / tracking wobble: narrow-band noise at 3-12 Hz per hand (physiological
                  # tremor 8-12 Hz, unsteady hands 3-6 Hz), RMS per episode up to 6 mm and 0.05 rad (head 30 %),
                  # coherent for ~0.3 s. Rewards score the clean operator: the policy must smooth it, not follow it
                  trem_hz=(3.0, 12.0), trem_p=(0.0, 0.006), trem_r=(0.0, 0.05), trem_tau=0.3,
                  bias_p=0.015, bias_r=0.07,            # per episode wrist calibration offset (m, rad); head half
                  odom=(0.0005, 0.03, 0.02))            # base pose estimate random walk: m/sqrt(s) + per m/s, rad per rad/s
    dr_physics = True                   # gains x0.75-1.25, link mass x0.85-1.15, hand payload 0-0.3 kg, torso pushes
    push_force = (10.0, 30.0)           # N, 0.2 s, on average every push_every s (<= 42 N m at the torso: survivable;
                                        # 90 N tips NOVA whatever the policy does: 62 kg x g x 8.5 cm = 52 N m)
    push_every = 4.0
    # stability / safety
    tilt_max = math.radians(2.0)
    zmp_out = -0.01                     # m: ZMP this far outside the wheel triangle ...
    zmp_out_s = 0.1                     # ... for this long terminates (a wheel is unloading)
    zmp_reserve = 0.03
    grace_s = 1.5                       # covers the 1 s glide-in after a reset
    drift_head = 0.45                   # m
    drift_wrist = 0.60                  # m
    drift_head_rot = math.radians(60.0)
    drift_body_yaw = math.radians(75.0)
    drift_stop = False                  # v40: falling behind no longer ends the episode (v39: 32 drift stops/min on LAFAN, the
                                        # operator outwalks the 0.6 m/s base ~18 % of the time); the tracking terms keep pulling
    collide_margin = 0.01               # m: modelled gap below this terminates; the model itself reads ~1.7 cm low
    # actuator limits (robot.JOINT_SPEC x SAFETY): speed and peak torque capped in the solver and checked here;
    # continuous torque = RMS over torque_tau s must stay under SAFETY x rated. Any breach terminates.
    torque_tau = 5.0
    limit_tol = 1.0                     # zero tolerance: the solver runs at robot.HEADROOM under the checked limit
    # swerve base (nova_hw subsystems/base.py): twist low-pass 0.08 s, modules flip instead of turning > 90 deg,
    # drive scaled by cos(turn of the commanded heading this tick), 0 past 30 deg (MABEL's rule; no gate on
    # the measured steer error, so a lagging wheel pushes off-line as on NOVA), steer 90 deg in ~0.17 s (steer_kp sweep)
    twist_tau = 0.08
    steer_rate = (math.pi / 2) / 0.17   # rad/s, randomised +-20 %
    # observation noise
    noise = dict(q=0.005, qd=0.08, base=0.05)   # white per step: a policy learns to chase it
    # reward weights (per second; multiplied by dt)
    w = dict(head_p=3.0, head_r=1.5, wrist_p=4.0, wrist_p_mid=4.0, wrist_p_fine=5.0, wrist_r=8.0, track_lin=-1.0,
             hand_vel=1.0, body_yaw=0.6, body_xy=0.4, lean=1.0, lean_excess=-5.0, neck=0.4, elbow=1.5,
             base_use=-0.25, base_acc=-0.1, steer=-0.01, lift_use=-2.5, act_sat=-0.5, action_rate=-0.06, action_acc=-0.01, hand_acc=-0.09, palm_acc=-0.0003,
             arm_acc=-1.25e-4, qf_acc=-3e-5, energy=-0.005, wrist_unreach=-3.0, torque=-0.02, joint_vel=-1e-3, joint_acc=-2e-7,
             joint_limit=-10.0, limits=-6.0, past_stop=-250.0, twist=-0.05, tilt=-60.0, tilt_rate=-2.0, zmp=-40.0, com_acc=-1.0, clearance=-20.0,
             terminated=-500.0, collide=-4500.0, alive=4.0)   # a stop costs 10, a collision 100; alive 4 keeps every second worth more than stopping (v34 netted +0.6/s and stopped on the torso end stop; v37's -100 stops drowned the signal)

    def __post_init__(self):
        self.scene.robot = R.articulation_cfg()


# ---------------------------------------------------------------- left/right mirror (symmetry augmentation, tasks.py)
# NOVA is mirror-symmetric about its x-z plane: left_arm_k = -right_arm_k, neck yaw / roll and lateral base motion
# flip sign. Each observation / action entry maps to (source index, sign); rsl-rl's Symmetry appends the mirrored
# copy of every mini-batch (Mittal et al., ICRA 2024), so neither arm can learn a habit the other does not share.
def _mirror_maps():
    P, S = [], []
    def blk(src, sg):                                    # append entries: source offsets (relative) + signs
        P.extend(src); S.extend(sg)
    def tgt(o):                                          # 27: (pos 3, 6D 6) x head, lw, rw
        H6, W6 = [1, -1, -1, 1, 1, -1], [1, 1, -1, -1, 1, 1]       # 6D = first two columns, row-major
        for k, src, r6 in ((0, 0, H6), (1, 2, W6), (2, 1, W6)):
            blk([o + 9 * src + i for i in range(9)], [1, -1, 1] + r6)
    def cur(o):                                          # 27: pos 9 (head, lw, rw) then 6D 18 (head, lw, rw)
        for src in (0, 2, 1): blk([o + 3 * src + i for i in range(3)], [1, -1, 1])
        for src, r6 in ((0, [1, -1, -1, 1, 1, -1]), (2, [1, 1, -1, -1, 1, 1]), (1, [1, 1, -1, -1, 1, 1])):
            blk([o + 9 + 6 * src + i for i in range(6)], r6)
    names = R.ACTUATED
    def joints(o, order):                                # one entry per joint name in `order`
        for n in order:
            if n.startswith(("left_arm", "right_arm")):
                m = n.replace("left", "@").replace("right", "left").replace("@", "right"); blk([o + order.index(m)], [-1])
            else:
                blk([o + order.index(n)], [-1 if n in ("neck_yaw", "neck_roll") else 1])
    pos = ["torso"] + R.NECK + R.ARM                     # action / filter order (env.i_pos)
    def acts(o):
        blk([o, o + 1, o + 2, o + 3], [1, -1, -1, 1]); joints(o + 4, pos)
    def vec(o, sg): blk([o + i for i in range(len(sg))], sg)
    def wrist_blk(o):                                    # 14: ORCA hinge L, R; solver residual L, R; palm error L, R
        blk([o + 1, o], [-1, -1])
        for b in (o + 2, o + 8):                         # rotation vectors: mirror about x-z = (-x, y, -z), sides swap
            blk([b + 3 + i for i in range(3)], [-1, 1, -1]); blk([b + i for i in range(3)], [-1, 1, -1])
    # actor: hist[0], hist[2], hist[5], cur, q, qd, base twist, act_obs
    for o in (0, 27, 54): tgt(o)
    cur(81); joints(108, names); joints(128, names); vec(148, [1, -1, -1]); acts(151); vec(173, [1])      # + mode bit
    blk([174, 176, 175], [1, 1, 1])                                                            # stream age: head, L, R
    wrist_blk(177)
    actor = (torch.tensor(P), torch.tensor(S, dtype=torch.float32)); P.clear(); S.clear()
    # critic: clean (targets, cur, q, qd, twist, act_obs) + privileged
    tgt(0); cur(27); joints(54, names); joints(74, names); vec(94, [1, -1, -1]); acts(97)
    o = 119
    vec(o, [1, -1, 1]); vec(o + 3, [-1, 1, -1]); vec(o + 6, [1, 1, 1])               # gravity, ang vel, lin z, zmp, clear
    for src in (0, 2, 1): blk([o + 9 + 3 * src + i for i in range(3)], [1, -1, 1])   # future head, lw, rw
    vec(o + 18, [1, -1]); vec(o + 20, [1, -1]); vec(o + 22, [1]); vec(o + 23, [-1, 1, -1])  # body yaw cs, sm xy, lean, neck
    for base in (o + 26, o + 32):                                                       # (le, re), (elbow L, elbow R)
        blk([base + 3 + i for i in range(3)], [1, -1, 1]); blk([base + i for i in range(3)], [1, -1, 1])
    vec(o + 38, [1, 1, 1, 1])                                                           # lift, kv, delay, mode bit
    blk([o + 42, o + 44, o + 43], [1, 1, 1])                                            # stream age
    wrist_blk(o + 45)
    critic = (torch.tensor(P), torch.tensor(S, dtype=torch.float32)); P.clear(); S.clear()
    acts(0); act = (torch.tensor(P), torch.tensor(S, dtype=torch.float32))
    assert len(actor[0]) == 191 and len(critic[0]) == 178 and len(act[0]) == 22, (len(actor[0]), len(critic[0]), len(act[0]))
    return {"policy": actor, "critic": critic, "act": act}


_MIRROR = None


def mirror_batch(env=None, obs=None, actions=None):
    """rsl-rl symmetry hook: returns (obs, actions) with the mirrored copy appended along the batch."""
    global _MIRROR
    _MIRROR = _MIRROR or _mirror_maps()
    def m(x, k):
        p, s = _MIRROR[k]; p, s = p.to(x.device), s.to(x.device)
        return torch.cat([x, x[..., p] * s], 0)
    o = None
    if obs is not None:
        o = obs.__class__({k: m(v, k) for k, v in obs.items()}, batch_size=[obs.batch_size[0] * 2], device=obs.device)
    a = m(actions, "act") if actions is not None else None
    return o, a


# ---------------------------------------------------------------- the environment
class NovaWbcEnv(WbcCore, DirectRLEnv):
    cfg: NovaWbcEnvCfg

    def __init__(self, cfg: NovaWbcEnvCfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        self.robot = self.scene["robot"]
        dev, N = self.device, self.num_envs
        self.jid = torch.tensor(self.robot.find_joints(R.ACTUATED, preserve_order=True)[0], device=dev)
        names = R.ACTUATED
        self.i_lift = [names.index(n) for n in R.LIFT]
        self.i_torso = names.index("torso")
        self.i_neck = [names.index(n) for n in R.NECK]
        self.i_arm = [names.index(n) for n in R.ARM]
        self.i_pos = [self.i_torso] + self.i_neck + self.i_arm        # 18 position-controlled joints, action order
        lim = self.robot.data.joint_pos_limits.torch[0, self.jid]
        self.q_lo, self.q_hi = lim[:, 0].clone(), lim[:, 1].clone()
        self.q_ready = self.robot.data.default_joint_pos.torch[0, self.jid].clone()

        # bodies and fixed offsets (link-local) of everything we measure
        fk = R.FK()
        self.body = {}
        def bid(link):
            if link not in self.body:
                ids, _ = self.robot.find_bodies(link)
                assert len(ids) == 1, f"body {link}: {ids} (bodies: {self.robot.body_names})"
                self.body[link] = ids[0]
            return self.body[link]
        T = R.tracked_offsets(fk)
        self.trk_body = torch.tensor([bid(T[k][0]) for k in ("head", "left_wrist", "right_wrist")], device=dev)
        self.trk_p = torch.tensor(np.stack([T[k][1] for k in ("head", "left_wrist", "right_wrist")]), dtype=torch.float32, device=dev)
        self.trk_R = torch.tensor(np.stack([T[k][2] for k in ("head", "left_wrist", "right_wrist")]), dtype=torch.float32, device=dev)
        self.elbow_body = torch.tensor([bid("l_elbow"), bid("r_elbow")], device=dev)
        self.shoulder_body = torch.tensor([bid("l_shoulder_3"), bid("r_shoulder_3")], device=dev)
        self.torso_body = bid("torso")
        # torso link axes at rest (REP): x fwd, y up, z right -> anatomical (x fwd, y left, z up)
        self.torso_R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], device=dev)
        # collision model (robot.collision_model): a distance field per part, spheres enclosing every arm/hand vertex
        parts, sph, pairs = R.collision_model()
        tt = lambda x: torch.tensor(np.array(x, dtype=np.float32), device=dev)
        self.sph_body = torch.tensor([bid(x[0]) for x in sph], device=dev)
        self.sph_c, self.sph_r = tt([x[1] for x in sph]), tt([x[2] for x in sph])
        # shoulder motors sit ~5 cm from the head and lift top by design: they get the termination, not the 4 cm band
        self.sph_sh = torch.tensor([str(x[4]) in ("shoulder_1", "shoulder_2") for x in sph], device=dev)
        self.sph_hand = torch.tensor(np.array([[str(x[4]) in R.HAND and str(x[3]) == s for s in "lr"] for x in sph]), device=dev)
        self.sdf = []                                     # (body, origin, extent, grid (1,1,z,y,x), sphere ids, hand of the part)
        for j, p in enumerate(parts):
            ids = torch.tensor(np.nonzero(pairs[:, j])[0], device=dev)
            if len(ids) == 0: continue
            g = tt(p["sdf"]).permute(2, 1, 0)[None, None].contiguous()
            fh = "lr".index(p["side"]) if p["side"] in ("l", "r") and p["name"] in R.HAND else -1
            self.sdf.append((bid(p["link"]), tt(p["origin"]), tt((np.array(p["sdf"].shape) - 1) * p["res"]), g, ids, fh))
        self.mass0 = self.robot.data.body_mass.torch[0].clone()
        self.masses = self.mass0[None].repeat(self.num_envs, 1)
        self.m_tot = self.masses.sum(-1)
        self.kp0 = self.robot.data.joint_stiffness.torch[0, self.jid].clone()
        self.kd0 = self.robot.data.joint_damping.torch[0, self.jid].clone()
        self.hand_bodies = [bid("l_hand_mount"), bid("r_hand_mount")]
        self.wheels = torch.tensor(WHEELS_B, device=dev)
        # gravity feed-forward (the robot-side gravity model): for each position joint, its child body, axis, and subtree
        names_pos = [R.ACTUATED[i] for i in self.i_pos]
        def chain(link):
            out = [link]
            while link in fk.J: link = fk.J[link]["parent"]; out.append(link)
            return out
        chains = [set(chain(b)) for b in self.robot.body_names]
        self.gc_body = torch.tensor([bid(fk.joint_child[n]) for n in names_pos], device=dev)
        self.gc_axis = tt([fk.J[fk.joint_child[n]]["axis"] / np.linalg.norm(fk.J[fk.joint_child[n]]["axis"]) for n in names_pos])
        self.gc_sub = tt([[float(fk.joint_child[n] in c) for c in chains] for n in names_pos])          # (J, B)
        self.gc_scale = torch.ones(self.num_envs, len(names_pos), device=dev)                         # calibration error
        self.ff_max = tt([R.FF_MAX_NM[R.joint_group(n)] for n in names_pos])                           # NOVA's per-joint clip
        # the lift is a non-back-drivable screw: it carries everything above each stage (exact, no calibration error)
        self.lf_body = torch.tensor([bid(fk.joint_child[n]) for n in R.LIFT], device=dev)
        self.lf_axis = tt([fk.J[fk.joint_child[n]]["axis"] for n in R.LIFT])
        self.lf_sub = tt([[float(fk.joint_child[n] in c) for c in chains] for n in R.LIFT])
        grp = lambda n: R.gain_group(R.joint_group(n))
        sp = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)                 # = the target clamp in _pre_physics_step
        lo = (self.q_lo[self.i_pos] + sp - self.q_ready[self.i_pos]) / self.cfg.action_scale
        hi = (self.q_hi[self.i_pos] - sp - self.q_ready[self.i_pos]) / self.cfg.action_scale
        self.a_lo = torch.cat([torch.full((4,), -2.5, device=dev), lo.clamp(min=-5.0)])
        self.a_hi = torch.cat([torch.full((4,), 2.5, device=dev), hi.clamp(max=5.0)])
        self.slew = tt([min(R.HW_GAINS[grp(n)][2], R.HEADROOM * R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0]) for n in names_pos])
        spec = [R.JOINT_SPEC[R.joint_group(n)] for n in names_pos]
        self.v_lim = tt([R.SAFETY * x[0] for x in spec]); self.t_pk = tt([R.SAFETY * x[1] for x in spec])
        self.t_rated = tt([R.SAFETY * x[2] for x in spec])
        self.ovl_hard = torch.tensor([R.joint_group(n) not in R.OVERLOAD_SOFT for n in names_pos], device=dev)
        self.mods = tt(WHEELS_B)                                                                       # (3,2)
        A = torch.zeros(6, 3, device=dev)
        for i, (x, y) in enumerate(WHEELS_B): A[2 * i] = torch.tensor([1.0, 0.0, -y]); A[2 * i + 1] = torch.tensor([0.0, 1.0, x])
        self.mods_pinv = torch.linalg.pinv(A)                                                          # module velocities -> twist
        # yaw of REP base frame = root yaw + pi (URDF base faces -x)
        self.wc = [R.wrist_chain(fk, s) for s in ("left", "right")]
        self.i_wr = [[self.i_pos.index(names.index(f"{s}_arm_{j}")) for j in (5, 6, 7)] for s in ("left", "right")]   # cols of q_f
        self.i_wr_all = self.i_wr[0] + self.i_wr[1]
        self.jid_hw = torch.tensor(self.robot.find_joints(R.HAND_WRIST, preserve_order=True)[0], device=dev)  # ORCA hinges
        lim_hw = self.robot.data.joint_pos_limits.torch[0, self.jid_hw]
        self.hw_lo, self.hw_hi = lim_hw[:, 0] + 0.08, lim_hw[:, 1] - 0.08
        self.hw_vlim = R.SAFETY * R.JOINT_SPEC["hand_wrist"][0]
        hz = torch.full((len(self.i_pos),), self.cfg.action_lpf_hz, device=dev); hz[self.i_wr_all] = self.cfg.wrist_lpf_hz
        self.beta = 1.0 - torch.exp(-2 * math.pi * hz * self.cfg.sim.dt * self.cfg.decimation)
        self.load_motions()
        self.alloc()

    # ------------------------------------------------------------ motion library
    def load_motions(self):
        z = np.load(self.cfg.motions)
        t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=self.device)
        self.m = dict(head_p=t("head_p"), lw_p=t("lw_p"), rw_p=t("rw_p"), le=t("le"), re=t("re"), ls=t("ls"), rs=t("rs"), sm=t("sm"),
                      head_R=quat_wxyz_to_mat(t("head_q")), lw_R=quat_wxyz_to_mat(t("lw_q")), rw_R=quat_wxyz_to_mat(t("rw_q")),
                      torso_yaw=t("torso_yaw"), lean=t("lean"), neck=t("neck"))
        self.c_start, self.c_len = t("starts").long(), t("lens").long()
        # shape-preserving retarget: each wrist / elbow about its own shoulder, scaled by robot arm reach / the
        # operator's (shoulder -> elbow -> wrist, per-clip median), the shoulders by the shoulder-width ratio
        fk = R.FK(); P = lambda j: fk.pose(fk.joint_child[j])[1]
        reach = np.linalg.norm(P("left_arm_4") - P("left_arm_2")) + np.linalg.norm(P("left_wrist") - P("left_arm_4"))
        width = np.linalg.norm(P("left_arm_2") - P("right_arm_2"))
        n = lambda a, b: np.linalg.norm(z[a] - z[b], axis=1)
        h_reach = 0.5 * (n("le", "ls") + n("lw_p", "le") + n("re", "rs") + n("rw_p", "re")); h_width = n("ls", "rs")
        med = lambda x: np.concatenate([np.full(l, np.median(x[s:s + l])) for s, l in zip(z["starts"], z["lens"])])
        self.m["arm_k"] = torch.tensor(ML.ROBOT_REACH / med(h_reach), dtype=torch.float32, device=self.device)[:, None]
        print(f"[wbc] retarget: robot reach {reach:.3f} m, shoulders {width:.3f} m; arm scale "
              f"{self.m['arm_k'].median().item():.2f} (median clip)")
        ego = np.array([m.startswith("egodex") for m in z["meta"]])
        self.c_ego = torch.tensor(ego, device=self.device)
        w = z["lens"].astype(np.float64)
        p_e = np.where(ego, w, 0); p_l = np.where(~ego, w, 0)
        p = (self.cfg.egodex_share * p_e / max(p_e.sum(), 1) + (1 - self.cfg.egodex_share) * p_l / max(p_l.sum(), 1)) if ego.any() and (~ego).any() else w / w.sum()
        self.c_p = torch.tensor(p / p.sum(), dtype=torch.float32, device=self.device)
        pu = p_e if ego.any() else w
        self.c_p_up = torch.tensor(pu / pu.sum(), dtype=torch.float32, device=self.device)   # upper-body mode: EgoDex only
        self.wrist_dz = float(z["wrist_dz"])
        print(f"[wbc] motions: {len(w)} clips ({ego.sum()} EgoDex), {w.sum() / 50 / 3600:.2f} h")


    # ------------------------------------------------------------ buffers
    def alloc(self):
        N, dev = self.num_envs, self.device
        z = lambda *s: torch.zeros(N, *s, device=dev)
        self.phase, self.rate, self.c_id = z(), torch.ones(N, device=dev), torch.zeros(N, dtype=torch.long, device=dev)
        # anchor: robot_world = Rz(anc_yaw) (human - anc_h) + anc_r
        self.anc_yaw, self.anc_h, self.anc_r = z(), z(3), z(3)
        # synthetic walk / turn and neck-only augmentation states
        self.aug_on, self.aug_yaw, self.aug_xy = z(), z(), z(2)
        self.aug_w, self.aug_v = z(), z(2)
        self.neck_on, self.neck_off, self.neck_tgt = z(), z(2), z(2)
        self.tgt = None                                               # current target dict (robot world)
        self.hist = z(10, 27)                                         # last 10 target observations (base frame)
        # the Vision Pro stream as received (robot world; head, left wrist, right wrist) and its fault states
        eye = lambda *s: torch.eye(3, device=dev).repeat(N, *s, 1, 1)
        self.rx_p, self.rx_R, self.rx_v = z(3, 3), eye(3), z(2, 3)
        self.buf_p, self.buf_R = z(8, 3, 3), eye(8, 3)                 # last 8 clean frames (latency)
        self.age = z(3)                                               # s since each part got a fresh tracked frame
        self.lat = torch.zeros(N, dtype=torch.long, device=dev); self.loss, self.npos, self.nrot = z(), z(), z()
        self.trem_w, self.trem_a = z(3), z(3, 2); self.trem_z = torch.zeros(N, 3, 6, dtype=torch.complex64, device=dev)
        self.slow_k = torch.ones(N, dtype=torch.long, device=dev); self.slow_t, self.out_t = z(), z()
        self.occ_t, self.occ_pred = z(2), z(2)
        self.spk_n = torch.zeros(N, 2, dtype=torch.long, device=dev); self.spk_p, self.spk_R = z(2, 3), eye(2)
        self.bias_p, self.bias_R = z(3, 3), eye(3)
        self.hold_p, self.hold_R = z(3, 3), eye(3)                    # reward reference while a part is lost
        self.lost_prev = torch.zeros(N, 3, dtype=torch.bool, device=dev)
        self.odo_xy, self.odo_yaw = z(2), z()                         # base pose estimate error (odometry)
        self.gov = torch.tensor([R.BASE_ACCEL, R.BASE_VMAX, R.BASE_WMAX, R.BASE_ACCEL * R.BASE_DECEL], device=dev).repeat(N, 1)   # tip governor: accel, speed, yaw rate, decel
        self.coast = torch.ones(N, dtype=torch.bool, device=dev)     # drives acked COAST: no locked-wheel speed cap
        self.drive_k = torch.ones(N, 3, device=dev)                   # open-loop duty drive: per-module speed error
        self.vb_obs = z(3)                                            # measured base twist: NOVA's drive feedback is 20 Hz
        self.rec_t = torch.full((N,), 10.0, device=dev)               # s since the stream last recovered
        self.n_step = 0; self.n_gov = 0
        self.actions = z(22); self.prev_actions = z(22); self.prev2_actions = z(22); self.raw_act = z(22)
        self.q_f = z(len(self.i_pos)); self.q_f_prev = z(len(self.i_pos)); self.sub_k = 0     # filtered position targets
        self.q_f1 = z(len(self.i_pos)); self.qf_acc = z(len(self.i_pos))                    # 1st filter stage, target accel
        self.sp, self.sp_v, self.v_ff = z(len(self.i_pos)), z(len(self.i_pos)), z(len(self.i_pos))   # nova_hw shaper state
        self.last_term = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)     # tilt, collide, drift, limit
        self.hw_wn = torch.full((len(self.i_pos),), self.cfg.hw_smooth_wn, device=self.device)      # shaper wn per q_f column
        self.hw_wn[self.i_wr_all] = self.cfg.hw_wrist_wn
        self.i_servo = [self.i_pos.index(R.ACTUATED.index(n)) for n in ("neck_pitch", "neck_roll")]       # columns of q_f
        self.f_lift = z(2); self.tau_g = z(len(self.i_pos)); self.tau = z(len(self.i_pos)); self.tau2 = z(len(self.i_pos))
        self.kp_env = z(len(self.i_pos)); self.kd_env = z(len(self.i_pos))
        self.tw_f = z(3); self.v_cmd_b = z(3); self.steer = z(3); self.steer_cmd = z(3); self.steer_prev = z(3); self.steer_k = torch.ones(N, device=dev)
        self.lim_kind = torch.zeros(N, 4, dtype=torch.bool, device=dev)      # overspeed, peak torque, overload, position
        self.hand_v_prev = z(2, 3); self.hand_acc = z(2, 3); self.tgt_v = z(2, 3); self.palm_w_prev = z(2, 3); self.palm_acc = z(2, 3)
        self.act_buf = z(self.cfg.action_delay[1] + 1, 22); self.delay = torch.zeros(N, dtype=torch.long, device=dev)
        self.v_ref_b = z(3)                                           # accel-limited base twist (vx, vy, wz), base frame
        self.v_ref_prev = z(3)
        self.lift_tgt = z(); self.q_tgt = z(len(R.ACTUATED)); self.v_tgt = z(len(R.ACTUATED))
        self.upper, self.sm0, self.sm_lp = z(), z(2), z(2)                              # mode bit; operator torso xy at the anchor (upper mode)
        self.kv, self.kw = torch.full((N,), 10.0, device=dev), torch.full((N,), 8.0, device=dev)
        self.com_v_prev, self.com_acc = z(3), z(3)
        self.zmp_margin, self.clear, self.clear_mv, self.clear_sh = z(), z(), z(), z()
        self.qd_prev = z(len(R.ACTUATED))
        self.ep_err = z(4)
        self.push_t, self.push_f = z(), z(3)
        self.zmp_out_t = z()
        self.blend_on, self.blend_t, self.blend_T = z(), z(), torch.ones(N, device=dev)
        self.blend_p, self.blend_R = z(3, 3), torch.eye(3, device=dev).repeat(N, 3, 1, 1)
        self.need_glide = torch.zeros(N, dtype=torch.bool, device=dev)
        self.clear_hand = torch.ones(self.num_envs, 2, device=dev); self.w_safe = z(2, 4)
        self.wik = z(2, 4); self.ik_buf = z(self.cfg.action_delay[1] + 1, 8)        # wrist solution (warm start), its delay line
        self.wres = z(2, 3)                                                           # what the solver could not reach (rotvec)
        self.hw_f1, self.hw_f, self.hw_sp, self.hw_spv = z(2), z(2), z(2), z(2)       # ORCA hinge: filter, servo profile
        self.oe_R = torch.eye(3, device=dev).repeat(N, 2, 1, 1); self.oe_s = z(2)   # one-euro: filtered palm, speed

    # ------------------------------------------------------------ robot geometry


    # ------------------------------------------------------------ targets

    # ------------------------------------------------------------ step
    def _pre_physics_step(self, actions):
        self.raw_act = actions                                         # before any clamp (r["act_sat"])
        a = actions.clamp(-5.0, 5.0)
        self.prev2_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        self.act_buf = torch.cat([a[:, None], self.act_buf[:, :-1]], 1)
        a = self.act_buf[torch.arange(self.num_envs, device=self.device), self.delay]
        self.actions = a
        dt = self.step_dt
        # base: twist command -> low-pass -> accel limit (0.75 m/s^2 floor) -> swerve modules -> realised twist
        cmd = torch.tanh(a[:, 0:3]) * torch.tensor([R.BASE_VMAX, R.BASE_VMAX, R.BASE_WMAX], device=self.device)
        cmd = cmd * (1 - self.upper[:, None])                          # upper-body mode: base locked
        # robot-side tip governor (stability.py limits, 10 Hz): speed, turn rate and accel only ever tighten
        sp = cmd[:, :2].norm(dim=-1, keepdim=True)
        cmd = torch.cat([cmd[:, :2] * (self.gov[:, 1:2] / sp.clamp_min(1e-6)).clamp(max=1.0),
                         torch.max(torch.min(cmd[:, 2:3], self.gov[:, 2:3]), -self.gov[:, 2:3])], -1)
        self.tw_f = self.tw_f + (1 - math.exp(-dt / self.cfg.twist_tau)) * (cmd - self.tw_f)
        vxy = R.base_ramp(self.v_cmd_b[:, :2], self.tw_f[:, :2], self.gov[:, 0:1], self.gov[:, 3:4], dt, torch)
        dw = (self.tw_f[:, 2] - self.v_cmd_b[:, 2]).clamp(-R.BASE_ALPHA * dt, R.BASE_ALPHA * dt)
        self.v_cmd_b = torch.cat([vxy, (self.v_cmd_b[:, 2] + dw)[:, None]], -1)
        self.v_ref_prev = self.v_ref_b.clone()
        self.steer_prev = self.steer.clone()
        self.v_ref_b = self.swerve(self.v_cmd_b, dt)
        # lift: rate command integrated to a target (both stages equal)
        rate = torch.tanh(a[:, 3]) * R.HEADROOM * R.LIFT_VLIM * (1 - self.upper)      # upper-body mode: lift locked
        self.lift_tgt = (self.lift_tgt + 0.5 * rate * dt).clamp(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)
        # position joints: command -> low-pass -> ramped across the substeps in _apply_action
        q = self.q_ready[self.i_pos][None] + self.cfg.action_scale * a[:, 4:]
        span = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)        # > nova_hw end_stop_margin 0.05 rad: fast moves overshoot
        if self.cfg.wrist_ik:                                                          # solved wrist + the policy's residual
            self.ik_buf = torch.cat([self.solve_wrist()[:, None], self.ik_buf[:, :-1]], 1)
            ik = self.ik_buf[torch.arange(self.num_envs, device=self.device), self.delay]
            ik = R.hand_guard(ik.view(-1, 2, 4), self.w_safe, self.clear_hand, self.cfg.hand_guard, torch).reshape(-1, 8)
            q[:, self.i_wr_all] = ik[:, [0, 1, 2, 4, 5, 6]] + self.cfg.wrist_res * a[:, 4:][:, self.i_wr_all]
            g = torch.max(torch.min(ik[:, [3, 7]], self.hw_hi), self.hw_lo)          # ORCA hinge goal: solver only
            b = self.beta[self.i_wr_all[0]]
            self.hw_f1 = self.hw_f1 + b * (g - self.hw_f1); self.hw_f = self.hw_f + b * (self.hw_f1 - self.hw_f)
        q = torch.max(torch.min(q, self.q_hi[self.i_pos] - span), self.q_lo[self.i_pos] + span)
        beta = self.beta
        qf_pp = self.q_f_prev
        self.q_f_prev = self.q_f.clone()
        self.q_f1 = self.q_f1 + beta * (q - self.q_f1)
        step = beta * (self.q_f1 - self.q_f)
        self.q_f = self.q_f + torch.max(torch.min(step, self.slew * dt), -self.slew * dt)     # NOVA's setpoint slew
        self.qf_acc = (self.q_f - 2 * self.q_f_prev + qf_pp) / (dt * dt)                     # what the motors are asked to do
        self.tau_g = self.gravity_ff(); self.f_lift = self.lift_ff()
        # nova_hw shaper (drivers/shaper.py): the 50 Hz goal -> critically damped follower, wn 40 rad/s (wrist 80) in 5 ms
        # sub-steps; the MIT frame holds (setpoint, v_ff = its velocity) until the next tick
        # neck pitch/roll are Feetech servos: no shaper, the firmware's trapezoid (speed / acc) to the goal, no v_ff
        wn, h, sv = self.hw_wn, 0.005, self.i_servo
        for _ in range(int(round(dt / h))):
            spv = self.sp_v + h * (wn * wn * (self.q_f - self.sp) - 2.0 * wn * self.sp_v)
            e = self.q_f[:, sv] - self.sp[:, sv]; v0 = self.sp_v[:, sv]
            vs = torch.sign(e) * torch.sqrt(2 * R.NECK_SERVO["acc"] * e.abs()).clamp(max=R.NECK_SERVO["speed"])   # brake in time
            vs = v0 + (vs - v0).clamp(-R.NECK_SERVO["acc"] * h, R.NECK_SERVO["acc"] * h)
            spv[:, sv] = vs; self.sp_v = spv
            self.sp = self.sp + h * self.sp_v
            e = self.hw_f - self.hw_sp; v0 = self.hw_spv                       # ORCA hinge: HLS3930 firmware profile
            vs = torch.sign(e) * torch.sqrt(2 * R.HAND_SERVO["acc"] * e.abs()).clamp(max=R.HAND_SERVO["speed"])
            self.hw_spv = v0 + (vs - v0).clamp(-R.HAND_SERVO["acc"] * h, R.HAND_SERVO["acc"] * h)
            self.hw_sp = self.hw_sp + h * self.hw_spv
        self.v_ff = self.sp_v.clone(); self.v_ff[:, sv] = 0.0
        self.sub_k = 0
        self.q_tgt[:, self.i_lift[0]] = self.lift_tgt
        self.q_tgt[:, self.i_lift[1]] = self.lift_tgt

    def solve_wrist(self):
        """robot.wrist_ik on the one-euro filtered received palms, for the forearm as measured (encoders + odometry yaw):
        exactly what deploy.Adapter runs."""
        q = self.robot.data.joint_pos.torch[:, self.jid]; qd = {n: q[:, i] for i, n in enumerate(R.ACTUATED)}
        _, yaw, _ = self.base_pose(); Rb = rot_z(yaw + self.odo_yaw) @ rot_z(torch.full_like(yaw, math.pi))
        span = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)
        lo, hi = self.q_lo[self.i_pos] + span, self.q_hi[self.i_pos] - span
        out = []
        for j, wc in enumerate(self.wc):
            A = Rb @ R.chain_rot(wc["chain"], qd, torch)
            ix = self.i_wr[j]
            L = torch.cat([lo[ix], self.hw_lo[j:j + 1]]); H = torch.cat([hi[ix], self.hw_hi[j:j + 1]])
            self.wik[:, j] = R.wrist_ik(A, self.oe_R[:, j], wc, self.wik[:, j], L, H, torch, iters=8)
            self.wres[:, j] = R.wrist_err(A, self.oe_R[:, j], wc, self.wik[:, j], torch)
            out.append(self.wik[:, j])
        return torch.cat(out, -1)

    def euro_step(self, Rx, reset=None):
        """One-euro filter (Casiez 2012) on the two received palm orientations (N,2,3,3)."""
        E, dt = self.cfg.euro, self.step_dt
        al = lambda hz: 1.0 / (1.0 + 1.0 / (2 * math.pi * hz * dt))
        D = self.oe_R.transpose(-1, -2) @ Rx
        v = torch.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]], -1) / 2
        sn = v.norm(dim=-1); th = torch.atan2(sn, (D[..., 0, 0] + D[..., 1, 1] + D[..., 2, 2] - 1) / 2)
        dv = v * (th / (sn + 1e-9))[..., None]
        self.oe_s = self.oe_s + al(E["d_hz"]) * (th / dt - self.oe_s)
        a = al(E["min_hz"] + E["beta"] * self.oe_s)
        self.oe_R = self.oe_R @ rotvec(dv * a[..., None])
        if reset is not None:
            self.oe_R[reset] = Rx[reset]; self.oe_s[reset] = 0


    def _apply_action(self):
        self.sub_k += 1
        # NOVA's MIT frame, held between 50 Hz ticks: tau = kp (sp - q) + kd (v_ff - qd) + t_ff
        self.q_tgt[:, self.i_pos] = self.sp
        self.robot.set_joint_position_target_index(target=self.q_tgt, joint_ids=self.jid)
        self.v_tgt[:, self.i_pos] = self.v_ff
        self.robot.set_joint_velocity_target_index(target=self.v_tgt, joint_ids=self.jid)
        self.robot.set_joint_effort_target_index(target=torch.cat([self.f_lift, self.tau_g], -1),
                                                 joint_ids=torch.cat([self.jid[self.i_lift], self.jid[self.i_pos]]))
        self.robot.set_joint_position_target_index(target=self.hw_sp, joint_ids=self.jid_hw)
        # swerve drive: force at the floor under the chassis tracking the base twist
        _, yaw, _ = self.base_pose()
        c, s = torch.cos(yaw), torch.sin(yaw)
        vb = self.v_ref_b
        v_ref_w = torch.stack([c * vb[:, 0] - s * vb[:, 1], s * vb[:, 0] + c * vb[:, 1]], -1)
        v = self.robot.data.root_com_lin_vel_w.torch[:, :2]
        w = self.robot.data.root_com_ang_vel_w.torch[:, 2]
        F = self.m_tot[:, None] * self.kv[:, None] * (v_ref_w - v)
        Fn = F.norm(dim=-1, keepdim=True)
        F = F * torch.clamp(0.8 * G * self.m_tot[:, None] / Fn.clamp_min(1e-6), max=1.0)       # traction limit
        F3 = torch.cat([F, torch.zeros(self.num_envs, 1, device=self.device)], -1)
        com = self.robot.data.body_com_pos_w.torch[:, 0]
        floor = torch.cat([com[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], -1)
        tau = torch.cross(floor - com, F3, dim=-1)
        tau[:, 2] += 4.0 * self.kw * (vb[:, 2] - w)
        # one call for both bodies: set_forces_and_torques_index resets every body of the given envs
        push = self.push_f * (self.push_t > 0).float()[:, None] * float(self.cfg.dr_physics)
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.stack([F3, push], 1), torques=torch.stack([tau, torch.zeros_like(push)], 1),
            body_ids=[0, self.torso_body], is_global=True)



    # ------------------------------------------------------------ per-step bookkeeping



    # ------------------------------------------------------------ observations




    def wrist_obs(self, yaw, cR, noisy):
        """14: ORCA hinge angles; what the wrist solver could not reach; palm error vs the received (actor) or clean
        (critic) target — rotation vectors in the base yaw frame, rad. The policy places the forearm so both stay small."""
        Ryt = rot_z(yaw + (self.odo_yaw if noisy else 0)).transpose(-1, -2)[:, None]
        hq = self.robot.data.joint_pos.torch[:, self.jid_hw]
        Rt = self.rx_R[:, 1:] if noisy else torch.stack([self.tgt["lw_R"], self.tgt["rw_R"]], 1)
        pe = rotlog(Rt @ cR[:, 1:].transpose(-1, -2))
        res = (Ryt @ self.wres[..., None])[..., 0]; pe = (Ryt @ pe[..., None])[..., 0]
        return torch.cat([hq, res.reshape(-1, 6), pe.reshape(-1, 6)], -1)

    def _get_observations(self):
        pos, yaw, Rb = self.base_pose()
        tob = self.target_obs(pos, yaw, noisy=True)
        self.hist = torch.cat([tob[:, None], self.hist[:, :-1]], 1)
        cp, cR = self.tracked_now()
        Ryt = rot_z(yaw).transpose(-1, -2)[:, None]
        cur = torch.cat([((Ryt @ (cp - pos[:, None])[..., None])[..., 0]).reshape(self.num_envs, 9),
                         six_d(Ryt @ cR).reshape(self.num_envs, 18)], -1)
        q = self.robot.data.joint_pos.torch[:, self.jid]; qd = self.robot.data.joint_vel.torch[:, self.jid]
        lin = self.robot.data.root_link_lin_vel_w.torch; ang = self.robot.data.root_link_ang_vel_w.torch
        c, s = torch.cos(yaw), torch.sin(yaw)
        vb = torch.stack([c * lin[:, 0] + s * lin[:, 1], -s * lin[:, 0] + c * lin[:, 1], ang[:, 2]], -1)
        n = self.cfg.noise
        upd = (torch.rand(self.num_envs, device=self.device) < 0.4) | (self.episode_length_buf <= 1)
        self.vb_obs = torch.where(upd[:, None], vb + torch.randn_like(vb) * n["base"], self.vb_obs)
        actor = torch.cat([self.hist[:, 0], self.hist[:, 2], self.hist[:, 5],
                           cur + torch.randn_like(cur) * 0.005,
                           q - self.q_ready + torch.randn_like(q) * n["q"],
                           0.1 * (qd + torch.randn_like(qd) * n["qd"]),
                           self.vb_obs, self.act_obs(), self.upper[:, None], self.age.clamp(max=1.0), self.wrist_obs(yaw, cR, True)], -1)
        # privileged
        t = self.tgt
        clean = torch.cat([self.target_obs(pos, yaw, noisy=False), cur, q - self.q_ready, 0.1 * qd, vb, self.act_obs()], -1)
        fut = self.compute_targets(self.phase + 25 * self.rate, step_aug=False)
        Rb = rot_z(yaw).transpose(-1, -2)
        loc = lambda p: (Rb @ (p - pos)[..., None])[..., 0]
        by = wrap(t["body_yaw"] - yaw)
        el_r = self.robot.data.body_link_pos_w.torch[:, self.elbow_body]
        priv = torch.cat([self.robot.data.projected_gravity_b.torch, self.robot.data.root_link_ang_vel_b.torch, lin[:, 2:3],
                          self.zmp_margin[:, None] * 10, self.clear[:, None] * 10,
                          loc(fut["head_p"]), loc(fut["lw_p"]), loc(fut["rw_p"]),
                          torch.stack([torch.cos(by), torch.sin(by)], -1), loc(t["sm"])[:, :2], t["lean"][:, None], t["neck"],
                          loc(t["le"]), loc(t["re"]), loc(el_r[:, 0]), loc(el_r[:, 1]),
                          self.lift_tgt[:, None] * 5, self.kv[:, None] / 10, self.delay[:, None].float(), self.upper[:, None],
                          self.age.clamp(max=1.0), self.wrist_obs(yaw, cR, False)], -1)
        return {"policy": actor, "critic": torch.cat([clean, priv], -1)}

    # ------------------------------------------------------------ rewards
    def _get_rewards(self):
        w, t = self.cfg.w, self.tgt
        pos, yaw, Rb = self.base_pose()
        cp, cR = self.tracked_now()
        e_h = (cp[:, 0] - t["head_p"]).norm(dim=-1)
        e_l = (cp[:, 1] - t["lw_p"]).norm(dim=-1); e_r = (cp[:, 2] - t["rw_p"]).norm(dim=-1)
        a_h = rot_angle(cR[:, 0], t["head_R"]); a_l = rot_angle(cR[:, 1], t["lw_R"]); a_r = rot_angle(cR[:, 2], t["rw_R"])
        K = lambda e, sg: torch.exp(-(e / sg) ** 2)
        r = {}
        r["head_p"] = K(e_h, 0.12); r["head_r"] = K(a_h, 0.35)
        # v31: + 5 cm kernel, fine 3 -> 2 cm, linear wrist term x4: the v28 retarget left the hands 13-15 cm off and the old
        # kernels gave almost no pull at that distance (K(0.13, 0.03) = 0)
        r["wrist_p"] = 0.5 * (K(e_l, 0.10) + K(e_r, 0.10)); r["wrist_p_mid"] = 0.5 * (K(e_l, 0.05) + K(e_r, 0.05))
        r["wrist_p_fine"] = 0.5 * (K(e_l, 0.02) + K(e_r, 0.02))
        rot_ok = self.c_ego[self.c_id].float()          # LAFAN1 has no finger markers: its palm frames are a guess, not scored
        # three scales: 0.2 rad (~11 deg) for precision, 0.5, and 1.0 rad so a palm 50+ deg off still feels a pull
        # (v15: 0.4 alone -> 57 deg). v27: weight 3 -> 8 and the 0.2 term: v25/v26 sat at ~30 deg (the most important term)
        # v28: + a 0.087 rad (5 deg) kernel: sigma 0.4 alone gives 0.95 at 5 deg, no gradient left (CHIP plateau)
        r["wrist_r"] = (K(a_l, 0.087) + K(a_r, 0.087) + K(a_l, 0.2) + K(a_r, 0.2) + K(a_l, 0.5) + K(a_r, 0.5)
                        + K(a_l, 1.0) + K(a_r, 1.0)) / 8 * rot_ok
        hv_now = self.robot.data.body_com_lin_vel_w.torch[:, self.trk_body[1:]]
        r["hand_vel"] = K((hv_now - self.tgt_v).norm(dim=-1), 0.3).mean(-1)            # move like the operator's hands
        r["track_lin"] = e_h + 2.0 * (e_l + e_r) + 0.1 * a_h + 0.6 * (a_l + a_r) * rot_ok
        # what the wrist solver cannot reach is the policy's doing: the forearm is placed so the palm is out of reach
        r["wrist_unreach"] = self.wres.norm(dim=-1).sum(-1) * rot_ok
        # human-likeness (privileged labels)
        r["body_yaw"] = K(wrap(t["body_yaw"] - yaw), 0.35)
        base_tgt = t["sm"][:, :2] - torch.stack([torch.cos(t["body_yaw"]), torch.sin(t["body_yaw"])], -1) * 0.096
        r["body_xy"] = K((pos[:, :2] - base_tgt).norm(dim=-1), 0.2)
        up = self.upper > 0                                               # base locked: nothing to score
        r["body_yaw"] = torch.where(up, torch.ones_like(e_h), r["body_yaw"]); r["body_xy"] = torch.where(up, torch.ones_like(e_h), r["body_xy"])
        q = self.robot.data.joint_pos.torch[:, self.jid]
        lean_t = (-t["lean"]).clamp(-1.0, 0.15)                           # torso joint: negative = forward (stop at 0.26)
        r["lean"] = K(q[:, self.i_torso] - lean_t, 0.15)
        r["lean_excess"] = (lean_t - q[:, self.i_torso] - 0.2).clamp_min(0).square()   # forward of the operator by > 11 deg
        tR = quat_xyzw_to_mat(self.robot.data.body_link_quat_w.torch[:, self.torso_body]) @ self.torso_R
        neck_lbl = rot_z(t["neck"][:, 0]) @ rot_y(t["neck"][:, 1]) @ rot_x(t["neck"][:, 2])
        r["neck"] = K(rot_angle(tR.transpose(-1, -2) @ cR[:, 0], neck_lbl), 0.35)
        bp = self.robot.data.body_link_pos_w.torch
        ua_l = torch.nn.functional.normalize(bp[:, self.elbow_body[0]] - bp[:, self.shoulder_body[0]], dim=-1)
        ua_r = torch.nn.functional.normalize(bp[:, self.elbow_body[1]] - bp[:, self.shoulder_body[1]], dim=-1)
        ang = lambda u, v: torch.acos((u * v).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
        r["elbow"] = 0.5 * (K(ang(ua_l, t["l_arm"]), 0.35) + K(ang(ua_r, t["r_arm"]), 0.35))
        # effort / smoothness: base and lift are slow, use them only when needed
        r["base_use"] = self.v_ref_b[:, :2].square().sum(-1) + 0.3 * self.v_ref_b[:, 2].square()
        r["base_acc"] = ((self.v_ref_b - self.v_ref_prev) / self.step_dt).square().sum(-1)
        r["steer"] = (wrap(self.steer - self.steer_prev) / self.step_dt).square().sum(-1).clamp(max=200.0)   # steering motors (rad/s)^2
        lr = torch.tanh(self.actions[:, 3]).abs()
        r["lift_use"] = lr + lr.square()                                  # L1 + L2: small bobs are not worth the lift
        r["act_sat"] = ((self.raw_act - self.a_hi).clamp_min(0) + (self.a_lo - self.raw_act).clamp_min(0)).sum(-1)
        r["action_rate"] = (self.actions - self.prev_actions).square().sum(-1)
        r["action_acc"] = (self.actions - 2 * self.prev_actions + self.prev2_actions).square().sum(-1)
        r["qf_acc"] = self.qf_acc.square().sum(-1).clamp(max=2e5)                     # chatter in the motor targets (rad/s^2)^2
        hv = self.robot.data.body_com_lin_vel_w.torch[:, self.trk_body[1:]]
        self.hand_acc = (hv - self.hand_v_prev) / self.step_dt; self.hand_v_prev = hv.clone()
        r["hand_acc"] = self.hand_acc.square().sum((-1, -2)).clamp(max=5000.0)      # shaking hands (m/s^2)^2
        pw = self.robot.data.body_link_ang_vel_w.torch[:, self.trk_body[1:]]
        self.palm_acc = (pw - self.palm_w_prev) / self.step_dt; self.palm_w_prev = pw.clone()
        r["palm_acc"] = self.palm_acc.square().sum((-1, -2)).clamp(max=2e4)         # wrist jitter (rad/s^2)^2 (SONIC: anti-shake)
        qd = self.robot.data.joint_vel.torch[:, self.jid]
        r["joint_vel"] = qd.square().sum(-1)
        qdd = (qd - self.qd_prev) / self.step_dt; self.qd_prev = qd.clone()
        r["joint_acc"] = qdd.square().sum(-1)
        r["arm_acc"] = qdd[:, self.i_arm].square().sum(-1)
        span = self.q_hi - self.q_lo
        r["joint_limit"] = ((self.q_lo + 0.08 * span - q).clamp_min(0) + (q - self.q_hi + 0.08 * span).clamp_min(0)).sum(-1)
        r["past_stop"] = self.lim_joint[:, 2].float().sum(-1) * (self.episode_length_buf > 10)      # joints past their end stop: 5 per step each, no stop (v39)
        twist_j = [self.i_arm[k] for k in (2, 4, 6, 9, 11, 13)]           # upper-arm, forearm and wrist rolls
        ip = self.i_pos
        bar = lambda x: ((x - 0.85) / 0.15).clamp(0, 1.5).square().sum(-1)      # from 85 % of the (already 80 %) limit
        rms = (self.tau2.sqrt() / self.t_rated) * self.ovl_hard                  # neck pitch/roll exceed at rest: reported only
        r["limits"] = (bar(qd[:, ip].abs() / self.v_lim) + bar(self.tau.abs() / self.t_pk) + bar(rms)
                       + bar(qd[:, self.i_lift].abs() / (0.5 * R.LIFT_VLIM))
                       + bar(self.robot.data.joint_vel.torch[:, self.jid_hw].abs() / self.hw_vlim))
        # energy: mechanical power |tau qd| (W) and copper loss ~ (tau / rated)^2, neck pitch/roll excluded (gravity-bound)
        self.power = (self.tau * qd[:, ip]).abs().sum(-1)
        r["energy"] = self.power.clamp(max=200.0)
        r["torque"] = ((self.tau / self.t_rated).square() * self.ovl_hard).sum(-1)
        r["twist"] = (q[:, twist_j] - self.q_ready[twist_j]).square().sum(-1)
        # stability
        g = self.robot.data.projected_gravity_b.torch
        r["tilt"] = g[:, :2].square().sum(-1)
        r["tilt_rate"] = self.robot.data.root_link_ang_vel_b.torch[:, :2].square().sum(-1)
        r["com_acc"] = self.com_acc[:, :2].square().sum(-1).clamp(max=25.0)          # 1 m/s^2 moves the ZMP ~10 cm
        r["zmp"] = ((self.cfg.zmp_reserve + 0.02 - self.zmp_margin) / 0.05).clamp(0, 3) ** 2
        band = lambda d, b: ((b - d) / (b - self.cfg.collide_margin)).clamp(0, 1.5)
        r["clearance"] = torch.maximum(band(self.clear_mv, 0.04), band(self.clear_sh, 0.02))   # 4 cm band; 2 cm for the shoulder motors
        r["terminated"] = self.reset_terminated.float()
        r["collide"] = self.last_term[:, 1].float()
        r["alive"] = torch.ones_like(e_h)
        total = sum(w[k] * r[k] for k in w) * self.step_dt
        log = self.extras.setdefault("log", {})
        for k in w: log[f"RewTerm/{k}"] = (w[k] * r[k]).mean()
        for k in ("head_p", "head_r", "wrist_p", "wrist_r", "hand_vel", "body_yaw", "neck", "elbow"):
            log[f"Rew/{k}"] = r[k].mean()
        log["Err/head_cm"] = e_h.mean() * 100; log["Err/wrist_cm"] = 0.5 * (e_l + e_r).mean() * 100
        log["Err/head_deg"] = torch.rad2deg(a_h).mean(); log["Err/wrist_deg"] = torch.rad2deg(0.5 * (a_l + a_r)).mean()
        log["Err/wrist_deg_ego"] = (torch.rad2deg(0.5 * (a_l + a_r)) * rot_ok).sum() / rot_ok.sum().clamp_min(1)
        log["Safety/zmp_margin_cm"] = self.zmp_margin.mean() * 100; log["Safety/clear_cm"] = self.clear.mean() * 100
        log["Safety/tilt_deg"] = torch.rad2deg(torch.asin(g[:, :2].norm(dim=-1).clamp(max=1))).mean()
        log["Base/speed"] = self.v_ref_b[:, :2].norm(dim=-1).mean()
        log["Smooth/hand_acc"] = self.hand_acc.norm(dim=-1).mean(); log["Smooth/palm_acc"] = self.palm_acc.norm(dim=-1).mean(); log["Err/wrist_unreach_deg"] = torch.rad2deg(self.wres.norm(dim=-1)).mean(); log["Smooth/action_rate"] = r["action_rate"].mean(); log["Smooth/qf_acc"] = r["qf_acc"].mean(); log["Smooth/power_W"] = self.power.mean()
        self.ep_err += torch.stack([e_h, 0.5 * (e_l + e_r), a_h, 0.5 * (a_l + a_r)], -1)
        return total

    def _get_dones(self):
        self._update_state()
        g = self.robot.data.projected_gravity_b.torch
        out = (self.zmp_margin < self.cfg.zmp_out) & (self.push_t <= 0)   # the ZMP estimate ignores external pushes
        self.zmp_out_t = torch.where(out, self.zmp_out_t + self.step_dt, torch.zeros_like(self.zmp_out_t))
        tilt = (torch.asin(g[:, :2].norm(dim=-1).clamp(max=1.0)) > self.cfg.tilt_max) | (self.zmp_out_t >= self.cfg.zmp_out_s - 1e-6)
        collide = self.clear < self.cfg.collide_margin
        cp, cR = self.tracked_now()
        t = self.tgt
        _, yaw, _ = self.base_pose()
        grace = (self.episode_length_buf * self.step_dt > self.cfg.grace_s) & (self.rec_t > 1.0)   # 1 s to catch up after a stream loss
        self.drift_kind = grace[:, None] & torch.stack([
            (cp[:, 0] - t["head_p"]).norm(dim=-1) > self.cfg.drift_head,
            torch.maximum((cp[:, 1] - t["lw_p"]).norm(dim=-1), (cp[:, 2] - t["rw_p"]).norm(dim=-1)) > self.cfg.drift_wrist,
            rot_angle(cR[:, 0], t["head_R"]) > self.cfg.drift_head_rot,
            (wrap(t["body_yaw"] - yaw).abs() > self.cfg.drift_body_yaw) & (self.upper == 0)], -1)   # head pos, wrist pos, head rot, body heading
        drift = self.drift_kind.any(-1)
        live = self.episode_length_buf > 10                                   # a reset drops the robot onto its wheels: skip 0.2 s
        limit = self.lim_kind[:, [0, 2]].any(-1) & live                        # peak torque: the solver clamps it (penalised); past
        # an end stop does not stop the episode either (v39): it costs past_stop every step, so it is never a way out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        log = self.extras.setdefault("log", {})
        log["Term/tilt"] = tilt.float().mean(); log["Term/collide"] = collide.float().mean(); log["Term/drift"] = drift.float().mean()
        log["Term/limit"] = limit.float().mean()
        for k, nm in enumerate(("speed", "peak", "overload", "position")):
            log[f"Limits/{nm}"] = (self.lim_kind[:, k] & live).float().mean()
        q = self.robot.data.joint_pos.torch[:, self.jid]
        for j, nm in enumerate(R.ACTUATED):
            lo = ((q[:, j] < self.q_lo[j] - 0.01) & live).float().mean(); hi = ((q[:, j] > self.q_hi[j] + 0.01) & live).float().mean()
            if lo > 0: log[f"PosLo/{nm}"] = lo
            if hi > 0: log[f"PosHi/{nm}"] = hi
        log["Limits/torque_rms_ratio"] = (self.tau2.sqrt() / self.t_rated).max(-1).values.mean()
        self.last_term = torch.stack([tilt, collide, drift, limit], -1)
        return tilt | collide | (drift & self.cfg.drift_stop) | limit, time_out

    # ------------------------------------------------------------ reset
    def _reset_idx(self, env_ids):
        if env_ids is None: env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n, dev = len(env_ids), self.device
        # clip + start + rate
        self.upper[env_ids] = (torch.rand(n, device=dev) < self.cfg.upper_share).float()
        up = self.upper[env_ids] > 0
        c = torch.multinomial(self.c_p, n, replacement=True)
        if up.any(): c[up] = torch.multinomial(self.c_p_up, int(up.sum()), replacement=True)
        rate = torch.empty(n, device=dev).uniform_(*self.cfg.speed_range)
        L = self.c_len[c].float()
        start = torch.rand(n, device=dev) * (L - 2.0 * 50).clamp_min(0) * 0.8
        self.c_id[env_ids] = c; self.rate[env_ids] = rate
        self.phase[env_ids] = start
        # augmentation flags
        self.aug_on[env_ids] = ((torch.rand(n, device=dev) < self.cfg.twist_aug) & ~up).float()
        self.neck_on[env_ids] = (torch.rand(n, device=dev) < self.cfg.neck_aug).float()
        for b in (self.aug_yaw, self.aug_w): b[env_ids] = 0
        for b in (self.aug_xy, self.aug_v, self.neck_off, self.neck_tgt): b[env_ids] = 0
        # robot initial state: random lift, ready arms, at the env origin, random heading
        lift = torch.empty(n, device=dev).uniform_(0.25, 0.75) * R.LIFT_STROKE
        lift[up] = torch.empty(int(up.sum()), device=dev).uniform_(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)   # locked anywhere in its travel
        q = self.q_ready[None].repeat(n, 1)
        q[:, self.i_lift[0]] = lift; q[:, self.i_lift[1]] = lift
        q[:, self.i_arm] += torch.randn(n, len(self.i_arm), device=dev) * 0.05
        self.robot.write_joint_position_to_sim_index(position=q, joint_ids=self.jid, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=self.jid, env_ids=env_ids)
        yaw0 = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        hq = yaw0 + math.pi                                               # URDF base faces -x
        root[:, 3:7] = torch.stack([torch.zeros_like(hq), torch.zeros_like(hq), torch.sin(hq / 2), torch.cos(hq / 2)], -1)
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
        # anchor: the human's start pose -> the robot's: torso heading -> base heading, head -> robot eye
        f0 = self.c_start[c].float() + start
        i0 = f0.long()
        # engage as the robot can do it: the operator's GAZE heading and head-estimated shoulders (no torso tracking)
        hR = self.m["head_R"][i0]; h_yaw = torch.atan2(hR[:, 1, 0], hR[:, 0, 0])
        self.anc_yaw[env_ids] = yaw0 - h_yaw
        self.anc_h[env_ids] = self.m["head_p"][i0]
        # robot eye in world at this lift (FK numbers: eye 1.374 m at lift 0, +2*stage; 0.176 m ahead of the base)
        eye_fwd = 0.176
        # put the human head where the robot eye is, shifted by the head's offset from the (estimated) shoulder line
        sl, sr = ML.shoulders_from_head(self.m["head_p"][i0], h_yaw, torch); sm = 0.5 * (sl + sr)
        self.sm0[env_ids] = sm[:, :2]; self.sm_lp[env_ids] = sm[:, :2]
        Rn = rot_z(yaw0 - h_yaw)
        off = (Rn @ (self.m["head_p"][i0] - sm)[..., None])[..., 0]
        base_xy = root[:, :2]
        fwd = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], -1)
        anc_r = torch.zeros(n, 3, device=dev)
        anc_r[:, :2] = base_xy + fwd * 0.096 + off[:, :2]                # torso line over the base, head where the human's is
        anc_r[:, 2] = 1.374 + 2 * lift + torch.randn(n, device=dev) * 0.03
        self.anc_r[env_ids] = anc_r
        # buffers
        self.lift_tgt[env_ids] = lift
        self.q_tgt[env_ids] = q; self.v_tgt[env_ids] = 0
        for b in (self.v_ref_b, self.v_ref_prev, self.actions, self.prev_actions, self.prev2_actions, self.act_buf, self.com_acc, self.ep_err): b[env_ids] = 0
        self.q_f[env_ids] = q[:, self.i_pos]; self.q_f_prev[env_ids] = q[:, self.i_pos]; self.q_f1[env_ids] = q[:, self.i_pos]
        qw = torch.cat([q[:, self.i_pos][:, self.i_wr_all].reshape(-1, 2, 3), torch.zeros(n, 2, 1, device=dev)], -1)
        self.wik[env_ids] = qw; self.w_safe[env_ids] = qw; self.clear_hand[env_ids] = 1.0; self.ik_buf[env_ids] = qw.reshape(-1, 8)[:, None]; self.wres[env_ids] = 0
        for b in (self.hw_f1, self.hw_f, self.hw_sp, self.hw_spv): b[env_ids] = 0
        self.robot.write_joint_position_to_sim_index(position=torch.zeros(n, 2, device=dev), joint_ids=self.jid_hw, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(n, 2, device=dev), joint_ids=self.jid_hw, env_ids=env_ids)
        self.sp[env_ids] = q[:, self.i_pos]; self.sp_v[env_ids] = 0; self.v_ff[env_ids] = 0
        self.hand_v_prev[env_ids] = 0; self.hand_acc[env_ids] = 0; self.tgt_v[env_ids] = 0; self.palm_w_prev[env_ids] = 0
        for b in (self.tw_f, self.v_cmd_b, self.steer, self.steer_cmd, self.steer_prev, self.tau): b[env_ids] = 0
        self.tau2[env_ids] = self.tau_g[env_ids].square()                 # the hold load, not a cold motor
        self.steer_k[env_ids] = torch.empty(n, device=dev).uniform_(0.8, 1.2)
        self.qd_prev[env_ids] = 0
        self.delay[env_ids] = torch.randint(self.cfg.action_delay[0], self.cfg.action_delay[1] + 1, (n,), device=dev)
        self.kv[env_ids] = torch.empty(n, device=dev).uniform_(6.0, 15.0)
        self.push_t[env_ids] = 0.0; self.zmp_out_t[env_ids] = 0.0
        ks = torch.ones(n, len(R.ACTUATED), device=dev)
        if self.cfg.dr_physics:
            ks = torch.empty(n, len(R.ACTUATED), device=dev).uniform_(0.75, 1.25)
            self.robot.write_joint_stiffness_to_sim_index(stiffness=self.kp0[None] * ks, joint_ids=self.jid, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim_index(damping=self.kd0[None] * ks.sqrt(), joint_ids=self.jid, env_ids=env_ids)
            m = self.mass0[None].repeat(n, 1) * torch.empty(n, len(self.mass0), device=dev).uniform_(0.85, 1.15)
            m[:, self.hand_bodies] += torch.empty(n, 2, device=dev).uniform_(0.0, 0.3)      # elbow j4 at 80 % rated caps it
            self.robot.set_masses_index(masses=m, env_ids=env_ids)
            self.masses[env_ids] = m; self.m_tot[env_ids] = m.sum(-1)
            self.gc_scale[env_ids] = torch.empty(n, len(self.i_pos), device=dev).uniform_(0.8, 1.2)
        self.kp_env[env_ids] = (self.kp0[None] * ks)[:, self.i_pos]; self.kd_env[env_ids] = (self.kd0[None] * ks.sqrt())[:, self.i_pos]
        self.kw[env_ids] = torch.empty(n, device=dev).uniform_(5.0, 12.0)
        self.drive_k[env_ids] = torch.empty(n, 3, device=dev).uniform_(0.85, 1.15)        # open-loop duty: speed error per module
        self.coast[env_ids] = torch.rand(n, device=dev) > 0.1                             # 10 %: a drive has not acked COAST
        self.gov[env_ids] = torch.tensor([R.BASE_ACCEL, R.BASE_VMAX, R.BASE_WMAX, R.BASE_ACCEL * R.BASE_DECEL], device=dev)
        self.com_v_prev[env_ids] = 0
        self.need_glide[env_ids] = True
        self.blend_on[env_ids] = 0.0
        # targets at the new start (no clock advance), history filled with the first observation
        self.tgt = self.compute_targets(self.phase, step_aug=False)
        self.stream_reset(env_ids)
        pos, yaw, _ = self.base_pose()
        tob = self.target_obs(pos, yaw, noisy=False)
        self.hist[env_ids] = tob[env_ids, None].repeat(1, self.hist.shape[1], 1)

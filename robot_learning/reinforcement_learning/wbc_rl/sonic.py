"""sonic.py — NOVA whole-body teleop, SONIC / Dyna-2.1 style, OpenArm v1 grippers on both arms, trained from scratch
with mjlab (MuJoCo Warp).

One RL controller tracks task-space targets through physics and outputs every joint:
  * targets (Dyna-2.1 "URR"): both wrist centres (pose), both elbows, chest (lean + heading), head (pose) and the
    footprint (base position) — the operator's landmarks, shape-preserving retarget (motion_lib.retarget_hand) onto
    NOVA's gripper arm; hands weigh most, then elbows, then chest (Dyna: "hand accuracy over chest accuracy");
  * SONIC's universal token: encoder -> FSQ token (32 x 8 levels) -> one decoder. The TEACHER encodes the full future
    targets (now, +60 ms, +200 ms, +500 ms) and is trained with PPO; the STUDENT encodes only what the Vision Pro
    sends (head + palms, with every stream fault) and is distilled onto the teacher's token and actions (DAgger)
    through the teacher's frozen decoder. The robot runs the student (deploy.SonicAdapter);
  * adaptive sampling (SONIC): clips that end in a stop or track badly are drawn more often;
  * domain randomisation (Dyna): link masses + inertia, PD gains, joint damping / friction / armature, gripper
    payload, command delays, pushes.
The plant and the operator are wbc_core.WbcCore (shared with env.py, verified against NOVA): nova_hw setpoint
shaper, datasheet speed / torque limits, gravity feed-forward, 3 cm/s lift, swerve base with the tip governor,
Vision Pro stream faults. mjlab's IdealPd actuator is NOVA's MIT frame: tau = kp (sp - q) + kd (v_ff - qd) + t_ff.
Data: EgoDex + LAFAN1 (data/motions_v4.npz) + optional operator Retarget Lab takes (data/avp_takes.npz, every 5th held out).

    cd robot_learning/reinforcement_learning/wbc_rl     (python = mjlab's venv, e.g. $MJLAB_PY)
    python sonic.py train --run s3 [--iters 6000] [--envs 4096] [--video]   # teacher (PPO); --video: 9 envs every 2000 steps
    python sonic.py distil --run s3s --teacher <model.pt> [--iters 1500]    # student (DAgger onto the token)
    python sonic.py eval <model.pt> [--student] [--clips mix|egodex|lafan|avp] [--no-faults]
    python sonic.py view <model.pt> [--envs 16]                             # live viewer (viser, http://localhost:8080)
"""
from __future__ import annotations

import argparse, json, math, os, sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mujoco  # noqa: E402
from rsl_rl.algorithms import PPO  # noqa: E402
from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.modules import MLP  # noqa: E402

from mjlab.actuator import IdealPdActuatorCfg  # noqa: E402
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.envs.mdp import dr  # noqa: E402
from mjlab.envs.mdp.terminations import time_out  # noqa: E402
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg  # noqa: E402
from mjlab.scene import SceneCfg  # noqa: E402
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: E402
from mjlab.terrains import TerrainEntityCfg  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402

import motion_lib as ML  # noqa: E402
import robot as R  # noqa: E402
from wbc_core import WHEELS_B, G, WbcCore, quat_wxyz_to_mat, rot_angle, rot_z, six_d, wrap  # noqa: E402

FUT = (0, 3, 10, 25)                 # target frames the teacher sees: now, +60 ms, +200 ms, +500 ms (policy steps)
REACH = 0.63 / 0.6573 * 0.4426       # m: shoulder -> wrist centre of NOVA's straight gripper arm (robot.py FK), same
                                     # elbow-just-off-straight margin as ML.ROBOT_REACH (0.63 of the ORCA arm's 0.657)
SRC = ("egodex", "lafan", "avp")
TASK = "Nova-Sonic-Teacher"


# ---------------------------------------------------------------- config (env.py's NOVA numbers + sonic's own)
@dataclass
class SonicCfg:
    motions: str = os.path.join(HERE, "data", "motions_v4.npz")
    avp_takes: str = os.path.join(HERE, "data", "avp_takes.npz")
    avp_share: float = 0.3            # of resets, when takes exist (the robot's own operator and headset)
    egodex_share: float = 0.5         # of the rest (else LAFAN1 walking)
    held_only: bool = False           # eval: draw the takes only from the held-out ones (avp_eval:)
    speed_range: tuple = (0.7, 1.4)   # operator playback rate
    twist_aug: float = 0.6            # synthetic walk / turn on top of EgoDex clips
    aug_walk_max: float = 0.35
    upper_share: float = 0.3          # upper-body mode (base and lift locked; the deploy toggle)
    neck_aug: float = 0.5
    scripted: bool = False
    action_scale: float = 0.5
    action_lpf_hz: float = 3.0        # NOVA's reference filter (deploy), 2nd order; the wrist joints faster:
    wrist_lpf_hz: float = 6.0         # arm_5-7 move light parts on 20.9 rad/s motors
    hw_smooth_wn: float = 40.0        # nova_hw setpoint shaper (rad/s), wrist joints 80
    hw_wrist_wn: float = 80.0
    action_delay: tuple = (1, 4)      # policy steps of output latency (20-80 ms)
    faults: bool = True
    stream: dict = field(default_factory=lambda: dict(
        lat=(0, 5), lat_jit=0.25, loss=(0.0, 0.15), slow_rate=1 / 6.0, outage_rate=1 / 15.0, occl_rate=1 / 8.0, predict=0.5,
        spike=0.004, noise_p=(0.001, 0.004), noise_r=(0.005, 0.04), trem_hz=(3.0, 12.0), trem_p=(0.0, 0.006),
        trem_r=(0.0, 0.05), trem_tau=0.3, bias_p=0.015, bias_r=0.07, odom=(0.0005, 0.03, 0.02)))
    dr_physics: bool = True
    payload: tuple = (0.0, 0.5)       # kg in each gripper
    push_force: tuple = (10.0, 30.0)
    push_every: float = 4.0
    tilt_max: float = math.radians(2.0)
    zmp_out: float = -0.01
    zmp_out_s: float = 0.1
    zmp_reserve: float = 0.03
    collide_margin: float = 0.01
    torque_tau: float = 5.0
    limit_tol: float = 1.0
    twist_tau: float = 0.08
    steer_rate: float = (math.pi / 2) / 0.17
    noise: dict = field(default_factory=lambda: dict(q=0.005, qd=0.08, base=0.05, cur=0.005))
    adapt: float = 3.0                # adaptive sampling: clip weight x (1 + adapt * failure score)
    pen_ramp: int = 1500              # iterations over which the smoothness penalties ramp 10 % -> 100 % (s2)
    far: float = 1.5                  # m: head or footprint this far off is a stop (s2)
    w: dict = field(default_factory=lambda: dict(
        wrist_p=6.0, wrist_r=6.0, elbow=1.5, head_p=2.0, head_r=1.5, lean=0.5, body_yaw=0.5, foot=1.0, hand_vel=1.0,
        action_rate=-0.05, action_acc=-0.01, hand_acc=-0.05, palm_acc=-2e-4, joint_acc=-2e-7, qf_acc=-3e-5,
        base_acc=-0.1, steer=-0.01, lift_use=-2.0, act_sat=-0.5, joint_limit=-10.0, past_stop=-250.0, limits=-6.0, track_lin=-2.0,
        torque=-0.02, tilt=-60.0, tilt_rate=-2.0, zmp=-40.0, com_acc=-1.0, clearance=-20.0,
        terminated=-500.0, collide=-4500.0, alive=2.0))
    SAFETY_TERMS: tuple = ("joint_limit", "past_stop", "limits", "tilt", "tilt_rate", "zmp", "clearance", "terminated", "collide")


# ---------------------------------------------------------------- NOVA as an mjlab entity
def nova_spec() -> mujoco.MjSpec:
    """data/nova_grip.xml (robot.py gripper): the three wheel spheres are the only colliders, frictionless (condim 1,
    priority over the floor), as the drive is a force at the floor; every mesh is visual."""
    spec = mujoco.MjSpec.from_file(R.GRIP_MJCF)
    for m in spec.meshes:                               # mesh paths in data/*.xml are relative to the file: pin them, so
        if not os.path.isabs(m.file):                   # they still resolve once mjlab attaches the spec to its scene
            m.file = os.path.normpath(os.path.join(os.path.dirname(R.GRIP_MJCF), m.file))
    for j in spec.joints:                              # the URDF's effort="100" placeholders became joint force caps
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE   # (100 N would drop the lift); the actuators cap at the datasheet
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.contype:
            g.condim, g.priority, g.friction = 1, 2, [1e-4, 0.0, 0.0]
    return spec


def nova_entity() -> EntityCfg:
    grp = {g: [n for n in R.ACTUATED if n not in R.LIFT and R.joint_group(n) == g] for g in R.JOINT_SPEC if g != "hand_wrist"}
    arm = {"arm_1": 0.05, "arm_2": 0.05, "arm_3": 0.03, "arm_4": 0.03, "torso": 0.2}
    acts = [IdealPdActuatorCfg(target_names_expr=tuple(names), stiffness=R.HW_GAINS[R.gain_group(g)][0],
                               damping=R.HW_GAINS[R.gain_group(g)][1], effort_limit=R.SAFETY * R.JOINT_SPEC[g][1],
                               armature=arm.get(g, 0.01)) for g, names in grp.items() if names]
    # the lift is a lead screw: its reflected inertia is huge (a 1e-4 kg m^2 rotor on a 5 mm lead is ~160 kg). Isaac's
    # implicit PD hid it; mjlab's PD is explicit and kd 4000 on the light stage-to-stage mode needs >= ~20 kg to be stable
    acts.append(IdealPdActuatorCfg(target_names_expr=tuple(R.LIFT), stiffness=40000.0, damping=4000.0, effort_limit=3000.0, armature=20.0))
    q0 = {n: R.READY.get(n, 0.0) for n in R.ACTUATED}; q0.update({n: R.LIFT_STROKE / 2 for n in R.LIFT})
    return EntityCfg(spec_fn=nova_spec, articulation=EntityArticulationInfoCfg(actuators=tuple(acts)),
                     init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, R.CHASSIS_Z + 0.002), rot=(0.0, 0.0, 0.0, 1.0), joint_pos=q0))


class _T:
    """Isaac Lab's `robot.data.<field>.torch` accessor."""
    __slots__ = ("torch",)

    def __init__(self, t): self.torch = t


def _xyzw(q):
    return torch.cat([q[..., 1:], q[..., :1]], -1)


class MjRobot:
    """The mjlab entity with the Isaac Lab articulation API wbc_core / SonicCore use (quaternions xyzw)."""
    def __init__(self, env: ManagerBasedRlEnv):
        self.env, self.e = env, env.scene["robot"]; self.data = self
        self.body_names = list(self.e.body_names)
        self.act_of = {}                                    # joint (local id) -> (actuator, column)
        for a in self.e.actuators:
            for c, n in enumerate(a.target_names): self.act_of[self.e.joint_names.index(n)] = (a, c)

    d = property(lambda s: s.e.data)
    joint_pos = property(lambda s: _T(s.d.joint_pos)); joint_vel = property(lambda s: _T(s.d.joint_vel))
    body_link_pos_w = property(lambda s: _T(s.d.body_link_pos_w)); body_link_quat_w = property(lambda s: _T(_xyzw(s.d.body_link_quat_w)))
    body_link_ang_vel_w = property(lambda s: _T(s.d.body_link_ang_vel_w))
    body_com_pos_w = property(lambda s: _T(s.d.body_com_pos_w)); body_com_lin_vel_w = property(lambda s: _T(s.d.body_com_lin_vel_w))
    root_link_pos_w = property(lambda s: _T(s.d.root_link_pos_w)); root_link_quat_w = property(lambda s: _T(_xyzw(s.d.root_link_quat_w)))
    root_link_lin_vel_w = property(lambda s: _T(s.d.root_link_lin_vel_w)); root_link_ang_vel_w = property(lambda s: _T(s.d.root_link_ang_vel_w))
    root_com_lin_vel_w = property(lambda s: _T(s.d.root_com_lin_vel_w)); root_com_ang_vel_w = property(lambda s: _T(s.d.root_com_ang_vel_w))
    root_link_ang_vel_b = property(lambda s: _T(s.d.root_link_ang_vel_b)); projected_gravity_b = property(lambda s: _T(s.d.projected_gravity_b))

    @property
    def body_mass(self):
        m = self.env.sim.model.body_mass
        m = m if m.dim() == 2 else m[None].expand(self.env.num_envs, -1)
        return _T(m[:, self.e.indexing.body_ids])

    def find_joints(self, names, preserve_order=True): return self.e.find_joints(names, preserve_order=preserve_order)
    def find_bodies(self, name): return self.e.find_bodies(name, preserve_order=True)
    def write_joint_position_to_sim_index(self, position, joint_ids, env_ids): self.e.write_joint_position_to_sim(position, joint_ids=joint_ids, env_ids=env_ids)
    def write_joint_velocity_to_sim_index(self, velocity, joint_ids, env_ids): self.e.write_joint_velocity_to_sim(velocity, joint_ids=joint_ids, env_ids=env_ids)
    def set_joint_position_target_index(self, target, joint_ids): self.e.set_joint_position_target(target, joint_ids=joint_ids)
    def set_joint_velocity_target_index(self, target, joint_ids): self.e.set_joint_velocity_target(target, joint_ids=joint_ids)
    def set_joint_effort_target_index(self, target, joint_ids): self.e.set_joint_effort_target(target, joint_ids=joint_ids)

    def write_root_pose(self, pos, yaw, env_ids):
        """Root link pose: position (n, 3), REP yaw (n,) -> the chassis frame (URDF base faces -x)."""
        h = yaw + math.pi
        q = torch.stack([torch.cos(h / 2), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h / 2)], -1)
        self.e.write_root_link_pose_to_sim(torch.cat([pos, q], -1), env_ids=env_ids)
        self.e.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=pos.device), env_ids=env_ids)

    def set_gains(self, kp, kd, joint_ids, env_ids):
        """Per env PD gains (n, J) for the given joints (NOVA's per-joint kp / kd x randomisation)."""
        for c, j in enumerate(joint_ids):
            a, k = self.act_of[int(j)]
            a.stiffness[env_ids, k] = kp[:, c]; a.damping[env_ids, k] = kd[:, c]

    def wrench(self, force, torque, body_ids):
        self.e.data.write_external_wrench(force, torque, body_ids=body_ids)


# ---------------------------------------------------------------- the env logic (env.py's plant + sonic's task)
class SonicCore(WbcCore):
    """Everything the managers below read: targets, stream, plant, observations, rewards, stops, resets."""
    def __init__(self, env: ManagerBasedRlEnv, cfg: SonicCfg):
        self.env, self.cfg = env, cfg
        self.robot = MjRobot(env)
        dev, N = self.device, self.num_envs
        names = R.ACTUATED
        self.jid = torch.tensor(self.robot.find_joints(names, preserve_order=True)[0], device=dev)
        self.i_lift = [names.index(n) for n in R.LIFT]
        self.i_torso = names.index("torso")
        self.i_neck = [names.index(n) for n in R.NECK]
        self.i_arm = [names.index(n) for n in R.ARM]
        self.i_pos = [self.i_torso] + self.i_neck + self.i_arm
        fk = R.FK(R.GRIP_URDF)
        lim = np.array([fk.J[fk.joint_child[n]]["lim"] for n in names])
        tt = lambda x: torch.tensor(np.array(x, dtype=np.float32), device=dev)
        self.q_lo, self.q_hi = tt(lim[:, 0]), tt(lim[:, 1])
        self.q_ready = tt([R.READY.get(n, 0.0) for n in names]); self.q_ready[self.i_lift] = R.LIFT_STROKE / 2
        self.body = {}
        def bid(link):
            if link not in self.body:
                ids, _ = self.robot.find_bodies(link)
                assert len(ids) == 1, f"body {link}: {ids} (bodies: {self.robot.body_names})"
                self.body[link] = ids[0]
            return self.body[link]
        assert bid("mabel_chassis") == 0, self.robot.body_names[:3]           # wbc_core reads the chassis as body 0
        T = R.tracked_offsets(fk, "gripper")
        self.trk_body = torch.tensor([bid(T[k][0]) for k in ("head", "left_wrist", "right_wrist")], device=dev)
        self.trk_p = tt(np.stack([T[k][1] for k in ("head", "left_wrist", "right_wrist")]))
        self.trk_R = tt(np.stack([T[k][2] for k in ("head", "left_wrist", "right_wrist")]))
        self.elbow_body = torch.tensor([bid("l_elbow"), bid("r_elbow")], device=dev)
        self.torso_body = bid("torso")
        parts, sph, pairs = R.collision_model(ee="gripper")
        self.sph_body = torch.tensor([bid(x[0]) for x in sph], device=dev)
        self.sph_c, self.sph_r = tt([x[1] for x in sph]), tt([x[2] for x in sph])
        self.sph_sh = torch.tensor([str(x[4]) in ("shoulder_1", "shoulder_2") for x in sph], device=dev)
        self.sph_hand = torch.tensor(np.array([[str(x[4]) in R.GRIP_HAND and str(x[3]) == s for s in "lr"] for x in sph]), device=dev)
        self.sdf = []
        for j, p in enumerate(parts):
            ids = torch.tensor(np.nonzero(pairs[:, j])[0], device=dev)
            if len(ids) == 0: continue
            g = tt(p["sdf"]).permute(2, 1, 0)[None, None].contiguous()
            fh = "lr".index(p["side"]) if p["side"] in ("l", "r") and p["name"] in R.GRIP_HAND else -1
            self.sdf.append((bid(p["link"]), tt(p["origin"]), tt((np.array(p["sdf"].shape) - 1) * p["res"]), g, ids, fh))
        self.mass0 = self.robot.data.body_mass.torch[0].clone()
        self.masses = self.mass0[None].repeat(N, 1); self.m_tot = self.masses.sum(-1)
        grp = lambda n: R.gain_group(R.joint_group(n)) if n not in R.LIFT else None
        self.kp0 = tt([R.HW_GAINS[grp(n)][0] if grp(n) else 40000.0 for n in names])
        self.kd0 = tt([R.HW_GAINS[grp(n)][1] if grp(n) else 4000.0 for n in names])
        self.hand_bodies = [bid("l_hand_mount"), bid("r_hand_mount")]
        self.wheels = tt(WHEELS_B)
        names_pos = [names[i] for i in self.i_pos]
        def chain(link):
            out = [link]
            while link in fk.J: link = fk.J[link]["parent"]; out.append(link)
            return out
        chains = [set(chain(b)) for b in self.robot.body_names]
        self.gc_body = torch.tensor([bid(fk.joint_child[n]) for n in names_pos], device=dev)
        self.gc_axis = tt([fk.J[fk.joint_child[n]]["axis"] / np.linalg.norm(fk.J[fk.joint_child[n]]["axis"]) for n in names_pos])
        self.gc_sub = tt([[float(fk.joint_child[n] in c) for c in chains] for n in names_pos])
        self.gc_scale = torch.ones(N, len(names_pos), device=dev)
        self.ff_max = tt([R.FF_MAX_NM[R.joint_group(n)] for n in names_pos])
        self.lf_body = torch.tensor([bid(fk.joint_child[n]) for n in R.LIFT], device=dev)
        self.lf_axis = tt([fk.J[fk.joint_child[n]]["axis"] for n in R.LIFT])
        self.lf_sub = tt([[float(fk.joint_child[n] in c) for c in chains] for n in R.LIFT])
        sp = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)
        lo = (self.q_lo[self.i_pos] + sp - self.q_ready[self.i_pos]) / cfg.action_scale
        hi = (self.q_hi[self.i_pos] - sp - self.q_ready[self.i_pos]) / cfg.action_scale
        self.a_lo = torch.cat([torch.full((4,), -2.5, device=dev), lo.clamp(min=-5.0)])
        self.a_hi = torch.cat([torch.full((4,), 2.5, device=dev), hi.clamp(max=5.0)])
        self.slew = tt([min(R.HW_GAINS[grp(n)][2], R.HEADROOM * R.SAFETY * R.JOINT_SPEC[R.joint_group(n)][0]) for n in names_pos])
        spec = [R.JOINT_SPEC[R.joint_group(n)] for n in names_pos]
        self.v_lim = tt([R.SAFETY * x[0] for x in spec]); self.t_pk = tt([R.SAFETY * x[1] for x in spec])
        self.t_rated = tt([R.SAFETY * x[2] for x in spec])
        self.ovl_hard = torch.tensor([R.joint_group(n) not in R.OVERLOAD_SOFT for n in names_pos], device=dev)
        self.mods = tt(WHEELS_B)
        A = torch.zeros(6, 3, device=dev)
        for i, (x, y) in enumerate(WHEELS_B): A[2 * i] = torch.tensor([1.0, 0.0, -y]); A[2 * i + 1] = torch.tensor([0.0, 1.0, x])
        self.mods_pinv = torch.linalg.pinv(A)
        self.i_wr_all = [self.i_pos.index(names.index(f"{s}_arm_{j}")) for s in ("left", "right") for j in (5, 6, 7)]
        hz = torch.full((len(self.i_pos),), cfg.action_lpf_hz, device=dev); hz[self.i_wr_all] = cfg.wrist_lpf_hz
        self.beta = 1.0 - torch.exp(-2 * math.pi * hz * self.step_dt)
        self.load_motions()
        self.alloc()
        self._k_rew = self._k_done = -1
        self.reset_state(torch.arange(N, device=dev))

    # host env (wbc_core reads these)
    num_envs = property(lambda s: s.env.num_envs)
    device = property(lambda s: s.env.device)
    step_dt = property(lambda s: s.env.step_dt)
    scene = property(lambda s: s.env.scene)
    extras = property(lambda s: s.env.extras)
    episode_length_buf = property(lambda s: s.env.episode_length_buf)
    common_step_counter = property(lambda s: s.env.common_step_counter)
    reset_terminated = property(lambda s: getattr(s.env, "reset_terminated", None) if getattr(s.env, "reset_terminated", None) is not None
                                else torch.zeros(s.num_envs, dtype=torch.bool, device=s.device))

    def euro_step(self, Rx, reset=None):              # env.py filters its wrist solver's input; no solver here
        pass

    # ------------------------------------------------------------ motion library: EgoDex + LAFAN1 + optional operator takes
    def load_motions(self):
        z = dict(np.load(self.cfg.motions))
        if os.path.exists(self.cfg.avp_takes):
            a = np.load(self.cfg.avp_takes); n0 = len(z["head_p"])
            z = {k: (np.concatenate([z[k], a[k]]) if k not in ("starts", "wrist_dz") else z[k]) for k in z}
            z["starts"] = np.concatenate([np.load(self.cfg.motions)["starts"], a["starts"] + n0])
        t = lambda k: torch.tensor(z[k], dtype=torch.float32, device=self.device)
        self.m = dict(head_p=t("head_p"), lw_p=t("lw_p"), rw_p=t("rw_p"), le=t("le"), re=t("re"), ls=t("ls"), rs=t("rs"), sm=t("sm"),
                      head_R=quat_wxyz_to_mat(t("head_q")), lw_R=quat_wxyz_to_mat(t("lw_q")), rw_R=quat_wxyz_to_mat(t("rw_q")),
                      torso_yaw=t("torso_yaw"), lean=t("lean"), neck=t("neck"))
        self.c_start, self.c_len = t("starts").long(), t("lens").long()
        meta = np.array([str(m) for m in z["meta"]])
        src = np.array([2 if m.startswith("avp") else 0 if m.startswith("egodex") else 1 for m in meta])
        held = np.array([m.startswith("avp_eval") for m in meta])
        # shape-preserving retarget onto the gripper arm: the operator's reach (shoulder -> elbow -> wrist, per-clip
        # median) -> NOVA's shoulder -> wrist centre. The takes have no elbows: the EgoDex median operator reach
        n = lambda a, b: np.linalg.norm(z[a] - z[b], axis=1)
        h_reach = 0.5 * (n("le", "ls") + n("lw_p", "le") + n("re", "rs") + n("rw_p", "re"))
        med = lambda x: np.concatenate([np.full(l, np.median(x[s:s + l])) for s, l in zip(z["starts"], z["lens"])])
        hr = med(h_reach); rows = lambda c: np.concatenate([np.full(l, src[i] == c) for i, l in enumerate(z["lens"])])
        hr[rows(2)] = np.median(hr[rows(0)]) if rows(0).any() else 0.55
        self.m["arm_k"] = torch.tensor(REACH / hr, dtype=torch.float32, device=self.device)[:, None]
        self.c_src = torch.tensor(src, device=self.device); self.c_held = torch.tensor(held, device=self.device)
        self.c_ego = self.c_src != 1                       # palms tracked (LAFAN1 has no finger markers) + synthetic walking
        self.c_body = self.c_src != 2                      # elbows / torso labels exist (the takes have only head + palms)
        w = z["lens"].astype(np.float64) * (np.where(src == 2, held, False) if self.cfg.held_only else ~held)
        has = (src == 2).any()
        share = {0: self.cfg.egodex_share * (1 - self.cfg.avp_share * has), 2: self.cfg.avp_share}
        share[1] = 1 - share[0] - (share[2] if has else 0)
        p = sum(share[s] * np.where(src == s, w, 0) / max(np.where(src == s, w, 0).sum(), 1) for s in (0, 1, 2) if (src == s).any())
        self.c_p0 = torch.tensor(p / p.sum(), dtype=torch.float32, device=self.device)
        pu = np.where(src != 1, w, 0)
        self.c_p_up0 = torch.tensor(pu / pu.sum(), dtype=torch.float32, device=self.device)
        self.c_fail = torch.zeros(len(w), device=self.device)
        self.c_p, self.c_p_up = self.c_p0.clone(), self.c_p_up0.clone()
        self.wrist_dz = float(z["wrist_dz"])
        print(f"[sonic] motions: {len(w)} clips ({[(SRC[s], int((src == s).sum())) for s in (0, 1, 2)]}), {z['lens'].sum() / 50 / 3600:.2f} h, "
              f"{held.sum()} held-out takes; arm scale {self.m['arm_k'].median().item():.2f} (gripper reach {REACH:.3f} m)")

    def resample_weights(self):
        """SONIC adaptive sampling: weight x (1 + adapt * failure score), score = EMA of stop / bad-tracking resets."""
        k = 1 + self.cfg.adapt * self.c_fail
        self.c_p = self.c_p0 * k; self.c_p /= self.c_p.sum()
        self.c_p_up = self.c_p_up0 * k; self.c_p_up /= self.c_p_up.sum().clamp_min(1e-9)

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
        self.gov = torch.tensor([R.BASE_ACCEL, R.BASE_VMAX, R.BASE_WMAX, R.BASE_ACCEL * R.BASE_DECEL], device=dev).repeat(N, 1)
        self.coast = torch.ones(N, dtype=torch.bool, device=dev); self.drive_k = torch.ones(N, 3, device=dev)
        self.vb_obs = z(3); self.rec_t = torch.full((N,), 10.0, device=dev)
        self.n_step = 0; self.n_gov = 0
        J = len(self.i_pos)
        self.actions, self.prev_actions, self.prev2_actions, self.raw_act = z(22), z(22), z(22), z(22)
        self.q_f, self.q_f_prev, self.q_f1, self.qf_acc = z(J), z(J), z(J), z(J)
        self.sp, self.sp_v, self.v_ff = z(J), z(J), z(J)
        self.last_term = torch.zeros(N, 4, dtype=torch.bool, device=dev)
        self.hw_wn = torch.full((J,), self.cfg.hw_smooth_wn, device=dev); self.hw_wn[self.i_wr_all] = self.cfg.hw_wrist_wn
        self.i_servo = [self.i_pos.index(R.ACTUATED.index(n)) for n in ("neck_pitch", "neck_roll")]
        self.f_lift = z(2); self.tau_g, self.tau, self.tau2 = z(J), z(J), z(J)
        self.kp_env, self.kd_env = self.kp0[None, self.i_pos].repeat(N, 1), self.kd0[None, self.i_pos].repeat(N, 1)
        self.tw_f, self.v_cmd_b, self.steer, self.steer_cmd, self.steer_prev = z(3), z(3), z(3), z(3), z(3)
        self.steer_k = torch.ones(N, device=dev)
        self.lim_kind = torch.zeros(N, 4, dtype=torch.bool, device=dev)
        self.lim_joint = torch.zeros(N, 3, len(R.ACTUATED), dtype=torch.bool, device=dev)
        self.hand_v_prev, self.hand_acc, self.tgt_v, self.palm_w_prev, self.palm_acc = z(2, 3), z(2, 3), z(2, 3), z(2, 3), z(2, 3)
        self.act_buf = z(self.cfg.action_delay[1] + 1, 22); self.delay = torch.zeros(N, dtype=torch.long, device=dev)
        self.v_ref_b, self.v_ref_prev = z(3), z(3)
        self.lift_tgt = z(); self.q_tgt = z(len(R.ACTUATED)); self.v_tgt = z(len(R.ACTUATED))
        self.upper, self.sm0, self.sm_lp = z(), z(2), z(2)
        self.kv, self.kw = torch.full((N,), 10.0, device=dev), torch.full((N,), 8.0, device=dev)
        self.com_v_prev, self.com_acc = z(3), z(3)
        self.zmp_margin, self.clear, self.clear_mv, self.clear_sh = z(), z(), z(), z()
        self.clear_hand = torch.ones(N, 2, device=dev)
        self.qd_prev = z(len(R.ACTUATED))
        self.ep_err, self.ep_n = z(4), z()
        self.push_t, self.push_f, self.zmp_out_t = z(), z(3), z()
        self.blend_on, self.blend_t, self.blend_T = z(), z(), torch.ones(N, device=dev)
        self.blend_p, self.blend_R = z(3, 3), torch.eye(3, device=dev).repeat(N, 3, 1, 1)
        self.need_glide = torch.zeros(N, dtype=torch.bool, device=dev)
        self.oe_R, self.oe_s = torch.eye(3, device=dev).repeat(N, 2, 1, 1), z(2)      # wbc_core.stream_reset writes them (unused)
        self.log = {}

    # ------------------------------------------------------------ actions -> NOVA's command chain (once per policy step)
    def pre_physics(self, actions):
        self.raw_act = actions
        a = actions.clamp(-5.0, 5.0)
        self.prev2_actions = self.prev_actions.clone(); self.prev_actions = self.actions.clone()
        self.act_buf = torch.cat([a[:, None], self.act_buf[:, :-1]], 1)
        a = self.act_buf[torch.arange(self.num_envs, device=self.device), self.delay]
        self.actions = a
        dt = self.step_dt
        # base: twist -> tip governor -> low-pass -> accel limit -> swerve modules (env.py, nova_hw base.py)
        cmd = torch.tanh(a[:, 0:3]) * torch.tensor([R.BASE_VMAX, R.BASE_VMAX, R.BASE_WMAX], device=self.device) * (1 - self.upper[:, None])
        sp = cmd[:, :2].norm(dim=-1, keepdim=True)
        cmd = torch.cat([cmd[:, :2] * (self.gov[:, 1:2] / sp.clamp_min(1e-6)).clamp(max=1.0),
                         torch.max(torch.min(cmd[:, 2:3], self.gov[:, 2:3]), -self.gov[:, 2:3])], -1)
        self.tw_f = self.tw_f + (1 - math.exp(-dt / self.cfg.twist_tau)) * (cmd - self.tw_f)
        vxy = R.base_ramp(self.v_cmd_b[:, :2], self.tw_f[:, :2], self.gov[:, 0:1], self.gov[:, 3:4], dt, torch)
        dw = (self.tw_f[:, 2] - self.v_cmd_b[:, 2]).clamp(-R.BASE_ALPHA * dt, R.BASE_ALPHA * dt)
        self.v_cmd_b = torch.cat([vxy, (self.v_cmd_b[:, 2] + dw)[:, None]], -1)
        self.v_ref_prev = self.v_ref_b.clone(); self.steer_prev = self.steer.clone()
        self.v_ref_b = self.swerve(self.v_cmd_b, dt)
        # lift: rate -> target, both stages equal, held off the stops
        rate = torch.tanh(a[:, 3]) * R.HEADROOM * R.LIFT_VLIM * (1 - self.upper)
        self.lift_tgt = (self.lift_tgt + 0.5 * rate * dt).clamp(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)
        # 18 position joints: target -> window clamp -> 2nd-order filter -> NOVA's setpoint slew
        q = self.q_ready[self.i_pos][None] + self.cfg.action_scale * a[:, 4:]
        span = ((self.q_hi - self.q_lo)[self.i_pos] * 0.04).clamp_min(0.08)
        q = torch.max(torch.min(q, self.q_hi[self.i_pos] - span), self.q_lo[self.i_pos] + span)
        qf_pp = self.q_f_prev; self.q_f_prev = self.q_f.clone()
        self.q_f1 = self.q_f1 + self.beta * (q - self.q_f1)
        self.q_f = self.q_f + torch.max(torch.min(self.beta * (self.q_f1 - self.q_f), self.slew * dt), -self.slew * dt)
        self.qf_acc = (self.q_f - 2 * self.q_f_prev + qf_pp) / (dt * dt)
        self.tau_g = self.gravity_ff(); self.f_lift = self.lift_ff()
        # nova_hw shaper (wn 40, wrist 80) in 5 ms sub-steps; neck pitch / roll are Feetech servos (trapezoid profile)
        wn, h, sv = self.hw_wn, 0.005, self.i_servo
        for _ in range(int(round(dt / h))):
            spv = self.sp_v + h * (wn * wn * (self.q_f - self.sp) - 2.0 * wn * self.sp_v)
            e = self.q_f[:, sv] - self.sp[:, sv]; v0 = self.sp_v[:, sv]
            vs = torch.sign(e) * torch.sqrt(2 * R.NECK_SERVO["acc"] * e.abs()).clamp(max=R.NECK_SERVO["speed"])
            spv[:, sv] = v0 + (vs - v0).clamp(-R.NECK_SERVO["acc"] * h, R.NECK_SERVO["acc"] * h)
            self.sp_v = spv; self.sp = self.sp + h * self.sp_v
        self.v_ff = self.sp_v.clone(); self.v_ff[:, sv] = 0.0
        self.q_tgt[:, self.i_lift[0]] = self.lift_tgt; self.q_tgt[:, self.i_lift[1]] = self.lift_tgt

    def apply_action(self):
        """Every physics step: NOVA's MIT frame held between ticks + the swerve drive as a force at the floor."""
        self.q_tgt[:, self.i_pos] = self.sp
        self.robot.set_joint_position_target_index(self.q_tgt, self.jid)
        self.v_tgt[:, self.i_pos] = self.v_ff
        self.robot.set_joint_velocity_target_index(self.v_tgt, self.jid)
        eff = torch.zeros_like(self.q_tgt); eff[:, self.i_lift] = self.f_lift; eff[:, self.i_pos] = self.tau_g
        self.robot.set_joint_effort_target_index(eff, self.jid)
        _, yaw, _ = self.base_pose()
        c, s = torch.cos(yaw), torch.sin(yaw); vb = self.v_ref_b
        v_ref_w = torch.stack([c * vb[:, 0] - s * vb[:, 1], s * vb[:, 0] + c * vb[:, 1]], -1)
        v = self.robot.data.root_com_lin_vel_w.torch[:, :2]; w = self.robot.data.root_com_ang_vel_w.torch[:, 2]
        F = self.m_tot[:, None] * self.kv[:, None] * (v_ref_w - v)
        F = F * torch.clamp(0.8 * G * self.m_tot[:, None] / F.norm(dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
        F3 = torch.cat([F, torch.zeros(self.num_envs, 1, device=self.device)], -1)
        com = self.robot.data.body_com_pos_w.torch[:, 0]
        floor = torch.cat([com[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], -1)
        tau = torch.cross(floor - com, F3, dim=-1); tau[:, 2] += 4.0 * self.kw * (vb[:, 2] - w)
        push = self.push_f * (self.push_t > 0).float()[:, None] * float(self.cfg.dr_physics)
        self.robot.wrench(torch.stack([F3, push], 1), torch.stack([tau, torch.zeros_like(push)], 1), [0, self.torso_body])

    # ------------------------------------------------------------ observations (computed once per step)
    def urr(self, t, pos, yaw, body):
        """Dyna's task-space targets in the base yaw frame (38): head pose 9, wrist poses 18, elbows 6, chest 3
        (lean, heading cos / sin), footprint 2. body = 0: the clip has no elbow / torso labels -> zeros there."""
        Rb = rot_z(yaw).transpose(-1, -2)
        loc = lambda p: (Rb @ (p - pos)[..., None])[..., 0]
        by = wrap(t["body_yaw"] - yaw)
        foot = t["sm"][:, :2] - torch.stack([torch.cos(t["body_yaw"]), torch.sin(t["body_yaw"])], -1) * 0.096
        b = body[:, None]
        return torch.cat([loc(t["head_p"]), six_d(Rb @ t["head_R"]), loc(t["lw_p"]), six_d(Rb @ t["lw_R"]), loc(t["rw_p"]), six_d(Rb @ t["rw_R"]),
                          loc(t["le"]) * b, loc(t["re"]) * b, t["lean"][:, None] * b, torch.stack([torch.cos(by), torch.sin(by)], -1),
                          loc(torch.cat([foot, t["sm"][:, 2:]], -1))[:, :2]], -1)

    def observations(self):
        pos, yaw, _ = self.base_pose()
        body = self.c_body[self.c_id].float()
        tob = self.target_obs(pos, yaw, noisy=True)
        self.hist = torch.cat([tob[:, None], self.hist[:, :-1]], 1)
        cp, cR = self.tracked_now()
        Ryt = rot_z(yaw).transpose(-1, -2)[:, None]
        cur = torch.cat([((Ryt @ (cp - pos[:, None])[..., None])[..., 0]).reshape(self.num_envs, 9), six_d(Ryt @ cR).reshape(self.num_envs, 18)], -1)
        q = self.robot.data.joint_pos.torch[:, self.jid]; qd = self.robot.data.joint_vel.torch[:, self.jid]
        lin = self.robot.data.root_link_lin_vel_w.torch; ang = self.robot.data.root_link_ang_vel_w.torch
        c, s = torch.cos(yaw), torch.sin(yaw)
        vb = torch.stack([c * lin[:, 0] + s * lin[:, 1], -s * lin[:, 0] + c * lin[:, 1], ang[:, 2]], -1)
        n = self.cfg.noise
        upd = (torch.rand(self.num_envs, device=self.device) < 0.4) | (self.episode_length_buf <= 1)   # 20 Hz drive feedback
        self.vb_obs = torch.where(upd[:, None], vb + torch.randn_like(vb) * n["base"], self.vb_obs)
        prop = torch.cat([cur + torch.randn_like(cur) * n["cur"], q - self.q_ready + torch.randn_like(q) * n["q"],
                          0.1 * (qd + torch.randn_like(qd) * n["qd"]), self.vb_obs, self.act_obs()], -1)          # 92
        mode = torch.stack([self.upper, body], -1)
        tgt = torch.cat([self.urr(self.tgt, pos, yaw, body)] + [self.urr(self.compute_targets(self.phase + k * self.rate, step_aug=False), pos, yaw, body)
                                                               for k in FUT[1:]] + [mode], -1)                     # 154
        vr = torch.cat([self.hist[:, 0], self.hist[:, 2], self.hist[:, 5], self.age.clamp(max=1.0), self.upper[:, None]], -1)   # 85
        el = self.robot.data.body_link_pos_w.torch[:, self.elbow_body]
        loc = lambda p: (rot_z(yaw).transpose(-1, -2) @ (p - pos)[..., None])[..., 0]
        clean = torch.cat([cur, q - self.q_ready, 0.1 * qd, vb, self.act_obs()], -1)
        priv = torch.cat([self.robot.data.projected_gravity_b.torch, self.robot.data.root_link_ang_vel_b.torch, lin[:, 2:3],
                          self.zmp_margin[:, None] * 10, self.clear[:, None] * 10, loc(el[:, 0]), loc(el[:, 1]),
                          self.lift_tgt[:, None] * 5, self.kv[:, None] / 10, self.delay[:, None].float(),
                          (self.masses[:, self.hand_bodies] - self.mass0[self.hand_bodies]), self.c_fail[self.c_id][:, None]], -1)   # 21
        self.obs = {"policy": torch.cat([vr, prop], -1), "tgt": tgt, "prop": prop, "vr": vr,
                    "critic": torch.cat([tgt, vr, clean, priv, self.target_obs(pos, yaw, noisy=False)], -1)}
        self.env.extras.setdefault("log", {}).update(self.log)       # metrics survive mjlab's per-reset log reset

    # ------------------------------------------------------------ rewards (computed once per step)
    def rewards(self):
        w, t = self.cfg.w, self.tgt
        pos, yaw, _ = self.base_pose()
        cp, cR = self.tracked_now()
        K = lambda e, sg: torch.exp(-(e / sg) ** 2)
        e_h = (cp[:, 0] - t["head_p"]).norm(dim=-1)
        e_w = torch.stack([(cp[:, 1] - t["lw_p"]).norm(dim=-1), (cp[:, 2] - t["rw_p"]).norm(dim=-1)], -1)
        a_h = rot_angle(cR[:, 0], t["head_R"])
        a_w = torch.stack([rot_angle(cR[:, 1], t["lw_R"]), rot_angle(cR[:, 2], t["rw_R"])], -1)
        rot_ok = self.c_ego[self.c_id].float(); body = self.c_body[self.c_id].float()
        bp = self.robot.data.body_link_pos_w.torch
        e_el = torch.stack([(bp[:, self.elbow_body[0]] - t["le"]).norm(dim=-1), (bp[:, self.elbow_body[1]] - t["re"]).norm(dim=-1)], -1)
        r = {}
        # SONIC-style tracking: three scales each, so a far-off frame still pulls (k5: sigma 12 cm alone was flat at 30 cm)
        r["wrist_p"] = ((K(e_w, 0.02) + K(e_w, 0.06) + K(e_w, 0.20)) / 3).mean(-1)
        r["wrist_r"] = ((K(a_w, 0.05) + K(a_w, 0.15) + K(a_w, 0.5)) / 3).mean(-1) * rot_ok
        r["elbow"] = K(e_el, 0.10).mean(-1) * body
        r["head_p"] = (K(e_h, 0.05) + K(e_h, 0.25)) / 2
        r["head_r"] = (K(a_h, 0.1) + K(a_h, 0.4)) / 2
        q = self.robot.data.joint_pos.torch[:, self.jid]
        lean_t = (-t["lean"]).clamp(-1.0, 0.15)
        r["lean"] = K(q[:, self.i_torso] - lean_t, 0.15) * body
        up = self.upper > 0; one = torch.ones_like(e_h)
        r["body_yaw"] = torch.where(up, one, K(wrap(t["body_yaw"] - yaw), 0.35))
        foot = t["sm"][:, :2] - torch.stack([torch.cos(t["body_yaw"]), torch.sin(t["body_yaw"])], -1) * 0.096
        e_f = (pos[:, :2] - foot).norm(dim=-1)
        r["foot"] = torch.where(up, one, (K(e_f, 0.10) + K(e_f, 0.40)) / 2)
        hv = self.robot.data.body_com_lin_vel_w.torch[:, self.trk_body[1:]]
        r["hand_vel"] = K((hv - self.tgt_v).norm(dim=-1), 0.3).mean(-1)
        r["track_lin"] = (e_h + e_w.mean(-1) + torch.where(up, torch.zeros_like(e_f), e_f)).clamp(max=3.0)
        r["base_acc"] = ((self.v_ref_b - self.v_ref_prev) / self.step_dt).square().sum(-1)
        r["steer"] = (wrap(self.steer - self.steer_prev) / self.step_dt).square().sum(-1).clamp(max=200.0)
        lr = torch.tanh(self.actions[:, 3]).abs(); r["lift_use"] = lr + lr.square()
        r["act_sat"] = ((self.raw_act - self.a_hi).clamp_min(0) + (self.a_lo - self.raw_act).clamp_min(0)).sum(-1)
        r["action_rate"] = (self.actions - self.prev_actions).square().sum(-1)
        r["action_acc"] = (self.actions - 2 * self.prev_actions + self.prev2_actions).square().sum(-1)
        r["qf_acc"] = self.qf_acc.square().sum(-1).clamp(max=2e5)
        self.hand_acc = (hv - self.hand_v_prev) / self.step_dt; self.hand_v_prev = hv.clone()
        r["hand_acc"] = self.hand_acc.square().sum((-1, -2)).clamp(max=5000.0)
        pw = self.robot.data.body_link_ang_vel_w.torch[:, self.trk_body[1:]]
        self.palm_acc = (pw - self.palm_w_prev) / self.step_dt; self.palm_w_prev = pw.clone()
        r["palm_acc"] = self.palm_acc.square().sum((-1, -2)).clamp(max=2e4)
        qd = self.robot.data.joint_vel.torch[:, self.jid]
        qdd = (qd - self.qd_prev) / self.step_dt; self.qd_prev = qd.clone()
        r["joint_acc"] = qdd.square().sum(-1)
        span = self.q_hi - self.q_lo
        r["joint_limit"] = ((self.q_lo + 0.08 * span - q).clamp_min(0) + (q - self.q_hi + 0.08 * span).clamp_min(0)).sum(-1)
        r["past_stop"] = self.lim_joint[:, 2].float().sum(-1) * (self.episode_length_buf > 10)
        ip = self.i_pos
        bar = lambda x: ((x - 0.85) / 0.15).clamp(0, 1.5).square().sum(-1)
        rms = (self.tau2.sqrt() / self.t_rated) * self.ovl_hard
        r["limits"] = bar(qd[:, ip].abs() / self.v_lim) + bar(self.tau.abs() / self.t_pk) + bar(rms) + bar(qd[:, self.i_lift].abs() / (0.5 * R.LIFT_VLIM))
        r["torque"] = ((self.tau / self.t_rated).square() * self.ovl_hard).sum(-1)
        g = self.robot.data.projected_gravity_b.torch
        r["tilt"] = g[:, :2].square().sum(-1)
        r["tilt_rate"] = self.robot.data.root_link_ang_vel_b.torch[:, :2].square().sum(-1)
        r["com_acc"] = self.com_acc[:, :2].square().sum(-1).clamp(max=25.0)
        r["zmp"] = ((self.cfg.zmp_reserve + 0.02 - self.zmp_margin) / 0.05).clamp(0, 3) ** 2
        band = lambda d, b: ((b - d) / (b - self.cfg.collide_margin)).clamp(0, 1.5)
        r["clearance"] = torch.maximum(band(self.clear_mv, 0.04), band(self.clear_sh, 0.02))
        r["terminated"] = self.reset_terminated.float()
        r["collide"] = self.last_term[:, 1].float()
        r["alive"] = one
        # s2: penalties ramp 10 % -> 100 % over pen_ramp iterations (safety terms and the linear pull always full)
        ramp = min(1.0, 0.1 + 0.9 * self.common_step_counter / (self.cfg.pen_ramp * 24))
        self.r = {k: (ramp * r[k] if w[k] < 0 and k not in self.cfg.SAFETY_TERMS and k != "track_lin" else r[k]) for k in w}
        L = self.log
        L["Sample/pen_ramp"] = ramp
        L["Err/head_cm"] = (e_h.mean() * 100).item(); L["Err/wrist_cm"] = (e_w.mean() * 100).item()
        L["Err/elbow_cm"] = ((e_el.mean(-1) * body).sum() / body.sum().clamp_min(1) * 100).item()
        L["Err/head_deg"] = torch.rad2deg(a_h).mean().item()
        L["Err/wrist_deg"] = ((torch.rad2deg(a_w.mean(-1)) * rot_ok).sum() / rot_ok.sum().clamp_min(1)).item()
        L["Err/foot_cm"] = (e_f.mean() * 100).item()
        L["Safety/clear_cm"] = (self.clear.mean() * 100).item(); L["Safety/zmp_margin_cm"] = (self.zmp_margin.mean() * 100).item()
        L["Smooth/hand_acc"] = self.hand_acc.norm(dim=-1).mean().item(); L["Base/speed"] = self.v_ref_b[:, :2].norm(dim=-1).mean().item()
        L["Sample/fail_mean"] = self.c_fail.mean().item(); L["Sample/fail_max"] = self.c_fail.max().item()
        self.ep_err += torch.stack([e_h, e_w.mean(-1), a_h, a_w.mean(-1)], -1); self.ep_n += 1
        self.r_last = dict(e_h=e_h, e_w=e_w, a_h=a_h, a_w=a_w, e_el=e_el, e_f=e_f, rot_ok=rot_ok, body=body)

    # ------------------------------------------------------------ stops (computed once per step, before rewards)
    def dones(self):
        self._update_state()
        g = self.robot.data.projected_gravity_b.torch
        out = (self.zmp_margin < self.cfg.zmp_out) & (self.push_t <= 0)
        self.zmp_out_t = torch.where(out, self.zmp_out_t + self.step_dt, torch.zeros_like(self.zmp_out_t))
        tilt = (torch.asin(g[:, :2].norm(dim=-1).clamp(max=1.0)) > self.cfg.tilt_max) | (self.zmp_out_t >= self.cfg.zmp_out_s - 1e-6)
        collide = self.clear < self.cfg.collide_margin
        live = self.episode_length_buf > 10
        limit = self.lim_kind[:, [0, 2]].any(-1) & live
        pos, _, _ = self.base_pose(); cp, _ = self.tracked_now(); t = self.tgt
        foot = t["sm"][:, :2] - torch.stack([torch.cos(t["body_yaw"]), torch.sin(t["body_yaw"])], -1) * 0.096
        far = live & (((cp[:, 0] - t["head_p"]).norm(dim=-1) > self.cfg.far) | (((pos[:, :2] - foot).norm(dim=-1) > self.cfg.far) & (self.upper == 0)))
        self.last_term = torch.stack([tilt, collide, far, limit], -1)
        for k, nm in enumerate(("tilt", "collide", "far", "limit")): self.log[f"Term/{nm}"] = self.last_term[:, k].float().mean().item()

    # ------------------------------------------------------------ reset (an mjlab reset event, after the DR events)
    def reset_state(self, env_ids):
        n, dev = len(env_ids), self.device
        if n == 0: return
        # adaptive sampling: the finished episodes score their clips (stopped, or wrist off by > 10 cm on average)
        done_n = self.ep_n[env_ids]; seen = done_n > 25
        if seen.any():
            ids = env_ids[seen]
            bad = (self.reset_terminated[ids] | (self.ep_err[ids, 1] / done_n[seen] > 0.10)).float()
            c = self.c_id[ids]
            self.c_fail.index_put_((c,), 0.8 * self.c_fail[c] + 0.2 * bad)
            self.resample_weights()
        self.upper[env_ids] = (torch.rand(n, device=dev) < self.cfg.upper_share).float()
        up = self.upper[env_ids] > 0
        c = torch.multinomial(self.c_p, n, replacement=True)
        if up.any(): c[up] = torch.multinomial(self.c_p_up, int(up.sum()), replacement=True)
        L = self.c_len[c].float()
        start = torch.rand(n, device=dev) * (L - 2.0 * 50).clamp_min(0) * 0.8
        self.c_id[env_ids] = c; self.rate[env_ids] = torch.empty(n, device=dev).uniform_(*self.cfg.speed_range); self.phase[env_ids] = start
        self.aug_on[env_ids] = ((torch.rand(n, device=dev) < self.cfg.twist_aug) & ~up).float()
        self.neck_on[env_ids] = (torch.rand(n, device=dev) < self.cfg.neck_aug).float()
        for b in (self.aug_yaw, self.aug_w, self.aug_xy, self.aug_v, self.neck_off, self.neck_tgt): b[env_ids] = 0
        lift = torch.empty(n, device=dev).uniform_(0.25, 0.75) * R.LIFT_STROKE
        lift[up] = torch.empty(int(up.sum()), device=dev).uniform_(R.LIFT_MARGIN, R.LIFT_STROKE - R.LIFT_MARGIN)
        q = self.q_ready[None].repeat(n, 1)
        q[:, self.i_lift[0]] = lift; q[:, self.i_lift[1]] = lift
        q[:, self.i_arm] += torch.randn(n, len(self.i_arm), device=dev) * 0.05
        self.robot.write_joint_position_to_sim_index(q, self.jid, env_ids)
        self.robot.write_joint_velocity_to_sim_index(torch.zeros_like(q), self.jid, env_ids)
        yaw0 = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
        pos0 = self.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, R.CHASSIS_Z + 0.002], device=dev)
        self.robot.write_root_pose(pos0, yaw0, env_ids)
        i0 = (self.c_start[c].float() + start).long()
        hR = self.m["head_R"][i0]; h_yaw = torch.atan2(hR[:, 1, 0], hR[:, 0, 0])
        self.anc_yaw[env_ids] = yaw0 - h_yaw; self.anc_h[env_ids] = self.m["head_p"][i0]
        sl, sr = ML.shoulders_from_head(self.m["head_p"][i0], h_yaw, torch); sm = 0.5 * (sl + sr)
        self.sm0[env_ids] = sm[:, :2]; self.sm_lp[env_ids] = sm[:, :2]
        off = (rot_z(yaw0 - h_yaw) @ (self.m["head_p"][i0] - sm)[..., None])[..., 0]
        fwd = torch.stack([torch.cos(yaw0), torch.sin(yaw0)], -1)
        anc_r = torch.zeros(n, 3, device=dev)
        anc_r[:, :2] = pos0[:, :2] + fwd * 0.096 + off[:, :2]
        anc_r[:, 2] = 1.374 + 2 * lift + torch.randn(n, device=dev) * 0.03
        self.anc_r[env_ids] = anc_r
        self.lift_tgt[env_ids] = lift; self.q_tgt[env_ids] = q; self.v_tgt[env_ids] = 0
        for b in (self.v_ref_b, self.v_ref_prev, self.actions, self.prev_actions, self.prev2_actions, self.act_buf, self.com_acc, self.ep_err, self.ep_n): b[env_ids] = 0
        self.q_f[env_ids] = q[:, self.i_pos]; self.q_f_prev[env_ids] = q[:, self.i_pos]; self.q_f1[env_ids] = q[:, self.i_pos]
        self.sp[env_ids] = q[:, self.i_pos]; self.sp_v[env_ids] = 0; self.v_ff[env_ids] = 0
        for b in (self.hand_v_prev, self.hand_acc, self.tgt_v, self.palm_w_prev, self.tw_f, self.v_cmd_b, self.steer, self.steer_cmd, self.steer_prev, self.tau, self.qd_prev):
            b[env_ids] = 0
        self.tau2[env_ids] = self.tau_g[env_ids].square()
        self.steer_k[env_ids] = torch.empty(n, device=dev).uniform_(0.8, 1.2)
        self.delay[env_ids] = torch.randint(self.cfg.action_delay[0], self.cfg.action_delay[1] + 1, (n,), device=dev)
        self.kv[env_ids] = torch.empty(n, device=dev).uniform_(6.0, 15.0)
        self.push_t[env_ids] = 0.0; self.zmp_out_t[env_ids] = 0.0
        # Dyna DR: PD gains here (masses + inertia, damping, friction, armature, payload: the mjlab events before this one)
        J = len(R.ACTUATED)
        ks = torch.empty(n, J, device=dev).uniform_(0.75, 1.25) if self.cfg.dr_physics else torch.ones(n, J, device=dev)
        kd = ks.sqrt() * (torch.empty_like(ks).uniform_(0.8, 1.25) if self.cfg.dr_physics else 1.0)
        self.robot.set_gains(self.kp0[None] * ks, self.kd0[None] * kd, self.jid.tolist(), env_ids)
        self.kp_env[env_ids] = (self.kp0[None] * ks)[:, self.i_pos]; self.kd_env[env_ids] = (self.kd0[None] * kd)[:, self.i_pos]
        m = self.robot.data.body_mass.torch[env_ids]
        self.masses[env_ids] = m; self.m_tot[env_ids] = m.sum(-1)
        if self.cfg.dr_physics: self.gc_scale[env_ids] = torch.empty(n, len(self.i_pos), device=dev).uniform_(0.8, 1.2)
        self.kw[env_ids] = torch.empty(n, device=dev).uniform_(5.0, 12.0)
        self.drive_k[env_ids] = torch.empty(n, 3, device=dev).uniform_(0.85, 1.15)
        self.coast[env_ids] = torch.rand(n, device=dev) > 0.1
        self.gov[env_ids] = torch.tensor([R.BASE_ACCEL, R.BASE_VMAX, R.BASE_WMAX, R.BASE_ACCEL * R.BASE_DECEL], device=dev)
        self.com_v_prev[env_ids] = 0
        self.need_glide[env_ids] = True; self.blend_on[env_ids] = 0.0
        self.tgt = self.compute_targets(self.phase, step_aug=False)
        self.stream_reset(env_ids)
        # the written pose (derived quantities are stale until mjlab's forward())
        pos = self.robot.data.root_link_pos_w.torch.clone(); pos[env_ids] = pos0
        yaw = self.base_pose()[1].clone(); yaw[env_ids] = yaw0
        self.hist[env_ids] = self.target_obs(pos, yaw, noisy=False)[env_ids, None].repeat(1, self.hist.shape[1], 1)


# ---------------------------------------------------------------- mjlab managers: thin views onto SonicCore
@dataclass(kw_only=True)
class NovaActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    sonic: SonicCfg = field(default_factory=SonicCfg)

    def build(self, env):
        return NovaAction(self, env)


class NovaAction(ActionTerm):
    """22 actions: base twist 3, lift rate 1, 18 position joints -> NOVA's command chain (SonicCore.pre_physics)."""
    def __init__(self, cfg: NovaActionCfg, env):
        super().__init__(cfg, env)
        env.nova = SonicCore(env, cfg.sonic)
        self._raw = torch.zeros(env.num_envs, 22, device=env.device)

    action_dim = property(lambda s: 22)
    raw_action = property(lambda s: s._raw)

    def process_actions(self, actions):
        self._raw = actions; self._env.nova.pre_physics(actions)

    def apply_actions(self):
        self._env.nova.apply_action()


def _obs(env, group):
    core = env.nova
    if group == "tgt": core.observations()         # "tgt" is the first group: compute every group once per step
    return core.obs[group]


def _rew(env, term):
    core = env.nova
    if core._k_rew != env.common_step_counter: core.rewards(); core._k_rew = env.common_step_counter
    return core.r[term]


def _stop(env, kind):
    core = env.nova
    if core._k_done != env.common_step_counter: core.dones(); core._k_done = env.common_step_counter
    return core.last_term[:, ("tilt", "collide", "far", "limit").index(kind)]


def _reset(env, env_ids):
    env.nova.reset_state(env_ids)


def make_env_cfg(n=4096, sonic: SonicCfg | None = None, play=False) -> ManagerBasedRlEnvCfg:
    sc = sonic or SonicCfg()
    groups = ("tgt", "prop", "vr", "policy", "critic")
    obs = {g: ObservationGroupCfg(terms={g: ObservationTermCfg(func=_obs, params={"group": g})}, concatenate_terms=True,
                                  enable_corruption=False) for g in groups}
    rew = {k: RewardTermCfg(func=_rew, weight=v, params={"term": k}) for k, v in sc.w.items()}
    stops = {k: TerminationTermCfg(func=_stop, params={"kind": k}) for k in ("tilt", "collide", "far", "limit")}
    stops["time_out"] = TerminationTermCfg(func=time_out, time_out=True)
    robot = SceneEntityCfg("robot")
    ev = {}
    if sc.dr_physics:                                 # Dyna: link masses + inertia, joint damping / friction / armature, payload
        ev["mass"] = EventTermCfg(func=dr.pseudo_inertia, mode="reset", params={"alpha_range": (math.log(0.85) / 2, math.log(1.15) / 2), "asset_cfg": robot})
        ev["payload"] = EventTermCfg(func=dr.body_mass, mode="reset", params={"ranges": sc.payload, "operation": "add",
                                                                               "asset_cfg": SceneEntityCfg("robot", body_names=["l_hand_mount", "r_hand_mount"])})
        arms = lambda: SceneEntityCfg("robot", joint_names=["torso"] + R.NECK + R.ARM)
        ev["damping"] = EventTermCfg(func=dr.joint_damping, mode="reset", params={"ranges": (0.0, 0.3), "operation": "abs", "asset_cfg": arms()})
        ev["friction"] = EventTermCfg(func=dr.joint_friction, mode="reset", params={"ranges": (0.0, 0.2), "operation": "abs", "asset_cfg": arms()})
        ev["armature"] = EventTermCfg(func=dr.joint_armature, mode="reset", params={"ranges": (0.8, 1.25), "operation": "scale", "asset_cfg": arms()})
    ev["nova_reset"] = EventTermCfg(func=_reset, mode="reset")
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": nova_entity()}, num_envs=n, env_spacing=3.0),
        observations=obs, actions={"nova": NovaActionCfg(sonic=sc)}, events=ev, rewards=rew, terminations=stops,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="mabel_chassis",
                            distance=7.0, elevation=-22.0, azimuth=135.0, max_extra_envs=8, width=960, height=540),
        sim=SimulationCfg(nconmax=6, njmax=30, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20, integrator="implicitfast")),
        decimation=4, episode_length_s=20.0)


# ---------------------------------------------------------------- SONIC token model (rsl_rl actor) + PPO with CAPS
class TokenNet(nn.Module):
    """encoder(targets | proprio) -> FSQ token -> decoder(token, proprio). x = [targets..., proprio (last n_prop)]."""
    Z, LEVELS = 32, 8

    def __init__(self, n_in, n_prop, n_out, enc=(512, 256), dec=(512, 384, 256), act="elu"):
        super().__init__()
        self.n_prop = n_prop
        self.enc = MLP(n_in, self.Z, list(enc), act)
        self.dec = MLP(self.Z + n_prop, n_out, list(dec), act)

    def encode(self, x):
        """Pre-quantisation token in (-1, 1) (what the student is distilled onto)."""
        return torch.tanh(self.enc(x))

    def quantise(self, z):
        h = (self.LEVELS - 1) / 2
        return z + ((torch.round(z * h) / h) - z).detach()      # FSQ, straight-through

    def forward(self, x):
        return self.dec(torch.cat([self.quantise(self.encode(x)), x[..., -self.n_prop:]], -1))

    def __getitem__(self, i):                                    # distributions index the last layers (mlp[-2])
        return self.dec[i]


class TokenModel(MLPModel):
    """MLPModel with TokenNet in place of the MLP; the LAST obs group must be "prop"."""
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(512, 384, 256), activation="elu",
                 obs_normalization=False, distribution_cfg=None):
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        assert self.obs_groups[-1] == "prop", self.obs_groups
        out = self.mlp[-1].out_features
        self.mlp = TokenNet(self.obs_dim, obs["prop"].shape[-1], out, dec=hidden_dims, act=activation)
        if self.distribution is not None: self.distribution.init_mlp_weights(self.mlp)

    def token(self, obs):
        return self.mlp.encode(self.get_latent(obs))


class CapsPPO(PPO):
    """PPO + a temporal smoothness loss on the policy's MEAN action (CAPS, Mysore et al. 2021; tasks.py's, for v21's
    20 ms command flips): after each update, CAPS_STEPS gradient steps on |mu(o_t) - mu(o_t+1)|^2."""
    CAPS_COEF, CAPS_STEPS, CAPS_PAIRS = 0.5, 2, 16384

    def update(self):
        st = self.storage
        obs, done = st.observations, st.dones[:-1, :, 0].bool()
        stats = super().update()
        T, N = done.shape
        ok = (~done).flatten().nonzero()[:, 0]
        tot = 0.0
        for _ in range(self.CAPS_STEPS):
            k = ok[torch.randint(len(ok), (min(self.CAPS_PAIRS, len(ok)),), device=self.device)]
            t, n = k // N, k % N
            a0 = self.actor(obs[t, n]); a1 = self.actor(obs[t + 1, n])
            w = torch.ones(a0.shape[-1], device=self.device); w[:4] = 0.3
            loss = self.CAPS_COEF * ((a0 - a1).square() * w).sum(-1).mean()
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step(); tot += loss.item()
        stats["caps"] = tot / self.CAPS_STEPS
        return stats


def rl_cfg(iters=6000, run="") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        max_iterations=iters, save_interval=250, experiment_name="nova_sonic_mj", run_name=run, logger="tensorboard",
        obs_groups={"actor": ("tgt", "prop"), "critic": ("critic",)},
        actor=RslRlModelCfg(class_name="sonic:TokenModel", hidden_dims=(512, 384, 256), obs_normalization=True,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(768, 512, 256), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(class_name="sonic:CapsPPO", learning_rate=5e-4, entropy_coef=0.003, num_learning_epochs=5,
                                       num_mini_batches=4, schedule="adaptive", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0))


def register(n=4096, iters=6000, run=""):
    from mjlab.tasks.registry import _REGISTRY, register_mjlab_task
    _REGISTRY.pop(TASK, None)
    register_mjlab_task(TASK, make_env_cfg(n), make_env_cfg(16, play=True), rl_cfg(iters, run))


# ---------------------------------------------------------------- train / distil / eval / view
def make_env(n, device="cuda:0", faults=True, clips="mix", seed=42, upper=None, render=False):
    from mjlab.rl import RslRlVecEnvWrapper
    sc = SonicCfg(faults=faults)
    if upper is not None: sc.upper_share = upper
    if clips == "egodex": sc.egodex_share, sc.avp_share = 1.0, 0.0
    elif clips == "lafan": sc.egodex_share, sc.avp_share = 0.0, 0.0
    elif clips == "avp": sc.avp_share, sc.held_only = 1.0, True
    c = make_env_cfg(n, sc); c.seed = seed
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(c, device=device, render_mode="rgb_array" if render else None))


def train(a):
    from mjlab.scripts.train import TrainConfig, launch_training
    register(a.envs, a.iters, a.run)
    agent = rl_cfg(a.iters, a.run); agent.seed = a.seed
    os.chdir(HERE)                                     # logs/rsl_rl/nova_sonic_mj/<time>_<run>
    launch_training(TASK, TrainConfig(env=make_env_cfg(a.envs), agent=agent, video=a.video, video_length=300, video_interval=2000))


def load_actor(env, pt, device):
    from dataclasses import asdict
    from mjlab.rl import MjlabOnPolicyRunner
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg(1)), None, device)
    runner.load(pt, map_location=device)
    return runner.alg.actor


def distil(a):
    """SONIC's second encoder: the student encodes the Vision Pro stream onto the teacher's token, through the teacher's
    FROZEN decoder (DAgger: the student drives, the teacher labels). loss = |a_s - a_t|^2 + LAT |z_s - z_t|^2."""
    from tensordict import TensorDict
    from torch.utils.tensorboard import SummaryWriter
    log_dir = os.path.join(HERE, "logs", "rsl_rl", "nova_sonic_mj", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{a.run}")
    os.makedirs(log_dir, exist_ok=True); tb = SummaryWriter(log_dir)
    env = make_env(a.envs, a.device, seed=a.seed)
    teacher = load_actor(env, a.teacher, a.device).eval()
    obs = env.get_observations().to(a.device)
    student = TokenModel(obs, {"actor": ["vr", "prop"]}, "actor", 22, obs_normalization=True).to(a.device)
    student.mlp.dec.load_state_dict(teacher.mlp.dec.state_dict())
    for p in student.mlp.dec.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=5e-4)
    LAT, STEPS, EPOCHS, BATCH = 1.0, 24, 4, 16384
    core = env.unwrapped.nova
    for it in range(a.iters):
        beta = max(0.0, 1.0 - it / max(1, a.iters // 5))         # the teacher drives at first, then the student alone
        buf = []
        with torch.no_grad():
            for _ in range(STEPS):
                a_t = teacher(obs); z_t = teacher.token(obs)
                student.update_normalization(obs)
                a_s = student(obs)
                use_t = (torch.rand(env.num_envs, 1, device=a.device) < beta).float()
                buf.append((obs["vr"].clone(), obs["prop"].clone(), a_t, z_t))
                obs, _, _, _ = env.step(use_t * a_t + (1 - use_t) * a_s)
                obs = obs.to(a.device)
        vr, pr, at, zt = (torch.cat([b[i] for b in buf]) for i in range(4))
        tot = 0.0
        for _ in range(EPOCHS):
            perm = torch.randperm(len(vr), device=a.device)
            for k in range(0, len(vr), BATCH):
                j = perm[k:k + BATCH]
                o = TensorDict({"vr": vr[j], "prop": pr[j]}, batch_size=[len(j)])
                loss = (student(o) - at[j]).square().mean() + LAT * (student.token(o) - zt[j]).square().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
                tot += loss.item()
        tb.add_scalar("Distil/loss", tot, it); tb.add_scalar("Distil/beta", beta, it)
        for k in ("Err/wrist_cm", "Err/wrist_deg", "Term/collide", "Term/tilt", "Term/far"):
            if k in core.log: tb.add_scalar(k, float(core.log[k]), it)
        if it % 50 == 0 or it == a.iters - 1:
            print(f"[distil] it {it} loss {tot:.4f} beta {beta:.2f} wrist {core.log.get('Err/wrist_cm', 0):.1f} cm "
                  f"{core.log.get('Err/wrist_deg', 0):.1f} deg", flush=True)
        if it % 250 == 0 or it == a.iters - 1:
            torch.save({"student": student.state_dict(), "teacher": a.teacher, "iter": it}, os.path.join(log_dir, f"student_{it}.pt"))


@torch.inference_mode()
def evaluate(a):
    env = make_env(a.envs, a.device, faults=not a.no_faults, clips=a.clips, seed=a.seed, upper=0.0)
    obs = env.get_observations().to(a.device)
    if a.student:
        ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
        pol = TokenModel(obs, {"actor": ["vr", "prop"]}, "actor", 22, obs_normalization=True).to(a.device)
        pol.load_state_dict(ck["student"])
    else:
        pol = load_actor(env, a.checkpoint, a.device)
    pol.eval(); U = env.unwrapped.nova; N = U.num_envs; dt = U.step_dt
    acc = {k: [] for k in ("src", "e_w", "a_w", "e_h", "a_h", "e_el", "turn", "hacc", "base", "clear", "rot_ok", "body")}
    ev = {k: torch.zeros(3, device=a.device) for k in ("collide", "tilt", "far", "limit", "t")}
    tR_prev = None
    for k in range(int(a.seconds / dt)):
        obs, _, done, _ = env.step(pol(obs)); obs = obs.to(a.device)
        src = U.c_src[U.c_id]
        tR = torch.stack([U.tgt["lw_R"], U.tgt["rw_R"]], 1)
        turn = torch.zeros(N, 2, device=a.device) if tR_prev is None else torch.rad2deg(rot_angle(tR, tR_prev)) / dt
        tR_prev = tR
        for j in range(3):
            m = src == j
            for i, nm in ((1, "collide"), (0, "tilt"), (2, "far"), (3, "limit")): ev[nm][j] += (U.last_term[:, i] & m).sum()
            ev["t"][j] += m.sum() * dt
        L = (U.episode_length_buf > 75) & ~done.bool()
        if k < 10: continue
        rl = U.r_last
        for nm, v in (("src", src), ("e_w", rl["e_w"]), ("a_w", rl["a_w"]), ("e_h", rl["e_h"]), ("a_h", rl["a_h"]), ("e_el", rl["e_el"]),
                      ("turn", turn), ("hacc", U.hand_acc.norm(dim=-1)), ("base", U.v_ref_b[:, :2].norm(dim=-1)), ("clear", U.clear),
                      ("rot_ok", rl["rot_ok"]), ("body", rl["body"])):
            acc[nm].append(v[L])
    A = {k: torch.cat(v).float() for k, v in acc.items()}
    out = {"checkpoint": a.checkpoint, "student": a.student, "faults": not a.no_faults, "clips": a.clips, "envs": N, "seconds": a.seconds}
    for j, nm in enumerate(SRC):
        m = A["src"] == j
        if m.sum() == 0: continue
        d = dict(wrist_cm=A["e_w"][m].mean().item() * 100, wrist_cm_p95=A["e_w"][m].flatten().quantile(0.95).item() * 100,
                 head_cm=A["e_h"][m].mean().item() * 100, head_deg=math.degrees(A["a_h"][m].mean().item()),
                 hand_acc=A["hacc"][m].mean().item(), base_speed=A["base"][m].mean().item(), clear_cm_min=A["clear"][m].min().item() * 100,
                 **{f"{s}_per_min": (ev[s][j] / ev["t"][j] * 60).item() for s in ("collide", "tilt", "far", "limit")})
        if A["body"][m].sum() > 0: d["elbow_cm"] = A["e_el"][m][A["body"][m] > 0].mean().item() * 100
        if A["rot_ok"][m].sum() > 0:
            aw, tw = A["a_w"][m].flatten(), A["turn"][m].flatten()
            d["palm_deg"] = math.degrees(aw.mean().item()); d["palm_deg_p95"] = math.degrees(aw.quantile(0.95).item())
            d["palm_deg_by_turn"] = {f"{lo}-{hi}": round(math.degrees(aw[(tw >= lo) & (tw < hi)].mean().item()), 2)
                                     for lo, hi in ((0, 10), (10, 45), (45, 90), (90, 1e9)) if ((tw >= lo) & (tw < hi)).any()}
        out[nm] = d
    js = json.dumps(out, indent=1).replace("1000000000.0", "inf")
    print(js)
    if a.out: open(a.out, "w").write(js)


def view(a):
    """mjlab's play with the viser viewer (a browser tab): a checkpoint driving a.envs NOVAs live."""
    from mjlab.scripts.play import PlayConfig, run_play
    register(a.envs)
    os.chdir(HERE)
    run_play(TASK, PlayConfig(checkpoint_file=os.path.abspath(a.checkpoint), num_envs=a.envs, viewer="viser"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--run", default="s3"); t.add_argument("--iters", type=int, default=6000)
    t.add_argument("--envs", type=int, default=4096); t.add_argument("--seed", type=int, default=42); t.add_argument("--video", action="store_true")
    d = sub.add_parser("distil"); d.add_argument("--run", default="s3s"); d.add_argument("--teacher", required=True)
    d.add_argument("--iters", type=int, default=1500); d.add_argument("--envs", type=int, default=4096)
    d.add_argument("--seed", type=int, default=42); d.add_argument("--device", default="cuda:0")
    e = sub.add_parser("eval"); e.add_argument("checkpoint"); e.add_argument("--student", action="store_true")
    e.add_argument("--clips", default="mix", choices=("mix", "egodex", "lafan", "avp")); e.add_argument("--no-faults", action="store_true")
    e.add_argument("--envs", type=int, default=512); e.add_argument("--seconds", type=float, default=20.0)
    e.add_argument("--seed", type=int, default=7); e.add_argument("--device", default="cuda:0"); e.add_argument("--out")
    v = sub.add_parser("view"); v.add_argument("checkpoint"); v.add_argument("--envs", type=int, default=16)
    a = ap.parse_args()
    {"train": train, "distil": distil, "eval": evaluate, "view": view}[a.cmd](a)

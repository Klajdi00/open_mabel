"""play.py — roll a checkpoint out headless, record videos with the target frames drawn, and score it.

    cd robot_learning/reinforcement_learning/wbc_rl
    env -u DISPLAY $ISAACLAB_PY play.py --checkpoint runs/<run>/model_1000.pt --videos 4 --seconds 12
    env -u DISPLAY $ISAACLAB_PY play.py --checkpoint none --videos 1     # zero action (sanity)

Video: the camera follows one robot; DESIRED frames are the big axes + spheres (head green, left wrist blue,
right wrist red), the robot's ACTUAL frames the small axes. A caption gives the clip, playback rate and errors.
Writes <out>/<tag>_<k>.mp4 + <tag>_<k>.traj.bin (body poses and frames for the site's 3D viewer; --traj: only
these, no rendering) and <out>/<tag>_metrics.json (per-episode tracking errors, terminations, safety).
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--videos", type=int, default=3)
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--eval_envs", type=int, default=256, help="envs scored in parallel (no rendering) after the videos")
ap.add_argument("--eval_seconds", type=float, default=20.0)
ap.add_argument("--out", default=os.path.join(HERE, "runs", "videos"))
ap.add_argument("--tag", default=None)
ap.add_argument("--source", default="mix", choices=["mix", "egodex", "lafan"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", default="whole", choices=["whole", "upper"], help="whole body, or upper body with base and lift locked (the deploy toggle)")
ap.add_argument("--rate", type=float, default=0.0, help="fixed human playback rate (0 = the training range)")
ap.add_argument("--faults", action="store_true", help="Vision Pro stream faults on (drops, occlusion, stalls, latency, odometry drift)")
ap.add_argument("--pushes", action="store_true", help="keep the training pushes on (off: scores teleop alone)")
ap.add_argument("--scenario", default="", choices=["", "neck", "twist", "walk"], help="scripted operator motion on a frozen pose")
ap.add_argument("--view", default="orbit", choices=["orbit", "front", "side"], help="camera: 40 deg front-left, straight front, or left side")
ap.add_argument("--clean", action="store_true", help="no caption, no target markers (the site's hero video)")
ap.add_argument("--traj", action="store_true", help="3D trajectories only (no rendering): <tag>_<k>.traj.bin for the site viewer")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
a = ap.parse_args()

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import env as E
import tasks


def mat_to_quat_xyzw(Rm):
    from isaaclab.utils.math import quat_from_matrix
    return quat_from_matrix(Rm)


def make_env(n, render):
    cfg = E.NovaWbcEnvCfg()
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = 6.0 if render else 3.0
    cfg.seed = a.seed
    if hasattr(cfg, "faults"): cfg.faults = a.faults
    if hasattr(cfg, "upper_share"): cfg.upper_share = 1.0 if a.mode == "upper" else 0.0
    if a.rate > 0:
        cfg.speed_range = (a.rate, a.rate)
    if a.scenario:
        cfg.scripted = True; cfg.speed_range = (1e-4, 1e-4); cfg.egodex_share = 1.0     # a frozen EgoDex pose
        cfg.push_force = (0.0, 0.0)
        cfg.drift_head = cfg.drift_wrist = 1e9; cfg.drift_head_rot = cfg.drift_body_yaw = 1e9     # measure the lag, don't reset
    if not a.pushes:
        cfg.push_force = (0.0, 0.0)
    if a.source != "mix":
        cfg.egodex_share = 1.0 if a.source == "egodex" else 0.0
    if render:
        cfg.sim.physics.load_visual_shapes = True
        cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(headless=True, window_width=a.width, window_height=a.height,
                                                         enable_markers=not a.clean, max_visible_envs=1, enable_shadows=True,
                                                         background_color=(0.92, 0.93, 0.95))]
    return E.NovaWbcEnv(cfg)


def load_policy(env):
    if a.checkpoint == "none":
        return lambda obs: torch.zeros(env.num_envs, 22, device=env.device)
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    import importlib.metadata as md
    agent = tasks.NovaWbcPPOCfg()
    try:
        agent = handle_deprecated_rsl_rl_cfg(agent, md.version("rsl-rl-lib"))
    except Exception:
        pass
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
    runner.load(a.checkpoint)
    pol = runner.get_inference_policy(device=env.device)
    return lambda obs: pol(wrapped.get_observations())


def markers():
    big = FRAME_MARKER_CFG.replace(prim_path="/Visuals/wbc/desired")
    for k in big.markers: big.markers[k].scale = (0.18, 0.18, 0.18)
    small = FRAME_MARKER_CFG.replace(prim_path="/Visuals/wbc/actual")
    for k in small.markers: small.markers[k].scale = (0.09, 0.09, 0.09)
    dots = VisualizationMarkersCfg(prim_path="/Visuals/wbc/dots", markers={
        "head": sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))),
        "lw": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.95))),
        "rw": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.2, 0.2)))})
    return VisualizationMarkers(big), VisualizationMarkers(small), VisualizationMarkers(dots)


def caption(img, lines):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    d.rectangle([0, 0, im.width, 34 * len(lines) + 12], fill=(255, 255, 255))
    for i, l in enumerate(lines):
        d.text((14, 8 + 34 * i), l, fill=(20, 20, 30), font=font)
    return np.asarray(im)


def scenario_profile(t, ramp):
    """0 -> 1 -> 0: 2 s still, `ramp` s out, 3 s hold, `ramp` s back, then still (smoothstep ramps)."""
    ss = lambda x: 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * (3 - 2 * x)
    return ss((t - 2.0) / ramp) - ss((t - 5.0 - ramp) / ramp)


HOLD = lambda ramp: (2.0 + ramp + 0.5, 5.0 + ramp)                        # the window scored as "held"


RAMP = {"neck": 2.0, "twist": 2.0, "walk": 4.0}                     # walk: 1 m in 4 s, peak 0.375 m/s (base cap 0.6)


def apply_scenario(env, t):
    k = scenario_profile(t, RAMP[a.scenario])
    if a.scenario == "neck":
        env.neck_on[:] = 1.0; env.neck_off[:, 0] = math.radians(45) * k; env.neck_off[:, 1] = 0.0
    elif a.scenario == "twist":
        env.aug_on[:] = 1.0; env.aug_yaw[:] = math.radians(45) * k
    elif a.scenario == "walk":
        fy = env.m["torso_yaw"][env.c_start[env.c_id] + env.phase.long()]          # the operator's forward, human frame
        env.aug_on[:] = 1.0; env.aug_xy[:, 0] = 1.0 * k * torch.cos(fy); env.aug_xy[:, 1] = 1.0 * k * torch.sin(fy)


TRAJ_SCAL = (("e_head_cm", 100), ("a_head_deg", 100), ("e_lw_cm", 100), ("e_rw_cm", 100), ("zmp_cm", 100), ("base_mps", 1000),
             ("torso_deg", 100), ("lift_cm", 100), ("rate", 1000), ("a_lw_deg", 100), ("a_rw_deg", 100),
             ("vx_cmd", 1000), ("vy_cmd", 1000), ("wz_cmd", 1000))
SKEL = ("ls", "rs", "le", "re")                 # operator shoulders / elbows (retargeted), drawn as a ghost on the site


def write_traj(path, rows, scal, env, tag, meta):
    """<tag>_<k>.traj.bin for the site's 3D viewer: uint32 header length, JSON header, then int16 rows (n, C).
    Row = every robot body then desired head/lw/rw then actual head/lw/rw as (x, y, z) in mm from `origin` and an
    xyzw quaternion x32767, then the scalars (TRAJ_SCAL, x scale)."""
    P = np.stack(rows)                                                      # (n, F, 7) float
    o = P[0, 0, :3].copy(); o[2] = 0.0
    pq = np.concatenate([np.clip((P[..., :3] - o) * 1000, -32767, 32767), np.clip(P[..., 3:] * 32767, -32767, 32767)], -1)
    S = np.clip(np.stack(scal) * np.array([k for _, k in TRAJ_SCAL]), -32767, 32767)
    data = np.concatenate([pq.reshape(len(P), -1), S], -1).round().astype("<i2")
    head = json.dumps({"fps": round(1 / env.step_dt), "n": len(P), "bodies": list(env.robot.body_names),
                       "frames": ["d_head", "d_lw", "d_rw", "a_head", "a_lw", "a_rw"] + ["d_" + k for k in SKEL], "origin": o.round(4).tolist(),
                       "scalars": [k for k, _ in TRAJ_SCAL], "scales": [v for _, v in TRAJ_SCAL], "tag": tag, "meta": meta,
                       "scenario": a.scenario}).encode()
    head += b" " * (-(len(head) + 4) % 8)
    with open(path, "wb") as f:
        f.write(np.uint32(len(head)).tobytes()); f.write(head); f.write(data.tobytes())


def record():
    """Rollouts of one robot. Always writes the 3D trajectory (<tag>_<k>.traj.bin); renders the mp4 unless --traj."""
    render = not a.traj
    env = make_env(1, render=render)
    policy = load_policy(env)
    if render:
        import imageio.v2 as imageio
        viz = env.sim.visualizers[0]
        big, small, dots = markers()
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or os.path.splitext(os.path.basename(a.checkpoint))[0]
    metas = [str(m) for m in np.load(env.cfg.motions)["meta"]]
    steps = int(a.seconds / env.step_dt)
    paths = []
    for k in range(a.videos):
        torch.manual_seed(a.seed * 1000 + k); np.random.seed(a.seed * 1000 + k)   # rollout k: same operator motion for any policy
        obs, _ = env.reset()
        frames, cam, rows, scal = [], None, [], []
        _, yaw_start, _ = env.base_pose(); yaw_start = yaw_start.clone(); log = []
        meta0 = metas[int(env.c_id[0])]
        for i in range(steps):
            if a.scenario:
                apply_scenario(env, (i + 1) * env.step_dt)
            with torch.inference_mode():
                act = policy(obs)
            obs, _, term, tout, _ = env.step(act)
            t = env.tgt
            cp, cR = env.tracked_now()
            dp = torch.stack([t["head_p"], t["lw_p"], t["rw_p"]], 1)[0]
            dR = torch.stack([t["head_R"], t["lw_R"], t["rw_R"]], 1)[0]
            pos, yaw, _ = env.base_pose()
            qj = env.robot.data.joint_pos.torch[0, env.jid]
            e_h = (cp[0, 0] - dp[0]).norm().item() * 100
            e_l, e_r = (cp[0, 1] - dp[1]).norm().item() * 100, (cp[0, 2] - dp[2]).norm().item() * 100
            a_h = math.degrees(E.rot_angle(cR[0, 0], dR[0]).item())
            torso, lift = -math.degrees(qj[env.i_torso].item()), 200 * qj[0].item()
            sk = torch.stack([t[k_][0] for k_ in SKEL], 0)
            fp = torch.cat([env.robot.data.body_link_pos_w.torch[0], dp, cp[0], sk], 0)
            fq = torch.cat([env.robot.data.body_link_quat_w.torch[0], mat_to_quat_xyzw(dR), mat_to_quat_xyzw(cR[0]),
                            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(len(SKEL), 1)], 0)
            rows.append(torch.cat([fp, fq], -1).cpu().numpy())
            scal.append([e_h, a_h, e_l, e_r, env.zmp_margin[0].item() * 100, env.v_ref_b[0, :2].norm().item(), torso, lift,
                         env.rate[0].item(), math.degrees(E.rot_angle(cR[0, 1], dR[1]).item()),
                         math.degrees(E.rot_angle(cR[0, 2], dR[2]).item()), *env.v_cmd_b[0].tolist()])
            if a.scenario:
                log.append([i * env.step_dt, math.degrees(E.wrap(yaw[0] - yaw_start[0]).item()), math.degrees(qj[env.i_neck[0]].item()),
                            pos[0, 0].item(), pos[0, 1].item(), e_h])
            if not render: continue
            if not a.clean:
                big.visualize(translations=dp, orientations=mat_to_quat_xyzw(dR))
                small.visualize(translations=cp[0], orientations=mat_to_quat_xyzw(cR[0]))
                dots.visualize(translations=dp, marker_indices=torch.tensor([0, 1, 2], device=env.device))
            look = pos[0].cpu().numpy() + np.array([0, 0, 1.2])
            cam_yaw = float(yaw_start[0]) if a.scenario else float(yaw[0])        # scenarios: a fixed camera shows the base turn
            yw = cam_yaw + math.radians({"orbit": 40, "front": 0, "side": 90}[a.view])
            if a.clean: look = look + 0.55 * np.array([math.sin(yw), -math.cos(yw), 0.0])   # robot in the right third, title on the left
            dist = 1.9 if a.clean else 2.2
            eye = look + np.array([dist * math.cos(yw), dist * math.sin(yw), 0.35])
            cam = eye if cam is None else 0.9 * cam + 0.1 * eye          # smooth follow
            viz.set_camera_view(cam.tolist(), look.tolist())
            img = viz.render_rgb_array()
            if img is None: continue
            meta = metas[int(env.c_id[0])]
            if a.clean: frames.append(np.asarray(img)[..., :3].copy()); continue
            frames.append(caption(np.asarray(img)[..., :3], [
                f"NOVA whole-body RL  |  {tag}  |  t={i * env.step_dt:5.1f}s  " + (f"operator: {a.scenario}" if a.scenario else f"rate x{env.rate[0].item():.2f}"),
                f"{meta[:70]}",
                f"error  head {e_h:4.1f} cm {a_h:4.1f} deg   wrists {0.5 * (e_l + e_r):4.1f} cm",
                f"zmp margin {env.zmp_margin[0].item() * 100:4.1f} cm   base {env.v_ref_b[0, :2].norm().item():.2f} m/s   "
                f"torso {torso:+3.0f} deg   lift {lift:3.0f} cm"]))
        if a.scenario:
            L = np.array(log); h0, h1 = HOLD(RAMP[a.scenario]); hold = (L[:, 0] > h0) & (L[:, 0] < h1)
            pre = (L[:, 0] > 1.5) & (L[:, 0] < 2.0)                               # settled, before the operator moves
            y0, n0 = L[pre, 1].mean(), L[pre, 2].mean()
            res = {"scenario": a.scenario, "base_yaw_change_deg": float(L[hold, 1].mean() - y0), "neck_yaw_change_deg": float(L[hold, 2].mean() - n0),
                   "base_moved_m": float(np.hypot(L[hold, 3] - L[pre, 3].mean(), L[hold, 4] - L[pre, 4].mean()).mean()),
                   "base_yaw_return_deg": float(L[-25:, 1].mean() - y0), "head_cm_mean": float(L[100:, 5].mean()),
                   "trace": L[::5].round(3).tolist()}
            if render:                                                            # --traj reruns keep the filmed scenario's numbers
                json.dump(res, open(os.path.join(a.out, f"{tag}_scenario.json"), "w"))
            print({k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in res.items() if k_ != "trace"}, flush=True)
        tp = os.path.join(a.out, f"{tag}_{k}.traj.bin")
        write_traj(tp, rows, scal, env, tag, meta0); paths.append(tp)
        if render:
            p = os.path.join(a.out, f"{tag}_{k}.mp4")
            imageio.mimwrite(p, frames, fps=int(round(1 / env.step_dt)), quality=7, macro_block_size=8)
            paths.append(p); print("wrote", p, len(frames), "frames", flush=True)
        print("wrote", tp, f"{os.path.getsize(tp) / 1e3:.0f} kB", flush=True)
    return paths


def evaluate():
    """Score the policy on eval_envs parallel envs for eval_seconds; errors count after the glide-in."""
    env = make_env(a.eval_envs, render=False)
    policy = load_policy(env)
    obs, _ = env.reset()
    steps = int(a.eval_seconds / env.step_dt)
    acc = {k: [] for k in ("head_cm", "head_deg", "wrist_cm", "wrist_deg", "body_yaw_deg", "zmp_cm", "clear_cm", "tilt_deg", "base_mps",
                           "torso_deg", "lean_err_deg", "upper_arm_deg", "ego",
                           "hand_hf_mm", "tgt_hf_mm", "hand_acc", "tgt_acc", "arm_da", "arm_qdd")}
    lost_n = torch.zeros(3, device=env.device); lost_v = 0.0; lost_vn = 0
    KINDS = ("tilt", "collide", "drift", "limit")[:env.last_term.shape[1]] if hasattr(env, "last_term") else ("tilt", "collide", "drift", "limit")
    terms_src = torch.zeros(2, len(KINDS), device=env.device); steps_src = torch.zeros(2, device=env.device)
    drift_src = torch.zeros(2, 4, device=env.device)        # seconds past each drift bound: head pos, wrist pos, head rot, body heading
    # actuator use vs the SAFETY x datasheet limits (robot.JOINT_SPEC): worst ratio seen per joint, live envs only
    n_pos = len(env.i_pos); lim_v = torch.zeros(n_pos, device=env.device); lim_t = torch.zeros_like(lim_v); lim_rms = torch.zeros_like(lim_v)
    lift_lo, lift_hi = 1.0, -1.0
    lift_v_max = 0.0; lim_kinds = torch.zeros(4, device=env.device); lim_stop = 0; past = 0
    terms = torch.zeros(len(KINDS), device=env.device); lift_travel = 0.0; lift_cmd = 0.0; lift_prev = env.lift_tgt.clone()
    # smoothness: wrist (actual and target) minus its 0.1 s EMA = shake; second difference = acceleration
    dt = env.step_dt; al = dt / 0.1
    sm = {"hand": None, "tgt": None}; p1 = {"hand": None, "tgt": None}; p2 = {"hand": None, "tgt": None}
    qd_prev = env.robot.data.joint_vel.torch[:, env.jid].clone(); arm = [4 + i for i in range(len(env.i_arm))]
    for i in range(steps):
        ego_before = env.c_ego[env.c_id].clone()
        a_prev = env.actions.clone()
        with torch.inference_mode():
            obs, _, term, tout, _ = env.step(policy(obs))
        cp_, _ = env.tracked_now()
        cur = {"hand": cp_[:, 1:].clone(), "tgt": torch.stack([env.tgt["lw_p"], env.tgt["rw_p"]], 1)}
        fresh = (env.episode_length_buf <= 3) | term | tout
        sv = {}
        for k in cur:
            if sm[k] is None: sm[k] = cur[k].clone(); p1[k] = cur[k].clone(); p2[k] = cur[k].clone()
            sm[k] = torch.where(fresh[:, None, None], cur[k], sm[k] + al * (cur[k] - sm[k]))
            sv[k + "_hf_mm"] = (cur[k] - sm[k]).norm(dim=-1).mean(-1) * 1000
            sv[k + "_acc"] = ((cur[k] - 2 * p1[k] + p2[k]) / dt ** 2).norm(dim=-1).mean(-1)
            p2[k] = torch.where(fresh[:, None, None], cur[k], p1[k]); p1[k] = cur[k].clone()
        sv["arm_da"] = (env.actions[:, arm] - a_prev[:, arm]).abs().mean(-1)
        qd = env.robot.data.joint_vel.torch[:, env.jid]
        sv["arm_qdd"] = ((qd - qd_prev)[:, env.i_arm] / dt).abs().mean(-1); qd_prev = qd.clone()
        terms += (env.last_term & term[:, None]).float().sum(0)
        if hasattr(env, "lim_joint"): lim_stop += env.lim_joint[term & env.last_term[:, 3]].float().sum(0)   # pre-reset breach
        for j, msk in enumerate((ego_before, ~ego_before)):
            terms_src[j] += (env.last_term & (term & msk)[:, None]).float().sum(0); steps_src[j] += msk.float().sum()
            if hasattr(env, "drift_kind"): drift_src[j] += (env.drift_kind & msk[:, None]).float().sum(0) * env.step_dt   # seconds behind
        lift_travel += env.robot.data.joint_vel.torch[:, env.jid[0]].abs().sum().item() * 2 * env.step_dt   # sim column incl. jitter
        lift_cmd += (env.lift_tgt - lift_prev).abs().clamp(max=0.001).sum().item() * 2; lift_prev = env.lift_tgt.clone()   # what NOVA's lift would execute (resets excluded)
        live = (env.episode_length_buf.float() * env.step_dt > env.cfg.grace_s) & ~term
        if not live.any(): continue
        if hasattr(env, "tau2"):
            lim_v = torch.maximum(lim_v, (qd[:, env.i_pos].abs() / env.v_lim)[live].amax(0))
            lim_t = torch.maximum(lim_t, (env.tau.abs() / env.t_pk)[live].amax(0))
            lim_rms = torch.maximum(lim_rms, (env.tau2.sqrt() / env.t_rated)[live].amax(0))
            lift_v_max = max(lift_v_max, float(qd[:, env.i_lift].abs()[live].max()))
            ql = env.robot.data.joint_pos.torch[:, env.jid[env.i_lift]][live]
            lift_lo, lift_hi = min(lift_lo, float(ql.min())), max(lift_hi, float(ql.max()))
            lim_kinds += env.lim_kind[live].float().sum(0)
            if hasattr(env, "lim_joint"): past += env.lim_joint[live][:, 2].float().sum(0)   # past an end stop (no stop from v39)
        t = env.tgt; cp, cR = env.tracked_now(); _, yaw, _ = env.base_pose()
        g = env.robot.data.projected_gravity_b.torch
        vals = dict(head_cm=(cp[:, 0] - t["head_p"]).norm(dim=-1) * 100, head_deg=torch.rad2deg(E.rot_angle(cR[:, 0], t["head_R"])),
                    wrist_cm=50 * ((cp[:, 1] - t["lw_p"]).norm(dim=-1) + (cp[:, 2] - t["rw_p"]).norm(dim=-1)),
                    wrist_deg=torch.rad2deg(0.5 * (E.rot_angle(cR[:, 1], t["lw_R"]) + E.rot_angle(cR[:, 2], t["rw_R"]))),
                    body_yaw_deg=torch.rad2deg(E.wrap(t["body_yaw"] - yaw).abs()), zmp_cm=env.zmp_margin * 100, clear_cm=env.clear * 100,
                    tilt_deg=torch.rad2deg(torch.asin(g[:, :2].norm(dim=-1).clamp(max=1))),
                    base_mps=env.v_ref_b[:, :2].norm(dim=-1), ego=env.c_ego[env.c_id].float())
        q = env.robot.data.joint_pos.torch[:, env.jid]
        vals["torso_deg"] = torch.rad2deg(-q[:, env.i_torso])                         # + = leaning forward
        vals["lean_err_deg"] = torch.rad2deg((q[:, env.i_torso] + t["lean"]).abs())   # robot lean vs the operator's
        bp = env.robot.data.body_link_pos_w.torch
        ua = lambda e, s: torch.nn.functional.normalize(bp[:, e] - bp[:, s], dim=-1)
        ang = lambda u, v: torch.rad2deg(torch.acos((u * v).sum(-1).clamp(-1, 1)))
        vals["upper_arm_deg"] = 0.5 * (ang(ua(env.elbow_body[0], env.shoulder_body[0]), t["l_arm"]) + ang(ua(env.elbow_body[1], env.shoulder_body[1]), t["r_arm"]))
        vals.update(sv)
        for k, v in vals.items(): acc[k].append(v[live].cpu())
        if hasattr(env, "lost_prev") and env.cfg.faults:                # a lost hand should hold still, not chase
            L = env.lost_prev & live[:, None]; lost_n += L.float().sum(0)
            hv = env.robot.data.body_com_lin_vel_w.torch[:, env.trk_body[1:]].norm(dim=-1)
            lost_v += float(hv[L[:, 1:]].sum()); lost_vn += int(L[:, 1:].sum())
    acc = {k: torch.cat(v).numpy() for k, v in acc.items()}
    env_min = a.eval_envs * a.eval_seconds / 60
    out = {"checkpoint": a.checkpoint, "env_minutes": env_min,
           "terminations_per_min": {k: float(terms[i]) / env_min for i, k in enumerate(KINDS)},
           "terminations_per_min_by_source": {s: {k: float(terms_src[j, i]) / max(float(steps_src[j]) * env.step_dt / 60, 1e-9)
                                                  for i, k in enumerate(KINDS)}
                                              for j, s in enumerate(("egodex", "lafan"))},
           "drift_s_per_min_by_kind": {s: {k: float(drift_src[j, i]) / max(float(steps_src[j]) * env.step_dt / 60, 1e-9)
                                         for i, k in enumerate(("head_pos", "wrist_pos", "head_rot", "body_heading"))}
                                     for j, s in enumerate(("egodex", "lafan"))},
           "lift_m_per_min": lift_travel / env_min, "lift_cmd_m_per_min": lift_cmd / env_min}
    if hasattr(env, "lost_prev") and env.cfg.faults:
        n_live = len(acc["head_cm"])
        out["stream"] = {"lost_share_head_l_r": [float(x) / max(n_live, 1) for x in lost_n],
                         "lost_hand_speed_mps": lost_v / max(lost_vn, 1)}
    for k in ("head_cm", "head_deg", "wrist_cm", "wrist_deg", "body_yaw_deg", "base_mps", "torso_deg", "lean_err_deg", "upper_arm_deg",
              "hand_hf_mm", "tgt_hf_mm", "hand_acc", "tgt_acc", "arm_da", "arm_qdd"):
        v = acc[k]
        if k == "wrist_deg" and (acc["ego"] > 0.5).any(): v = v[acc["ego"] > 0.5]   # LAFAN1 palms are a guess (no finger markers)
        out[k] = {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
        v = acc[k]
        for src, m in (("egodex", acc["ego"] > 0.5), ("lafan", acc["ego"] < 0.5)):
            if m.any(): out[k][src] = float(v[m].mean())
    out["zmp_cm"] = {"mean": float(acc["zmp_cm"].mean()), "p01": float(np.percentile(acc["zmp_cm"], 1)), "min": float(acc["zmp_cm"].min())}
    out["clear_cm"] = {"p01": float(np.percentile(acc["clear_cm"], 1)), "min": float(acc["clear_cm"].min())}
    out["tilt_deg"] = {"p99": float(np.percentile(acc["tilt_deg"], 99)), "max": float(acc["tilt_deg"].max())}
    if hasattr(env, "tau2"):
        import robot as R
        names = [R.ACTUATED[i] for i in env.i_pos]
        out["limits"] = {"safety": R.SAFETY, "lift_stage_mps_max": lift_v_max, "lift_stage_cap": 0.5 * R.LIFT_VLIM,
                         "lift_stage_range_m": [lift_lo, lift_hi], "lift_stage_allowed_m": [getattr(R, "LIFT_MARGIN", 0.0), R.LIFT_STROKE - getattr(R, "LIFT_MARGIN", 0.0)],
                         "breach_env_steps": dict(zip(("speed", "peak", "overload", "position"), lim_kinds.tolist())),
                         "worst_ratio": {n: {"speed": float(v), "peak": float(t), "rms": float(r)}
                                         for n, v, t, r in zip(names, lim_v, lim_t, lim_rms)},
                         # the steps above exclude stops (reset on the same step); these are the breaches that stopped one
                         "past_stop_s_per_robot_min": {n: round(float(c) * env.step_dt / max(float(steps_src.sum()) * env.step_dt / 60, 1e-9), 3)
                                                       for n, c in zip(R.ACTUATED, (past.tolist() if torch.is_tensor(past) else [0] * len(R.ACTUATED))) if c},
                         "stops_by_joint": {f"{n}/{k}": int(c) for i, k in enumerate(("speed", "overload", "position"))
                                            for n, c in zip(R.ACTUATED, (lim_stop[i].tolist() if torch.is_tensor(lim_stop) else [0] * len(R.ACTUATED))) if c}}
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or os.path.splitext(os.path.basename(a.checkpoint))[0]
    p = os.path.join(a.out, f"{tag}_metrics.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return p


if __name__ == "__main__":
    out = record() if a.videos > 0 else [evaluate()]
    print(json.dumps({"outputs": out}))

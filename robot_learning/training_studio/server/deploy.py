"""
Closed-loop MuJoCo deployment — the policy drives the physics sim on demand.

Loads the MABEL scene + a trained checkpoint, then at each control step renders
the selected cameras, reads scoped proprioception, runs the network, SCATTERS the
policy's scoped action back onto the full 59-DOF ctrl (unselected joints held),
and steps physics. Streams the trajectory (actuator positions for the 3D viewer +
predicted actions for the live graphs). MuJoCo handles deployment physics.
"""
import os

import numpy as np

from . import config as CFG
from . import embodiment as EMB
from . import eval as EV


def _render_size():
    cams = {c["name"]: (c.get("h", 270), c.get("w", 480)) for c in EMB.cameras()}
    return cams


def run(checkpoint_dir, ds_meta_root, selection, device="cpu", control_steps=120,
        physics_substeps=None, on_step=None, progress=None):
    """Run the policy in closed-loop MuJoCo. Returns the recorded trajectory."""
    import mujoco
    import torch
    from lerobot.utils.control_utils import predict_action

    if not CFG.SCENE_XML or not os.path.exists(CFG.SCENE_XML):
        raise FileNotFoundError("sim scene not available (data_collection.config.SCENE_XML)")

    model = mujoco.MjModel.from_xml_path(CFG.SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]

    policy, pre, post, cfg, meta = EV._load_policy(checkpoint_dir, ds_meta_root, device)
    dev = torch.device(device)
    cams = [k.split("observation.images.")[-1] for k in getattr(cfg, "input_features", {})
            if k.startswith("observation.images.")]
    sizes = _render_size()
    renderers = {}
    for c in cams:
        h, w = sizes.get(c, (270, 480))
        try:
            renderers[c] = mujoco.Renderer(model, height=h, width=w)
        except Exception:
            renderers[c] = None

    sidx = selection["state_indices"]
    scatter = selection["scatter_map"]
    # control period: default to model timestep so one physics step per control step
    substeps = physics_substeps or max(1, int(round((1.0 / CFG.LOG_FPS) / model.opt.timestep)))

    task = "mabel task"
    policy.reset()
    traj = []
    for step in range(control_steps):
        proprio_full = np.asarray(data.actuator_length, dtype=np.float32).copy()
        obs = {"observation.state": proprio_full[sidx]}
        for c in cams:
            r = renderers.get(c)
            if r is not None:
                try:
                    r.update_scene(data, camera=c)
                    obs[f"observation.images.{c}"] = r.render()
                except Exception:
                    obs[f"observation.images.{c}"] = np.zeros((270, 480, 3), np.uint8)
        act = predict_action(obs, policy, dev, pre, post, use_amp=False, task=task)
        act = np.asarray(act.detach().to("cpu")).reshape(-1)[: len(scatter)]
        # scatter the scoped action onto the full ctrl (hold the rest)
        for j, gi in enumerate(scatter):
            data.ctrl[gi] = float(act[j])
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        frame = {
            "step": step,
            "ctrl": [round(float(x), 5) for x in data.ctrl],
            "actuator_positions": [round(float(x), 5) for x in data.actuator_length],
            "actuator_names": act_names,
            "action": [round(float(x), 5) for x in act],
            "action_indices": scatter,
        }
        traj.append(frame)
        if on_step:
            on_step(frame)
        if progress:
            progress(step + 1, control_steps, stage=0)

    for r in renderers.values():
        if r is not None:
            try:
                r.close()
            except Exception:
                pass
    return {"frames": len(traj), "trajectory": traj, "actuator_names": act_names,
            "cameras": cams, "action_dim": len(scatter)}

#!/usr/bin/env python3
"""The collision demo: drive both palms straight INTO the chest, guard OFF vs ON.

The grid sweep shows retargeting quality; it does not show the guard doing its
job, because a workspace grid mostly asks for poses that are already clear. This
asks for the one thing the guard exists to refuse — a hand commanded into the
robot's own body — and puts the two answers side by side with the live gap.
"""
import os, sys, numpy as np, mujoco
import imageio.v2 as imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_front_workspace as H
from mabel.teleop_engine.workspace_table import ArmChain, palm_target, PALM_ROLLS
from mabel.teleop_engine.arm import ArmRetargeter
from mabel.teleop_engine.hand import HandOptimizer
from mabel.teleop_engine.collision import ArmCollisionProjector

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/frontws_collision.mp4"
FPS, W, Hh = 30, 960, 540
m, d0 = H.load_model()

def pane(guard_on):
    d = mujoco.MjData(m); d.qpos[:] = d0.qpos
    d.ctrl[:] = 0.0; d.ctrl[m.actuator("Lift").id] = 0.3175
    mujoco.mj_forward(m, d)
    chains = {s: ArmChain(m, s) for s in H.SIDES}
    arms = {s: ArmRetargeter(m, s, HandOptimizer(m, s).R_rob) for s in H.SIDES}
    for s in H.SIDES:
        chains[s].sync(d); arms[s].sync(d)
    proj = ArmCollisionProjector(m, d, arms) if guard_on else None
    return dict(d=d, chains=chains, arms=arms, proj=proj,
                tag="guard OFF" if not guard_on else "guard ON",
                rend=mujoco.Renderer(m, height=Hh, width=W // 2))

panes = [pane(False), pane(True)]
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = (-0.50, 0.0, 0.95); cam.distance, cam.azimuth, cam.elevation = 1.85, 18.0, -10.0
act = {s: [m.actuator(f"{s.capitalize()} Arm {i}").id for i in range(1, 8)] for s in H.SIDES}
wact = {s: m.actuator(f"{s.capitalize()} Wrist").id for s in H.SIDES}

# A path that starts clear in front and walks straight back into the sternum.
def target(side, u):
    """u in [0,1]: 0 = out in front, 1 = deep inside the chest."""
    x = 0.42 - 0.42 * u                     # shoulder-frame forward
    y = (-0.16 if side == "right" else 0.16) * (1.0 - u)   # converge on the midline
    return np.array([x, y, -0.12])

N = int(6.5 * FPS)
with imageio.get_writer(OUT, fps=FPS, macro_block_size=1) as w:
    for f in range(N):
        u = min(1.0, max(0.0, (f / FPS - 0.6) / 3.2))
        tiles = []
        for p in panes:
            d = p["d"]; sol = {}; wr = {}
            for s in H.SIDES:
                ch, rt = p["chains"][s], p["arms"][s]
                ch.sync(d); rt.sync(d)
                p_s = target(s, u)
                R_s = palm_target(np.array([1.0, 0.0, 0.0]), PALM_ROLLS["down"], s)
                hk, p_w, R_w = H._hk_world(ch, p_s, R_s)
                sol[s] = rt.solve(hk, dt=1.0 / FPS)
                wr[s] = float(rt.ik.qpos[rt.orca_wrist_qadr])
                p[f"tgt_{s}"] = (p_w, R_w)
            if p["proj"] is not None:
                sol["right"], sol["left"], wr["right"], wr["left"] = p["proj"].project(
                    sol["right"], sol["left"], 1.0 / FPS,
                    wrist_r=wr["right"], wrist_l=wr["left"])
            for s in H.SIDES:
                d.ctrl[act[s]] = sol[s][:7]
                d.ctrl[wact[s]] = wr[s]
            for _ in range(int((1.0 / FPS) / m.opt.timestep)):
                mujoco.mj_step(m, d)
            r = p["rend"]; r.update_scene(d, camera=cam)
            for s in H.SIDES:
                H._draw_target(r.scene, *p[f"tgt_{s}"], size=0.10)
            tiles.append(H._label(r.render(), p["tag"], "reaching into the chest",
                                  H._body_gap(m, d)))
        w.append_data(np.concatenate(tiles, axis=1))
        if f % 60 == 0:
            print(f"  {f}/{N}")
print("video ->", OUT, os.path.getsize(OUT), "bytes")

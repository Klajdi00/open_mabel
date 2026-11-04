#!/usr/bin/env python3
"""Bake the arm's rigid-body parameters out of the MJCF into `arm_params.py`.

    python3 controller/mabel/basic/extract_arm_params.py

BUILD TIME ONLY. This is the one place MuJoCo is allowed near the gravity model,
and it is allowed here for the same reason a URDF is: what it emits are
*a-priori model constants* — link transforms, masses, centre-of-mass offsets —
the kind of thing that ships with the robot. Nothing it produces is simulator
state. At RUNTIME the gravity model imports only numpy and reads only measured
joint angles, so the identical code runs against the physical arm.

Each of the seven arm joints rotates about its own local +Z at its body origin,
so one fixed (pos, quat) per joint fully describes the chain. Every body that is
not itself an arm link (the wrist, the hand mount, all 17 ORCA finger links) is
LUMPED into its nearest arm ancestor at the hand's rest pose: the fingers are
light and sit close to the wrist axis, so their motion barely changes the arm's
gravity load, and carrying 20+ extra links at runtime would buy nothing. The
approximation is measured, not assumed — see `exp_gravity_model.py` E3.
"""
import os
import sys

import numpy as np
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
MJCF = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_params.py")

SIDES = {"left": "Left Arm {}", "right": "Right Arm {}"}


def chain_bodies(model, side):
    return [model.body(model.joint(SIDES[side].format(i)).bodyid[0])
            for i in range(1, 8)]


def extract(model, data, side):
    """(pos, quat, mass, com) per link, with descendants lumped in."""
    bodies = chain_bodies(model, side)
    chain_ids = [b.id for b in bodies]

    # nearest arm ancestor for every body in the model
    owner = {}
    for bid in range(model.nbody):
        cur = bid
        while cur > 0 and cur not in chain_ids:
            cur = int(model.body_parentid[cur])
        if cur in chain_ids:
            owner.setdefault(cur, []).append(bid)

    mujoco.mj_forward(model, data)          # rest pose: hands open, arm at qpos0
    links = []
    for b in bodies:
        m_tot, mc = 0.0, np.zeros(3)
        R = np.array(data.body(b.id).xmat).reshape(3, 3)
        p = np.array(data.body(b.id).xpos)
        for bid in owner.get(b.id, []):
            m_i = float(model.body_mass[bid])
            if m_i <= 0:
                continue
            # the body's OWN com in world, then into the arm link's frame
            c_w = (np.array(data.body(bid).xpos)
                   + np.array(data.body(bid).xmat).reshape(3, 3)
                   @ np.array(model.body_ipos[bid]))
            mc += m_i * (R.T @ (c_w - p))
            m_tot += m_i
        com = (mc / m_tot) if m_tot > 0 else np.zeros(3)
        links.append({
            "name": b.name,
            "pos": np.array(model.body_pos[b.id], dtype=float),
            "quat": np.array(model.body_quat[b.id], dtype=float),
            "mass": float(m_tot),
            "com": com,
        })

    # world->shoulder-base rotation at the rest pose, so the runtime model can be
    # told how the torso is leaning instead of assuming upright
    parent = int(model.body_parentid[bodies[0].id])
    base_R = np.array(data.body(parent).xmat).reshape(3, 3)
    return links, base_R


def fmt(v):
    return "(" + ", ".join(f"{x: .8f}" for x in np.asarray(v).ravel()) + ")"


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    out = ['"""Arm rigid-body constants, GENERATED — do not hand-edit.',
           "",
           "Emitted by `extract_arm_params.py` from the canonical MJCF. These are",
           "a-priori model data (link transforms, masses, COM offsets), the same",
           "class of thing a URDF carries — NOT simulator state. `gravity.py`",
           "consumes them with numpy alone, so the same model runs on hardware.",
           "",
           "Re-run the extractor whenever the MJCF's arm geometry or masses change;",
           "`test_gravity_model.py` fails if these drift from the model.",
           '"""',
           "", "ARM = {"]
    for side in ("left", "right"):
        links, base_R = extract(model, data, side)
        out.append(f'    "{side}": {{')
        out.append(f'        "base_R": {fmt(base_R)},   # world<-torso at rest')
        out.append('        "links": [')
        total = 0.0
        for L in links:
            total += L["mass"]
            out.append(f'            # {L["name"]}')
            out.append(f'            dict(pos={fmt(L["pos"])},')
            out.append(f'                 quat={fmt(L["quat"])},')
            out.append(f'                 mass={L["mass"]: .6f},')
            out.append(f'                 com={fmt(L["com"])}),')
        out.append("        ],")
        out.append(f'        "total_mass": {total:.6f},')
        out.append("    },")
        print(f"{side:5s}: 7 links, {total:.3f} kg total")
    out.append("}")
    out.append("")
    out.append("GRAVITY = (0.0, 0.0, -9.81)   # m/s^2, world frame")
    out.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print("wrote", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
extract_arm_model.py
====================

Build a standalone, fixed-base 7-DOF MuJoCo model of the MABEL right arm by
extracting the exact link inertials / joints / meshes from the source-of-truth
``mabel_full.xml``.

Two things this script gets right that the previous version did not, and both of
them are fatal on hardware:

1. **The base orientation is DERIVED, not invented.**
   The old version hard-coded ``euler="-0.35 0.15 0"`` as a "representative"
   mounting pose, chosen because it made the identifiability look good.  On the
   real robot the arm hangs off ``chassis -> lift -> body_mount -> torso ->
   r_arm_mount``, and the gravity vector in the arm's base frame is
   ``[0, -9.66, -1.70]``, not ``[1.47, 3.33, -9.11]``.  **That is a 99.9 degree
   error.**  Since the whole identification is "how does gravity project onto
   each joint axis", a base frame rotated ~100 degrees from reality makes every
   regressor column, every optimal posture and every identified parameter wrong
   -- silently, because the least-squares still converges.  We now run MuJoCo's
   forward kinematics on the full robot at the *calibration configuration* and
   bake the resulting world<-mount transform in.

2. **The hand is INCLUDED, not pruned.**
   The old version dropped everything below ``r_hand_mount`` with a note saying
   the hand "can later be lumped in as a payload".  It never was.  The ORCA hand
   is **1.117 kg -- 3.5x the mass of the 0.319 kg link it bolts to** -- and it is
   rigidly attached (no joint between ``r_hand_mount`` and ``right_tower``).
   Calibrating without it means the terminal link's identified first moment has
   to absorb a kilogram that the model says is not there, which drives its
   implied COM metres away from the part.  We now compute the composite rigid
   inertia of the whole hand subtree and merge it into ``r_hand_mount``.

The calibration configuration
-----------------------------
``--torso`` and ``--lift`` set the pose of the chain *above* the arm.  The torso
is a hinge, so its angle rotates the arm's base frame and therefore changes the
gravity direction the arm sees.  **Whatever you pass here, the real robot must be
parked at the same values during collection** -- the generated model records them
so the hardware collector can assert it.

Usage:
    python3 tools/extract_arm_model.py                    # torso=0, lift=0, hand on
    python3 tools/extract_arm_model.py --torso 0.3        # torso parked at 0.3 rad
    python3 tools/extract_arm_model.py --no-hand          # bare arm (hand removed)
"""
import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(CALIB_ROOT))
FULL_XML = os.path.join(REPO_ROOT, "simulation", "mabel_mujoco", "models", "mabel_full.xml")
MESH_ABS = os.path.join(REPO_ROOT, "simulation", "mabel_mujoco", "assets", "meshes")
OUT_XML = os.path.join(CALIB_ROOT, "models", "right_arm_sysid.xml")
OUT_META = os.path.join(CALIB_ROOT, "models", "right_arm_sysid.json")
MESHDIR_REL = os.path.relpath(MESH_ABS, os.path.dirname(OUT_XML))

ARM_JOINTS = [
    ("Right Arm 1", "motor_DM8009", "r_j1"),
    ("Right Arm 2", "motor_DM8009", "r_j2"),
    ("Right Arm 3", "motor_DM4340", "r_j3"),
    ("Right Arm 4", "motor_DM4340", "r_j4"),
    ("Right Arm 5", "motor_DM4310", "r_j5"),
    ("Right Arm 6", "motor_DM4310", "r_j6"),
    ("Right Arm 7", "motor_DM4310", "r_j7"),
]
MOUNT = "r_arm_mount"
HAND_ROOT = "r_hand_mount"


# --------------------------------------------------------------------------- #
#  Kinematics off the full robot                                               #
# --------------------------------------------------------------------------- #
def full_model_state(torso: float, lift: float):
    """Forward-kinematics the full robot at the calibration configuration."""
    m = mujoco.MjModel.from_xml_path(FULL_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    def setj(name, val):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return False
        d.qpos[m.jnt_qposadr[jid]] = val
        return True

    setj("Torso", torso)
    # the lift is a two-stage prismatic pair driven together
    setj("Lift Lower", lift / 2.0)
    setj("Lift Upper", lift / 2.0)
    mujoco.mj_forward(m, d)
    return m, d


def mount_pose(m, d):
    """World <- r_arm_mount transform (pos, quat wxyz) and gravity in that frame."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, MOUNT)
    pos = d.xpos[b].copy()
    R = d.xmat[b].reshape(3, 3).copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    g_world = m.opt.gravity.copy()
    return pos, quat, R.T @ g_world


# --------------------------------------------------------------------------- #
#  Rigid composition of the hand into the terminal link                        #
# --------------------------------------------------------------------------- #
def subtree_bodies(m, root_id):
    out, stack = [], [root_id]
    while stack:
        b = stack.pop()
        out.append(b)
        stack += [i for i in range(m.nbody) if m.body_parentid[i] == b]
    return out


def lump_hand(m, d):
    """Composite (mass, com, inertia) of r_hand_mount + the whole rigid hand,
    expressed in the r_hand_mount body frame.

    The hand's finger joints are held at their model zero during calibration, so
    the subtree is rigid and this composition is exact.  (The fingers are only
    ~0.19 kg of the 1.117 kg hand; the 0.9 kg tower is rigidly bolted on.)
    """
    hm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT)
    bodies = subtree_bodies(m, hm)

    M = 0.0
    com_w = np.zeros(3)
    for b in bodies:
        mb = float(m.body_mass[b])
        M += mb
        com_w += mb * d.xipos[b]
    com_w /= max(M, 1e-12)

    I_w = np.zeros((3, 3))
    for b in bodies:
        mb = float(m.body_mass[b])
        Rb = d.ximat[b].reshape(3, 3)                      # world <- principal axes
        Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T        # about ITS own com, in world
        r = d.xipos[b] - com_w
        I_w += Ib + mb * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    # express in the r_hand_mount body frame
    R_hm = d.xmat[hm].reshape(3, 3)
    com_local = R_hm.T @ (com_w - d.xpos[hm])
    I_local = R_hm.T @ I_w @ R_hm

    evals, evecs = np.linalg.eigh(I_local)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1.0
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, evecs.reshape(-1))
    return M, com_local, np.clip(evals, 1e-9, None), quat, len(bodies)


def fmt(v, n=9):
    return " ".join(f"{x:.{n}g}" for x in np.atleast_1d(v))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torso", type=float, default=0.0,
                    help="torso hinge angle (rad) the robot will be PARKED at")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="lift extension (m) the robot will be PARKED at")
    ap.add_argument("--no-hand", action="store_true",
                    help="calibrate the bare arm (hand physically removed)")
    args = ap.parse_args()

    m, d = full_model_state(args.torso, args.lift)
    pos, quat, g_mount = mount_pose(m, d)

    print(f"calibration configuration: torso = {args.torso} rad, lift = {args.lift} m")
    print(f"  r_arm_mount world pos   : {fmt(pos, 5)}")
    print(f"  world<-mount quat (wxyz): {fmt(quat, 6)}")
    print(f"  GRAVITY in mount frame  : {fmt(g_mount, 5)}  (this is what drives everything)")

    # ---- the arm subtree, straight out of the full model --------------------
    tree = ET.parse(FULL_XML)
    root = tree.getroot()
    arm = None
    for body in root.iter("body"):
        if body.get("name") == MOUNT:
            arm = copy.deepcopy(body)
            break
    if arm is None:
        raise RuntimeError(f"{MOUNT} not found in {FULL_XML}")

    # re-parent onto the world at the DERIVED pose
    arm.set("pos", fmt(pos))
    arm.set("quat", fmt(quat))
    for k in ("euler", "xyaxes", "zaxis"):
        arm.attrib.pop(k, None)

    # ---- hand: lump it in, or drop it --------------------------------------
    hand_body = None
    for body in arm.iter("body"):
        if body.get("name") == HAND_ROOT:
            hand_body = body
            break
    if hand_body is None:
        raise RuntimeError(f"{HAND_ROOT} not found under {MOUNT}")

    hand_mass = 0.0
    if args.no_hand:
        for child in list(hand_body.findall("body")):
            hand_body.remove(child)
        print("\n  hand: REMOVED (bare-arm calibration)")
    else:
        M, com, diag, iq, nb = lump_hand(m, d)
        own = float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_ROOT)])
        hand_mass = M - own
        for child in list(hand_body.findall("body")):
            hand_body.remove(child)
        for g in list(hand_body.findall("geom")):
            pass  # keep the mount's own geoms for the viewer
        inert = hand_body.find("inertial")
        if inert is None:
            inert = ET.SubElement(hand_body, "inertial")
        inert.set("mass", f"{M:.9g}")
        inert.set("pos", fmt(com))
        inert.set("diaginertia", fmt(diag))
        inert.set("quat", fmt(iq))
        print(f"\n  hand: LUMPED into {HAND_ROOT} ({nb} rigid bodies)")
        print(f"    link alone      : {own:.4f} kg")
        print(f"    + ORCA hand     : {hand_mass:.4f} kg")
        print(f"    = terminal link : {M:.4f} kg   com {fmt(com, 4)} m")

    # ---- assets -------------------------------------------------------------
    src_asset = root.find("asset")
    assets = []
    for mesh in src_asset.findall("mesh"):
        nm, f = mesh.get("name", ""), mesh.get("file", "")
        if nm.startswith("m_r_") and "arms/right" in f:
            mm = copy.deepcopy(mesh)
            mm.set("file", f.split("assets/meshes/")[-1])
            assets.append(mm)
    for mat in src_asset.findall("material"):
        if mat.get("name", "").startswith("mat_ra"):
            assets.append(copy.deepcopy(mat))

    # ---- assemble -----------------------------------------------------------
    mj = ET.Element("mujoco", {"model": "mabel_right_arm_sysid"})
    ET.SubElement(mj, "compiler", {"angle": "radian", "eulerseq": "XYZ",
                                   "meshdir": MESHDIR_REL})
    ET.SubElement(mj, "option", {"gravity": fmt(m.opt.gravity), "timestep": "0.002",
                                 "integrator": "implicitfast"})
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "mesh", {"scale": "0.001 0.001 0.001"})
    dv = ET.SubElement(default, "default", {"class": "ra_visual"})
    ET.SubElement(dv, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0",
                               "group": "2"})
    dc = ET.SubElement(default, "default", {"class": "ra_collision"})
    ET.SubElement(dc, "geom", {"type": "mesh", "contype": "0", "conaffinity": "0",
                               "group": "1", "rgba": "1 1 1 0"})
    # armature stays (it is a motor property, from motor_calibration); damping and
    # frictionloss are ZERO here on purpose -- the identification models friction
    # explicitly and cancels it (mabel_sysid.friction), rather than baking a
    # guess into the plant it is fitting.
    for cls, arm_val in (("motor_DM8009", "0.10"), ("motor_DM4340", "0.08"),
                         ("motor_DM4310", "0.05")):
        dd = ET.SubElement(default, "default", {"class": cls})
        ET.SubElement(dd, "joint", {"type": "hinge", "limited": "true",
                                    "damping": "0.0", "frictionloss": "0.0",
                                    "armature": arm_val})

    asset = ET.SubElement(mj, "asset")
    for a in assets:
        asset.append(a)
    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", {"name": "sun", "pos": "0 0 4", "dir": "0 0 -1",
                                "diffuse": "0.8 0.8 0.8"})
    wb.append(arm)

    act = ET.SubElement(mj, "actuator")
    force = {"motor_DM8009": "40", "motor_DM4340": "27", "motor_DM4310": "7"}
    for jname, cls, alias in ARM_JOINTS:
        fr = force[cls]
        ET.SubElement(act, "position", {"name": f"act_{alias}", "joint": jname,
                                        "kp": "100", "kv": "20",
                                        "forcerange": f"-{fr} {fr}"})

    out = ET.ElementTree(mj)
    ET.indent(out, space="  ")
    os.makedirs(os.path.dirname(OUT_XML), exist_ok=True)
    out.write(OUT_XML, encoding="unicode", xml_declaration=False)

    meta = {
        "source": os.path.relpath(FULL_XML, REPO_ROOT),
        "calibration_config": {"torso_rad": args.torso, "lift_m": args.lift,
                               "hand_attached": not args.no_hand},
        "mount_pos_world": pos.tolist(),
        "mount_quat_world": quat.tolist(),
        "gravity_in_mount_frame": g_mount.tolist(),
        "hand_mass_kg": hand_mass,
        "WARNING": ("The real robot MUST be parked at this torso angle and lift "
                    "height during collection. The torso is a hinge: changing it "
                    "rotates the arm's base frame and therefore the gravity "
                    "direction every identified parameter is referenced to."),
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {OUT_XML}")
    print(f"wrote {OUT_META}")


if __name__ == "__main__":
    main()

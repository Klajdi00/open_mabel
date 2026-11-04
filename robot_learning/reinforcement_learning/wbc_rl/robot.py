"""robot.py — NOVA for Isaac Lab / Newton: the training model, the articulation config, the tracked frames.

    python robot.py                    # data/nova_w4.xml -> USD (data/usd_w4/), print tracked-frame offsets + collision model
    python robot.py gripper            # the same for the OpenArm v1 gripper model (data/nova_grip.*, sonic.py)
    python robot.py [gripper] --rebuild  # first regenerate data/nova_*.{urdf,xml} from a robot description (NOVA_DESC)
    export_glb()                       # site/robot.glb: the sim's visual meshes for a 3D viewer

The derived training models are SHIPPED in data/ (nova_w4.* = ORCA v1 hands, nova_grip.* = OpenArm v1 grippers);
their meshes are referenced by relative path from simulation/mabel_mujoco/assets/meshes/ (identical parts) and
assets/meshes/openarm_gripper_v1/. Rebuilding them is optional and needs a robot description directory
(env NOVA_DESC) with urdf/nova.urdf, urdf/orca_v1_hands.xml, urdf/openarm_gripper_v1.xml and build_hand_variants.py.

Training model = the robot description URDF with
  * the ORCA v1 hands (MABEL's hands) in place of the v2 meshes,
    every hand joint frozen (fingers + ORCA wrist belong to the hand optimiser, not this controller);
  * a FLOATING base standing on three frictionless spheres at the swerve wheel contacts, so tipping is
    real physics: the drive is a force applied at the floor (env.py), like wheel traction;
  * no other collision geometry (arm/body contact is checked analytically, COLLISION below).
Actuated: lift_lower, lift_upper (the column; always commanded equal), torso pitch, neck yaw/pitch/roll,
both 7-DoF arms = 23 joints. URDF base_link faces -x; BASE_YAW turns it to REP-103 (x forward).

Tracked frames (link-local offsets, computed here by FK so they follow the URDF):
  head  : eye point 0.10 m forward / 0.08 m up of the `head` link (front face of the head mesh), axes x fwd, y left, z up
  wrists: the ORCA v1 wrist JOINT position, anatomical palm axes (x wrist->middle knuckle, y pinky->index knuckle)
          — the same construction motion_lib.py applies to the human hand.
"""
import math, os, re, xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DESC = os.path.abspath(os.path.expanduser(os.environ.get("NOVA_DESC", "")))   # optional: only for --rebuild
SRC_URDF = os.path.join(DESC, "urdf", "nova.urdf")
V1_HANDS = os.path.join(DESC, "urdf", "orca_v1_hands.xml")
TRAIN_URDF = os.path.join(HERE, "data", "nova_w4.urdf")         # v29+: the ORCA wrist hinge is actuated (4-joint wrist)
TRAIN_MJCF = os.path.join(HERE, "data", "nova_w4.xml")      # what the sim loads (URDF -> MuJoCo -> MJCF -> USD)
USD_DIR = os.path.join(HERE, "data", "usd_w4")
# OpenArm v1 parallel gripper on both arms (sonic.py): urdf/openarm_gripper_v1.xml seated by
# build_hand_variants.compose, fingers frozen closed (the opening is the gripper's own servo, not this controller)
GRIPPER_SRC = os.path.join(DESC, "urdf", "openarm_gripper_v1.xml")
GRIP_URDF = os.path.join(HERE, "data", "nova_grip.urdf")
GRIP_MJCF = os.path.join(HERE, "data", "nova_grip.xml")
GRIP_USD = os.path.join(HERE, "data", "usd_grip")
WRIST_CENTRE = (0.0, 0.0, -0.02)     # in <s>_hand_mount: on the J7 axis where J5 / J6 / J7 meet (MABEL end_effector.wrist_centre)
GRIPPER_STROKE = 0.044               # m per finger

WHEEL_R = 0.038                       # sphere at each wheel contact (wheel link origin is 0.038 m up)
CHASSIS_Z = 0.07775                   # root height when the spheres touch the floor
ARM = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(1, 8)]
NECK = ["neck_yaw", "neck_pitch", "neck_roll"]
LIFT = ["lift_lower", "lift_upper"]
ACTUATED = LIFT + ["torso"] + NECK + ARM
# the ORCA hand's own wrist hinge (Feetech HLS3930), parallel to arm_6 at arm_7 = 0: the 4th wrist joint. Solver-driven
# (wrist_ik), not a policy output; on NOVA it is commanded through the hand subsystem (the robot-side plant routes it).
HAND_WRIST = ["left_wrist", "right_wrist"]
LIFT_STROKE = 0.3175                  # per stage
LIFT_VMAX = 0.03                      # m/s, the WHOLE column, hard cap -> each stage moves at half
LIFT_VLIM = LIFT_VMAX                 # 3 cm/s: the full cap is allowed for the lift; 80 % margin elsewhere
LIFT_MARGIN = 0.02                    # m per stage kept off each hard stop: the column stays 4 cm inside its
                                      # measured 317.5-952.5 mm travel (lift datasheet) at top and bottom
BASE_VMAX, BASE_WMAX = 0.6, 1.2       # m/s, rad/s the policy may ask for (hardware ceiling 1.04 / 3.0)
BASE_ACCEL = 0.8                      # m/s^2 nova_hw config BASE accel (body twist); the tip governor caps it lower
BASE_DECEL = 1.5                      # x accel when slowing (config decel_ratio), never past the governor's accel_max
BASE_ALPHA = 2.5                      # rad/s^2 config yaw_accel
STEER_COS_GATE = math.radians(30.0)   # base.py: a module drives at cos(turn of its COMMANDED heading), 0 past 30 deg in one tick

REST = {"left_arm_1": 0.041, "left_arm_2": -0.124, "left_arm_4": 1.49,      # the robot's REST_POSE
        "right_arm_2": 0.124, "right_arm_4": -1.571}
# working pose: elbows bent ~70 deg, forearms forward — where teleop actually lives; used for resets
# (kp, kd, setpoint slew rad/s) as NOVA runs them: NOVA's calibration file (both arms equal)
HW_GAINS = {"arm_1": (37.0, 3.0, 2.5), "arm_2": (25.0, 4.0, 2.5), "arm_3": (18.0, 1.6, 3.0), "arm_4": (72.0, 3.0, 3.0),
            "arm_5": (10.0, 1.05, 4.0), "arm_6": (6.0, 0.8, 4.0), "arm_7": (13.0, 0.97, 4.0),
            "torso": (90.0, 4.0, 1.5), "neck": (6.0, 0.5, 2.5),
            "neck_servo": (40.0, 1.0, 6.28), "hand_wrist": (40.0, 1.0, 0.95 * 4.71)}  # neck pitch/roll: Feetech HLS3915 firmware position loop (stiff, its own P;
                                              # kp/kd here approximate it), no shaper / v_ff: goals go straight to the servo
# NOVA's neck servo profile (calibration file, neck/feetech/servo: speed 4095, acc 200, limit 500 permille):
# 4095 steps/s = 6.28 rad/s, 200 x 100 steps/s^2 = 30.7 rad/s^2. Bench (neck tuning): a 15 deg step rises in
# 105 ms, settles in 181 ms, 1 % overshoot, 0 ticks steady-state error: the profile alone takes 2 sqrt(0.26/30.7) =
# 184 ms, so the servo loop is stiff and holds the head (no sag); kp 40 / kd 1 reproduces that
NECK_SERVO = dict(speed=6.28, acc=30.7)
# ORCA wrist servo profile (calibration file, hand/orca-*/servo: speed 3000 steps/s = 4.6 rad/s, acc 80 x 100 = 12.3 rad/s^2),
# speed capped at HEADROOM x the datasheet
HAND_SERVO = dict(speed=0.95 * 4.71, acc=12.3)

# Gravity feed-forward ceiling per joint group as NOVA applies it (the robot's gravity-comp config ff_max_nm):
# the robot-side plant clips the model torque here; the neck gets none (yaw: no g_scale, pitch/roll: servos).
FF_MAX_NM = {"arm_1": 30.0, "arm_2": 30.0, "arm_3": 12.0, "arm_4": 14.0, "arm_5": 6.0, "arm_6": 6.0, "arm_7": 6.0,
             "torso": 10.0, "neck_yaw": 0.0, "neck_pitch": 0.0, "neck_roll": 0.0}

# Physical limits per joint group: (max speed rad/s, peak torque N m, rated = continuous torque N m, source).
# Training keeps every joint at or under SAFETY x these (solver caps + termination + a barrier penalty).
# SAFETY 0.8 until v35; since then the full datasheet ratings (match real life, no 80 % cap).
SAFETY = 1.0
HEADROOM = 0.95                       # solver caps, slews and the lift command sit 5 % under SAFETY x limit: zero-tolerance checks
JOINT_SPEC = {
    "arm_1": (18.1, 40.0, 20.0, "HM9816G9 datasheet motors/docs/hm: 172.9 rpm max, peak 40, rated 20 N m"),
    "arm_2": (18.1, 40.0, 20.0, "HM9816G9, as arm_1"),
    "arm_3": (5.50, 27.0, 9.0, "DM-J4340-2EC manual motors/docs/damiao p.9: peak 27, rated 9 N m; 100 rpm no-load is the 48 V figure, NOVA's arm bus is 24 V (HM9816G9 18-36 V): 52.5 rpm"),
    "arm_4": (5.50, 27.0, 9.0, "DM-J4340-2EC, as arm_3"),
    "arm_5": (20.9, 7.0, 3.0, "DM-J4310-2EC manual motors/docs/damiao p.9: 200 rpm no-load at 24 V, peak 7, rated 3 N m"),
    "arm_6": (20.9, 7.0, 3.0, "DM-J4310-2EC, as arm_5"),
    "arm_7": (20.9, 7.0, 3.0, "DM-J4310-2EC, as arm_5"),
    "torso": (12.6, 400.0, 100.0, "DM-J10422P-2EC manual motors/docs/damiao: 120 rpm no-load, peak 400, rated 100"),
    "neck_yaw": (20.9, 7.0, 3.0, "DM-J4310-2EC, as arm_5"),
    "neck_pitch": (10.5, 1.39, 0.44, "HLS3915M-C001 spec servos/docs: 100 rpm no-load, stall 14.2 kg cm, rated 4.5 kg cm"),
    "neck_roll": (10.5, 1.39, 0.44, "HLS3915M-C001, as neck_pitch"),
    "hand_wrist": (4.71, 3.43, 0.85, "HLS3930M-C001 spec neck/docs: 45 rpm no-load, stall 35 kg cm, rated 8.7 kg cm"),
}
OVERLOAD_SOFT = ("neck_pitch", "neck_roll")   # holding the head level already needs ~0.44 N m: penalised, not terminated


def gain_group(g):
    """JOINT_SPEC group -> HW_GAINS key."""
    return "neck_servo" if g in ("neck_pitch", "neck_roll") else "neck" if g.startswith("neck") else g


def joint_group(name):
    return "torso" if name == "torso" else name if name.startswith("neck") else "hand_wrist" if name in HAND_WRIST else "arm_" + name[-1]


READY = {"left_arm_1": -0.35, "left_arm_2": 0.15, "left_arm_4": 0.30,     # exact mirror of the right (env.mirror_batch)
         "right_arm_1": 0.35, "right_arm_2": -0.15,
         "right_arm_4": -0.30}


# ---------------------------------------------------------------- URDF surgery
def _subtree(root, top):
    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    links, joints, stack = set(), [], [top]
    while stack:
        l = stack.pop()
        for j in kids.get(l, []):
            c = j.find("child").get("link"); links.add(c); joints.append(j); stack.append(c)
    return links, joints


def _compose_grippers():
    """nova.urdf with the OpenArm v1 gripper on both hand mounts (the rig the Control Center draws)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_hand_variants", os.path.join(DESC, "build_hand_variants.py"))
    bhv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bhv)
    return bhv.compose({"left": "gripper", "right": "gripper"})


def _need_desc():
    if not os.environ.get("NOVA_DESC") or not os.path.exists(SRC_URDF):
        raise FileNotFoundError("rebuilding the training model needs NOVA_DESC=<robot description dir> with urdf/nova.urdf; "
                                "the shipped data/nova_w4.xml / data/nova_grip.xml do not need it")


def build_urdf(path=TRAIN_URDF, ee="orca"):
    _need_desc()
    root = ET.parse(SRC_URDF).getroot() if ee == "orca" else _compose_grippers()
    for j in root.findall("joint"):                          # floating base: drop world
        if j.find("parent").get("link") == "world": root.remove(j)
    for l in root.findall("link"):
        if l.get("name") == "world": root.remove(l)
    if ee == "orca":
        for mount in ("l_hand_mount", "r_hand_mount"):        # v2 hand out
            links, joints = _subtree(root, mount)
            for j in joints: root.remove(j)
            for l in root.findall("link"):
                if l.get("name") in links: root.remove(l)
        for e in ET.parse(V1_HANDS).getroot():                # v1 hand in
            root.append(e)
    keep = ACTUATED + (HAND_WRIST if ee == "orca" else [])
    for j in root.findall("joint"):                           # freeze hands / gripper fingers and wheels
        n = j.get("name")
        if n not in keep and j.get("type") != "fixed":
            j.set("type", "fixed")
            for t in ("axis", "limit", "mimic"):
                if j.find(t) is not None: j.remove(j.find(t))
    for l in root.findall("link"):                            # collisions: only the wheel spheres
        for c in l.findall("collision"): l.remove(c)
        if l.get("name") in ("fl_wheel", "fr_wheel", "b_wheel"):
            c = ET.SubElement(l, "collision"); ET.SubElement(c, "origin", xyz="0 0 0", rpy="0 0 0")
            g = ET.SubElement(c, "geometry"); ET.SubElement(g, "sphere", radius=str(WHEEL_R))
        for m in l.iter("mesh"):
            m.set("filename", re.sub(r"^package://[^/]+", DESC, m.get("filename")))
    for j in root.findall("joint"):                           # the URDF's effort/velocity are placeholders (100 / 10)
        lim = j.find("limit")
        if lim is not None and j.get("name") in keep: lim.set("velocity", "10")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ET.ElementTree(root).write(path)
    return path


def build_mjcf(path=TRAIN_MJCF, ee="orca"):
    """URDF -> MJCF through MuJoCo's own parser. The chassis gets a free joint, then static bodies are fused
    (hand + wheels into their movers); meshes are visual only, the three wheel spheres the only colliders."""
    import re, mujoco
    src = open(build_urdf(TRAIN_URDF if ee == "orca" else GRIP_URDF, ee)).read()
    hint = '<mujoco><compiler discardvisual="false" fusestatic="{}" strippath="false"/></mujoco>'
    src = re.sub(r"(<robot[^>]*>)", r"\1" + hint.format("false"), src, count=1)
    # MuJoCo names a mesh after its file stem, so meshes/right/3_shoulder_to_elbow.stl silently became the LEFT
    # arm's mesh (same stem). Point the right arm at uniquely named links to its own files.
    alias = os.path.join(HERE, "data", "mesh_alias"); os.makedirs(alias, exist_ok=True)
    def own(mo):
        f = mo.group(1); dst = os.path.join(alias, "right_" + os.path.basename(f))
        if not os.path.lexists(dst): os.symlink(f, dst)
        return f'filename="{dst}"'
    src = re.sub(r'filename="([^"]*/meshes/right/[^"/]+\.stl)"', own, src)
    tmp = path + ".tmp.urdf"
    open(tmp, "w").write(src)
    mujoco.mj_saveLastXML(path, mujoco.MjModel.from_xml_path(tmp)); os.remove(tmp)
    root = ET.parse(path).getroot()
    chassis = root.find("worldbody").find("body")
    assert chassis.get("name") == "mabel_chassis", chassis.get("name")
    chassis.set("pos", f"0 0 {CHASSIS_Z}")
    chassis.insert(0, ET.Element("freejoint", name="root"))
    root.find("compiler").set("fusestatic", "true")
    for g in root.iter("geom"):
        if g.get("type", "sphere") == "sphere":                       # MuJoCo omits type for its default (sphere)
            g.attrib.update(contype="1", conaffinity="1", friction="0 0 0", group="0")
        else:
            g.attrib.update(contype="0", conaffinity="0", group="1")
    ET.ElementTree(root).write(path)
    mujoco.mj_saveLastXML(path, mujoco.MjModel.from_xml_path(path))   # re-save fused
    return path


def build_usd(ee="orca", rebuild=False):
    """MJCF -> USD with Isaac Lab's kit-free converter. Run in its own process (python robot.py [gripper]): the
    importer's fallback PhysX schemas must register before anything builds the USD schema registry."""
    import isaacsim.asset.importer.utils.impl.physx_types  # noqa: F401  (registers the fallback schemas)
    import shutil
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    usd_dir = USD_DIR if ee == "orca" else GRIP_USD
    shutil.rmtree(usd_dir, ignore_errors=True)                         # it would otherwise write nova_w4_2/, _3/, ...
    mj = TRAIN_MJCF if ee == "orca" else GRIP_MJCF
    if rebuild: mj = build_mjcf() if ee == "orca" else build_mjcf(GRIP_MJCF, "gripper")
    c = MjcfConverter(MjcfConverterCfg(asset_path=mj, usd_dir=usd_dir, fix_base=False, self_collision=False,
                                       force_usd_conversion=True))
    return c.usd_path


# ---------------------------------------------------------------- FK on the training URDF (numpy, for offsets)
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


class FK:
    def __init__(self, urdf=TRAIN_URDF):
        root = ET.parse(urdf).getroot(); self.J = {}
        for j in root.findall("joint"):
            o = j.find("origin"); v = lambda k, d: np.array([float(x) for x in (o.get(k, d) if o is not None else d).split()])
            ax = j.find("axis")
            self.J[j.find("child").get("link")] = dict(name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"),
                                                       xyz=v("xyz", "0 0 0"), R=_rpy(*v("rpy", "0 0 0")),
                                                       axis=np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else None,
                                                       lim=(float(j.find("limit").get("lower", 0)), float(j.find("limit").get("upper", 0)))
                                                       if j.find("limit") is not None else None)
        self.joint_child = {d["name"]: c for c, d in self.J.items()}

    def pose(self, link, q=None):
        q = q or {}
        if link not in self.J: return np.eye(3), np.zeros(3)
        d = self.J[link]; R, p = self.pose(d["parent"], q)
        p = p + R @ d["xyz"]; R = R @ d["R"]
        a = float(q.get(d["name"], 0.0))
        if d["type"] in ("revolute", "continuous") and a:
            k = d["axis"] / np.linalg.norm(d["axis"]); K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = R @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)
        elif d["type"] == "prismatic" and a:
            p = p + R @ (d["axis"] * a)
        return R, p

    def rel(self, link, ref):
        """Pose of `link` in `ref`'s frame (fixed joints only between them)."""
        Rr, pr = self.pose(ref); Rl, pl = self.pose(link)
        return Rr.T @ Rl, Rr.T @ (pl - pr)


def tracked_offsets(fk=None, ee="orca"):
    """{name: (body, pos_in_body (3,), R_body_frame (3,3))} for head / left wrist / right wrist.
    Gripper: the wrist CENTRE on <s>_hand_mount (the operator's wrist maps there, as in MABEL's Retarget Lab) and
    the ORCA palm's anatomical axes at ORCA wrist 0, fixed in the mount: x = approach, so the operator's palm
    orientation carries over to the gripper unchanged (the lab's gripper_frame rule)."""
    fk = fk or FK()
    out = {}
    # head link axes: x right, y up, z back -> anatomical (x fwd, y left, z up)
    R_head = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    out["head"] = ("head", R_head @ np.array([0.10, 0.0, 0.08]), R_head)
    if ee == "gripper":
        orca = FK(TRAIN_URDF); T = tracked_offsets(orca)
        for s, side in (("l", "left"), ("r", "right")):
            body, _, C = T[side + "_wrist"]
            out[side + "_wrist"] = (f"{s}_hand_mount", np.array(WRIST_CENTRE), orca.rel(body, f"{s}_hand_mount")[0] @ C)
        return out
    for s, side in (("l", "left"), ("r", "right")):
        ref = fk.joint_child[f"{side}_wrist"]                         # the palm, after the ORCA wrist hinge
        w = np.zeros(3)                                               # wrist JOINT origin = child link origin
        k = {f: fk.rel(fk.joint_child[f"{side}_{f}_mcp"], ref)[1] for f in ("index", "middle", "pinky")}
        x = k["middle"] - w; x /= np.linalg.norm(x)
        y = k["index"] - k["pinky"]; y -= (y @ x) * x; y /= np.linalg.norm(y)
        out[side + "_wrist"] = (ref, w, np.stack([x, y, np.cross(x, y)], -1))
    return out


# ---------------------------------------------------------------- wrist solver (numpy or torch: xp)
# The palm orientation is solved exactly by the three wrist joints (arm_5 forearm roll, arm_6 pitch +-45 deg, arm_7 roll)
# for the forearm as MEASURED (encoders, so upstream sag is corrected); the policy places the arm and adds a small
# residual. Training (env.py) and the robot (deploy.py) call these same functions.
def wrist_chain(fk, side):
    """Fixed rotations / axes from the URDF base to the elbow link; the four wrist joints as (fixed frame, axis):
    arm_5, arm_6, arm_7 and the ORCA hinge; C = the palm axes in the palm link."""
    s = side[0]; link = f"{s}_elbow"; chain = []
    while link in fk.J:
        d = fk.J[link]
        chain.append((d["R"], d["axis"] / np.linalg.norm(d["axis"]) if d["type"] == "revolute" else None, d["name"]))
        link = d["parent"]
    palm = fk.joint_child[f"{side}_wrist"]
    F8 = fk.J[palm]["R"]; link = fk.J[palm]["parent"]                 # hand_mount -> ... -> tower (fixed), then the hinge
    while link != f"{s}_hand_mount":
        F8 = fk.J[link]["R"] @ F8; link = fk.J[link]["parent"]
    z = np.array([0.0, 0.0, 1.0]); ax8 = fk.J[palm]["axis"] / np.linalg.norm(fk.J[palm]["axis"])
    joints = [(fk.J[f"{s}_forearm"]["R"], z), (fk.J[f"{s}_wrist"]["R"], z), (fk.J[f"{s}_hand_mount"]["R"], z), (F8, ax8)]
    tr = tracked_offsets(fk)[side + "_wrist"]
    assert tr[0] == palm
    return dict(chain=chain[::-1], joints=joints, C=tr[2],
                names=[f"{side}_arm_5", f"{side}_arm_6", f"{side}_arm_7", f"{side}_wrist"])


_TC = {}


def _tt(x, ref):
    """numpy constant -> torch tensor like `ref`, cached (a host-to-GPU copy per call made the solver 10x slower)."""
    x = np.asarray(x, dtype=float); key = (x.tobytes(), x.shape, ref.dtype, ref.device)
    if key not in _TC:
        import torch
        _TC[key] = torch.tensor(x, dtype=ref.dtype, device=ref.device)
    return _TC[key]


def _axis_rot(k, a, xp):
    if xp is np: a = np.asarray(a, dtype=float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    K, K2, I = (xp.asarray(K, dtype=a.dtype), xp.asarray(K @ K, dtype=a.dtype), xp.asarray(np.eye(3), dtype=a.dtype)) if xp is np else \
        tuple(_tt(x, a) for x in (K, K @ K, np.eye(3)))
    return I + xp.sin(a)[..., None, None] * K + (1 - xp.cos(a))[..., None, None] * K2


def base_ramp(p, t, acc, dec, dt, xp):
    """nova_hw subsystems/base._accel_limit on the body (vx, vy), (..., 2): within 90 deg of the current direction the
    new direction is taken at once and only the speed ramps (acc up, dec down); from rest, to rest or reversing the
    vector ramps through zero."""
    sp = xp.hypot(p[..., 0:1], p[..., 1:2]); st = xp.hypot(t[..., 0:1], t[..., 1:2])
    a = xp.where(st < sp, dec, acc) * dt
    s = sp + (st - sp).clip(-a, a)
    same = (st > 1e-9) & (sp > 1e-9) & (p[..., 0:1] * t[..., 0:1] + p[..., 1:2] * t[..., 1:2] >= 0)
    d = t - p; k = (a / xp.hypot(d[..., 0:1], d[..., 1:2]).clip(min=1e-9)).clip(max=1.0)
    return xp.where(same, t / st.clip(min=1e-9) * s, p + k * d)


def chain_rot(chain, q, xp):
    """Orientation of the last link in the URDF base frame; q: {joint name: (...) angle}."""
    R = None
    for Rf, ax, name in chain:
        Rf = xp.asarray(Rf) if xp is np else _tt(Rf, next(iter(q.values())))
        R = Rf if R is None else R @ Rf
        if ax is not None: R = R @ _axis_rot(ax, q[name], xp)
    return R


def wrist_ik(A, Rt, wc, q, lo, hi, xp, iters=6, w_mid=(0.0, 0.3, 0.0, 0.3)):
    """Wrist joints q (...,4) so that A F5 R(q5) F6 R(q6) F7 R(q7) F8 R(q8) C = Rt (A: the elbow link frame), clamped to
    [lo, hi], warm-started at q. Damped Gauss-Newton; w_mid pulls the two pitch joints (arm_6 +-45 deg, the ORCA hinge)
    toward their centres, so the bend is shared like a human wrist and roll goes to the forearm."""
    T = (lambda x: xp.asarray(x)) if xp is np else (lambda x: _tt(x, q))
    tr = lambda x: x.swapaxes(-1, -2) if xp is np else x.transpose(-1, -2)
    clip = (lambda x, a, b: np.minimum(np.maximum(x, a), b)) if xp is np else (lambda x, a, b: xp.minimum(xp.maximum(x, a), b))
    M = tr(A) @ Rt @ tr(T(wc["C"]))
    F = [T(f) for f, _ in wc["joints"]]; K = [T(k) for _, k in wc["joints"]]
    W = T(np.array(w_mid, dtype=float)); I3 = T(1e-4 * np.eye(3)); I4 = T(np.eye(len(wc["joints"])))
    for _ in range(iters):
        Rc, cols = None, []
        for i, ((_, k), Fi, Ki) in enumerate(zip(wc["joints"], F, K)):
            Rc = Fi if Rc is None else Rc @ Fi
            cols.append((Rc @ Ki[..., None])[..., 0])
            Rc = Rc @ _axis_rot(k, q[..., i], xp)
        E = M @ tr(Rc)                                                          # remaining rotation, left frame
        v = xp.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1) / 2
        sn = (v * v).sum(-1) ** 0.5; cs = (E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2] - 1) / 2
        th = xp.arctan2(sn, cs) if xp is np else xp.atan2(sn, cs)
        e = v * (th / (sn + 1e-9))[..., None]
        J = xp.stack([c + 0 * e for c in cols], -1)
        # damped pseudo-inverse for the orientation (exact), the centring preference only in its null space; a joint
        # at a limit that would be pushed further is dropped from the solve and the others take its share
        act = q * 0 + 1
        for _ in range(2):
            Ja = J * act[..., None, :]
            Jp = tr(Ja) @ xp.linalg.inv(Ja @ tr(Ja) + I3)
            dq = ((Jp @ e[..., None])[..., 0] + ((I4 - Jp @ Ja) @ (-W * q)[..., None])[..., 0]) * act
            out = ((q <= lo + 1e-4) & (dq < 0)) | ((q >= hi - 1e-4) & (dq > 0))
            act = act * (1 - out * 1.0) if xp is np else act * (~out).to(act.dtype)
        q = clip(q + dq, lo, hi)
    return q


def hand_guard(q, safe, clear, band, xp):
    """Collision guard on the wrist solver's command (env.py and deploy.py): q (..., 2, 4) the solution per hand,
    clear (..., 2) the hand's measured clearance (m), safe (..., 2, 4) its last command with room (updated in place).
    Above band[1] the solution passes; below it the command slides back towards `safe`, reaching it at band[0]."""
    s = xp.clip((clear - band[0]) / (band[1] - band[0]), 0.0, 1.0)[..., None]
    out = safe + s * (q - safe)
    ok = clear >= band[1]
    safe[ok] = q[ok]
    return out


def wrist_err(A, Rt, wc, q, xp):
    """Remaining rotation (rotation vector, left = world frame) between the solved palm and Rt."""
    tr = lambda x: x.swapaxes(-1, -2) if xp is np else x.transpose(-1, -2)
    T = (lambda x: xp.asarray(x)) if xp is np else (lambda x: _tt(x, q))
    Rc = A
    for i, (f, k) in enumerate(wc["joints"]):
        Rc = Rc @ T(f) @ _axis_rot(k, q[..., i], xp)
    E = Rt @ tr(Rc @ T(wc["C"]))
    v = xp.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1) / 2
    sn = (v * v).sum(-1) ** 0.5; cs = (E[..., 0, 0] + E[..., 1, 1] + E[..., 2, 2] - 1) / 2
    th = xp.arctan2(sn, cs) if xp is np else xp.atan2(sn, cs)
    return v * (th / (sn + 1e-9))[..., None]


# ---------------------------------------------------------------- collision model (meshes -> distance fields)
# NOVA's rule: only the fingers of the two hands may touch each other. Arm to body, hand to arm,
# hand to body, anything to the lift or base is a crash. Built from the visual meshes:
#   every part gets a distance field: unsigned distance to its mesh surface on an SDF_RES grid in its body frame;
#   every arm/hand part is also wrapped in spheres that enclose all of its vertices (k-means, radius = farthest vertex).
# Gap between part A and part B = min over A's spheres of (B's distance field at the centre - radius); conservative by
# how loosely the spheres fit (~1 cm), exact on B's side.
BODY_PARTS = {"chassis": ("mabel_chassis", None), "lift_mid": ("lift_mid", None), "lift_top": ("lift_top", None),
              "torso": ("torso", None), "neck_1": ("neck_1", None), "neck_2": ("neck_2", None), "head": ("head", None)}
ARM_PARTS = {"shoulder_1": ("shoulder_1", None, 16), "shoulder_2": ("shoulder_2", None, 16), "upper": ("shoulder_3", None, 32),
             "elbow": ("elbow", None, 32), "forearm": ("forearm", None, 24), "wrist": ("wrist", None, 16),
             "hand_wrist": ("hand_mount", ("arm_to_hand", "tower"), 24), "palm": ("palm__anchor", ("palm",), 20),
             "fingers": ("palm__anchor", ("thumb", "index", "middle", "ring", "pinky"), 48)}
HAND = ("hand_wrist", "palm", "fingers")
SHOULDER = ("shoulder_1", "shoulder_2", "upper")
SDF_RES, SDF_MARGIN = 0.008, 0.10
SDF_CACHE = os.path.join(HERE, "data", "collision_sdf_w4.npz")
# gripper (sonic.py): everything below J7 is fused into <s>_hand_mount; the fingers are drawn closed and swept to
# fully open (the mirrored finger slides along mount -z, the other +z). Gripper jaws may touch the other jaws only.
GRIP_ARM_PARTS = {k: v for k, v in ARM_PARTS.items() if k not in ("hand_wrist", "palm", "fingers")}
GRIP_ARM_PARTS.update({"gripper": ("hand_mount", ("link7", "hand"), 24), "fingers": ("hand_mount", ("finger",), 24)})
GRIP_HAND = ("gripper", "fingers")
GRIP_SDF = os.path.join(HERE, "data", "collision_sdf_grip.npz")


def _mesh_parts(m, body):
    """[(mesh name, vertices, faces)] of a MuJoCo body's visible meshes, in the body frame."""
    import mujoco
    b, out, seen = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body), [], set()
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_rgba[g, 3] == 0: continue
        k = m.geom_dataid[g]
        if k in seen: continue
        seen.add(k)
        Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ Rg.reshape(3, 3).T + m.geom_pos[g]
        f = m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]]
        out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, k), v.astype(np.float64), f))
    return out


def _kmeans(v, n, surf=None):
    """n spheres enclosing every vertex of v; centres from k-means on the (area-uniform) surface samples."""
    v_all, v = v, (surf if surf is not None else v)
    c = [v.mean(0)]
    for _ in range(n - 1):
        c.append(v[((v[:, None] - np.array(c)[None]) ** 2).sum(-1).min(1).argmax()])
    c = np.array(c)
    for _ in range(25):
        lab = ((v[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([v[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(n)])
    v = np.concatenate([v_all, v])
    lab = ((v[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    return [(c[i], float(np.linalg.norm(v[lab == i] - c[i], axis=1).max())) for i in range(n) if (lab == i).any()]


def _part_mesh(m, body, keys):
    ps = [(v, f) for nm, v, f in _mesh_parts(m, body) if keys is None or any(k in nm for k in keys)]
    V, F, o = [], [], 0
    for v, f in ps:
        V.append(v); F.append(f + o); o += len(v)
    return np.concatenate(V), np.concatenate(F)


def _sdf(v, f):
    """Unsigned distance to the surface (area-sampled points + vertices) on a grid around the part."""
    import trimesh
    from scipy.spatial import cKDTree
    pts = np.concatenate([v, trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 60000, seed=0)[0]])
    lo, hi = v.min(0) - SDF_MARGIN, v.max(0) + SDF_MARGIN
    n = np.ceil((hi - lo) / SDF_RES).astype(int) + 1
    g = np.stack(np.meshgrid(*[lo[i] + SDF_RES * np.arange(n[i]) for i in range(3)], indexing="ij"), -1)
    return lo, cKDTree(pts).query(g.reshape(-1, 3), workers=-1)[0].reshape(n).astype(np.float32)


def _swept_fingers(m, body):
    """Gripper finger meshes (in the mount frame) closed and fully open, one mesh: what the jaws can occupy."""
    V, F, o = [], [], 0
    for nm, v, f in _mesh_parts(m, body):
        if "finger" not in nm: continue
        for t in (0.0, 0.5, 1.0):
            V.append(v + np.array([0.0, 0.0, (-1.0 if "flip" in nm else 1.0) * t * GRIPPER_STROKE])); F.append(f + o); o += len(v)
    return np.concatenate(V), np.concatenate(F)


def collision_model(pad=0.003, ee="orca"):
    """parts: [{name, side, link, origin, res, sdf (x,y,z grid)}]; spheres: [(link, centre, radius, side, part)];
    pairs: (sphere, part) bool — which sphere/part gaps count as a crash. Cached in data/collision_sdf*.npz."""
    mj, cache, arm, hand = (TRAIN_MJCF, SDF_CACHE, ARM_PARTS, HAND) if ee == "orca" else (GRIP_MJCF, GRIP_SDF, GRIP_ARM_PARTS, GRIP_HAND)
    src = [p for p in (mj, __file__) if os.path.exists(p)]
    if os.path.exists(cache) and os.path.getmtime(cache) > max(os.path.getmtime(p) for p in src):
        z = np.load(cache, allow_pickle=True)
        return list(z["parts"]), list(z["spheres"]), z["pairs"]
    import mujoco                                     # only to rebuild the cache
    m = mujoco.MjModel.from_xml_path(mj)
    parts, spheres = [], []
    for name, (body, keys) in BODY_PARTS.items():
        lo, d = _sdf(*_part_mesh(m, body, keys))
        parts.append(dict(name=name, side=None, link=body, origin=lo, res=SDF_RES, sdf=d))
    for s in ("l", "r"):
        for name, (link, keys, n) in arm.items():
            link = ("left_" if s == "l" else "right_") + link if link == "palm__anchor" else f"{s}_{link}"   # after the ORCA hinge
            v, f = _swept_fingers(m, link) if ee == "gripper" and name == "fingers" else _part_mesh(m, link, keys)
            lo, d = _sdf(v, f)
            parts.append(dict(name=name, side=s, link=link, origin=lo, res=SDF_RES, sdf=d))
            import trimesh
            surf = trimesh.sample.sample_surface(trimesh.Trimesh(v, f, process=False), 6000, seed=1)[0]
            spheres += [(link, c, r + pad, s, name) for c, r in _kmeans(v, n, surf)]
    pairs = np.zeros((len(spheres), len(parts)), bool)
    for i, (_, _, _, s, a) in enumerate(spheres):
        for j, p in enumerate(parts):
            if p["side"] is None:                    # arm vs body: the shoulder sits on the torso
                pairs[i, j] = not (a in SHOULDER and p["name"] == "torso")
            elif p["side"] != s:                     # left vs right (one direction is enough): all but fingers-fingers
                pairs[i, j] = s == "l" and not (a == "fingers" and p["name"] == "fingers") and not (a in SHOULDER and p["name"] in SHOULDER)
            else:                                    # own arm: the hand may not reach its upper arm or shoulder
                pairs[i, j] = a in hand and p["name"] in SHOULDER
    np.savez(cache, parts=np.array(parts, dtype=object), spheres=np.array(spheres, dtype=object), pairs=pairs)
    return parts, spheres, pairs


GLB_FIXED = {"mabel_chassis": 12_000, "lift_top": 8_000}       # big, plain parts: a small fixed budget


def export_glb(path=os.path.join(HERE, "site", "robot.glb"), faces=260_000):
    """The sim's visual meshes as one GLB for the site's 3D viewer: one node per MuJoCo body (= Isaac body, same
    name), mesh in that body's frame, vertex colours from the MJCF, decimated to ~`faces` triangles in total. The
    chassis and lift column get a fixed budget, the arms, hands and head share the rest (the viewer zooms on them)."""
    import mujoco, trimesh, fast_simplification, open3d as o3d
    m = mujoco.MjModel.from_xml_path(TRAIN_MJCF)
    parts, seen = {}, set()
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_rgba[g, 3] == 0: continue
        key = (m.geom_bodyid[g], m.geom_dataid[g], *np.round(m.geom_pos[g], 5), *np.round(m.geom_quat[g], 5))
        if key in seen: continue                                            # the URDF lists some visuals twice
        seen.add(key)
        k = m.geom_dataid[g]; va, nv, fa, nf = m.mesh_vertadr[k], m.mesh_vertnum[k], m.mesh_faceadr[k], m.mesh_facenum[k]
        Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        v = m.mesh_vert[va:va + nv] @ Rg.reshape(3, 3).T + m.geom_pos[g]
        parts.setdefault(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]), []).append(
            (v, m.mesh_face[fa:fa + nf], m.geom_rgba[g]))
    total = sum(len(f) for b, ps in parts.items() if b not in GLB_FIXED for _, f, _ in ps)
    share = (faces - sum(GLB_FIXED.values())) / total
    scene = trimesh.Scene()
    for body, ps in parts.items():
        V, F, C = [], [], []
        nb = sum(len(f) for _, f, _ in ps)
        for v, f, rgba in ps:
            want = max(60, int(len(f) * (GLB_FIXED[body] / nb if body in GLB_FIXED else share)))  # tiny parts keep >= 60
            if len(f) > 1.1 * want:
                v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_count=want)
                if len(f2) > 2 * want:                                       # many tiny islands: cluster vertices instead
                    t = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
                    t = t.simplify_vertex_clustering(float(np.ptp(v, 0).max()) / max(8.0, math.sqrt(want) * 0.6))
                    v2, f2 = np.asarray(t.vertices), np.asarray(t.triangles)
                v, f = v2, f2
            F.append(f + sum(len(x) for x in V)); V.append(v); C.append(np.tile((rgba * 255).astype(np.uint8), (len(v), 1)))
        mesh = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), vertex_colors=np.concatenate(C), process=False)
        scene.add_geometry(mesh, node_name=body, geom_name=body)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene.export(path)
    return path, sum(len(g.faces) for g in scene.geometry.values())


def articulation_cfg(ee="orca"):
    """The Isaac Lab ArticulationCfg (imports Isaac Lab, so call it only inside the sim process)."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    import glob
    usd_dir, stem = (USD_DIR, "nova_w4") if ee == "orca" else (GRIP_USD, "nova_grip")
    hits = glob.glob(os.path.join(usd_dir, "**", "*.usd*"), recursive=True)
    assert hits, f"no USD yet: run  python robot.py {ee}  once"
    usd_path = os.path.join(usd_dir, stem, stem + ".usda")
    wr = HAND_WRIST if ee == "orca" else []
    q0 = {k: 0.0 for k in ACTUATED + wr}; q0.update(READY); q0["lift_lower"] = q0["lift_upper"] = LIFT_STROKE / 2
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path, activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, CHASSIS_Z + 0.002), rot=(0.0, 0.0, 0.0, 1.0), joint_pos=q0),
        actuators={
            # NOVA's own PD (NOVA's calibration file, arm/openarm-dual, torso, neck gains) with its
            # gravity feed-forward added by env.py, as in teleop (gravityComp on). Gains randomised +-25 % (env.py).
            # Torque and speed are capped in the solver at SAFETY x the datasheet (JOINT_SPEC).
            **{g: ImplicitActuatorCfg(joint_names_expr=[f".*_{g}" if g.startswith("arm") else ".*_wrist" if g == "hand_wrist" else g],
                                      stiffness=HW_GAINS[gain_group(g)][0], damping=HW_GAINS[gain_group(g)][1],
                                      effort_limit_sim=SAFETY * JOINT_SPEC[g][1], velocity_limit_sim=HEADROOM * SAFETY * JOINT_SPEC[g][0],
                                      armature={"arm_1": 0.05, "arm_2": 0.05, "arm_3": 0.03, "arm_4": 0.03, "torso": 0.2}.get(g, 0.01))
               for g in JOINT_SPEC if g != "hand_wrist" or ee == "orca"},
            "lift": ImplicitActuatorCfg(joint_names_expr=["lift_.*"], stiffness=40000.0, damping=4000.0,
                                        effort_limit_sim=3000.0, armature=1.0,                    # a screw: never faster than its motor
                                        velocity_limit_sim=HEADROOM * 0.5 * LIFT_VLIM),
        },
    )


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ee = args[0] if args else "orca"                           # orca | gripper
    print("wrote", build_usd(ee, rebuild="--rebuild" in sys.argv))
    fk = FK(TRAIN_URDF if ee == "orca" else GRIP_URDF)
    for k, (b, p, R) in tracked_offsets(fk, ee).items():
        print(f"{k:12s} body {b:14s} pos {p.round(4)}\n{R.round(3)}")
    parts, spheres, pairs = collision_model(ee=ee)
    print(len(parts), "distance fields,", len(spheres), "spheres,", pairs.sum(), "checked sphere/part pairs")

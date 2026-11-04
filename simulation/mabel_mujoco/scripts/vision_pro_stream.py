"""
vision_pro_stream.py — Apple Vision Pro tracking → ZMQ for MABEL teleop.

This is the proven dexterous_hand/vision_pro/connect_full_avp.py streamer, kept
verbatim in structure (always-on skeleton viewer, per-frame publish paced by the
viewer's vsync, the same terminal readout), with only the additions MABEL's
retargeter needs on top of head + wrist poses:

  * right_joints / left_joints — the full NAMED hand skeleton (snake_case wire
    names) the per-finger ORCA retargeter consumes (hand_ik_retarget.py).
  * right_pinch / left_pinch  — live thumb↔index distance (m).
  * right_elbow / left_elbow + right_forearm_wrist / left_forearm_wrist — the
    forearm joints (27-joint skeleton only; null on the old 25-joint app).
  * odom_mat / body_yaw / head_yaw / neck_twist / head_height — the human's
    GLOBAL pose. odom_mat is the TORSO base_link frame on the floor; its heading
    is the FORWARD normal of the two-elbow line (BodyOdom), so it points where the
    body faces / moves and is independent of head yaw. body_yaw is that heading,
    head_yaw is the raw gaze heading, neck_twist = head_yaw - body_yaw. The viewer
    draws the base frame, a head triad, body-forward (green) vs gaze (magenta)
    floor spokes, forearm/elbow links, and a floor->head support line.
  * CONFLATE on the PUB socket — keeps ONLY the newest frame in the send queue so
    a momentarily slow consumer can never build a backlog (the streaming lag).

Published JSON (consumed by mabel.avp_receiver.AVPFrame):
  { "head_mat":[16], "odom_mat":[16], "body_yaw":float, "head_yaw":float,
    "neck_twist":float, "head_height":float,
    "right_hand_mats":[[16]x21], "left_hand_mats":[[16]x21],
    "right_joints":{name:[16]}, "left_joints":{name:[16]},
    "right_pinch":float, "left_pinch":float,
    "right_elbow":[16], "left_elbow":[16],
    "right_forearm_wrist":[16], "left_forearm_wrist":[16], "timestamp":float }

Run:
  python   scripts/vision_pro_stream.py --ip <headset-ip>            # HEADLESS (fast)
  mjpython scripts/vision_pro_stream.py --ip <headset-ip> --viewer   # + skeleton viewer
  python   scripts/vision_pro_stream.py --ip <headset-ip> --check    # verify tracking
By default the streamer is HEADLESS (no MuJoCo viewer) — lower latency, and runs
under plain `python`. Pass --viewer to pull up the always-on skeleton viewer (that
mode needs `mjpython` on macOS). Then, in another terminal:
  mjpython scripts/retarget_teleop.py --source avp
"""
import argparse
import math
import time
import sys

import mujoco
import mujoco.viewer
from avp_stream import VisionProStreamer
import zmq
import json
import numpy as np

# MediaPipe 21-joint connections for the skeleton viewer.
MP_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)     # Little
]

# ARKit 27-joint skeleton → snake_case wire names (indices 0-24; 25-26 forearm
# are dropped). These are the names the per-finger retargeter keys on.
WIRE_NAME = {
    0: "wrist",
    1: "thumb_knuckle", 2: "thumb_intermediate_base",
    3: "thumb_intermediate_tip", 4: "thumb_tip",
    5: "index_metacarpal", 6: "index_knuckle", 7: "index_intermediate_base",
    8: "index_intermediate_tip", 9: "index_tip",
    10: "middle_metacarpal", 11: "middle_knuckle", 12: "middle_intermediate_base",
    13: "middle_intermediate_tip", 14: "middle_tip",
    15: "ring_metacarpal", 16: "ring_knuckle", 17: "ring_intermediate_base",
    18: "ring_intermediate_tip", 19: "ring_tip",
    20: "little_metacarpal", 21: "little_knuckle", 22: "little_intermediate_base",
    23: "little_intermediate_tip", 24: "little_tip",
}


def _axis_triad(length, r):
    """Three RGB box geoms (X=red, Y=green, Z=blue) along the body's local axes —
    a coordinate frame so you can SEE a body's orientation, not just its position."""
    h = length / 2.0
    return (f'<geom type="box" pos="{h} 0 0" size="{h} {r} {r}" rgba="1 0.2 0.2 1"/>'
            f'<geom type="box" pos="0 {h} 0" size="{r} {h} {r}" rgba="0.2 1 0.2 1"/>'
            f'<geom type="box" pos="0 0 {h}" size="{r} {r} {h}" rgba="0.3 0.5 1 1"/>')


def create_mp_scene():
    xml = """
    <mujoco model="vision_pro_strict_matrices">
        <asset>
            <texture type="skybox" builtin="gradient" rgb1=".3 .5 .9" rgb2="0 0 0" width="256" height="256"/>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".2 .2 .2" width="512" height="512"/>
            <material name="grid_mat" texture="grid" texrepeat="20 20" reflectance="0.1"/>
        </asset>
        <worldbody>
            <light pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
            <geom name="floor" type="plane" size="10 10 .05" material="grid_mat"/>

            <body name="head" mocap="true">
                <geom type="box" size="0.1 0.04 0.06" rgba=".5 .7 1 .3"/>
                """ + _axis_triad(0.12, 0.004) + """
                <site name="head_s"/>
            </body>

            <!-- Human TORSO odom / base_link (head-yaw-decoupled): on the floor. -->
            <body name="base" mocap="true">
                <geom type="cylinder" size="0.06 0.004" rgba="1 1 1 0.35"/>
                """ + _axis_triad(0.30, 0.007) + """
                <site name="base_s"/>
            </body>
            <!-- Heading spokes on the floor: body forward (green) vs gaze (magenta).
                 Turn only your head and the magenta swings while green stays put. -->
            <body name="body_fwd" mocap="true">
                <geom type="sphere" size="0.02" rgba="0.2 1 0.2 1"/>
                <site name="body_fwd_s"/>
            </body>
            <body name="head_fwd" mocap="true">
                <geom type="sphere" size="0.02" rgba="1 0.2 1 1"/>
                <site name="head_fwd_s"/>
            </body>
    """

    def build_hand_bodies(prefix, rgba):
        h_xml = ""
        for i in range(21):
            h_xml += f'''
            <body name="{prefix}mp_{i}" mocap="true">
                <geom type="sphere" size="0.01" rgba="{rgba}"/>
                <site name="{prefix}s_{i}"/>
            </body>'''
        # Forearm chain: forearmWrist (fa_w) and elbow / forearmArm (fa_e).
        h_xml += f'''
            <body name="{prefix}fa_w" mocap="true">
                <geom type="sphere" size="0.013" rgba="{rgba}"/>
                <site name="{prefix}fa_w_s"/>
            </body>
            <body name="{prefix}fa_e" mocap="true">
                <geom type="sphere" size="0.022" rgba="1 0.55 0.0 1.0"/>
                <site name="{prefix}fa_e_s"/>
            </body>'''
        return h_xml

    xml += build_hand_bodies("right_", "0.1 0.8 0.2 1.0")
    xml += build_hand_bodies("left_", "0.2 0.8 0.8 1.0")

    xml += "</worldbody><tendon>"

    def build_hand_tendons(prefix, rgba):
        t_xml = ""
        for idx, (p, c) in enumerate(MP_CONNECTIONS):
            t_xml += f'<spatial name="{prefix}link_{idx}" rgba="{rgba}" width="0.002"><site site="{prefix}s_{p}"/><site site="{prefix}s_{c}"/></spatial>'
        # Forearm segment: wrist -> forearmWrist -> elbow (thicker, orange).
        t_xml += (f'<spatial name="{prefix}fa_link0" rgba="1 0.55 0.0 0.5" width="0.005">'
                  f'<site site="{prefix}s_0"/><site site="{prefix}fa_w_s"/></spatial>')
        t_xml += (f'<spatial name="{prefix}fa_link1" rgba="1 0.55 0.0 0.5" width="0.005">'
                  f'<site site="{prefix}fa_w_s"/><site site="{prefix}fa_e_s"/></spatial>')
        return t_xml

    xml += build_hand_tendons("right_", "0.1 0.8 0.2 0.3")
    xml += build_hand_tendons("left_", "0.2 0.8 0.8 0.3")

    # Vertical support line: human base on the floor up to the head.
    xml += ('<spatial name="support" rgba="1 1 1 0.25" width="0.003">'
            '<site site="base_s"/><site site="head_s"/></spatial>')
    # Heading spokes from the base: body-forward (green) and gaze (magenta).
    xml += ('<spatial name="body_spoke" rgba="0.2 1 0.2 0.6" width="0.006">'
            '<site site="base_s"/><site site="body_fwd_s"/></spatial>')
    xml += ('<spatial name="gaze_spoke" rgba="1 0.2 1 0.6" width="0.005">'
            '<site site="base_s"/><site site="head_fwd_s"/></spatial>')

    xml += "</tendon></mujoco>"
    return xml


# avp_stream applies a Y-up→Z-up rotation (its YUP2ZUP) to EVERY matrix it
# returns. MABEL's retargeter (config.AVP_TO_ROBOT) is tuned for the RAW ARKit
# (Y-up) frame the MabelTeleop app sends over the bridge, so we convert
# avp_stream's Z-up matrices BACK to Y-up before publishing. Then BOTH sources
# share one frame + one AVP_TO_ROBOT. Without this, "move hand forward" (Z-up +Y)
# maps onto a VERTICAL robot axis → the robot moves up/down instead of forward.
# ZUP2YUP is the inverse of avp_stream's YUP2ZUP: [x,y,z] -> [x, z, -y].
ZUP2YUP = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])


def _to_arkit(M):
    """avp_stream Z-up world matrix → raw ARKit Y-up world matrix (app frame)."""
    return ZUP2YUP @ np.asarray(M, dtype=float)


def get_mp_aligned_mats(m):
    """AVP 27-mat hand → MediaPipe 21-mat ordering ([0]=wrist, [4]=thumb tip,
    [8]=index tip), the layout the wrist-pose + pinch consumer expects."""
    return [
        m[0],                         # Wrist
        m[1], m[2], m[3], m[4],       # Thumb
        m[6], m[7], m[8], m[9],       # Index
        m[11], m[12], m[13], m[14],   # Middle
        m[16], m[17], m[18], m[19],   # Ring
        m[21], m[22], m[23], m[24],   # Pinky
    ]


def named_joints(m):
    """Full NAMED world skeleton {wire_name: [16]} for per-finger retargeting,
    converted to the ARKit (Y-up) app frame."""
    return {WIRE_NAME[i]: _to_arkit(m[i]).flatten().tolist()
            for i in range(min(25, len(m)))}


def pinch_distance(m):
    """Live world thumb-tip(4) ↔ index-tip(9) distance [m]. Computed directly
    (not from avp_stream's cached scalar, which goes stale through array views)."""
    return float(np.linalg.norm(np.asarray(m[4], float)[:3, 3]
                                - np.asarray(m[9], float)[:3, 3]))


# Forearm joints in the AVP arm skeleton (present only when the app streams the
# 27-joint format; the 25-joint format stops at the fingertips).
FOREARM_WRIST_IDX = 25   # forearmWrist — just proximal of the wrist
ELBOW_IDX = 26           # forearmArm   — the elbow end of the forearm


def has_forearm(m):
    """True when this hand carries the forearm/elbow joints (27-joint skeleton)."""
    return m is not None and len(m) >= 27


def forearm_arkit(m):
    """(forearm_wrist[16], elbow[16]) in the ARKit Y-up app frame, or (None, None)
    when the headset/app only streams the 25-joint (no-forearm) skeleton."""
    if not has_forearm(m):
        return None, None
    return (_to_arkit(m[FOREARM_WRIST_IDX]).flatten().tolist(),
            _to_arkit(m[ELBOW_IDX]).flatten().tolist())


def elbow_zup(m):
    """World (Z-up viewer frame) elbow position [3] for the body-heading normal,
    or None when this hand has no forearm joints."""
    if not has_forearm(m):
        return None
    return np.asarray(m[ELBOW_IDX], float)[:3, 3]


def head_yaw(head_zup):
    """Gaze heading: yaw about world-up (Z) of the HEAD, radians. After
    avp_stream's rotate_head the head's local +Y column is the world forward
    direction, so the heading is atan2(forward_y, forward_x). NOTE this tracks
    the neck, not the torso — see BodyOdom for the body frame."""
    R = np.asarray(head_zup, float)[:3, :3]
    fwd = R[:, 1]                       # head local +Y == world forward
    return math.atan2(fwd[1], fwd[0])


def wrap_to_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def neck_pivot_zup(head_zup, back, down):
    """World point of the cervical (neck) pivot — the point the eyes orbit when
    you swivel your head. Sits `back` metres behind and `down` metres below the
    eyes in the head's own frame (local +Y forward, +Z up). Because it is rigidly
    offset from the head, R_head turns it back onto the same world point as the
    head rotates, so it does NOT slew sideways when you merely turn your head —
    unlike the raw head position. This is the body's position proxy."""
    M = np.asarray(head_zup, float)
    R, p = M[:3, :3], M[:3, 3]
    return p + R @ np.array([0.0, -back, -down])   # back along -fwd, down along -up


def elbow_normal_yaw(elbow_r_zup, elbow_l_zup):
    """Torso facing (yaw about world-up Z) from the two elbows: the FORWARD normal
    of the elbow line on the floor plane. The line right_elbow - left_elbow points
    to the body's right; forward is that turned +90deg about up, i.e. (-dy, dx).
    Returns None if the elbows coincide. Independent of head yaw, and swings to
    point where the torso (and hence locomotion) is aimed."""
    if elbow_r_zup is None or elbow_l_zup is None:
        return None
    d = np.asarray(elbow_r_zup, float)[:2] - np.asarray(elbow_l_zup, float)[:2]
    if np.linalg.norm(d) < 1e-3:
        return None
    return math.atan2(d[0], -d[1])     # atan2(forward_y, forward_x), forward=(-dy,dx)


class BodyOdom:
    """Estimate the human's TORSO odom / base_link frame.

      * Heading — the FORWARD normal of the two-elbow line (elbow_normal_yaw),
        low-passed. This points where the torso faces / where you are moving, and
        is naturally independent of head yaw (turning your neck does not move your
        elbows). Falls back to holding the last heading — or the gaze heading on
        the very first frame — when both elbows aren't tracked.
      * Position — the neck pivot (head-rotation-invariant) dropped to the floor
        (z=0), low-passed. Turning the head leaves it put; stepping/leaning moves
        it.
    """

    def __init__(self, heading_tau=0.25, pos_tau=0.4,
                 neck_back=0.08, neck_down=0.10):
        self.heading_tau = heading_tau   # heading low-pass time constant [s]
        self.pos_tau = pos_tau           # position low-pass time constant [s]
        self.neck_back = neck_back
        self.neck_down = neck_down
        self.body_yaw = None             # latched torso heading [rad]
        self.base_xy = None              # latched floor position [m]
        self.last_t = None

    def update(self, head_zup, now, elbow_r=None, elbow_l=None):
        """Advance the estimate. `elbow_r`/`elbow_l` are the world (Z-up) elbow
        positions (3-vectors) or None. Returns (base_mat_zup 4x4, body_yaw,
        neck_twist) where neck_twist = head_yaw - body_yaw (gaze vs torso)."""
        hy = head_yaw(head_zup)
        dt = max(now - self.last_t, 1e-3) if self.last_t is not None else None

        # --- heading: forward normal of the elbow line, low-passed ---
        target = elbow_normal_yaw(elbow_r, elbow_l)
        if target is None:                       # no elbows -> hold (or gaze at init)
            target = self.body_yaw if self.body_yaw is not None else hy
        if self.body_yaw is None or dt is None:
            self.body_yaw = target
        else:
            a = 1.0 - math.exp(-dt / max(self.heading_tau, 1e-3))
            self.body_yaw = wrap_to_pi(self.body_yaw
                                       + a * wrap_to_pi(target - self.body_yaw))

        # --- position: low-passed neck pivot dropped to the floor ---
        xy = neck_pivot_zup(head_zup, self.neck_back, self.neck_down)[:2]
        if self.base_xy is None or dt is None:
            self.base_xy = np.array(xy, float)
        else:
            a = 1.0 - math.exp(-dt / max(self.pos_tau, 1e-3))
            self.base_xy = (1.0 - a) * self.base_xy + a * xy
        self.last_t = now

        c, s = math.cos(self.body_yaw), math.sin(self.body_yaw)
        M = np.eye(4)
        M[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        M[0, 3], M[1, 3], M[2, 3] = self.base_xy[0], self.base_xy[1], 0.0
        return M, self.body_yaw, wrap_to_pi(hy - self.body_yaw)


def build_payload(avp_data, body=None):
    """Pure payload builder (no viewer side effects) — shared by --check. All
    matrices are converted from avp_stream's Z-up to the ARKit Y-up app frame.
    `body` is a persistent BodyOdom (pass the same instance across frames so the
    torso heading hysteresis has history); a transient one is made if omitted."""
    body = body if body is not None else BodyOdom()
    head = avp_data.head
    payload = {"timestamp": time.time(),
               "head_mat": _to_arkit(head).flatten().tolist()
               if head is not None else None}
    # Human global pose: TORSO odom / base_link (head-yaw-decoupled), plus the
    # raw gaze heading and the neck twist between them, and standing height.
    if head is not None:
        er, el = elbow_zup(avp_data.right), elbow_zup(avp_data.left)
        base_mat, body_yaw, neck_twist = body.update(head, time.time(), er, el)
        payload["odom_mat"] = _to_arkit(base_mat).flatten().tolist()
        payload["body_yaw"] = float(body_yaw)
        payload["head_yaw"] = float(head_yaw(head))
        payload["neck_twist"] = float(neck_twist)
        payload["head_height"] = float(np.asarray(head, float)[2, 3])
    else:
        payload["odom_mat"] = None
        payload["body_yaw"] = None
        payload["head_yaw"] = None
        payload["neck_twist"] = None
        payload["head_height"] = None
    for side, hand in (("right", avp_data.right), ("left", avp_data.left)):
        if hand is not None and len(hand) > 0:
            payload[f"{side}_hand_mats"] = [_to_arkit(x).flatten().tolist()
                                            for x in get_mp_aligned_mats(hand)]
            payload[f"{side}_joints"] = named_joints(hand)
            payload[f"{side}_pinch"] = pinch_distance(hand)
            fa_w, elbow = forearm_arkit(hand)
            payload[f"{side}_forearm_wrist"] = fa_w
            payload[f"{side}_elbow"] = elbow
        else:
            payload[f"{side}_hand_mats"] = None
            payload[f"{side}_joints"] = None
            payload[f"{side}_pinch"] = 0.0
            payload[f"{side}_forearm_wrist"] = None
            payload[f"{side}_elbow"] = None
    return payload


# ── rigorous tracking self-check (head + both hands) ─────────────────────────
def run_check(streamer, seconds=6.0):
    """Sample the headset for N seconds and print a PASS/FAIL report so you can
    confirm head + hand tracking are live and sane before teleoperating."""
    G, R0, Y, E, B = "\033[92m", "\033[91m", "\033[93m", "\033[0m", "\033[1m"
    ok = lambda c: f"{G}PASS{E}" if c else f"{R0}FAIL{E}"
    n = head_ok = head_orth = 0
    head_pos = []
    heading = []        # gaze (head) heading samples [deg]
    body_head = []      # torso (body) heading samples [deg]
    twist = []          # neck twist head-minus-body [deg]
    body = BodyOdom()   # persistent so the heading hysteresis has history
    stats = {"right": dict(present=0, jmin=99, finger=[], pinch=[], forearm=0),
             "left":  dict(present=0, jmin=99, finger=[], pinch=[], forearm=0)}
    print(f"\nsampling the headset for {seconds:.0f}s — move your head and hands, "
          f"open/close a pinch…")
    t0 = time.time()
    while time.time() - t0 < seconds:
        tr = streamer.get_latest()
        if tr is None:
            time.sleep(0.005); continue
        p = build_payload(tr, body); n += 1
        if p["head_mat"]:
            M = np.asarray(p["head_mat"]).reshape(4, 4)
            head_ok += 1; head_pos.append(M[:3, 3].copy())
            if np.linalg.norm(M[:3, :3].T @ M[:3, :3] - np.eye(3)) < 1e-2:
                head_orth += 1
        if p.get("head_yaw") is not None:
            heading.append(math.degrees(p["head_yaw"]))
            body_head.append(math.degrees(p["body_yaw"]))
            twist.append(math.degrees(p["neck_twist"]))
        for side in ("right", "left"):
            nm = p[f"{side}_joints"]
            if not nm:
                continue
            s = stats[side]; s["present"] += 1; s["jmin"] = min(s["jmin"], len(nm))
            s["pinch"].append(p[f"{side}_pinch"] * 100.0)
            if p.get(f"{side}_elbow") is not None:
                s["forearm"] += 1
            kn, tip = nm.get("index_knuckle"), nm.get("index_tip")
            if kn and tip:
                s["finger"].append(np.linalg.norm(
                    np.asarray(kn).reshape(4, 4)[:3, 3]
                    - np.asarray(tip).reshape(4, 4)[:3, 3]) * 100.0)
        time.sleep(0.005)
    dt = time.time() - t0
    span = (np.ptp(np.array(head_pos), axis=0) if len(head_pos) > 1 else np.zeros(3))
    head_pass = head_ok > 0.8 * n and head_orth > 0.8 * max(head_ok, 1)
    print(f"\n{B}══ Vision Pro tracking self-check ══{E}  ({n} frames, {n/max(dt,1e-6):.0f} Hz)\n")
    gspan = float(np.ptp(np.array(heading))) if len(heading) > 1 else 0.0
    bspan = float(np.ptp(np.array(body_head))) if len(body_head) > 1 else 0.0
    tmax = float(np.max(np.abs(twist))) if twist else 0.0
    print(f"  HEAD   {ok(head_pass)}  present {head_ok/max(n,1)*100:3.0f}% · "
          f"orthonormal {head_orth}/{head_ok} · moved "
          f"{span[0]*100:.1f}/{span[1]*100:.1f}/{span[2]*100:.1f} cm (xyz)")
    print(f"  ODOM         gaze swept {gspan:5.1f}° · body swept {bspan:5.1f}° · "
          f"neck twist max {tmax:4.0f}° · height "
          f"{np.mean([h[2] for h in head_pos])*100 if head_pos else 0:5.1f} cm")
    print(f"  {Y}↳ body heading is the elbow-line normal: shake your head and it "
          f"stays put; turn your torso and it follows.{E}")
    all_pass = head_pass; frozen = False
    for side in ("right", "left"):
        s = stats[side]; frac = s["present"] / max(n, 1)
        fcm = np.array(s["finger"]) if s["finger"] else np.array([0.0])
        pinch = np.array(s["pinch"]) if s["pinch"] else np.array([0.0])
        pvar = float(np.ptp(pinch)); jmin = s["jmin"] if s["present"] else 0
        hp = frac > 0.5 and jmin >= 21 and 3.0 < fcm.mean() < 14.0
        all_pass = all_pass and hp
        if hp and pvar < 0.5:
            frozen = True
        fa_frac = s["forearm"] / max(n, 1)
        fa_str = (f"{G}elbow {fa_frac*100:3.0f}%{E}" if s["forearm"]
                  else f"{Y}no elbow{E}")
        print(f"  {side.upper():5s}  {ok(hp)}  present {frac*100:3.0f}% · joints≥{jmin} · "
              f"index {fcm.mean():4.1f} cm · pinch {pinch.min():4.1f}–{pinch.max():4.1f} cm "
              f"({Y if pvar < 0.5 else G}Δ{pvar:.1f}{E}) · {fa_str}")
    if frozen:
        print(f"  {Y}note{E}   a hand tracked but its pinch never changed — pinch "
              f"thumb↔index during the check to confirm it updates.")
    print(f"\n  {B}OVERALL {ok(all_pass)}{E}  "
          f"{'tracking is live and sane.' if all_pass else 'see FAILs above.'}\n")


def stream_frame(avp_data, body, socket, start, model=None, data=None):
    """Build + publish one frame's payload and print the terminal readout. When
    `model`/`data` are given (i.e. --viewer), the skeleton mocap bodies are also
    driven; headless they are None and all viewer side effects are skipped."""
    viewer_on = model is not None and data is not None

    head_mat = avp_data.head
    if viewer_on:
        data.mocap_pos[model.body('head').mocapid[0]] = head_mat[:3, 3]
        mujoco.mju_mat2Quat(data.mocap_quat[model.body('head').mocapid[0]],
                            head_mat[:3, :3].flatten())

    # Human TORSO odom / base_link — heading from the two-elbow forward normal,
    # so it points where the body faces / moves and is independent of head yaw.
    er = elbow_zup(getattr(avp_data, 'right', None))
    el = elbow_zup(getattr(avp_data, 'left', None))
    base_mat, body_yaw, neck_twist = body.update(head_mat, start, er, el)
    gaze_yaw = head_yaw(head_mat)
    if viewer_on:
        base_id = model.body('base').mocapid[0]
        data.mocap_pos[base_id] = base_mat[:3, 3]   # raw Z-up for THIS viewer
        mujoco.mju_mat2Quat(data.mocap_quat[base_id], base_mat[:3, :3].flatten())

        # Floor heading spokes: body-forward (green) vs gaze (magenta).
        bxy = base_mat[:2, 3]
        for nm_body, ang in (("body_fwd", body_yaw), ("head_fwd", gaze_yaw)):
            end = np.array([bxy[0] + 0.45 * math.cos(ang),
                            bxy[1] + 0.45 * math.sin(ang), 0.02])
            data.mocap_pos[model.body(nm_body).mocapid[0]] = end

    payload = {"head_mat": _to_arkit(head_mat).flatten().tolist(),
               "odom_mat": _to_arkit(base_mat).flatten().tolist(),
               "body_yaw": float(body_yaw),
               "head_yaw": float(gaze_yaw),
               "neck_twist": float(neck_twist),
               "head_height": float(head_mat[2, 3]),
               "timestamp": time.time()}

    wrist_pos_r = None
    wrist_pos_l = None
    elbow_pos_r = None
    elbow_pos_l = None

    def hide_forearm(prefix):
        """Park the forearm bodies below the floor when the app isn't streaming
        forearm joints, so stale poses don't linger (viewer only)."""
        if not viewer_on:
            return
        for tag in ("fa_w", "fa_e"):
            bid = model.body(f"{prefix}{tag}").mocapid[0]
            data.mocap_pos[bid] = (0.0, 0.0, -5.0)

    def process_hand(raw_mats, side):
        prefix = f"{side}_"
        mp_mats = get_mp_aligned_mats(raw_mats)
        serialized = []
        for i, mat in enumerate(mp_mats):
            if viewer_on:
                body_id = model.body(f"{prefix}mp_{i}").mocapid[0]
                data.mocap_pos[body_id] = mat[:3, 3]    # raw Z-up for THIS viewer
                mujoco.mju_mat2Quat(data.mocap_quat[body_id], mat[:3, :3].flatten())
            serialized.append(_to_arkit(mat).flatten().tolist())  # Y-up to publish
        # MABEL additions: named skeleton (finger IK) + pinch.
        payload[f"{side}_joints"] = named_joints(raw_mats)
        payload[f"{side}_pinch"] = pinch_distance(raw_mats)
        # Forearm / elbow (only in the 27-joint skeleton).
        elbow_pos = None
        if has_forearm(raw_mats):
            if viewer_on:
                for tag, idx in (("fa_w", FOREARM_WRIST_IDX), ("fa_e", ELBOW_IDX)):
                    mat = np.asarray(raw_mats[idx], float)
                    bid = model.body(f"{prefix}{tag}").mocapid[0]
                    data.mocap_pos[bid] = mat[:3, 3]
                    mujoco.mju_mat2Quat(data.mocap_quat[bid], mat[:3, :3].flatten())
            elbow_pos = np.asarray(raw_mats[ELBOW_IDX], float)[:3, 3]
            fa_w, elbow = forearm_arkit(raw_mats)
            payload[f"{side}_forearm_wrist"] = fa_w
            payload[f"{side}_elbow"] = elbow
        else:
            hide_forearm(prefix)
            payload[f"{side}_forearm_wrist"] = None
            payload[f"{side}_elbow"] = None
        return serialized, mp_mats[0][:3, 3], elbow_pos

    if getattr(avp_data, 'right', None) is not None and len(avp_data.right) > 0:
        r_mats, wrist_pos_r, elbow_pos_r = process_hand(avp_data.right, "right")
        payload["right_hand_mats"] = r_mats
    else:
        hide_forearm("right_")

    if getattr(avp_data, 'left', None) is not None and len(avp_data.left) > 0:
        l_mats, wrist_pos_l, elbow_pos_l = process_hand(avp_data.left, "left")
        payload["left_hand_mats"] = l_mats
    else:
        hide_forearm("left_")

    socket.send_string(json.dumps(payload))

    fps = 1 / (time.time() - start + 0.0001)

    hp = head_mat[:3, 3]
    base_inv = np.linalg.inv(base_mat)   # world(Z-up) -> body frame
    in_body = lambda p: (base_inv @ np.array([p[0], p[1], p[2], 1.0]))[:3]
    mode = "viewer" if viewer_on else "headless"
    output = [
        "\033[H",
        "\033[1;34m" + "═" * 62 + "\033[0m",
        f"\033[1;32m VISION PRO STREAMER \033[0m | FPS: {fps:>4.1f} | {mode}",
        "─" * 62,
        f"\033[1;35m HEAD:\033[0m         X: {hp[0]:>6.3f} | Y: {hp[1]:>6.3f} | "
        f"Z: {hp[2]:>6.3f}  height {hp[2]*100:5.1f}cm",
        f"\033[1;37m ODOM/BASE:\033[0m    x: {base_mat[0,3]:>6.3f} | "
        f"y: {base_mat[1,3]:>6.3f}  body {math.degrees(body_yaw):>6.1f}° "
        f"gaze {math.degrees(gaze_yaw):>6.1f}° twist {math.degrees(neck_twist):>5.0f}°",
        "─" * 62,
    ]
    if wrist_pos_r is not None:
        pr = payload.get("right_pinch", 0.0) * 100
        br = in_body(wrist_pos_r)
        output.append(f"\033[1;33m RIGHT WRIST:\033[0m   X: {wrist_pos_r[0]:>6.3f} | "
                      f"Y: {wrist_pos_r[1]:>6.3f} | Z: {wrist_pos_r[2]:>6.3f}  "
                      f"pinch {pr:4.1f}cm  body[{br[0]:>5.2f},{br[1]:>5.2f}]")
        if elbow_pos_r is not None:
            output.append(f"\033[0;33m RIGHT ELBOW:\033[0m   X: {elbow_pos_r[0]:>6.3f} | "
                          f"Y: {elbow_pos_r[1]:>6.3f} | Z: {elbow_pos_r[2]:>6.3f}")
        else:
            output.append("\033[0;33m RIGHT ELBOW:   (no forearm joints)\033[0m")
    else:
        output.append("\033[1;31m RIGHT WRIST:   Not Detected\033[0m")
    if wrist_pos_l is not None:
        pl = payload.get("left_pinch", 0.0) * 100
        bl = in_body(wrist_pos_l)
        output.append(f"\033[1;36m LEFT WRIST: \033[0m   X: {wrist_pos_l[0]:>6.3f} | "
                      f"Y: {wrist_pos_l[1]:>6.3f} | Z: {wrist_pos_l[2]:>6.3f}  "
                      f"pinch {pl:4.1f}cm  body[{bl[0]:>5.2f},{bl[1]:>5.2f}]")
        if elbow_pos_l is not None:
            output.append(f"\033[0;36m LEFT ELBOW: \033[0m   X: {elbow_pos_l[0]:>6.3f} | "
                          f"Y: {elbow_pos_l[1]:>6.3f} | Z: {elbow_pos_l[2]:>6.3f}")
        else:
            output.append("\033[0;36m LEFT ELBOW:    (no forearm joints)\033[0m")
    else:
        output.append("\033[1;31m LEFT WRIST:    Not Detected\033[0m")
    output.extend(["\033[1;34m" + "═" * 62 + "\033[0m", "\033[K"])
    sys.stdout.write("\n".join(output))
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", "--room", dest="ip", type=str, required=True,
                        help="Vision Pro IP (shown in the Tracking Streamer app)")
    parser.add_argument("--addr", type=str, default="tcp://*:5556",
                        help="ZMQ PUB bind address (consumer connects to :5556)")
    parser.add_argument("--viewer", action="store_true",
                        help="pull up the always-on MuJoCo skeleton viewer (needs "
                             "mjpython on macOS). Default is HEADLESS for speed.")
    parser.add_argument("--check", nargs="?", type=float, const=6.0, default=None,
                        metavar="SECONDS",
                        help="verify head + hand tracking for N seconds, then exit")
    parser.add_argument("--heading-tau", type=float, default=0.25, metavar="SEC",
                        help="low-pass time constant for the torso/odom heading "
                             "(from the elbow-line normal); larger = smoother, "
                             "slower (default 0.25s)")
    args = parser.parse_args()

    if args.check is not None:
        run_check(VisionProStreamer(ip=args.ip), seconds=args.check)
        return

    # --- ZMQ: CONFLATE keeps ONLY the newest frame queued, so a slow consumer
    #     never makes us accumulate a stale backlog (the streaming-lag fix). ---
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)     # newest-only send queue
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)       # drop pending messages immediately on close
    socket.bind(args.addr)
    # ----------------------------------

    streamer = VisionProStreamer(ip=args.ip)
    body = BodyOdom(heading_tau=args.heading_tau)   # torso odom estimator

    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("\033[2J\033[3J\033[H")
    sys.stdout.flush()

    if not args.viewer:
        # HEADLESS: no MuJoCo model/viewer — just build + publish each frame.
        while True:
            start = time.time()
            avp_data = streamer.get_latest()
            if avp_data:
                stream_frame(avp_data, body, socket, start)
            # 200Hz loop: fast enough for AVP, slow enough for ZMQ to transmit.
            time.sleep(0.005)
        return

    model = mujoco.MjModel.from_xml_string(create_mp_scene())
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            avp_data = streamer.get_latest()
            if avp_data:
                stream_frame(avp_data, body, socket, start, model, data)
            viewer.sync()
            mujoco.mj_step(model, data)
            # 200Hz loop: fast enough for AVP, slow enough for ZMQ to transmit.
            time.sleep(0.005)


if __name__ == "__main__":
    main()

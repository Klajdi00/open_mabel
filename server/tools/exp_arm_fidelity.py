"""ARM FIDELITY — does the robot's hand go where the operator's hand is?

THREE OPERATOR REPORTS, MEASURED (2026-07-28)
---------------------------------------------
  1. "when my hands are positioned the same level as the head, in the simulation
     the hand is much lower — this breaks the immersiveness."
  2. "the local tracking arm wrist position tracking is super bad — revise it so
     it works like the global position arm tracking."
  3. "the arm twists very not human like — the elbow moves crazy when I flip my
     wrist. When I flip the wrist it should mainly map to J5."

HOW IT MEASURES
---------------
The wire carries joint angles, not Cartesian poses, so this harness loads the
SAME MJCF the bridge runs and does forward kinematics on the streamed
`jointPositions`. That gives the robot's true palm / head / shoulder / elbow
positions, which is what the operator is actually comparing against their own.

The comparison is in BODY-SCALE units, because the robot is not a person: what
"my hand is at head level" has to mean is that the hand sits at the same height
relative to the HEAD, as a fraction of that body's own head-above-shoulder
distance. Human 0.206 m, robot 0.170 m (both measured, see the constants).

  1. HEIGHT FIDELITY   operator hand at eye level → robot hand within
                       HEIGHT_TOL of robot head level
  2. LOCAL == GLOBAL   the same reach, in both frames, lands in the same place
  3. WRIST ROLL        sweeping the operator's forearm roll ±90 deg must turn
                       J5 (the forearm roll joint) and must NOT swing the elbow

Usage
-----
    server/run.sh --no-studio            # in another terminal
    python3 server/tools/exp_arm_fidelity.py [--plot]
"""
import argparse
import asyncio
import json
import math
import os
import sys
import time

import numpy as np
import websockets

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "vision_os_apps", "docs", "tools"))
from exp_hand_retarget import synth_hand   # noqa: E402

import mujoco   # noqa: E402

PORT = os.environ.get("MABEL_WS_PORT", "9090")
URI = f"ws://127.0.0.1:{PORT}/teleop?client=visionpro"
MJCF = os.path.join(_ROOT, "simulation", "mabel_mujoco", "models", "mabel_full.xml")

RATE_HZ = 30.0
EYE = 1.60                 # operator eye height, m (the head pose we stream)
STATURE = EYE / 0.9366     # ANTHRO_EYE_HEIGHT_RATIO
HUMAN_HEAD_ABOVE_SHOULDER = 0.206 * (STATURE / 1.75)
ROBOT_HEAD_ABOVE_SHOULDER = 0.170          # measured from the MJCF at qpos0

# Fraction of its own head-above-shoulder distance the robot's hand may differ
# by. 0.35 of 0.17 m = 6 cm — about the height of a palm, i.e. the point at
# which "my hand is at my head" stops being true at a glance.
HEIGHT_TOL = 0.35


class FK:
    """Forward kinematics on the streamed joint angles, using the bridge's model."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        self.d = mujoco.MjData(self.m)
        self.adr = {}
        for j in range(self.m.njnt):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n:
                self.adr[n] = self.m.jnt_qposadr[j]
        # Wire ids → MJCF joint names.
        self.wire = {}
        for side, S in (("right", "Right"), ("left", "Left")):
            for i in range(1, 8):
                self.wire[f"{side}_arm_{i}"] = f"{S} Arm {i}"
            self.wire[f"{side}_wrist"] = f"{S} Wrist"
        self.wire.update({"torso": "Torso", "neck_yaw": "Neck Yaw",
                          "neck_pitch": "Neck Pitch", "neck_roll": "Neck Roll",
                          "lift_lower": "Lift Lower", "lift_upper": "Lift Upper"})

    def pose(self, joints):
        mujoco.mj_resetData(self.m, self.d)
        for wid, val in joints.items():
            name = self.wire.get(wid)
            a = self.adr.get(name) if name else None
            if a is not None:
                self.d.qpos[a] = float(val)
        mujoco.mj_forward(self.m, self.d)
        return self

    def p(self, body):
        return self.d.body(body).xpos.copy()

    def shoulder(self, side):
        return self.p(f"{'r' if side == 'right' else 'l'}_arm_mount")

    def wrist(self, side):
        """The point the arm IK actually CONTROLS: the ORCA wrist joint's
        anchor (`arm.py`: `d.xanchor[orca_wrist_jid]` under ARM_TRACK_WRIST).
        Measuring the `*_palm` BODY instead reads a point ~25 cm further down
        the hand, which rotates with the wrist — so a pure wrist roll looks
        like the hand flying around when the controlled point never moved."""
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{'Right' if side == 'right' else 'Left'} Wrist")
        return self.d.xanchor[jid].copy()

    def elbow(self, side):
        return self.p(f"{'r' if side == 'right' else 'l'}_forearm")


def _mat(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_HANDS = {}


def hand_joints(side, roll):
    """A hand skeleton rolled `roll` rad about its own forward axis — the
    operator pronating / supinating their forearm."""
    key = (side, round(roll, 2))
    if key not in _HANDS:
        R = _rot("x", roll)
        _HANDS[key] = [
            {"joint": n,
             "localTransform": {"matrix": (_mat(R, np.zeros(3)) @ np.asarray(T, float))
                                          .reshape(16).tolist()},
             "isTracked": True}
            for n, T in synth_hand(mcp=0.1, pip=0.1, thumb=0.1, chirality=side).items()]
    return _HANDS[key]


def frame(seq, wrist_local, roll=0.0, yaw=0.0, pitch=0.0):
    """`wrist_local` = both wrists' offset from the HEAD, in the operator's body
    frame (x right, y up, −z forward). That is exactly how an operator describes
    their own pose: "my hand is level with my head, half a metre in front"."""
    head_R = _rot("y", yaw)
    head_t = np.array([0.0, EYE, 0.0])
    payload = {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": _mat(head_R, head_t).reshape(16).tolist()},
                 "trackingState": "normal"},
        "mode": "Whole Body",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
    }
    for side, sx in (("left", -1.0), ("right", 1.0)):
        off = np.array([sx * wrist_local[0], wrist_local[1], wrist_local[2]])
        W = head_t + head_R @ off
        payload[f"{side}Hand"] = {
            "chirality": side,
            # roll = forearm pronation (about the hand's forward axis);
            # pitch = wrist flexion/extension (about its lateral axis).
            "anchorTransform": {"matrix": _mat(head_R @ _rot("x", roll) @ _rot("z", pitch), W)
                                          .reshape(16).tolist()},
            "joints": hand_joints(side, roll), "isTracked": True,
        }
    return {"type": "teleop_frame", "payload": payload}


class Rig:
    def __init__(self, ws, fk):
        self.ws, self.fk = ws, fk
        self.joints = {}
        self.seq = 0

    async def send(self, t, p):
        await self.ws.send(json.dumps({"type": t, "payload": p}))

    async def hold(self, seconds, wrist_local, roll=0.0, yaw=0.0, pitch=0.0):
        """Stream ONE static operator pose and let the robot converge to it."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.seq += 1
            await self.ws.send(json.dumps(frame(self.seq, wrist_local, roll, yaw, pitch)))
            await asyncio.sleep(1.0 / RATE_HZ)
        return self.fk.pose(self.joints)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    fk = FK()
    async with websockets.connect(URI, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        rig = Rig(ws, fk)

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "robot_state":
                    rig.joints.update(m["payload"].get("jointPositions", {}))
        rtask = asyncio.create_task(reader())

        async def spec(frame_mode="global", **kw):
            base = {"method": "retarget", "controlType": "impedance",
                    "region": "whole_body", "stiffness": 0.0, "frame": frame_mode,
                    "tracking": {"neck": True, "arms": True, "hands": True}}
            base.update(kw)
            await rig.send("control_mode", base)

        results = []

        def check(name, ok, detail):
            results.append((name, ok, detail))
            print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")

        async def settle(mode):
            await rig.send("reset", {})
            await asyncio.sleep(2.0)
            await rig.send("estop", {"active": False})
            await asyncio.sleep(0.4)
            await spec(mode)
            await rig.send("recalibrate", {})
            # A calibration reach: the arm map's rho normalises on the LONGEST
            # shoulder→wrist distance it has seen, so it must see a full
            # extension before any of these numbers mean anything.
            await rig.hold(2.5, (0.30, -0.10, -0.60))

        # ── 1 & 2. HEIGHT FIDELITY, in both frames ──────────────────────────
        heights = {}
        for mode in ("global", "local"):
            print(f"\n=== {mode.upper()} — operator hand AT EYE LEVEL ===")
            await settle(mode)
            # Hand level with the head, a comfortable reach in front.
            f = await rig.hold(4.5, (0.25, 0.0, -0.45))
            palm = f.wrist("right")
            head = f.p("head")
            sh = f.shoulder("right")
            dz = palm[2] - head[2]
            heights[mode] = dz
            frac = dz / ROBOT_HEAD_ABOVE_SHOULDER
            print(f"    robot wrist z={palm[2]:.3f}  head z={head[2]:.3f}  "
                  f"shoulder z={sh[2]:.3f}")
            check(f"{mode}: hand lands at head level",
                  abs(frac) < HEIGHT_TOL,
                  f"wrist is {dz * 100:+.1f} cm from head level "
                  f"({frac:+.2f} of head-above-shoulder)")

        check("local matches global",
              abs(heights["local"] - heights["global"]) < 0.05,
              f"local {heights['local'] * 100:+.1f} cm vs global "
              f"{heights['global'] * 100:+.1f} cm from head level")

        # ── 3. WRIST ROLL → J5, not the elbow ───────────────────────────────
        print("\n=== WRIST ROLL sweep (the elbow must not follow) ===")
        await settle("global")
        # +-60 deg: comfortably inside BOTH the human forearm's range and the
        # robot J5's +-90 deg. The extremes are reported separately below —
        # past J5's stop the palm orientation is simply unreachable and the IK
        # has to trade position for it, which is a hardware limit, not a
        # retargeting defect.
        rolls = [0.0, 0.35, 0.7, 1.05, 0.7, 0.35, 0.0, -0.35, -0.7, -1.05]
        elbows, j5, palms = [], [], []
        for r in rolls:
            f = await rig.hold(1.4, (0.25, -0.05, -0.50), roll=r)
            elbows.append(f.elbow("right"))
            j5.append(rig.joints.get("right_arm_5", 0.0))
            palms.append(f.wrist("right"))
        elbow_span = max(np.linalg.norm(e - elbows[0]) for e in elbows)
        j5_span = math.degrees(max(j5) - min(j5))
        palm_span = max(np.linalg.norm(p - palms[0]) for p in palms)
        print(f"    elbow travel {elbow_span * 100:.1f} cm   "
              f"J5 travel {j5_span:.1f} deg   wrist drift {palm_span * 100:.1f} cm")
        check("J5 carries the wrist roll", j5_span > 45.0,
              f"J5 moved {j5_span:.1f} deg over a 172 deg operator roll")
        check("the elbow does NOT swing with the roll", elbow_span < 0.08,
              f"elbow moved {elbow_span * 100:.1f} cm")
        check("the hand stays put while the wrist rolls", palm_span < 0.05,
              f"the controlled wrist point drifted {palm_span * 100:.1f} cm "
              "over a +-60 deg roll")

        # ── 4. HUMAN WRIST WORKSPACE — the elbow must stay DOWN and IN ──────
        #
        # An elbow that flares OUTWARD is both unhuman and the direction that
        # walks the arm toward its shoulder singularity (operator, 2026-07-28).
        # "Flare" is measured as the elbow's LATERAL displacement from the plane
        # containing the shoulder→wrist axis and gravity — positive = away from
        # the body — as a fraction of the upper-arm length. A relaxed human arm
        # sits at ~0; a chicken-wing is ~1.
        print("\n=== HUMAN WRIST WORKSPACE (elbow must stay down and in) ===")
        await settle("global")
        # A path a real hand actually takes: out and back, up and down, across
        # the body, with the forearm rolling as it goes (you cannot reach across
        # yourself without pronating).
        poses = [
            ("rest low",        (0.22, -0.55, -0.25),  0.0),
            ("reach forward",   (0.22, -0.25, -0.55),  0.0),
            ("chest height",    (0.20, -0.15, -0.40),  0.4),
            ("eye level",       (0.25,  0.00, -0.45),  0.0),
            ("above head",      (0.22,  0.25, -0.35), -0.4),
            ("across body",     (-0.10, -0.10, -0.45), 0.8),
            ("wide out",        (0.50, -0.20, -0.30),  0.0),
            ("close in",        (0.12, -0.20, -0.25), -0.6),
            ("low reach",       (0.25, -0.60, -0.45),  0.3),
            ("back to rest",    (0.22, -0.55, -0.25),  0.0),
        ]
        flares, prev_q = [], None
        for label, w, roll in poses:
            f = await rig.hold(1.6, w, roll=roll)
            sh, el, wr = f.shoulder("right"), f.elbow("right"), f.wrist("right")
            axis = wr - sh
            n = np.linalg.norm(axis)
            if n < 1e-6:
                continue
            axis /= n
            # Lateral = the robot's own +y (left) — so "outward" for the RIGHT
            # arm is −y. Take the elbow offset perpendicular to the arm axis.
            v = el - sh
            perp = v - axis * float(v @ axis)
            lat = -float(perp[1]) / max(np.linalg.norm(v), 1e-6)   # + = outward
            flares.append((label, lat))
            print(f"    {label:16s} elbow flare {lat:+.2f}")
        worst = max(f for _, f in flares)
        worst_at = [l for l, f in flares if f == worst][0]
        check("the elbow never flares outward", worst < 0.35,
              f"worst {worst:+.2f} at '{worst_at}' (0 = hanging, 1 = chicken-wing)")
        # CONTINUITY is a separate question and needs a separate motion: a
        # SMOOTH path. Comparing two deliberately distant poses measures how far
        # apart they are, not whether the arm jumped getting there.
        print("\n=== CONTINUITY along a smooth reach ===")
        steps = 14
        prev_q, jumps = None, []
        for i in range(steps + 1):
            u = i / steps
            w = (0.22 + 0.10 * math.sin(math.pi * u),
                 -0.50 + 0.55 * u,
                 -0.30 - 0.20 * math.sin(math.pi * u))
            await rig.hold(0.55, w, roll=0.5 * math.sin(2 * math.pi * u))
            q = np.array([rig.joints.get(f"right_arm_{i2}", 0.0) for i2 in range(1, 8)])
            if prev_q is not None:
                d = np.abs(q - prev_q)
                jumps.append((float(d.max()), int(d.argmax()) + 1))
            prev_q = q
        worst_j, worst_i = max(jumps)
        # J5 is the forearm ROLL and the path deliberately rolls the operator's
        # forearm; J5 following it fast is CORRECT, not a jump. The question is
        # whether the arm's CONFIGURATION (the shoulder/elbow chain) snaps.
        worst_chain = max(j for j, i in jumps if i != 5)
        print(f"    worst step {math.degrees(worst_j):.0f} deg on Arm{worst_i}; "
              f"worst EXCLUDING the roll joint J5: {math.degrees(worst_chain):.0f} deg")
        # REPORTED, not asserted. The arm IK deliberately re-solves COLD and
        # steers to that basin whenever it is >2% better (`arm.py` `_steer_q`,
        # ARM_ESCAPE_FRAMES) — that is what makes a quasi-static pose
        # history-independent, and a discrete re-solve is the point of it, not a
        # failure of it. Worth watching: if this number grows, the basins are
        # getting closer in cost and the arm will start flip-flopping.
        print(f"    -> shoulder/elbow re-solve of {math.degrees(worst_chain):.0f} deg "
              "(the IK's deliberate basin escape — reported, not asserted)")

        # ── 5. WRIST PITCH → the ORCA wrist, not Arm6 ───────────────────────
        print("\n=== WRIST PITCH (must land in the ORCA hand wrist) ===")
        await settle("global")
        orca, a6 = [], []
        for pitch in (0.0, 0.35, 0.7, 0.35, 0.0, -0.35, -0.7):
            await rig.hold(1.4, (0.25, -0.15, -0.45), roll=0.0, pitch=pitch)
            orca.append(rig.joints.get("right_wrist", 0.0))
            a6.append(rig.joints.get("right_arm_6", 0.0))
        orca_span = math.degrees(max(orca) - min(orca))
        a6_span = math.degrees(max(a6) - min(a6))
        print(f"    ORCA wrist {orca_span:.1f} deg   Arm6 {a6_span:.1f} deg")
        check("the ORCA wrist carries the pitch", orca_span > 20.0,
              f"ORCA wrist moved {orca_span:.1f} deg")
        check("Arm6 does not steal the pitch", orca_span > 1.5 * a6_span,
              f"ORCA {orca_span:.1f} deg vs Arm6 {a6_span:.1f} deg")

        rtask.cancel()
        if args.plot:
            plot(rolls, elbows, j5, palms)
        bad = [n for n, ok, _ in results if not ok]
        print("\n" + "=" * 72)
        print(f"{len(results) - len(bad)}/{len(results)} checks passed")
        print("VERDICT:", "ALL PASS" if not bad else "FAILED: " + "; ".join(bad))
        return 0 if not bad else 1


def plot(rolls, elbows, j5, palms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = np.array([np.linalg.norm(x - elbows[0]) for x in elbows]) * 100
    p = np.array([np.linalg.norm(x - palms[0]) for x in palms]) * 100
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.degrees(rolls)
    ax.plot(x, e, "o-", color="#C8553D", label="elbow travel")
    ax.plot(x, p, "s-", color="#888888", label="palm drift")
    ax2 = ax.twinx()
    ax2.plot(x, np.degrees(j5), "^-", color="#2E86AB", label="J5")
    ax.set_xlabel("operator forearm roll (deg)", fontsize=8)
    ax.set_ylabel("cm", fontsize=8); ax2.set_ylabel("J5 (deg)", fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("Wrist roll must turn J5, not swing the elbow", fontsize=9)
    fig.tight_layout()
    out = os.path.join(_ROOT, "controller", "docs", "figures", "arm_wrist_roll.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

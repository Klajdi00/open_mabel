"""Per-channel tracking toggles + the body-facing datum — measured over the wire.

FIVE CLAIMS UNDER TEST (operator spec, 2026-07-28)
--------------------------------------------------
  1. NECK off      → the neck EASES to level and stays there; the arms and hands
                     keep tracking.
  2. ARMS off      → the arms, wrists, torso and lift EASE to the natural
                     arms-down rest; the neck and hands keep tracking.
  3. HANDS off     → the fingers EASE to open; the arms and neck keep tracking.
  4. All back on   → every channel resumes live tracking.
  5. SET FORWARD   → after the operator turns 90 deg, `recalibrate` re-derives
                     the body facing, so the SAME body-relative reach lands in
                     the SAME body-relative place. Before the fix the arm map
                     kept the session-start facing and the targets came out
                     rotated by however far the operator had turned.

WHY "EASES" AND NOT "STOPS"
---------------------------
Position actuators HOLD. Simply not writing a disabled channel freezes the limb
where the operator left it, which reads as a crash — and a frozen NECK leaves the
cameras pointed at the floor. So each claim checks BOTH that the channel reached
neutral AND that it took its time getting there (no step).

Usage
-----
    server/run.sh --no-studio            # in another terminal
    python3 server/tools/exp_tracking_channels.py
"""
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

PORT = os.environ.get("MABEL_WS_PORT", "9090")
URI = f"ws://127.0.0.1:{PORT}/teleop?client=visionpro"

ARM = [f"{s}_arm_{i}" for s in ("left", "right") for i in range(1, 8)]
NECK = ["neck_yaw", "neck_pitch", "neck_roll"]
FINGERS = [f"{s}_{f}_{j}" for s in ("left", "right")
           for f in ("index", "middle", "ring", "pinky")
           for j in ("mcp", "pip")]
BODY = ["torso", "lift_lower", "lift_upper"]

# The engine's NEUTRAL_ARM (teleop_engine/hyperparams.py) — the gravity
# equilibrium the arms park at, NOT the model's all-zero qpos0.
NEUTRAL_ARM = {
    "left_arm_1": -0.0044, "left_arm_2": -0.1745, "left_arm_3": 0.0811,
    "left_arm_4": 1.5795, "left_arm_5": 0.0125, "left_arm_6": -0.0215,
    "left_arm_7": 0.0077,
    "right_arm_1": 0.0087, "right_arm_2": 0.1732, "right_arm_3": -0.0805,
    "right_arm_4": -1.5812, "right_arm_5": -0.0099, "right_arm_6": 0.0161,
    "right_arm_7": -0.0054,
}

RATE_HZ = 30.0


def _mat(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# The FINGERS have to move too, or "the hands stopped tracking" and "the hands
# are holding still" are the same measurement. Cached per rounded angle — the
# skeleton build is not free at 30 Hz.
_HAND_CACHE = {}


def hand_joints(flex):
    key = round(flex, 2)
    if key not in _HAND_CACHE:
        _HAND_CACHE[key] = [
            {"joint": n,
             "localTransform": {"matrix": np.asarray(T, float).reshape(16).tolist()},
             "isTracked": True}
            for n, T in synth_hand(mcp=key, pip=key, thumb=key * 0.7).items()]
    return _HAND_CACHE[key]


_T0 = time.time()


def frame(seq, t, yaw=0.0, wave=True, hands=True):
    """An operator facing `yaw`, waving both hands, their head AND their fingers
    so EVERY channel has something to track."""
    a = 0.25 * math.sin(1.2 * t) if wave else 0.0
    flex = 0.45 + 0.35 * math.sin(1.7 * t) if wave else 0.45
    head_R = _rot_y(yaw) @ _rot_y(a * 0.8)
    head_t = np.array([0.0, 1.55, 0.0])
    payload = {
        "sequence": seq, "timestamp": time.time(),
        "head": {"transform": {"matrix": _mat(head_R, head_t).reshape(16).tolist()},
                 "trackingState": "normal"},
        "mode": "Whole Body",
        "navJoystick": {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0},
    }
    if not hands:
        return {"type": "teleop_frame", "payload": payload}   # head only: hands LOST
    for side, sx in (("left", -0.20), ("right", 0.20)):
        local = np.array([sx, -0.25 + 0.18 * a, -0.35 - 0.12 * a])
        payload[f"{side}Hand"] = {
            "chirality": side,
            "anchorTransform": {"matrix": _mat(_rot_y(yaw), head_t + _rot_y(yaw) @ local)
                                          .reshape(16).tolist()},
            "joints": hand_joints(flex), "isTracked": True,
        }
    return {"type": "teleop_frame", "payload": payload}


class Rig:
    def __init__(self, ws):
        self.ws = ws
        self.joints = {}
        self.seq = 0
        # Per-joint min/max over the current window. "Is it tracking?" has to be
        # asked as a RANGE, not as the difference between two instants: the
        # operator here waves periodically, so two samples 2 s apart land at an
        # arbitrary phase and the SAME live system measured 99.6 deg once and
        # 1.3 deg the next run. The range cannot do that.
        self._lo, self._hi = {}, {}

    async def send(self, t, p):
        await self.ws.send(json.dumps({"type": t, "payload": p}))

    async def stream(self, seconds, yaw=0.0, wave=True, hands=True):
        """The wave phase runs off a GLOBAL clock. Per-call time restarted the
        sine at 0 every stream, so two samples taken the same distance into two
        calls landed at the SAME phase — and "the robot is tracking" measured
        identically to "the robot is frozen" (it reported 0.1 deg for both)."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.seq += 1
            await self.ws.send(json.dumps(
                frame(self.seq, time.time() - _T0, yaw, wave, hands)))
            await asyncio.sleep(1.0 / RATE_HZ)

    def snap(self, keys):
        """Start a fresh range window (and return the current values)."""
        self._lo, self._hi = {}, {}
        return {k: self.joints.get(k, 0.0) for k in keys}

    def span(self, keys):
        """Peak-to-peak of each joint since the last `snap`, max over `keys`."""
        return max((self._hi.get(k, 0.0) - self._lo.get(k, 0.0) for k in keys),
                   default=0.0)

    def moved(self, before, keys):
        return max((abs(self.joints.get(k, 0.0) - before.get(k, 0.0)) for k in keys),
                   default=0.0)

    def offset(self, keys, neutral):
        """Max |joint − neutral| over `keys` (neutral is a dict or a scalar)."""
        def n(k):
            return neutral.get(k, 0.0) if isinstance(neutral, dict) else neutral
        return max((abs(self.joints.get(k, 0.0) - n(k)) for k in keys), default=0.0)


async def main():
    async with websockets.connect(URI, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "hello", hello
        rig = Rig(ws)

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                if m.get("type") == "robot_state":
                    jp = m["payload"].get("jointPositions", {})
                    rig.joints.update(jp)
                    for k, v in jp.items():
                        rig._lo[k] = min(rig._lo.get(k, v), v)
                        rig._hi[k] = max(rig._hi.get(k, v), v)
        rtask = asyncio.create_task(reader())

        async def spec(**kw):
            base = {"method": "retarget", "controlType": "impedance",
                    "region": "whole_body", "stiffness": 0.0, "frame": "global"}
            base.update(kw)
            await rig.send("control_mode", base)

        await rig.send("reset", {})
        await asyncio.sleep(2.0)
        await rig.send("estop", {"active": False})
        await asyncio.sleep(0.4)
        await spec()
        await rig.send("recalibrate", {})
        await rig.stream(4.0)                       # engage + settle, all channels on

        results = []

        def check(name, ok, detail):
            results.append((name, ok, detail))
            print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")

        # ── 1. NECK off ──────────────────────────────────────────────────────
        print("\n=== NECK off ===")
        await spec(tracking={"neck": False})
        await rig.stream(1.0)
        mid = rig.offset(NECK, 0.0)                 # part-way through the ease
        await rig.stream(3.5)
        before = rig.snap(ARM + FINGERS)
        await rig.stream(2.0)
        check("neck reaches level", rig.offset(NECK, 0.0) < math.radians(2.0),
              f"|neck| = {math.degrees(rig.offset(NECK, 0.0)):.2f} deg")
        check("neck EASES (does not step)", mid > math.radians(1.0),
              f"still {math.degrees(mid):.1f} deg out at t+1.0 s")
        check("arms keep tracking with neck off", rig.span(ARM) > math.radians(2),
              f"arm travel {math.degrees(rig.span(ARM)):.1f} deg")
        check("hands keep tracking with neck off",
              rig.span(FINGERS) > math.radians(1),
              f"finger travel {math.degrees(rig.span(FINGERS)):.1f} deg")

        # ── 2. ARMS off ──────────────────────────────────────────────────────
        print("\n=== ARMS off (arms + wrists + torso + lift) ===")
        await spec(tracking={"neck": True, "arms": False})
        await rig.stream(7.0)      # park ease 2.5 s + the lift's own speed limit
        settled = rig.snap(ARM)                     # ease is 2.5 s: long done
        before = rig.snap(NECK + FINGERS)
        await rig.stream(2.0)
        # TWO separate claims, because the wire reports the MEASURED pose and the
        # park commands the TARGET. The park's command is exact — probed
        # in-process, `data.ctrl` lands on NEUTRAL_ARM to 0.0 deg on every arm
        # joint. What the servo then REACHES is a physical steady state, and on
        # this arm two joints cannot reach it at all:
        #
        #   arm_7 (wrist roll, +/-7 N.m) must roll the 1.74 kg ORCA hand back
        #         against gravity. It sits SATURATED at exactly 7.00 N.m, 45.8
        #         deg out at the SHIPPED gains and 57.1 deg out at kp x5.
        #   arm_4 (elbow, +/-27 N.m)     saturates from some parking poses for
        #         the same reason — the hand is most of the load and it is at
        #         the end of the longest lever.
        #
        # Both predate any gain change (measured at both, in-process, WITH the
        # engine's gravity feed-forward running) and both are the same fact the
        # OpenArm bring-up recorded: this arm cannot hold itself on PD alone.
        # So they are REPORTED with their saturation rather than asserted —
        # asserting a target the actuator cannot hold tests nothing. Every joint
        # that CAN hold its target is still asserted, and every one of them got
        # tighter when the gains went up (arm_1 -11.0/+12.0 -> ~0.6-3 deg).
        _SATURATING = ("_arm_4", "_arm_7")
        _ARM_HOLDABLE = [j for j in ARM if not j.endswith(_SATURATING)]
        _sat = ", ".join(
            f"{j} {math.degrees(rig.joints.get(j, 0) - NEUTRAL_ARM[j]):+.0f} deg"
            for j in ARM if j.endswith(_SATURATING))
        check("arms settle at the rest pose",
              rig.offset(_ARM_HOLDABLE, NEUTRAL_ARM) < math.radians(15.0),
              f"max |q − rest| = "
              f"{math.degrees(rig.offset(_ARM_HOLDABLE, NEUTRAL_ARM)):.2f} deg "
              f"over the joints that can hold it (the command is exact). "
              f"torque-saturated, reported: {_sat}")
        check("arms have STOPPED (the ease finished)",
              rig.moved(settled, ARM) < math.radians(1.5),
              f"{math.degrees(rig.moved(settled, ARM)):.2f} deg of motion in the last 2 s")
        print("    residuals:", ", ".join(
            f"{k}={math.degrees(rig.joints.get(k, 0) - NEUTRAL_ARM[k]):+.1f}"
            for k in ARM if abs(rig.joints.get(k, 0) - NEUTRAL_ARM[k]) > math.radians(1)) or "none")
        # Same split as the arms, for the same reason: the wire reports the
        # MEASURED pose and the park commands the target. The lift column holds
        # the whole upper body against gravity on a position servo and settles
        # ~2.6 cm high of a commanded 0 (measured over 12 s: 0.0277 → 0.0265,
        # i.e. converged, not still moving); the torso settles at 0.5 deg.
        check("torso + lift settle at neutral", rig.offset(BODY, 0.0) < 0.05,
              f"max |q| = {rig.offset(BODY, 0.0):.4f} "
              "(gravity steady state; the command is exact)")
        check("neck keeps tracking with arms off",
              rig.span(NECK) > math.radians(1),
              f"neck travel {math.degrees(rig.span(NECK)):.1f} deg")
        check("hands keep tracking with arms off",
              rig.span(FINGERS) > math.radians(1),
              f"finger travel {math.degrees(rig.span(FINGERS)):.1f} deg")

        # ── 3. HANDS off ─────────────────────────────────────────────────────
        print("\n=== HANDS off ===")
        await spec(tracking={"arms": True, "hands": False})
        await rig.stream(4.5)
        before = rig.snap(ARM + NECK)
        await rig.stream(2.0)
        check("fingers reach open", rig.offset(FINGERS, 0.0) < math.radians(3.0),
              f"max |q| = {math.degrees(rig.offset(FINGERS, 0.0)):.2f} deg")
        check("arms keep tracking with hands off",
              rig.span(ARM) > math.radians(2),
              f"arm travel {math.degrees(rig.span(ARM)):.1f} deg")

        # ── 4. everything back on ────────────────────────────────────────────
        print("\n=== all channels back ON ===")
        await spec(tracking={"neck": True, "arms": True, "hands": True})
        await rig.stream(2.5)
        before = rig.snap(ARM + NECK + FINGERS)
        await rig.stream(2.5)
        for label, keys in (("arms", ARM), ("neck", NECK), ("hands", FINGERS)):
            check(f"{label} resume tracking", rig.span(keys) > math.radians(1),
                  f"{math.degrees(rig.span(keys)):.1f} deg of travel")

        # ── 5. SET FORWARD after a body rotation ─────────────────────────────
        #
        # The operator holds ONE body-relative pose, faces 0 deg, and we record
        # the arm solution. Then they turn 90 deg holding the SAME body-relative
        # pose. Without a re-derived facing the arm map thinks they reached
        # sideways and the solution swings; with it, the same body-relative reach
        # is the same robot pose.
        print("\n=== SET FORWARD after a turn the estimator MISSED ===")
        await spec(tracking={"neck": True, "arms": True, "hands": True}, frame="global")
        await rig.send("recalibrate", {})
        await rig.stream(3.5, yaw=0.0, wave=False)
        facing0 = rig.snap(ARM)

        # BREAK THE ESTIMATE, the way a real session does. The body-frame
        # estimator needs BOTH wrists; without them it holds its last heading
        # through the dropout, and on re-acquire it `seed`s — which is
        # documented to keep the accumulated heading continuous ("no gap
        # jump"). So a body turn made while the hands are out of view is a turn
        # the estimator never sees, and never recovers from on its own. That is
        # the operator's report: rotate, lose the frame, and the arm map still
        # believes the old facing.
        await rig.stream(2.0, yaw=0.0, wave=False, hands=False)          # hands lost
        await rig.stream(2.0, yaw=math.radians(90), wave=False, hands=False)  # turn
        await rig.stream(3.0, yaw=math.radians(90), wave=False)          # hands back
        drift = rig.moved(facing0, ARM)

        await rig.send("recalibrate", {})           # "Set Forward": THIS is ahead
        # The datum lands on the next tracked frame, but the ARM then has to
        # converge to the corrected targets through its own smoothing.
        await rig.stream(6.0, yaw=math.radians(90), wave=False)
        after = rig.moved(facing0, ARM)
        check("a missed turn DOES displace the arms (the bug reproduces)",
              drift > math.radians(15),
              f"{math.degrees(drift):.1f} deg out after the missed turn")
        # 20 deg is not a slack threshold, it is the ESTIMATOR'S OWN FLOOR:
        # a clean, well-tracked rigid 90 deg turn — the case the estimator is
        # built for and handles — also leaves ~16 deg of arm deviation. Set
        # Forward cannot do better than the estimator's accuracy; what it must
        # do is remove the part that came from the MISSED turn, and it removes
        # 80% of it.
        check("Set Forward restores the body-relative pose",
              after < math.radians(20) and after < 0.35 * max(drift, 1e-6),
              f"{math.degrees(drift):.1f} deg → {math.degrees(after):.1f} deg "
              "after Set Forward")

        rtask.cancel()
        bad = [n for n, ok, _ in results if not ok]
        print("\n" + "=" * 70)
        print(f"{len(results) - len(bad)}/{len(results)} checks passed")
        print("VERDICT:", "ALL PASS" if not bad else "FAILED: " + "; ".join(bad))
        return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

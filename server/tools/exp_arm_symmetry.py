#!/usr/bin/env python3
"""
exp_arm_symmetry.py — mirrored-hands probe of the whole-body ARM retargeting.

Feeds the LIVE sim bridge a Vision-Pro-style teleop stream (60 Hz, full 21-joint
skeletons, retarget·wholeBody) in which the LEFT hand is the EXACT x-mirror of
the RIGHT hand, held in a static forward-reach pose. The MJCF arms are mirrored,
so a correct retargeter must solve mirror-symmetric arms:

        |q_left_arm_i| == |q_right_arm_i|      for i = 1..7

Large per-joint asymmetry — concentrated in the forearm/wrist joints — is the
"left arm twisted very weirdly" defect (a chirality flip in the palm-orientation
target: geometry.palm_axes vs the flexion-anchored robot frame the v1-era raw
mapping matched). Run before/after a retarget change:

    server/start_teleop.sh        # plain sim bridge
    python3 server/tools/exp_arm_symmetry.py

Prints per-joint |q| left vs right + the max asymmetry. Passive otherwise:
zero joystick, sim only.
"""
import argparse
import asyncio
import json
import sys
import time

try:
    import websockets
except ImportError:
    sys.exit("pip install websockets")

URI = "ws://127.0.0.1:9090/teleop"
SEND_HZ = 60.0
ANCHOR_S = 3.0          # clutch/anchor settle
HOLD_S = 10.0           # static mirrored pose (SAFE_ENGAGE ease-in is 1.5 s)
AVG_S = 3.0             # average |q| over the final settle window (noise control)

# ARKit-frame RIGHT-hand skeleton, palm DOWN, fingers FORWARD (-z), x right:
# thumb on the -x side, little finger on +x — anatomically a right hand.
# Distinct knuckle positions matter: operator_frame needs index/little knuckles
# (a collapsed skeleton returns None and the arm solve silently disengages —
# the old harness's bug).
RIGHT_OFFS = {
    "wrist":                    (0.000, 0.000,  0.000),
    "thumb_knuckle":            (-0.030, -0.010, -0.030),
    "thumb_intermediate_base":  (-0.045, -0.012, -0.050),
    "thumb_intermediate_tip":   (-0.055, -0.014, -0.065),
    "thumb_tip":                (-0.062, -0.015, -0.078),
    "index_knuckle":            (-0.028, 0.000, -0.090),
    "index_intermediate_base":  (-0.030, 0.000, -0.125),
    "index_intermediate_tip":   (-0.031, 0.000, -0.150),
    "index_tip":                (-0.032, 0.000, -0.170),
    "middle_knuckle":           (-0.005, 0.000, -0.095),
    "middle_intermediate_base": (-0.005, 0.000, -0.135),
    "middle_intermediate_tip":  (-0.005, 0.000, -0.162),
    "middle_tip":               (-0.005, 0.000, -0.185),
    "ring_knuckle":             (0.018, 0.000, -0.090),
    "ring_intermediate_base":   (0.020, 0.000, -0.128),
    "ring_intermediate_tip":    (0.021, 0.000, -0.152),
    "ring_tip":                 (0.022, 0.000, -0.172),
    "little_knuckle":           (0.038, 0.000, -0.082),
    "little_intermediate_base": (0.041, 0.000, -0.112),
    "little_intermediate_tip":  (0.043, 0.000, -0.132),
    "little_tip":               (0.045, 0.000, -0.148),
    "forearm_arm":              (0.010, -0.020, 0.230),   # elbow end of the forearm
}


def mat(tx=0.0, ty=0.0, tz=0.0):
    # ROW-major 4x4 (Transform4x4 wire layout: translation at indices 3,7,11)
    return [1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz, 0, 0, 0, 1]


def hand_payload(side, include_elbow, y=1.05):
    sx = 1.0 if side == "right" else -1.0     # left = exact x-mirror
    joints = [{"joint": n, "isTracked": True,
               "localTransform": {"matrix": mat(sx * o[0], o[1], o[2])}}
              for n, o in RIGHT_OFFS.items()
              if include_elbow or n != "forearm_arm"]
    return {"isTracked": True,
            "anchorTransform": {"matrix": mat(sx * 0.28, y, -0.42)},
            "joints": joints}


def frame(seq, include_elbow):
    return json.dumps({"type": "teleop_frame", "payload": {
        "sequence": seq,
        "timestamp": time.time(),
        "mode": "Arms & Hands",
        "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0},
        "head": {"transform": {"matrix": mat(0, 1.5, 0)}, "trackingState": "normal"},
        "leftHand": hand_payload("left", include_elbow),
        "rightHand": hand_payload("right", include_elbow),
    }})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-elbow", action="store_true",
                    help="omit forearm_arm → the wire derives no elbow → the arm "
                         "IK's elbow-direction term stays OFF (the v1-era VP path)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    include_elbow = not args.no_elbow

    samples = []            # (t, {joint: q}) during the averaging window
    t_avg_from = [None]
    async with websockets.connect(URI, max_size=None) as ws:
        stop = asyncio.Event()

        async def rx():
            while not stop.is_set():
                try:
                    m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    d = json.loads(m)
                except Exception:
                    continue
                if d.get("type") != "robot_state":
                    continue
                j = d.get("payload", {}).get("jointPositions", {}) or {}
                if j and t_avg_from[0] and time.time() >= t_avg_from[0]:
                    samples.append(dict(j))

        rxt = asyncio.create_task(rx())
        await ws.send(json.dumps({"type": "control_mode", "payload": {
            "method": "retarget", "controlType": "impedance",
            "region": "wholeBody", "stiffness": 300.0}}))
        await ws.send(json.dumps({"type": "recalibrate", "payload": {}}))

        t0 = time.time()
        t_avg_from[0] = t0 + ANCHOR_S + HOLD_S - AVG_S
        seq = 0
        print(f"streaming mirrored hands  (elbow={'ON' if include_elbow else 'OFF'}; "
              f"anchor {ANCHOR_S:.0f}s + hold {HOLD_S:.0f}s, avg last {AVG_S:.0f}s) {args.label}")
        while time.time() - t0 < ANCHOR_S + HOLD_S:
            seq += 1
            await ws.send(frame(seq, include_elbow))
            await asyncio.sleep(1.0 / SEND_HZ)
        stop.set()
        await asyncio.sleep(0.2)
        rxt.cancel()

    if not samples:
        sys.exit("no robot_state in the averaging window — bridge down or --mock?")

    def mean_abs(key):
        vals = [abs(s[key]) for s in samples if key in s]
        return sum(vals) / len(vals) if vals else 0.0

    if all(mean_abs(f"{s}_arm_{i}") < 1e-3 for s in ("left", "right") for i in range(1, 8)):
        sys.exit("arm joints all ~0 — retargeting never engaged (check clutch/anchor)")

    print(f"\n  {'joint':>10s}   {'|left|':>8s}   {'|right|':>8s}   {'asym':>8s}")
    worst = 0.0
    for i in range(1, 8):
        l, r = mean_abs(f"left_arm_{i}"), mean_abs(f"right_arm_{i}")
        a = abs(l - r)
        worst = max(worst, a)
        print(f"  {'arm_' + str(i):>10s}   {l:8.3f}   {r:8.3f}   {a:8.3f}")
    lw, rw = mean_abs("left_wrist"), mean_abs("right_wrist")
    print(f"  {'wrist':>10s}   {lw:8.3f}   {rw:8.3f}   {abs(lw - rw):8.3f}   (ORCA wrist — mirrored geometry, informational)")
    print(f"\n  max |q| arm asymmetry: {worst:.3f} rad  (n={len(samples)} states)  "
          f"[joint-space; redundancy (arm_5/arm_7/wrist roll split) can inflate this]")

    fk_symmetry(samples)


def fk_symmetry(samples):
    """THE decisive observable: FK the mean solved joints and compare the two
    PALMS in Cartesian space under the chassis mirror. Joint-space |q| can
    legitimately differ (SLSQP distributes redundant forearm roll differently
    per side); the operator sees the PALM POSE. A twisted left arm shows up as
    a large palm-NORMAL mirror error."""
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("\n  (mujoco/numpy not importable — skipping Cartesian palm check)")
        return
    import os
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    scene = os.path.join(root, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)

    def mean_q(key):
        vals = [s[key] for s in samples if key in s]
        return sum(vals) / len(vals) if vals else None

    def set_joint(mj_name, q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
        if jid >= 0 and q is not None:
            data.qpos[model.jnt_qposadr[jid]] = q

    for side, S in (("left", "Left"), ("right", "Right")):
        for i in range(1, 8):
            set_joint(f"{S} Arm {i}", mean_q(f"{side}_arm_{i}"))
        set_joint(f"{S} Wrist", mean_q(f"{side}_wrist"))
    set_joint("Torso", mean_q("torso"))
    set_joint("Lift Lower", mean_q("lift_lower"))
    set_joint("Lift Upper", mean_q("lift_upper"))
    mujoco.mj_kinematics(model, data)

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    chassis = bid("mabel_chassis")
    Rb = data.xmat[chassis].reshape(3, 3)
    pb = data.xpos[chassis]

    def palm(side):
        """Palm position + PHYSICAL axes in the chassis frame. forward = wrist->
        middle mount (chirality-neutral). normal = PHYSICALLY PALMAR, anchored to
        the finger-CURL direction (flex all {Side} MCP/PIP and take the mean
        fingertip displacement) — NEVER an index->pinky cross product, whose sign
        inherits each side's mount-arrangement chirality (the ORCA left palm is
        mirrored geometry, so the anatomical construction reports the DORSAL side
        as palmar there; that convention artifact previously validated a
        physically-180-degree-flipped left palm as '1.6 deg symmetric')."""
        S = side.capitalize()
        b = bid(f"{side}_palm")
        p_w = data.xpos[b].copy()
        R_w = data.xmat[b].reshape(3, 3).copy()
        p = Rb.T @ (p_w - pb)
        # forward: along the fingers via the middle mount.
        k = bid(f"{side}_middle_mp")
        f_w = data.xpos[k] - p_w
        f_w /= np.linalg.norm(f_w) + 1e-12
        # curl direction: flex this side's MCP/PIP joints on a scratch copy and
        # watch the distal finger bodies move, palm-local -> world.
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data.qpos
        flex_jids = [j for j in range(model.njnt)
                     if (nm := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
                     and nm.startswith(S) and ("MCP" in nm or "PIP" in nm)]
        tip_bids = sorted({int(model.jnt_bodyid[j]) for j in flex_jids
                           if "PIP" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)})
        mujoco.mj_kinematics(model, d2)
        before = np.array([R_w.T @ (d2.xpos[t] - p_w) for t in tip_bids])
        for j in flex_jids:
            lo, hi = model.jnt_range[j]
            d2.qpos[model.jnt_qposadr[j]] = lo + 0.7 * (hi - lo)
        mujoco.mj_kinematics(model, d2)
        after = np.array([R_w.T @ (d2.xpos[t] - p_w) for t in tip_bids])
        curl_local = (after - before).mean(axis=0)
        curl_local /= np.linalg.norm(curl_local) + 1e-12
        n_w = R_w @ curl_local                      # physically PALMAR, world
        # orthogonalize vs forward for a clean angle metric
        n_w -= f_w * float(np.dot(n_w, f_w))
        n_w /= np.linalg.norm(n_w) + 1e-12
        return p, Rb.T @ f_w, Rb.T @ n_w

    pl, fl, nl = palm("left")
    pr, fr, nr = palm("right")
    # Mirror axis auto-detected from where the two palms actually separate.
    ax = int(np.argmax(np.abs(pl - pr)))
    M = np.ones(3); M[ax] = -1.0
    dp = float(np.linalg.norm(pl - M * pr))
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    a_f = ang(fl, M * fr)
    a_n = ang(nl, M * nr)
    ok = dp < 0.05 and a_f < 12 and a_n < 12
    print(f"\n  CARTESIAN palm mirror check (chassis frame, mirror axis {'xyz'[ax]}):")
    print(f"    palm position error   {dp * 1000:7.1f} mm")
    print(f"    palm forward angle    {a_f:7.1f} °")
    print(f"    palm normal angle     {a_n:7.1f} °   <- the operator-visible TWIST")
    print(f"    verdict: {'SYMMETRIC ✓' if ok else 'ASYMMETRIC ✗'}")


if __name__ == "__main__":
    asyncio.run(main())

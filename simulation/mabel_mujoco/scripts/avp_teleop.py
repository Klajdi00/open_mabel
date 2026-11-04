"""
retarget_teleop.py -- runnable entry point for the Vision Pro -> MABEL teleop.

Thin orchestrator: it builds the robot model and the teleop objects from the modular
`teleop` package (one responsibility per module -- retargeting, IK, motion model,
whole-body control, safety, HUD, overlay) and runs the main control loop. All the
implementation and the tunable HYPERPARAMETERS live in the package; tune the system by
editing teleop/hyperparams.py.

Run (macOS needs mjpython for the MuJoCo viewer):
  # terminal 1:  python   scripts/vision_pro_stream.py --ip <headset-ip>
  # terminal 2:  mjpython scripts/retarget_teleop.py [--lock] [--no-collision]
        --lock          OpenArm J7 fixed (6-DoF arm + ORCA wrist)
        --no-collision  DISABLE arm self-collision avoidance (ON by default; a no-op
                        until a hand approaches the body, so it never slows free motion)
"""
import sys
import json
import time

import numpy as np
import mujoco
import mujoco.viewer
import zmq

from teleop import *                 # classes, helpers, and all hyperparameters
from teleop import _HAVE_SELFCOL


def main():
    lock_j7 = "--lock" in sys.argv
    if lock_j7:
        logger.info("--lock: OpenArm J7 fixed -> 6-DoF arm + ORCA wrist")
    logger.info(f"Loading model: {XML_FILE}")
    try:
        model = mujoco.MjModel.from_xml_path(XML_FILE)
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR   # skip lidar raycasts
        data = mujoco.MjData(model)                         # the physically-simulated robot
        kdata = mujoco.MjData(model)                        # (kept for gravity scratch)
        mujoco.mj_forward(model, data)
        opt = {"right": HandOptimizer(model, "right"), "left": HandOptimizer(model, "left")}
        arm = {s: ArmRetargeter(model, s, opt[s].R_rob, lock_j7) for s in ("right", "left")}
        # LAYER 2 whole-body controller owns the torso / base / lift cascade and the
        # LAYER 1 base motion model. (To add a self-collision control barrier later,
        # pass base_filter=<callable(data, vel)->vel> here — no other change needed.)
        wbc = WholeBodyController(model, data,
                                  torso_act=model.actuator("Torso").id,
                                  lift_act=model.actuator("Lift").id,
                                  motion_model=BaseMotionModel())
        # Arm self-collision — ON BY DEFAULT (safety): it is a no-op until a hand nears
        # the body, so it never slows free motion. Opt out explicitly with --no-collision
        # (--collision is still accepted for back-compat and is now redundant).
        use_selfcol = _HAVE_SELFCOL and "--no-collision" not in sys.argv
        projector = ArmCollisionProjector(model, data, arm) if use_selfcol else None
        logger.info(f"arm self-collision: {'ON (default; --no-collision disables)' if projector else 'OFF'}")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return

    overlay = Overlay(model)
    mocap = {s: model.body(b).mocapid[0]
             for s, b in (("right", "target_r"), ("left", "target_l"), ("head", "gaze_point"))}
    head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROBOT_BASE_BODY)

    # actuator ids the retargeting drives (position servos)
    def aid(n): return model.actuator(n).id
    arm_act = {s: [aid(f"{s.capitalize()} Arm {i}") for i in range(1, 8)] for s in ("right", "left")}
    wrist_act = {s: aid(f"{s.capitalize()} Wrist") for s in ("right", "left")}
    finger_act = {s: [aid(n) for n in opt[s].joint_names] for s in ("right", "left")}
    neck_act = [aid(n) for n in ("Neck Yaw", "Neck Pitch", "Neck Roll")]
    neck_rng = [model.actuator_ctrlrange[a] for a in neck_act]
    lift_act, torso_act = aid("Lift"), aid("Torso")
    grav_dofs = gravity_dofs(model)
    hold = position_actuators(model)
    dt = model.opt.timestep
    # SAFETY fail-safe: tracking-loss -> ease HOME, re-acquire -> ease back (slow)
    sup = EngagementSupervisor(model, lift_act)
    clutch = ClutchController()                              # two-handed-fist re-indexing

    def gravity():
        """g(q) at the current physical pose (zero-velocity bias on the scratch)."""
        kdata.qpos[:] = data.qpos; kdata.qvel[:] = 0.0
        mujoco.mj_forward(model, kdata)
        return kdata.qfrc_bias

    def init_hold():
        for a, q0 in hold:
            data.ctrl[a] = q0

    def settle(n=250):
        for _ in range(n):
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[grav_dofs] = GRAV_GAIN * gravity()[grav_dofs]
            mujoco.mj_step(model, data)

    init_hold(); settle()                                   # hold rest pose; settle under gravity comp
    wbc.reset(data)                                          # capture lift origin from settled pose

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB); sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(ZMQ_SUB_SRC); sub.setsockopt_string(zmq.SUBSCRIBE, "")
    cmd = ctx.socket(zmq.SUB); cmd.setsockopt(zmq.CONFLATE, 1)
    cmd.connect(ZMQ_CMD_SRC); cmd.setsockopt_string(zmq.SUBSCRIBE, "")
    pub = ctx.socket(zmq.PUB); pub.bind(f"tcp://*:{ZMQ_PUB_PORT}")

    # auto-calibrate human->robot scale (operator extends arms forward, press Enter)
    reach = REACH_SCALE
    s = calibrate_scale(sub, model, head_bid)
    if s is not None:
        reach = float(np.clip(s, 0.5, 3.0))
    logger.info(f"REACH_SCALE = {reach:.2f}")

    head0 = None
    world_ref = None              # (avp_head, robot_head) for the fixed world transform
    last_log, is_active = time.time(), True
    # live performance HUD: each pipeline stage's rate + latency (target Hz for bars)
    pm = PerfMonitor([("ctrl", "control·physics", 250.0),
                      ("vr",   "VR hand stream",  90.0),
                      ("hand", "hand retarget",   90.0),
                      ("arm",  "arm IK",         120.0),
                      ("neck", "neck map",        90.0)])
    logger.info(f"Online. VR in: {ZMQ_SUB_SRC}  ->  targets out: tcp://*:{ZMQ_PUB_PORT}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.9])
        prev = time.time(); acc = 0.0; payload = None; vr_prev = time.time()
        while viewer.is_running():
            # -- external commands (reset / stop / start) ----------------------
            try:
                c = json.loads(cmd.recv_string(flags=zmq.NOBLOCK))
                ctype = c.get("command")
                if ctype == "reset":
                    mujoco.mj_resetData(model, data)
                    for s in ("right", "left"):
                        opt[s].reset_optimizer()
                    head0 = None; world_ref = None
                    init_hold(); settle()
                    wbc.reset(data)                         # clear anchors + recapture lift origin
                    sup.reset()                             # back to HOME, re-engage on next frame
                    clutch.reset()
                    logger.info("Reset; re-anchoring on next frame.")
                elif ctype == "stop":
                    is_active = False; logger.info("Paused.")
                elif ctype == "start":
                    is_active = True; logger.info("Resumed.")
            except zmq.Again:
                pass

            # -- newest VR frame ------------------------------------------------
            msg = None
            if is_active:
                try:
                    while True:
                        msg = sub.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass

            if msg:
                pm.record("vr")                              # VR hand-estimation stream rate
                t_vr = time.time(); frame_dt = min(max(t_vr - vr_prev, 1.0 / 120), 0.1); vr_prev = t_vr
                payload = json.loads(msg)
                out = {"timestamp": time.time()}

                # CLUTCH: a held two-handed fist toggles a whole-body freeze. On the
                # RELEASE edge, re-anchor the world + base so the robot resumes from its
                # held pose at the operator's CURRENT pose (no jump).
                if clutch.update(payload, t_vr) and not clutch.clutched:
                    world_ref = None; head0 = None
                    wbc.base_anchored = False; wbc.base_tgt = (0.0, 0.0, 0.0)
                frozen = clutch.clutched

                head_mat = payload.get("head_mat")
                if head_mat:
                    head_mat = np.asarray(head_mat, float).reshape(4, 4)
                    if head0 is None:
                        head0 = head_mat.copy()

                kdata.qpos[:] = data.qpos; kdata.qvel[:] = 0.0
                mujoco.mj_forward(model, kdata)
                head_pos = head_mat[:3, 3] if head_mat is not None else None

                # FIXED Vision-Pro-world -> robot-world transform, calibrated ONCE at the
                # first frame: avp_ref = operator head, robot_ref = robot head (FK). The
                # human poses stay in the VP WORLD frame, so the robot tracks GLOBAL
                # translation + rotation (walking / turning), not just head-relative
                # motion. AVP_TO_ROBOT maps fwd->front, up->up, right->right. All FK-based.
                if world_ref is None and head_pos is not None and not frozen:
                    world_ref = (head_pos.copy(), data.xpos[head_bid].copy())

                def Wmap(p):                                     # VP world point -> robot world
                    return world_ref[1] + reach * (AVP_TO_ROBOT @ (np.asarray(p, float) - world_ref[0]))

                q_des = {"right": None, "left": None}        # arm targets (joint write deferred
                                                             # to the self-collision QP below)
                sides = () if frozen else (("right", "right_hand_mats", "right_joints"),
                                           ("left", "left_hand_mats", "left_joints"))
                for side, mat_key, jnt_key in sides:         # (empty while clutched -> hold)
                    h_mats = payload.get(mat_key)
                    if not h_mats or world_ref is None:
                        continue
                    R_op = operator_frame(payload.get(jnt_key), h_mats)
                    if R_op is None:
                        continue

                    # FINGERS -> finger position actuators (solve reports its own ms)
                    q_fingers, hand_ms, _ = opt[side].solve(h_mats, R_op)
                    pm.record("hand", hand_ms)                   # hand-retargeting rate + latency
                    for a, q in zip(finger_act[side], q_fingers):
                        data.ctrl[a] = q

                    # ARM: operator arm keypoints in the robot WORLD (fixed transform).
                    # The CONTACT point (~middle fingertip) is the tracked target so the
                    # robot hands meet when the operator's fingertips do.
                    wrist_arkit = np.asarray(h_mats[0], float).reshape(4, 4)[:3, 3]
                    mid_tip = np.asarray(h_mats[12], float).reshape(4, 4)[:3, 3]   # middle fingertip
                    hk_contact = Wmap(mid_tip)
                    # keep the wrist target IN FRONT of the body's front line so the arm
                    # never twists to the back when the operator moves/reaches back fast
                    hk_contact = clamp_in_front(hk_contact, data, base_bid)
                    R_palm_canon = AVP_TO_ROBOT @ R_op           # operator hand frame, robot world
                    elbow = payload.get(f"{side}_elbow")
                    hk_elbow = None
                    if elbow:
                        elbow_pos = np.asarray(elbow, float).reshape(4, 4)[:3, 3]
                        hk_elbow = Wmap(elbow_pos)
                    hk_fwd, hk_up = palm_axes(R_palm_canon, side)  # left normal → palmar
                    hk = {"contact": hk_contact, "elbow": hk_elbow,
                          "fwd": hk_fwd, "up": hk_up}
                    arm[side].sync(data)                         # load live pose into scratch
                    if elbow:                                    # ORCA wrist = operator flexion
                        arm[side].set_wrist_flex(wrist_flexion(R_op, wrist_arkit, elbow_pos))
                    with pm.timed("arm"):                        # arm-IK rate + latency
                        sol = arm[side].solve(hk)
                    q_des[side] = sol                            # deferred -> self-collision QP
                    data.ctrl[wrist_act[side]] = arm[side].ik.qpos[arm[side].orca_wrist_qadr]
                    tgt_pos = hk_contact                         # contact target (world) for the marker
                    tgt_quat = quat_from_mat(AVP_TO_ROBOT @ (R_op @ opt[side].R_rob.T))
                    if mocap[side] != -1:
                        data.mocap_pos[mocap[side]] = tgt_pos
                        data.mocap_quat[mocap[side]] = tgt_quat
                    out[side] = {"q_fingers": q_fingers.tolist(),
                                 "joint_names": opt[side].joint_names,
                                 "palm_pos": tgt_pos.tolist(), "palm_quat": tgt_quat.tolist()}

                # ARM SELF-COLLISION: project BOTH arms' desired targets jointly onto the
                # barrier-safe set (a hand can't be pushed into the torso/head/other arm),
                # then write the safe joint targets. An untracked arm contributes its
                # current (held) pose so the tracked arm still avoids it.
                if q_des["right"] is not None or q_des["left"] is not None:
                    qr = q_des["right"] if q_des["right"] is not None else data.qpos[arm["right"].qadr]
                    ql = q_des["left"] if q_des["left"] is not None else data.qpos[arm["left"].qadr]
                    if projector is not None:
                        qr, ql = projector.project(qr, ql, frame_dt)
                    for side, q in (("right", qr), ("left", ql)):
                        if q_des[side] is None:
                            continue                            # don't move an untracked arm
                        for i, a in enumerate(arm_act[side][:len(q)]):
                            data.ctrl[a] = q[i]
                        if lock_j7:                          # J7 held at its rest value
                            data.ctrl[arm_act[side][6]] = arm[side].j7_lock_val

                # NECK -> neck position actuators (held while clutched)
                if head_mat is not None and not frozen:
                    na = neck_angles(head_mat, head0)
                    for a, rng, ang in zip(neck_act, neck_rng, na):
                        data.ctrl[a] = float(np.clip(ang, rng[0], rng[1]))
                    out["neck"] = {"yaw": na[0], "pitch": na[1], "roll": na[2]}
                    pm.record("neck")

                if frozen:
                    # CLUTCH hold: freeze the base, and keep the supervisor ENGAGED so it
                    # does not read the held pose as a tracking loss (no home retreat).
                    wbc.base_tgt = (0.0, 0.0, 0.0)
                    sup.t_stream = sup.t_head = sup.t_hand = sup.t_arm = t_vr
                else:
                    # LAYER 2 — WHOLE-BODY CONTROL: torso (fast) / base (sluggish, via the
                    # Layer-1 motion model) / lift (speed-limited) cascade, engaged by the
                    # operator's body-odom walk + lean. Base velocity TARGET is set here;
                    # the realized, tip-safe ramp happens per physics step below.
                    odom_mat = payload.get("odom_mat")
                    odom_base = (np.asarray(odom_mat, float).reshape(4, 4)[:3, 3]
                                 if odom_mat else head_pos)    # body base on the floor (walk signal)
                    wbc.update_targets(data, head_pos, odom_base,
                                       payload.get("body_yaw"), payload.get("head_height"))
                    # SAFETY: snapshot the freshly-retargeted pose as the live target +
                    # refresh per-signal freshness (only trusted when fully tracked).
                    sup.observe(payload, data, time.time())

                if out.keys() - {"timestamp"}:
                    pub.send_string(json.dumps(out))

                pm.set_status(R="right" in out, L="left" in out, neck="neck" in out,
                              base=wbc.base_anchored, base_v=wbc.base_cmd,
                              lift=float(data.ctrl[lift_act]),
                              torso=float(data.ctrl[torso_act]),
                              selfcol=(projector.min_dist if projector else None),
                              mode=("CLUTCH (hold)" if frozen else sup.state()))
            else:
                if is_active and time.time() - last_log > 2.0:
                    logger.warning(f"Waiting for VR data on {ZMQ_SUB_SRC} ...")
                    last_log = time.time()

            # -- WHOLE-BODY PHYSICS: gravity-comp position servos + swerve base,
            #    advanced by the real elapsed wall-time (capped) so it tracks
            #    realtime and chases the freshest operator frame.
            now = time.time(); wall_dt = min(now - prev, MAX_CATCHUP); acc += wall_dt; prev = now
            # SAFETY fail-safe: ramp the home<->live blend; on re-acquire re-anchor the
            # world/base so the robot eases back to the operator's CURRENT pose (no jump),
            # and freeze the base whenever tracking is lost (never drive blind).
            engaged = sup.step(now, wall_dt)
            if sup.reengaged_edge:
                world_ref = None; head0 = None
                wbc.base_anchored = False; wbc.base_tgt = (0.0, 0.0, 0.0)
            if not engaged:
                wbc.base_tgt = (0.0, 0.0, 0.0)   # stop the base on any tracking loss
            sup.apply(data)                      # blend upper body + lift: HOME <-> live
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[grav_dofs] = GRAV_GAIN * gravity()[grav_dofs]
            while acc >= dt:
                wbc.base_step(data, dt)          # LAYER 1 ramp (tip-safe) + swerve drive
                mujoco.mj_step(model, data)
                pm.record("ctrl")                # control / physics step rate
                acc -= dt
            pm.maybe_render()                    # refresh the live HUD (in-place, ~2 Hz)

            # -- overlay: the operator's EXACT tracked head + hands, mapped into the
            #    robot world (same transform as the arm targets), so the robot wrist
            #    should land on the ghost wrist.
            if SHOW_OVERLAY and payload is not None and world_ref is not None:
                if payload.get("head_mat") and head_bid != -1:
                    scn = viewer.user_scn; scn.ngeom = 0
                    overlay.tracking(scn, payload, world_ref, reach)    # VP world -> robot world (fixed)
                    rh = data.xpos[head_bid].copy()
                    overlay.gaze(scn, rh, -data.xmat[head_bid].reshape(3, 3)[:, 2])
            viewer.sync()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
exp_loop_latency.py — the COMPLETE teleop loop latency budget, measured.

Times every stage of the closed loop on the production scene (mabel_home):

  human motion
    → ARKit hand-skeleton estimation (Vision Pro, modelled: sampling theory)
    → app send loop (30 Hz, modelled)
    → network uplink (scenario parameter: LAN / Tailscale, from measured RTTs)
    → bridge ingest: JSON decode → TeleopFrame → AVP adapter        [measured]
    → arm/wrist retargeting (SE(3) clutch) + whole-body IK solve    [measured]
    → finger keypoint-IK (damped GN, 20 Hz worker)                  [measured]
    → physics: mj_step under the real-time accumulator              [measured]
    → actuator tracking: position-servo step response t90           [measured]
    → camera render + GPU readback (the macOS fixed stall)          [measured]
    → JPEG encode → UDP chunk/send → reassemble → decode            [measured]
    → display scan-out (90 Hz, modelled)

Sampling-stage model: a zero-order-hold stream at period T contributes a
uniformly distributed age U(0, T) to a randomly timed event → MEAN T/2,
WORST T. All modelled entries state this and are labelled (m); everything
else is timed on the real code (mean over N reps, p95 in the table).

Outputs:
  • per-stage table (mean / p95 ms)
  • three composed budgets × two network scenarios (LAN, Tailscale)
  • figures → server/docs/figures/loop_latency.pdf  +  arm_step_response.pdf

Run:  python3 server/tools/exp_loop_latency.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "controller"))
sys.path.insert(0, os.path.join(_ROOT, "server"))
sys.path.insert(0, os.path.join(_ROOT, "vision_pro", "docs", "tools"))

import cv2
import mujoco
from mabel.teleop_controller import TeleopController
from mabel.hand_ik_retarget import HandIKRetargeter, human_keypoints
from teleopframe_adapter import to_avp_frame
from wire_protocol import WireCodec
from sim_cameras import camera_plan, HEAD_W, HEAD_H
from exp_hand_retarget import synth_hand            # vision_pro/docs/tools

FIGS = os.path.normpath(os.path.join(_HERE, "..", "docs", "figures"))
N = 300
SEED = 7

IDENT = [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# Network scenarios: one-way delay = RTT/2. LAN RTT ≈ 2 ms (same Wi-Fi,
# measured by the e2e harness); Tailscale RTT = 41 ms measured between the
# operator's Mac and Vision Pro on the same campus network (direct WireGuard
# path, `tailscale ping`, 2026-06-12).
SCENARIOS = {"LAN": 1.0, "Tailscale": 20.5}

APP_SEND_HZ = 30.0          # WebSocketLink.pushFrame
ARKIT_HZ = 90.0             # visionOS hand-anchor update rate
TELEMETRY_HZ = 20.0         # bridge _send_state
FINGER_HZ = 20.0            # hand IK worker SOLVE_PERIOD
DISPLAY_HZ = 90.0           # headset refresh
BRIDGE_LOOP_HZ = 30.0       # bridge outer loop (newest-frame pickup)


def wire_json(seq, pose):
    joints = [{"joint": n, "localTransform": {"matrix": np.asarray(T).reshape(16).tolist()},
               "isTracked": True} for n, T in pose.items()]
    payload = {"sequence": seq, "timestamp": time.time(),
               "head": {"transform": {"matrix": IDENT}, "trackingState": "normal"},
               "rightHand": {"chirality": "right", "anchorTransform": {"matrix": IDENT},
                             "joints": joints, "isTracked": True},
               "leftHand": {"chirality": "left", "anchorTransform": {"matrix": IDENT},
                            "joints": joints, "isTracked": True},
               "mode": "Arms & Hands",
               "navJoystick": {"lx": 0, "ly": 0, "rx": 0, "ry": 0}}
    return json.dumps({"type": "teleop_frame", "payload": payload})


def bench(fn, n=N):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    a = 1e3 * np.asarray(ts)
    return float(a.mean()), float(np.percentile(a, 95))


def main():
    np.random.seed(SEED)
    os.makedirs(FIGS, exist_ok=True)
    scene = os.path.join(_ROOT, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    model.opt.timestep = 0.004
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    ctl = TeleopController(model, data)
    ctl.set_mode(TeleopController.MODE_MANIP)

    raw = wire_json(1, synth_hand(mcp=0.7, pip=0.8, thumb=0.5))
    print(f"scene: mabel_home.xml   wire frame: {len(raw) / 1024:.1f} kB (two full hands)")

    M = {}          # name → (mean, p95)

    def add(name, fn, n=N):
        M[name] = bench(fn, n)
        print(f"  {name:42s} {M[name][0]:8.3f} ms  (p95 {M[name][1]:.3f})")

    # ════ 1. uplink compute (bridge, per fresh frame) ═══════════════════════
    print("\n[1] bridge ingest + retargeting (per fresh 30 Hz frame)")
    add("json decode + envelope", lambda: WireCodec.decode_envelope(raw))
    _, payload = WireCodec.decode_envelope(raw)
    add("decode_teleop_frame (2 hands)", lambda: WireCodec.decode_teleop_frame(payload))
    tf = WireCodec.decode_teleop_frame(payload)
    add("to_avp_frame adapter", lambda: to_avp_frame(tf))
    frame = to_avp_frame(tf)
    ctl.calibrate(frame)
    add("wrist retarget (SE3 clutch update)", lambda: ctl.retgt.update(frame))

    # ════ 2. per physics substep: IK + step ═════════════════════════════════
    print("\n[2] whole-body IK + physics (per 4 ms substep)")
    add("controller.step (whole-body IK)", lambda: ctl.step(frame, None, 0.004))
    add("mj_step (physics)", lambda: mujoco.mj_step(model, data))
    add("finger stream apply (cached)",
        lambda: ctl.hand_retgt.apply("Right", frame.right_joints))

    # ════ 3. finger GN solve (background worker @ 20 Hz/hand) ═══════════════
    print("\n[3] finger keypoint-IK (worker thread)")
    rt = ctl.hand_retgt
    hand = rt.hands["Right"]
    h_raw = human_keypoints(frame.right_joints, "right")
    h_shape = rt._calibrate(h_raw, hand["lengths"], hand["kn0"])
    add("GN finger solve (4 it, 16 DOF)", lambda: rt._solve_gn(hand, h_shape, h_raw), n=100)

    # ════ 4. arm tracking dynamics: actuator step response ══════════════════
    # The IK writes position-servo targets; the robot then has to physically
    # get there. Step a representative arm actuator and measure t90 of the
    # joint — this is CONTROL BANDWIDTH, present on sim and real alike.
    print("\n[4] actuator step response (position-servo tracking)")
    t90 = {}
    trace = {}

    def hold_pose():
        """Set every actuator's target to its joint's current position so the
        robot HOLDS while one joint is stepped (otherwise the unpowered arm
        sags under gravity and the response never converges)."""
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            if 0 <= j < model.njnt and model.jnt_type[j] >= 2:
                data.ctrl[a] = data.qpos[model.jnt_qposadr[j]]
        for _ in range(100):                    # settle under hold (0.4 s)
            mujoco.mj_step(model, data)

    for act_name, step_rad in [("Left Arm 2", 0.3), ("Left Wrist", 0.3)]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        jid = model.actuator_trnid[aid, 0]
        qadr = model.jnt_qposadr[jid]
        hold_pose()
        q0 = data.qpos[qadr]
        data.ctrl[aid] = data.ctrl[aid] + step_rad
        ts, qs = [], []
        t_start = data.time
        for k in range(int(2.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            ts.append(data.time - t_start)
            qs.append(data.qpos[qadr])
        qs = np.asarray(qs) - q0
        ts = np.asarray(ts)
        idx = np.argmax(qs >= 0.9 * step_rad) if (qs >= 0.9 * step_rad).any() else -1
        t90[act_name] = float(ts[idx]) * 1e3 if idx >= 0 else float("nan")
        # First-order time constant from the 63.2% crossing — the phase lag a
        # CONTINUOUSLY moving IK target sees is ≈ τ, the teleop-relevant number.
        i63 = np.argmax(qs >= 0.632 * step_rad) if (qs >= 0.632 * step_rad).any() else -1
        tau = float(ts[i63]) * 1e3 if i63 >= 0 else float("nan")
        t90[act_name + " τ"] = tau
        trace[act_name] = (ts, qs / step_rad)
        print(f"  {act_name:42s} t90 = {t90[act_name]:6.1f} ms   τ ≈ {tau:6.1f} ms  (0.3 rad step)")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ════ 5. return path: render → encode → transport → decode ══════════════
    print("\n[5] video return path")
    renderer = mujoco.Renderer(model, height=HEAD_H, width=HEAD_W)
    cam = "zed_left" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "zed_left") >= 0 else None

    def render_once():
        renderer.update_scene(data, camera=cam)
        return renderer.render()
    add(f"render+readback ({HEAD_W}x{HEAD_H})", render_once, n=60)
    rgb = render_once()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    add("JPEG encode q60", lambda: cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60]), n=200)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    jpeg = buf.tobytes()
    print(f"  {'(frame size q60)':42s} {len(jpeg)/1024:8.1f} kB")
    add("JPEG decode (cv2 proxy for ImageIO)",
        lambda: cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR), n=200)

    # UDP transport: chunk+send+reassemble on loopback (real protocol stack).
    from udp_video import UDPVideoSender
    from exp_udp_video import RefReceiver
    sender = UDPVideoSender(port=9491, host="127.0.0.1")
    sender.start()
    rx = RefReceiver(("127.0.0.1", 9491), ["main"])
    rx.subscribe(); time.sleep(0.3)
    lat_udp = []
    import hashlib
    for i in range(150):
        f = jpeg + bytes([i & 0xFF])            # unique per frame
        h = hashlib.sha1(f).hexdigest()
        t0 = time.monotonic()
        sender.send("main", f)
        for _ in range(2000):
            if h in rx.received:
                lat_udp.append((rx.received[h] - t0) * 1e3); break
            time.sleep(0.0002)
        time.sleep(0.004)
    rx.stop(); sender.close()
    M["UDP chunk+send+reassemble"] = (float(np.mean(lat_udp)), float(np.percentile(lat_udp, 95)))
    print(f"  {'UDP chunk+send+reassemble':42s} {np.mean(lat_udp):8.3f} ms  (p95 {np.percentile(lat_udp, 95):.3f})")

    # Head camera FRAME AGE from the render budget: the weighted scheduler
    # gives the head w of W total slots; slots/s = 1/render_cost.
    weights, _, _ = camera_plan(stereo=True)
    W = sum(weights.values())
    head_share = weights.get("zed_left", 8) / W
    render_ms = M[f"render+readback ({HEAD_W}x{HEAD_H})"][0]
    slots_per_s = 1000.0 / render_ms
    head_fps = head_share * slots_per_s
    head_age = 1000.0 / (2 * head_fps)
    print(f"  camera scheduler: {slots_per_s:.1f} slots/s, head share {head_share:.2f} "
          f"→ {head_fps:.1f} fps → mean frame age {head_age:.1f} ms")

    # ════ 6. composed budgets ════════════════════════════════════════════════
    half = lambda hz: 1000.0 / (2 * hz)
    ingest = sum(M[k][0] for k in ["json decode + envelope", "decode_teleop_frame (2 hands)",
                                   "to_avp_frame adapter", "wrist retarget (SE3 clutch update)"])
    substep = sum(M[k][0] for k in ["controller.step (whole-body IK)", "mj_step (physics)",
                                    "finger stream apply (cached)"])

    def budgets(net_ms):
        cmd_arm = [("ARKit hand estimate (m, 90 Hz ZOH)", half(ARKIT_HZ)),
                   ("app send loop (m, 30 Hz ZOH)", half(APP_SEND_HZ)),
                   ("uplink network (m)", net_ms),
                   ("bridge pickup (m, 30 Hz ZOH)", half(BRIDGE_LOOP_HZ)),
                   ("ingest + wrist retarget", ingest),
                   ("whole-body IK + physics substep", substep)]
        cmd_fin = cmd_arm + [("finger worker (m, 20 Hz ZOH)", half(FINGER_HZ)),
                             ("GN finger solve", M["GN finger solve (4 it, 16 DOF)"][0])]
        vis = [("head camera frame age (scheduler)", head_age),
               ("render + GPU readback", render_ms),
               ("JPEG encode q60", M["JPEG encode q60"][0]),
               ("UDP transport", M["UDP chunk+send+reassemble"][0]),
               ("downlink network (m)", net_ms),
               ("JPEG decode (proxy)", M["JPEG decode (cv2 proxy for ImageIO)"][0]),
               ("display scan-out (m, 90 Hz)", half(DISPLAY_HZ))]
        return cmd_arm, cmd_fin, vis

    results = {}
    print("\n=== composed budgets (mean, ms) ===")
    for net, ow in SCENARIOS.items():
        cmd_arm, cmd_fin, vis = budgets(ow)
        a, f, v = sum(x for _, x in cmd_arm), sum(x for _, x in cmd_fin), sum(x for _, x in vis)
        results[net] = (cmd_arm, cmd_fin, vis)
        print(f"\n  [{net}]  one-way network = {ow:.1f} ms")
        print(f"    command path, ARM/WRIST  → actuator target : {a:6.1f} ms  (+ servo tracking τ ≈ {t90.get('Left Arm 2 τ', float('nan')):.0f} ms)")
        print(f"    command path, FINGERS    → actuator target : {f:6.1f} ms")
        print(f"    visual return, sim state → photons         : {v:6.1f} ms")
        print(f"    GLASS-TO-GLASS reaction (arm cmd + visual) : {a + v:6.1f} ms")

    # ════ figures ════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.6), sharex=True)
    for ax, net in zip(axes, SCENARIOS):
        cmd_arm, _, vis = results[net]
        for row, (label, parts) in enumerate([("command (arm)", cmd_arm), ("visual return", vis)]):
            leftv = 0.0
            colors = plt.cm.copper(np.linspace(0.2, 0.9, len(parts)))
            for (k, vms), c in zip(parts, colors):
                ax.barh([row], [vms], left=leftv, color=c,
                        label=k if net == "LAN" and row == (0 if "cmd" in label else 1) else None)
                leftv += vms
        ax.set_yticks([0, 1]); ax.set_yticklabels(["command\n(arm)", "visual\nreturn"], fontsize=8)
        tot_a = sum(v for _, v in cmd_arm); tot_v = sum(v for _, v in vis)
        ax.set_title(f"{net}: command {tot_a:.0f} ms · visual {tot_v:.0f} ms · "
                     f"glass-to-glass {tot_a + tot_v:.0f} ms", fontsize=9)
        ax.grid(alpha=0.25, axis="x")
    axes[1].set_xlabel("ms")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, fontsize=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.16), ncols=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "loop_latency.pdf"), bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(5.2, 2.8))
    for name, (ts, qn) in trace.items():
        ax2.plot(ts * 1e3, qn, label=f"{name} (τ ≈ {t90[name + ' τ']:.0f} ms)")
    ax2.axhline(0.9, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("time [ms]"); ax2.set_ylabel("normalized response")
    ax2.set_xlim(0, 2000); ax2.set_title("Position-servo step response (0.3 rad)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGS, "arm_step_response.pdf"))
    print(f"\nfigures → {FIGS}/loop_latency.pdf, arm_step_response.pdf")


if __name__ == "__main__":
    main()

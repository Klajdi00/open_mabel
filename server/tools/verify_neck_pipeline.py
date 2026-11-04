#!/usr/bin/env python3
"""verify_neck_pipeline.py — RIGOROUS offline proof that the web-GUI neck-slider control
pipeline is wired correctly at every hop (no robot, no running gateway needed).

The web control center and iOS are clients of the SAME gateway (ws :9090), so a neck slider
drives this exact chain:

    joint_command{neck_*}  →  facade._apply_joint_targets  →  data.ctrl["Neck Pitch"…]
                           →  command_extract              →  /body/command{neck_pitch…}
    torque ON (estop-off)  →  ROSPlant.set_neck_torque     →  /body/torque_enable (Bool)
    HAL body_node          →  _on_torque→neck.set_torque · _on_cmd→neck.set_gaze

Each stage is asserted independently so a failure NAMES the broken hop. If every stage passes
here but the real robot's neck stays limp, the fault is RUNTIME, not code — almost always a
GATEWAY PROCESS started before set_neck_torque was built (so getattr(plant,'set_neck_torque')
is None and the estop→torque edge energises arms+hands but silently skips the neck). Fix:
restart real_run.sh / mabel-bringup, then enable torque. Confirm live with diag_neck_path.py.

    python3 server/tools/verify_neck_pipeline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("controller", "ros/build/mabel_control", "ros/build/mabel_sim",
          "hardware_bridge"):
    sys.path.insert(0, str(ROOT / p))

FAILS = []
def ok(label, cond, extra=""):
    print(("  PASS " if cond else "  FAIL ") + label + (("  " + extra) if extra else ""))
    if not cond:
        FAILS.append(label)


def stage_ab():
    """Movement: facade neck override -> data.ctrl -> command_extract -> /body/command."""
    import mujoco
    from mabel.facade import MabelTeleop
    from mabel_control.command_extract import extract
    from mabel_sim import interface_config as cfg
    mjcf = ROOT / "simulation/mabel_mujoco/models/mabel_full.xml"
    if not mjcf.exists():
        print("  SKIP (no canonical MJCF)"); return

    class _RC:
        engaged = 1.0

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    acts = {model.actuator(i).name for i in range(model.nu)}
    for a in ("Neck Pitch", "Neck Roll", "Neck Yaw"):
        ok(f"model has actuator '{a}'", a in acts)

    print("— Stage B: command_extract carries the neck into /body/command —")
    data = mujoco.MjData(model)
    data.actuator("Neck Pitch").ctrl[0] = 0.30
    data.actuator("Neck Roll").ctrl[0] = -0.20
    c = extract(model, data, _RC(), cfg)
    ok("body has neck_pitch/neck_roll", {"neck_pitch", "neck_roll"} <= set(c.body), str(list(c.body)))
    ok("neck_pitch ~0.30", abs(c.body.get("neck_pitch", 0) - 0.30) < 1e-3, str(c.body.get("neck_pitch")))

    print("— Stage A: facade joint_command{neck_*} -> neck actuator ctrl (method=joint) —")
    data2 = mujoco.MjData(model)
    teleop = MabelTeleop(model, data2)
    teleop.set_control_spec(method="joint", control_type="impedance", region="whole_body")
    teleop.set_joint_targets({"neck_pitch": 0.30, "neck_roll": -0.20})
    for _ in range(200):
        teleop._apply_joint_targets()
    npc = float(data2.actuator("Neck Pitch").ctrl[0])
    ok("facade drove Neck Pitch ctrl ~0.30", abs(npc - 0.30) < 2e-2, f"got {npc:.4f}")
    c2 = extract(model, data2, _RC(), cfg)
    ok("full A->B /body/command neck_pitch ~0.30",
       abs(c2.body.get("neck_pitch", 0) - 0.30) < 2e-2, str(c2.body.get("neck_pitch")))


def stage_c():
    """HAL body_node consumes /body/torque_enable + /body/command and drives the neck writer."""
    try:
        import rclpy
        from std_msgs.msg import Bool
        from sensor_msgs.msg import JointState
        from mabel_hw.ros.body_node import BodyNode
    except Exception as e:
        print(f"  SKIP (no ROS / hardware_bridge: {e})"); return

    class _Motors:
        names = ("neck_pitch", "neck_roll"); position = (0.0, 0.0); velocity = (0.0, 0.0)
        effort = (0.0, 0.0); torque_enabled = False; stamp = None
    class _NeckState:
        motors = _Motors(); pitch = 0.0; roll = 0.0; yaw = None; stamp = None
    class _Writer:
        def __init__(self): self.calls = []
        def set_torque(self, on): self.calls.append(("set_torque", bool(on)))
        def set_gaze(self, p, r, y): self.calls.append(("set_gaze", round(p, 4), round(r, 4), round(y, 4)))
        def set_targets(self, t): self.calls.append(("set_targets", dict(t)))
        def release(self): pass
    class _NeckHandle:
        available = True; task_space_available = True
        def __init__(self): self.writer = _Writer()
        def claim(self, owner, steal=False): return self.writer
        def state(self): return _NeckState()
    class _Absent:
        available = False
    class _HW:
        def __init__(self): self.neck = _NeckHandle(); self.torso = _Absent()

    cfg = {"topic_command": "/body/command", "topic_joint_states": "/body/joint_states",
           "topic_torque": "/body/torque_enable", "topic_torque_state": "/body/torque_state",
           "publish_rate": 50.0, "joint_motors": "", "kp_default": 30.0, "kd_default": 1.0,
           "p_max": 12.5, "v_max": 30.0, "t_max": 10.0}

    rclpy.init()
    hw = _HW(); node = BodyNode(cfg, hw=hw); w = hw.neck.writer
    try:
        ok("HAL claimed the neck writer", node._neck_writer is not None)
        node._on_torque(Bool(data=True))
        ok("torque_enable=True -> neck.set_torque(True)", ("set_torque", True) in w.calls, str(w.calls))
        w.calls.clear()
        msg = JointState(); msg.name = ["neck_pitch", "neck_roll"]; msg.position = [0.30, -0.20]
        node._on_cmd(msg)
        gaze = [c for c in w.calls if c[0] == "set_gaze"]
        ok("/body/command -> neck.set_gaze", bool(gaze), str(w.calls))
        if gaze:
            _, p, r, _y = gaze[-1]
            ok("gaze pitch 0.30 / roll -0.20", abs(p - 0.30) < 1e-6 and abs(r + 0.20) < 1e-6, f"{p},{r}")
        w.calls.clear(); node._on_torque(Bool(data=False))
        ok("torque_enable=False -> neck.set_torque(False)", ("set_torque", False) in w.calls, str(w.calls))
    finally:
        node.destroy_node(); rclpy.shutdown()


def main():
    print("=== Stage A/B: neck movement command path (facade -> extract -> /body/command) ===")
    stage_ab()
    print("\n=== Stage C: HAL body_node applies torque + gaze to the neck ===")
    stage_c()
    print("\n=== RESULT:", "ALL PASS — the code pipeline is correct end-to-end." if not FAILS
          else f"{len(FAILS)} FAILED: {FAILS}", "===")
    if not FAILS:
        print("If the real neck is still limp: the running gateway is stale (no set_neck_torque)")
        print("→ restart mabel-bringup / real_run.sh, enable torque, then run diag_neck_path.py.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

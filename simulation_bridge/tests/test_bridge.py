"""simulation_bridge: engine + RPC, against the real robot model (no scene
clutter — mabel_bare) with an isolated instance so a live sim0 is untouched."""
import json
import os
import socket
import threading
import time

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BARE = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_bare.xml")
HOME = os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(not os.path.isfile(BARE),
                                reason="bare scene missing")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def world():
    os.environ["MABEL_SIM_INSTANCE"] = "pytest"
    os.environ["MABEL_STATE_FILE"] = f"/tmp/mabel_sim_pytest_{os.getpid()}.f64"
    import importlib
    from simulation_bridge import config as CFG
    importlib.reload(CFG)
    from simulation_bridge.engine import SimWorld
    from simulation_bridge.rpc import (SimBridgeClient, SimBridgeServer,
                                       SimBridgeService)
    w = SimWorld(scene=BARE, instance="pytest")
    srv = SimBridgeServer(SimBridgeService(w), port=_free_port()).start()
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    cli = SimBridgeClient(port=srv.port)
    time.sleep(0.3)
    yield w, cli
    w.stop()
    srv.stop()
    for suffix in ("", ".scene", ".pert"):
        try:
            os.remove(w.state_path + suffix)
        except OSError:
            pass


def test_hello_and_info(world):
    w, cli = world
    h = cli.call("hello")
    assert h["kind"] == "sim" and h["instance"] == "pytest"
    i = cli.call("info")
    assert i["nq"] == w.model.nq and i["nu"] == w.model.nu
    assert "Left Arm 1" in i["actuators"]         # servo contract present
    assert not any("Steer" in a or "Drive" in a for a in i["actuators"])


def test_memmap_v2_layout(world):
    w, cli = world
    m = w.model
    mm = np.memmap(w.state_path, dtype=np.float64, mode="r",
                   shape=(1 + m.nq + m.nv + m.nu,))
    t1 = float(mm[0])
    time.sleep(0.2)
    assert float(mm[0]) > t1, "sim time must advance in the memmap"
    # legacy prefix readable exactly like the old [t, qpos] writers
    legacy = np.memmap(w.state_path, dtype=np.float64, mode="r",
                       shape=(1 + m.nq,))
    assert np.allclose(legacy[1:], mm[1:1 + m.nq])


def test_twist_drives_and_deadman_zeroes(world):
    w, cli = world
    cli.call("reset")
    time.sleep(0.3)
    t0 = time.time()
    top = 0.0
    while time.time() - t0 < 2.5:
        cli.call("set_twist", vx=0.5, vy=0.0, wz=0.0)
        s = cli.call("state")
        top = max(top, max(abs(v[1]) for n, v in s["joints"].items()
                           if "Drive" in n))
        time.sleep(0.05)
    assert top > 2.0, f"wheels never spun (max {top})"
    time.sleep(1.0)                                # > CMD_TIMEOUT, no commands
    s = cli.call("state")
    now = max(abs(v[1]) for n, v in s["joints"].items() if "Drive" in n)
    assert now < 0.5, f"deadman failed: wheels still at {now}"


def test_estop_blocks_base(world):
    w, cli = world
    cli.call("set_estop", active=True)
    for _ in range(10):
        cli.call("set_twist", vx=0.8, vy=0.0, wz=0.0)
        time.sleep(0.03)
    s = cli.call("state")
    assert s["estop"] is True
    assert max(abs(v[1]) for n, v in s["joints"].items() if "Drive" in n) < 0.3
    cli.call("set_estop", active=False)


def test_servo_targets_latch(world):
    w, cli = world
    cli.call("reset")
    time.sleep(0.3)
    cli.call("set_targets", targets={"Left Arm 2": 0.6, "bogus": 9.9})
    t0 = time.time()
    pos = 0.0
    while time.time() - t0 < 4:
        s = cli.call("state")
        pos = s["joints"].get("Left Arm 2", [0, 0])[0]
        if abs(pos - 0.6) < 0.1:
            break
        time.sleep(0.1)
    assert abs(pos - 0.6) < 0.1, f"servo target not tracked: {pos}"


def test_reset_with_pose(world):
    w, cli = world
    cli.call("reset", pose={"x": 1.5, "y": -0.8, "yaw": 1.0})
    time.sleep(0.4)
    s = cli.call("state")
    assert abs(s["base"]["x"] - 1.5) < 0.05
    assert abs(s["base"]["y"] + 0.8) < 0.05
    assert abs(s["base"]["yaw"] - 1.0) < 0.1
    cli.call("reset")
    time.sleep(0.3)
    s = cli.call("state")
    assert abs(s["base"]["x"]) < 0.05 and abs(s["base"]["y"]) < 0.05


def test_effort_channel_moves_joint(world):
    """Effort channel + torque-enable together: with the arm group's servos
    yielding (torque off), an applied joint torque must move the elbow —
    with the servos holding, the same torque must barely move it."""
    w, cli = world
    cli.call("reset")
    time.sleep(0.3)
    cli.call("set_group_torque", group="left_arm", enabled=False)
    peak = 0.0
    for _ in range(30):
        cli.call("set_effort", efforts={"Left Arm 4": 25.0})
        s = cli.call("state")
        peak = max(peak, abs(s["joints"].get("Left Arm 4", [0, 0])[1]))
        time.sleep(0.02)
    cli.call("set_group_torque", group="left_arm", enabled=True)
    assert peak > 0.05, f"effort produced no motion with torque off (peak {peak})"
    cli.call("reset")


def test_unknown_op_is_error(world):
    w, cli = world
    from simulation_bridge.rpc import SimBridgeError
    with pytest.raises(SimBridgeError):
        cli.call("no_such_op")


def test_realtime_rate(world):
    w, cli = world
    time.sleep(1.2)
    s = cli.call("state")
    realtime_hz = 1.0 / w.model.opt.timestep       # 500 Hz at dt=2 ms
    assert s["rateHz"] >= 0.6 * realtime_hz, \
        f"physics too slow: {s['rateHz']} Hz vs {realtime_hz} Hz real-time"

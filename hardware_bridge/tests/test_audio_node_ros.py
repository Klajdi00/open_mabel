"""The audio DDS adapter against a real rclpy graph, with the sound card faked.

Marked `ros`: auto-skipped where rclpy is absent (the Mac), runs for real on
thor. Complements test_ros_nodes_static.py (which only freezes the module's
SHAPE) by exercising the parts that can silently rot without a live graph: the
latched mute state a late dashboard depends on, demand-driven capture keyed on
DDS subscriber count, and the drop-newest queue that keeps speech in order.

The CARD is faked (MockAudio) so this needs no hardware; experiments/exp_audio.py
is the real-device counterpart.
"""
import json
import threading
import time

import pytest

pytest.importorskip("rclpy")
pytestmark = pytest.mark.ros

import rclpy                                                        # noqa: E402
from rclpy.executors import MultiThreadedExecutor                   # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import (DurabilityPolicy, QoSProfile,                # noqa: E402
                       qos_profile_sensor_data)
from std_msgs.msg import Bool, String, UInt8MultiArray              # noqa: E402

from mabel_hw import config                                         # noqa: E402
from mabel_hw.mock import MockAudio                                 # noqa: E402
from mabel_hw.ros.audio_node import AudioNode                       # noqa: E402

FRAME = 640


class _Hw:
    """Minimal stand-in for the facade: the node only reads `_audio` off it."""

    def __init__(self, audio):
        self._audio = audio


class Probe(Node):
    def __init__(self, cfg):
        super().__init__("audio_probe")
        self.mic, self.health, self.state = [], [], []
        self.create_subscription(UInt8MultiArray, cfg["topic_mic"],
                                 lambda m: self.mic.append(bytes(m.data)),
                                 qos_profile_sensor_data)
        self.create_subscription(String, cfg["topic_health"],
                                 lambda m: self.health.append(json.loads(m.data)), 5)
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, cfg["topic_state"],
                                 lambda m: self.state.append(bool(m.data)), latched)
        self.spk = self.create_publisher(UInt8MultiArray, cfg["topic_speaker"], 10)
        self.en = self.create_publisher(Bool, cfg["topic_enable"], 10)


@pytest.fixture()
def graph():
    """A live node + probe on their own context, spinning in a background thread."""
    cfg = dict(config.AUDIO)
    if not rclpy.ok():
        rclpy.init()
    core = MockAudio()
    node = AudioNode(cfg, hw=_Hw(core))
    probe = Probe(cfg)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    ex.add_node(probe)
    t = threading.Thread(target=ex.spin, daemon=True)
    t.start()
    time.sleep(0.5)
    yield node, probe, core, ex
    try:
        ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        probe.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.fixture()
def lone_node():
    """The node WITHOUT an executor: nothing spins, so its own timers never race
    a test that wants to inspect internal state directly."""
    if not rclpy.ok():
        rclpy.init()
    node = AudioNode(dict(config.AUDIO), hw=_Hw(MockAudio()))
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _until(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_topics_are_advertised_with_the_contract_types(graph):
    node, _, _, _ = graph
    topics = dict(node.get_topic_names_and_types())
    for tp, ty in (("/audio/mic", "std_msgs/msg/UInt8MultiArray"),
                   ("/audio/speaker", "std_msgs/msg/UInt8MultiArray"),
                   ("/audio/enable", "std_msgs/msg/Bool"),
                   ("/audio/state", "std_msgs/msg/Bool"),
                   ("/audio/device_health", "std_msgs/msg/String")):
        assert tp in topics, f"{tp} not advertised"
        assert ty in topics[tp], f"{tp} is {topics[tp]}, want {ty}"


def test_mute_state_reaches_a_LATE_subscriber(graph):
    """TRANSIENT_LOCAL or a dashboard that starts later shows mute as unknown
    forever (the torque_state scar, mabel_control/ros_plant.py)."""
    _, probe, _, _ = graph
    assert _until(lambda: len(probe.state) > 0), "latched state never delivered"


def test_mic_pcm_flows_to_dds(graph):
    node, probe, core, _ = graph
    assert _until(lambda: node._unsub is not None), "node never subscribed to the core"
    for _ in range(30):
        core.spin()
        time.sleep(0.005)
    assert _until(lambda: len(probe.mic) >= 10), f"only {len(probe.mic)} frames"
    assert len(probe.mic[0]) == FRAME


def test_speaker_pcm_flows_from_dds(graph):
    _, probe, core, _ = graph
    pcm = bytes(range(256)) * 2
    m = UInt8MultiArray()
    m.data = list(pcm)
    probe.spk.publish(m)
    assert _until(lambda: pcm in core.played), "speaker frame never reached the card"


def test_enable_false_mutes_and_echoes_on_the_latched_topic(graph):
    _, probe, core, _ = graph
    probe.en.publish(Bool(data=False))
    assert _until(lambda: core._muted is True), "enable=false did not mute"
    assert _until(lambda: probe.state and probe.state[-1] is False)
    probe.en.publish(Bool(data=True))
    assert _until(lambda: core._muted is False), "enable=true did not unmute"


def test_capture_follows_dds_demand(graph):
    """An idle graph must release the sound card — a calibration studio or
    speaker-test has to be able to open it."""
    node, probe, _, ex = graph
    assert _until(lambda: node._unsub is not None), "no capture with a subscriber"
    ex.remove_node(probe)
    probe.destroy_node()
    assert _until(lambda: node._unsub is None, timeout=6.0), \
        "card still held with no DDS subscriber"


def test_queue_overflow_drops_newest_and_never_reorders(lone_node):
    """Latest-wins would reorder speech. On overflow the node drops the NEWEST
    frame, so what does get through stays in sequence.

    Uses the NON-spinning node on purpose: with an executor running, the node's
    own _drain timer races this test for the queue and the assertions measure the
    race, not the policy (it read seq=8 at index 0 before this was split out).
    """
    node = lone_node
    for i in range(node._q.maxsize + 10):
        node._enqueue(type("C", (), {"pcm": bytes([i % 256]) * FRAME, "seq": i})())
    assert node._q.qsize() == node._q.maxsize
    got = [node._q.get_nowait().seq for _ in range(node._q.maxsize)]
    assert got == sorted(got), "frames left the queue out of order"
    assert got == list(range(node._q.maxsize)), "dropped the OLDEST, not the newest"
    assert node._dropped == 10


def test_health_json_shape(graph):
    _, probe, _, _ = graph
    assert _until(lambda: len(probe.health) > 0), "no device_health published"
    h = probe.health[-1]
    assert {"speaker", "mic", "connected", "card"} <= set(h), h
    assert {"open", "volumePct", "queuedFrames"} <= set(h["speaker"])
    assert {"open", "volumePct", "muted", "rms", "frames"} <= set(h["mic"])

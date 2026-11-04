"""audio_node — USB speaker + mic DDS adapter.

The sound card (device resolution, the ALSA pipes, the demand-driven capture,
the speaker FIFO) lives in :class:`mabel_hw.subsystems.audio.AudioSubsystem`;
this node only moves bytes between it and DDS. Contract:

  Subscribed:
    <topic_speaker>  std_msgs/UInt8MultiArray  raw S16_LE PCM to play, one frame
                                               per message (20 ms @ 16 kHz mono)
    <topic_enable>   std_msgs/Bool             mic mute: false = muted
  Published:
    <topic_mic>      std_msgs/UInt8MultiArray  raw S16_LE PCM from the mic,
                                               sensor QoS (a dropped frame is a
                                               click, never worth a resend)
    <topic_state>    std_msgs/Bool             LATCHED mic-enabled state, so a
                                               dashboard that starts later sees
                                               the current mode without waiting
    <topic_health>   std_msgs/String (JSON)    {speaker:{...}, mic:{...}} @2 Hz

WHY UInt8MultiArray AND NOT audio_common_msgs/AudioData: audio_common is not
installed on thor and pulling an apt package in to carry a byte array would buy
nothing — the wire is raw S16_LE either way, and the format is fixed by
config.AUDIO (rate/channels), not per-message. `layout` is left empty for the
same reason: a frame is a flat byte run, and the apps/:9610 plane parse it with
the same struct the sound card uses.

MIC CAPTURE IS DEMAND-DRIVEN, so this node only subscribes to the subsystem
while something is actually subscribed to <topic_mic> — an idle graph must not
hold the card open (a calibration studio or speaker-test needs to grab it).
"""

import array
import json
import queue

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool, String, UInt8MultiArray

from ._util import add_publisher_watchdog
from ..subsystems.audio import AudioSubsystem as CoreAudio


def build(cfg, hw=None, **_):
    return AudioNode(cfg, hw=hw)


class AudioNode(Node):
    def __init__(self, cfg, hw=None):
        super().__init__("mabel_hw_audio")
        self.cfg = cfg

        if hw is not None and hw._audio is not None:
            self.core = hw._audio
            self._owns_core = False
        else:
            self.core = CoreAudio(cfg,
                                  logger=lambda m: self.get_logger().warn(m))
            self._owns_core = True

        self.pub_mic = self.create_publisher(UInt8MultiArray, cfg["topic_mic"],
                                             qos_profile_sensor_data)
        # Latched, like every other torque_state-shaped signal in the fleet: a
        # VOLATILE subscriber never receives a TRANSIENT_LOCAL publisher's last
        # value, so a late dashboard would show the mute mode as unknown forever.
        latched = QoSProfile(depth=1)
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.pub_state = self.create_publisher(Bool, cfg["topic_state"], latched)
        self.pub_health = self.create_publisher(String, cfg["topic_health"], 5)

        self.create_subscription(UInt8MultiArray, cfg["topic_speaker"],
                                 self._on_speaker, 10)
        self.create_subscription(Bool, cfg["topic_enable"], self._on_enable, 10)
        add_publisher_watchdog(self, cfg["topic_speaker"])

        # Capture pushes frames from ITS thread; publish from the executor thread
        # via a queue (rclpy publishers are not guaranteed thread-safe across
        # contexts). Deeper than the lidar's 1-deep latest-wins slot ON PURPOSE:
        # audio is a SEQUENCE, so coalescing to the newest frame would drop
        # speech. 25 frames = 0.5 s of slack; past that the card is outrunning
        # the executor and dropping is the only honest option.
        self._q = queue.Queue(maxsize=25)
        self._dropped = 0
        self._unsub = None
        # The drain timer only exists WHILE the mic is subscribed. It used to run
        # unconditionally at 200 Hz, waking this node 200x/s to poll a queue that
        # is always empty when nobody is listening — pure scheduling pressure in
        # the process that publishes the joint states the apps render. It is
        # created in _follow_demand() with the subscription and destroyed with it.
        self._drain_timer = None
        self.create_timer(0.5, self._publish_health)
        self.create_timer(1.0, self._follow_demand)

        self._publish_state()
        self.get_logger().info(
            f"audio online: card '{cfg['card']}' -> {cfg['topic_mic']} "
            f"({cfg['rate']} Hz x{cfg['channels']} S16_LE, "
            f"{cfg['frame_ms']} ms frames); speaker on {cfg['topic_speaker']}.")

    # ---- command intake --------------------------------------------------------
    def _on_speaker(self, msg: UInt8MultiArray):
        self.core.play(bytes(msg.data))

    def _on_enable(self, msg: Bool):
        self.core.set_muted(not bool(msg.data))
        self._publish_state()

    # ---- mic: only hold the card while someone is listening --------------------
    def _follow_demand(self):
        """Mirror DDS demand onto the subsystem's subscriber count, so an idle
        graph releases the sound card instead of pinning it open."""
        try:
            wanted = self.count_subscribers(self.cfg["topic_mic"]) > 0
        except Exception:                                 # noqa: BLE001
            wanted = True                                 # can't tell: keep it up
        if wanted and self._unsub is None:
            self._unsub = self.core.on_chunk(self._enqueue)
            # 5 ms: one wire frame is 20 ms, so this drains well inside a period
            # without the queue ever backing up. Only while someone listens.
            self._drain_timer = self.create_timer(0.005, self._drain)
        elif not wanted and self._unsub is not None:
            self._unsub()
            self._unsub = None
            if self._drain_timer is not None:
                self.destroy_timer(self._drain_timer)
                self._drain_timer = None
            while not self._q.empty():          # stale speech is never replayed
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break

    def _enqueue(self, chunk):
        try:
            self._q.put_nowait(chunk)
        except queue.Full:
            # Never shuffle out the oldest to make room (the latest-wins idiom
            # used for poses/clouds): that reorders speech. Drop the newest and
            # say so — a gap the consumer can see via `seq` beats a scramble.
            self._dropped += 1
            if self._dropped % 50 == 1:
                self.get_logger().warn(
                    f"mic publish queue full — dropped {self._dropped} frames; "
                    "the ROS executor is not draining audio fast enough.",
                    throttle_duration_sec=10.0)

    def _drain(self):
        for _ in range(8):                # bounded: never starve the executor
            try:
                chunk = self._q.get_nowait()
            except queue.Empty:
                return
            msg = UInt8MultiArray()
            # array.array('B'), NOT list(): rclpy serialises a uint8[] from a
            # buffer at C speed, while a list makes 640 Python ints per frame —
            # 32k int objects/s at 50 Hz, on the GIL, both here and again in the
            # subscriber. Same bytes on the wire, a fraction of the cost.
            msg.data = array.array("B", chunk.pcm)
            self.pub_mic.publish(msg)

    # ---- feedback --------------------------------------------------------------
    def _publish_state(self):
        st = self.core.state()
        self.pub_state.publish(Bool(data=not st.muted))

    def _publish_health(self):
        h = dict(self.core.health())
        st = self.core.state()
        h["connected"] = st.connected
        h["card"] = st.card
        self.pub_health.publish(String(data=json.dumps(h)))

    def destroy_node(self):
        try:
            if self._unsub is not None:
                self._unsub()
            if self._drain_timer is not None:
                self.destroy_timer(self._drain_timer)
            if self._owns_core:
                self.core.close()
        finally:
            super().destroy_node()


# legacy class name (pre-mabel_hw shims + old importers)
AudioSubsystem = AudioNode

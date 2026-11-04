"""
audio_streams.py — the :9610 voice plane: robot mic up, operator speaker down.

The ONE duplex plane in the 96xx block. Read data_streams.py's module docstring
before touching it: the "every listener is read-only broadcast" rule has a
narrow, argued exception here, and the argument is what keeps it narrow — audio
carries NO actuation authority (it cannot move the robot and never reaches
controller/'s gate), so hosting it off :9090 costs no safety while saving the
50 Hz control loop from a continuous PCM firehose.

    device --(AUDIO 0x07, ws binary)--> AudioHub.on_speaker --> plant/HAL --> bar
    bar    --> AudioHub.publish_mic --(AUDIO 0x07)--> every subscribed device

★ AUDIO IS A SEQUENCE, NOT A SNAPSHOT — the one place this file deliberately
breaks the house pattern. Every other hub here is latest-wins, and the reasoning
is written down in cloud_streams.py: "clouds are snapshots — a slow client gets a
fresher cloud, never a backlog". That does NOT transfer. Dropping the oldest
frame of speech does not make the voice fresher, it makes it *scrambled*: the
listener hears a click and a word out of order. So this hub uses a bounded FIFO
and, on overflow, drops the WHOLE backlog to resync (one clean gap, then correct
audio) rather than shuffling frames. `seq` in the 0x07 header lets the client see
exactly that gap.

DEMAND-DRIVEN, like the cloud planes: `enabled()` is false until someone
connects, so the robot's sound card is never opened for a room nobody is
listening to.

Auth + hosting are the shared ones (`data_streams.serve_hub_ws`): the port IS the
selector, `?key=<MABEL_TOKEN>` gates it, and `compression=None` because PCM is
incompressible and latency-critical (deflate would burn CPU to grow the frame).
"""
from __future__ import annotations

import asyncio
import time

import data_streams
from map_stream import AUDIO_CODEC_PCM16, decode_audio, encode_audio

__all__ = ["AudioHub", "AUDIO"]


class AudioHub:
    """Fan-out (mic) + fan-in (speaker) for the voice plane.

    Same subscriber surface as MapStreamHub/TelemetryHub, so
    data_streams.serve_hub_ws hosts it unchanged and it can move to another
    process later without touching a client.
    """

    # ~0.5 s of speech at 20 ms/frame. Deep enough to ride out a scheduler
    # hiccup; shallow enough that a stalled client can never accumulate a
    # noticeable delay before the resync below fires.
    _QUEUE_MAX = 25

    def __init__(self) -> None:
        self._subs: dict = {}
        self._loop = None
        self._seq = 0
        self._on_speaker = None            # set by the owner (bridge/plant sink)
        self._notice = None                # set when this rig has no sound card
        self._demand_cb = None             # called on the FIRST/LAST subscriber
        self._rate = data_streams.AUDIO_RATE
        self._channels = data_streams.AUDIO_CHANNELS
        self._stats = {"micFrames": 0, "spkFrames": 0, "rejected": 0, "resyncs": 0}

    # ── hub surface ───────────────────────────────────────────────────────────
    def attach_loop(self, loop) -> None:
        self._loop = loop

    def enabled(self) -> bool:
        """True only while a device is connected — an unwatched robot must not
        hold its sound card open (the mic subsystem keys capture off this)."""
        return bool(self._subs)

    def set_speaker_sink(self, sink) -> None:
        """Register where inbound speaker PCM goes (the plant / HAL facade).

        A sink and not a subclass so the SAME hub serves sim and real, and the
        wire cannot tell which is behind it.

        Called from the CONTROL thread, after make_plant() — which happens long
        after the network thread binds this port. That ordering is why the port
        binds unconditionally and availability is wired in later, rather than the
        server asking the plant at bind time (it does not exist yet).
        """
        self._on_speaker = sink
        self._notice = None

    def set_demand_cb(self, cb) -> None:
        """cb(active: bool) on the first subscriber and the last one.

        This is what makes voice cost ZERO when nobody is talking, and it is not
        an optimisation — it is the whole reason the gateway can host this plane.
        The gateway is the process whose GIL the 50 Hz joint stream lives in (see
        perception_server.py: the map/cloud planes were EVICTED from it for
        exactly this). A permanently-subscribed mic pumps 50 Hz of PCM through
        that GIL for an empty room, and measurably drags proprio off 50 Hz.
        """
        self._demand_cb = cb

    def _demand(self, active: bool) -> None:
        cb = self._demand_cb
        if cb is None:
            return
        try:
            cb(active)
        except Exception:                   # noqa: BLE001 — never kill a socket
            pass

    def set_unavailable(self, reason: str) -> None:
        """This rig has no sound card: answer subscribers with an explicit notice
        instead of a hello, so a client can tell "sim has no bar" from "the plane
        is broken". Never a refused connection — the house rule for an absent
        source (data_streams.NoticeSource)."""
        self._on_speaker = None
        self._notice = reason

    def stats(self) -> dict:
        return dict(self._stats, subscribers=len(self._subs))

    # ── mic: robot -> devices ─────────────────────────────────────────────────
    def publish_mic(self, pcm: bytes, *, rate: int = 0, channels: int = 0) -> None:
        """Broadcast one mic frame. Safe to call from ANY thread (the HAL's
        capture thread does) — the encode happens here, once, and the send is
        hopped onto the network loop.
        """
        if not self._subs or not pcm:
            return                          # unwatched: encode nothing, cost nothing
        frame = encode_audio(pcm, seq=self._seq, rate=rate or self._rate,
                             channels=channels or self._channels,
                             codec=AUDIO_CODEC_PCM16)
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self._stats["micFrames"] += 1
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._fanout, frame)

    def _fanout(self, frame: bytes) -> None:
        for q in list(self._subs.values()):
            self._offer(q, frame)

    def _offer(self, q: asyncio.Queue, frame: bytes) -> None:
        """Bounded enqueue with RESYNC-on-overflow.

        NOT the latest-wins _offer() used by every other hub in this server: that
        one pops the oldest to make room, which reorders a voice stream. Here a
        full queue means the client is not draining, so its backlog is already
        stale speech — drop ALL of it and start clean. One gap, correct audio
        after; the client sees it via the 0x07 `seq` jump.
        """
        if q.full():
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._stats["resyncs"] += 1
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    # ── speaker: devices -> robot ─────────────────────────────────────────────
    def _ingest_speaker(self, raw: bytes) -> None:
        got = decode_audio(raw)
        if got is None:
            self._stats["rejected"] += 1
            return                          # malformed: never hand it to the card
        pcm, _seq, rate, channels, codec = got
        if codec != AUDIO_CODEC_PCM16:
            # A client announcing Opus would otherwise have its compressed bytes
            # played as PCM — loud noise into a speaker in a room with people.
            self._stats["rejected"] += 1
            return
        if rate != self._rate or channels != self._channels:
            # Refuse rather than resample: the plug layer on the robot converts
            # OUR format, and silently accepting 48 kHz here would play it at 16k
            # (a third speed, an octave and a half down). The client is told the
            # expected format in `hello` and must send it.
            self._stats["rejected"] += 1
            return
        sink = self._on_speaker
        if sink is None:
            return                          # no plant on this rig: drop, don't queue
        self._stats["spkFrames"] += 1
        try:
            sink(pcm)
        except Exception:                   # noqa: BLE001 — a sink must never kill the socket
            pass

    # ── subscribers ───────────────────────────────────────────────────────────
    async def add_subscriber(self, ws) -> None:
        """hello (the format contract) + live mic frames out, speaker frames in.

        Inbound BINARY frames are AUDIO 0x07 speaker records; inbound TEXT is
        JSON control ({"type":"ping"}). The frame type is the discriminator, the
        same way the binary map planes tell a notice from a record.
        """
        import json

        if self._notice is not None:
            # No source on this rig — explain, heartbeat, never refuse.
            frame = data_streams.notice_frame(self._notice, data_streams.AUDIO_PORT)
            try:
                while True:
                    await ws.send(frame)
                    await asyncio.sleep(5.0)
            except Exception:                   # noqa: BLE001
                pass
            return
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAX)
        # hello goes through the QUEUE, not a direct ws.send(): the writer task
        # below owns this socket, and two coroutines awaiting ws.send() on one
        # connection is exactly what websockets forbids. Enqueue first so it is
        # the first thing on the wire regardless of scheduling.
        q.put_nowait(json.dumps({
            "type": "hello", "t": time.time(),
            # The client MUST send this exact format — _ingest_speaker refuses
            # anything else rather than play it at the wrong speed.
            "codec": "pcm16", "rate": self._rate, "channels": self._channels,
            "frameMs": data_streams.AUDIO_FRAME_MS,
            "record": "0x07", "duplex": True,
        }))
        first = not self._subs
        self._subs[ws] = q
        if first:
            self._demand(True)              # a listener arrived: open the tap
        writer = asyncio.create_task(self._writer(ws, q))
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    self._ingest_speaker(bytes(raw))
                    continue
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict) and msg.get("type") == "ping":
                        await q.put(json.dumps({"type": "pong", "t": msg.get("t"),
                                                "serverT": time.time()}))
                except Exception:           # noqa: BLE001
                    pass
        except Exception:                   # noqa: BLE001
            pass
        finally:
            self._subs.pop(ws, None)
            if not self._subs:
                self._demand(False)         # last one left: shut the tap off
            writer.cancel()

    async def _writer(self, ws, q: asyncio.Queue) -> None:
        """Single sender per socket (websockets forbids concurrent ws.send)."""
        try:
            while True:
                await ws.send(await q.get())
        except (asyncio.CancelledError, Exception):   # noqa: BLE001
            pass


# Module singleton, like TELEMETRY / MAP_HUB.
AUDIO = AudioHub()

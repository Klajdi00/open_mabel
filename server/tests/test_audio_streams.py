"""The :9610 voice plane — the AUDIO 0x07 record and the AudioHub.

Pure asyncio + struct, no websockets server and no sound card, so these run
anywhere. Two things here are load-bearing and easy to "clean up" into bugs, so
they are pinned hard:

  * the hub RESYNCS on overflow instead of latest-wins (every other hub in this
    server drops the oldest — that reorders speech);
  * a frame in the wrong format is REFUSED, never resampled or best-efforted
    (playing 48 kHz PCM at 16 kHz is an octave and a half down, into a real
    speaker in a room with people).

server/ modules import each other flat, so the server dir goes on sys.path the
same way the studios' contract tests locate it: walk up until wire_protocol.py is
in sight rather than hard-coding a depth.
"""
import asyncio
import json
import os
import struct
import sys

import pytest

# These tests drive coroutines with asyncio.run() rather than pytest-asyncio: the
# repo runs pytest with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 (ROS's launch_testing
# plugin breaks collection), which also disables pytest-asyncio. One less plugin
# to depend on, and the suite runs identically on the Mac and on thor.

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_server():
    d = _HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "wire_protocol.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("server/ (wire_protocol.py) not found above " + _HERE)


sys.path.insert(0, _find_server())

import data_streams                                                   # noqa: E402
from audio_streams import AudioHub                                    # noqa: E402
from map_stream import (AUDIO_CODEC_OPUS, AUDIO_CODEC_PCM16,          # noqa: E402
                        decode_audio, encode_audio)

RATE = data_streams.AUDIO_RATE
CH = data_streams.AUDIO_CHANNELS
FRAME = int(RATE * CH * 2 * data_streams.AUDIO_FRAME_MS / 1000)       # 640 B
PCM = b"\x11\x22" * (FRAME // 2)


# ── the wire record ───────────────────────────────────────────────────────────

def test_audio_record_round_trips_through_the_live_encoder():
    f = encode_audio(PCM, seq=7, rate=RATE, channels=CH)
    assert f[0] == 0x07
    assert len(f) == 16 + FRAME                    # 16-byte header, 2.5% overhead
    assert decode_audio(f) == (PCM, 7, RATE, CH, AUDIO_CODEC_PCM16)


def test_audio_header_layout_is_frozen():
    """The apps + the browser parse these offsets by hand — they are contract."""
    f = encode_audio(PCM, seq=9, rate=RATE, channels=CH)
    tag, codec, ch, seq, rate, n = struct.unpack("<BBHIII", f[:16])
    assert (tag, codec, ch, seq, rate, n) == (0x07, 0, CH, 9, RATE, FRAME)


def test_seq_wraps_at_u32_rather_than_overflowing():
    assert decode_audio(encode_audio(b"ab", seq=2 ** 32 + 5,
                                     rate=RATE, channels=CH))[1] == 5


@pytest.mark.parametrize("bad,why", [
    (b"", "empty"),
    (b"\x05" + b"\x00" * 20, "wrong record tag (a CLOUD frame)"),
    (b"\x07\x00", "truncated header"),
    (encode_audio(PCM, seq=1, rate=RATE, channels=CH)[:-10], "truncated payload"),
])
def test_malformed_frames_decode_to_none(bad, why):
    assert decode_audio(bad) is None, why


def test_channels_must_be_positive():
    with pytest.raises(ValueError):
        encode_audio(PCM, seq=0, rate=RATE, channels=0)


# ── the hub ───────────────────────────────────────────────────────────────────

class FakeWS:
    """Minimal duplex websocket: `outbox` collects sends, `inbox` is replayed.

    `linger` models a socket that stays OPEN after the last inbound message —
    which is what a real one does. Without it the iterator ends the instant the
    inbox drains, add_subscriber() returns, and its `finally` cancels the writer
    before anything queued (hello, pong, mic frames) can reach the wire; a test
    would then be measuring teardown, not behaviour.
    """

    def __init__(self, inbox=(), linger=0.05):
        self.outbox = []
        self._inbox = list(inbox)
        self._linger = linger

    async def send(self, frame):
        self.outbox.append(frame)

    def __aiter__(self):
        async def _gen():
            for m in self._inbox:
                yield m
            await asyncio.sleep(self._linger)   # stay open, like a real socket
        return _gen()


@pytest.fixture()
def hub():
    h = AudioHub()
    h.attach_loop(asyncio.get_event_loop_policy().new_event_loop())
    return h


def test_hub_is_demand_driven(hub):
    """No subscriber ⇒ disabled ⇒ the HAL never opens the sound card."""
    assert hub.enabled() is False
    hub._subs[object()] = asyncio.Queue()
    assert hub.enabled() is True


def test_publish_mic_costs_nothing_unwatched(hub):
    """Not merely 'sends nothing' — it must not even ENCODE (an unwatched robot
    pays zero, the same rule telemetry states for its snapshot slots)."""
    hub.publish_mic(PCM)
    assert hub.stats()["micFrames"] == 0


def test_hello_states_the_format_contract():
    ws = FakeWS()          # no inbound traffic; the hello must arrive anyway

    async def _go():
        hub = AudioHub()
        hub.attach_loop(asyncio.get_running_loop())
        await hub.add_subscriber(ws)
    asyncio.run(_go())
    hello = json.loads(ws.outbox[0])
    assert hello["type"] == "hello"
    assert (hello["codec"], hello["rate"], hello["channels"]) == ("pcm16", RATE, CH)
    assert hello["duplex"] is True and hello["record"] == "0x07"


def test_speaker_pcm_reaches_the_sink():
    got, stats = [], {}

    async def _go():
        hub = AudioHub()
        hub.attach_loop(asyncio.get_running_loop())
        hub.set_speaker_sink(got.append)
        await hub.add_subscriber(FakeWS([encode_audio(PCM, seq=0, rate=RATE,
                                                      channels=CH)]))
        stats.update(hub.stats())
    asyncio.run(_go())
    assert got == [PCM]
    assert stats["spkFrames"] == 1


@pytest.mark.parametrize("frame,why", [
    (encode_audio(PCM, seq=0, rate=48000, channels=CH),
     "48 kHz would play at 16 kHz — an octave and a half down"),
    (encode_audio(PCM, seq=0, rate=RATE, channels=2),
     "stereo bytes played as mono is double-speed garbage"),
    (encode_audio(PCM, seq=0, rate=RATE, channels=CH, codec=AUDIO_CODEC_OPUS),
     "Opus bytes played as PCM is loud noise"),
    (b"\x05" + b"\x00" * 32, "not an audio record at all"),
], ids=["wrong_rate", "wrong_channels", "wrong_codec", "wrong_record"])
def test_wrong_format_is_refused_never_best_efforted(frame, why):
    got, stats = [], {}

    async def _go():
        hub = AudioHub()
        hub.attach_loop(asyncio.get_running_loop())
        hub.set_speaker_sink(got.append)
        await hub.add_subscriber(FakeWS([frame]))
        stats.update(hub.stats())
    asyncio.run(_go())
    assert got == [], why
    assert stats["rejected"] == 1


def test_a_throwing_sink_never_kills_the_socket():
    ws = FakeWS([encode_audio(PCM, seq=0, rate=RATE, channels=CH)])
    left = {}

    async def _go():
        hub = AudioHub()
        hub.attach_loop(asyncio.get_running_loop())
        hub.set_speaker_sink(
            lambda _p: (_ for _ in ()).throw(RuntimeError("plant down")))
        await hub.add_subscriber(ws)           # must not raise
        left["subs"] = ws in hub._subs
    asyncio.run(_go())
    assert left["subs"] is False               # and must clean up


def test_ping_pongs_on_the_control_path():
    ws = FakeWS([json.dumps({"type": "ping", "t": 123.0})])

    async def _go():
        hub = AudioHub()
        hub.attach_loop(asyncio.get_running_loop())
        await hub.add_subscriber(ws)
    asyncio.run(_go())
    pongs = [json.loads(m) for m in ws.outbox if '"pong"' in str(m)]
    assert pongs and pongs[0]["t"] == 123.0


def test_overflow_resyncs_instead_of_reordering(hub):
    """★ THE inversion of the house rule. Latest-wins would pop the OLDEST frame
    to make room — which hands the listener speech out of order. A full queue
    means the client is not draining, so its backlog is already stale: drop it
    all and start clean."""
    q = asyncio.Queue(maxsize=hub._QUEUE_MAX)
    for i in range(hub._QUEUE_MAX):
        hub._offer(q, encode_audio(PCM, seq=i, rate=RATE, channels=CH))
    assert q.full()
    hub._offer(q, encode_audio(PCM, seq=999, rate=RATE, channels=CH))   # overflow

    assert hub.stats()["resyncs"] == 1
    assert q.qsize() == 1, "backlog not dropped — this is the latest-wins bug"
    assert decode_audio(q.get_nowait())[1] == 999, "kept a stale frame"


def test_no_overflow_keeps_every_frame_in_order(hub):
    q = asyncio.Queue(maxsize=hub._QUEUE_MAX)
    for i in range(hub._QUEUE_MAX):
        hub._offer(q, encode_audio(PCM, seq=i, rate=RATE, channels=CH))
    seqs = [decode_audio(q.get_nowait())[1] for _ in range(hub._QUEUE_MAX)]
    assert seqs == list(range(hub._QUEUE_MAX)), "frames reordered without overflow"
    assert hub.stats()["resyncs"] == 0


def test_mic_seq_is_monotonic_across_frames(hub):
    """The client's ONLY gap detector — if it doesn't advance, a drop is invisible."""
    hub._subs[object()] = asyncio.Queue(maxsize=8)
    hub._loop = None                              # skip the fanout hop
    for _ in range(3):
        hub.publish_mic(PCM)
    assert hub._seq == 3


# ── the port contract ─────────────────────────────────────────────────────────

def test_audio_port_is_registered_and_collision_free():
    assert data_streams.AUDIO_PORT == 9610
    assert data_streams.AUDIO_PORT in data_streams.ALL_STREAM_PORTS
    assert len(set(data_streams.ALL_STREAM_PORTS)) == \
        len(data_streams.ALL_STREAM_PORTS), "96xx port collision"
    assert data_streams.port_map()["audio"] == 9610


def test_wire_format_matches_the_hal():
    """server and hardware_bridge must agree byte-for-byte or the bar plays
    chipmunks; the HAL side is frozen in its contract.json."""
    assert (data_streams.AUDIO_RATE, data_streams.AUDIO_CHANNELS,
            data_streams.AUDIO_FRAME_MS) == (16000, 1, 20)
    assert FRAME == 640


def test_audio_is_gated_by_the_master_kill_switch(monkeypatch):
    monkeypatch.setenv("MABEL_DATA_STREAMS", "0")
    assert data_streams.streams_enabled() is False

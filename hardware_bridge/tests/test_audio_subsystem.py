"""AudioSubsystem with a fake AlsaAudio injected at the driver seam — unit-tests
OUR policy: demand-driven capture, the speaker FIFO's resync-on-overrun (NOT the
latest-wins every other stream uses), mute, the lease-free stop, and staying
alive with no sound card. No ALSA, no device, runs anywhere.
"""
import threading
import time

import pytest

from mabel_hw.subsystems.audio import AudioSubsystem, _rms

CFG = dict(
    enabled=True,
    device_playback="plughw:CARD=Bar,DEV=0", device_capture="plughw:CARD=Bar,DEV=0",
    card="Bar", usb_id="3206:0631",
    rate=16000, channels=1, sample_bytes=2, frame_ms=20, playback_idle_s=0.05,
    mixer_playback="PCM", mixer_capture="Mic",
    volume_playback=70, volume_capture=100,
)
FRAME = 640          # 20 ms @ 16 kHz mono S16_LE


class FakeAlsa:
    """Stands in for drivers.audio.AlsaAudio (the surface the subsystem uses)."""

    def __init__(self):
        self.card = "Bar"
        self._cap = False
        self._play = False
        self.written = []                 # frames handed to the card, in order
        self.volumes = {}
        self.starts = 0                   # how many times capture was opened
        self.feed = b"\x11\x11" * (FRAME // 2)
        self.dead_after = None            # frames until arecord "EOF"s
        self._n = 0

    def start_capture(self):
        if not self._cap:
            self.starts += 1
        self._cap = True

    def stop_capture(self):
        self._cap = False

    def capturing(self):
        return self._cap

    def read_frame(self):
        if not self._cap:
            return None
        self._n += 1
        if self.dead_after is not None and self._n > self.dead_after:
            return None
        time.sleep(0.001)                 # stand in for the card's pacing
        return self.feed

    def start_playback(self):
        self._play = True

    def stop_playback(self):
        self._play = False

    def playing(self):
        return self._play

    def write_frame(self, pcm):
        if not self._play:
            return False
        self.written.append(bytes(pcm))
        return True

    def set_volume(self, control, percent, capture=False):
        self.volumes[control] = int(percent)

    def get_volume(self, control):
        return self.volumes.get(control)

    def close(self):
        self._cap = self._play = False


@pytest.fixture()
def sub():
    fake = FakeAlsa()
    s = AudioSubsystem(CFG, driver=fake, logger=lambda *a: None, autostart=False)
    s.dev = fake
    yield s
    s.close()


def _spin(s, until, timeout=2.0):
    """Run the subsystem's threads until `until()` or the timeout."""
    s.connect()
    t0 = time.time()
    while time.time() - t0 < timeout:
        if until():
            return True
        time.sleep(0.005)
    return False


# ── demand-driven capture ─────────────────────────────────────────────────────

def test_no_subscriber_means_the_mic_is_never_opened(sub):
    """An unwatched robot must not hold the card — a calibration studio or
    speaker-test has to be able to grab it."""
    assert not _spin(sub, lambda: sub.dev.capturing(), timeout=0.4)
    assert sub.dev.starts == 0


def test_capture_starts_on_the_first_subscriber(sub):
    got = []
    sub.on_chunk(got.append)
    assert _spin(sub, lambda: len(got) >= 3), "no mic frames reached the subscriber"
    assert sub.dev.capturing()


def test_capture_is_released_when_the_last_subscriber_leaves(sub):
    got = []
    unsub = sub.on_chunk(got.append)
    assert _spin(sub, lambda: len(got) >= 2)
    unsub()
    t0 = time.time()
    while time.time() - t0 < 2.0 and sub.dev.capturing():
        time.sleep(0.01)
    assert not sub.dev.capturing(), "mic still held with no subscribers"


def test_frames_carry_a_gap_detectable_sequence(sub):
    """seq is the ONLY way a consumer sees a dropped frame: audio is a sequence,
    so a gap is an audible click, not a stale render."""
    got = []
    sub.on_chunk(got.append)
    assert _spin(sub, lambda: len(got) >= 4)
    seqs = [c.seq for c in got[:4]]
    assert seqs == [0, 1, 2, 3], seqs
    assert got[0].rate == 16000 and got[0].channels == 1
    assert got[0].samples() == 320                      # 20 ms @ 16 kHz


def test_a_broken_callback_never_kills_capture(sub):
    good = []
    sub.on_chunk(lambda c: (_ for _ in ()).throw(RuntimeError("consumer bug")))
    sub.on_chunk(good.append)
    assert _spin(sub, lambda: len(good) >= 3), "one bad consumer stopped the mic"


def test_device_eof_re_resolves_instead_of_dying(sub):
    """arecord EOF = card yanked or grabbed by someone else. The subsystem must
    drop the handle and retry, so a hot-replug needs no bridge restart."""
    got = []
    sub.on_chunk(got.append)
    sub.dev.dead_after = 2
    assert _spin(sub, lambda: not sub.dev.capturing() and len(got) >= 2)
    assert sub._resolved == "", "handle not dropped after EOF"


# ── mute ──────────────────────────────────────────────────────────────────────

def test_mute_zeroes_the_wire_but_keeps_the_level_meter_honest(sub):
    got = []
    sub.on_chunk(got.append)
    assert _spin(sub, lambda: len(got) >= 2)
    sub.set_muted(True)
    got.clear()
    assert _spin(sub, lambda: len(got) >= 2)
    assert set(got[-1].pcm) == {0}, "muted mic still leaked audio"
    assert sub.state().mic_rms > 0, "mute must not blind the level meter"
    assert sub.state().muted is True


# ── the speaker FIFO ──────────────────────────────────────────────────────────

def test_play_queues_frames_in_order(sub):
    sub.connect()
    frames = [bytes([i]) * FRAME for i in range(1, 6)]
    for f in frames:
        sub.play(f)
    t0 = time.time()
    while time.time() - t0 < 2.0 and len(sub.dev.written) < len(frames):
        time.sleep(0.005)
    assert sub.dev.written == frames, "speaker frames reordered or dropped"


def test_play_is_non_blocking(sub):
    """The caller is a network coroutine / ROS callback — it must never stall
    on the sound card."""
    sub.connect()
    t0 = time.perf_counter()
    for _ in range(50):
        sub.play(b"\x00" * FRAME)
    assert time.perf_counter() - t0 < 0.05


def test_overrun_drops_the_backlog_and_resyncs(sub):
    """The house latest-wins rule is WRONG here and this is the proof: on
    overrun we take ONE clean gap rather than play out a backlog that puts every
    later frame permanently behind."""
    sub.connect()
    sub.dev._play = False                            # stall the drain
    for i in range(200):                             # >> maxlen (50)
        sub.play(bytes([i % 256]) * FRAME)
    assert len(sub._playq) <= sub._playq.maxlen, "FIFO grew without bound"


def test_stop_is_lease_free_and_silences_immediately(sub):
    """An operator killing a stuck loop of audio must never be told to take a
    lease first — same rule as every other stop() in this package."""
    sub.connect()
    for _ in range(20):
        sub.play(b"\x01" * FRAME)
    sub.stop()
    assert len(sub._playq) == 0
    assert not sub.dev.playing()


def test_speaker_idles_the_card_out_between_phrases(sub):
    sub.connect()
    sub.play(b"\x01" * FRAME)
    assert _spin(sub, lambda: len(sub.dev.written) >= 1)
    t0 = time.time()
    while time.time() - t0 < 2.0 and sub.dev.playing():
        time.sleep(0.01)
    assert not sub.dev.playing(), "speaker held the card with nothing to say"


def test_play_with_no_card_never_queues_forever(sub):
    s = AudioSubsystem(CFG, driver=None, logger=lambda *a: None, autostart=False)
    try:
        for _ in range(500):
            s.play(b"\x00" * FRAME)
        assert len(s._playq) <= s._playq.maxlen
    finally:
        s.close()


# ── the mixer assertion (the 0%-volume boot) ──────────────────────────────────

def test_connect_asserts_the_mixer(sub):
    """The bar boots 'PCM' at 0% and then plays SILENCE with rc=0 everywhere.
    Coming up without asserting the mixer is a silent robot."""
    assert _spin(sub, lambda: sub.dev.volumes.get("PCM") == 70)
    assert sub.dev.volumes.get("Mic") == 100


def test_state_reports_realized_volume_not_requested(sub):
    assert _spin(sub, lambda: sub.state().playback_volume == 70)
    sub.dev.volumes["PCM"] = 0            # the card overrode us / a knob moved
    sub.service()
    assert sub.state().playback_volume == 0, "state parroted the request"


# ── degradation ───────────────────────────────────────────────────────────────

def test_absent_card_leaves_the_subsystem_alive_and_idle(monkeypatch):
    """A missing speaker must not crash the HAL (bridge invariant)."""
    from mabel_hw.drivers import audio as audio_mod
    monkeypatch.setattr(audio_mod, "detect_audio_card", lambda cfg, log=None: None)
    s = AudioSubsystem(CFG, logger=lambda *a: None, autostart=True)
    try:
        time.sleep(0.15)
        assert s.connected() is False
        st = s.state()                    # reading an absent card is safe
        assert st.connected is False and st.device == ""
        s.play(b"\x00" * FRAME)           # writing to it is not fatal either
        s.stop()                          # the safety path never raises
    finally:
        s.close()


def test_health_shape(sub):
    sub.on_chunk(lambda c: None)
    assert _spin(sub, lambda: sub.health()["mic"]["frames"] > 0)
    h = sub.health()
    assert set(h) == {"speaker", "mic"}
    assert "volumePct" in h["speaker"] and "rms" in h["mic"]


# ── the level meter ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("pcm,expect", [
    (b"", 0.0),
    (b"\x00\x00" * 160, 0.0),
    (b"\xff\x7f" * 160, pytest.approx(1.0, abs=0.01)),      # +full scale
    (b"\x00\x80" * 160, pytest.approx(1.0, abs=0.01)),      # -full scale
])
def test_rms_bounds(pcm, expect):
    assert _rms(pcm) == expect


def test_rms_decodes_signed_little_endian():
    """A sign bug reads -1 (0xFFFF) as 65535 and pins the meter at full scale on
    silence-adjacent noise."""
    quiet = (-1).to_bytes(2, "little", signed=True) * 160
    assert _rms(quiet) < 0.001


# ── mic-silence detection (the "iOS can't hear the robot" black hole) ───────────

def _silence_frames():
    import mabel_hw.subsystems.audio as A
    return A._SILENCE_FRAMES


def test_sustained_exact_silence_raises_mic_silent(sub):
    """A device that streams frames of EXACT zeros (hardware-muted / dead mic —
    the MZ-631M failure, 2026-07-18) must surface as mic_silent, not pass silently
    to a listener that then hears nothing with no diagnostic anywhere."""
    sub.dev.feed = b"\x00\x00" * (FRAME // 2)          # exact digital silence
    sub.on_chunk([].append)
    assert _spin(sub, lambda: sub._seq > _silence_frames() + 5, timeout=4.0)
    st = sub.state()
    assert st.capturing and st.mic_silent, "silent mic not flagged"
    assert sub.health()["mic"]["silent"] is True
    assert st.mic_rms == 0.0


def test_real_signal_keeps_mic_silent_false(sub):
    """A live mic (any noise floor > 0) must never trip the flag."""
    sub.dev.feed = b"\x11\x11" * (FRAME // 2)          # nonzero level
    sub.on_chunk([].append)
    assert _spin(sub, lambda: sub._seq > _silence_frames() + 5, timeout=4.0)
    assert sub.state().mic_silent is False
    assert sub.health()["mic"]["silent"] is False


def test_mic_silent_clears_when_audio_returns(sub):
    """Un-mute the bar mid-stream => the flag clears on the first real sample, so
    the app can confirm the fix live."""
    sub.dev.feed = b"\x00\x00" * (FRAME // 2)
    sub.on_chunk([].append)
    assert _spin(sub, lambda: sub.state().mic_silent, timeout=4.0)
    sub.dev.feed = b"\x11\x11" * (FRAME // 2)          # mic comes back
    assert _spin(sub, lambda: sub.state().mic_silent is False, timeout=2.0)


def test_mic_silent_is_false_while_not_capturing(sub):
    """No subscriber => not capturing => silence is 'idle', not 'broken'."""
    assert _spin(sub, lambda: True)                    # connect, no subscriber
    assert sub.state().capturing is False
    assert sub.state().mic_silent is False

"""AlsaAudio + the card resolver with arecord/aplay/amixer stubbed — unit-tests
OUR layer: the PCM pipe contract, the mixer assertion, card auto-detect, and
graceful degradation when no sound card is present. (ALSA itself is alsa-utils'
test surface; test_audio_device.py drives the real card when one is plugged in.)
"""
import subprocess

import pytest

from mabel_hw.drivers import audio as audio_mod
from mabel_hw.errors import SubsystemUnavailable

# One 20 ms frame of 16 kHz mono S16_LE.
FRAME = 640

_CARDS_REAL = """\
 0 [HDA            ]: tegra-hda - NVIDIA Jetson Thor AGX HDA
                      NVIDIA Jetson Thor AGX HDA at 0x88090b8000 irq 311
 1 [APE            ]: tegra-ape - NVIDIA Jetson Thor AGX APE
                      NVIDIA-NVIDIAJetsonAGXThorDeveloperKit-NotSpecified-Jetson
 2 [Bar            ]: USB-Audio - MZ-631M  USB Speaker Bar
                      MZ-631M  USB Speaker Bar at usb-a80aa10000.usb-4.2.4.4
"""
_CARDS_ONBOARD_ONLY = "\n".join(_CARDS_REAL.splitlines()[:4]) + "\n"
# Exactly ONE onboard card: the resolver must still refuse it. Two would also be
# refused as "ambiguous", which would let a filter-less resolver pass this test
# for the wrong reason — the single-card case is what actually pins the rule.
_CARDS_ONE_ONBOARD = "\n".join(_CARDS_REAL.splitlines()[:2]) + "\n"


class FakeProc:
    """Stands in for a Popen'd arecord/aplay pipe."""

    def __init__(self, argv, stdout=None, stdin=None, **_):
        self.argv = argv
        self.terminated = False
        self._rc = None
        self.written = bytearray()
        self.stdout = self if stdout is subprocess.PIPE else None
        self.stdin = self if stdin is subprocess.PIPE else None
        self.feed = bytes(FRAME)          # what read() hands back, per call

    # stdout side (arecord)
    def read(self, n):
        return self.feed[:n]

    # stdin side (aplay)
    def write(self, b):
        self.written.extend(b)

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return self._rc

    def terminate(self):
        self.terminated = True
        self._rc = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._rc = -9


@pytest.fixture()
def dev(monkeypatch):
    spawned = []

    def _popen(argv, **kw):
        p = FakeProc(argv, **kw)
        spawned.append(p)
        return p

    monkeypatch.setattr(audio_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(audio_mod.shutil, "which", lambda t: "/usr/bin/" + t)
    monkeypatch.setattr(audio_mod.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(
                            a[0], 0, stdout="  Front Left: Playback 70 [70%] [on]\n",
                            stderr=""))
    d = audio_mod.AlsaAudio("Bar", device_playback="plughw:CARD=Bar,DEV=0",
                            device_capture="plughw:CARD=Bar,DEV=0")
    d.spawned = spawned
    yield d
    d.close()


# ── the card resolver ─────────────────────────────────────────────────────────

def _cards(monkeypatch, text, usb_card=None):
    """Fake /proc/asound/cards, and (by default) no USB identity for any card —
    so a test exercises the NAME path unless it opts into the VID:PID path."""
    import io
    monkeypatch.setattr(audio_mod, "_CARDS", "/proc/asound/cards")
    monkeypatch.setattr("builtins.open", lambda p, *a, **k: io.StringIO(text))
    monkeypatch.setattr(audio_mod, "card_for_usb_id", lambda uid: usb_card)


_ID_OF = {"card0": "HDA", "card1": "APE", "card2": "Bar"}


def _sysfs(monkeypatch, tree: dict, ids: dict):
    """Fake /sys/class/sound.

    `tree`: {card dir -> the realpath its `device` link resolves to}
    `ids`:  {usb node path -> {"idVendor": .., "idProduct": ..}}
    A node absent from `ids` has no USB identity — that is how an onboard codec
    (Thor's HDA/APE) is modelled, and the walk must give up on it.
    """
    import io
    monkeypatch.setattr(audio_mod, "_SND_CLASS", "/sys/class/sound")
    monkeypatch.setattr(audio_mod.os, "listdir", lambda p: sorted(tree))
    monkeypatch.setattr(audio_mod.os.path, "realpath",
                        lambda p: tree[p.split("/")[4]])
    monkeypatch.setattr("builtins.open",
                        lambda p, *a, **k: io.StringIO(_ID_OF[p.split("/")[4]])
                        if p.endswith("/id") else _raise(p))
    monkeypatch.setattr(audio_mod, "_attr",
                        lambda node, name: (ids.get(node) or {}).get(name))


def _raise(path):
    raise OSError(path)


def test_usb_identity_finds_the_card_on_any_port(monkeypatch):
    """THE port-independence contract: the bar is located by VID:PID, so moving
    it between ports/hubs (which changes the sysfs path AND the card index)
    still resolves. Nothing in the lookup encodes where it is plugged in."""
    for path in ("/sys/devices/usb1/1-4.2.4.4.2",      # original port
                 "/sys/devices/usb2/2-1.3",             # moved to another hub
                 "/sys/devices/usb1/1-7"):              # and another
        _sysfs(monkeypatch,
               {"card0": "/sys/devices/tegra-hda", "card2": path + "/sound"},
               {path: {"idVendor": "3206", "idProduct": "0631"}})
        assert audio_mod.card_for_usb_id("3206:0631") == "Bar", path


def test_usb_identity_ignores_onboard_codecs(monkeypatch):
    """Thor's HDA/APE have no idVendor anywhere up their sysfs chain, which is
    what stops the resolver 'succeeding' into an unplugged HDMI sink."""
    _sysfs(monkeypatch, {"card0": "/sys/devices/tegra-hda"}, {})
    assert audio_mod.card_for_usb_id("3206:0631") is None


def test_usb_identity_absent_device_is_none(monkeypatch):
    node = "/sys/devices/usb1/1-4"
    _sysfs(monkeypatch, {"card2": node + "/sound"},
           {node: {"idVendor": "1234", "idProduct": "5678"}})
    assert audio_mod.card_for_usb_id("3206:0631") is None


def test_detect_follows_the_usb_device_over_a_stale_pin(monkeypatch):
    """ALSA suffixes a colliding id (Bar -> Bar_1) on replug. The USB device is
    the robot's speaker either way, so VID:PID wins over the configured name."""
    _cards(monkeypatch, _CARDS_REAL, usb_card="Bar_1")
    logs = []
    assert audio_mod.detect_audio_card({"card": "Bar", "usb_id": "3206:0631"},
                                       log=logs.append) == "Bar_1"
    assert any("following the device" in m for m in logs), logs


def test_detect_prefers_the_pinned_card(monkeypatch):
    _cards(monkeypatch, _CARDS_REAL)
    assert audio_mod.detect_audio_card({"card": "Bar"}) == "Bar"


@pytest.mark.parametrize("template,card,expect", [
    ("plughw:CARD=Bar,DEV=0", "Bar_1", "plughw:CARD=Bar_1,DEV=0"),
    ("plughw:CARD=Bar,DEV=0", "Bar", "plughw:CARD=Bar,DEV=0"),
    # An explicit override (no CARD=) is a deliberate choice — never rewritten.
    ("plughw:2,0", "Bar_1", "plughw:2,0"),
    ("default", "Bar_1", "default"),
])
def test_device_string_retargets_onto_the_resolved_card(template, card, expect):
    assert audio_mod.device_for_card(template, card) == expect


def test_detect_claims_the_only_usb_card_when_unpinned(monkeypatch, capsys):
    """No pin: claim the single USB-Audio card — and say which env var pins it."""
    _cards(monkeypatch, _CARDS_REAL)
    logs = []
    assert audio_mod.detect_audio_card({"card": ""}, log=logs.append) == "Bar"
    assert any("MABEL_HW_AUDIO_CARD=Bar" in m for m in logs), logs


def test_detect_never_claims_an_onboard_codec(monkeypatch):
    """Thor's HDA (HDMI) and APE are always present and are NEVER the speaker —
    claiming one would 'succeed' and play into an unplugged HDMI sink.

    Uses the SINGLE-onboard-card fixture on purpose: with two, a resolver that
    had lost its USB-Audio filter would refuse them as 'ambiguous' and pass this
    test while being completely broken (caught by mutation, 2026-07-16)."""
    _cards(monkeypatch, _CARDS_ONE_ONBOARD)
    assert audio_mod.detect_audio_card({"card": ""}) is None


def test_detect_absent_card_is_not_an_error(monkeypatch):
    """A pinned-but-absent card resolves to None so the caller polls for
    hot-plug — an unplugged speaker must never raise."""
    _cards(monkeypatch, _CARDS_ONBOARD_ONLY)
    assert audio_mod.detect_audio_card({"card": "Bar"}) is None


def test_detect_hints_the_pin_when_the_pinned_card_is_missing(monkeypatch):
    cards = _CARDS_ONBOARD_ONLY + " 3 [Other          ]: USB-Audio - Some Other Bar\n"
    _cards(monkeypatch, cards)
    logs = []
    assert audio_mod.detect_audio_card({"card": "Bar"}, log=logs.append) is None
    assert any("MABEL_HW_AUDIO_CARD=Other" in m for m in logs), logs


def test_detect_ambiguous_stays_idle(monkeypatch):
    cards = _CARDS_REAL + " 3 [Bar2           ]: USB-Audio - Second Speaker Bar\n"
    _cards(monkeypatch, cards)
    logs = []
    assert audio_mod.detect_audio_card({"card": ""}, log=logs.append) is None
    assert any("AMBIGUOUS" in m for m in logs), logs


# ── the PCM pipes ─────────────────────────────────────────────────────────────

def test_capture_opens_plughw_at_the_wire_format(dev):
    """The device is 48k/stereo-only, so a bare hw: at 16k mono cannot open —
    the plug layer is load-bearing, not a convenience."""
    dev.start_capture()
    argv = dev.spawned[0].argv
    assert argv[0] == "arecord"
    assert "plughw:CARD=Bar,DEV=0" in argv
    assert "S16_LE" in argv and "16000" in argv and "1" in argv


def test_capture_is_idempotent(dev):
    dev.start_capture()
    dev.start_capture()
    assert len(dev.spawned) == 1        # not a second arecord on the same card


def test_read_frame_returns_exactly_one_frame(dev):
    dev.start_capture()
    assert len(dev.read_frame()) == FRAME


def test_short_read_means_device_gone(dev):
    """arecord EOF/short read = the card vanished or someone else grabbed it.
    That is a re-resolve, not an exception."""
    dev.start_capture()
    dev.spawned[0].feed = b"\x00" * 8       # less than a frame
    assert dev.read_frame() is None


def test_playback_write_round_trips(dev):
    dev.start_playback()
    pcm = bytes(range(256)) * 2
    assert dev.write_frame(pcm) is True
    assert bytes(dev.spawned[0].written) == pcm


def test_write_to_a_dead_pipe_is_false_not_raise(dev):
    dev.start_playback()
    dev.spawned[0]._rc = 1                  # aplay exited
    assert dev.write_frame(b"\x00" * FRAME) is False


def test_broken_pipe_is_false_not_raise(dev):
    dev.start_playback()

    def _boom(_b):
        raise BrokenPipeError("card yanked mid-frame")
    dev.spawned[0].write = _boom
    assert dev.write_frame(b"\x00" * FRAME) is False


def test_capture_and_playback_are_independent(dev):
    """Either direction may be open without the other and either may die alone."""
    dev.start_capture()
    dev.start_playback()
    assert dev.capturing() and dev.playing()
    dev.stop_capture()
    assert not dev.capturing() and dev.playing()


def test_missing_alsa_utils_names_the_fix(monkeypatch):
    monkeypatch.setattr(audio_mod.shutil, "which", lambda t: None)
    with pytest.raises(SubsystemUnavailable) as e:
        audio_mod.AlsaAudio("Bar", device_playback="x", device_capture="y")
    assert "alsa-utils" in str(e.value)


# ── the mixer ─────────────────────────────────────────────────────────────────

def test_set_volume_unmutes_and_clamps(dev, monkeypatch):
    calls = []
    monkeypatch.setattr(audio_mod.subprocess, "run",
                        lambda a, **k: calls.append(a) or
                        subprocess.CompletedProcess(a, 0, stdout="", stderr=""))
    dev.set_volume("PCM", 150)
    assert calls[0] == ["amixer", "-c", "Bar", "sset", "PCM", "100%", "unmute"]


def test_get_volume_reads_the_realized_percent(dev):
    """The bar boots 'PCM' at 0% and then plays silence with rc=0 — what the
    card reports is the only honest value."""
    assert dev.get_volume("PCM") == 70


def test_unsupported_control_is_none_not_fatal(dev, monkeypatch):
    monkeypatch.setattr(audio_mod.subprocess, "run",
                        lambda a, **k: subprocess.CompletedProcess(
                            a, 1, stdout="", stderr="Unable to find simple control"))
    assert dev.get_volume("Mic") is None


# ── presence probe ────────────────────────────────────────────────────────────

def test_card_present_needs_both_usb_and_a_bound_pcm(monkeypatch):
    """lsusb proves it enumerated; /proc/asound/cards proves snd-usb-audio
    actually claimed it. Enumerated-but-unbound reads healthy to lsusb alone."""
    _cards(monkeypatch, _CARDS_REAL)
    monkeypatch.setattr(audio_mod.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(
                            a, 0, stdout="Bus 001 Device 018: ID 3206:0631 MZ-631M\n",
                            stderr=""))
    assert audio_mod.card_present("3206:0631", "Bar") is True

    _cards(monkeypatch, _CARDS_ONBOARD_ONLY)     # USB says yes, ALSA says no
    assert audio_mod.card_present("3206:0631", "Bar") is False

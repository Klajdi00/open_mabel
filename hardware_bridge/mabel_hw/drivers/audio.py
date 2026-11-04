"""audio.py — the ALSA device layer: two PCM pipes and a mixer.

Protocol + raw frames only, no policy (the capture thread, the retry ladder and
the typed snapshots live one layer up in mabel_hw.subsystems.audio).

WHY SUBPROCESSES AND NOT A BINDING. thor has no sounddevice/pyaudio/pyalsaaudio
and no ffmpeg; `alsa-utils` is present and is what actually drives this card
today. `arecord`/`aplay` are ALSA's own reference clients: they hold the PCM
open, do the period pacing in C, and hand us bytes over a pipe — the same
worker-process shape the cameras already use, for the same reason (the copy loop
never touches the GIL that the control loop needs). A binding would add a build
dep to buy nothing: we would still be memcpy-ing S16 frames.

THE PLUG LAYER IS LOAD-BEARING. The current card (JMTek 0c76:1203) is 2ch both
ways; we speak 16 kHz MONO. `plughw:` makes the kernel downmix (and resample,
which the old 48 kHz-only MZ-631M bar depended on outright). Every device string
here must stay a plughw: string — see config.AUDIO for the full device story.

Nothing in this module may import rclpy (tests/test_mabel_hw_pure.py).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from typing import Optional

from ..errors import SubsystemUnavailable

__all__ = ["AudioError", "AlsaAudio", "detect_audio_card", "card_present",
           "card_for_usb_id", "device_for_card"]


class AudioError(RuntimeError):
    """The sound card refused an operation (open/read/write/mixer)."""


_CARDS = "/proc/asound/cards"
_SND_CLASS = "/sys/class/sound"


def card_for_usb_id(usb_id: str) -> Optional[str]:
    """ALSA card *id* of the USB device with this VID:PID, or None if absent.

    THE PORT-INDEPENDENT IDENTITY. Walks /sys/class/sound/card*/device up to the
    owning USB device node and matches idVendor:idProduct, so the sound card is
    found on ANY port, hub, or card index — nothing here encodes where it is
    plugged in:

        card2 -> id "Device" -> 4.2.3 -> 0c76:1203

    Onboard codecs (Thor's HDA/APE) have no idVendor anywhere up their sysfs
    chain, so they can never match — which is what stops the resolver from
    "succeeding" into an unplugged HDMI sink.
    """
    want = (usb_id or "").strip().lower()
    if ":" not in want:
        return None
    vid, pid = want.split(":", 1)
    try:
        cards = os.listdir(_SND_CLASS)
    except OSError:
        return None
    for entry in sorted(cards):
        if not entry.startswith("card"):
            continue
        base = os.path.join(_SND_CLASS, entry)
        try:
            with open(os.path.join(base, "id")) as fh:
                cid = fh.read().strip()
        except OSError:
            continue
        node = os.path.realpath(os.path.join(base, "device"))
        # Walk toward / until a node carries idVendor (the USB device itself).
        for _ in range(8):
            v = _attr(node, "idVendor")
            if v is not None:
                if v.lower() == vid and (_attr(node, "idProduct") or "").lower() == pid:
                    return cid
                break
            parent = os.path.dirname(node)
            if parent == node or parent == "/":
                break
            node = parent
    return None


def _attr(node: str, name: str) -> Optional[str]:
    try:
        with open(os.path.join(node, name)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def card_present(usb_id: str = "", card: str = "") -> bool:
    """True when the sound card is on the bus AND the kernel bound a PCM to it.

    Two-part on purpose, mirroring the RealSense probe in real_run.sh: lsusb
    proves the device enumerated, /proc/asound/cards proves snd-usb-audio
    actually claimed it. A device that enumerates but never binds is the failure
    mode a bare lsusb check reports as healthy.
    """
    if card and _card_index(card) is None:
        return False
    if not usb_id:
        return bool(card)
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True,
                             timeout=5.0).stdout
    except Exception:                                    # noqa: BLE001
        return bool(card) and _card_index(card) is not None
    return usb_id.lower() in out.lower()


def _card_index(card: str) -> Optional[int]:
    """ALSA card *index* for a card id, or None when that id is not loaded."""
    try:
        with open(_CARDS) as fh:
            text = fh.read()
    except OSError:
        return None
    # Lines look like: " 2 [Device         ]: USB-Audio - USB PnP Audio Device"
    for m in re.finditer(r"^\s*(\d+)\s+\[([^\]]+)\]", text, re.M):
        if m.group(2).strip() == card:
            return int(m.group(1))
    return None


def detect_audio_card(cfg: dict, log=None) -> Optional[str]:
    """Resolve the card id to open, or None when no USB audio device is present.

      1. the VID:PID's card, wherever it is plugged in  (the authoritative path);
      2. the configured card id, if it is loaded        (explicit override);
      3. otherwise the single USB-Audio card that is NOT an onboard codec —
         logged loudly with the exact env var to pin it;
      4. otherwise None: the caller stays idle and retries (hot-plug support).

    PORT-INDEPENDENT BY CONSTRUCTION. Step 1 keys on the USB device's identity,
    not its location, so the bar can move between ports/hubs (or renumber from
    card 2 to card 3) and still resolve — and the caller re-runs this every
    couple of seconds, so a replug reconnects with no bridge restart.

    Deliberately NOT serial-based, unlike the hands' detect_side_port: these
    cards report hard-coded serials (the JMTek dongle "000000000000", the old
    bar "000000010000"), so a serial registry would mis-bind two identical
    devices to each other. VID:PID is the real identity.

    Never raises — an absent speaker must not kill the bridge.
    """
    log = log or (lambda *a, **k: None)
    want = (cfg.get("card") or "").strip()

    by_usb = card_for_usb_id(cfg.get("usb_id") or "")
    if by_usb is not None:
        if want and by_usb != want:
            # The card id moved (ALSA suffixes on collision, e.g. Bar -> Bar_1).
            # The USB device is the robot's speaker either way — follow it.
            log(f"audio: {cfg.get('usb_id')} is card '{by_usb}', not the pinned "
                f"'{want}' — following the device (ALSA renamed or renumbered it).")
        return by_usb

    if want and _card_index(want) is not None:
        return want

    try:
        with open(_CARDS) as fh:
            text = fh.read()
    except OSError:
        return None

    # USB-Audio cards only: Thor's onboard HDA (HDMI out) and APE (the SoC's own
    # audio complex) are always present and are never the speaker bar — claiming
    # one would "succeed" and play into an unconnected HDMI sink.
    usb = [(int(m.group(1)), m.group(2).strip())
           for m in re.finditer(r"^\s*(\d+)\s+\[([^\]]+)\]:\s*USB-Audio", text, re.M)]
    if want:
        if usb:
            log(f"audio: pinned card '{want}' is not loaded, but a USB audio card "
                f"is: {', '.join(c for _, c in usb)}. If that is the speaker bar, "
                f"pin it: MABEL_HW_AUDIO_CARD={usb[0][1]}")
        return None
    if len(usb) == 1:
        log(f"audio: no pinned card — claiming the only USB audio card "
            f"'{usb[0][1]}'. Pin it so this can never mis-bind: "
            f"MABEL_HW_AUDIO_CARD={usb[0][1]}")
        return usb[0][1]
    if len(usb) > 1:
        log(f"audio: {len(usb)} USB audio cards present — AMBIGUOUS, staying "
            f"idle. Pin one (cat /proc/asound/cards): "
            + ", ".join(f"{c}=index {i}" for i, c in usb))
    return None


def device_for_card(template: str, card: str) -> str:
    """Retarget an ALSA device string onto the card we actually resolved.

    The configured string names a card (``plughw:CARD=Bar,DEV=0``) but detection
    may land on a different one — a second bar, or ALSA suffixing the id to
    ``Bar_1`` after a replug. Substituting keeps ONE source of truth for the
    resolved card and stops the driver opening a card nobody detected.

    A string with no CARD= (a raw ``plughw:2,0``, or a user's own alias) is
    returned untouched: an explicit override is a deliberate choice.
    """
    if "CARD=" not in template:
        return template
    return re.sub(r"CARD=[^,]+", f"CARD={card}", template)


class AlsaAudio:
    """One duplex sound card: an arecord pipe, an aplay pipe, and amixer.

    Capture and playback are INDEPENDENT streams — either may be open without the
    other, and either may die without taking the other down. Both are opened
    lazily (`start_capture`/`start_playback`) so an unwatched robot never holds
    the card, and so a speaker-only rig costs nothing.
    """

    def __init__(self, card: str, *, device_playback: str, device_capture: str,
                 rate: int = 16000, channels: int = 1, frame_ms: int = 20,
                 logger=None):
        self.card = card
        self._dev_p = device_playback
        self._dev_c = device_capture
        self.rate = int(rate)
        self.channels = int(channels)
        self.frame_bytes = int(rate * channels * 2 * frame_ms / 1000)
        self._log = logger or (lambda *a, **k: None)
        self._cap: Optional[subprocess.Popen] = None
        self._play: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        for tool in ("arecord", "aplay"):
            if shutil.which(tool) is None:
                raise SubsystemUnavailable(
                    f"{tool} not found — the audio subsystem needs alsa-utils "
                    f"(sudo apt install alsa-utils)")

    # ── capture ───────────────────────────────────────────────────────────────
    def start_capture(self) -> None:
        """Open the mic stream. Idempotent."""
        with self._lock:
            if self._cap is not None and self._cap.poll() is None:
                return
            self._cap = self._spawn(
                ["arecord", "-D", self._dev_c, "-t", "raw", "-f", "S16_LE",
                 "-r", str(self.rate), "-c", str(self.channels),
                 # Period ~= one wire frame: arecord hands us a frame as soon as
                 # the card has one, instead of buffering into added latency.
                 "--period-size", str(self.frame_bytes // (2 * self.channels)),
                 "-q"], stdout=subprocess.PIPE)

    def read_frame(self) -> Optional[bytes]:
        """Exactly one frame of PCM, or None when the stream is gone.

        Blocking + fixed-size: arecord paces us at the card's real 16 kHz, which
        IS our clock. Never size a read by "what's available" — that turns this
        into thousands of tiny reads/s (the RPLIDAR lesson, drivers/rplidar.py).
        """
        cap = self._cap
        if cap is None or cap.stdout is None:
            return None
        buf = cap.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return None                              # EOF: device gone / stopped
        return buf

    def stop_capture(self) -> None:
        with self._lock:
            self._cap = _kill(self._cap)

    def capturing(self) -> bool:
        return self._cap is not None and self._cap.poll() is None

    # ── playback ──────────────────────────────────────────────────────────────
    def start_playback(self) -> None:
        """Open the speaker stream. Idempotent."""
        with self._lock:
            if self._play is not None and self._play.poll() is None:
                return
            self._play = self._spawn(
                ["aplay", "-D", self._dev_p, "-t", "raw", "-f", "S16_LE",
                 "-r", str(self.rate), "-c", str(self.channels), "-q", "-"],
                stdin=subprocess.PIPE)

    def write_frame(self, pcm: bytes) -> bool:
        """Queue one frame for the speaker. False when the stream is gone.

        aplay's own ALSA buffer is the jitter buffer; a short write means the
        card went away mid-frame, which is a reopen, not an error to raise.
        """
        play = self._play
        if play is None or play.stdin is None or play.poll() is not None:
            return False
        try:
            play.stdin.write(pcm)
            play.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            return False

    def stop_playback(self) -> None:
        with self._lock:
            self._play = _kill(self._play)

    def playing(self) -> bool:
        return self._play is not None and self._play.poll() is None

    # ── mixer ─────────────────────────────────────────────────────────────────
    def set_volume(self, control: str, percent: int, capture: bool = False) -> None:
        """Set a simple mixer control and unmute it.

        This is not optional politeness: the old MZ-631M bar booted with 'PCM'
        at 0% and then played perfect, silent audio with a zero exit status. The
        current card's control is 'Speaker' (config.AUDIO mixer_playback).
        """
        if not control:
            return
        pct = max(0, min(100, int(percent)))
        self._amixer(["sset", control, f"{pct}%",
                      "cap" if capture else "unmute"])

    def get_volume(self, control: str) -> Optional[int]:
        """Realized volume % as the CARD reports it, or None if unsupported."""
        if not control:
            return None
        out = self._amixer(["sget", control])
        if out is None:
            return None
        m = re.search(r"\[(\d{1,3})%\]", out)
        return int(m.group(1)) if m else None

    def _amixer(self, args: list) -> Optional[str]:
        try:
            r = subprocess.run(["amixer", "-c", self.card] + args,
                               capture_output=True, text=True, timeout=5.0)
        except Exception as e:                            # noqa: BLE001
            self._log(f"audio: amixer {' '.join(args)} failed: {e}")
            return None
        if r.returncode != 0:
            # A card without this control is normal (HDMI has no 'Mic'); say it
            # once at debug volume rather than failing the subsystem.
            self._log(f"audio: amixer {' '.join(args)} -> rc={r.returncode} "
                      f"{r.stderr.strip()}")
            return None
        return r.stdout

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def close(self) -> None:
        self.stop_capture()
        self.stop_playback()

    def _spawn(self, argv: list, **pipes) -> subprocess.Popen:
        try:
            return subprocess.Popen(argv, stderr=subprocess.DEVNULL,
                                    # Own process group: our cleanup kills the
                                    # pipe without signalling the whole bridge.
                                    start_new_session=True, **pipes)
        except OSError as e:
            raise AudioError(f"cannot start {argv[0]} on {self.card}: {e}") from e


def _kill(proc: Optional[subprocess.Popen]) -> None:
    """Terminate a PCM pipe and reap it. Always returns None (assign it back)."""
    if proc is None:
        return None
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
    except Exception:                                     # noqa: BLE001
        try:
            proc.kill()
        except Exception:                                 # noqa: BLE001
            pass
    for pipe in (proc.stdin, proc.stdout):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:                                 # noqa: BLE001
            pass
    return None

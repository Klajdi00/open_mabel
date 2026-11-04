"""audio.py — the speaker + mic as a typed, ROS-free subsystem.

Single owner of the sound card. Consumers get frames, not devices:

    chunk = audio.latest()             # types.AudioChunk | None  (newest mic frame)
    unsub = audio.on_chunk(cb)         # push per 20 ms capture frame
    audio.play(pcm)                    # queue S16_LE bytes for the speaker

BOTH DIRECTIONS ARE DEMAND-DRIVEN. Nothing opens the card until someone asks:
capture starts on the first `on_chunk` subscriber and stops when the last one
leaves; playback opens on the first `play()` and idles out after
``playback_idle_s``. An unwatched robot holds no PCM and burns no CPU — the same
rule the cloud planes follow, and the reason a `speaker-test` or a calibration
studio can still grab the card when MABEL is not talking.

AUDIO IS A SEQUENCE, NOT A SNAPSHOT. Every other stream in this package is
latest-wins because a late consumer wants the *freshest* pose/cloud. That
reasoning inverts here: dropping the oldest frame of speech is an audible click.
The playback path is therefore a bounded FIFO that, when it overruns, drops the
WHOLE backlog and resyncs (one clean gap) rather than dribbling stale frames
forever. See `_play_run`.

The device story (why plughw is load-bearing, the mixer assertion, why we pin
the card id and not the serial) is in config.AUDIO — read it before changing a
device string.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional

from ..types import AudioChunk, AudioState, monotonic

__all__ = ["AudioSubsystem"]


class AudioSubsystem:
    def __init__(self, cfg: dict, *, driver=None, logger=None,
                 autostart: bool = True):
        self._cfg = cfg
        self._log = logger or (lambda *a, **k: None)
        self._stop = threading.Event()
        self._latest: Optional[AudioChunk] = None
        self._cbs: list = []
        self._seq = 0            # mic frames captured since start
        self._played = 0         # speaker frames written to the card
        self._rms = 0.0
        # A live mic always has a noise floor > 0; EXACT zero means the device is
        # delivering digital silence (hardware mute button, or a dead mic). We flag
        # it only after a sustained run so a brief real gap never trips it, and only
        # while capturing (silence with no capture is just "idle", not "broken").
        self._silent_run = 0            # consecutive exact-silence frames
        self._mic_silent = False
        self._warned_silent = False
        self._muted = False
        self._dev = None                      # AlsaAudio | None (injected or built)
        self._injected = driver is not None
        if driver is not None:
            self._dev = driver
        self._resolved = ""                   # card id we actually opened
        self._dev_p = cfg["device_playback"]   # retargeted onto _resolved
        self._dev_c = cfg["device_capture"]
        self._vol_p: Optional[int] = None     # realized mixer %, not requested
        self._vol_c: Optional[int] = None
        self._thread = None
        self._play_thread = None
        # Bounded FIFO, ~1 s of speech at 20 ms/frame. Deep enough to ride out a
        # scheduler hiccup, shallow enough that a stalled card can never build a
        # minute of delay before anyone notices.
        self._playq: collections.deque = collections.deque(maxlen=50)
        self._play_wake = threading.Event()
        self._last_play = 0.0
        if autostart:
            self.connect()

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mabel-hw-audio")
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._play_wake.set()
        dev = self._dev
        if dev is not None and not self._injected:
            try:
                dev.close()
            except Exception:                             # noqa: BLE001
                pass

    def connected(self) -> bool:
        """The card is present. NOT the same as 'we are capturing' — a speaker
        with nobody listening to the mic is connected and idle, not down."""
        if self._dev is not None and self._injected:
            return True
        return bool(self._resolved)

    # ── read ──────────────────────────────────────────────────────────────────
    def latest(self) -> Optional[AudioChunk]:
        return self._latest

    def on_chunk(self, callback):
        """Subscribe to mic frames. Capture starts on the first subscriber."""
        self._cbs.append(callback)

        def _unsub():
            try:
                self._cbs.remove(callback)
            except ValueError:
                pass
        return _unsub

    def state(self) -> AudioState:
        dev = self._dev
        return AudioState(
            device=self._dev_c if self._resolved else "",
            card=self._resolved or (self._cfg.get("card") or ""),
            connected=self.connected(),
            capturing=bool(dev is not None and dev.capturing()),
            playing=bool(dev is not None and dev.playing()),
            muted=self._muted,
            rate=int(self._cfg["rate"]),
            channels=int(self._cfg["channels"]),
            playback_volume=self._vol_p,
            capture_volume=self._vol_c,
            mic_rms=self._rms,
            mic_silent=bool(self._mic_silent and dev is not None
                            and dev.capturing()),
            stamp=monotonic())

    def health(self) -> dict:
        """Per-direction health for the GUI (camelCase, rounded — house style)."""
        st = self.state()
        # `frames` on BOTH sides is a monotonic counter, not a rate: a consumer
        # sampling health at its own cadence differentiates it to get the true
        # throughput (50/s while talking, 0 while idle). Publishing a rate here
        # would just be this object guessing at the sampler's clock.
        return {"speaker": {"open": st.playing, "volumePct": st.playback_volume,
                            "queuedFrames": len(self._playq),
                            "frames": self._played},
                "mic": {"open": st.capturing, "volumePct": st.capture_volume,
                        "muted": st.muted, "rms": round(st.mic_rms, 4),
                        "silent": st.mic_silent, "frames": self._seq}}

    # ── write (lease-gated at the facade) ─────────────────────────────────────
    def play(self, pcm: bytes) -> None:
        """Queue one frame of S16_LE PCM for the speaker.

        Non-blocking by contract: the caller is a network coroutine or a ROS
        callback and must never stall on the sound card.
        """
        if not pcm:
            return
        self._playq.append(bytes(pcm))
        self._play_wake.set()

    def set_volume(self, percent: int, capture: bool = False) -> None:
        dev = self._dev
        if dev is None:
            return
        ctl = self._cfg["mixer_capture" if capture else "mixer_playback"]
        dev.set_volume(ctl, percent, capture=capture)
        self._refresh_mixer()

    def set_muted(self, muted: bool) -> None:
        """Mute the MIC at our layer: capture keeps running (so liveness and the
        level meter stay honest) but frames are zeroed before they leave."""
        self._muted = bool(muted)

    # ── safety (lease-FREE) ───────────────────────────────────────────────────
    def stop(self) -> None:
        """Silence the speaker NOW: drop the backlog and close the PCM.

        Lease-free like every other stop() in this package — an operator killing
        a stuck loop of audio must never be told to acquire a lease first.
        """
        self._playq.clear()
        dev = self._dev
        if dev is not None:
            try:
                dev.stop_playback()
            except Exception:                             # noqa: BLE001
                pass

    # ── capture thread ────────────────────────────────────────────────────────
    def _run(self):
        while not self._stop.is_set():
            try:
                self._session()
            except Exception as e:                        # noqa: BLE001
                self._log(f"audio error ({e}); retrying in 5 s.")
                self._stop.wait(5.0)

    def _session(self):
        from ..drivers.audio import AlsaAudio, detect_audio_card, device_for_card

        card = detect_audio_card(self._cfg, log=self._log) \
            if not self._injected else self._cfg.get("card", "")
        if card is None:
            self._resolved = ""
            self._stop.wait(2.0)                 # absent; poll for hot-plug
            return
        # Retarget the device strings onto the card DETECTION actually found —
        # the config names one id ('Device', or 'MabelAudio' under the udev
        # rule), but the card may have moved ports and come back suffixed
        # ('Device_1'). Opening the configured name here would open a card
        # nobody resolved (or nothing at all).
        self._dev_p = device_for_card(self._cfg["device_playback"], card)
        self._dev_c = device_for_card(self._cfg["device_capture"], card)
        if not self._injected:
            self._dev = AlsaAudio(
                card,
                device_playback=self._dev_p, device_capture=self._dev_c,
                rate=int(self._cfg["rate"]), channels=int(self._cfg["channels"]),
                frame_ms=int(self._cfg["frame_ms"]), logger=self._log)
        self._resolved = card
        # Assert the mixer BEFORE anyone can play into it (the 0%-volume boot).
        self._dev.set_volume(self._cfg["mixer_playback"],
                             int(self._cfg["volume_playback"]))
        self._dev.set_volume(self._cfg["mixer_capture"],
                             int(self._cfg["volume_capture"]), capture=True)
        self._refresh_mixer()
        self._log(f"audio online: card '{card}' "
                  f"{self._cfg['rate']} Hz x{self._cfg['channels']} S16_LE "
                  f"(speaker {self._dev_p}, "
                  f"mic {self._dev_c}, "
                  f"volume {self._vol_p}%/{self._vol_c}%).")
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(target=self._play_run, daemon=True,
                                                 name="mabel-hw-audio-play")
            self._play_thread.start()
        try:
            self._capture_loop()
        finally:
            if not self._injected:
                try:
                    self._dev.stop_capture()
                except Exception:                         # noqa: BLE001
                    pass

    def _capture_loop(self):
        """Demand-driven capture: open on the first subscriber, close on the last."""
        dev = self._dev
        while not self._stop.is_set():
            if not self._cbs:                    # NO LISTENER: do not hold the mic
                if dev.capturing():
                    dev.stop_capture()
                    self._log("audio: no mic subscribers — capture released.")
                self._stop.wait(0.5)
                continue
            if not dev.capturing():
                dev.start_capture()
                self._seq = 0
                self._silent_run = 0          # re-evaluate silence per session
                self._mic_silent = False
                self._log("audio: mic subscriber — capture started.")
            pcm = dev.read_frame()
            if pcm is None:
                # EOF from arecord: the card went away (or someone else grabbed
                # it). Drop the handle so the outer session re-resolves it — a
                # hot-replug must not need a bridge restart.
                dev.stop_capture()
                if self._cbs:
                    self._log("audio: mic stream ended (device gone or busy) — "
                              "re-resolving in 2 s.")
                self._resolved = ""
                self._stop.wait(2.0)
                return
            self._emit(pcm)

    def _emit(self, pcm: bytes):
        self._rms = _rms(pcm)
        # Sustained EXACT silence => the device is muted/dead (not us). ~2 s at
        # 50 fps before flagging, so a real pause never trips it; any real sample
        # (noise floor) clears it immediately.
        if self._rms <= _SILENCE_EPS:
            self._silent_run += 1
            if self._silent_run >= _SILENCE_FRAMES and not self._mic_silent:
                self._mic_silent = True
                if not self._warned_silent:
                    self._warned_silent = True
                    self._log("⚠ mic is CAPTURING but delivering exact digital "
                              "silence — the card streams frames, every sample is "
                              "zero. Not a software mute (mixer open, verified): "
                              "this is device-level. Check the mic's physical "
                              "wiring/plug on the dongle, and `amixer -c <card> "
                              "sget Mic` cswitch. (History: the old MZ-631M bar's "
                              "vendor DSP GATED the mic below a VAD threshold, "
                              "2026-07-18 — zeros in a quiet room, real audio only "
                              "above a sound threshold, no ALSA control for it.)")
        else:
            self._silent_run = 0
            if self._mic_silent:
                self._mic_silent = False
                self._warned_silent = False
                self._log("mic recovered — real audio flowing again.")
        if self._muted:
            pcm = b"\x00" * len(pcm)
        chunk = AudioChunk(pcm=pcm, seq=self._seq, rate=int(self._cfg["rate"]),
                           channels=int(self._cfg["channels"]), stamp=monotonic())
        self._seq += 1
        self._latest = chunk
        for cb in list(self._cbs):
            try:
                cb(chunk)
            except Exception as e:               # a consumer never kills capture
                self._log(f"audio on_chunk callback failed: {e}")

    # ── playback thread ───────────────────────────────────────────────────────
    def _play_run(self):
        """Drain the FIFO into aplay; idle the PCM out when nobody is talking."""
        idle_s = float(self._cfg.get("playback_idle_s", 2.0))
        while not self._stop.is_set():
            if not self._playq:
                dev = self._dev
                if (dev is not None and dev.playing()
                        and time.monotonic() - self._last_play > idle_s):
                    dev.stop_playback()          # release the card between phrases
                self._play_wake.wait(0.2)
                self._play_wake.clear()
                continue
            dev = self._dev
            if dev is None:
                self._playq.clear()              # no card: never queue forever
                continue
            if not dev.playing():
                dev.start_playback()
            # OVERRUN = RESYNC, not catch-up. A backlog at the cap means the card
            # is slower than the sender; playing it out would put every later
            # frame permanently behind. One clean gap beats unbounded latency.
            if len(self._playq) >= self._playq.maxlen:
                self._playq.clear()
                self._log("audio: speaker backlog overran — dropped it and "
                          "resynced (sender is faster than the card).")
                continue
            try:
                pcm = self._playq.popleft()
            except IndexError:
                continue
            if dev.write_frame(pcm):
                self._played += 1
            else:
                dev.stop_playback()              # reopen on the next frame
            self._last_play = time.monotonic()

    # ── pacing ────────────────────────────────────────────────────────────────
    def service(self) -> None:
        """~1 Hz slow path: keep the realized mixer values honest."""
        if self._dev is not None and self._resolved:
            self._refresh_mixer()

    def _refresh_mixer(self):
        dev = self._dev
        if dev is None:
            return
        self._vol_p = dev.get_volume(self._cfg["mixer_playback"])
        self._vol_c = dev.get_volume(self._cfg["mixer_capture"])


# A frame counts as "silence" only at EXACT zero level (<~0.3 LSB of full scale) —
# a live mic's noise floor is always well above this, so it cleanly separates a
# muted/dead device from a quiet room. ~2 s (50 fps) before flagging, so a genuine
# pause never trips it.
_SILENCE_EPS = 1e-5
_SILENCE_FRAMES = 100


def _rms(pcm: bytes) -> float:
    """Level of one S16_LE frame, 0..1 of full scale.

    stdlib `audioop` is gone in 3.13 and numpy is a lazy import elsewhere in this
    package, so this stays a plain int loop over 320 samples — ~20 us/frame at
    50 frames/s, i.e. 0.1% of one core. Cheap enough to always be honest.
    """
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    total = 0
    for i in range(0, n * 2, 2):
        s = pcm[i] | (pcm[i + 1] << 8)
        if s >= 0x8000:
            s -= 0x10000
        total += s * s
    return min(1.0, (total / n) ** 0.5 / 32768.0)

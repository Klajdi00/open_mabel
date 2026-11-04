#!/usr/bin/env python3
"""exp_audio.py — drive the REAL sound card through AudioSubsystem and report.

The audio counterpart of exp_hw_api.py: it exercises the subsystem against the
physical speaker bar (no ALSA calls of its own) and prints a pass/fail line per
contract clause, so "is the audio stack healthy on this robot?" is one command:

    python3 experiments/exp_audio.py            # full check (plays a short tone)
    python3 experiments/exp_audio.py --quiet    # skip the tone (mic + mixer only)

Run it after touching the audio driver/subsystem, after a device swap, and when
the operator reports "MABEL can't hear/talk" — it separates the three failures
that look identical from the app: no card, card-but-no-mixer (the 0%-volume
boot), and card-but-digital-silence (a device-level mute the software cannot
see). It never moves the robot.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mabel_hw import config                                          # noqa: E402
from mabel_hw.subsystems.audio import AudioSubsystem                 # noqa: E402

PASS, FAIL, INFO = "\033[32m✓\033[0m", "\033[31m✗\033[0m", "\033[2m▸\033[0m"
WARN = "\033[33m⚠\033[0m"


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  {PASS if ok else FAIL} {label}" + (f"   {detail}" if detail else ""))
    return ok


def observe(ok: bool, label: str, detail: str = "") -> None:
    """A check whose failure is HARDWARE, not a regression — warn, never fail.

    Kept distinct from check() so the summary line can't say ALL CHECKS PASSED
    under a red ✗: a muted bar is a true report about the world, and calling it
    a failed check would train the reader to ignore this script's exit code.
    """
    print(f"  {PASS if ok else WARN} {label}" + (f"   {detail}" if detail else ""))


def tone(seq: int, n: int, sr: int, hz: float = 660.0, secs: float = 0.5) -> bytes:
    buf = bytearray()
    for i in range(n):
        t = (seq * n + i) / sr
        env = min(1.0, min(t * 8, (secs - t) * 8)) if t < secs else 0.0
        buf += int(0.25 * env * math.sin(2 * math.pi * hz * t) * 32767) \
            .to_bytes(2, "little", signed=True)
    return bytes(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true",
                    help="skip the speaker tone (mic + mixer checks only)")
    args = ap.parse_args()

    cfg = config.AUDIO
    sub = AudioSubsystem(cfg, logger=lambda m: print(f"  {INFO} {m}"))
    time.sleep(1.5)
    ok = True

    print("\n  device")
    st = sub.state()
    ok &= check(st.connected, "card resolved",
                f"{st.card!r} → {st.device or '(none)'}")
    if not st.connected:
        print("\n  No sound card. Plug the USB audio dongle in, or pin it:\n"
              "    MABEL_HW_AUDIO_CARD=<id from /proc/asound/cards>\n")
        sub.close()
        return 1
    # The old bar booted its PCM control at 0% and then played SILENCE with a
    # zero exit status everywhere — a working device is not an audible one.
    ok &= check(st.playback_volume == int(cfg["volume_playback"]),
                "playback mixer asserted",
                f"{st.playback_volume}% (want {cfg['volume_playback']}%)")
    ok &= check(st.capture_volume == int(cfg["volume_capture"]),
                "capture mixer asserted", f"{st.capture_volume}%")

    print("\n  mic")
    ok &= check(not st.capturing, "idle with no subscriber",
                "the card is free for a studio / speaker-test")
    got: list = []
    unsub = sub.on_chunk(got.append)
    t0 = time.time()
    while len(got) < 25 and time.time() - t0 < 5.0:
        time.sleep(0.02)
    ok &= check(len(got) >= 25, "frames flow on subscribe", f"{len(got)} frames")
    if got:
        rate = len(got) / max(got[-1].stamp - got[0].stamp, 1e-6)
        ok &= check(40 < rate < 60, "wire pacing", f"{rate:.1f} frames/s (want ~50)")
        ok &= check(len(got[0].pcm) == int(cfg["rate"] * cfg["channels"] * 2
                                           * cfg["frame_ms"] / 1000),
                    "frame size", f"{len(got[0].pcm)} B")
        ok &= check([c.seq for c in got[:5]] == [0, 1, 2, 3, 4], "sequence intact")
        peak = max((abs(int.from_bytes(c.pcm[i:i + 2], "little", signed=True))
                    for c in got for i in range(0, len(c.pcm), 2)), default=0)
        # NOT a failure: the pipeline can be perfect while the DEVICE mutes its
        # own mic (physical mute button). Distinguishing the two is the point of
        # this script — silence here with everything above green means hardware.
        observe(peak > 0, "mic hears something",
                f"peak={peak} rms={sub.state().mic_rms:.4f}"
                + ("" if peak else "  ← device sends DIGITAL SILENCE. The stack "
                                   "is fine; the DEVICE is mute: check the mic "
                                   "plug/wiring on the dongle and `amixer -c "
                                   f"{sub.state().card} sget Mic` cswitch."))
    unsub()
    time.sleep(1.0)
    ok &= check(not sub.state().capturing, "card released on last unsubscribe")

    if not args.quiet:
        print("\n  speaker")
        sr = int(cfg["rate"])
        n = int(sr * cfg["frame_ms"] / 1000)
        for k in range(25):
            sub.play(tone(k, n, sr))
        time.sleep(1.2)
        ok &= check(len(sub._playq) == 0, "FIFO drained to the card",
                    "you should have heard a 660 Hz beep")
        t0 = time.time()
        while sub.state().playing and time.time() - t0 < 4.0:
            time.sleep(0.05)
        ok &= check(not sub.state().playing, "card released when idle",
                    f"after {cfg['playback_idle_s']}s")

    print(f"\n  health  {sub.health()}")
    sub.close()
    print(f"\n  {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

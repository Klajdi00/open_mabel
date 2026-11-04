#!/usr/bin/env python3
"""
exp_audio_duplex.py — is the :9610 voice plane ACTUALLY full duplex?

The question this answers is narrow and load-bearing: when a client starts
sending speaker frames UP, does the mic keep coming DOWN — during, and after?
The iOS/visionOS symptom ("press Talk and the robot's room goes silent, and
stays silent after you let go") has two possible homes: the server/HAL, or the
client's own audio stack. This probe is the SERVER-side half of that answer, so
the app fix is not aimed at the wrong layer.

Method — one socket, three phases, one metric (downlink frames/s):

    LISTEN   listen only ........................ baseline mic rate
    DUPLEX   listen + send 50 Hz speaker frames .. must equal the baseline
    AFTER    listen only again .................. must equal the baseline

A full-duplex plane holds ~50 Hz through all three. A drop in DUPLEX means the
sound card or hub is half-duplex; a drop only in AFTER means the uplink left
something wedged.

The uplink payload is a short sine burst at a deliberately LOW amplitude — this
plays out of a real speaker in a real room, so it is a polite test tone, not a
full-scale square wave. `--silent` sends digital zeros instead (still exercises
the whole ingest path: decode, format check, sink, ALSA write) when you would
rather not make noise.

Usage
-----
  python3 server/tools/exp_audio_duplex.py --host 192.168.1.42
  python3 server/tools/exp_audio_duplex.py --host thor.local --secs 6 --silent
  python3 server/tools/exp_audio_duplex.py --host 192.168.1.42 --figure

Exits non-zero if any phase FAILs, so it works as a regression gate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AUDIO_TAG = 0x07
AUDIO_HEADER = 16
CODEC_PCM16 = 0

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<40s} {detail}")


# ── the AUDIO 0x07 record (mirrors server/map_stream.py) ─────────────────────
def encode_audio(pcm: bytes, seq: int, rate: int, channels: int) -> bytes:
    return (struct.pack("<BBHIII", AUDIO_TAG, CODEC_PCM16, channels,
                        seq & 0xFFFFFFFF, rate, len(pcm)) + pcm)


def decode_audio(raw: bytes):
    if len(raw) < AUDIO_HEADER or raw[0] != AUDIO_TAG:
        return None
    tag, codec, ch, seq, rate, n = struct.unpack("<BBHIII", raw[:AUDIO_HEADER])
    if len(raw) < AUDIO_HEADER + n:
        return None
    return raw[AUDIO_HEADER:AUDIO_HEADER + n], seq, rate, ch, codec


def rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    acc = 0
    for i in range(0, n * 2, 2):
        s = pcm[i] | (pcm[i + 1] << 8)
        if s >= 0x8000:
            s -= 0x10000
        acc += s * s
    return min(1.0, (acc / n) ** 0.5 / 32768.0)


def tone(samples: int, rate: int, phase: float, hz: float = 440.0,
         amp: float = 0.12) -> tuple[bytes, float]:
    """A low-amplitude sine burst, continuous across calls via `phase`.

    Continuity matters: restarting the phase every 20 ms frame puts a step
    discontinuity in the waveform 50×/s, which is a buzz on top of the tone and
    would make "does it sound right out of the bar?" unanswerable.
    """
    out = bytearray()
    step = 2.0 * math.pi * hz / rate
    for i in range(samples):
        v = int(amp * 32767 * math.sin(phase + i * step))
        out += struct.pack("<h", v)
    return bytes(out), (phase + samples * step) % (2.0 * math.pi)


# ── the probe ────────────────────────────────────────────────────────────────
async def run(host: str, port: int, secs: float, token: str,
              silent: bool) -> dict:
    import websockets

    url = f"ws://{host}:{port}/?client=probe"
    if token:
        url += f"&key={token}"
    print(f"\n:{port} voice plane  →  {url}\n")

    phases = [("LISTEN", False), ("DUPLEX", True), ("AFTER", False)]
    report: dict = {"phases": {}, "hello": None, "gaps": 0, "trace": []}

    async with websockets.connect(url, ping_interval=None,
                                  max_size=4 << 20) as ws:
        # hello carries the format we MUST send in (the hub refuses anything
        # else rather than resampling it).
        rate, channels, frame_ms = 16000, 1, 20
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=5.0)
        except asyncio.TimeoutError:
            check("hello arrives", False, "no hello within 5 s")
            return report
        if isinstance(first, str):
            j = json.loads(first)
            report["hello"] = j
            if j.get("type") == "notice":
                check("rig has a sound card", False, j.get("text", "notice"))
                return report
            rate = int(j.get("rate", rate))
            channels = int(j.get("channels", channels))
            frame_ms = int(j.get("frameMs", frame_ms))
            check("hello arrives", True,
                  f"{rate} Hz x{channels} {frame_ms} ms duplex={j.get('duplex')}")
        else:
            check("hello arrives", False, "first frame was binary")
            return report

        frame_samples = rate * frame_ms // 1000
        tx_seq = 0
        phase = 0.0
        last_rx_seq = None

        for label, talking in phases:
            t0 = time.monotonic()
            rx = 0
            tx = 0
            levels: list[float] = []
            next_tx = t0
            print(f"  {label:<7s} {'listen + TALK' if talking else 'listen only'}"
                  f" for {secs:.0f} s …")
            while time.monotonic() - t0 < secs:
                # Uplink at the wire's own 50 Hz cadence.
                now = time.monotonic()
                if talking and now >= next_tx:
                    if silent:
                        pcm = b"\x00" * (frame_samples * channels * 2)
                    else:
                        pcm, phase = tone(frame_samples * channels, rate, phase)
                    await ws.send(encode_audio(pcm, tx_seq, rate, channels))
                    tx_seq += 1
                    tx += 1
                    next_tx += frame_ms / 1000.0
                    if next_tx < now:            # never spin a backlog
                        next_tx = now + frame_ms / 1000.0
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.005)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, str):
                    continue
                got = decode_audio(raw)
                if got is None:
                    continue
                pcm, seq, _r, _c, _codec = got
                if last_rx_seq is not None and seq != (last_rx_seq + 1) & 0xFFFFFFFF:
                    report["gaps"] += 1
                last_rx_seq = seq
                rx += 1
                levels.append(rms(pcm))

            dt = time.monotonic() - t0
            report["phases"][label] = {
                "rxHz": rx / dt, "txHz": tx / dt, "rx": rx, "tx": tx,
                "secs": dt,
                "rmsMean": statistics.fmean(levels) if levels else 0.0,
                "rmsMax": max(levels) if levels else 0.0,
            }
            report["trace"].append((label, levels))
            p = report["phases"][label]
            print(f"          rx {p['rxHz']:6.1f} Hz   tx {p['txHz']:5.1f} Hz   "
                  f"mic rms mean {p['rmsMean']:.4f} max {p['rmsMax']:.4f}")

    return report


def verdict(report: dict, secs: float) -> None:
    ph = report.get("phases", {})
    if not ph:
        return
    base = ph.get("LISTEN", {}).get("rxHz", 0.0)
    print()
    check("downlink alive while listening", base > 5.0, f"{base:.1f} Hz")

    for label in ("DUPLEX", "AFTER"):
        got = ph.get(label, {}).get("rxHz", 0.0)
        # 60% of baseline is a generous floor: it passes normal jitter and fails
        # the symptom under test (downlink going to ZERO).
        ok = base > 5.0 and got >= 0.6 * base
        check(f"downlink survives {label}", ok,
              f"{got:.1f} Hz vs {base:.1f} Hz baseline "
              f"({100 * got / base if base else 0:.0f}%)")

    up = ph.get("DUPLEX", {}).get("txHz", 0.0)
    check("uplink ran at the wire rate", up >= 40.0, f"{up:.1f} Hz")
    check("no downlink seq gaps", report.get("gaps", 0) == 0,
          f"{report.get('gaps', 0)} gaps")


def figure(report: dict, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x0 = 0
    for label, levels in report.get("trace", []):
        xs = list(range(x0, x0 + len(levels)))
        ax.plot(xs, levels, lw=0.9, label=label)
        if x0:
            ax.axvline(x0, color="0.7", lw=0.8, ls="--")
        x0 += len(levels)
    ax.set_xlabel("downlink mic frame #")
    ax.set_ylabel("frame RMS (0…1)")
    ax.set_title(":9610 downlink through a talk cycle")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    print(f"\n  figure → {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9610)
    ap.add_argument("--secs", type=float, default=5.0,
                    help="seconds per phase (default 5)")
    ap.add_argument("--token", default=os.getenv("MABEL_TOKEN", ""))
    ap.add_argument("--silent", action="store_true",
                    help="send digital zeros instead of a test tone")
    ap.add_argument("--figure", action="store_true",
                    help="write docs/figures/audio_duplex.pdf")
    a = ap.parse_args()

    report = asyncio.run(run(a.host, a.port, a.secs, a.token, a.silent))
    verdict(report, a.secs)

    if a.figure and report.get("trace"):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        figure(report, os.path.join(here, "docs", "figures", "audio_duplex.pdf"))

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed"
          + (f" — FAILED: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

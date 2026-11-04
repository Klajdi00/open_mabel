"""The ZED's side-by-side 'head_stereo' frame carries TWO eyes, so it is pushed at
2 x net_width — each eye keeps the width every other stream gets (2026-10-16).
A push without `width` must keep the old behaviour (net_width)."""
import cv2
import numpy as np

from mabel_hw.cameras.video_pipe import PipeVideo


class _Sock:
    """Stand-in for the unix DGRAM socket (a real socket's sendto is read-only)."""
    def __init__(self):
        self.sent = []

    def sendto(self, pkt, path):
        self.sent.append(pkt)


def _capture(pv, monkeypatch):
    stub = _Sock()
    monkeypatch.setattr(pv, "_sock", stub)
    return stub.sent


def _jpeg_width(pkt):
    n = pkt[0]
    jpeg = pkt[1 + n + 8:]
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[1]


def test_stereo_frame_keeps_per_eye_net_width(monkeypatch):
    pv = PipeVideo(net_width=640, jpeg_quality=60)
    sent = _capture(pv, monkeypatch)
    sbs = np.zeros((720, 2560, 3), np.uint8)              # 1280x720 per eye
    pv.push(("head_stereo",), sbs, width=1280)
    assert _jpeg_width(sent[-1]) == 1280                  # = 2 x 640: each eye 640


def test_default_push_is_unchanged(monkeypatch):
    pv = PipeVideo(net_width=640, jpeg_quality=60)
    sent = _capture(pv, monkeypatch)
    pv.push(("main",), np.zeros((720, 1280, 3), np.uint8))
    assert _jpeg_width(sent[-1]) == 640


def test_one_encode_serves_every_name(monkeypatch):
    pv = PipeVideo(net_width=640, jpeg_quality=60)
    sent = _capture(pv, monkeypatch)
    pv.push(("head_stereo", "head_sbs"), np.zeros((720, 2560, 3), np.uint8), width=1280)
    names = [p[1:1 + p[0]].decode() for p in sent]
    assert names == ["head_stereo", "head_sbs"]
    assert sent[0][1 + sent[0][0] + 8:] == sent[1][1 + sent[1][0] + 8:]   # same JPEG bytes

"""MJPEG single-frame extractor."""
from control_center.server.video_proxy import extract_jpeg, SOI, EOI


def _jpeg(payload=b"x" * 50):
    return SOI + payload + EOI


def test_extracts_first_complete_jpeg():
    frame = _jpeg()
    stream = [b"--frame\r\nContent-Type: image/jpeg\r\n\r\n", frame[:20],
              frame[20:], b"\r\n--frame..."]
    assert extract_jpeg(stream) == frame


def test_skips_partial_then_finds_next():
    f1 = _jpeg(b"a" * 30)
    # stream starts mid-jpeg (no SOI) then carries a full one
    stream = [b"leftover-bytes" + EOI, b"junk", f1]
    assert extract_jpeg(stream) == f1


def test_none_when_no_jpeg():
    assert extract_jpeg([b"no image here", b"still nothing"]) is None


def test_bounded_memory_still_finds_late_frame():
    big = [b"\x00" * 1_000_000 for _ in range(5)]      # 5 MB of noise > cap
    frame = _jpeg(b"z" * 100)
    assert extract_jpeg(big + [frame]) == frame

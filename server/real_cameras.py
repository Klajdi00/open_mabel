"""
real_cameras.py — real USB camera sources for the MABEL teleop pipeline.

These replace the MuJoCo-rendered cameras (sim_cameras.py) with the operator's
actual hardware, while feeding the SAME PushCapture fan-out — so the MJPEG/UDP/
H.264 machinery and the Vision Pro contract (the /camera/* paths) are unchanged;
only the pixels become real.

Two cameras, both USB on the Mac (not yet on thor):

  · HEAD — a "3D USB Camera" that outputs ONE hardware-stitched side-by-side
    stereo frame (1280×480 → two 640×480 eyes). We split it down the middle,
    optionally rectify+undistort each eye with its calibration, and push:
        left  half → /camera/head_left/stream.mjpg   (+ /camera/main = left mono)
        right half → /camera/head_right/stream.mjpg

  · WRIST — a "Global Shutter Camera" (mono), optionally undistorted, pushed to
    the wrist path(s).

Calibration comes from the ROS camera_info YAMLs already in the workspace
(ros/src/mabel_sensors/config): head_left_cam.yaml, head_right_cam.yaml,
right_wrist_cam.yaml. Each carries K (camera_matrix), D (distortion), R
(rectification) and P (projection). For the head eyes we use R+P directly — that
is exactly the stereo rectification that row-aligns the two views for a
comfortable headset image. For the mono wrist we undistort with K,D into an
optimal new camera matrix (the stereo P there is not meant for mono use).
"""
from __future__ import annotations

import glob
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

# Where the calibration YAMLs live (relative to the repo root, ../ros).
_HERE = os.path.abspath(os.path.dirname(__file__))
CALIB_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "ros", "src", "mabel_sensors", "config"))

# The MJPEG paths the apps pull (kept identical to sim_cameras.py).
HEAD_LEFT_PATH  = "/camera/head_left/stream.mjpg"
HEAD_RIGHT_PATH = "/camera/head_right/stream.mjpg"
HEAD_MAIN_PATH  = "/camera/main/stream.mjpg"          # mono fallback = left eye
# The headset's stereo source: RECTIFIED left|right halves of ONE stitched frame
# side by side — same exposure by construction (unlike two eye streams, which the
# client could pair a frame apart). Not the :9620 'head_sbs', which is the RAW,
# un-rectified stitched frame (vertical misalignment in a headset).
HEAD_STEREO_PATH = "/camera/head_stereo/stream.mjpg"
WRIST_LEFT_PATH  = "/camera/wrist_left/stream.mjpg"
WRIST_RIGHT_PATH = "/camera/wrist_right/stream.mjpg"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CamCalib:
    """One pinhole calibration parsed from a ROS camera_info YAML."""
    name: str
    width: int
    height: int
    K: np.ndarray            # 3×3 intrinsic
    D: np.ndarray            # 1×5 distortion (plumb_bob)
    R: np.ndarray            # 3×3 rectification
    P: np.ndarray            # 3×4 projection (rectified)

    @classmethod
    def load(cls, path: str) -> "CamCalib":
        import yaml
        with open(path) as f:
            y = yaml.safe_load(f)

        def mat(key, r, c):
            return np.array(y[key]["data"], dtype=np.float64).reshape(r, c)

        return cls(
            name=y.get("camera_name", os.path.basename(path)),
            width=int(y["image_width"]), height=int(y["image_height"]),
            K=mat("camera_matrix", 3, 3),
            D=np.array(y["distortion_coefficients"]["data"], dtype=np.float64).reshape(1, -1),
            R=mat("rectification_matrix", 3, 3),
            P=mat("projection_matrix", 3, 4),
        )

    def scaled_to(self, out_w: int, out_h: int) -> "CamCalib":
        """Return this calibration rescaled to a different image size.

        The camera was calibrated at one resolution (640×480) but the sensor can
        stream at a higher one (e.g. 1280×720 per eye). Intrinsics scale per-axis:
        fx,cx by sx = out_w/width and fy,cy by sy = out_h/height (handles a 4:3 →
        16:9 mode change correctly). Distortion D is in normalized coords, so it is
        resolution-independent; R is a pure rotation, unchanged; P scales like K
        (its baseline term P[0,3] scales with sx, consistent with fx)."""
        if (out_w, out_h) == (self.width, self.height):
            return self
        sx, sy = out_w / self.width, out_h / self.height
        K = self.K.copy(); K[0] *= sx; K[1] *= sy
        P = self.P.copy(); P[0] *= sx; P[1] *= sy
        return CamCalib(self.name, out_w, out_h, K, self.D, self.R, P)

    def stereo_rectify_maps(self, out_size: tuple[int, int]):
        """Undistort + stereo-rectify maps (uses R and P → row-aligned eyes)."""
        return cv2.initUndistortRectifyMap(
            self.K, self.D, self.R, self.P, out_size, cv2.CV_16SC2)

    def mono_undistort_maps(self, out_size: tuple[int, int], alpha: float = 0.0):
        """Undistort maps for a mono camera into an optimal new K (alpha=0 hides
        the black border from the strong barrel distortion; 1 keeps all pixels)."""
        newK, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (self.width, self.height), alpha, out_size)
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.K, self.D, None, newK, out_size, cv2.CV_16SC2)
        return mapx, mapy


def load_calib(filename: str) -> Optional[CamCalib]:
    """Load a calibration YAML by name from CALIB_DIR; None if missing/unreadable."""
    path = os.path.join(CALIB_DIR, filename)
    try:
        return CamCalib.load(path)
    except Exception as e:           # noqa: BLE001 — calibration is optional
        print(f"[real-cam] calibration '{filename}' unavailable ({e}); "
              f"streaming raw (no rectify)")
        return None


def head_fov() -> Optional[tuple]:
    """The head stereo camera's TRUE field of view, from the rectified projection
    (head_left_cam.yaml P) — what the Vision Pro panel must subtend so the world
    is at real-life angular scale. Returns (fovy_deg, aspect, hfov_deg), where:

        fovy  = 2·atan((H/2)/fy)      vertical FOV
        hfov  = 2·atan((W/2)/fx)      horizontal FOV
        aspect = tan(hfov/2)/tan(fovy/2)   the TANGENT-FOV ratio the panel wants
                 (NOT the pixel W/H — that is only equal when fx==fy)

    Resolution-independent: scaling the calibration with the capture size scales
    fx,fy with W,H, leaving these angles unchanged. Both rectified eyes share the
    same P (stereo rectification), so the left eye defines it."""
    cal = load_calib("head_left_cam.yaml")
    if cal is None:
        return None
    fx, fy = float(cal.P[0, 0]), float(cal.P[1, 1])
    w, h = cal.width, cal.height
    fovy = math.degrees(2 * math.atan((h / 2) / fy))
    hfov = math.degrees(2 * math.atan((w / 2) / fx))
    aspect = math.tan(math.radians(hfov / 2)) / math.tan(math.radians(fovy / 2))
    return fovy, aspect, hfov


# ---------------------------------------------------------------------------
# Device discovery (resolve the two cameras by their resolution signature)
# ---------------------------------------------------------------------------

@dataclass
class Devices:
    head: Optional[int]
    wrist_right: Optional[int]
    wrist_left: Optional[int]


def _probe_index(i: int) -> Optional[dict]:
    """Open one OpenCV index and report its resolution AND whether it actually
    delivers frames — some devices (e.g. a 2nd identical UVC cam that loses the
    USB-bandwidth race on macOS) open and report a size but never stream."""
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release()
        return None
    dw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    dh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Quick liveness: a few reads to confirm it isn't a stone-dead index. (A 2nd
    # identical UVC cam can pass this yet starve once the high-bandwidth head is
    # also streaming — that USB-bandwidth case is handled at runtime by the
    # stall→placeholder, not here, so the probe stays fast.)
    good = 0
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None and f.size > 0:
            good += 1
        time.sleep(0.03)
    live = good >= 2
    # Ask for a wide stitched-stereo mode; only the 3D head cam honors it.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ww = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return {"index": i, "dw": dw, "dh": dh, "wide_w": ww, "live": live}


def _co_stream_ok(a: int, b: int) -> bool:
    """True if device `b` still delivers frames while `a` is also open. Two
    identical UVC global-shutter cams on one USB controller often CAN'T — only
    one streams at a time — which we must detect before assigning both wrists."""
    ca = cv2.VideoCapture(a)
    cb = cv2.VideoCapture(b)
    time.sleep(0.8)
    gb = 0
    for _ in range(15):
        ca.read()
        ok, f = cb.read()
        if ok and f is not None and f.size > 0:
            gb += 1
        time.sleep(0.03)
    ca.release(); cb.release()
    time.sleep(0.3)
    return gb >= 3


def resolve_devices(head_arg="auto", wrist_right_arg="auto", wrist_left_arg="auto",
                    swap_wrists: bool = False, max_probe: int = 3) -> Devices:
    """Map the head + the two wrist cameras to OpenCV indices.

    Signatures on this Mac (verified): the head "3D USB Camera" is the only device
    that returns a stitched ≥3:1 wide frame (2560×720); the two "Global Shutter
    Camera"s default to 1280×720. The built-in FaceTime / iPhone continuity
    cameras are 1920×1080 and are skipped. The two global-shutter cams are
    identical, so they're assigned by index order (lowest = right, next = left);
    pass --swap-wrists or explicit indices if mirrored. Override any with an int
    (or env MABEL_CAM_HEAD / MABEL_CAM_WRIST_RIGHT / MABEL_CAM_WRIST_LEFT).
    """
    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    head = as_int(head_arg)
    wr = as_int(wrist_right_arg)
    wl = as_int(wrist_left_arg)
    if None not in (head, wr, wl):
        return Devices(head, wr, wl)

    # Probe a bounded range and STOP at the first index that won't open, so we
    # never touch the slow FaceTime / iPhone-continuity cameras beyond the USB
    # cams (which would add seconds to startup).
    probes = []
    for i in range(max_probe):
        p = _probe_index(i)
        if p is None:
            break
        probes.append(p)
    # Head = widest stitched device (wide_w == 2560, i.e. side-by-side stereo).
    if head is None:
        wide = [p for p in probes if p["wide_w"] >= 2560 and p["live"]]
        head = wide[0]["index"] if wide else None
    # Wrist candidates = global-shutter cams (≤1280×720, not the head, not a
    # 1080p built-in/continuity camera) that ACTUALLY STREAM — a 2nd identical
    # UVC cam that opens but never yields a frame is excluded so we don't leave a
    # permanently black tile.
    cand = [p["index"] for p in probes
            if p["index"] != head and p["live"] and p["dw"] <= 1280 and p["dh"] <= 720]
    dead = [p["index"] for p in probes
            if p["index"] != head and not p["live"] and p["dw"] <= 1280 and p["dh"] <= 720]
    if dead:
        print(f"[real-cam] note: camera index(es) {dead} open but deliver NO frames "
              f"(USB bandwidth / duplicate device on macOS) — ignored. Try a different "
              f"USB port/controller for the 2nd wrist cam, or set MABEL_CAM_WRIST_LEFT.")
    if swap_wrists:
        cand = list(reversed(cand))
    # First candidate → right wrist, second → left wrist (each overridable).
    auto_wl = wrist_left_arg in (None, "auto")
    if wr is None:
        wr = cand[0] if len(cand) >= 1 else None
    if wl is None:
        wl = next((c for c in cand if c != wr), None)
    # Two identical USB global-shutter cams usually can't co-stream (one bus,
    # macOS serves one at a time). If the AUTO-picked left wrist can't deliver
    # while the right is open, drop it → the server mirrors the right into both
    # tiles (live video everywhere) instead of a dead/placeholder left tile.
    if auto_wl and wl is not None and wr is not None and wl != wr:
        if not _co_stream_ok(wr, wl):
            print(f"[real-cam] the two wrist cams (idx {wr} & {wl}) can't stream at the "
                  f"same time — identical USB global-shutter cams on one controller, so "
                  f"macOS serves only one. Using idx {wr} for BOTH wrist tiles. For a real "
                  f"LEFT view, plug the 2nd wrist cam into a SEPARATE USB controller "
                  f"(another port/bus, or a powered hub), then it auto-detects "
                  f"(or force it: --wrist-left-device {wl} / MABEL_CAM_WRIST_LEFT={wl}).")
            wl = None
    return Devices(head, wr, wl)


# ---------------------------------------------------------------------------
# Capture threads (push JPEG bytes into the sink callbacks)
# ---------------------------------------------------------------------------

JpegSink = Callable[[bytes], None]


def _placeholder_jpeg(text: str, w: int = 640, h: int = 480, quality: int = 60) -> bytes:
    img = np.zeros((h, w, 3), dtype="uint8")
    cv2.putText(img, text, (18, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2)
    cv2.putText(img, "Waiting for camera…", (18, h // 2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


class _BaseCamera:
    """Common open / reconnect / rate-limit loop. Subclasses implement _process()."""

    def __init__(self, name: str, device: int, width: int, height: int,
                 fps: int = 30, jpeg_quality: int = 80, flip180: bool = False):
        self.name = name
        self.device = device
        self.req_w, self.req_h = width, height
        self.fps = max(1, fps)
        self.encode = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        self.flip180 = flip180
        self.last_error: Optional[str] = "no frames yet"
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sinks: dict[str, JpegSink] = {}

    def set_sink(self, key: str, sink: JpegSink) -> None:
        self._sinks[key] = sink

    def _emit(self, key: str, jpeg: bytes) -> None:
        sink = self._sinks.get(key)
        if sink is not None:
            sink(jpeg)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"realcam-{self.name}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            self.last_error = f"cannot open device {self.device!r}"
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        # Ask the sensor for our target rate — V4L2 otherwise often defaults to a
        # lower fps mode (e.g. 30 when the camera can do 60), which directly sets
        # the floor on frame-interval latency.
        try:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        except Exception:
            pass
        # Small internal buffer → always the freshest frame (low latency).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[real-cam] {self.name:<12} device={self.device} opened {aw}×{ah}")
        self.last_error = None
        return cap

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        cap: Optional[cv2.VideoCapture] = None
        last = 0.0
        fails = 0
        opened_at = 0.0
        last_good = 0.0
        stalled_ph = False
        while self._running:
            if cap is None or not cap.isOpened():
                cap = self._open()
                if cap is None:
                    self._broadcast_placeholder()
                    time.sleep(min(2.0 * (fails + 1), 10.0)); fails += 1
                    continue
                fails = 0
                opened_at = last_good = time.monotonic()
                stalled_ph = False
            now = time.monotonic()
            if now - last < interval:
                time.sleep(0.001); continue
            last = now
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                # A camera that opened but never delivers (a 2nd identical UVC cam
                # losing the USB race on macOS) would otherwise hang its tile —
                # serve a placeholder so the stream stays alive + visibly waiting.
                if not stalled_ph and now - last_good > 2.0:
                    self.last_error = "opened but no frames (USB bandwidth / duplicate device)"
                    self._broadcast_placeholder()
                    stalled_ph = True
                if fails >= 5:                       # give up + reopen after sustained failure
                    cap.release(); cap = None
                    time.sleep(min(0.3 * fails, 3.0))
                else:
                    time.sleep(0.05)
                continue
            fails = 0
            last_good = now
            stalled_ph = False
            if self.flip180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            try:
                self._process(frame)
            except Exception as e:            # noqa: BLE001
                self.last_error = f"process error: {e}"
        if cap:
            cap.release()

    def _jpeg(self, bgr: np.ndarray) -> Optional[bytes]:
        ok, buf = cv2.imencode(".jpg", bgr, self.encode)
        return buf.tobytes() if ok else None

    # subclasses override:
    def _process(self, frame: np.ndarray) -> None: ...
    def _broadcast_placeholder(self) -> None: ...


class StereoHeadCamera(_BaseCamera):
    """The stitched 3D USB head cam: split → (rectify) → left/right/main JPEGs.

    Sink keys: 'left', 'right', 'main'.
    """

    def __init__(self, device: int, *, stitched_w: int = 3200, stitched_h: int = 1200,
                 out_w: int = 0, out_h: int = 0,
                 rectify: bool = True, fps: int = 30, jpeg_quality: int = 80,
                 flip180: bool = False):
        super().__init__("head", device, stitched_w, stitched_h, fps, jpeg_quality, flip180)
        self.eye_w = stitched_w // 2
        self.eye_h = stitched_h
        # Demand gate for the optional 'sbs' sink (set by real_camera_server when
        # the :9620 dedicated stream is on): None = always emit when sink wired.
        self.wants_sbs = None
        # Streamed per-eye output size. We CAPTURE + rectify at full resolution
        # (sharp, correct geometry) then AREA-AVERAGE down to this size before
        # JPEG — supersampling anti-aliasing. The Vision Pro panel samples the
        # texture at roughly this density; sending a much denser texture makes
        # high-frequency detail (text) alias into rainbow moiré on the curved
        # panel (bilinear, no mipmaps). 0 = stream native (can shimmer). The FOV
        # is unchanged by this resize, so real-life scale is preserved.
        self.out_w = out_w if out_w > 0 else self.eye_w
        self.out_h = out_h if out_h > 0 else self.eye_h
        self._maps = None
        if rectify:
            left = load_calib("head_left_cam.yaml")
            right = load_calib("head_right_cam.yaml")
            if left and right:
                # Rectify each eye at its FULL split resolution: scale the (640×480)
                # calibration up to the actual eye size so we keep every pixel
                # instead of downscaling to the calibration size.
                self._maps = {
                    "left":  left.scaled_to(self.eye_w, self.eye_h)
                                 .stereo_rectify_maps((self.eye_w, self.eye_h)),
                    "right": right.scaled_to(self.eye_w, self.eye_h)
                                 .stereo_rectify_maps((self.eye_w, self.eye_h)),
                }
                print(f"[real-cam] head stereo rectification ON @ {self.eye_w}×{self.eye_h}/eye "
                      f"(calibration {left.width}×{left.height} scaled up)")
            else:
                print("[real-cam] head rectification OFF (calibration missing) — raw split")

    def _eye(self, frame: np.ndarray, side: str) -> np.ndarray:
        half = frame[:, :self.eye_w] if side == "left" else frame[:, self.eye_w:]
        if self._maps is not None:
            # Maps are built at the eye size; guard against a mode that returned a
            # different split width than requested.
            if (half.shape[1], half.shape[0]) != (self.eye_w, self.eye_h):
                half = cv2.resize(half, (self.eye_w, self.eye_h))
            mapx, mapy = self._maps[side]
            half = cv2.remap(half, mapx, mapy, cv2.INTER_LINEAR)
        # Supersample down to the streamed size with area averaging (anti-alias).
        if (half.shape[1], half.shape[0]) != (self.out_w, self.out_h):
            half = cv2.resize(half, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return half

    def _process(self, frame: np.ndarray) -> None:
        # Optional raw side-by-side tap (the :9620 hi-res dedicated stream): the
        # native hardware-stitched frame, pre-split/pre-rectify — full detail,
        # no resample. Demand-gated by wants_sbs so the extra full-res JPEG
        # encode only happens while a :9620 client is actually connected.
        sbs_sink = self._sinks.get("sbs")
        if sbs_sink is not None and (self.wants_sbs is None or self.wants_sbs()):
            js = self._jpeg(frame)
            if js:
                self._emit("sbs", js)
        left = self._eye(frame, "left")
        right = self._eye(frame, "right")
        if "stereo" in self._sinks and left.shape[0] == right.shape[0]:
            js2 = self._jpeg(np.hstack((left, right)))
            if js2:
                self._emit("stereo", js2)
        jl, jr = self._jpeg(left), self._jpeg(right)
        if jl:
            self._emit("left", jl)
            self._emit("main", jl)         # mono panel = left eye
        if jr:
            self._emit("right", jr)

    def _broadcast_placeholder(self) -> None:
        ph = _placeholder_jpeg("Head camera", self.out_w, self.out_h)
        for k in ("left", "right", "main"):
            self._emit(k, ph)


class MonoCamera(_BaseCamera):
    """A mono USB cam (the wrist global-shutter): optional undistort → JPEG.

    Sink keys: every key passed in `sink_keys` receives the same frame (so one
    physical wrist cam can fill both wrist tiles while we have a single one).
    """

    def __init__(self, device: int, calib_file: Optional[str], *, out_w: int = 1280,
                 out_h: int = 720, undistort: bool = True, fps: int = 30,
                 jpeg_quality: int = 80, flip180: bool = False,
                 sink_keys: tuple[str, ...] = ("a",), label: str = "wrist"):
        super().__init__(label, device, out_w, out_h, fps, jpeg_quality, flip180)
        self.out_w, self.out_h = out_w, out_h
        self.sink_keys = sink_keys
        self._maps = None
        if undistort and calib_file:
            cal = load_calib(calib_file)
            if cal:
                # Undistort at the FULL output resolution: scale the (640×480)
                # calibration up to the capture size first so no pixels are lost.
                self._maps = cal.scaled_to(out_w, out_h).mono_undistort_maps(
                    (out_w, out_h), alpha=0.0)
                print(f"[real-cam] {label} undistortion ON @ {out_w}×{out_h} "
                      f"(from {calib_file})")

    def _process(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != (self.out_w, self.out_h):
            frame = cv2.resize(frame, (self.out_w, self.out_h))
        if self._maps is not None:
            frame = cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        jpeg = self._jpeg(frame)
        if jpeg:
            for k in self.sink_keys:
                self._emit(k, jpeg)

    def _broadcast_placeholder(self) -> None:
        ph = _placeholder_jpeg("Wrist camera", self.out_w, self.out_h)
        for k in self.sink_keys:
            self._emit(k, ph)

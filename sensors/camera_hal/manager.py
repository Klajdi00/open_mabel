"""CameraManager -- bring up every available camera and grab them concurrently.

Each camera gets its own daemon thread running a tight read() loop, so a slow or
absent device never stalls the others. Consumers (the viewer) just ask for the
`latest()` frame per camera. This is the "dedicated hardware bring-up" layer; it
has no ROS and no GUI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .base import CameraBase, CameraError
from .frame import CameraFrame


@dataclass
class CameraStatus:
    name: str
    model: str
    state: str = "init"        # init | waiting | running | unavailable | error
    detail: str = ""           # hint / error text for the dashboard
    fps: float = 0.0
    frames: int = 0
    latest: CameraFrame | None = field(default=None, repr=False)


class _Worker(threading.Thread):
    """Supervises ONE camera forever: (re)connect -> stream -> on any drop, fall
    back to retrying open(). A camera that is absent at start-up -- or unplugged
    and re-plugged while running -- is picked up automatically, with no need to
    restart the node. Consumers see it come and go purely through `status`."""

    #: seconds to wait between (re)connection attempts (absent / failed device)
    RECONNECT_INTERVAL = 2.0
    #: if no frame arrives for this long while "running", treat the device as gone
    #: and drop back to the reconnect loop (covers hot-unplug AND a wedged SDK)
    STALL_TIMEOUT = 3.0

    def __init__(self, cam: CameraBase):
        super().__init__(name=f"cam-{cam.name}", daemon=True)
        self.cam = cam
        self.status = CameraStatus(name=cam.name, model=cam.model)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self.status, k, v)

    def _try_open(self) -> bool:
        """One connection attempt. True once the device is open and streaming."""
        # Cheap presence probe FIRST so an absent camera stays quiet -- no noisy
        # SDK open() failure logged every couple of seconds while we wait for it.
        try:
            present = self.cam.is_present()
        except Exception:                           # noqa: BLE001
            present = True                          # probe blew up -> let open() decide
        if not present:
            self._set(state="waiting", detail="device not connected — waiting…",
                      fps=0.0)
            return False
        try:
            self.cam.open()
        except CameraError as e:
            self._set(state="unavailable", detail=e.hint or str(e), fps=0.0)
            return False
        except Exception as e:                      # noqa: BLE001 - surface anything
            self._set(state="error", detail=str(e), fps=0.0)
            return False
        self._set(state="running",
                  detail=self.cam.intrinsics.pretty() if self.cam.intrinsics else "")
        return True

    def _stream(self) -> None:
        """Pump frames until the device stalls/errors or we're asked to stop."""
        t_prev = time.monotonic()
        t_last_frame = t_prev
        ema = 0.0
        while not self._stop_evt.is_set():
            try:
                frame = self.cam.read()
            except Exception as e:                  # noqa: BLE001 - read blew up == gone
                self._set(detail=f"read error: {e}")
                return
            now = time.monotonic()
            if frame is None:
                # A transient miss is normal; a persistent one means the device
                # has dropped -> bail out so run() can reconnect it.
                if now - t_last_frame > self.STALL_TIMEOUT:
                    self._set(detail="stream stalled — device gone?")
                    return
                continue
            t_last_frame = now
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                inst = 1.0 / dt
                ema = inst if ema == 0 else 0.9 * ema + 0.1 * inst
            with self._lock:
                self.status.latest = frame
                self.status.fps = ema
                self.status.frames += 1

    def run(self) -> None:
        while not self._stop_evt.is_set():
            if self._try_open():
                self._stream()
                try:
                    self.cam.close()
                except Exception:                   # noqa: BLE001
                    pass
                # Mark the drop so the dashboard/ROS status reflect it during the
                # gap before the next reconnect attempt.
                self._set(state="waiting", detail="reconnecting…", fps=0.0)
            # Back off before retrying (open failed OR the stream dropped), but
            # stay responsive to stop().
            self._stop_evt.wait(self.RECONNECT_INTERVAL)
        try:
            self.cam.close()
        except Exception:                           # noqa: BLE001
            pass

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self.status.latest

    def stop(self) -> None:
        self._stop_evt.set()


class CameraManager:
    def __init__(self, cameras: list[CameraBase]):
        self._workers = [_Worker(c) for c in cameras]

    def start(self) -> None:
        for w in self._workers:
            w.start()

    def statuses(self) -> list[CameraStatus]:
        return [w.status for w in self._workers]

    def latest(self, name: str) -> CameraFrame | None:
        for w in self._workers:
            if w.cam.name == name:
                return w.latest()
        return None

    def running_names(self) -> list[str]:
        return [w.cam.name for w in self._workers if w.status.state == "running"]

    def stop(self) -> None:
        for w in self._workers:
            w.stop()
        for w in self._workers:
            w.join(timeout=2.0)

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

"""
pose_logger.py — Pretty-print TeleopFrame data received from the Vision Pro.

Designed for validation: run the server, open the app, and watch joints
update in the terminal to confirm the pipeline is working end-to-end.
"""
from __future__ import annotations
import math
import time
from collections import deque
from typing import Deque

from wire_protocol import TeleopFrame, Transform4x4

# ANSI colours (no external deps)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_MAGENTA= "\033[35m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_DIM    = "\033[2m"


def _xyz(t: Transform4x4) -> str:
    x, y, z = t.position_xyz()
    return f"({x:+.3f}, {y:+.3f}, {z:+.3f})"


def _euler_from_matrix(m: list[float]) -> tuple[float, float, float]:
    """Extract roll/pitch/yaw (ZYX) from a row-major 4x4 rotation matrix.
    Returns degrees."""
    # Row-major: m[row*4 + col]
    r00, r01, r02 = m[0], m[1], m[2]
    r10, r11, r12 = m[4], m[5], m[6]
    r20, r21, r22 = m[8], m[9], m[10]

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r20))))
    if abs(r20) < 0.9999:
        roll = math.degrees(math.atan2(r21, r22))
        yaw  = math.degrees(math.atan2(r10, r00))
    else:
        roll = math.degrees(math.atan2(-r12, r11))
        yaw  = 0.0
    return roll, pitch, yaw


def _rpy(t: Transform4x4) -> str:
    r, p, y = _euler_from_matrix(t.matrix)
    return f"R{r:+.1f}° P{p:+.1f}° Y{y:+.1f}°"


class PoseLogger:
    """
    Stateful logger: tracks frame rate, prints concise summaries, and
    optionally dumps full joint tables every N frames.
    """

    def __init__(self, log_every_n: int = 30, verbose_matrices: bool = False):
        self.log_every_n = log_every_n
        self.verbose = verbose_matrices
        self._count = 0
        self._t_window: Deque[float] = deque(maxlen=60)

    # ------------------------------------------------------------------

    def log(self, frame: TeleopFrame) -> None:
        self._count += 1
        now = time.monotonic()
        self._t_window.append(now)

        if self._count % max(1, self.log_every_n) != 0:
            return

        fps = self._fps()
        self._print_frame(frame, fps)

    # ------------------------------------------------------------------

    def _fps(self) -> float:
        if len(self._t_window) < 2:
            return 0.0
        dt = self._t_window[-1] - self._t_window[0]
        return (len(self._t_window) - 1) / dt if dt > 0 else 0.0

    def _print_frame(self, f: TeleopFrame, fps: float) -> None:
        head_pos  = _xyz(f.head.transform)
        head_rpy  = _rpy(f.head.transform)
        tracking  = f.head.trackingState.value.upper()

        lh = "—"
        rh = "—"
        lh_wrist = "—"
        rh_wrist = "—"
        l_joints_tracked = 0
        r_joints_tracked = 0

        if f.leftHand and f.leftHand.isTracked:
            lh = _xyz(f.leftHand.anchorTransform)
            lh_wrist = _rpy(f.leftHand.anchorTransform)
            l_joints_tracked = sum(1 for j in f.leftHand.joints if j.isTracked)
        if f.rightHand and f.rightHand.isTracked:
            rh = _xyz(f.rightHand.anchorTransform)
            rh_wrist = _rpy(f.rightHand.anchorTransform)
            r_joints_tracked = sum(1 for j in f.rightHand.joints if j.isTracked)

        print(
            f"\n{_BOLD}{'─'*68}{_RESET}\n"
            f"  {_BOLD}seq{_RESET} {f.sequence:<8}  "
            f"{_BOLD}fps{_RESET} {fps:.1f}  "
            f"{_BOLD}mode{_RESET} {_YELLOW}{f.mode.value}{_RESET}  "
            f"{_BOLD}tracking{_RESET} {_GREEN if tracking == 'NORMAL' else _YELLOW}{tracking}{_RESET}\n"
            f"\n"
            f"  {_BOLD}{_CYAN}HEAD{_RESET}     pos {head_pos}  orient {head_rpy}\n"
            f"  {_BOLD}{_CYAN}L WRIST{_RESET}  pos {lh}  orient {lh_wrist}  "
            f"joints {l_joints_tracked}/27\n"
            f"  {_BOLD}{_MAGENTA}R WRIST{_RESET}  pos {rh}  orient {rh_wrist}  "
            f"joints {r_joints_tracked}/27",
            flush=True,
        )

        if self.verbose and f.leftHand and f.leftHand.isTracked:
            print(f"\n  {_DIM}Left hand joints:{_RESET}")
            for j in f.leftHand.joints:
                x, y, z = j.localTransform.position_xyz()
                tracked = "✓" if j.isTracked else "✗"
                print(f"    {tracked} {j.joint:<34} ({x:+.3f}, {y:+.3f}, {z:+.3f})")

        if self.verbose and f.rightHand and f.rightHand.isTracked:
            print(f"\n  {_DIM}Right hand joints:{_RESET}")
            for j in f.rightHand.joints:
                x, y, z = j.localTransform.position_xyz()
                tracked = "✓" if j.isTracked else "✗"
                print(f"    {tracked} {j.joint:<34} ({x:+.3f}, {y:+.3f}, {z:+.3f})")

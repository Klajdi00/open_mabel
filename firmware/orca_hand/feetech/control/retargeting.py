#!/usr/bin/env python3
"""MABEL ORCA Hand — hand retargeting from MediaPipe landmarks.

Captures a live webcam feed, detects the operator's hand using MediaPipe,
maps the 21 hand landmarks to MABEL joint angles, and streams position
commands to the physical hand.

Requires:
    pip install mediapipe opencv-python

Retargeting model (right operator → MABEL left hand):
  • Finger curl:  metacarpal–proximal–distal landmark angle → MCP / PIP
  • Abduction:    angle between adjacent metacarpals → ABD joints
  • Thumb:        multi-segment mapping (CMC, MP, IP)
  • Wrist:        wrist roll from forearm landmark

Usage:
    python retargeting.py
    python retargeting.py --cam 1 --mirror     # external camera, mirror view
    python retargeting.py --scale 0.8          # reduce range of motion
    python retargeting.py --no-robot           # dry-run (visualise only)
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

_FEETECH   = Path(__file__).resolve().parent.parent
_ORCA_ROOT = _FEETECH.parent / "orca_core"
for _p in (str(_FEETECH), str(_ORCA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.hand_config import (
    BAUDRATE, TOTAL, JOINT_ID, ID_JOINT, POSES,
    detect_port, load_calibration, pct_to_raw,
    RED, GREEN, YELLOW, CYAN, BOLD, RESET,
)

try:
    import cv2
    import mediapipe as mp
except ImportError:
    print(f"{RED}Missing dependencies.  Install with:{RESET}")
    print("    pip install mediapipe opencv-python")
    sys.exit(1)

try:
    from orca_core.hardware.feetech_client import FeetechClient, DEFAULT_POS_SCALE, POSITION_DIRECTION
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False

# ─── MediaPipe landmark indices ───────────────────────────────────────────────
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST     = 0
THUMB_CMC = 1;  THUMB_MCP  = 2;  THUMB_IP  = 3;  THUMB_TIP  = 4
INDEX_MCP = 5;  INDEX_PIP  = 6;  INDEX_DIP = 7;  INDEX_TIP  = 8
MIDDLE_MCP= 9;  MIDDLE_PIP =10;  MIDDLE_DIP=11;  MIDDLE_TIP =12
RING_MCP  =13;  RING_PIP   =14;  RING_DIP  =15;  RING_TIP   =16
PINKY_MCP =17;  PINKY_PIP  =18;  PINKY_DIP =19;  PINKY_TIP  =20


def _angle_3pts(a, b, c) -> float:
    """Angle at vertex b formed by points a, b, c (radians)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.acos(float(np.clip(cos, -1, 1)))


def _lm_to_np(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


# ─── Landmark → joint angle mapping ──────────────────────────────────────────

def landmarks_to_joints(lms, scale: float = 1.0) -> dict[str, float]:
    """Map MediaPipe hand landmarks to MABEL joint angles (percent 0-100).

    Returns {joint_name: pct}.  Only joints that can be reliably estimated
    are included; the rest hold their last position.
    """
    pts = [_lm_to_np(lms[i]) for i in range(21)]

    def curl(mcp_i, pip_i, tip_i) -> float:
        """Finger curl 0 (extended) → 100 (fully flexed)."""
        ang = _angle_3pts(pts[mcp_i - 1], pts[mcp_i], pts[pip_i])
        # angle ~π when extended, decreases when flexed
        pct = (1.0 - ang / math.pi) * 100.0 * 2.0
        return max(0.0, min(100.0, pct * scale))

    def abduction(a_mcp, b_mcp) -> float:
        """Abduction between two fingers: 0=adducted, 100=spread."""
        v1 = pts[a_mcp] - pts[WRIST]
        v2 = pts[b_mcp] - pts[WRIST]
        ang = _angle_3pts(pts[a_mcp], pts[WRIST], pts[b_mcp])
        pct = (ang / (math.pi / 6)) * 50.0   # ~30° = 100%
        return max(0.0, min(100.0, pct))

    out: dict[str, float] = {}

    # Finger flexion
    out["index_mcp"]  = curl(INDEX_MCP,  INDEX_PIP,  INDEX_TIP)
    out["index_pip"]  = curl(INDEX_PIP,  INDEX_DIP,  INDEX_TIP)
    out["middle_mcp"] = curl(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    out["middle_pip"] = curl(MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
    out["ring_mcp"]   = curl(RING_MCP,   RING_PIP,   RING_TIP)
    out["ring_pip"]   = curl(RING_PIP,   RING_DIP,   RING_TIP)
    out["pinky_mcp"]  = curl(PINKY_MCP,  PINKY_PIP,  PINKY_TIP)
    out["pinky_pip"]  = curl(PINKY_PIP,  PINKY_DIP,  PINKY_TIP)

    # Thumb
    out["thumb_cmc"] = curl(THUMB_CMC, THUMB_MCP, THUMB_TIP)
    out["thumb_mp"]  = curl(THUMB_MCP, THUMB_IP,  THUMB_TIP)
    out["thumb_ip"]  = curl(THUMB_IP,  THUMB_TIP, THUMB_MCP)

    # Abduction (spread between fingers)
    out["index_abd"]  = abduction(INDEX_MCP,  MIDDLE_MCP)
    out["middle_abd"] = abduction(MIDDLE_MCP, RING_MCP)
    out["ring_abd"]   = abduction(RING_MCP,   PINKY_MCP)
    out["pinky_abd"]  = abduction(PINKY_MCP,  RING_MCP)
    out["thumb_abd"]  = abduction(THUMB_CMC,  INDEX_MCP)

    return out


# ─── Low-pass filter (one-euro style, simplified) ─────────────────────────────

class LowPassFilter:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._state: dict[str, float] = {}

    def update(self, values: dict[str, float]) -> dict[str, float]:
        out = {}
        for k, v in values.items():
            prev = self._state.get(k, v)
            s = prev + self.alpha * (v - prev)
            self._state[k] = s
            out[k] = s
        return out


# ─── Robot writer thread ──────────────────────────────────────────────────────

class RobotWriter:
    """Sends latest joint targets to the hand at a fixed rate."""

    def __init__(self, port: str, calib: dict):
        self.calib   = calib
        self.ids     = list(range(1, TOTAL + 1))
        self.client  = FeetechClient(self.ids, port=port, baudrate=BAUDRATE)
        self.client.connect()
        self._latest: dict[str, float] = {}
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, joints: dict[str, float]) -> None:
        with self._lock:
            self._latest = dict(joints)

    def stop(self):
        self._stop.set()
        try:
            self.client.disconnect()
        except Exception:
            pass

    def _run(self):
        from orca_core.hardware.feetech_client import DEFAULT_POS_SCALE, POSITION_DIRECTION
        hz = 30
        dt = 1.0 / hz
        while not self._stop.is_set():
            with self._lock:
                joints = dict(self._latest)
            if joints:
                mids, rads = [], []
                for jname, pct in joints.items():
                    mid = JOINT_ID.get(jname)
                    if mid:
                        raw  = float(pct_to_raw(mid, pct, self.calib))
                        rad  = raw * DEFAULT_POS_SCALE * POSITION_DIRECTION
                        mids.append(mid)
                        rads.append(rad)
                if mids:
                    try:
                        self.client.write_positions_sync(
                            mids, np.array(rads), speed=1200, torque=500
                        )
                    except Exception:
                        pass
            time.sleep(dt)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="MABEL hand retargeting via MediaPipe")
    parser.add_argument("--port",     type=str)
    parser.add_argument("--cam",      type=int, default=0, help="Camera index")
    parser.add_argument("--mirror",   action="store_true", help="Flip camera view")
    parser.add_argument("--scale",    type=float, default=1.0,
                        help="RoM scale factor 0-1 (default 1.0 = full range)")
    parser.add_argument("--alpha",    type=float, default=0.25,
                        help="Low-pass filter strength 0-1 (default 0.25)")
    parser.add_argument("--no-robot", action="store_true",
                        help="Visualise only, do not send to robot")
    args = parser.parse_args()

    port  = args.port or (detect_port() if not args.no_robot else "dry-run")
    calib = load_calibration()

    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}  MABEL — HAND RETARGETING (MediaPipe){RESET}")
    print(f"{BOLD}{'='*50}{RESET}")
    print(f"{CYAN}  Camera: {args.cam}   Scale: {args.scale}   α: {args.alpha}{RESET}")
    if args.no_robot:
        print(f"{YELLOW}  DRY RUN — no robot commands will be sent.{RESET}")
    print(f"\n  Press {BOLD}Q{RESET} in the video window to quit.\n")

    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"{RED}Cannot open camera {args.cam}{RESET}")
        return 1

    lpf    = LowPassFilter(alpha=args.alpha)
    writer = RobotWriter(port, calib) if (not args.no_robot and ORCA_AVAILABLE) else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lms    = results.multi_hand_landmarks[0].landmark
                joints = landmarks_to_joints(lms, scale=args.scale)
                joints = lpf.update(joints)

                if writer:
                    writer.update(joints)

                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    results.multi_hand_landmarks[0],
                    mp_hands.HAND_CONNECTIONS,
                )

                # Overlay joint values
                y = 30
                for jname, pct in list(joints.items())[:8]:
                    cv2.putText(frame, f"{jname}: {pct:.0f}%",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    y += 18
            else:
                cv2.putText(frame, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("MABEL Retargeting", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        if writer:
            writer.stop()

    print(f"\n{GREEN}Retargeting stopped.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

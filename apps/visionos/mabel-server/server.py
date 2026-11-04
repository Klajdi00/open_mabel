"""
Mabel teleop reference server.

This is the robot-side counterpart to the Vision Pro app. It accepts
pose frames from the headset, prints basic retargeting info, and
publishes a fake RobotState back once a second. 

This updated version includes the Spatial Virtual Joystick logic 
for Base Navigation via right-hand pinch gestures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.server import WebSocketServerProtocol

log = logging.getLogger("mabel.teleop")

# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

HEADSET_CLIENT_NAME = "VisionPro-MabelTeleop"

@dataclass
class Envelope:
    type: str
    payload: dict[str, Any]

    @classmethod
    def from_bytes(cls, data: bytes | str) -> "Envelope":
        obj = json.loads(data)
        return cls(type=obj["type"], payload=obj.get("payload", {}))

    def to_bytes(self) -> bytes:
        return json.dumps({"type": self.type, "payload": self.payload}).encode("utf-8")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

TeleopFrameCallback = Callable[[dict[str, Any]], Awaitable[None]]

class TeleopServer:
    def __init__(self, on_teleop_frame: Optional[TeleopFrameCallback] = None) -> None:
        self.on_teleop_frame = on_teleop_frame or self._default_frame_handler
        self.last_frame_ts: float = 0.0
        self.frame_count: int = 0
        self.last_rtt_ms: Optional[float] = None
        
        # Spatial Joystick State
        self.is_pinched = False
        self.joystick_origin: Optional[tuple[float, float, float]] = None

    async def handle(self, ws: WebSocketServerProtocol) -> None:
        log.info("headset connected from %s", ws.remote_address)
        state_task = asyncio.create_task(self._publish_robot_state(ws))
        try:
            async for message in ws:
                try:
                    env = Envelope.from_bytes(message)
                except Exception as e:
                    log.warning("bad envelope: %s", e)
                    continue
                await self._dispatch(ws, env)
        except websockets.ConnectionClosed:
            pass
        finally:
            state_task.cancel()
            log.info("headset disconnected; received %d frames", self.frame_count)

    async def _dispatch(self, ws: WebSocketServerProtocol, env: Envelope) -> None:
        if env.type == "teleop_frame":
            self.frame_count += 1
            self.last_frame_ts = time.monotonic()
            await self.on_teleop_frame(env.payload)

        elif env.type == "hello":
            client = env.payload.get("client", "<unknown>")
            version = env.payload.get("version", "?")
            log.info("handshake from %s v%s", client, version)

        elif env.type == "ping":
            pong = Envelope("pong", {"t": time.monotonic()})
            await ws.send(pong.to_bytes())

        elif env.type == "pong":
            t = env.payload.get("t")
            if isinstance(t, (int, float)):
                self.last_rtt_ms = (time.monotonic() - float(t)) * 1000.0

    # -----------------------------------------------------------------------
    # Retargeting & Virtual Joystick Logic
    # -----------------------------------------------------------------------

    async def _default_frame_handler(self, payload: dict[str, Any]) -> None:
        # The mode string comes from the updated SwiftUI app
        mode = payload.get("mode", "Arms & Hands")
        right_hand = payload.get("rightHand")

        if mode == "Base Driving":
            await self._process_base_navigation(right_hand)
        else:
            # If we just switched back to manipulation, ensure base brakes are applied
            if self.is_pinched:
                self.is_pinched = False
                self.joystick_origin = None
                log.info("🕹️ BASE BRAKE: Mode switched, zeroing velocities.")
                # TODO: Publish geometry_msgs/Twist with 0.0 to your ROS base topic
                
            await self._process_manipulation(payload)

    async def _process_base_navigation(self, right_hand: Optional[dict[str, Any]]) -> None:
        if not right_hand or not right_hand.get("isTracked"):
            if self.is_pinched:
                self._apply_brakes("Tracking lost")
            return

        # Get world transforms for thumb and index tips
        thumb_tx = joint_world_transform(right_hand, "thumb_tip")
        index_tx = joint_world_transform(right_hand, "index_tip")

        if not thumb_tx or not index_tx:
            return

        # Extract 3D translations (X, Y, Z) from the row-major matrices
        thumb_pos = extract_translation(thumb_tx)
        index_pos = extract_translation(index_tx)

        # Calculate Euclidean distance between fingers
        pinch_distance = math.dist(thumb_pos, index_pos)
        PINCH_THRESHOLD = 0.025 # 2.5 cm

        if pinch_distance < PINCH_THRESHOLD:
            if not self.is_pinched:
                # PINCH JUST STARTED: Set the origin point
                self.is_pinched = True
                self.joystick_origin = thumb_pos
                log.info("🕹️ JOYSTICK ENGAGED: Origin set.")
            else:
                # PINCH DRAGGING: Calculate delta from origin
                dx = thumb_pos[0] - self.joystick_origin[0] # Strafe (Left/Right)
                dy = thumb_pos[1] - self.joystick_origin[1] # Lift (Up/Down)
                dz = thumb_pos[2] - self.joystick_origin[2] # Surge (Forward/Back)

                # ARKit coordinate system: -Z is forward.
                # Pushing hand forward makes dz negative.
                # Let's map these to standard ROS REP-103 Twist velocities:
                # Multiply by a sensitivity scalar (e.g., 5.0) to convert meters of drag to m/s
                SENSITIVITY = 5.0
                
                v_x = -dz * SENSITIVITY  # Forward/Backward
                v_y = -dx * SENSITIVITY  # Strafe Left/Right
                v_z = dy * SENSITIVITY   # Standing Desk Lift

                # Apply a small deadzone so micro-jitters don't move the 8k robot
                DEADZONE = 0.05
                v_x = v_x if abs(v_x) > DEADZONE else 0.0
                v_y = v_y if abs(v_y) > DEADZONE else 0.0
                v_z = v_z if abs(v_z) > DEADZONE else 0.0

                if self.frame_count % 10 == 0:
                    log.info(f"🚀 DRIVE CMD -> vX(fwd): {v_x:.2f}, vY(strf): {v_y:.2f}, vZ(lift): {v_z:.2f}")
                
                # TODO: Publish these to your ROS Swerve Drive /cmd_vel topic

        else:
            if self.is_pinched:
                self._apply_brakes("Pinch released")

    def _apply_brakes(self, reason: str) -> None:
        self.is_pinched = False
        self.joystick_origin = None
        log.info(f"🛑 BRAKE ({reason}): Zeroing velocities.")
        # TODO: Publish 0.0 velocities to ROS

    async def _process_manipulation(self, payload: dict[str, Any]) -> None:
        # Standard arm & hand retargeting logs
        if self.frame_count % 60 == 1:
            seq = payload.get("sequence")
            left = payload.get("leftHand")
            left_tracked = bool(left and left.get("isTracked"))
            log.info(f"🦾 MANIPULATION MODE -> Seq: {seq} | L-Tracked: {left_tracked}")

    # -- outbound state ----------------------------------------------------

    async def _publish_robot_state(self, ws: WebSocketServerProtocol) -> None:
        seq = 0
        try:
            while True:
                seq += 1
                state = {
                    "timestamp": time.time(),
                    "mode": "teleop" if self.frame_count > 0 else "idle",
                    "battery": { "percentage": 0.85, "voltage": 24.9, "charging": False },
                    "jointPositions": {},
                    "diagnostics": [],
                    "latencyMs": self.last_rtt_ms,
                }
                env = Envelope("robot_state", state)
                try:
                    await ws.send(env.to_bytes())
                except websockets.ConnectionClosed:
                    return
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

# ---------------------------------------------------------------------------
# Pose decode helpers
# ---------------------------------------------------------------------------

def unpack_transform(t: dict[str, Any]) -> list[float]:
    """Returns the 16 floats of a 4x4 row-major transform."""
    m = t.get("matrix")
    if not isinstance(m, list) or len(m) != 16:
        raise ValueError(f"bad transform: {t!r}")
    return [float(x) for x in m]

def extract_translation(matrix: list[float]) -> tuple[float, float, float]:
    """Extracts X, Y, Z translation from a 16-element row-major 4x4 matrix."""
    # In a row-major array, translations are at indices 3, 7, and 11
    return (matrix[3], matrix[7], matrix[11])

def joint_world_transform(hand: dict[str, Any], joint_name: str) -> Optional[list[float]]:
    if not hand or not hand.get("isTracked"):
        return None
    anchor = unpack_transform(hand["anchorTransform"])
    for j in hand.get("joints", []):
        if j["joint"] == joint_name and j.get("isTracked"):
            local = unpack_transform(j["localTransform"])
            return _matmul4(anchor, local)
    return None

def _matmul4(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * 16
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = sum(a[i * 4 + k] * b[k * 4 + j] for k in range(4))
    return out

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Mabel teleop reference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--path", default="/teleop")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def handler(ws: WebSocketServerProtocol) -> None:
        if ws.path != args.path:
            log.warning("rejecting connection on %s (expected %s)", ws.path, args.path)
            await ws.close()
            return
        server = TeleopServer()
        await server.handle(ws)

    async with websockets.serve(handler, args.host, args.port, max_size=4 * 1024 * 1024):
        log.info("serving on ws://%s:%d%s", args.host, args.port, args.path)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

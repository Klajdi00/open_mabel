"""
teleop_receiver.py — Async WebSocket server that receives TeleopFrames
from the Vision Pro and echoes back mock RobotState telemetry.

Protocol (mirrors WebSocketTelemetryClient.swift):
  - Inbound:  WireEnvelope { type: "teleop_frame" | "ping" | "pong" | "hello" }
  - Outbound: WireEnvelope { type: "robot_state" | "ping" | "pong" }

The server also manages a heartbeat: if a client goes silent for
HEARTBEAT_TIMEOUT seconds it is considered gone and its connection closed.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Awaitable

import websockets
from websockets.server import WebSocketServerProtocol

from wire_protocol import (
    WireCodec, WireMessageType, RobotState, RobotMode, Battery,
    TeleopFrame,
)

log = logging.getLogger("teleop_receiver")

# How long (s) without any message before we close the connection
HEARTBEAT_TIMEOUT = 10.0
# How often (s) we broadcast a fresh mock RobotState to connected clients
ROBOT_STATE_INTERVAL = 0.5


class TeleopReceiver:
    """
    Manages all active Vision Pro connections. Emits received TeleopFrames
    via the `on_frame` callback so the rest of the server can act on them.
    """

    def __init__(
        self,
        on_frame: Callable[[TeleopFrame], Awaitable[None]] | None = None,
    ):
        self._on_frame = on_frame
        self._clients: set[WebSocketServerProtocol] = set()
        self._last_rx: dict[WebSocketServerProtocol, float] = {}

        # Mutable mock state — replace with real robot data in production
        self._robot_state = RobotState(
            mode=RobotMode.IDLE,
            battery=Battery(percentage=85.0, voltage=24.2, isCharging=False),
            latencyMs=0.0,
        )

    # ------------------------------------------------------------------
    # WebSocket handler (one Task per connection)
    # ------------------------------------------------------------------

    async def handler(self, ws: WebSocketServerProtocol) -> None:
        """Called by websockets for every new connection."""
        addr = ws.remote_address
        log.info("Client connected: %s", addr)
        self._clients.add(ws)
        self._last_rx[ws] = time.monotonic()

        heartbeat_task = asyncio.create_task(self._heartbeat_guard(ws))
        state_task     = asyncio.create_task(self._state_broadcaster(ws))

        try:
            async for raw in ws:
                self._last_rx[ws] = time.monotonic()
                await self._handle_message(ws, raw)
        except websockets.exceptions.ConnectionClosed as e:
            log.info("Client disconnected: %s  code=%s", addr, e.code)
        except Exception as e:
            log.warning("Unexpected error from %s: %s", addr, e)
        finally:
            heartbeat_task.cancel()
            state_task.cancel()
            self._clients.discard(ws)
            self._last_rx.pop(ws, None)
            log.info("Cleaned up: %s", addr)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(
        self, ws: WebSocketServerProtocol, raw: bytes | str
    ) -> None:
        try:
            msg_type, payload = WireCodec.decode_envelope(raw)
        except (ValueError, KeyError) as e:
            log.warning("Bad envelope: %s", e)
            return

        if msg_type == WireMessageType.TELEOP_FRAME:
            try:
                frame = WireCodec.decode_teleop_frame(payload)
                if self._on_frame:
                    await self._on_frame(frame)
                # Update latency estimate from round-trip timestamp
                dt_ms = (time.time() - frame.timestamp) * 1000
                self._robot_state.latencyMs = round(dt_ms, 1)
            except (KeyError, ValueError) as e:
                log.warning("Bad teleop_frame: %s", e)

        elif msg_type == WireMessageType.PING:
            pong = WireCodec.encode_pong()
            await ws.send(pong)

        elif msg_type == WireMessageType.PONG:
            pass  # acknowledged; _last_rx already updated above

        elif msg_type == WireMessageType.HELLO:
            client_info = payload
            log.info("Hello from %s: %s", ws.remote_address, client_info)
            # Immediately send current robot state so the dashboard shows data
            await ws.send(WireCodec.encode_robot_state(self._robot_state))

        else:
            log.debug("Unhandled message type: %s", msg_type)

    # ------------------------------------------------------------------
    # Background tasks per connection
    # ------------------------------------------------------------------

    async def _heartbeat_guard(self, ws: WebSocketServerProtocol) -> None:
        """Close the connection if we haven't heard from the client recently."""
        while True:
            await asyncio.sleep(2.0)
            idle = time.monotonic() - self._last_rx.get(ws, 0)
            if idle > HEARTBEAT_TIMEOUT:
                log.warning(
                    "Heartbeat timeout (%.1fs idle) — closing %s",
                    idle, ws.remote_address,
                )
                await ws.close(1001, "heartbeat timeout")
                return

    async def _state_broadcaster(self, ws: WebSocketServerProtocol) -> None:
        """Send a fresh RobotState to this client at ROBOT_STATE_INTERVAL."""
        while True:
            await asyncio.sleep(ROBOT_STATE_INTERVAL)
            # ws.closed (v12+) or not ws.open (v11)
            is_closed = getattr(ws, "closed", not getattr(ws, "open", True))
            if is_closed:
                return
            self._robot_state.timestamp = time.time()
            try:
                await ws.send(WireCodec.encode_robot_state(self._robot_state))
            except websockets.exceptions.ConnectionClosed:
                return

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_mock_state(self, **kwargs) -> None:
        """Patch the mock RobotState (for testing/demo purposes)."""
        for k, v in kwargs.items():
            if hasattr(self._robot_state, k):
                setattr(self._robot_state, k, v)

    @property
    def client_count(self) -> int:
        return len(self._clients)

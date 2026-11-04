"""
telemetry_stream.py — the :9600 telemetry plane (typed-JSON WebSocket).

A read-only, dedicated-port broadcast of everything the :9090 robot_state
under-reports, at per-class rates:

  proprio     @50 Hz  — every streamed joint's {q, dq, tau, temp} + base pose
  control     @50 Hz  — the command INTO the robot (mode, desired joints, twist)
  status      @5 Hz   — estop / controlMode / torque / battery / policy /
                        recording / dataCollection / driver / latency
  lidar       @~5 Hz  — full 2D sweep, sent only when a fresh sweep exists
  collection  @1 Hz   — episode + bag folder enumeration (dirs, counts, sizes)
  network     @1 Hz   — host IPs, port map, THIS client's transport route

Joint keys are the SAME iOS joint ids robot_state.jointPositions uses
(left_arm_1 … right_pinky_pip), so clients reuse their existing id tables —
no new mapping. Unavailable values are null; unavailable subsystems are
{"available": false, "reason": ...} (e.g. battery in sim, temperature in sim).

Threading contract (the camera-contention lesson: newest-only, no reliability
layer, zero cost unwatched):
  * The CONTROL thread only reference-swaps plain dicts into snapshot slots
    (set_proprio / set_control), and only when enabled() — an unwatched robot
    pays nothing, not even dict building.
  * Broadcaster coroutines on the NETWORK asyncio loop encode each frame ONCE
    per tick and fan out to per-socket latest-wins queues (drop-oldest).
  * Status/collection/network are built ON the network loop from reference-
    swapped state via provider callables — never on the control thread.

The bridge wires it up:  TELEMETRY.set_providers(...) + serve via
data_streams.serve_hub_ws(host, TELEMETRY_PORT, TELEMETRY, ...).
"""
from __future__ import annotations

import asyncio
import json
import time

__all__ = ["TelemetryHub", "TELEMETRY"]


class TelemetryHub:
    """Fan-out hub for the telemetry plane (same subscriber surface as
    MapStreamHub, so data_streams.serve_hub_ws hosts it unchanged)."""

    _QUEUE_MAX = 8          # a queue holds ≤ a few frames; latest-wins on overflow

    # rates (Hz) — advertised in hello so clients don't hardcode them
    RATES = {"proprioHz": 50, "controlHz": 50, "statusHz": 5,
             "lidarHz": 5, "collectionHz": 1, "networkHz": 1}

    def __init__(self) -> None:
        self._subs: dict = {}                  # ws -> asyncio.Queue[str]
        self.loop: asyncio.AbstractEventLoop | None = None
        # (version, dict) snapshot slots, reference-swapped by the control thread.
        self._proprio: tuple[int, dict | None] = (0, None)
        self._control: tuple[int, dict | None] = (0, None)
        # Provider callables (network-loop side), set by the bridge:
        #   hello() -> dict          status() -> dict       collection() -> dict
        #   lidar() -> dict|None     network(ws) -> dict
        self._providers: dict = {}
        self._started = False

    # ── producer side (control thread) ──────────────────────────────────────
    def enabled(self) -> bool:
        return bool(self._subs)

    def set_proprio(self, d: dict) -> None:
        self._proprio = (self._proprio[0] + 1, d)

    def set_control(self, d: dict) -> None:
        self._control = (self._control[0] + 1, d)

    # ── wiring (network thread, once) ────────────────────────────────────────
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        if not self._started:
            self._started = True
            loop.create_task(self._fast_broadcaster())
            loop.create_task(self._mid_broadcaster())
            loop.create_task(self._slow_broadcaster())

    def set_providers(self, **providers) -> None:
        self._providers.update(providers)

    def _call(self, name: str, *a):
        fn = self._providers.get(name)
        if fn is None:
            return None
        try:
            return fn(*a)
        except Exception:
            return None

    # ── subscriber side (network asyncio loop) ──────────────────────────────
    async def add_subscriber(self, ws) -> None:
        """Register a client: hello + a full snapshot immediately (so a UI
        paints without waiting out the slow tick), then live frames until the
        socket closes. Inbound frames: {"type":"ping","t":...} → pong (echoed
        t, client computes RTT); everything else is ignored."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAX)
        hello = dict(self._call("hello") or {})
        hello.update(type="hello", t=time.time(), rates=dict(self.RATES))
        self._offer(q, json.dumps(hello))
        for name in ("status", "collection"):
            snap = self._call(name)
            if snap:
                self._offer(q, self._frame(name, snap))
        net = self._call("network", ws)
        if net:
            self._offer(q, self._frame("network", net))
        self._subs[ws] = q
        writer = asyncio.create_task(self._writer(ws, q))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if isinstance(msg, dict) and msg.get("type") == "ping":
                        self._offer(q, json.dumps(
                            {"type": "pong", "t": msg.get("t"), "serverT": time.time()}))
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._subs.pop(ws, None)
            writer.cancel()

    async def _writer(self, ws, q: asyncio.Queue) -> None:
        """Single sender per socket (websockets forbids concurrent ws.send)."""
        try:
            while True:
                await ws.send(await q.get())
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _frame(mtype: str, payload: dict) -> str:
        return json.dumps({"type": mtype, "t": time.time(), **payload})

    def _offer(self, q: asyncio.Queue, frame: str) -> None:
        """Latest-wins enqueue: drop the oldest frame rather than block or grow."""
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    def _offer_all(self, frame: str) -> None:
        for q in list(self._subs.values()):
            self._offer(q, frame)

    # ── broadcasters (encode once per tick, fan out) ─────────────────────────
    async def _fast_broadcaster(self) -> None:
        """proprio + control at a STRICT 50 Hz wire cadence — the latest
        snapshot is (re)sent every tick, exactly like the :9090 robot_state
        stream, so the wire rate never tracks control-loop load. Freshness is
        carried honestly by each message's `stamp` (the control thread's
        sample time): under heavy load samples refresh at the outer-loop rate
        while the wire stays at 50 Hz."""
        period = 0.02
        loop = asyncio.get_event_loop()
        next_t = loop.time()
        while True:
            next_t += period
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = loop.time()
                await asyncio.sleep(0)
            if not self._subs:
                continue
            _, d = self._proprio
            if d is not None:
                self._offer_all(self._frame("proprio", d))
            _, d = self._control
            if d is not None:
                self._offer_all(self._frame("control", d))

    async def _mid_broadcaster(self) -> None:
        """status @5 Hz + the lidar sweep whenever a fresh one exists."""
        lidar_sent = None
        while True:
            await asyncio.sleep(0.2)
            if not self._subs:
                continue
            st = self._call("status")
            if st:
                self._offer_all(self._frame("status", st))
            li = self._call("lidar")
            if li is not None and li.get("stamp") != lidar_sent:
                lidar_sent = li.get("stamp")
                self._offer_all(self._frame("lidar", li))

    async def _slow_broadcaster(self) -> None:
        """collection (shared) + network (per-socket transport) @1 Hz."""
        while True:
            await asyncio.sleep(1.0)
            if not self._subs:
                continue
            col = self._call("collection")
            if col:
                self._offer_all(self._frame("collection", col))
            for ws, q in list(self._subs.items()):
                net = self._call("network", ws)
                if net:
                    self._offer(q, self._frame("network", net))


TELEMETRY = TelemetryHub()

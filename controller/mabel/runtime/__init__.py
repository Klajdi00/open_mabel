"""runtime — operator-loop instrumentation and I/O (not control law).

The helpers a live teleop/controller loop uses for telemetry and capture, kept out
of the control path so the math packages stay dependency-light:

    PerfMonitor     live per-stage rate/latency HUD           (hud.py)
    Dashboard       terminal ANSI status redraw               (dashboard.py)
    EpisodeRecorder demo capture -> curation pipeline          (recorder.py)

Old flat paths (`mabel.hud`, `mabel.dashboard`, `mabel.recorder`) still resolve via
the lazy compatibility finder in `mabel/__init__.py`.
"""
from .dashboard import Dashboard          # noqa: F401
from .hud import PerfMonitor              # noqa: F401
from .recorder import EpisodeRecorder     # noqa: F401

__all__ = ["Dashboard", "PerfMonitor", "EpisodeRecorder"]

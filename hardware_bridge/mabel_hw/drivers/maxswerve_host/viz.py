"""RerunLogger — live visualizer for MAXSwerve control states via Rerun.

Rerun is an OPTIONAL dependency: this module imports it, but the core package
(`maxswerve_host`) does not, so `import maxswerve_host` works without rerun. Install
with `pip install -e ".[viz]"` (or `pip install rerun-sdk`).

Each per-module signal is logged as a single 3-element `rr.Scalars`, so all three
modules appear as named lines (m1/m2/m3) in one plot. Feed it the SwerveClient state:

    from maxswerve_host import SwerveClient
    from maxswerve_host.viz import RerunLogger

    sw = SwerveClient().connect()
    viz = RerunLogger(spawn=True)            # or web=True on a headless server
    while True:
        st = sw.getState()
        if st: viz.log_state(st, rtt=sw.last_rtt())

Tested against rerun-sdk 0.33.
"""

from __future__ import annotations

import numpy as np
import rerun as rr

_MODULES = ["m1", "m2", "m3"]

# Per-module signals logged as 3-element Scalars (one plot, three named lines).
_SERIES_PATHS = [
    "drive/velocity_mps",
    "drive/velocity_cmd_mps",
    "drive/position_rot",
    "drive/torque_est_nm",
    "steer/angle_deg",
    "steer/angle_cmd_deg",
    "health/drive_voltage_v",
    "health/drive_current_a",
    "health/drive_temp_c",
    "health/steer_current_a",
    "health/steer_temp_c",
]


class RerunLogger:
    def __init__(
        self,
        app_id: str = "maxswerve_live",
        spawn: bool = True,
        web: bool = False,
        save: str | None = None,
    ):
        """Start a Rerun recording stream and a viewer.

        spawn: open the native viewer (needs a desktop). Ignored if web=True.
        web:   serve a browser viewer (headless servers) — prints the URL.
        save:  also record everything to this .rrd file for offline replay.
        """
        rr.init(app_id, spawn=bool(spawn and not web))
        if save:
            rr.save(save)
        if web:
            uri = rr.serve_grpc()
            rr.serve_web_viewer(connect_to=uri, open_browser=False)
        self._t0: float | None = None
        self._styled = False

    def _set_time(self, recv_time: float):
        if self._t0 is None:
            self._t0 = recv_time
        rr.set_time("t", duration=recv_time - self._t0)

    def _ensure_styling(self):
        # Static, one-time: name the three series lines m1/m2/m3 on each plot.
        if self._styled:
            return
        self._styled = True
        for path in _SERIES_PATHS:
            rr.log(path, rr.SeriesLines(names=_MODULES), static=True)

    def log_state(self, st, rtt: float | None = None, cmd=None):
        """Log one control-state snapshot. cmd=(drive[3], steer[3]) overlays setpoints."""
        self._ensure_styling()
        self._set_time(st.recv_time)
        rr.log("drive/velocity_mps", rr.Scalars(st.drive_velocity))
        rr.log("drive/position_rot", rr.Scalars(st.drive_position))
        rr.log("drive/torque_est_nm", rr.Scalars(st.drive_torque_est))
        rr.log("steer/angle_deg", rr.Scalars(st.steer_angle * 360.0))
        rr.log("diag/age_ms", rr.Scalars(st.age() * 1e3))
        rr.log("diag/live_mask", rr.Scalars(float(st.live_mask)))
        if rtt is not None:
            rr.log("diag/rtt_ms", rr.Scalars(rtt * 1e3))
        if cmd is not None:
            drive_cmd, steer_cmd = cmd
            rr.log("drive/velocity_cmd_mps", rr.Scalars(np.asarray(drive_cmd, float)))
            rr.log("steer/angle_cmd_deg", rr.Scalars(np.asarray(steer_cmd, float) * 360.0))

    def log_health(self, h):
        """Log one health snapshot (lower rate)."""
        self._set_time(h.recv_time)
        rr.log("health/drive_voltage_v", rr.Scalars(h.drive_voltage))
        rr.log("health/drive_current_a", rr.Scalars(h.drive_current))
        rr.log("health/drive_temp_c", rr.Scalars(h.drive_temperature))
        rr.log("health/steer_current_a", rr.Scalars(h.steer_current))
        rr.log("health/steer_temp_c", rr.Scalars(h.steer_temperature))

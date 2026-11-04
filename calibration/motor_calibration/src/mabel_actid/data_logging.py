"""
data_logging.py
===============

A single container for a calibration run plus CSV/JSON (de)serialisation, so the
collection step (hardware *or* simulator) and the identification step are fully
decoupled: collect once, re-fit as many times as you like.

Columns are joint/output side, SI units:
    t      s        time
    q      rad      joint position
    w      rad/s    joint velocity (motor feedback)
    a      rad/s^2  joint acceleration (filtered-differentiated, may be NaN raw)
    tau    N*m      torque the motor reported it applied (output side)
    w_cmd  rad/s    commanded velocity (staircase) -- for steady detection
    steady bool     True on settled constant-velocity samples
"""
from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class CalibrationLog:
    t: np.ndarray
    q: np.ndarray
    w: np.ndarray
    a: np.ndarray
    tau: np.ndarray
    w_cmd: np.ndarray
    steady: np.ndarray
    motor: str = ""
    note: str = ""
    rate: float = 0.0

    def __len__(self) -> int:
        return int(self.t.size)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "q", "w", "a", "tau", "w_cmd", "steady"])
            for i in range(len(self)):
                w.writerow([self.t[i], self.q[i], self.w[i], self.a[i],
                            self.tau[i], self.w_cmd[i], int(self.steady[i])])
        meta = {"motor": self.motor, "note": self.note, "rate": self.rate,
                "n": len(self)}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationLog":
        path = Path(path)
        rows = np.genfromtxt(path, delimiter=",", names=True)
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(
            t=rows["t"], q=rows["q"], w=rows["w"], a=rows["a"],
            tau=rows["tau"], w_cmd=rows["w_cmd"],
            steady=rows["steady"].astype(bool),
            motor=meta.get("motor", ""), note=meta.get("note", ""),
            rate=meta.get("rate", 0.0))

    # --------------------------------------------------------- derived data
    def steady_points(self, vel_tol: float = 0.25, vel_abs: float = 0.01):
        """Average ``(w, tau)`` over each settled constant-velocity segment.

        Returns ``(w_levels, tau_levels)`` -- one clean friction sample per
        staircase step.  Segments are split wherever ``steady`` toggles.

        Within a segment, only samples where the *actual* velocity has converged
        to the commanded one (``|w - w_cmd| <= vel_tol*|w_cmd| + vel_abs``) are
        averaged.  This rejects the stick / break-away phase at the start of a
        low-speed step, where the joint has not begun to move yet and the torque
        is not the running friction -- the dominant low-speed ID error.
        """
        s = self.steady
        edges = np.flatnonzero(np.diff(s.astype(int)) != 0) + 1
        segs = np.split(np.arange(len(self)), edges)
        ws, taus = [], []
        for seg in segs:
            if not (seg.size and s[seg[0]]):
                continue
            wc = float(np.median(self.w_cmd[seg]))
            keep = np.abs(self.w[seg] - wc) <= vel_tol * abs(wc) + vel_abs
            sel = seg[keep]
            if sel.size < max(3, 0.2 * seg.size):    # too little convergence
                continue
            ws.append(float(np.mean(self.w[sel])))
            taus.append(float(np.mean(self.tau[sel])))
        return np.asarray(ws), np.asarray(taus)

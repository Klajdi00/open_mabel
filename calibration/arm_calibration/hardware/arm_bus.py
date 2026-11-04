"""
arm_bus.py
==========

The 7-joint arm on one DaMiao CAN bus, in **MJCF joint coordinates**.

Wraps :class:`mabel_actid.damiao.DamiaoBus` (the same transport the motor
calibration uses -- serial / gs_usb on macOS, socketcan on Linux) and adds the
one thing the identification cannot work without: the **frame map** between what
the motor encoder reports and what the MJCF model means by a joint angle.

    q_motor = sign_j * q_mjcf + offset_j
    tau_motor = sign_j * tau_mjcf

If ``sign`` is wrong for a joint, every torque that joint reports is negated and
the fitted gravity sinusoid is mirrored -- the identification will still converge,
to a confidently wrong answer.  There is no way to detect this from the sweep
data alone, which is why :mod:`00_frame_check` exists and why this module
**refuses to run without a saved, verified frame calibration**.

SAFETY
------
* Every joint that is not under test must HOLD its posture -- an unpowered arm
  falls.  So all 7 motors are enabled for the whole session.
* Motion is always one joint at a time, rest-to-rest, inside the envelope in
  :class:`mabel_sysid.excitation.SafetyLimits`.
* A torque watchdog aborts (and disables) if any joint exceeds its budget.
* Ctrl-C, exceptions and normal exit all disable every motor.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import numpy as np

# the shared DaMiao transport lives in the motor-calibration package
_HERE = os.path.dirname(os.path.abspath(__file__))
_ACTID_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "motor_calibration", "src"))
if _ACTID_SRC not in sys.path:
    sys.path.insert(0, _ACTID_SRC)
_SYSID_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SYSID_SRC not in sys.path:
    sys.path.insert(0, _SYSID_SRC)

from mabel_actid import session as _session          # noqa: E402
from mabel_actid.motors import get as get_motor      # noqa: E402
from mabel_sysid.excitation import (                 # noqa: E402
    JOINT_MOTORS, JOINT_TAU_CONT, SafetyLimits)

FRAME_FILE = os.path.abspath(
    os.path.join(_HERE, "..", "results", "arm_frame_calibration.json"))

# CAN ids 1..7, matching firmware/openarm/scripts/teleop_gui.py
CAN_IDS = [1, 2, 3, 4, 5, 6, 7]

# Position-hold gains for the joints that are NOT under test.  Deliberately soft:
# stiff enough to hold the posture against gravity, soft enough that a collision
# pushes the arm rather than fighting you.  (kp, kd) per joint.
HOLD_GAINS = [(28.0, 2.0), (28.0, 2.0), (22.0, 1.6), (22.0, 1.6),
              (12.0, 0.8), (10.0, 0.8), (10.0, 0.8)]


class FrameNotCalibrated(RuntimeError):
    pass


class ArmBus:
    """The 7-DOF arm, addressed in MJCF joint coordinates."""

    def __init__(self, backend: Optional[str] = None, *, channel: str = "can0",
                 port: Optional[str] = None, lim: Optional[SafetyLimits] = None,
                 require_frame: bool = True):
        self.lim = lim or SafetyLimits()
        self.nv = 7
        self.sign = np.ones(self.nv)
        self.offset = np.zeros(self.nv)
        self._frame_ok = False
        if os.path.exists(FRAME_FILE):
            with open(FRAME_FILE) as f:
                fc = json.load(f)
            self.sign = np.asarray(fc["sign"], float)
            self.offset = np.asarray(fc["offset"], float)
            self._frame_ok = bool(fc.get("verified", False))
        if require_frame and not self._frame_ok:
            raise FrameNotCalibrated(
                f"no VERIFIED frame calibration at {FRAME_FILE}.\n"
                "Run  python3 hardware/00_frame_check.py  first: it establishes the\n"
                "encoder zero and the sign of every joint against the MJCF model.\n"
                "Without it a sign error silently mirrors the identified parameters.")

        backend = backend or _session.default_backend()
        self.backend = backend
        self.bus = _session.make_bus(backend, channel=channel, port=port)
        for j, cid in enumerate(CAN_IDS):
            self.bus.add_motor(cid, get_motor(JOINT_MOTORS[j]))
        self._enabled = False

    # -------------------------------------------------------------- frame map
    def to_motor(self, q_mjcf: np.ndarray) -> np.ndarray:
        return self.sign * np.asarray(q_mjcf, float) + self.offset

    def to_mjcf(self, q_motor: np.ndarray) -> np.ndarray:
        return (np.asarray(q_motor, float) - self.offset) / self.sign

    # ---------------------------------------------------------------- lifecycle
    def enable(self) -> None:
        for cid in CAN_IDS:
            self.bus.clear_faults(cid)
            self.bus.enable(cid)
            time.sleep(0.02)
        self._enabled = True
        time.sleep(0.1)

    def disable(self) -> None:
        try:
            for cid in CAN_IDS:
                self.bus.disable(cid)
                time.sleep(0.01)
        finally:
            self._enabled = False

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *exc):
        self.disable()
        self.bus.shutdown(disable_all=True)

    # ------------------------------------------------------------------ state
    def read(self) -> dict:
        """Current joint state in MJCF coordinates."""
        qm, vm, tm = [], [], []
        for cid in CAN_IDS:
            fb = self.bus.get(cid)
            qm.append(fb.pos); vm.append(fb.vel); tm.append(fb.tau)
        qm = np.asarray(qm); vm = np.asarray(vm); tm = np.asarray(tm)
        return {"q": self.to_mjcf(qm), "qd": vm / self.sign,
                "tau": tm / self.sign, "q_motor": qm, "tau_motor": tm}

    # ----------------------------------------------------------------- command
    def hold(self, q_mjcf: np.ndarray, *, active: Optional[int] = None,
             kp_active: Optional[float] = None,
             kd_active: Optional[float] = None) -> None:
        """One MIT frame per joint: hold ``q_mjcf``.

        ``active`` (a joint index) may use different gains -- the joint under
        test is position-tracked along its sweep with the same soft gains, so
        nothing special is needed, but the hook is here.
        """
        qm = self.to_motor(q_mjcf)
        for j, cid in enumerate(CAN_IDS):
            kp, kd = HOLD_GAINS[j]
            if active is not None and j == active:
                kp = kp_active if kp_active is not None else kp
                kd = kd_active if kd_active is not None else kd
            self.bus.mit(cid, float(qm[j]), 0.0, kp, kd, 0.0)

    def watchdog(self, st: dict) -> None:
        """Abort if any joint is pulling more than its share of continuous torque."""
        budget = self.lim.tau_frac * JOINT_TAU_CONT[: self.nv]
        over = np.abs(st["tau"]) > budget
        if over.any():
            j = int(np.argmax(np.abs(st["tau"]) / budget))
            raise RuntimeError(
                f"TORQUE WATCHDOG: joint {j+1} at {st['tau'][j]:.2f} N*m "
                f"(budget {budget[j]:.2f}). Disabling. Check for a collision, a "
                f"wrong frame map, or a posture the arm cannot hold.")

    # ------------------------------------------------------- trajectory replay
    def play(self, Q: np.ndarray, *, dt: float = 0.01, active: Optional[int] = None,
             on_sample=None, label: str = "") -> None:
        """Track a (T, 7) MJCF joint trajectory at ``dt``, with the watchdog live.

        ``on_sample(i, t, state)`` is called every step -- the collector uses it
        to log.  Blocks for the trajectory's duration.
        """
        T = Q.shape[0]
        t0 = time.time()
        for i in range(T):
            self.hold(Q[i], active=active)
            st = self.read()
            self.watchdog(st)
            if on_sample is not None:
                on_sample(i, i * dt, st)
            # pace the loop
            nxt = t0 + (i + 1) * dt
            slp = nxt - time.time()
            if slp > 0:
                time.sleep(slp)
            if label and i % 200 == 0:
                pct = 100.0 * i / max(T - 1, 1)
                sys.stdout.write(f"\r  {label}: {pct:5.1f}%   ")
                sys.stdout.flush()
        if label:
            sys.stdout.write(f"\r  {label}: 100.0%   \n")


def save_frame(sign, offset, verified: bool, notes: str = "") -> str:
    os.makedirs(os.path.dirname(FRAME_FILE), exist_ok=True)
    with open(FRAME_FILE, "w") as f:
        json.dump({"sign": list(map(float, sign)),
                   "offset": list(map(float, offset)),
                   "verified": bool(verified), "notes": notes,
                   "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return FRAME_FILE

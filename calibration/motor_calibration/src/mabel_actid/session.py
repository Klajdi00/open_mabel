"""
session.py
==========

The calibration *engine*, factored out of ``hardware/collect.py`` so the exact
same drive loop, identification, and plotting are shared by the command-line
tool and the GUI — one implementation, no drift.

Three pieces:

* :func:`make_bus`        — open the right transport for this machine.  On macOS
  there is no SocketCAN, so the default backend is the **DM-USB2FDCAN serial**
  adapter; on Linux it defaults to SocketCAN ``can0`` but ``serial`` works too.
* :func:`run_collection`  — drive the velocity staircase + frequency chirp on an
  already-open bus, returning the two :class:`CalibrationLog`s and saving their
  CSVs.  Progress / per-step logging / abort are delivered through :class:`Hooks`
  so a GUI can show them live; the CLI passes the default no-op hooks.
* :func:`identify_and_save` / :func:`render_raw_plots` — the offline fit + figures.

The physics (excitation profiles, friction/inertia fit) lives in
:mod:`trajectories`, :mod:`identify`, :mod:`metrics`; this module only owns the
real-time motor drive and file layout.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .motors import MotorProfile, get as get_motor
from .trajectories import velocity_staircase, chirp_profile
from .data_logging import CalibrationLog
from . import identify, metrics

# motor_calibration root = .../src/mabel_actid/session.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def result_dir(*parts: str) -> str:
    d = RESULTS_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


# --------------------------------------------------------------------------- #
#  transport                                                                  #
# --------------------------------------------------------------------------- #
def default_backend() -> str:
    """Best transport for this machine, matching the firmware teleop GUI.

    Linux -> ``socketcan``.  macOS -> ``serial`` if a DaMiao serial adapter is
    present, else ``gs_usb`` (CANable/candleLight, e.g. the Jhoinrch USB-CAN).
    """
    from .transports import auto_interface
    return auto_interface()


def make_bus(backend: Optional[str] = None, *, channel: str = "can0",
             port: Optional[str] = None, bitrate: int = 1_000_000):
    """Construct (but do not enable) the actuator bus for this platform.

    ``backend``  ``serial`` (DM-USB2FDCAN), ``gs_usb`` (CANable/candleLight over
                 USB — the macOS path for those adapters), ``socketcan`` (Linux
                 ``can0``), or ``sim`` (synthetic motor).  ``None`` ->
                 :func:`default_backend`.
    ``port``     serial device for ``serial``; ``None`` auto-detects.
    ``channel``  SocketCAN/gs_usb channel name.
    """
    backend = backend or default_backend()
    if backend == "sim":
        return SimBus()
    if backend in ("serial", "gs_usb", "socketcan"):
        from .damiao import DamiaoBus
        return DamiaoBus(interface=backend, channel=channel, port=port,
                         bitrate=bitrate)
    raise ValueError(
        f"unknown backend '{backend}' (use serial / gs_usb / socketcan / sim)")


# --------------------------------------------------------------------------- #
#  simulated bus (no hardware) — for demoing / testing the full pipeline      #
# --------------------------------------------------------------------------- #
class SimBus:
    """A drop-in :class:`damiao.DamiaoBus` that drives a *known* synthetic motor.

    Lets the GUI (and CLI) run an entire calibration — live plots, live fitting,
    final identification — with no adapter or motor attached.  Each MIT command
    advances a 1-DOF friction+inertia plant one control step, so the recovered
    parameters should land near the ground truth baked in here.
    """
    def __init__(self, dt: float = 1.0 / 500.0):
        from .damiao import MotorFeedback
        from .friction_model import FrictionParams
        self._FB = MotorFeedback
        self.dt = dt
        self.truth = FrictionParams(Fc_pos=0.42, Fc_neg=0.44, Fv=0.20,
                                    Fs_pos=0.55, Fs_neg=0.57, v_s=0.25, J=0.018)
        # ground-truth DC-motor torque-speed envelope (Isaac DCMotor)
        self.tau_stall = 24.0      # saturation_effort
        self.w_noload = 6.0        # velocity_limit
        self.tau_cont = 12.0       # effort_limit (continuous clip)
        self._q = 0.0
        self._w = 0.0
        self._enabled = False
        self._fb = MotorFeedback()
        self._rng = np.random.default_rng(0)
        self.profiles = {}

    def add_motor(self, can_id, profile):
        self.profiles[can_id] = profile

    def _step(self, p_des, v_des, kp, kd, t_ff):
        from .friction_model import friction_torque
        tau_cmd = kp * (p_des - self._q) + kd * (v_des - self._w) + t_ff
        # Isaac DCMotor torque-speed clip: deliverable torque shrinks with speed
        # and is capped by the continuous limit. Binds only at high torque/speed,
        # so the low-torque friction staircase is unaffected.
        v = self._w
        top = self.tau_stall * (1.0 - v / self.w_noload)
        bot = self.tau_stall * (-1.0 - v / self.w_noload)
        max_eff = min(top, self.tau_cont)
        min_eff = max(bot, -self.tau_cont)
        applied = min(max(tau_cmd, min_eff), max_eff)
        tau_fr = float(friction_torque(self.truth, np.array([self._w]))[0])
        acc = (applied - tau_fr) / max(self.truth.J, 1e-6)
        self._w += acc * self.dt
        self._q += self._w * self.dt
        fb = self._FB()
        fb.t = time.perf_counter()
        fb.pos, fb.vel = self._q, self._w
        fb.tau = applied + float(self._rng.normal(0.0, 0.02))
        fb.err = 1 if self._enabled else 0
        fb.t_mos, fb.t_rotor = 30.0, 35.0
        self._fb = fb

    def mit(self, can_id, p_des, v_des, kp, kd, t_ff):
        self._step(p_des, v_des, kp, kd, t_ff)

    def hold_velocity(self, can_id, w, kd):
        self._step(0.0, w, 0.0, kd, 0.0)

    def track_position(self, can_id, q, w, kp, kd):
        self._step(q, w, kp, kd, 0.0)

    def torque(self, can_id, t_ff):
        self._step(0.0, 0.0, 0.0, 0.0, t_ff)

    def enable(self, can_id):
        self._enabled = True
        self._fb.err = 1

    def disable(self, can_id):
        self._enabled = False
        self._fb.err = 0

    def clear_faults(self, can_id):
        pass

    def set_zero(self, can_id):
        self._q = 0.0

    def refresh_status(self, can_id):
        pass

    def get(self, can_id):
        out = self._FB()
        for s in self._FB.__slots__:
            setattr(out, s, getattr(self._fb, s))
        return out

    def shutdown(self, disable_all=True):
        self._enabled = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()


# --------------------------------------------------------------------------- #
#  collection config + hooks                                                  #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CollectConfig:
    """Excitation parameters — defaults match the gentle ``run_hardware.sh`` run."""
    rate: float = 500.0
    vel_frac: float = 0.10          # max staircase speed as fraction of rated v_max
    levels: int = 12
    dwell: float = 1.5
    kd: float = 2.0                 # velocity-loop gain
    stair_kp: float = 20.0          # position-ramp gain (forces through stiction)
    chirp: bool = True
    chirp_amp: float = 0.5          # rad
    chirp_f0: float = 0.2
    chirp_f1: float = 3.0
    chirp_dur: float = 20.0
    chirp_kp: float = 40.0
    chirp_kd: float = 2.0
    # torque-speed envelope sweep (Isaac DCMotor: saturation_effort, velocity_limit)
    torque_speed: bool = True
    ts_frac: float = 0.5            # commanded torque as fraction of rated t_max
    ts_ramp: float = 0.4            # s to ramp torque in (avoid a jerk)
    ts_dur: float = 2.5             # s max per direction
    ts_ceil: float = 0.6            # abort/cut when |w| exceeds this fraction of v_max


@dataclasses.dataclass
class Hooks:
    """Optional callbacks so a GUI can follow a run; all default to no-ops."""
    log: Callable[[str], None] = lambda msg: None
    progress: Callable[[float], None] = lambda frac: None      # 0.0..1.0
    phase: Callable[[str], None] = lambda name: None
    should_abort: Callable[[], bool] = lambda: False
    # live sample for on-the-fly plotting: (phase, t, q, w, tau, w_cmd), called
    # at ~SAMPLE_HZ during the drive so a GUI can plot the motion as it happens.
    sample: Callable[[str, float, float, float, float, float], None] = \
        lambda *a: None


SAMPLE_HZ = 25.0      # rate at which live samples are pushed to hooks.sample


class Aborted(Exception):
    """Raised inside the drive loop when ``hooks.should_abort()`` goes True."""


# --------------------------------------------------------------------------- #
#  drive loop                                                                 #
# --------------------------------------------------------------------------- #
def _run_reference(bus, can_id, t_ref, command_fn, rate, hooks, span, phase=""):
    """Execute a timed reference at ``rate`` Hz, sampling feedback each tick.

    ``command_fn(i, t)`` sends one MIT frame and returns the commanded velocity.
    ``span`` = ``(lo, hi)`` maps this segment onto the overall 0..1 progress bar.
    ``phase`` tags the live samples pushed to ``hooks.sample`` (for live plots).
    Returns arrays ``(t, q, w, tau, w_cmd)``.  Raises :class:`Aborted` if asked.
    """
    dt = 1.0 / rate
    n = t_ref.size
    ts = np.zeros(n); qs = np.zeros(n); ws = np.zeros(n)
    taus = np.zeros(n); wcmd = np.zeros(n)
    lo, hi = span
    stride = max(1, int(round(rate / SAMPLE_HZ)))   # push ~SAMPLE_HZ live samples
    t0 = time.perf_counter()
    for i in range(n):
        target = t0 + t_ref[i]
        wcmd[i] = command_fn(i, t_ref[i])
        fb = bus.get(can_id)
        ts[i] = time.perf_counter() - t0
        qs[i], ws[i], taus[i] = fb.pos, fb.vel, fb.tau
        if i % stride == 0:
            hooks.sample(phase, ts[i], qs[i], ws[i], taus[i], wcmd[i])
        if (i & 0x3F) == 0:                       # ~every 64 ticks
            hooks.progress(lo + (hi - lo) * (i / max(1, n)))
            if hooks.should_abort():
                raise Aborted()
        while time.perf_counter() < target + dt:
            pass
    hooks.progress(hi)
    return ts, qs, ws, taus, wcmd


def run_collection(bus, profile: MotorProfile, can_id: int, cfg: CollectConfig,
                   hooks: Hooks = Hooks(), name: Optional[str] = None
                   ) -> dict:
    """Drive staircase (+ optional chirp + torque-speed) and save the logs.

    The bus must already be constructed; this enables the motor on entry and
    ALWAYS disables it on exit (including on abort / error).  ``name`` is the
    instance/label the results are filed under (``results/<name>/...``); it
    defaults to the motor *type* name so a quick run still works.  ``profile``
    always supplies the physical ranges.  Returns a dict of the logs by phase.
    """
    motor = name or profile.name        # instance key for the results directory
    v_max = cfg.vel_frac * profile.v_max
    hooks.log(f"motor {motor}  id {can_id}  rated v_max {profile.v_max:.1f} rad/s "
              f"-> using {v_max:.2f} rad/s")

    outdir = result_dir(motor, "logs")

    # allocate the 0..1 progress bar across the enabled phases
    _wt = {"staircase": 2.0, "chirp": 2.0, "torque_speed": 1.0}
    _enabled = (["staircase"] + (["chirp"] if cfg.chirp else [])
                + (["torque_speed"] if cfg.torque_speed else []))
    _tot = sum(_wt[p] for p in _enabled)
    spans, _acc = {}, 0.0
    for _p in _enabled:
        _lo = _acc / _tot
        _acc += _wt[_p]
        spans[_p] = (_lo, _acc / _tot)

    try:
        bus.clear_faults(can_id)
        time.sleep(0.05)
        bus.enable(can_id)
        time.sleep(0.2)
        fb = bus.get(can_id)
        # byte0 high nibble is a STATE code (0=disabled,1=enabled); >=8 are faults
        if fb.err >= 8:
            hooks.log(f"WARNING: motor fault code 0x{fb.err:X}")
        elif fb.err != 1:
            hooks.log(f"WARNING: motor not enabled (state {fb.err}); check power/CAN")

        # ---- staircase ----------------------------------------------------
        hooks.phase("staircase")
        stair = velocity_staircase(v_max=v_max, n_levels=cfg.levels,
                                   dwell=cfg.dwell, settle=0.5, rate=cfg.rate)
        hooks.log(f"staircase: {stair.levels.size} levels, ~{stair.t[-1]:.0f}s "
                  f"... keep clear of the shaft")

        if cfg.stair_kp > 0:
            # Realise the velocity staircase as a POSITION RAMP, re-anchored to the
            # motor's actual position at each level, so the motor is forced through
            # stiction even at low speed (pure velocity-hold stalls below break-away).
            excursion = float(np.max(np.abs(stair.levels))) * cfg.dwell
            hooks.log(f"  position-ramp drive (kp={cfg.stair_kp}, kd={cfg.kd}); "
                      f"per-level excursion <= {excursion:.2f} rad")
            if excursion > 0.9 * profile.p_max:
                raise ValueError(
                    f"ramp excursion {excursion:.2f} rad exceeds 90% of p_max "
                    f"{profile.p_max:.2f}; lower vel_frac or dwell")
            _anchor = {"v": None, "q0": 0.0, "t0": 0.0}

            def stair_cmd(i, t):
                v = float(stair.w_cmd[i])
                if _anchor["v"] is None or v != _anchor["v"]:
                    _anchor.update(v=v, q0=bus.get(can_id).pos, t0=t)
                q_ref = _anchor["q0"] + v * (t - _anchor["t0"])
                bus.track_position(can_id, q_ref, v, cfg.stair_kp, cfg.kd)
                return v
        else:
            def stair_cmd(i, _t):
                bus.hold_velocity(can_id, stair.w_cmd[i], cfg.kd)
                return stair.w_cmd[i]

        ts, qs, ws, taus, wcmd = _run_reference(
            bus, can_id, stair.t, stair_cmd, cfg.rate, hooks, spans["staircase"],
            phase="staircase")
        stair_log = CalibrationLog(
            t=ts, q=qs, w=ws, a=np.full_like(ts, np.nan), tau=taus,
            w_cmd=wcmd, steady=stair.steady, motor=motor,
            note="velocity staircase (hardware)", rate=cfg.rate)
        p1 = os.path.join(outdir, "staircase.csv")
        stair_log.save(p1)
        hooks.log(f"  saved {os.path.relpath(p1, ROOT)}")

        # gently bring velocity back to zero before the next test
        for w in np.linspace(stair.w_cmd[-1], 0.0, int(0.5 * cfg.rate)):
            bus.hold_velocity(can_id, float(w), cfg.kd)
            time.sleep(1.0 / cfg.rate)

        # ---- chirp --------------------------------------------------------
        chirp_log = None
        if cfg.chirp:
            hooks.phase("chirp")
            t, q, w, a = chirp_profile(cfg.chirp_amp, cfg.chirp_f0,
                                       cfg.chirp_f1, cfg.chirp_dur, cfg.rate)
            q0 = bus.get(can_id).pos
            hooks.log(f"chirp: {cfg.chirp_f0}->{cfg.chirp_f1} Hz, "
                      f"{cfg.chirp_dur:.0f}s, amp {cfg.chirp_amp} rad")

            def chirp_cmd(i, _t):
                bus.track_position(can_id, q0 + q[i], w[i],
                                   cfg.chirp_kp, cfg.chirp_kd)
                return w[i]

            ts, qs, ws, taus, wcmd = _run_reference(
                bus, can_id, t, chirp_cmd, cfg.rate, hooks, spans["chirp"],
                phase="chirp")
            acc = identify.differentiate(ts, ws, window=0.02)
            chirp_log = CalibrationLog(
                t=ts, q=qs, w=ws, a=acc, tau=taus, w_cmd=wcmd,
                steady=np.zeros_like(ts, bool), motor=motor,
                note="frequency chirp (hardware)", rate=cfg.rate)
            p2 = os.path.join(outdir, "chirp.csv")
            chirp_log.save(p2)
            hooks.log(f"  saved {os.path.relpath(p2, ROOT)}")

        # ---- torque-speed envelope ----------------------------------------
        ts_log = None
        if cfg.torque_speed:
            hooks.phase("torque-speed")
            ts_log = _run_torque_speed(bus, can_id, profile, cfg, hooks,
                                       spans["torque_speed"])
            p3 = os.path.join(outdir, "torque_speed.csv")
            ts_log.save(p3)
            hooks.log(f"  saved {os.path.relpath(p3, ROOT)}")
        return {"staircase": stair_log, "chirp": chirp_log, "torque_speed": ts_log}
    finally:
        try:
            bus.disable(can_id)
        except Exception:
            pass


def _run_torque_speed(bus, can_id, profile, cfg, hooks, span):
    """Max-effort free-acceleration sweep, both directions, to trace the motor's
    torque–speed envelope (Isaac DCMotor: ``saturation_effort``/``velocity_limit``).

    SAFETY: SHAFT MUST BE FREE.  Pure-torque command ramped to ``ts_frac`` of
    rated torque; speed is hard-capped at ``ts_ceil`` of rated v_max (cut + brake
    on reach), and each direction is time-bounded by ``ts_dur``.
    """
    rate = cfg.rate
    dt = 1.0 / rate
    stride = max(1, int(round(rate / SAMPLE_HZ)))
    w_ceil = cfg.ts_ceil * profile.v_max
    tau_cmd_max = cfg.ts_frac * profile.t_max
    lo, hi = span
    hooks.log(f"torque-speed: ±{tau_cmd_max:.1f} N·m, speed ceiling "
              f"±{w_ceil:.1f} rad/s ... SHAFT MUST BE FREE")

    seg_t, seg_q, seg_w, seg_tau = [], [], [], []
    for k, sign in enumerate((+1.0, -1.0)):
        t0 = time.perf_counter()
        while True:
            el = time.perf_counter() - t0
            ramp = min(1.0, el / max(cfg.ts_ramp, 1e-3))
            bus.torque(can_id, sign * ramp * tau_cmd_max)
            fb = bus.get(can_id)
            seg_t.append(el if sign > 0 else cfg.ts_dur + el)
            seg_q.append(fb.pos); seg_w.append(fb.vel); seg_tau.append(fb.tau)
            if len(seg_w) % stride == 0:
                hooks.sample("torque_speed", seg_t[-1], fb.pos, fb.vel, fb.tau, 0.0)
                hooks.progress(lo + (hi - lo) * (0.5 * k + 0.5 * min(1.0, el / cfg.ts_dur)))
                if hooks.should_abort():
                    raise Aborted()
            if abs(fb.vel) >= w_ceil or el >= cfg.ts_dur:
                break
            time.sleep(dt)
        # brake back toward zero before reversing
        for _ in range(int(0.6 * rate)):
            bus.hold_velocity(can_id, 0.0, cfg.kd)
            time.sleep(dt)

    hooks.progress(hi)
    t = np.asarray(seg_t); w = np.asarray(seg_w); tau = np.asarray(seg_tau)
    acc = identify.differentiate(t, w, window=0.02) if t.size > 10 else np.zeros_like(t)
    return CalibrationLog(
        t=t, q=np.asarray(seg_q), w=w, a=acc, tau=tau,
        w_cmd=np.zeros_like(t), steady=np.zeros_like(t, bool),
        motor=profile.name, note="torque-speed sweep (hardware)", rate=rate)


# --------------------------------------------------------------------------- #
#  identification + plotting (offline)                                        #
# --------------------------------------------------------------------------- #
def identify_and_save(name: str, *, motor_type: Optional[str] = None,
                      model: str = "stribeck", hooks: Hooks = Hooks(),
                      timestamp: Optional[str] = None):
    """Fit the model to ``results/<name>/logs/`` and write the FINAL calibration.

    ``name``        instance/label the run is filed under (``results/<name>/``).
    ``motor_type``  the DaMiao model (a ``MOTORS`` key) supplying the physical
                    ranges + datasheet envelope; defaults to ``name``.

    Writes, under ``results/<name>/``:
      * ``calibration.json`` — the full record (name, type, profile, all fitted
        params, fit quality, MuJoCo attrs, Isaac DCMotorCfg w/ provenance, plots);
      * ``mujoco_joint.xml`` — a ready-to-paste ``<joint .../>`` snippet;
      * ``isaaclab_actuator.py`` — a ready-to-paste ``DCMotorCfg`` snippet;
      * ``params.json`` — kept for back-compat (export_mujoco.py etc.);
      * ``plots/*.png``.
    Returns the :class:`identify.ActuatorResult`.
    """
    motor_type = motor_type or name
    logs = RESULTS_DIR / name / "logs"
    stair = CalibrationLog.load(logs / "staircase.csv")
    chirp_path = logs / "chirp.csv"
    dyn = CalibrationLog.load(chirp_path) if chirp_path.exists() else None

    res = identify.identify_actuator(stair, dynamic=dyn, model=model)
    p = res.params
    hooks.log(f"friction fit: R^2 {res.curve.r2:.5f}  RMS {res.curve.rms:.4f} N*m "
              f"({res.report['n_levels']} levels, model={model})")
    if res.inertia is not None:
        hooks.log(f"inertia fit:  J {p.J:.5f} kg*m^2  RMS {res.inertia.rms:.4f} N*m")
    hooks.log("MuJoCo: " + "  ".join(f'{k}={v}'
              for k, v in p.mujoco_attrs().items()))

    plots = Path(result_dir(name, "plots"))
    w_lvl, tau_lvl = stair.steady_points()
    metrics.plot_friction_curve(w_lvl, tau_lvl, p,
                                str(plots / "friction_curve.png"),
                                title=f"{name} friction curve")
    if dyn is not None:
        metrics.plot_torque_fit(dyn, p, str(plots / "torque_fit.png"),
                                title=f"{name} dynamic torque fit")

    # ---- torque-speed envelope (Isaac DCMotor) ----------------------------
    from .motors import MOTORS
    profile = MOTORS.get(motor_type)
    ts_path = logs / "torque_speed.csv"
    dc = None
    if ts_path.exists():
        tslog = CalibrationLog.load(ts_path)
        dc = identify.fit_torque_speed(
            tslog.w, tslog.tau,
            tau_cont=(profile.tau_cont if profile else 0.0),
            w_max_hint=((profile.w_noload or profile.v_max) if profile else 0.0))
        if dc.observed:
            hooks.log(f"torque-speed: stall {dc.tau_stall:.1f} N*m, no-load "
                      f"{dc.w_noload:.1f} rad/s (loaded droop fit, R^2 {dc.r2:.3f})")
        else:
            hooks.log(f"torque-speed: no-load speed {dc.w_noload:.1f} rad/s MEASURED; "
                      f"stall torque needs a load → using datasheet peak.")
        metrics.plot_torque_speed(tslog, dc, str(plots / "torque_speed.png"),
                                  title=f"{name} torque-speed envelope")

    isaac = isaaclab_block(p, dc, profile)
    hooks.log("Isaac DCMotor: " + "  ".join(
        f"{k}={v['value']}" for k, v in isaac.items() if v['value'] is not None))

    # which plot files actually exist
    plot_files = {n: f"plots/{n}.png" for n in
                  ("friction_curve", "torque_fit", "torque_speed",
                   "staircase_data", "chirp_data")
                  if (plots / f"{n}.png").exists()}

    record = {
        "name": name,
        "motor_type": motor_type,
        "timestamp": timestamp or "",
        "model": model,
        "profile": (dataclasses.asdict(profile) if profile else None),
        "fit_quality": {
            "friction_r2": res.report.get("friction_r2"),
            "friction_rms_Nm": res.report.get("friction_rms_Nm"),
            "inertia_rms_Nm": res.report.get("inertia_rms_Nm"),
            "n_levels": res.report.get("n_levels"),
        },
        "params": res.report.get("params"),
        "dc_motor": (dc.as_dict() if dc is not None else None),
        "mujoco": p.mujoco_attrs(),
        "isaaclab": isaac,
        "plots": plot_files,
    }
    base = RESULTS_DIR / name
    (base / "calibration.json").write_text(json.dumps(record, indent=2))
    # back-compat params.json (flat report + isaac)
    report = {"motor": name, **res.report}
    if dc is not None:
        report["dc_motor"] = dc.as_dict()
    report["isaaclab"] = isaac
    (base / "params.json").write_text(json.dumps(report, indent=2))
    # ready-to-paste sim configs
    (base / "mujoco_joint.xml").write_text(_mujoco_snippet(name, motor_type, p))
    (base / "isaaclab_actuator.py").write_text(_isaaclab_snippet(name, isaac))
    hooks.log(f"wrote {os.path.relpath(base / 'calibration.json', ROOT)} "
              f"+ mujoco_joint.xml + isaaclab_actuator.py + plots")
    return res


def _mujoco_snippet(name: str, motor_type: str, p) -> str:
    a = p.mujoco_attrs()
    return (f'<!-- MABEL actuator calibration: {name}  (type {motor_type}) -->\n'
            f'<!-- frictionloss=Coulomb friction, damping=viscous, armature=reflected inertia -->\n'
            f'<joint name="{name}" '
            f'frictionloss="{a["frictionloss"]}" '
            f'damping="{a["damping"]}" '
            f'armature="{a["armature"]}"/>\n')


def _isaaclab_snippet(name: str, isaac: dict) -> str:
    def g(k):
        return isaac.get(k, {}).get("value")
    lines = [f"# MABEL actuator calibration: {name}",
             "# † MEASURED on the bench; others are datasheet/policy — see comments.",
             "from isaaclab.actuators import DCMotorCfg",
             "",
             f'{name}_actuator = DCMotorCfg(',
             '    joint_names_expr=[".*"],']
    for k in ("effort_limit", "saturation_effort", "velocity_limit",
              "stiffness", "damping", "armature", "friction"):
        v = g(k)
        src = isaac.get(k, {}).get("source", "")
        if k == "stiffness" and v is None:
            val = '{".*": 0.0}'
        elif k == "damping" and v is None:
            val = '{".*": 0.0}'
        elif v is None:
            val = "None  # <-- FILL FROM DATASHEET"
        else:
            val = repr(v)
        lines.append(f"    {k}={val},  # {src}")
    lines.append(")")
    return "\n".join(lines) + "\n"


def isaaclab_block(fp, dc, profile) -> dict:
    """Assemble the Isaac Lab ``DCMotorCfg`` parameters with a provenance tag on
    each, so it's explicit which were *measured* on the bench vs taken from the
    datasheet vs left to the control policy.

    ``fp`` FrictionParams, ``dc`` DCMotorParams|None, ``profile`` MotorProfile|None.
    """
    t_max = profile.t_max if profile else 0.0
    v_max = profile.v_max if profile else 0.0
    cont = (profile.tau_cont if profile and profile.tau_cont else 0.0)

    # velocity_limit: the free-shaft sweep DOES measure this (terminal speed)
    vlim_measured = bool(dc and dc.w_noload > 0)
    vlim = (dc.w_noload if vlim_measured else
            (profile.w_noload if profile and profile.w_noload else (v_max or None)))
    # saturation_effort: only "measured" if a real droop was fit (needs a load)
    sat_measured = bool(dc and dc.observed)
    sat = (dc.tau_stall if sat_measured else
           (profile.tau_stall if profile and profile.tau_stall else (t_max or None)))

    def f(x, n):                          # plain python float (no np.float64 repr)
        return round(float(x), n) if x is not None else None

    return {
        "armature":          {"value": f(fp.J, 8),  "source": "MEASURED (chirp inertia sweep)"},
        "friction":          {"value": f(fp.Fc, 6), "source": "MEASURED (velocity staircase)"},
        "velocity_limit":    {"value": f(vlim, 4),
                              "source": ("MEASURED (max-effort terminal speed ≈ no-load)"
                                         if vlim_measured else
                                         "datasheet / MIT v_max (sweep not run)")},
        "saturation_effort": {"value": f(sat, 3),
                              "source": ("MEASURED (loaded torque-speed droop)" if sat_measured else
                                         "DATASHEET / peak t_max — FLAG: stall torque needs a load "
                                         "(dynamometer/brake); not measurable on a free shaft")},
        "effort_limit":      {"value": (f(cont, 3) if cont else None),
                              "source": "DATASHEET — FLAG: continuous/thermal limit, not bench-measurable"},
        "damping":           {"value": f(fp.Fv, 6),
                              "source": "viscous Fv MEASURED; but Isaac PD `damping` is a CONTROL gain — "
                                        "set per policy (this is the physical value for reference)"},
        "stiffness":         {"value": None,
                              "source": "CONTROL POLICY — choose per task (not a physical parameter)"},
    }


def render_raw_plots(motor: str, hooks: Hooks = Hooks()) -> None:
    """Eyeball plots of the raw staircase / chirp data (independent of the fit)."""
    # plot_data.py lives in hardware/ and imports its sibling `_bootstrap`, so
    # the hardware dir must be importable before we exec it.
    import importlib.util
    hwdir = ROOT / "hardware"
    if str(hwdir) not in sys.path:
        sys.path.insert(0, str(hwdir))
    spec = importlib.util.spec_from_file_location("_plot_data", hwdir / "plot_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    logs = RESULTS_DIR / motor / "logs"
    plots = Path(result_dir(motor, "plots"))
    if (logs / "staircase.csv").exists():
        mod.plot_staircase(str(logs / "staircase.csv"), False,
                           str(plots / "staircase_data.png"))
    if (logs / "chirp.csv").exists():
        mod.plot_dynamic(str(logs / "chirp.csv"), False,
                         str(plots / "chirp_data.png"))
    hooks.log("raw data plots updated")

"""
SimWorld — the in-process MuJoCo test bench for the `mabel` control library.

Unlike the other studios (which proxy the out-of-process teleop bridge), this
backend OWNS physics: a background thread paces mj_step at real time and runs
the library controllers directly, with per-stage wall-clock instrumentation.

WHAT CHANGED (2026-07): the lab is now a MINI-GATEWAY around the SAME control
seam the iOS / Vision Pro apps drive on the real robot — `mabel.MabelTeleop`.
It mirrors the apps' four control axes 1:1, so the input here is identical to
the input there (the wire strings and facade calls match `server/
sim_teleop_bridge.py`):

    TORQUE ON/OFF   the master gate. OFF => arms limp (servos neutralised, zero
                    effort — the honest analog of the robot's set_arms_torque
                    (False)); boot state. ON => resume_hold() makes the command
                    equal the MEASURED pose, so torque holds where it is now (no
                    lurch), exactly like the app's ENABLE TORQUE.
    STIFF / SOFT    control_type impedance (K>0, springs to the target) vs
                    compliance (K≈0 gravity-comp float, fully back-drivable, and
                    in whole_body the base admits the push and glides). Engages
                    in the Joint / Wrist methods (the WBC torque engine).
    REGION          arm | upper_body | whole_body — grows the torque law outward.
    MODE (method)   navigation (drive: base joystick) · joint (sliders ->
                    impedance targets) · wrist (task-space: WholeBodyIK ->
                    impedance targets) · retarget (whole-body operator teleop
                    from a Vision Pro stream or the mock operator).

Every plotted number is measured, never synthesized: timings via
time.perf_counter around the real calls, wrench truth from the injected
xfrc_applied, tracking errors from FK on the live model. The research
instrumentation (ZMP tip-usage, command-odometry, wrench estimation, MPJPE /
jerk / latency) is re-homed under the mode that produces it.
"""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

import numpy as np
import mujoco

import sys
from . import config as CFG
sys.path.insert(0, str(CFG.CONTROLLER_DIR))

import mabel.config as C                                            # noqa: E402
from mabel.facade import MabelTeleop                                # noqa: E402
from mabel.whole_body_controller import WholeBodyIK, swerve_ik, BaseFollow  # noqa: E402
from mabel.whole_body_controller.engine import _MJ_TO_IOS           # noqa: E402
from mabel.whole_body_controller.admittance import ContactEstimator # noqa: E402
from mabel.whole_body_controller.admittance.wrench_estimation import (  # noqa: E402
    surface_point, body_subtree)
from mabel.motion_model import tip_usage, max_speed_dir             # noqa: E402
from mabel.motion_model.base_response import BaseMotionModel        # noqa: E402
from mabel.dynamic_models.regions import region_joints              # noqa: E402
from mabel.dynamic_models.model import _dof                        # noqa: E402
from mabel.types import OperatorPose                                # noqa: E402
from mabel.sources.wire import app_joystick                         # noqa: E402

from . import joints as J                                          # noqa: E402

# The four control axes, mirroring the app / wire protocol exactly.
METHODS = ("navigation", "joint", "wrist", "retarget")
CONTROL_TYPES = ("impedance", "compliance")
REGIONS = ("arm", "upper_body", "whole_body")
WBC_METHODS = ("joint", "wrist")        # methods the compliance (Soft) engine can run
DEFAULT_STIFFNESS = 20.0                # Nm/rad — Soft/impedance tunable (unused in position Stiff)
# IK actuator -> wire joint id (arms + torso from the engine map, + the ORCA wrists,
# so the wrist-IK targets reach the position servos through _apply_joint_targets).
_IK_WIRE = {**_MJ_TO_IOS, "Left Wrist": "left_wrist", "Right Wrist": "right_wrist"}

CONTROL_DECIM = 2           # control every 2nd step (= 125 Hz at dt 4 ms)
NAV_FRAME_HZ = 60.0         # headless drive "VR frame" cadence (see _control)
TS_DRIVE = float(os.environ.get("MABEL_CTS_TIMESTEP_DRIVE", "0.008"))
                            # coarse DRIVE physics step (see set_control_spec)

# Overlay skeleton points sent to the viewer (subset of the ARKit skeleton):
# fingertips + knuckles per hand, so the operator's HAND SHAPE reads in the 3D
# view next to the robot (whole-body retargeting debug).
_TIP_JOINTS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
               "thumb_knuckle", "index_knuckle", "middle_knuckle",
               "ring_knuckle", "little_knuckle")


class _Timer:
    """Exponentially-smoothed per-stage wall-clock timing [ms]."""
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.ms = {}
        self.peak = {}

    def add(self, key, dt_s):
        ms = dt_s * 1e3
        cur = self.ms.get(key)
        self.ms[key] = ms if cur is None else (1 - self.alpha) * cur + self.alpha * ms
        self.peak[key] = max(self.peak.get(key, 0.0), ms)

    def snapshot(self):
        return {k: round(v, 3) for k, v in self.ms.items()}, \
               {k: round(v, 2) for k, v in self.peak.items()}


class SimWorld:
    def __init__(self, scene=CFG.SCENE, state_file=None):
        if state_file is None:
            state_file = CFG.STATE_FILE
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        # Realtime on a laptop: raw mj_step on this scene costs ~3.2 ms wall, so
        # the native 2 ms timestep cannot pace 1x. wbc_stability validated this
        # scene at dt = 4 ms — use that.
        self.model.opt.timestep = float(CFG.TIMESTEP)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        self.lock = threading.RLock()

        self.qadr, self.mj_of_wire = J.build_maps(self.model)
        self._scratch = mujoco.MjData(self.model)   # FK of the reference

        # Position-servo discovery + the actuator<->wire-id map (for the wrist-IK
        # -> joint-target bridge, and the servo-restore on torque enable).
        self._pos_acts, self._act_id, self._act_jq = [], {}, {}
        for aid in range(self.model.nu):
            if self.model.actuator_biasprm[aid, 1] < 0.0:   # position servo
                nm = self.model.actuator(aid).name
                self._pos_acts.append(nm)
                self._act_id[nm] = aid
                try:
                    self._act_jq[nm] = int(self.model.joint(nm).qposadr[0])
                except Exception:
                    pass                   # tendon (Lift)

        self._chassis = self.model.body(C.CHASSIS_BODY).id
        # The GLB rig's base_link corresponds to the MJCF ROOT (free-joint) body,
        # NOT the chassis body (child offset ~0.23 m in x). Feeding chassis coords
        # shifted the whole rendered robot off the true world frame.
        self._root = self._chassis
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self._root = int(self.model.jnt_bodyid[j])
                break
        self._palm_l = self.model.body(C.LEFT_PALM).id
        self._palm_r = self.model.body(C.RIGHT_PALM).id
        try:
            self._head = self.model.body(C.HEAD_BODY).id
        except Exception:
            self._head = None
        # Palm-LOCAL forward unit vectors (wrist -> middle finger mount), used to
        # draw/drag the wrist anchor at the VISUAL palm centre.
        self._palm_fwd = {}
        d0 = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, d0)
        for side, bid in (("l", self._palm_l), ("r", self._palm_r)):
            try:
                mp = self.model.body(f"{'left' if side == 'l' else 'right'}_middle_mp").id
                R = d0.xmat[bid].reshape(3, 3)
                f = R.T @ (d0.xpos[mp] - d0.xpos[bid])
                self._palm_fwd[side] = f / (np.linalg.norm(f) + 1e-9)
            except Exception:
                self._palm_fwd[side] = np.zeros(3)
        self.PALM_VIS_OFFSET = 0.5 * C.HAND_REACH   # ~palm centre [m]

        # Snapshot the pristine actuator gains so torque-enable can restore the
        # servos the limp state neutralised.
        self._gains0 = self.model.actuator_gainprm.copy()
        self._bias0 = self.model.actuator_biasprm.copy()
        self._qpos0 = None                     # settled home, captured in start()

        # ── THE control seam: one MabelTeleop, driven exactly like the bridge ──
        self.teleop = MabelTeleop(self.model, self.data)
        self.plant = self.teleop.wbc.plant     # SimPlant(model, data) — reuse it

        # App-mirror control state (defaults = the facade's own defaults).
        self.torque_on = False                 # boot LIMP (arms limp on connect)
        self.spec = {"method": "navigation", "type": "impedance",
                     "region": "arm", "stiffness": DEFAULT_STIFFNESS}
        # Boot method is navigation → adopt the coarse DRIVE timestep from the
        # start (set_control_spec applies the same policy on every change).
        self.model.opt.timestep = TS_DRIVE
        self.dt = TS_DRIVE
        self.self_collision = bool(C.SELF_COLLISION_ENABLE)
        self.sc_method = "cbf"                  # IK self-collision law: slide, don't stop
        self.wrist_human = True                # anatomical IK posture (VP parity)
        self.hand_side = "both"

        self.nav = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0}
        self.timer = _Timer()
        self._t = 0.0
        self._tickno = 0
        self._limped = False                   # arm servos currently neutralised

        # wrist-IK task-space bench (lazy)
        self.ik: WholeBodyIK | None = None

        # diagnostics: external wrench injector + trajectory generator
        self.push = None                       # {"body","bid","F","point","tau","until"}
        self.traj = {"kind": "hold", "joint": "left_arm_2", "amp": 0.3,
                     "period": 4.0, "t0": 0.0, "q0": {}}
        self.estimator: ContactEstimator | None = None   # wrench estimation in Soft
        self._est_w = None                     # last estimated contact wrench (persists between heavy ticks)
        self._tau_sum = np.zeros(self.model.nv)
        self._tau_n = 0

        # command odometry (drive mode) + the whole-body base drive: sticks OR
        # (Stiff wrist) the task-space follower, arbitrated + tip-gated inside
        # BaseFollow (the ported sim-teleop whole-body cascade).
        self.odom = BaseMotionModel()
        self._last_base = (0.0, 0.0, 0.0)      # last commanded safe twist (vx, vy, w)
        self.base_follow = BaseFollow(self.model)
        self._base_v = (0.0, 0.0, 0.0)         # mirror of base_follow's realised twist

        # BIDIRECTIONAL memmap for the external mjpython viewer mirror.
        self._mm = None
        self._pert_off = 1 + self.model.nq
        self._viewer_push = None
        if state_file:
            try:
                self._mm = np.memmap(state_file, dtype=np.float64, mode="w+",
                                     shape=(1 + self.model.nq + 8,))
                self._mm[:] = 0.0
            except OSError:
                self._mm = None
        self._running = False
        self._thread = None
        self.status_msg = "arms limp — press ENABLE TORQUE"

        # retarget pose ingest (lock-free mailbox, like the bridge)
        self._pose = None
        self._mailbox = None
        self._pose_stamp = 0.0
        self._pose_seq = 0
        self._frame_gaps = deque(maxlen=120)
        self._last_frame_wall = None
        self.vp_status = {"source": "off", "device": None, "connected": False,
                          "rate_hz": 0.0, "jitter_ms": 0.0, "age_ms": None,
                          "frames": 0, "calibration": "none"}
        self._ee_hist = deque(maxlen=8)

        # plot ring buffers + 3D overlays
        self.plots = {}
        self._plot_next = 0.0
        self.overlay = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        self.settle()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="simworld")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _seed_model_defaults(self):
        """Apply the MJCF's `home` keyframe (qpos + ctrl) if the model ships one.

        The lift is assembled HALF-RAISED in the MJCF, and its position servo is
        stiff (kp 10000): leaving ctrl at zero drags the stages to the bottom
        within ~0.5 s, so settling from a bare MjData would capture a bottomed
        home no matter what the model's default pose says. Keyframe 0 carries the
        matching lift command — the same reset the sim bridge uses
        (simulation_bridge/engine.py). Models without a keyframe are unaffected.
        """
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)

    def settle(self, steps=800):
        """Settle to gravity equilibrium with the servos ON (home pose), capture
        it, then drop into the boot LIMP state (arms limp until ENABLE TORQUE)."""
        self._seed_model_defaults()
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self._qpos0 = self.data.qpos.copy()
        # capture the upright home base pose — the viewer renders the base as a
        # rigid DELTA from this, so falls / tip-over show (the rig node is placed
        # by base {x,y,yaw} + the height drop dz + the body-frame rotation qd).
        rb = self.data.body(self._root)
        self._root_home_z = float(rb.xpos[2])
        self._root_home_quat = np.array(rb.xquat, dtype=float)
        self.teleop.controller.hold_current()      # every servo := measured pose
        self._go_limp()                            # boot limp

    def _tick_once(self):
        """One physics step + control + decimated heavy work + plot sampling.

        The TORQUE law runs EVERY physics step (250 Hz) — a stiff impedance
        spring under an 8 ms (125 Hz) ZOH goes unstable, so the facade's control
        law is stepped at the physics rate (matching the real sim bridge). The
        EXPENSIVE, decimated work (wrist-IK QP solve, momentum observer, native-
        viewer read) runs every CONTROL_DECIM-th step."""
        with self.lock:
            self._consume_mailbox()
            heavy = (self._tickno % CONTROL_DECIM == 0)
            if heavy:
                self._read_viewer_push()
            t0 = time.perf_counter()
            try:
                self._control(heavy)
            except Exception as e:          # never kill physics on a bug
                self.status_msg = f"control error: {e!r}"
            self.timer.add("control_ms", time.perf_counter() - t0)
            t1 = time.perf_counter()
            mujoco.mj_step(self.model, self.data)
            self.timer.add("physics_ms", time.perf_counter() - t1)
            if self.estimator is not None:
                # hand the observer the interval MEAN torque, not a point sample
                # (a fast-slewing joint's ZOH mismatch reads as a phantom torque).
                self._tau_sum += self.data.qfrc_applied
                self._tau_sum += self.data.qfrc_actuator
                self._tau_n += 1
            self._t += self.dt
            self._tickno += 1
            if self._mm is not None and self._tickno % 4 == 0:
                self._mm[0] = self._t
                self._mm[1:self._pert_off] = self.data.qpos
            if self._t >= self._plot_next:
                self._plot_next = self._t + 1.0 / CFG.PLOT_HZ
                try:
                    self._sample_plots()
                except Exception as e:
                    self.status_msg = f"plot error: {e!r}"

    def _loop(self):
        next_t = time.perf_counter()
        while self._running:
            next_t += self.dt
            self._tick_once()
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:                               # behind: resync AND yield so the
                next_t = time.perf_counter()    # ingest/WS threads never starve
                time.sleep(0.001)

    def step_sync(self, seconds):
        """Advance synchronously (experiment harness + tests). Only valid while
        the realtime thread is NOT running."""
        if self._running:
            raise RuntimeError("step_sync while the realtime thread is running")
        for _ in range(int(round(seconds / self.dt))):
            self._tick_once()

    # ── torque gate (the master ON/OFF) ─────────────────────────────────────
    def set_torque(self, on):
        """Master torque ON/OFF, mirroring the app's ENABLE TORQUE / E-STOP.

        ON  : resume_hold() so the COMMANDED pose == the MEASURED pose (hold
              where it is now, no lurch), restore the servos the limp state
              neutralised, and re-arm the active mode.
        OFF : neutralise the arm/wrist position servos and apply zero effort so
              the arms go limp (the sim analog of set_arms_torque(False)); base
              stops. Everything else keeps holding.
        """
        with self.lock:
            on = bool(on)
            if on == self.torque_on:
                return
            self.torque_on = on
            if on:
                self._restore_servos()
                self.teleop.resume_hold()          # command := measured, no lurch
                self.teleop._wbc_dirty = True      # re-enter the torque engine if WBC
                self._limped = False
                self._sync_estimator()
                self.status_msg = "TORQUE ON — holding current pose"
            else:
                self._go_limp()

    def _arm_actuators(self):
        """Arm + ORCA-wrist position servos (the joints that go limp)."""
        out = []
        for j in region_joints("arm"):          # 7 arm + wrist per side
            try:
                out.append(self.model.actuator(j).name)
            except KeyError:
                pass
        return out

    def _go_limp(self):
        """Neutralise the arm servos and stop the wheels — the LIMP state."""
        if self.teleop.wbc.active:
            self.teleop.wbc.exit()              # restore, then we re-neutralise
        for a in self._arm_actuators():
            try:
                self.plant.disable_servo(a)     # gain/bias := 0 -> no servo force
            except KeyError:
                pass
        # zero every velocity/effort actuator (wheels) so the base cannot creep
        for aid in range(self.model.nu):
            if self._bias0[aid, 1] >= 0.0:
                self.data.ctrl[aid] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.push = None
        self.torque_on = False
        self._limped = True
        self.estimator = None
        self.base_follow.reset()               # no stale twist on re-enable
        self._base_v = (0.0, 0.0, 0.0)
        self.status_msg = "arms limp — press ENABLE TORQUE"

    def _restore_servos(self):
        """Undo the limp neutralisation: restore the pristine gains."""
        self.model.actuator_gainprm[:] = self._gains0
        self.model.actuator_biasprm[:] = self._bias0

    def _limp_hold(self):
        """The de-energised LIMP torque law: servos neutralised, arms driven by
        gravity/bias feedforward + light damping only (no spring, no target).

        This is the SIM analog of the real arm's torque-off "damping hold" — the
        arm is fully back-drivable and floats in place. (True zero torque is
        physically honest but useless on a bench: with no motor holding them the
        heavy arms free-fall through gravity and collapse into deep self/body
        contact, overflowing the contact solver.) It accepts NO control input and
        engages NO whole-body admittance — that is what separates it from Soft."""
        b = self.teleop.wbc.dyn.bias()
        self.plant.begin()
        for j in region_joints("arm"):
            self.plant.apply_effort(j, b[_dof(self.model, j)] - 0.8 * self.plant.qd(j))
        self.plant.commit()

    # ── control-spec + input passthrough (the app wire mapping) ─────────────
    def set_control_spec(self, method=None, control_type=None, region=None,
                         stiffness=None):
        """control_mode message: method | control_type | region | stiffness.
        Mirrors sim_teleop_bridge -> MabelTeleop.set_control_spec exactly."""
        with self.lock:
            if method is not None:
                if method not in METHODS:
                    raise ValueError(f"unknown method {method!r}")
                self.spec["method"] = method
            if control_type is not None:
                if control_type not in CONTROL_TYPES:
                    raise ValueError(f"unknown control_type {control_type!r}")
                self.spec["type"] = control_type
            if region is not None:
                if region not in REGIONS:
                    raise ValueError(f"unknown region {region!r}")
                self.spec["region"] = region
                if self.ik is not None:            # wrist bench: region changes DOFs
                    self._sync_ik_scratch(from_target=False)
                    self.ik.sync_to_current(freeze_posture=True)
            if stiffness is not None:
                self.spec["stiffness"] = max(0.0, float(stiffness))
            # entering / leaving a WBC method flips the wrench-estimation mount
            self._sync_estimator()
            # MODE-DEPENDENT TIMESTEP (the bridge's own real-time trick,
            # sim_teleop_bridge TS_DRIVE/TS_MANIP): DRIVE runs coarse physics —
            # base/lift/neck dynamics are fine at 8 ms, and this scene's raw
            # mj_step (~3.9 ms wall) left dt=4 ms with a 1.04x physics CEILING
            # before any control cost (measured RT 0.4-0.5x while driving:
            # "drive feels dead"). Manipulation methods keep the validated
            # 4 ms for grasp/contact accuracy. Mid-run timestep changes are
            # cheap and safe (the bridge does exactly this).
            ts = TS_DRIVE if self.spec["method"] == "navigation" else float(CFG.TIMESTEP)
            if abs(self.model.opt.timestep - ts) > 1e-12:
                self.model.opt.timestep = ts
                self.dt = ts

    # legacy single-axis helpers (tests / simple wiring)
    def set_method(self, method):   self.set_control_spec(method=method)
    def set_region(self, region):   self.set_control_spec(region=region)
    def set_control_type(self, t):  self.set_control_spec(control_type=t)
    def set_stiffness(self, k):     self.set_control_spec(stiffness=k)

    def set_joint_targets(self, joints):
        """joint_command message: {wire joint id: value}. Accumulated by the
        facade; consumed by the WBC impedance (Joint mode) / direct override."""
        with self.lock:
            self.teleop.set_joint_targets(joints or {})

    def apply_external_force(self, forces):
        """external_force message: {body: [fx,fy,fz]} world-frame operator push
        the compliance / whole-body engine admits."""
        with self.lock:
            self.teleop.apply_external_force(forces or {})

    def recalibrate(self):
        with self.lock:
            self.teleop.recalibrate()

    def set_self_collision(self, on, method=None):
        with self.lock:
            self.self_collision = bool(on)
            if method in ("stop", "cbf"):
                self.sc_method = method
            self.teleop.set_self_collision(self.self_collision, self.sc_method)
            if self.ik is not None:
                self.ik.set_self_collision(self.self_collision, method=self.sc_method)

    def set_sc_method(self, method):
        if method not in ("stop", "cbf"):
            raise ValueError(f"sc method must be stop|cbf, got {method!r}")
        self.set_self_collision(self.self_collision, method)

    def set_wrist_human(self, on):
        with self.lock:
            self.wrist_human = bool(on)
            if self.ik is not None:
                self._apply_wrist_posture()

    def set_hand_side(self, side):
        if side not in ("left", "right", "both"):
            raise ValueError(f"unknown side {side!r}")
        with self.lock:
            self.hand_side = side

    def reset_robot(self):
        """Respawn at the settled home pose, zero velocities + odometry, re-home
        the controller, and drop back to LIMP."""
        with self.lock:
            self._restore_servos()
            self.data.qpos[:] = self._qpos0
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            if self.model.nkey > 0:                # keep the lift at its default
                self.data.ctrl[:] = self.model.key_ctrl[0]
            mujoco.mj_forward(self.model, self.data)
            self.teleop.reset()                # engine: clears refs + re-anchors
            self.ik = None
            self.push = None
            self.base_follow.reset()
            self.odom = BaseMotionModel()
            x, y, yaw = self._true_base()
            self.odom.reset(x, y, yaw)
            self.plots.clear()
            self._t = 0.0
            self._plot_next = 0.0
            self.overlay = {}
            self._go_limp()

    # ── wrist task-space (WholeBodyIK -> impedance targets) ─────────────────
    def _ensure_wrist_ik(self):
        if self.ik is None:
            # The IK is bound to a SCRATCH MjData, NOT the physics data. In Stiff it
            # plans on its OWN integrated target pose (decoupled from the physical
            # arm, which lags the target under impedance). Solving the IK against the
            # measured, lagging arm cascades the IK integrator with the impedance
            # loop and the palm OSCILLATES; planning on the target state converges
            # cleanly and the impedance just tracks it. See _drive_wrist_ik.
            self._ik_data = mujoco.MjData(self.model)
            self._ik_data.qpos[:] = self.data.qpos
            mujoco.mj_kinematics(self.model, self._ik_data)
            self.ik = WholeBodyIK(self.model, self._ik_data)
            self.ik.set_self_collision(self.self_collision, method=self.sc_method)
            self._apply_wrist_posture()
            self.ik.sync_to_current(freeze_posture=True)

    def _sync_ik_scratch(self, from_target):
        """Refresh the IK's scratch MjData: base pose + non-managed joints always
        from the REAL physics state; the arm/torso/lift joints from the IK's own
        `target_q` (from_target=True → plan on the commanded pose) or from the real
        measured pose (from_target=False → Soft / re-anchor)."""
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        if from_target:
            for name, val in self.ik.target_q.items():
                # Sync-Lift cascade: the "Lift" target must move BOTH stage joints
                # (writing only Lift Lower plans on a half-lifted pose).
                joints = (("Lift Lower", "Lift Upper") if name == "Lift" else (name,))
                for joint in joints:
                    try:
                        d.qpos[self.model.joint(joint).qposadr[0]] = val
                    except Exception:
                        pass
        mujoco.mj_kinematics(self.model, d)

    def _apply_wrist_posture(self):
        """HUMAN-LIKE redundancy resolution from the VP whole-body teleop's
        constraints (config.ARM_REG_W: roll resolves at the forearm, hard return
        on shoulder-roll + wrist-yaw), mapped onto the DLS null-space weights.
        wrist_human=False restores the generic uniform posture for A/B."""
        if self.ik is None:
            return
        if self.wrist_human:
            w_arm = (np.asarray(C.ARM_REG_W, dtype=float) / 0.08)
            w_wrist = float(C.ARM_REG_WRIST) / 0.08
            # Lift/Torso null-space return 4x the arms: the LAZY sections drift
            # back to rest through the redundancy, so height/lean the ARMS can
            # supply is not slowly absorbed by the body (measured: an 11 cm lift
            # rise for a 5 cm up-drag with the uniform 1.0 pull).
            self.ik.posture_w = np.concatenate(
                [[4.0, 4.0], w_arm, [w_wrist], w_arm, [w_wrist]])
            self.ik.kp_rot = 0.8
        else:
            self.ik.posture_w = np.ones(len(self.ik.dofs))
            self.ik.kp_rot = C.IK_KP_ROT

    def set_wrist_target(self, side, pos):
        """`pos` is the VISUAL palm-centre point the user dragged; the IK target
        is the palm body origin, so subtract the current vis offset."""
        if side not in ("l", "r"):
            raise ValueError(f"side must be 'l' or 'r', got {side!r}")
        with self.lock:
            if self.spec["method"] != "wrist":
                raise ValueError("wrist targets only apply in the wrist method")
            self._ensure_wrist_ik()
            p = np.asarray(pos, dtype=float)[:3]
            if not np.isfinite(p).all():
                raise ValueError("non-finite wrist target")
            setattr(self.ik, f"target_{side}_pos", p - self._palm_vis_ofs(side))

    def wrist_sync(self):
        with self.lock:
            if self.ik is not None:
                self._sync_ik_scratch(from_target=False)
                self.ik.sync_to_current(freeze_posture=True)

    def _drive_wrist_ik(self, dt, solve=True):
        """Solve task-space IK from the dragged palm targets and push the result
        into the WBC engine as joint targets (Stiff springs to them; Soft floats,
        with the targets riding the measured pose so re-stiffening is bumpless).
        The QP is DECIMATED (runs on the heavy tick); the resulting joint targets
        persist in the facade and the impedance tracks them every physics step."""
        self._ensure_wrist_ik()
        arms_only = (self.spec["region"] == "arm")
        if self.spec["type"] == "compliance":      # SOFT: float, don't command
            if solve:
                self._sync_ik_scratch(from_target=False)   # anchor to the measured pose
                self.ik.sync_to_current()
        elif solve:                                # STIFF: advance the target, decoupled
            self._sync_ik_scratch(from_target=True)        # plan on the commanded pose
            t0 = time.perf_counter()
            dq = self.ik.solve(arms_only=arms_only)
            if self.spec["region"] == "whole_body" and self.base_follow.active:
                # The BASE is still driving to the target — do NOT integrate the
                # slow body DOFs toward it meanwhile ("you don't lean into a thing
                # you're walking to"). Without this the torso accumulated to FULL
                # lean at its rate cap during the chase, and the robot pitched
                # over right as the eager base arrived (margin +0.014 → fell).
                # Ease lift + torso back toward rest instead; they may assist
                # again once the base is inside the reach band.
                dq[0] = 0.5 * (self.ik.q_rest[0] - self.ik.target_q["Lift"])
                dq[1] = 0.5 * (self.ik.q_rest[1] - self.ik.target_q["Torso"])
            self.ik.apply(dq, CONTROL_DECIM * dt, arms_only=arms_only)
            self.timer.add("ik_solve_ms", time.perf_counter() - t0)
            jt = {_IK_WIRE[a]: float(v) for a, v in self.ik.target_q.items()
                  if a in _IK_WIRE}
            if not arms_only:
                # The 18-DOF solve recruits the LIFT too. Its (joint-unit) target
                # rides the wire as the two stage keys — _apply_joint_targets sums
                # them back into the tendon ctrl. _IK_WIRE has no "Lift" entry, so
                # without this the solved lift motion was silently dropped ("the
                # lift is stuck" in upper_body / whole_body wrist mode).
                jt["lift_lower"] = jt["lift_upper"] = float(self.ik.target_q["Lift"])
            self.teleop.set_joint_targets(jt)
        if solve:
            self.overlay["targets"] = self._ik_target_overlay(self.ik)

    # ── diagnostics: external wrench + trajectory ───────────────────────────
    def apply_push(self, body, force, seconds=1.0):
        with self.lock:
            bid = self.model.body(body).id      # raises KeyError on bad name
            F = np.asarray(force, dtype=float)[:3]
            com = self.data.xipos[bid].copy()
            pt, hit = surface_point(self.model, self.data,
                                    body_subtree(self.model, bid), com, F)
            tau = np.cross(pt - com, F) if hit else np.zeros(3)
            self.push = {"body": body, "bid": bid, "F": F,
                         "point": pt if hit else com, "tau": tau,
                         "until": self._t + float(seconds)}

    def set_trajectory(self, kind="hold", joint=None, amp=None, period=None):
        with self.lock:
            if kind not in ("hold", "step", "sine"):
                raise ValueError(f"unknown trajectory {kind!r}")
            self.traj["kind"] = kind
            if joint:
                self.traj["joint"] = joint
            if amp is not None:
                self.traj["amp"] = float(amp)
            if period is not None:
                self.traj["period"] = max(0.5, float(period))
            self.traj["t0"] = self._t
            self.traj["q0"] = {}

    def _traj_target(self):
        """Diagnostic joint driver (Joint mode): feed a step/sine to one joint's
        WBC impedance target so its tracking can be measured."""
        if self.traj["kind"] == "hold" or self.spec["method"] != "joint":
            return
        wid = self.traj["joint"]
        mj = None
        for m, i in _MJ_TO_IOS.items():
            if i == wid:
                mj = m
                break
        if mj is None:
            return
        if wid not in self.traj["q0"]:
            self.traj["q0"][wid] = float(self.plant.q(mj))
            self.traj["t0"] = self._t
        q0 = self.traj["q0"][wid]
        tau = self._t - self.traj["t0"]
        if self.traj["kind"] == "step":
            tgt = q0 + (self.traj["amp"] if tau > 0.5 else 0.0)
        else:
            tgt = q0 + self.traj["amp"] * math.sin(2 * math.pi * tau / self.traj["period"])
        jo = self.model.joint(mj)
        if jo.limited[0]:
            lo, hi = jo.range
            tgt = min(max(tgt, lo + 0.02), hi - 0.02)
        self.teleop.set_joint_targets({wid: tgt})

    def _sync_estimator(self):
        """Mount the momentum-observer contact estimator whenever a WBC method
        (joint / wrist) is live and torque is on — the arms are then torque-driven,
        so a diagnostic push is localised + its wrench estimated against the ground
        truth in Stiff (the arm resists) and Soft (the arm yields) alike."""
        want = (self.torque_on and self.spec["method"] in WBC_METHODS)
        if want and self.estimator is None:
            self.estimator = ContactEstimator(self.model, self.data, gain=35.0)
            self._tau_sum[:] = 0.0
            self._tau_n = 0
        elif not want:
            self.estimator = None

    # ── external wrench application (push + viewer Ctrl+drag) ────────────────
    def _apply_external_wrenches(self):
        """Zero + re-apply the diagnostic push and the native-viewer drag to
        xfrc BEFORE the control law runs (so whole-body compliance glides under
        it and the estimator's ground truth is honest)."""
        self.data.xfrc_applied[:] = 0.0
        true_w = None
        if self.push is not None:
            if self._t <= self.push["until"]:
                bid, F, tau = self.push["bid"], self.push["F"], self.push["tau"]
                self.data.xfrc_applied[bid, :3] = F
                self.data.xfrc_applied[bid, 3:] = tau
                nf = float(F @ F)
                if nf > 1e-9:
                    p_line = self.data.xipos[bid] + np.cross(F, tau) / nf
                    pt, hit = surface_point(self.model, self.data,
                                            body_subtree(self.model, bid), p_line, F)
                    if hit:
                        self.push["point"] = pt
                true_w = {"body": self.push["body"],
                          "point": [round(float(v), 4) for v in self.push["point"]],
                          "force": [round(float(v), 3) for v in self.push["F"]]}
            else:
                self.push = None
        if self._viewer_push is not None:
            bid, w = self._viewer_push
            self.data.xfrc_applied[bid, :] += w
            f, m = w[:3], w[3:]
            nf = float(f @ f)
            if nf > 1e-6:
                com = self.data.xipos[bid]
                p_line = com + np.cross(f, m) / nf
                pt, hit = surface_point(self.model, self.data,
                                        body_subtree(self.model, bid), p_line, f)
                true_w = {"body": self.model.body(bid).name,
                          "point": [round(float(v), 4) for v in (pt if hit else com)],
                          "force": [round(float(v), 3) for v in f]}
        return true_w

    # ── native-viewer perturbation (Ctrl+drag in the mjpython mirror) ───────
    def _read_viewer_push(self):
        if self._mm is None:
            return
        po = self._pert_off
        if self._mm[po] > 0.5:
            bid = int(self._mm[po + 1])
            if 0 <= bid < self.model.nbody:
                self._viewer_push = (bid, np.array(self._mm[po + 2:po + 8]))
            else:
                self._viewer_push = None
        else:
            self._viewer_push = None

    # ── whole-body base drive (sticks / task-space follower → tip-safe twist) ─
    def _drive_base(self, dt):
        """Drive the mobile base in the whole_body region (any manipulation mode),
        plus a right-stick (ry) lift jog. The twist target is arbitrated inside
        the library's BaseFollow: a deliberate nav-stick deflection wins; otherwise
        (Stiff wrist mode) the LazyBase task-space follower creeps the base to keep
        the DRAGGED palm targets inside the arms' comfort box — drag a target out
        of reach and the chassis drives after it (the ported sim-teleop whole-body
        cascade; previously nothing mapped wrist targets to the base, so it never
        moved). Every twist passes through safe_base_step inside BaseFollow.
        Command odometry integrates the same safe twist."""
        joy = app_joystick(self.nav)
        dz = lambda v: 0.0 if abs(v) < C.NAV_DEADZONE else v
        if joy is None:
            stick, ry = None, 0.0
        else:
            lx, ly, rx, ry = joy
            stick = (dz(ly) * C.MAX_LIN_SPEED, dz(lx) * C.MAX_LIN_SPEED,
                     dz(rx) * C.MAX_ANG_SPEED)
        p_l = p_r = None
        if (self.spec["method"] == "wrist" and self.spec["type"] == "impedance"
                and self.ik is not None):
            p_l, p_r = self.ik.target_l_pos, self.ik.target_r_pos
        self._base_v = self.base_follow.step(self.data, dt, self._lift_z(),
                                             stick=stick, p_l=p_l, p_r=p_r)
        swerve_ik(*self._base_v, self.data)
        self._last_base = self._base_v
        self.odom.step(*self._base_v, dt)
        if abs(ry) > 0.04:                               # right-stick lift jog
            cur = float(self.data.actuator("Lift").ctrl[0])
            newv = min(C.LIFT_RANGE[1], max(0.0, cur + ry * 0.5 * dt))
            self.data.actuator("Lift").ctrl[0] = newv
            self.teleop.joint_targets["lift_lower"] = newv / 2.0
            self.teleop.joint_targets["lift_upper"] = newv / 2.0

    # ── central control dispatch ────────────────────────────────────────────
    def _control(self, heavy=True):
        if not self.torque_on:
            # LIMP is a gravity-comp float — let a native-viewer Ctrl+drag / push
            # still shove the back-drivable arm around (manipulate the limp robot).
            self._apply_external_wrenches()
            self._limp_hold()
            return

        method = self.spec["method"]
        ctype = self.spec["type"]
        region = self.spec["region"]
        dt = self.dt
        true_w = self._apply_external_wrenches()
        pose = None

        if method in ("navigation", "retarget"):
            # KINEMATIC path via the facade: position servos hold the arms while the
            # base joystick (drive) or the operator retarget stream drives motion.
            self.teleop.set_control_spec(method=method, control_type=ctype,
                                         region=region, stiffness=self.spec["stiffness"])
            joy = app_joystick(self.nav)
            if method == "retarget":
                pose = self._pose
                if pose is not None and not self.teleop.calibrated:
                    self.teleop.calibrate(pose)
            else:                                    # a headless frame carries the stick
                # VR-CADENCE PACING (2026-07-20 — the "drive feels dead" fix):
                # the engine memoizes its per-frame work on pose.timestamp, so a
                # FRESH stamp every 250 Hz physics tick ran _payload + retarget +
                # the gravity-comp mj_forward at the full physics rate —
                # update_ms ~3 ms of the 4 ms budget, RT 0.4x, and the tip-safe
                # accel ramp then unfolded at a third of real time. Quantizing
                # the stamp to 60 Hz restores the real bridge's cadence: frame
                # work at VR rate, the cheap base ramp/swerve every substep.
                pose = OperatorPose(
                    timestamp=math.floor(self._t * NAV_FRAME_HZ) / NAV_FRAME_HZ)
            t0 = time.perf_counter()
            cmd = self.teleop.update(pose, joy, dt)
            self.timer.add("update_ms", time.perf_counter() - t0)
            self._last_base = tuple(cmd.base) if cmd.base is not None else (0.0, 0.0, 0.0)
            if method == "navigation":
                self.odom.step(*self._last_base, dt)

        elif ctype == "compliance":
            # SOFT (joint / wrist): the WBC compliance engine — arms float, back-
            # drivable; whole-body admits the push and glides. Passive (the operator
            # moves it — in the viewer with Ctrl+drag), like the apps' Soft.
            self.teleop.set_control_spec(method=method, control_type="compliance",
                                         region=region, stiffness=self.spec["stiffness"])
            if method == "wrist":
                self._drive_wrist_ik(dt, solve=heavy)   # targets ride the pose (bumpless)
            t0 = time.perf_counter()
            cmd = self.teleop.update(None, None, dt)
            self.timer.add("update_ms", time.perf_counter() - t0)
            self._last_base = tuple(cmd.base) if cmd.base is not None else (0.0, 0.0, 0.0)

        else:
            # STIFF (joint / wrist): POSITION SERVOS hold the targets — rigid and
            # rock-steady, exactly like the apps ("Stiff = a joint target the servo
            # holds", NOT a host-side torque-impedance loop, which oscillates chasing
            # a live-dragged target — the bug iOS/VP fixed the same way). Wrist runs a
            # decoupled scratch IK → joint targets; the base is drivable in whole_body.
            if self.teleop.wbc.active:
                self.teleop.wbc.exit()               # servos active
                self.teleop.controller.hold_current()   # freeze at current pose (bumpless Soft→Stiff)
                if self.ik is not None:
                    self._sync_ik_scratch(from_target=False)
                    self.ik.sync_to_current(freeze_posture=True)   # wrist targets ride the pose
            if method == "wrist":
                self._drive_wrist_ik(dt, solve=heavy)   # scratch IK -> joint_targets
            elif heavy:
                self._traj_target()                  # diagnostic joint driver
            t0 = time.perf_counter()
            self.teleop._apply_joint_targets()       # joint_targets -> position servos
            # gravity feedforward so the servos REACH the target instead of sagging
            # below it under the arm's own weight (the ~5 cm tracking offset).
            b = self.teleop.wbc.dyn.bias()
            self.plant.begin()
            for j in region_joints(region):
                self.plant.apply_effort(j, b[_dof(self.model, j)])
            if region in ("upper_body", "whole_body"):
                # The tendon LIFT is excluded from region_joints — without its own
                # feedforward the kp=1e4 servo sags ~28 mm under the ~290 N upper
                # body, and the IK (which plans on the TARGET lift height) leaves
                # the palms exactly that far off: the measured 31 mm upper-region
                # wrist error vs 8 mm in the arm region.
                for j in ("Lift Lower", "Lift Upper"):
                    self.plant.apply_effort(j, b[_dof(self.model, j)])
            self.plant.commit()
            self.timer.add("update_ms", time.perf_counter() - t0)
            if region == "whole_body":
                self._drive_base(dt)
            else:
                swerve_ik(0.0, 0.0, 0.0, self.data)
                self._last_base = (0.0, 0.0, 0.0)

        # wrench estimation (Soft / torque-driven arms): observer residual -> wrench.
        # DECIMATED (the observer is fed the interval-MEAN torque; a point sample
        # of a fast-slewing joint reads as a phantom contact).
        if self.estimator is not None and heavy:
            if self._tau_n > 0:
                tau_cmd = self._tau_sum / self._tau_n
                self._tau_sum[:] = 0.0
                self._tau_n = 0
            else:
                tau_cmd = self.data.qfrc_applied + self.data.qfrc_actuator
            self.estimator.update(tau_cmd, CONTROL_DECIM * dt)
            self._est_w = None
            if self.estimator.contacts:
                c = max(self.estimator.contacts,
                        key=lambda c: float(np.linalg.norm(c["force"])))
                self._est_w = {"body": c["body"], "joint": c["joint"],
                               "point": [round(float(v), 4) for v in c["point"]],
                               "force": [round(float(v), 3) for v in c["force"]]}
        est_w = self._est_w if self.estimator is not None else None
        if true_w is not None or est_w is not None:
            self.overlay["wrench"] = {"true": true_w, "est": est_w}
        else:
            self.overlay.pop("wrench", None)

        # retarget overlays (operator skeleton + IK targets), decimated
        if method == "retarget" and pose is not None and heavy:
            self.overlay["skeleton"] = self._skeleton_overlay(pose)
            self.overlay["targets"] = self._target_overlay()
            self._ee_hist.append((self._t,
                                  self.data.body(self._palm_l).xpos.copy(),
                                  self.data.body(self._palm_r).xpos.copy()))

    # ── retarget pose ingest (VP node / mock) ───────────────────────────────
    def ingest_pose(self, pose: OperatorPose, source: str):
        """Called by the VP node / mock source. LOCK-FREE (single attribute
        assignment, atomic under the GIL) so a saturated loop never starves it."""
        self._mailbox = (pose, source, time.time())

    def _consume_mailbox(self):
        box = self._mailbox
        if box is None or box[2] == self._pose_stamp:
            return
        pose, device, now = box
        self._pose = pose
        self._pose_seq += 1
        if self._last_frame_wall is not None:
            self._frame_gaps.append(now - self._last_frame_wall)
        self._last_frame_wall = now
        self._pose_stamp = now
        self.vp_status["device"] = device
        self.vp_status["connected"] = True
        self.vp_status["frames"] = self._pose_seq

    def apply_profile(self, profile):
        with self.lock:
            if profile is not None:
                self.teleop.apply_profile(profile)
            self.vp_status["calibration"] = getattr(profile, "user", None) or "applied"

    def _vp_stats(self):
        gaps = list(self._frame_gaps)
        if gaps:
            mean = sum(gaps) / len(gaps)
            self.vp_status["rate_hz"] = round(1.0 / mean, 1) if mean > 0 else 0.0
            var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
            self.vp_status["jitter_ms"] = round(math.sqrt(var) * 1e3, 2)
        stale = (time.time() - self._pose_stamp) if self._pose_stamp else None
        self.vp_status["age_ms"] = round(stale * 1e3, 1) if stale is not None else None
        if stale is not None and stale > 1.0:
            self.vp_status["connected"] = False

    # ── geometry helpers ────────────────────────────────────────────────────
    def _lift_z(self):
        q = self.data.joint(C.LIFT_JOINT).qpos[0]
        return min(C.LIFT_RANGE[1], max(0.0, q / 0.303 * C.LIFT_RANGE[1]))

    def _true_base(self):
        b = self.data.body(self._root)
        R = b.xmat.reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        return float(b.xpos[0]), float(b.xpos[1]), yaw

    def _base_delta(self):
        """Height drop `dz` + body-frame rotation delta `qd` [w,x,y,z] of the base
        from the settled upright home — the viewer renders the base as this rigid
        delta on top of {x,y,yaw}, so a fall / tip-over is visible (was: always
        upright). Identity at home."""
        rb = self.data.body(self._root)
        q0 = getattr(self, "_root_home_quat", None)
        if q0 is None:
            return 0.0, [1.0, 0.0, 0.0, 0.0]
        q0inv = q0 * np.array([1.0, -1.0, -1.0, -1.0])   # conjugate (unit quat)
        qd = np.empty(4)
        mujoco.mju_mulQuat(qd, q0inv, np.array(rb.xquat, dtype=float))  # q0^-1 * q
        return float(rb.xpos[2]) - self._root_home_z, [round(float(v), 5) for v in qd]

    def _palm_vis_ofs(self, side):
        bid = self._palm_l if side == "l" else self._palm_r
        R = self.data.body(bid).xmat.reshape(3, 3)
        return R @ (self._palm_fwd[side] * self.PALM_VIS_OFFSET)

    def _ik_target_overlay(self, ik):
        out = {}
        for key, tgt, bid in (("l", getattr(ik, "target_l_pos", None), self._palm_l),
                              ("r", getattr(ik, "target_r_pos", None), self._palm_r)):
            if tgt is None:
                continue
            ofs = self._palm_vis_ofs(key)
            actual = self.data.body(bid).xpos + ofs
            out[key] = {"target": [round(float(v), 4) for v in (tgt + ofs)],
                        "actual": [round(float(v), 4) for v in actual]}
        return out

    def _target_overlay(self):
        ik = getattr(self.teleop.controller, "ik", None)
        return self._ik_target_overlay(ik) if ik is not None else {}

    def _skeleton_overlay(self, pose):
        try:
            head_r = self.data.body(self.model.body("head").id).xpos
        except Exception:
            head_r = np.array([0.0, 0.0, 1.4])
        if pose.head is None:
            return None
        h_op = pose.head[:3, 3]
        M = C.AVP_TO_ROBOT

        def mapped(p):
            return [round(float(v), 4) for v in (head_r + M @ (np.asarray(p) - h_op))]

        pts = {"head": mapped(h_op)}
        for side, hand, joints, elbow in (
                ("l", pose.left_hand, pose.left_joints, pose.left_elbow),
                ("r", pose.right_hand, pose.right_joints, pose.right_elbow)):
            if hand is not None and len(hand):
                pts[f"wrist_{side}"] = mapped(hand[0][:3, 3])
            if elbow is not None:
                pts[f"elbow_{side}"] = mapped(elbow[:3, 3])
            if joints:
                for tip in _TIP_JOINTS:
                    m = joints.get(f"{tip}")
                    if m is not None:
                        pts[f"{side}_{tip}"] = mapped(m[:3, 3])
        return pts

    def _robot_pinch(self, side):
        try:
            a = self.data.body(f"{side}_thumb_dp").xpos
            b = self.data.body(f"{side}_index_ip").xpos
            return float(np.linalg.norm(a - b))
        except Exception:
            return None

    def _ee_jerk(self):
        h = list(self._ee_hist)
        if len(h) < 4:
            return None
        ts = [x[0] for x in h]
        ps = [0.5 * (x[1] + x[2]) for x in h]
        dt = (ts[-1] - ts[0]) / (len(ts) - 1)
        if dt <= 0:
            return None
        v = np.diff(ps, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        jk = np.diff(a, axis=0) / dt
        return float(np.sqrt((jk ** 2).sum(axis=1).mean()))

    # ── plots (re-homed per mode) ────────────────────────────────────────────
    def _plot(self, series, value):
        buf = self.plots.get(series)
        if buf is None:
            buf = self.plots[series] = deque(
                maxlen=int(CFG.PLOT_WINDOW_S * CFG.PLOT_HZ))
        buf.append((round(self._t, 3),
                    None if value is None else round(float(value), 5)))

    def _sample_plots(self):
        method = self.spec["method"]
        if method == "navigation":
            self._sample_drive()
        elif method in ("joint", "wrist"):
            self._sample_manip()
        elif method == "retarget":
            self._sample_retarget()

    def _sample_drive(self):
        z = self._lift_z()
        dz = lambda v: 0.0 if abs(v) < C.NAV_DEADZONE else v
        j = app_joystick(self.nav) or (0.0, 0.0, 0.0, 0.0)
        lx, ly, rx, _ = j
        self._plot("cmd_vx", dz(ly) * C.MAX_LIN_SPEED)
        self._plot("cmd_vy", dz(lx) * C.MAX_LIN_SPEED)
        self._plot("cmd_w", dz(rx) * C.MAX_ANG_SPEED)
        vx, vy, w = self._last_base
        self._plot("safe_vx", vx); self._plot("safe_vy", vy); self._plot("safe_w", w)
        self._plot("tip_usage", tip_usage(vx, vy, w, 0, 0, 0, z))
        v_fwd, v_back, v_lat = max_speed_dir(z)
        self._plot("cap_fwd", v_fwd); self._plot("cap_lat", v_lat); self._plot("lift_z", z)
        x, y, yaw = self._true_base()
        self._plot("odom_err", math.hypot(x - self.odom.x, y - self.odom.y))
        self.overlay["odom"] = {"true": [x, y, yaw],
                                "odom": [self.odom.x, self.odom.y, self.odom.theta]}

    def _sample_manip(self):
        # test joint: q vs the WBC impedance target
        wid = self.traj["joint"]
        mj = next((m for m, i in _MJ_TO_IOS.items() if i == wid), None)
        if mj is not None:
            q = self.plant.q(mj)
            qdes = self.teleop.wbc._q_des.get(wid, q) if self.teleop.wbc.active else q
            self._plot("q", q); self._plot("q_des", qdes); self._plot("q_err", qdes - q)
            # total applied joint torque vs the pure-gravity feed-forward the WBC
            # injects: with gravity comp carrying the load, (tau - tau_ff) is just
            # the impedance error term. The engine writes straight to MuJoCo and
            # caches no per-joint total, so read it back (qfrc_applied is the WBC's
            # Soft torque, qfrc_actuator the Stiff position servo).
            dof = _dof(self.model, mj)
            self._plot("tau", float(self.data.qfrc_applied[dof]
                                    + self.data.qfrc_actuator[dof]))
            self._plot("tau_ff", float(self.teleop.wbc.dyn.gravity()[dof]))
        # region-wide tracking error (impedance)
        if self.teleop.wbc.active:
            errs = []
            for j in self.teleop.wbc.managed:
                ios = _MJ_TO_IOS.get(j)
                if ios and ios in self.teleop.wbc._q_des:
                    errs.append(abs(self.teleop.wbc._q_des[ios] - self.plant.q(j)))
            if errs:
                self._plot("err_max", max(errs))
                self._plot("err_rms", math.sqrt(sum(e * e for e in errs) / len(errs)))
        # self-collision "soft wall" RESPONSE: peak barrier torque the guard adds
        # to push capsules apart at the margin (0 when the layer is off / clear).
        # Not cached on the engine — recompute from the live qvel, gated EXACTLY
        # like engine.step (self_collision flag AND guard built) so a disabled
        # layer reads a true 0 even though the guard object persists.
        wbc = self.teleop.wbc
        self._plot("tau_sc",
                   float(np.max(np.abs(wbc.guard.barrier_force(self.data.qvel))))
                   if (wbc.self_collision and wbc.guard is not None) else 0.0)
        # wrist IK tracking — the REAL palm vs the dragged target (the IK's own
        # e_pos_l is the scratch planner's error, ~0; measure the physical arm).
        if self.spec["method"] == "wrist" and self.ik is not None:
            tg = self.overlay.get("targets") or {}
            for side in ("l", "r"):
                v = tg.get(side)
                if v is not None:
                    err = np.linalg.norm(np.asarray(v["target"]) - np.asarray(v["actual"]))
                    self._plot(f"ee_err_{side}_mm", err * 1e3)
            # orientation error too — position-only is incomplete for an SE(3) IK.
            # e_rot_{l,r} is the IK's body-frame rotation-vector residual (rad).
            self._plot("ee_rot_l_deg", math.degrees(float(np.linalg.norm(self.ik.e_rot_l))))
            self._plot("ee_rot_r_deg", math.degrees(float(np.linalg.norm(self.ik.e_rot_r))))
            self._plot("ik_solve_ms", self.timer.ms.get("ik_solve_ms"))
        # wrench estimation (Soft)
        w = self.overlay.get("wrench") or {}
        tw, ew = w.get("true"), w.get("est")
        self._plot("F_true", np.linalg.norm(tw["force"]) if tw else 0.0)
        self._plot("F_est", np.linalg.norm(ew["force"]) if ew else 0.0)
        if tw and ew:
            a = np.asarray(tw["force"]); b = np.asarray(ew["force"])
            den = np.linalg.norm(a) * np.linalg.norm(b)
            self._plot("F_cos", float(a @ b / den) if den > 1e-6 else None)
            self._plot("pt_err_cm", 100 * np.linalg.norm(
                np.asarray(tw["point"]) - np.asarray(ew["point"])))
        else:
            self._plot("F_cos", None); self._plot("pt_err_cm", None)

    def _sample_retarget(self):
        self._vp_stats()
        # EE tracking: the robot palm vs the operator's wrist mapped into the
        # robot frame (skeleton overlay). The facade's kinematic generator does
        # not expose its internal palm-target error (status() returns 0), so this
        # measured palm-to-mapped-wrist distance is the honest tracking quantity
        # (it carries a fixed hand-length offset — dynamics, not absolute error).
        sk = self.overlay.get("skeleton") or {}
        ee = []
        for side, bid in (("l", self._palm_l), ("r", self._palm_r)):
            w = sk.get(f"wrist_{side}")
            if w is not None:
                d = float(np.linalg.norm(self.data.body(bid).xpos - np.asarray(w)))
                self._plot(f"ee_err_{side}_mm", d * 1e3)
                ee.append(d)
        # MPJPE proxy: mean over the tracked keypoints (both wrists)
        self._plot("mpjpe_mm", 1e3 * (sum(ee) / len(ee)) if ee else None)
        vx, vy, w = self._last_base
        self._plot("base_v", math.hypot(vx, vy)); self._plot("base_w", w)
        self._plot("tip_usage", tip_usage(vx, vy, w, 0, 0, 0, self._lift_z()))
        self._plot("jerk", self._ee_jerk())
        self._plot("stream_hz", self.vp_status["rate_hz"])
        self._plot("retarget_ms", self.timer.ms.get("update_ms"))
        self._plot("age_ms", self.vp_status["age_ms"])

    # ── snapshot for /ws/state ──────────────────────────────────────────────
    def snapshot(self):
        with self.lock:
            jp = {w: float(self.data.qpos[adr]) for w, adr in self.qadr.items()}
            x, y, yaw = self._true_base()
            dz, qd = self._base_delta()
            ms, peak = self.timer.snapshot()
            self._vp_stats()
            snap = {
                "t": round(self._t, 3),
                "torque": self.torque_on,
                "spec": dict(self.spec),
                "region": self.spec["region"],
                "method": self.spec["method"],
                "status": self.status_msg,
                "selfCollision": self.self_collision,
                "scMethod": self.sc_method,
                "wristHuman": self.wrist_human,
                "handSide": self.hand_side,
                "jointPositions": jp,
                "base": {"x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4)},
                # full base pose delta from the upright home (viewer renders falls /
                # tip-over) — kept OUT of `base` so `base` stays all-scalar for the
                # finiteness checks that iterate base.values().
                "basePose": {"z": round(float(self.data.body(self._root).xpos[2]), 4),
                             "dz": round(dz, 4), "qd": qd},
                "bodies": {
                    "head": ([round(float(v), 4) for v in self.data.body(self._head).xpos]
                             if self._head is not None else None),
                    "palm_l": [round(float(v), 4) for v in self.data.body(self._palm_l).xpos],
                    "palm_r": [round(float(v), 4) for v in self.data.body(self._palm_r).xpos],
                },
                "odom": {"x": round(self.odom.x, 4), "y": round(self.odom.y, 4),
                         "yaw": round(self.odom.theta, 4)},
                "lift": round(self._lift_z(), 4),
                "traj": {k: self.traj[k] for k in ("kind", "joint", "amp", "period")},
                "vp": dict(self.vp_status),
                "timing": {"avg": ms, "peak": peak},
                "overlay": dict(self.overlay),
                "plots": {k: list(v)[-1] for k, v in self.plots.items() if len(v)},
            }
            if self.spec["method"] == "retarget":
                snap["calibrated"] = self.teleop.calibrated
            return snap

    def plot_history(self):
        with self.lock:
            return {k: list(v) for k, v in self.plots.items()}

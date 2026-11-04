"""
One retargeter, kinematically: the engine's command IS the robot's pose.

`KinematicRetargeter` wraps a TeleopEngine configured as one variant
(variants.py) and steps it through `_retarget(payload, None, now)` — the same
per-frame path `update()` takes, minus the engagement supervisor's ease and the
plant. After each solve every position-servo target is written straight into
qpos (an IDEAL servo) and the kinematics are recomputed, so:

  * the robot you see is exactly what the map asked for — no lag, no sag;
  * the engine's next solve reads that pose back as "measured", which is what a
    perfectly tracking robot would feed it (trust region, collision projector
    and the torso-aware shoulder all stay self-consistent);
  * the BASE is integrated kinematically too: the engine runs in GLOBAL scope,
    its realised base twist (head-overflow follow + body heading + workspace
    assist, ramped through the tip-safe motion model exactly as on the robot)
    is passed through the per-module swerve model (wheel-speed / torque caps,
    steer slew, drive choked while a wheel turns — motion_model/swerve_modules.py)
    and the twist the three modules deliver is integrated into the free joint, so
    the robot walks and turns in the world and is NOT treated as holonomic.
    No wheel contact, no slip beyond what the model reports as scrub.

`run_worker` is the process entry point: it builds one KinematicRetargeter and
answers frames from a pipe. One process per variant — see variants.py for why
that is required, not just faster.

Per-frame metrics (`metrics()`): see METRICS below and the lab doc
(docs/retarget_lab.tex, §Metrics) for each definition and its caveats.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np

from . import config as CFG

# metric id -> (label, unit, higher_is_better, per_side)
METRICS = {
    "cos_reach":  ("Reach-direction cosine", "", True, True),
    "cos_upper":  ("Upper-arm cosine", "", True, True),
    "cos_fore":   ("Forearm cosine", "", True, True),
    "place_pct":  ("Placement error", "% reach", False, True),
    "ext_err":    ("Extension error |Δρ|", "", False, True),
    "elbow_err":  ("Elbow-angle error", "°", False, True),
    "orient_deg": ("Palm orientation error", "°", False, True),
    "track_mm":   ("Own-target error (control)", "mm", False, True),
    "aperture_h": ("Pinch aperture · human", "mm", None, True),
    "aperture_r": ("Pinch aperture · robot", "mm", None, True),
    "sat_pct":    ("Joint-limit saturation", "%", False, False),
    "qvel":       ("Peak arm joint speed", "rad/s", False, False),
    "selfcol":    ("Self-collision contacts", "", False, False),
    "solve_ms":   ("Solve time", "ms", False, False),
}

_SIDES = (("left", "l"), ("right", "r"))


def _ensure_paths():
    for p in (str(CFG.CONTROLLER_DIR), str(CFG.PKG.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def sw_radius():
    from mabel.motion_model.swerve_modules import WHEEL_RADIUS
    return WHEEL_RADIUS


def _wire_name(n):
    return n.strip().lower().replace(" ", "_")


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _cos(a, b):
    ua, ub = _unit(a), _unit(b)
    return float(ua @ ub) if ua is not None and ub is not None else np.nan


def _ang(a, b):
    c = _cos(a, b)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1)))) if np.isfinite(c) else np.nan


def _interior(sh, el, wr):
    """Elbow interior angle [deg] of the chain sh-el-wr (180 = straight)."""
    return _ang(sh - el, wr - el)


class KinematicRetargeter:
    def __init__(self, vid, scene=None, frame_clock=True, model=None):
        _ensure_paths()
        os.environ.setdefault("MABEL_LOG_LEVEL", "WARNING")
        import mujoco
        from mabel.teleop_engine import engine as ENG
        from mabel.teleop_engine.hand import HandOptimizer
        from .variants import VARIANTS, configure

        self.mj = mujoco
        self.vid = vid
        self._physical_provider = None
        self._physical_now = time.monotonic
        self._physical_budget_s = 0.025
        self._physical_publication_reserve_s = 0.005
        self._physical_phase_start = None
        self._physical_deadline = None
        self._physical_accepted = None
        self._physical_previous = None
        self._physical_rejected = False
        self._physical_reject_reason = ""
        self._physical_record_context = None
        spec = VARIANTS[vid]
        # the model is reused across resets (loading it is ~1.7 of the ~2.3 s)
        m = self.model = model if model is not None else mujoco.MjModel.from_xml_path(scene or CFG.SCENE)
        d = self.data = mujoco.MjData(m)
        # `home` keyframe: base upright on the floor, lift half-extended.
        # (qpos0 is the assembled pose and is not a standing pose.)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)

        # ideal-servo map: every single-joint POSITION actuator (biasprm[1] < 0)
        # except the swerve wheels, plus the lift tendon (ctrl = stage SUM).
        self._servo = []
        for a in range(m.nu):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            if "Steer" in nm or "Drive" in nm or not m.actuator_biasprm[a, 1] < 0:
                continue
            j = int(m.actuator_trnid[a, 0])
            self._servo.append((a, int(m.jnt_qposadr[j]), *m.jnt_range[j],
                                bool(m.jnt_limited[j])))
        J = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self._lift_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "Lift")
        self._lift_q = [int(m.jnt_qposadr[J(n)]) for n in ("Lift Lower", "Lift Upper")
                        if J(n) >= 0]
        # seed every servo target with the keyframe pose (hold, no lurch)
        for a, qa, *_ in self._servo:
            d.ctrl[a] = d.qpos[qa]
        if self._lift_act >= 0 and self._lift_q:
            d.ctrl[self._lift_act] = sum(d.qpos[q] for q in self._lift_q)

        # lab-wide whole-body settings (config.py): set BEFORE the engine is
        # built; all three are module globals read at call time
        # (the engine's clutch is teleop_engine/safety.py, which binds its OWN
        # CLUTCH_ENABLE from hyperparams at import — patching mabel.config does
        # nothing; measured: the take still froze 9.4 s until this pointed here)
        import mabel.teleop_engine.safety as TS
        import mabel.teleop_engine.wholebody as WB
        TS.CLUTCH_ENABLE = bool(CFG.CLUTCH)
        WB.TORSO_LEAN_MODE = CFG.TORSO_LEAN_MODE
        WB.HEADING_MODE = ENG.HEADING_MODE = CFG.HEADING_MODE
        WB.BASE_TRANSLATION_MODE = CFG.BASE_TRANSLATION_MODE
        ENG.ARM_MAP = spec["arm_map"]
        finger = "anatomical" if spec["fingers"] == "anatomical" else "keypoint"
        self.eng = ENG.TeleopEngine(m, d, arm_map=spec["arm_map"], finger_solver=finger)
        self.spec = configure(self.eng, m, vid)
        self.direct = bool(self.spec.get("direct"))
        self.base_on = bool(CFG.BASE_FOLLOW)
        self.eng.set_retarget_scope("global" if self.base_on else "local")   # apply_settings refines
        self.chassis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mabel_chassis")
        fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_root")
        self._free_q = int(m.jnt_qposadr[fj]) if fj >= 0 else None
        self._steer = []
        for k in ("FL", "FR", "B"):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{k} Steer")
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{k} Steer")
            if a >= 0 and j >= 0:
                self._steer.append((a, int(m.jnt_qposadr[j])))
        self._drive_q = {}
        for k in ("FL", "FR", "B"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{k} Drive")
            if j >= 0:
                self._drive_q[k.lower()] = int(m.jnt_qposadr[j])
        self.swerve = None
        self.settings = dict(getattr(self, "settings", None) or
                             {"scope": CFG.SCOPE if self.base_on else "local", "robot": CFG.BASE_PROFILE})
        if CFG.SWERVE_MODEL:
            from mabel.motion_model.swerve_modules import SwerveModules
            self.swerve = SwerveModules(profile=self.settings["robot"])
            # start from the keyframe's steer angles
            sq = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a).split()[0].lower(): float(d.qpos[q])
                  for a, q in self._steer}
            self.swerve.theta[:] = [sq.get(k, 0.0) for k in self.swerve.keys]
        self._base_real = (0.0, 0.0, 0.0)
        # physical servo: qpos address -> slew key
        self._slew_key = {}
        for a, qa, *_ in self._servo:
            j = int(m.actuator_trnid[a, 0])
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if " Arm " in nm:
                self._slew_key[qa] = ("arm", int(nm.split()[-1]) - 1)
            elif nm == "Torso":
                self._slew_key[qa] = ("torso", None)
            elif nm.startswith("Neck"):
                self._slew_key[qa] = ("neck", None)
        self.slew_sat = 0.0
        # ablation channels (config.CHANNELS): "fixed" = own parks, eased like the engine's channels
        from mabel.teleop_engine.safety import ChannelPark
        e = self.eng
        self.lift_home = float(m.key_ctrl[kid][self._lift_act]) if kid >= 0 and self._lift_act >= 0 else 0.0
        self.park = {"torso": ChannelPark([e.torso_act], [0.0]),
                     "neck": ChannelPark(e.neck_act, [0.0] * len(e.neck_act)),
                     "lift": ChannelPark([self._lift_act], [self.lift_home]) if self._lift_act >= 0 else None}
        for ch in CFG.CHANNELS:
            self.settings.setdefault(ch, "free")
        self.apply_settings(self.settings)

        # wire-named joints the viewer renders
        self.qnames = {}
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm and m.jnt_type[j] in (2, 3):         # slide / hinge
                self.qnames[_wire_name(nm)] = int(m.jnt_qposadr[j])

        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.head_bid = B("head")
        self.palm = {s: B(f"{s}_palm") for s, _ in _SIDES}
        self.tip = {s: (B(f"{s}_thumb_dp"), B(f"{s}_index_ip")) for s, _ in _SIDES}
        self.R_rob = {s: HandOptimizer(m, s).R_rob for s, _ in _SIDES}
        self.arm_q = {s: [int(m.jnt_qposadr[J(f"{s.capitalize()} Arm {i}")])
                          for i in range(1, 8)] for s, _ in _SIDES}
        self.arm_rng = {s: [m.jnt_range[J(f"{s.capitalize()} Arm {i}")]
                            for i in range(1, 8)] for s, _ in _SIDES}
        self.arm_act = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                              f"{s.capitalize()} Arm {i}")
                            for i in range(1, 8)] for s, _ in _SIDES}
        # self-collision = one ARM touching another segment (other arm/trunk);
        # contacts inside one arm (adjacent links, a closing hand) do not count.
        par = np.asarray(m.body_parentid, int)
        seg = np.zeros(m.nbody, int)
        for name, tag in (("l_arm_mount", 1), ("r_arm_mount", 2)):
            root = B(name)
            if root < 0:
                continue
            for b in range(m.nbody):
                a = b
                while a > 0 and a != root:
                    a = par[a]
                if a == root:
                    seg[b] = tag
        self._gseg = seg[np.asarray(m.geom_bodyid, int)]
        self._groot = np.asarray(m.body_rootid, int)[np.asarray(m.geom_bodyid, int)]
        self._rob_root = int(m.body_rootid[self.head_bid])

        # capture the targets the map emits each frame (no second call: the
        # map is stateful — head-anchor low-pass, workspace re-anchor)
        self._tgt = {}
        orig = self.eng._map_wrist_targets

        def _capture(payload, _orig=orig):
            out = _orig(payload)
            self._tgt = {k: np.asarray(v, float)[:3].copy() for k, v in (out or {}).items()}
            return out
        self.eng._map_wrist_targets = _capture

        self._prev_arm = None
        self._prev_t = None
        self._calibrated = False
        self.n = 0
        self.clock = install_frame_clock() if frame_clock else None

    def apply_settings(self, settings):
        """Live switches from the studio: scope (whole body "global" / upper body
        "local") and robot ("mabel" / "nova": swerve profile + joint slews)."""
        st = dict(self.settings)
        st.update({k: v for k, v in (settings or {}).items() if v is not None})
        if settings is not None and "operator_profile" in settings:
            from copy import deepcopy
            from dataclasses import asdict, is_dataclass
            profile = settings["operator_profile"]
            if profile is not None and is_dataclass(profile):
                profile = asdict(profile)
            if profile is not None and not isinstance(profile, dict):
                raise TypeError("operator_profile must be a serialized dict, dataclass, or None")
            # Keep the same explicit measured profile across lab worker resets
            # and deployment adapter rebuilds; no user/default file is loaded.
            st["operator_profile"] = deepcopy(profile)
        sc = str(st.get("scope") or "global").lower()
        st["scope"] = self.eng.set_retarget_scope(sc if self.base_on else "local")
        rb = str(st.get("robot") or "mabel").lower()
        st["robot"] = rb if rb in CFG.SLEWS else "mabel"
        from mabel.motion_model.relative_follow import identified_response
        response = identified_response(CFG.NOVA_DIR / "experiments/base_id/20261023_drive_candidate.json")
        self.base_follow_response = response
        # NOVA whole body: 1:1 POSITION follow since 2026-10-29 (operator: "use position instead of velocity ...
        # when human move forward 1 meter the robot needs to move forward 1 meter"). MABEL_NOVA_BASE_FOLLOW=relative
        # restores the 2026-10-17..28 relative-velocity follower (box + workspace assist) exactly.
        nova_follow = os.environ.get("MABEL_NOVA_BASE_FOLLOW", "position").strip().lower()
        one_to_one = st["robot"] == "nova" and nova_follow != "relative"
        tr = self.eng.wbc.tracker
        if tr.set_follow_mode("relative" if st["robot"] == "nova" and not one_to_one else "position",
                              response_s=response["response_s"]):
            self.eng.wbc.base_anchored = False
        if tr.one_to_one != one_to_one:
            self.eng.wbc.base_anchored = False
        tr.one_to_one = one_to_one
        # 1.5 1/s, 3 cm: 1-D sim of NOVA's measured drive (0.3-0.5 s dead time, 0.8 m/s^2, 0.3 m/s cap) gives
        # <= 5 cm overshoot and ~1 cm final error on a walked 1 m step; the engine's 3.5 rings (+4..8 cm).
        from mabel.teleop_engine.hyperparams import BASE_KP
        tr.kp = float(os.environ.get("MABEL_NOVA_BASE_KP", "1.5")) if one_to_one else BASE_KP
        tr.err_band = float(os.environ.get("MABEL_NOVA_BASE_BAND", "0.03")) if one_to_one else 0.0
        if self.swerve is not None and st["robot"] != self.settings.get("robot"):
            from mabel.motion_model.swerve_modules import SwerveModules
            th = self.swerve.theta.copy()
            self.swerve = SwerveModules(steer0=th, profile=st["robot"])
        self.slews = CFG.SLEWS[st["robot"]]
        if self.spec.get("arm_speed") == "nova":
            # v4 lifts the MABEL preset's legacy 0.8 rad/s arm servo cap.
            # Keep every other preset channel/IK guard and old snapshot intact.
            self.slews = {**self.slews, "arm": list(CFG.SLEWS["nova"]["arm"])}
        for ch, modes in CFG.CHANNELS.items():
            v = str(st.get(ch) or "free").lower()
            st[ch] = v if v in modes else "free"
        self.eng.track_neck = st["neck"] != "fixed"    # fixed: the engine's neck law is not run
        self.eng.enable_gripper_contact = bool(self.spec.get("gripper_contact", False))
        self._robot_limits(st["robot"])
        ee = str(st.get("ee") or "orca").lower()
        st["ee"] = self.eng.set_end_effector(ee if ee in CFG.EE_KINDS else "orca")
        if "operator_profile" in st:
            self.eng.apply_profile(st["operator_profile"])
        self.settings = st
        return st

    def _park_channels(self, dt):
        """After the solve: each "fixed" channel's command eased to home (the arms then
        reach without it — that is the ablation)."""
        for ch, p in self.park.items():
            if p is not None:
                p.update(self.data, self.settings[ch] != "fixed", dt)

    def _robot_limits(self, robot):
        """The robot's own command windows ON the engine, so the lab's output IS what that robot
        commands — there is no separate deploy channel. NOVA: its neck windows (CFG.NOVA_NECK_RANGE
        = its calibrated servo windows, pitch 90 deg up / 53 deg down) and its torso command range
        (CFG.NOVA_TORSO_RANGE); the kinematic servo clamps to the same windows. MABEL: its MJCF."""
        e = self.eng
        if not hasattr(self, "_mjcf_lim"):
            self._mjcf_lim = ([tuple(map(float, r)) for r in e.neck_rng],
                              (float(e.wbc.torso.lo), float(e.wbc.torso.hi)), list(self._servo))
        neck, torso, servo = self._mjcf_lim
        if robot == "nova":
            neck = [tuple(CFG.NOVA_NECK_RANGE[k]) for k in ("neck_yaw", "neck_pitch", "neck_roll")]
            torso = tuple(CFG.NOVA_TORSO_RANGE)
        e.neck_rng = [np.array(r, float) for r in neck]
        e.wbc.torso.lo, e.wbc.torso.hi = torso
        # IK speed caps: the robot's (CFG.NOVA_IK_CAPS), or the engine's own defaults for MABEL. The QP cap is a
        # module global (one robot per worker process); the trust rates are per arm.
        import mabel.teleop_engine.collision as _col
        from mabel.teleop_engine import hyperparams as _hp
        caps = CFG.NOVA_IK_CAPS if robot == "nova" else None
        _col.ARM_QP_VMAX = caps["qp_vmax"] if caps else _hp.ARM_QP_VMAX
        for a in e.arm.values():
            if not hasattr(a, "_trust_distal0"):
                a._trust_distal0 = a.trust_rate_distal
            a.trust_rate_distal = caps["distal"] if caps else a._trust_distal0
            a.trust_override = dict(caps["override"]) if caps else {}
        lim = {e.torso_act: torso, **dict(zip(e.neck_act, neck))}
        self._servo = [(a, qa, *(lim[a] if a in lim else (lo, hi)), on or a in lim)
                       for a, qa, lo, hi, on in servo]

    # ── one frame ──────────────────────────────────────────────────────────
    def reanchor(self):
        """Re-zero live intent at the current pose, retaining the loaded solver.

        Episode reset/seek still rebuilds all state. A live re-anchor keeps
        measured geometry, limits, reach tables and the stable-head window.
        """
        e, d = self.eng, self.data
        e.calibrate()
        e.wbc.reset(d)
        e.wbc.tracker.relative.reset()
        e._ws_assist.reset()
        e._ws_offset[:] = 0.0
        e.sup.rebase()
        for arm in e.arm.values():
            arm.sync(d)
            arm.last_q = np.clip(d.qpos[arm.opt_qadr], arm.lo, arm.hi).copy()
            arm._warm, arm._stuck = True, 0
            arm._prev_contact = arm._steer_q = None
        self._prev_t = self._prev_arm = None
        self._calibrated = True
        self._tgt = {}
        self._base_cmd_ext = self._base_real = (0.0, 0.0, 0.0)
        self._physical_accepted = self._physical_previous = None
        self._physical_record_context = None

    def reset(self):
        """Fresh engine state (seek / new episode) without re-loading the model."""
        keep = dict(self.settings)
        physical_required = self.eng.arm_feedback_required
        provider = self._physical_provider
        budget, reserve, physical_now = (self._physical_budget_s,
                                        self._physical_publication_reserve_s, self._physical_now)
        self.__init__(self.vid, frame_clock=self.clock is not None, model=self.model)
        self.apply_settings(keep)
        self._physical_now = physical_now
        if provider is not None:
            self.set_physical_feedback_provider(provider, compute_budget_s=budget,
                                                publication_reserve_s=reserve)
        if physical_required:
            self.eng.arm_feedback_required = True
            self.eng.arm_feedback_reason = "fresh physical arm feedback required after reset"

    def set_physical_feedback(self, position, valid=True, reason="", age_s=None, compute_budget_s=0.0):
        """Require explicit hardware arm feedback without changing the Lab plant.

        ``position`` contains all 14 wire names (left_arm_1..7/right_arm_1..7),
        in radians. The hardware adapter validates source/transport ages and live
        bits; this boundary also rejects incomplete or non-finite joint values.
        Calling with an invalid sample holds the gated physical gripper path.
        Ordinary virtual Lab runs never call this method.
        """
        names = [f"{s}_arm_{i}" for s in ("right", "left") for i in range(1, 8)]
        position = position if isinstance(position, dict) else {}
        values = {}
        for name in names:
            value = position.get(name)
            if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool) and np.isfinite(value):
                values[name] = float(value)
        complete = len(values) == len(names)
        age_valid = (isinstance(age_s, (int, float, np.floating)) and not isinstance(age_s, bool)
                     and np.isfinite(age_s) and 0.0 <= age_s <= .1)
        budget_valid = (isinstance(compute_budget_s, (int, float, np.floating))
                        and not isinstance(compute_budget_s, bool) and np.isfinite(compute_budget_s)
                        and 0.0 <= compute_budget_s <= .1)
        if age_valid and budget_valid:
            age_valid = age_s + compute_budget_s <= .1
        self.eng.arm_feedback_required = True
        self.eng.arm_measured = values
        self.eng.arm_feedback_valid = bool(valid and complete and age_valid and budget_valid)
        self.eng.arm_feedback_age_s = float(age_s) if age_valid else None
        self.eng.arm_feedback_compute_s = float(compute_budget_s) if budget_valid else 0.0
        self.eng.arm_feedback_reason = str(reason or ("physical arm feedback incomplete/non-finite" if not complete else
                                                     "physical feedback age/compute budget missing/stale"
                                                     if not age_valid or not budget_valid else ""))

    def set_physical_feedback_provider(self, provider, *, compute_budget_s=0.025,
                                       publication_reserve_s=0.005):
        """Install a nonblocking late hardware snapshot provider, gated to v5.

        The callback returns position/valid/reason/age_s/torso. Its source age is
        measured at this late boundary; the deterministic compute budget is an
        additional explicit replay input. Wall time separately rejects overruns.
        """
        if not callable(provider):
            raise TypeError("physical feedback provider must be callable")
        if (not np.isfinite(compute_budget_s) or not np.isfinite(publication_reserve_s)
                or not 0.0 <= publication_reserve_s < compute_budget_s <= .1):
            raise ValueError("require 0 <= publication reserve < compute budget <= 100 ms")
        self._physical_provider = provider
        self._physical_budget_s = float(compute_budget_s)
        self._physical_publication_reserve_s = float(publication_reserve_s)
        self.eng.physical_feedback_refresh = self._refresh_physical_feedback
        if self.eng.enable_gripper_contact:
            self.eng.arm_feedback_required = True
            self.eng.arm_feedback_valid = False
            self.eng.arm_feedback_reason = "waiting for late physical feedback"

    def _refresh_physical_feedback(self):
        if not self.eng.enable_gripper_contact or self._physical_provider is None:
            return
        start = self._physical_now()
        self._physical_phase_start = start
        self._physical_deadline = (start + self._physical_budget_s
                                   - self._physical_publication_reserve_s)
        try:
            sample = self._physical_provider()
            if not isinstance(sample, dict):
                raise ValueError("provider did not return a complete snapshot")
            self.set_physical_feedback(sample.get("position"), valid=sample.get("valid") is True,
                                       reason=sample.get("reason", ""), age_s=sample.get("age_s"),
                                       compute_budget_s=self._physical_budget_s)
            torso = sample.get("torso")
            finite = (isinstance(torso, (int, float, np.floating)) and not isinstance(torso, bool)
                      and np.isfinite(torso))
            self.eng.neck_torso_measured = float(torso) if finite else None
        except Exception as exc:
            self.set_physical_feedback({}, valid=False,
                                       reason=f"late physical feedback unavailable: {type(exc).__name__}",
                                       compute_budget_s=self._physical_budget_s)
            self.eng.neck_torso_measured = None

    def _physical_state(self):
        projector = self.eng.projector
        acts = np.r_[self.eng.arm_act["right"], [self.eng.wrist_act["right"]],
                     self.eng.arm_act["left"], [self.eng.wrist_act["left"]]].astype(int)
        return (self.data.qpos[projector.qadr].copy(), self.data.ctrl[acts].copy(),
                None if projector.q_cmd is None else projector.q_cmd.copy())

    def _begin_physical_command(self):
        self._physical_phase_start = self._physical_deadline = None
        self._physical_rejected = False
        self._physical_reject_reason = ""
        self._physical_record_context = None
        if not (self.eng.enable_gripper_contact and self.eng.arm_feedback_required):
            self._physical_previous = None
            return
        if self._physical_accepted is None:
            self._physical_accepted = self._physical_state()
        self._physical_previous = tuple(None if value is None else value.copy()
                                        for value in self._physical_accepted)

    def accept_physical_command(self, record=None, *, commit=True):
        """Final publication check; False restores the prior accepted arm state.

        A live adapter uses ``step(..., defer_physical_accept=True)`` and calls
        this immediately before sending. Rechecking an already accepted frame
        can still reject it before publication; that frame's previous snapshot
        remains available until the next step starts.
        """
        if self._physical_previous is None:
            return True
        projector = self.eng.projector
        invalid = self._physical_phase_start is not None and projector.physical_feedback_hold
        late = self._physical_deadline is not None and self._physical_now() > self._physical_deadline
        if self._physical_rejected or late or invalid:
            if not self._physical_rejected:
                self._physical_reject_reason = (projector.contact_guard_reason if invalid else
                                               "physical command publication deadline exceeded; previous command held")
            self._physical_rejected = True
            qpos, ctrl, command = self._physical_previous
            acts = np.r_[self.eng.arm_act["right"], [self.eng.wrist_act["right"]],
                         self.eng.arm_act["left"], [self.eng.wrist_act["left"]]].astype(int)
            self.data.qpos[projector.qadr] = qpos
            self.data.ctrl[acts] = ctrl
            projector.q_cmd = None if command is None else command.copy()
            self._physical_accepted = tuple(None if value is None else value.copy()
                                            for value in self._physical_previous)
            projector.physical_feedback_hold = True
            projector.contact_guard_reason = self._physical_reject_reason
            self.mj.mj_kinematics(self.model, self.data)
            self.mj.mj_comPos(self.model, self.data)
            if record is not None and self._physical_record_context is not None:
                # This is a held frame, so rebuilding its visual record cannot
                # release a late newly computed closure.
                updated = self._record(*self._physical_record_context)
                record.clear()
                record.update(updated)
            return False
        if commit:
            self._physical_accepted = self._physical_state()
        return True

    @property
    def physical_command_deadline(self):
        """Absolute monotonic delivery deadline shared across local processes.

        The local acceptance check is earlier by the configured transport
        reserve. A receiver must still discard a target delivered after this
        deadline; acknowledging actual acceptance is a separate IPC contract.
        """
        return (None if self._physical_phase_start is None else
                self._physical_phase_start + self._physical_budget_s)

    def step(self, pose, now, collide=True, frame=None, defer_physical_accept=False):
        """Retarget one OperatorPose at clock `now` [s]; returns the frame record.
        `frame` (the raw {kind, p}) is only read by variants that need the wire flags (RL)."""
        eng, m, d = self.eng, self.model, self.data
        self._tgt = {}
        if self.clock is not None:
            self.clock.now = float(now)
        # A deployment adapter can inject measured wheel/lift qpos between
        # frames. Refresh BOTH geometry and cdof before mapping or collision
        # Jacobians; mj_kinematics alone leaves mj_jac's world-frame data stale.
        self.mj.mj_kinematics(m, d)
        self.mj.mj_comPos(m, d)
        self._begin_physical_command()
        t0 = time.perf_counter()
        payload = eng._payload(pose)
        if not self._calibrated and pose.head is not None:
            eng.calibrate(pose)
            self._calibrated = True
        dt = (now - self._prev_t) if self._prev_t is not None else 1.0 / 60.0
        eng._frame_dt = float(np.clip(dt, 1e-3, 0.1))
        # The base follower gets the UNCLIPPED interval (2026-10-28): with the 0.1 s clip a 0.15-0.23 s Wi-Fi gap
        # was divided by 0.1 s (operator speed overestimated up to 2.3x) and a real stall could never reach
        # relative_follow.STALL_S, so it was never rebased. Torso / lift slews keep the clipped dt.
        eng._follow_dt = float(max(dt, 1e-3))
        eng._retarget(payload, None, now)
        self.accept_physical_command(commit=False)
        if self.direct:
            self._direct_copy(pose)
        self._park_channels(eng._frame_dt)
        if self.base_on and self._free_q is not None:
            self._base_step(eng._frame_dt)
        solve_ms = (time.perf_counter() - t0) * 1e3
        self._servo_ideal(eng._frame_dt)
        if collide:
            self.mj.mj_fwdPosition(m, d)
        else:
            self.mj.mj_kinematics(m, d)
        self._physical_record_context = (pose, payload, now, solve_ms, collide)
        rec = self._record(*self._physical_record_context)
        self.accept_physical_command(rec, commit=not defer_physical_accept)
        self._prev_t = now
        self.n += 1
        return rec

    def _base_step(self, dt):
        """The engine's base command, realised exactly. Mirrors what
        TeleopEngine.update() does at the point of consumption (workspace-assist
        twist added to the target, then wbc.base_step ramps it through the
        tip-safe motion model), then integrates the chassis-frame twist into the
        free joint instead of driving wheels."""
        eng, d = self.eng, self.data
        w = eng.wbc
        if (getattr(eng, "_ws_assist", None) is not None and eng.track_arms
                and not eng.nav_mode and getattr(w, "base_in_loop", False)
                and w.allow_base_assist):
            ax, ay = eng._ws_assist.base_twist()
            if ax or ay:
                bt = tuple(w.base_tgt or (0.0, 0.0, 0.0))
                w.base_tgt = (bt[0] + ax, bt[1] + ay, bt[2])
        w.base_step(d, dt)
        self._apply_base(w.base_cmd, dt)

    def _apply_base(self, cmd, dt):
        """A chassis-frame twist COMMAND (MuJoCo chassis: front -x) -> what three swerve
        modules deliver -> integrated into the free joint; steer / wheel joints follow.
        Shared by the engine's base and any other twist source (the RL policy)."""
        d = self.data
        self._base_cmd_ext = tuple(float(v) for v in cmd)
        if self.swerve is not None:                  # what three modules can deliver
            vx, vy, om = self.swerve.step(cmd, dt)
        else:
            vx, vy, om = (float(v) for v in cmd)
        self._base_real = (vx, vy, om)
        qa = self._free_q
        Rb = d.xmat[self.chassis].reshape(3, 3)
        v = Rb @ np.array([vx, vy, 0.0])
        d.qpos[qa] += v[0] * dt
        d.qpos[qa + 1] += v[1] * dt
        if om:
            h = 0.5 * om * dt                        # world-z rotation, pre-multiplied
            qz = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
            q = d.qpos[qa + 3:qa + 7].copy()
            out = np.empty(4)
            self.mj.mju_mulQuat(out, qz, q)
            self.mj.mju_normalize4(out)
            d.qpos[qa + 3:qa + 7] = out
        if self.swerve is not None:                  # wheels where the model has them
            sw = self.swerve
            for a, qs in self._steer:
                k = self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_ACTUATOR, a).split()[0].lower()
                d.qpos[qs] = float(sw.theta[sw.keys.index(k)])
            for k, qd in self._drive_q.items():
                d.qpos[qd] += float(sw.s[sw.keys.index(k)]) / sw_radius() * dt
        else:
            for a, qs in self._steer:                # wheels point where swerve_ik says
                d.qpos[qs] = float(d.ctrl[a])

    def retrack_base(self, dt):
        """Between operator frames (a robot gateway in 1:1 position follow): re-close the base position loop on
        the pose set_base_pose just mirrored and ramp it through the same motion model. False = not applicable."""
        w = self.eng.wbc
        if not (getattr(w, "base_in_loop", False) and w.tracker.one_to_one):
            return False
        self.mj.mj_kinematics(self.model, self.data)       # xpos from the pose just set (set_base_pose writes qpos)
        tgt = w.tracker.recommand()
        if tgt is None:
            return False
        w.base_tgt = tgt
        w.base_step(self.data, float(dt))
        self._apply_base(w.base_cmd, float(dt))
        return True

    def set_base_pose(self, x, y, yaw):
        """Put the chassis at a MEASURED world pose (x, y, yaw; MuJoCo world). For a
        robot gateway (NOVA) in whole-body scope: the engine's base twist then corrects
        from where the robot really is, instead of integrating its own open-loop model
        and drifting from the robot. Unused by the lab (its model IS the measurement)."""
        if self._free_q is None or self.chassis < 0:
            return
        d, qa = self.data, self._free_q
        R = d.xmat[self.chassis].reshape(3, 3)
        dyaw = float(yaw) - float(np.arctan2(R[1, 0], R[0, 0]))
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        if abs(dyaw) > 1e-9:
            h = 0.5 * dyaw
            out = np.empty(4)
            self.mj.mju_mulQuat(out, np.array([np.cos(h), 0.0, 0.0, np.sin(h)]), d.qpos[qa + 3:qa + 7].copy())
            self.mj.mju_normalize4(out)
            d.qpos[qa + 3:qa + 7] = out
            self.mj.mj_kinematics(self.model, d)
        d.qpos[qa] += float(x) - float(d.xpos[self.chassis][0])
        d.qpos[qa + 1] += float(y) - float(d.xpos[self.chassis][1])
        self.mj.mj_kinematics(self.model, d)

    def base_pose(self):
        """Chassis world pose in the gateway's robot_state.base shape."""
        d = self.data
        c = self.chassis
        if c < 0:
            return None
        R = d.xmat[c].reshape(3, 3)
        out = {"x": float(d.xpos[c][0]), "y": float(d.xpos[c][1]), "z": float(d.xpos[c][2]),
               "yaw": float(np.arctan2(R[1, 0], R[0, 0])),
               "quat": [float(q) for q in d.xquat[c]],
               "steer": [float(d.qpos[qs]) for _, qs in self._steer],
               "cmd": [float(v) for v in getattr(self, "_base_cmd_ext", self.eng.wbc.base_cmd)],
               "real": [float(v) for v in self._base_real]}
        out["followMode"] = self.eng.wbc.tracker.follow_mode
        out["followResponse"] = dict(getattr(self, "base_follow_response", {}))
        sw = self.swerve
        if sw is not None and sw.last:
            out.update(wheel_v=[float(v) for v in sw.last["wheel_v"]],
                       wheel_v_req=[float(v) for v in sw.last["wheel_v_req"]],
                       wheel_a=[float(v) for v in sw.last["wheel_a"]],
                       steer_rate=[float(v) for v in sw.last["steer_rate"]],
                       steer_err=[float(v) for v in sw.last["steer_err"]],
                       scrub=float(sw.last["scrub"]), desat=float(sw.last["desat"]))
        return out

    def _servo_ideal(self, dt=None):
        """ctrl -> qpos. SERVO "ideal": at once. SERVO "slew" (default): each joint
        moves toward its target at no more than the robot's setpoint slew (rad/s,
        lift m/s) — the HAL's hard rate limit — so the next solve reads back a pose
        that lags exactly as the robot's would. `slew_sat` = the fraction of slewed
        joints that hit their limit this frame (demand above what the robot can do)."""
        d = self.data
        slew = CFG.SERVO == "slew" and dt is not None and dt > 0
        sl = self.slews
        n_sat = n = 0
        for a, qa, lo, hi, lim in self._servo:
            v = float(d.ctrl[a])
            v = min(max(v, lo), hi) if lim else v
            key = self._slew_key.get(qa) if slew else None
            if key is not None:
                vmax = sl["arm"][key[1]] if key[0] == "arm" else sl[key[0]]
                step = vmax * dt
                dq = v - float(d.qpos[qa])
                n += 1
                if abs(dq) > step:
                    n_sat += 1
                    v = float(d.qpos[qa]) + (step if dq > 0 else -step)
            d.qpos[qa] = v
        if self._lift_act >= 0 and self._lift_q:
            tot = float(d.ctrl[self._lift_act])
            if slew:
                cur = sum(float(d.qpos[q]) for q in self._lift_q)
                step = sl["lift"] * dt
                n += 1
                if abs(tot - cur) > step:
                    n_sat += 1
                    tot = cur + (step if tot > cur else -step)
            half = tot / len(self._lift_q)
            for q in self._lift_q:
                d.qpos[q] = half
        self.slew_sat = n_sat / n if n else 0.0

    def _direct_copy(self, pose):
        """Joint-angle copy (the ablation's `direct` column, methods.py):
        operator shoulder->elbow->wrist angles onto Arm 1/2/3/4, clipped."""
        from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT
        R = np.asarray(AVP_TO_ROBOT, float)
        d, m = self.data, self.model
        if pose.head is None:
            return
        h = R @ np.asarray(pose.head, float)[:3, 3]
        for side, _ in _SIDES:
            hand = getattr(pose, f"{side}_hand")
            el = getattr(pose, f"{side}_elbow")
            if hand is None or el is None:
                continue
            w = R @ np.asarray(hand[0], float)[:3, 3]
            e = R @ np.asarray(el, float)[:3, 3]
            s = h + np.array([0.0, 0.129 if side == "right" else -0.129, -0.22])
            u = _unit(e - s)
            v = _unit(w - e)
            if u is None or v is None:
                continue
            elbow = float(np.arccos(np.clip(u @ v, -1, 1)))
            pitch = float(np.arctan2(-u[2], np.linalg.norm(u[:2])))
            yaw = float(np.arctan2(u[1], u[0]))
            roll = float(np.arctan2(v[2], np.linalg.norm(v[:2])))
            for i, val in enumerate((yaw, pitch, roll, -elbow)):
                a = self.arm_act[side][i]
                lo, hi = m.actuator_ctrlrange[a]
                d.ctrl[a] = float(np.clip(val, lo, hi))

    def _grip_mount(self, side):
        """The gripper's mount (end_effector.gripper_frame) in the J7 body frame, row-major 4x4, cached."""
        cache = self.__dict__.setdefault("_grip_mounts", {})
        if side not in cache:
            from mabel.teleop_engine.end_effector import gripper_frame
            cache[side] = gripper_frame(self.model, side, self.palm[side], self.R_rob[side]).ravel().tolist()
        return cache[side]

    # ── measurement ────────────────────────────────────────────────────────
    def _record(self, pose, payload, now, solve_ms, collide):
        eng, m, d = self.eng, self.model, self.data
        anchored = eng._world_ref is not None
        q = {n: float(d.qpos[a]) for n, a in self.qnames.items()}
        Mb = np.asarray(getattr(eng, "_frame_M", None) if getattr(eng, "_frame_M", None)
                        is not None else eng._body_map_M(), float)
        head_h = (np.asarray(pose.head, float)[:3, 3] if pose.head is not None else None)
        head_r = d.xpos[self.head_bid].copy()
        met = {}
        geo = {}
        for side, s in _SIDES:
            ar = eng.arm[side]
            wr = ar._ee_point(d)[0].copy()            # the tracked point (ORCA wrist / gripper base face)
            wa = d.xanchor[ar.orca_wrist_jid]           # the ORCA wrist anchor: the viewer's GLB<->MJ calibration
            sh = d.xpos[ar.shoulder_bid].copy()
            el = d.xpos[ar.elbow_bid].copy()
            Rp = d.xmat[self.palm[side]].reshape(3, 3) @ self.R_rob[side]
            tgt = self._tgt.get(side)
            g = {"wr": wr.tolist(), "wa": wa.tolist(), "tgt": tgt.tolist() if tgt is not None else None}
            if eng.ee == "gripper":
                g["mount"] = self._grip_mount(side)     # OpenArm link7 frame in the J7 body (constant)
            hand = getattr(pose, f"{side}_hand")
            elb = getattr(pose, f"{side}_elbow")
            sh_h = eng.shoulder.shoulder_world(side)
            L_h = eng.shoulder.arm_length(side) if eng.shoulder.ready(side) else None
            L_r = float(eng.arm_reach[side])
            nan = np.nan
            vals = dict.fromkeys(("cos_reach", "cos_upper", "cos_fore", "place_pct",
                                  "ext_err", "elbow_err", "orient_deg", "track_mm",
                                  "aperture_h", "aperture_r"), nan)
            tb, ib = self.tip[side]
            if eng.ee == "gripper":
                vals["aperture_r"] = float(eng.gripper_aperture[side].open * 1e3)
            elif tb >= 0 and ib >= 0:
                vals["aperture_r"] = float(np.linalg.norm(d.xpos[tb] - d.xpos[ib]) * 1e3)
            if anchored and hand is not None and head_h is not None:
                w_h = np.asarray(hand[0], float)[:3, 3]
                if tgt is not None:
                    vals["track_mm"] = float(np.linalg.norm(tgt - wr) * 1e3)
                joints = getattr(pose, f"{side}_joints") or {}
                if "thumb_tip" in joints and "index_tip" in joints:
                    vals["aperture_h"] = float(np.linalg.norm(
                        np.asarray(joints["thumb_tip"])[:3, 3]
                        - np.asarray(joints["index_tip"])[:3, 3]) * 1e3)
                if L_h:
                    p_h = Mb @ (w_h - head_h) / L_h
                    p_r = (wr - head_r) / L_r
                    vals["place_pct"] = float(np.linalg.norm(p_h - p_r) * 100)
                if sh_h is not None:
                    v_sw_h = Mb @ (w_h - sh_h)
                    v_sw_r = wr - sh
                    vals["cos_reach"] = _cos(v_sw_h, v_sw_r)
                    if L_h:
                        vals["ext_err"] = abs(float(np.linalg.norm(v_sw_h)) / L_h
                                              - float(np.linalg.norm(v_sw_r)) / L_r)
                    if elb is not None:
                        e_h = np.asarray(elb, float)[:3, 3]
                        vals["cos_upper"] = _cos(Mb @ (e_h - sh_h), el - sh)
                        vals["cos_fore"] = _cos(Mb @ (w_h - e_h), wr - el)
                        vals["elbow_err"] = abs(_interior(Mb @ sh_h, Mb @ e_h, Mb @ w_h)
                                                - _interior(sh, el, wr))
                        g["op_el"] = (head_r + Mb @ (e_h - head_h)).tolist()
                # palm frame, exactly as the solver is given it
                from mabel.teleop_engine.geometry import operator_frame, palm_axes
                R_op = operator_frame(payload.get(f"{side}_joints"),
                                      payload.get(f"{side}_hand_mats"))
                if R_op is not None:
                    fwd, up = palm_axes(Mb @ R_op, side)
                    vals["orient_deg"] = max(_ang(np.asarray(up), Rp[:, 2]),
                                             _ang(np.asarray(fwd), Rp[:, 0]))
            for k, v in vals.items():
                met[f"{k}_{s}"] = v
            geo[side] = g
        # whole-robot measures
        arm_now = np.array([d.qpos[a] for sd, _ in _SIDES for a in self.arm_q[sd]])
        sat, n = 0, 0
        for sd, _ in _SIDES:
            for a, (lo, hi) in zip(self.arm_q[sd], self.arm_rng[sd]):
                if hi > lo:
                    n += 1
                    v = d.qpos[a]
                    sat += min(v - lo, hi - v) < 0.01 * (hi - lo)
        met["sat_pct"] = 100.0 * sat / max(n, 1)
        dt = (now - self._prev_t) if self._prev_t is not None else None
        met["qvel"] = (float(np.max(np.abs(arm_now - self._prev_arm))) / dt
                       if self._prev_arm is not None and dt and dt > 1e-4 else np.nan)
        self._prev_arm = arm_now
        n_sc = 0
        if collide:
            g, sg, rr = self._groot, self._gseg, self._rob_root
            for c in range(int(d.ncon)):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if g[g1] == rr and g[g2] == rr and sg[g1] != sg[g2]:
                    n_sc += 1
        met["selfcol"] = float(n_sc)
        met["solve_ms"] = solve_ms
        record = {"t": float(now), "anchored": bool(anchored), "q": q, "geo": geo,
                "head": head_r.tolist(), "m": met, "base": self.base_pose(),
                "torso": float(d.qpos[eng.torso_qadr]),
                "clutched": bool(eng.clutch.clutched), "slew_sat": float(self.slew_sat),
                "ee": eng.ee, "gripper": eng.gripper_open,
                "settings": dict(self.settings)}
        if eng.enable_gripper_contact and eng.arm_feedback_required and eng.projector is not None:
            projector = eng.projector
            record["physical_contact"] = {
                "feedback_valid": bool(eng.arm_feedback_valid),
                "hold": bool(projector.physical_feedback_hold),
                "reason": projector.contact_guard_reason,
                "closing_lead_mm": float(1000.0 * projector.closing_lead),
                "closing_lead_limit_mm": float(1000.0 * projector.closing_lead_limit),
                "feedback_age_ms": (1000.0 * eng.arm_feedback_age_s
                                    if eng.arm_feedback_age_s is not None else None),
                "compute_budget_ms": 1000.0 * eng.arm_feedback_compute_s,
                "effective_age_ms": (1000.0 * (eng.arm_feedback_age_s + eng.arm_feedback_compute_s)
                                     if eng.arm_feedback_age_s is not None else None),
                "publication_rejected": bool(self._physical_rejected),
                "publication_phase_ms": (1000.0 * (self._physical_now() - self._physical_phase_start)
                                         if self._physical_phase_start is not None else None),
            }
        return record


class FrameClock:
    """`time` stand-in whose monotonic()/time() read the EPISODE clock.

    The finger adapters pace their solves (30 Hz) and run their one-euro filters
    on time.monotonic() — right on the robot, where frames arrive in real time,
    but it makes a replay depend on how loaded the machine is (measured: the same
    clip twice through the same engine differed by up to 0.26 rad on the thumb).
    Each worker process rebinds those modules' `time` to this clock, so a variant
    sees exactly the timing the operator produced. Everything else delegates."""

    def __init__(self):
        import time as _t
        self._t = _t
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def __getattr__(self, k):
        return getattr(self._t, k)


def install_frame_clock():
    """Rebind the wall clock of the frame-paced modules (process-wide: call only
    in a worker / evaluation process)."""
    import mabel.hand_retargeting.keypoint_ik as KI
    import mabel.teleop_engine.hand_anatomical as HA
    import mabel.teleop_engine.hand_keypoint as HK
    clk = FrameClock()
    for m in (HK, HA, KI):
        m.time = clk
    return clk


def pose_from_frame(frame):
    """{kind: wire|avp, p: payload} -> OperatorPose (None if undecodable)."""
    from mabel.types import OperatorPose
    try:
        if frame["kind"] == "wire":
            return OperatorPose.from_teleop_frame(frame["p"])
        return OperatorPose.from_payload(frame["p"])
    except Exception:
        return None


def make_retargeter(vid, **kw):
    """The right retargeter class for a variant: the TeleopEngine (KinematicRetargeter)
    or a learned whole-body policy (rl_variant.RLRetargeter)."""
    from .variants import VARIANTS
    if VARIANTS[vid].get("rl"):
        from .rl_variant import RLRetargeter
        return RLRetargeter(vid, **kw)
    return KinematicRetargeter(vid, **kw)


def _home_record(kr):
    """The record of the freshly reset (home keyframe) pose, so the viewer and the headset show home at once,
    with no frame needed (a paused replay or an idle headset would otherwise keep the old pose on screen)."""
    rec = getattr(kr, "_record", None)
    if rec is None:
        return None
    from mabel.types import OperatorPose
    kr.mj.mj_forward(kr.model, kr.data)
    return rec(OperatorPose(), {}, 0.0, 0.0, False)


def run_worker(vid, conn):
    """Process entry. Protocol (pickled tuples over a Pipe):
         in : ("frame", seq, t, frame) | ("reset",) | ("stop",) | ("set", {scope, robot, torso, neck, lift})
         out: ("ready", vid, info) | ("rec", vid, seq, rec) | ("error", vid, msg)
    Latest-wins: if several frames are queued, all but the newest are dropped
    (counted) — the live view must never lag the operator. Offline evaluation
    does not use the pool; it steps KinematicRetargeter directly, every frame."""
    try:
        kr = make_retargeter(vid)
        # the gripper's mount on J7 goes out with "ready" so the viewer can build it before any toggle, and its
        # reach tables are measured in the background so the first switch to it is instant
        conn.send(("ready", vid, {"nq": kr.model.nq,
                                  "ee_mount": {s: kr._grip_mount(s) for s in ("right", "left")}}))
        import threading
        threading.Thread(target=kr.eng.precompute_end_effector, args=("gripper",), daemon=True,
                         name="ee-precompute").start()
    except Exception:
        conn.send(("error", vid, traceback.format_exc()))
        return
    dropped = 0
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "set":
            kr.apply_settings(msg[1])
            continue
        if msg[0] in ("reset", "reanchor"):
            try:
                (getattr(kr, "reanchor", kr.reset) if msg[0] == "reanchor" else kr.reset)()
                if msg[0] == "reset" and (home := _home_record(kr)) is not None:
                    conn.send(("rec", vid, -1, home))
            except Exception:
                conn.send(("error", vid, traceback.format_exc()))
            continue
        if msg[0] == "stop":
            return
        latest = msg
        while conn.poll():
            nxt = conn.recv()
            if nxt[0] == "set":                     # a switch is never dropped
                kr.apply_settings(nxt[1])
                continue
            if nxt[0] == "frame" and latest[0] == "frame":
                dropped += 1
            if nxt[0] == "reset" and latest[0] == "reset":
                continue                         # back-to-back resets are one rebuild
            latest = nxt
            if nxt[0] in ("reset", "reanchor", "stop"):
                break
        if latest[0] == "stop":
            return
        if latest[0] in ("reset", "reanchor"):
            try:
                (getattr(kr, "reanchor", kr.reset) if latest[0] == "reanchor" else kr.reset)()
                if latest[0] == "reset" and (home := _home_record(kr)) is not None:
                    conn.send(("rec", vid, -1, home))
            except Exception:
                conn.send(("error", vid, traceback.format_exc()))
            continue
        _, seq, t, frame = latest
        pose = pose_from_frame(frame)
        if pose is None:
            continue
        try:
            rec = kr.step(pose, t, frame=frame)
            rec["dropped"] = dropped
            conn.send(("rec", vid, seq, rec))
        except Exception:
            conn.send(("error", vid, traceback.format_exc()))

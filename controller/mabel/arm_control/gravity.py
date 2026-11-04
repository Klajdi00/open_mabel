"""gravity.py — arm gravity feed-forward, in the DaMiao raw-motor frame.

WHERE THIS LIVES AND WHY. This is the real-robot arm control law. It belongs under
``controller/`` with the rest of the control code (dynamic model, compliance, collision,
PD) so it is modular, debuggable, and reused identically by the hardware bridge, the sim,
and the experiments — the hardware bridge only *hosts* the arms Teensy link; the math is
here. It is a thin, frame-aware layer over :class:`mabel.dynamic_models.MujocoDynamics`
(the shared inverse-dynamics engine): MujocoDynamics computes gravity from the MJCF, this
module maps it into the raw motor frame each joint's firmware actually speaks and enforces
the safety envelope.

THE LAW. The arms run the DaMiao MIT law  T = kp·(q* − q) + kd·(q̇* − q̇) + t_ff. kd is
packed into 12 bits over [0, 5], so kd ≤ 5 is a HARDWARE cap; a critically damped loop
then caps kp ≈ 15 on the shoulder, and against ~10 N·m of gravity a pure PD droops
~40° at steady state (measured — see docs/gravity_ff_tracking.pdf). No (kp, kd) fixes it.
Gravity MUST be fed forward, not fought:  t_ff = τ_g(q)  deletes the gravity term and
frees kp for tracking.

FRAME TRAP. The firmware commands RAW MOTOR angles; gravity is computed in the MODEL
frame. They relate by the calibration `dir`, and torque transforms the same way:
    q_model = model_mid + dir·(q_raw − raw_mid)   ⇒   τ_raw = dir · τ_model.
A wrong `dir` drives the joint DOWN at full feed-forward torque — hence the safety below.

HANDS (the shake bug this file must never re-introduce). The canonical MJCF carries the
1.12 kg ORCA hand per side at the very END of the chain (max lever arm). If the hands are
NOT bolted on but the model still includes them, the feed-forward LIFTS 1.12 kg of
phantom mass: the FF pushes up, the PD fights back, and the arm SHAKES (observed live
2026-07-14). So the hand mass is a FIRST-CLASS, EXPLICIT choice — :class:`HandsConfig` —
not an env afterthought, and it DEFAULTS to NONE (the hands are off). Getting it wrong is
now a loud, logged decision, not a silent 1.12 kg error.
"""
from __future__ import annotations

import enum
import math
import os

from ..dynamic_models.model import MujocoDynamics

_TAU = 2.0 * math.pi

# firmware joint id (l_j1..r_j7) -> MJCF joint name
_MJCF_JOINT = {f"{s}_j{k}": f"{'Left' if s == 'l' else 'Right'} Arm {k}"
               for s in ("l", "r") for k in range(1, 8)}
# OPTIONAL posture joints (2026-07-29). These ride the SAME mj_forward as the arms:
# when the caller's pose carries them they are set on the model and their gravity
# torque is emitted; when absent the evaluation is byte-identical to before (they
# stay at qpos0 and are not returned), so existing arm-only callers (armctl, the
# studio) are untouched. Two effects for the price of one evaluation:
#   1. τ_torso becomes ARM-POSE-AWARE — the torso subtree carries both arms, so an
#      extended arm moves the subtree CoM and the single-sinusoid hold law
#      (torso_control.hold_tau, fitted at ONE arm pose) goes wrong by several N·m.
#   2. τ_arm becomes TORSO-LEAN-AWARE — gravity enters the shoulder through the
#      trunk's orientation; evaluating the arms at torso=0 while the robot leans
#      −40° was a silent 10-20% shoulder error (see basic/gravity.py BASE
#      ORIENTATION note).
_EXTRA_MJCF_JOINT = {"torso": "Torso"}
# the end-effector subtree roots — everything below these is "the hand".
_HAND_ROOTS = ("left_tower", "right_tower")

# Safety envelope (shared by every consumer; overridable per-instance). The DEFAULTS come from
# the shared controller/control_param.json (arm.gravity_feedforward) — the SAME file the setup
# studio's armctl and mabel_hw's gains read — so editing that one file retunes the gravity FF on
# every path. A missing file keeps the historical constants below, so nothing breaks without it.
try:
    from ..control_params import arm_gravity_ff as _arm_ff
    _FF = _arm_ff() or {}
except Exception:
    _FF = {}
FF_CLAMP = float(_FF.get("clamp", 0.6))       # |t_ff| <= FF_CLAMP * t_max per joint
FF_RAMP_S = float(_FF.get("ramp_s", 1.5))     # fade t_ff in over this long on (re)engage
FF_POSE_PAD = float(_FF.get("pose_pad", 0.12))  # rad (~7°) slack past the MJCF limit
FF_SCALE_DEFAULT = float(_FF.get("scale", 1.0))  # global τ_g trim when no env/arg override
# Per-joint trims measured against the REAL arm (control_param.json
# arm.gravity_feedforward.scale_by_joint). 2026-07-28 no-hands sysid
# (hardware_bridge/experiments/exp_arm_gravity_sysid.py): measured holding torque vs
# the hands-stripped MJCF fit at scale l_j4≈0.96, l_j2≈0.72, l_j1≈0.40 — the model
# slightly overestimates the shoulders. Multiplies ON TOP of the global scale.
FF_SCALE_BY_JOINT = {str(k): float(v)
                     for k, v in (_FF.get("scale_by_joint") or {}).items()}


class HandsConfig(enum.Enum):
    """Which end-effector mass the gravity model carries. EXPLICIT so the 1.12 kg
    phantom-hand shake can never come back silently."""
    NONE = "none"         # no hands bolted on (default, 2026-07) — hand mass stripped
    ORCA = "orca"         # ORCA hands attached — full MJCF mass kept

    @classmethod
    def from_env(cls, default: "HandsConfig" = None) -> "HandsConfig":
        """MABEL_ARMS_HANDS = none|orca (or the legacy MABEL_HW_ARMS_FF_HANDS=1 => orca).

        When no env is set, the default comes from the shared registry
        (control_param.json arm.gravity_feedforward.hands) via ``registry_default()``
        unless the caller passes one — so bolting the hands on is ONE registry edit,
        not an env hunt across launch scripts. An explicit constructor arg still wins.
        """
        default = default or cls.registry_default()
        v = os.getenv("MABEL_ARMS_HANDS", "").strip().lower()
        if v in ("none", "off", "0"):
            return cls.NONE
        if v in ("orca", "hands", "on", "1"):
            return cls.ORCA
        if os.getenv("MABEL_HW_ARMS_FF_HANDS", "0").strip() in ("1", "true", "yes", "on"):
            return cls.ORCA
        return default

    @classmethod
    def registry_default(cls) -> "HandsConfig":
        """The registry's hands choice (control_param.json arm.gravity_feedforward.hands).
        Unset/unparsable degrades to NONE — the phantom-mass shake must stay opt-in."""
        try:
            return cls(str(_FF.get("hands", "none")).strip().lower())
        except ValueError:
            return cls.NONE


class ArmGravityCompensator:
    """Per-joint gravity feed-forward in the raw motor frame, on top of MujocoDynamics.

    Parameters
    ----------
    mjcf_path : str            path to the canonical robot MJCF
    dir_by_joint : {fw_id: ±1} calibration direction (raw↔model sign) per firmware joint
    tmax_by_joint : {fw_id: N·m} per-joint torque ceiling (from the manifest) for the clamp
    hands : HandsConfig        NONE (default) strips the hand mass; ORCA keeps it
    mount_tilt_deg : float     TRUE shoulder-mount tilt on the metal (MJCF draws −10°; the
                               physical arm measures 0°). Env MABEL_ARMS_MOUNT_TILT_DEG.
    ff_scale : float           global trim on τ_g (1=full, 0=pure PD for A/B). Env
                               MABEL_ARMS_FF_SCALE (legacy MABEL_HW_ARMS_FF_SCALE).
    ff_clamp / ff_ramp_s / pose_pad : safety-envelope overrides (see module constants)

    ``tff(model_angles, ramp)`` returns {fw_id: t_ff [N·m, raw frame]} for the measured
    MODEL-frame pose. Missing MuJoCo or an unloadable model degrades to {} (t_ff = 0) —
    gravity off is safe, a crash is not.
    """

    def __init__(self, mjcf_path, dir_by_joint, tmax_by_joint, *,
                 hands: HandsConfig = None, mount_tilt_deg=None, ff_scale=None,
                 ff_clamp=FF_CLAMP, ff_ramp_s=FF_RAMP_S, pose_pad=FF_POSE_PAD, log=None):
        self._log = log or (lambda *a, **k: None)
        self._dir = dict(dir_by_joint or {})
        self._tmax = dict(tmax_by_joint or {})
        self._hands = hands if hands is not None else HandsConfig.from_env()
        self._mount_tilt = (float(mount_tilt_deg) if mount_tilt_deg is not None
                            else float(os.getenv("MABEL_ARMS_MOUNT_TILT_DEG",
                                                 os.getenv("MABEL_HW_ARMS_MOUNT_TILT_DEG",
                                                           "0.0"))))
        self._ff_scale = (float(ff_scale) if ff_scale is not None
                          else float(os.getenv("MABEL_ARMS_FF_SCALE",
                                               os.getenv("MABEL_HW_ARMS_FF_SCALE",
                                                         str(FF_SCALE_DEFAULT)))))
        self.ff_clamp, self.ff_ramp_s, self.pose_pad = ff_clamp, ff_ramp_s, pose_pad
        self._mj = self._m = None
        self.dyn: MujocoDynamics | None = None
        self._qadr: dict = {}
        self._dofadr: dict = {}
        self._qlim: dict = {}
        self._qadr_x: dict = {}          # optional posture joints (torso)
        self._dofadr_x: dict = {}
        self._qlim_x: dict = {}
        self._hand_mass0: dict = {}      # body id -> original mass (for restore)
        self._pose_bad = 0
        self._eval_err = 0
        try:
            import mujoco
            self._mj = mujoco
            self._m = mujoco.MjModel.from_xml_path(str(mjcf_path))
            self.dyn = MujocoDynamics(self._m, mujoco.MjData(self._m))
            self._index_joints(mujoco)
            self._record_hand_masses(mujoco)
            self._fix_arm_mounts(mujoco, mjcf_path)
            self.set_hands(self._hands)                        # applies mass + logs
            self._log(f"arm gravity ready — {len(self._qadr)} joints, "
                      f"hands={self._hands.value}, mount_tilt={self._mount_tilt:+.1f}°, "
                      f"ff_scale={self._ff_scale:g}")
        except Exception as e:                                 # noqa: BLE001
            self._log(f"arm gravity unavailable ({e}); t_ff = 0 (pure PD — arms droop).")
            self._m = self.dyn = None

    # ── build ────────────────────────────────────────────────────────────────
    def _index_joints(self, mujoco) -> None:
        for fw, mj in _MJCF_JOINT.items():
            i = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, mj)
            if i >= 0:
                self._qadr[fw] = int(self._m.jnt_qposadr[i])
                self._dofadr[fw] = int(self._m.jnt_dofadr[i])
                self._qlim[fw] = (float(self._m.jnt_range[i][0]),
                                  float(self._m.jnt_range[i][1]))
        # Optional posture joints: indexed apart so _pose_ok never REQUIRES them.
        self._qadr_x, self._dofadr_x, self._qlim_x = {}, {}, {}
        for fw, mj in _EXTRA_MJCF_JOINT.items():
            i = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_JOINT, mj)
            if i >= 0:
                self._qadr_x[fw] = int(self._m.jnt_qposadr[i])
                self._dofadr_x[fw] = int(self._m.jnt_dofadr[i])
                self._qlim_x[fw] = (float(self._m.jnt_range[i][0]),
                                    float(self._m.jnt_range[i][1]))

    def _hand_subtree(self, mujoco) -> set:
        """Body ids of both hands: the *_tower roots and everything below them."""
        roots = [i for i in range(self._m.nbody)
                 if (mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                 in _HAND_ROOTS]
        drop, changed = set(roots), True
        while changed:
            changed = False
            for b in range(self._m.nbody):
                if self._m.body_parentid[b] in drop and b not in drop:
                    drop.add(b)
                    changed = True
        return drop

    def _record_hand_masses(self, mujoco) -> None:
        for b in self._hand_subtree(mujoco):
            self._hand_mass0[b] = float(self._m.body_mass[b])

    def set_hands(self, hands: HandsConfig) -> None:
        """Switch the gravity model between HANDS-OFF and HANDS-ON at runtime.

        NONE zeros the hand subtree's mass (the FF then lifts only the arm); ORCA
        restores the original MJCF masses. Uses MujocoDynamics.set_link_mass so the
        subtree/inertia bookkeeping stays consistent. Kinematics are untouched, so the
        arm links stay put — only what the FF lifts changes."""
        if self.dyn is None:
            return
        self._hands = hands
        total = 0.0
        for b, m0 in self._hand_mass0.items():
            want = m0 if hands is HandsConfig.ORCA else 0.0
            self._m.body_mass[b] = want
            self._m.body_inertia[b] = (self._m.body_inertia[b] if hands is HandsConfig.ORCA
                                       else 0.0)
            total += m0
        self.dyn._refresh()                                    # mj_setConst
        if hands is HandsConfig.NONE:
            self._log(f"arm gravity: HANDS OFF — {total:.3f} kg of hand mass stripped "
                      "(feed-forward lifts the ARM only; set hands=ORCA when refitted).")
        else:
            self._log(f"arm gravity: HANDS ON — {total:.3f} kg ORCA mass INCLUDED.")

    def _fix_arm_mounts(self, mujoco, mjcf_path) -> None:
        """Tilt the shoulder mounts to where they physically are (the MJCF draws −10°,
        the metal measures 0°). Read from the XML so a corrected MJCF makes this a no-op;
        it can never double-apply. Right-multiply by Rx(Δ) (mounts compose Rz·Ry·Rx)."""
        import xml.etree.ElementTree as ET

        import numpy as np
        try:
            root = ET.parse(str(mjcf_path)).getroot()
        except Exception as e:                                 # noqa: BLE001
            self._log(f"arm gravity: MJCF re-read for mount tilt failed ({e}); "
                      "using the drawn mount angle.")
            return
        want = math.radians(self._mount_tilt)
        for body in root.iter("body"):
            if body.get("name") not in ("l_arm_mount", "r_arm_mount"):
                continue
            eul = (body.get("euler") or "").split()
            if not eul:
                continue
            try:
                rx_model = float(eul[0])
            except ValueError:
                continue
            delta = want - rx_model
            if abs(delta) < 1e-6:
                continue
            bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))
            if bid < 0:
                continue
            dq, out = np.zeros(4), np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, np.array([1.0, 0.0, 0.0]), float(delta))
            mujoco.mju_mulQuat(out, np.array(self._m.body_quat[bid], dtype=float), dq)
            self._m.body_quat[bid] = out
            self._log(f"arm gravity: {body.get('name')} re-tilted "
                      f"{math.degrees(rx_model):+.1f}° → {self._mount_tilt:+.1f}° "
                      "(matches the metal).")

    # ── query ────────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.dyn is not None

    @property
    def hands(self) -> HandsConfig:
        return self._hands

    def _pose_ok(self, model_angles) -> bool:
        for fw, (lo, hi) in self._qlim.items():
            v = model_angles.get(fw)
            if v is None or not math.isfinite(v):
                return False
            if v < lo - self.pose_pad or v > hi + self.pose_pad:
                return False
        return True

    def _extra_ok(self, fw: str, v) -> bool:
        """An optional posture joint is used only when its value is sane; a bad
        one is DROPPED (evaluated at qpos0, no τ emitted) rather than killing the
        whole pose — a stale torso calibration must not zero the arms' FF."""
        if v is None or not math.isfinite(v):
            return False
        lo, hi = self._qlim_x[fw]
        return (lo - self.pose_pad) <= v <= (hi + self.pose_pad)

    def tff(self, model_angles: dict, ramp: float = 1.0,
            frame: str = "raw") -> dict:
        """{fw_id: t_ff [N·m]} for the measured MODEL-frame pose.

        ``ramp`` in [0,1] fades the feed-forward in on engage (caller times it).

        ``frame`` picks the OUTPUT convention — they differ by the calibration sign
        (τ_raw = dir·τ_model), so getting it wrong drives the joint DOWN at full
        feed-forward torque:

          "raw"   (default) the DaMiao motor frame the firmware speaks. For a caller
                  that talks to the motor directly. Needs `dir_by_joint`.
          "model" the MJCF/URDF frame. **This is what /arms/command.effort must
                  carry**: mabel_hw.subsystems.joint_group.command() dir-corrects
                  effort into the raw frame itself, so publishing "raw" there would
                  DOUBLE-apply dir and push every dir=-1 joint the wrong way into its
                  stop. A "model" caller needs no dir map at all — which is how the
                  gateway stays free of any calibration import.

        The ff_clamp is applied in the requested frame; |τ| is identical either way
        (dir = ±1), so the clamp is unaffected by the choice."""
        if frame not in ("raw", "model"):
            raise ValueError(f"frame must be 'raw' or 'model', got {frame!r}")
        if self.dyn is None or ramp <= 0.0 or not model_angles:
            return {}
        if not self._pose_ok(model_angles):
            self._pose_bad += 1
            if self._pose_bad in (1, 400):
                self._log("⚠ arm pose outside the MJCF limits — gravity t_ff = 0 "
                          "(a joint's calibration is stale; gravity would be computed "
                          "for a pose the arm is not in)")
            return {}
        self._pose_bad = 0
        try:
            mj, m, d = self._mj, self._m, self.dyn.data
            mj.mj_resetData(m, d)                  # valid free-joint quat; never hand-zero qpos
            for fw, adr in self._qadr.items():
                v = model_angles.get(fw)
                if v is not None and math.isfinite(v):
                    d.qpos[adr] = float(v)
            # Optional posture joints (torso): set when the caller supplied a sane
            # value; emit τ only for the ones actually set this cycle.
            extra_live = []
            for fw, adr in self._qadr_x.items():
                if fw in model_angles and self._extra_ok(fw, model_angles[fw]):
                    d.qpos[adr] = float(model_angles[fw])
                    extra_live.append(fw)
            d.qvel[:] = 0.0                        # so bias = pure gravity g(q)
            mj.mj_forward(m, d)                    # REQUIRED: compute kinematics for this pose
            g = self.dyn.bias()                    # qfrc_bias at q̇=0 = g(q) (shared engine)
            out = {}
            dofs = dict(self._dofadr)
            dofs.update({fw: self._dofadr_x[fw] for fw in extra_live})
            for fw, dof in dofs.items():
                tau_model = (float(g[dof]) * self._ff_scale * float(ramp)
                             * FF_SCALE_BY_JOINT.get(fw, 1.0))
                if not math.isfinite(tau_model):
                    continue
                tau = (float(self._dir.get(fw, 1)) * tau_model   # -> raw motor frame
                       if frame == "raw" else tau_model)             # -> model frame
                cap = self.ff_clamp * float(self._tmax.get(fw, 0.0) or 0.0)
                if cap > 0:
                    tau = max(-cap, min(cap, tau))
                out[fw] = tau
            self._eval_err = 0
            return out
        except Exception as e:                     # noqa: BLE001
            self._eval_err += 1
            if self._eval_err in (1, 50):
                self._log(f"arm gravity eval failed ({e}) — t_ff = 0 this cycle")
            return {}

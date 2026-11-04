"""
A learned whole-body controller as a lab variant: NOVA's RL policy (NOVA/controller/wbc_rl).

The policy replaces retargeter + motion model + whole-body controller: Vision Pro head and
wrists (+ its own joint state, odometry and twist) -> base twist, lift rate, torso, neck and 14
arm targets, at 50 Hz. It is run here through NOVA's OWN deploy path, imported, not copied:
`deploy.Adapter` (retarget map, observation, action filter, NOVA's slews, lift ramp) +
`deploy.Policy` on the exported numpy weights (`checkpoints/v<N>_policy.npz`, every exported version), exactly as
NOVA's `sim2sim.py` runs it on a Mac. Only the PLANT is the lab's: the commands are realised
by the same servo (ideal or the robot's slews) and the same swerve-module base as every other
variant, so the comparison is the map, not the physics.

Conventions, measured (not assumed):
  * NOVA's training model is MABEL's kinematics with a lift that starts 317.5 mm higher: with
    each lift stage +0.15875 m (half a stroke) the head and both hand mounts agree to 0.00 mm
    over random poses. NOVA stage q -> lab stage q + LIFT_OFF; the lab model's lift range is
    widened by that much for this variant (geometry is a translation; the viewer is too).
  * the URDF base faces -x like the MJCF chassis; the adapter's odometry yaw is the REP
    heading (forward = +x), i.e. the chassis yaw + pi, and its twist is NOVA's body frame
    (+x forward) -> the chassis frame is (-vx, -vy, wz).
  * scope "local" (upper body) = the policy's own upper-body mode bit (base and lift held).
Fingers are not in the policy (NOVA uses its hand optimiser): the lab's finger solver runs as
for every other variant. The TeleopEngine still runs each frame as the MEASURING INSTRUMENT —
its operator-side estimators (shoulders, body frame, anchor) define the metrics for every
variant — and its robot commands are discarded.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from .worker import KinematicRetargeter

NOVA_ROOT = Path(os.environ.get("MABEL_RTL_NOVA", str(Path(__file__).resolve().parents[4] / "NOVA")))
WBC_RL = NOVA_ROOT / "controller" / "wbc_rl"
LIFT_OFF = 0.15875                       # m per stage: NOVA's lift zero sits half a lab stroke up
YAW_REP = math.pi                        # adapter heading = chassis yaw + pi


def _import_deploy():
    for p in (str(WBC_RL), str(NOVA_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import deploy as D                   # noqa: E402  NOVA/controller/wbc_rl/deploy.py
    import robot as R                    # noqa: E402
    return D, R


class RLRetargeter(KinematicRetargeter):
    def __init__(self, vid, **kw):
        super().__init__(vid, **kw)
        from .variants import VARIANTS
        D, R = _import_deploy()
        self.D, self.R = D, R
        spec = VARIANTS[vid]
        self.policy_path = WBC_RL / "checkpoints" / spec["rl"]
        self.wrist_ik = bool(spec.get("rl_wrist_ik", False))   # v28+: deploy's exact wrist solver; earlier: the policy's wrist
        if not getattr(self, "_rl_policy", None):
            self._rl_policy = D.Policy(str(self.policy_path))
        self.ad = D.Adapter(self._rl_policy, upper=(self.settings.get("scope") == "local"))
        m, mj = self.model, self.mj
        self._lift_jid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n) for n in ("Lift Lower", "Lift Upper")]
        for j in self._lift_jid:
            m.jnt_range[j][1] = max(float(m.jnt_range[j][1]), R.LIFT_STROKE + LIFT_OFF)
        self._servo = [(a, qa, lo, (max(hi, R.LIFT_STROKE + LIFT_OFF) if qa in self._lift_q else hi), lim)
                       for a, qa, lo, hi, lim in self._servo]
        self._qadr = {n: self.qnames[n] for n in R.ACTUATED}
        self._act = {}                                   # wire joint name -> position actuator
        for a, qa, *_ in self._servo:
            for n, q in self._qadr.items():
                if q == qa:
                    self._act[n] = a
        self._op = {}                                    # last received operator frames (held when lost)
        self._seen = {"head": None, "l": None, "r": None}
        self._t_pol = None
        self._q_prev = None
        self._rl_track = {}
        self._rl_orient = {}
        self._t_engage = None
        self._rl_cmd = None

    def apply_settings(self, settings):
        st = super().apply_settings(settings)
        if getattr(self, "ad", None) is not None:
            self.ad.upper = float(st.get("scope") == "local")
        return st

    # ── NOVA-convention robot state from the lab model ─────────────────────────
    def _q20(self):
        d = self.data
        q = np.array([float(d.qpos[self._qadr[n]]) for n in self.R.ACTUATED])
        q[:2] -= LIFT_OFF
        return q

    def _odom(self):
        b = self.base_pose() or {}
        return (float(b.get("x", 0.0)), float(b.get("y", 0.0)), float(b.get("yaw", 0.0)) + YAW_REP)

    def _vb(self):
        vx, vy, w = getattr(self, "_base_real", (0.0, 0.0, 0.0))
        return np.array([-vx, -vy, w])                   # chassis frame -> NOVA body frame

    # ── operator frames (deploy's own conversions) ──────────────────────────────
    def _update_op(self, pose, now):
        D = self.D
        if pose is None:
            return
        if pose.head is not None:
            hp, hR = D.head_from_avp(np.asarray(pose.head, float))
            self._op["head_p"], self._op["head_R"] = hp, hR
            self._seen["head"] = now
        for side, k in (("left", "l"), ("right", "r")):
            joints = getattr(pose, f"{side}_joints", None)
            if not joints:
                continue
            hand = {"anchorTransform": {"matrix": np.eye(4).ravel().tolist()},
                    "joints": [{"joint": n, "localTransform": {"matrix": np.asarray(M, float).ravel().tolist()},
                                "isTracked": True} for n, M in joints.items()]}
            r = D.palm_from_avp(hand)
            if r is not None:
                self._op[k + "w_p"], self._op[k + "w_R"] = r[0], r[1]
                self._seen[k] = now

    def _update_op_wire(self, payload, now):
        """NOVA's deploy contract on the RAW headset frame: the head only while trackingState is
        normal, a hand only while isTracked (deploy.palm_from_avp -> hand_retarget.wire_positions,
        which since NOVA 03c43fd keeps ARKit's estimated joints whatever their own isTracked says)
        — otherwise the part is held and its age grows, exactly what the policy was trained to see."""
        D = self.D
        hd = payload.get("head") or {}
        M = (hd.get("transform") or {}).get("matrix") if isinstance(hd.get("transform"), dict) else hd.get("transform")
        if M is not None and hd.get("trackingState", "normal") == "normal":
            hp, hR = D.head_from_avp(np.asarray(M, float).reshape(4, 4))
            self._op["head_p"], self._op["head_R"] = hp, hR
            self._seen["head"] = now
        for key, k in (("leftHand", "l"), ("rightHand", "r")):
            h = payload.get(key)
            if not h or not h.get("isTracked"):
                continue
            r = D.palm_from_avp(h)
            if r is not None:
                self._op[k + "w_p"], self._op[k + "w_R"] = r[0], r[1]
                self._seen[k] = now

    def _ready(self):
        return all(k in self._op for k in ("head_p", "lw_p", "rw_p"))

    def _tick(self, q, qd, age):
        """One 50 Hz policy tick = deploy.Adapter.step, split as NOVA's sim2sim.py runs it so the wrist
        solver is per policy (pre-v28 policies: `--no-wrist-ik`). No ORCA wrist hinge in the lab model:
        its measurement is its own command (hq=None in deploy)."""
        ad, od = self.ad, self._odom()
        ad.hq, ad.yaw = ad.hw_f.copy(), od[2]
        t = ad.targets(self._op); ad.blend_t += self.D.DT
        ad.euro(np.stack([t["lw_R"], t["rw_R"]]))
        obs = ad.observe(ad.target_obs(t, od), q, qd, self._vb(), age, t)
        a = ad.pi(obs)
        return ad.act(a, ad.solve_wrist(q, od) if self.wrist_ik else None)

    # ── one frame ────────────────────────────────────────────────────────────────
    def step(self, pose, now, collide=True, frame=None):
        import time
        eng, m, d = self.eng, self.model, self.data
        self._tgt = {}
        if self.clock is not None:
            self.clock.now = float(now)
        t0 = time.perf_counter()
        payload = eng._payload(pose)
        if not self._calibrated and pose.head is not None:
            eng.calibrate(pose)
            self._calibrated = True
        dt = (now - self._prev_t) if self._prev_t is not None else 1.0 / 60.0
        eng._frame_dt = float(np.clip(dt, 1e-3, 0.1))
        # the instrument: the engine's operator-side estimators (its robot output is discarded)
        keep = d.ctrl.copy()
        eng._retarget(payload, None, now)
        fingers = {a for side in ("left", "right") for a in eng.finger_act[side]}
        for a in range(m.nu):
            if a not in fingers:
                d.ctrl[a] = keep[a]
        if frame is not None and frame.get("kind") == "wire":
            self._update_op_wire(frame["p"], now)           # the deploy contract (head/hand tracking flags)
        else:
            self._update_op(pose, now)                      # EgoDex payloads
        if self._ready():
            D = self.D
            if not self.ad.engaged:
                self.ad.engage(self._op, self._q20(), self._odom())
                self._t_pol = now
                self._t_engage = now
                self._q_prev = self._q20()
            # the policy at its own 50 Hz on the episode clock (0, 1 or more ticks per frame)
            while self._t_pol + D.DT <= now + 1e-9:
                self._t_pol += D.DT
                q = self._q20()
                qd = (q - self._q_prev) / D.DT if self._q_prev is not None else np.zeros_like(q)
                self._q_prev = q
                age = np.array([max(0.0, self._t_pol - (self._seen[k] if self._seen[k] is not None else self._t_pol))
                                for k in ("head", "l", "r")])
                cmd = self._tick(q, qd, age)
                self._rl_cmd = cmd
                for n, v in cmd["joints"].items():
                    a = self._act.get(n)
                    if a is not None:
                        d.ctrl[a] = float(v)
                if self._lift_act >= 0:
                    d.ctrl[self._lift_act] = 2.0 * (float(cmd["lift"]) + LIFT_OFF)
                tw = cmd["twist"]
                self._base_cmd_rep = (float(tw[0]), float(tw[1]), float(tw[2]))
                # own-target error, in the policy's own convention (its target vs its FK frame)
                tg = self.ad.targets(self._op, step=False)
                cp, _ = self.ad.tracked(q)
                x, y, yw = self._odom()
                Rb = D.rot_z(yw)
                _, cR = self.ad.tracked(q)
                for j, (k, side) in enumerate((("lw", "left"), ("rw", "right")), start=1):
                    cur = Rb @ cp[j] + np.array([x, y, self.R.CHASSIS_Z])
                    self._rl_track[side] = float(np.linalg.norm(tg[k + "_p"] - cur) * 1e3)
                    self._tgt[side] = np.asarray(tg[k + "_p"], float)
                    Rc = Rb @ cR[j]                          # the policy's OWN palm frame vs its target
                    c = (np.trace(Rc.T @ tg[k + "_R"]) - 1.0) / 2.0
                    self._rl_orient[side] = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
        if self.base_on and self._free_q is not None and getattr(self, "_base_cmd_rep", None):
            vx, vy, w = self._base_cmd_rep
            self._apply_base((-vx, -vy, w), eng._frame_dt)
        solve_ms = (time.perf_counter() - t0) * 1e3
        self._servo_ideal(eng._frame_dt)
        if collide:
            self.mj.mj_fwdPosition(m, d)
        else:
            self.mj.mj_kinematics(m, d)
        rec = self._record(pose, payload, now, solve_ms, collide)
        for side, s in (("left", "l"), ("right", "r")):
            if side in self._rl_track:
                rec["m"][f"track_mm_{s}"] = self._rl_track[side]
        # the policy's 1 s engage glide is its ease-in: excluded like the engine's pre-anchor
        # frames (and as NOVA's sim2sim excludes the first 1.5 s)
        settled = self._t_engage is not None and now - self._t_engage >= 1.5
        rec["anchored"] = bool(rec.get("anchored")) and settled
        rec["rl"] = {"engaged": bool(self.ad.engaged), "upper": bool(self.ad.upper),
                     "own_orient_deg": dict(self._rl_orient), "own_track_mm": dict(self._rl_track),
                     "lift_stage": float(self._rl_cmd["lift"]) if self._rl_cmd else None,
                     "policy": self.policy_path.name}
        self._prev_t = now
        self.n += 1
        return rec

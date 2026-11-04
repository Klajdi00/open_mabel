"""target_filter.py — CBF projection for ABSOLUTE joint targets (the manipulate path).

WHY THIS EXISTS. Every wrist-tracking client plans in ITS OWN head and ships joint
targets: the web control center solves wrist IK in the browser (viewport3d._solveArm →
joint_command), and the iOS / Vision Pro apps solve it on-device (method "wrist" →
joint_command). None of those targets ever met the self-collision guard — it lived only
in the server-side WholeBodyIK (the retarget path) and the WBC torque soft-wall — so a
wrist target inside the torso was simply executed. This module is the ONE choke point:
the facade filters every manipulate-method joint target through it, so web, iOS and
visionOS all inherit the SAME avoidance law without any app change.

THE LAW (control barrier function, velocity-projection form). The filter maintains a
PLAN configuration q_plan that glides from the current pose toward the raw target at a
bounded rate. Each tick the step is projected through
:meth:`SelfCollisionGuard.project`:

    q̇' = argmin ‖q̇' − q̇‖   s.t.   ḋ_k ≥ −α (d_k − d_s)  for every active pair k

— the minimal correction removes ONLY each barrier's normal (gap-closing) excess and
keeps the TANGENTIAL component, so a target inside the body makes the hand slide along
the torso/head/other arm and settle at the nearest reachable point on the boundary,
instead of stopping dead or penetrating. The integrated q_plan IS the collision-avoided
path — :meth:`plan_path` exposes it whole for previews and tests.

The capsules are evaluated at the PLAN pose (a scratch MjData), never the lagging
physical pose, for the same servo-lag reason WholeBodyIK evaluates its barrier at the
target pose: the committed target must stay outside the margin so the arm settling on
it does too.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ... import config as C
from .capsule_guard import SelfCollisionGuard

__all__ = ["JointTargetProjector"]

# The guarded actuator set: everything the manipulate methods command that moves the
# capsule model. The Lift is excluded — it translates torso + arms together, so the
# self-collision geometry is (to first order) invariant along it.
_GUARDED = tuple(n for n in C.IK_ACTUATORS if n != "Lift")

#: rad/s — how fast the filter may ease a pose OUT of a violated margin when the
#: operator commands nothing (the anti-oscillation cap on unsolicited motion).
_ESCAPE_RATE = 0.05


class JointTargetProjector:
    """Receding-horizon CBF filter for absolute joint targets, actuator-name keyed."""

    #: max joint step (rad, ∞-norm) per projected sub-step — bounds the CBF
    #: linearization error (see step()).
    SUBSTEP_RAD = 0.02

    def __init__(self, model, data, guard: SelfCollisionGuard | None = None,
                 vmax: float | None = None):
        self.model = model
        self.data = data
        self.guard = guard if guard is not None else SelfCollisionGuard(model, data)
        #: rad/s plan glide rate — how fast the plan flows toward the raw target.
        #: Matches the studio's SAFE_BLEND_VEL-class budgets; per-tick the step is
        #: vmax·dt, so the projection linearization stays valid.
        self.vmax = float(C.SC_TARGET_VMAX if vmax is None else vmax)
        self.names = _GUARDED
        self._qadr = {n: int(model.joint(n).qposadr[0]) for n in self.names}
        self._dof = {n: int(model.joint(n).dofadr[0]) for n in self.names}
        self._scratch = mujoco.MjData(model)
        self._plan: dict = {}          # {actuator: rad} — the committed avoided path head
        #: telemetry: tightest capsule gap seen at the last filtered step
        self.min_dist = np.inf

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def reset(self):
        """Drop the plan — next step() reseeds from the measured pose. Call on
        mode entry / resume so a stale plan can never command a jump."""
        self._plan.clear()

    def _measured(self, name):
        return float(self.data.qpos[self._qadr[name]])

    # ── the per-tick filter ───────────────────────────────────────────────────
    def step(self, targets: dict, dt: float) -> dict:
        """Advance the plan one tick toward ``targets`` ({actuator: rad}) and
        return the filtered absolute targets to command. Names outside the
        guarded set pass through untouched; guarded names not present drop out
        of the plan (they reseed from measured when they reappear)."""
        guarded = {n: float(v) for n, v in targets.items() if n in self._qadr}
        for n in list(self._plan):
            if n not in guarded:
                del self._plan[n]
        if not guarded:
            return dict(targets)
        dt = min(max(float(dt), 1e-3), 0.1)
        for n in guarded:
            self._plan.setdefault(n, self._measured(n))

        names = list(guarded)
        q_plan = np.array([self._plan[n] for n in names])
        q_tgt = np.array([guarded[n] for n in names])
        step = np.clip(q_tgt - q_plan, -self.vmax * dt, self.vmax * dt)

        # SUB-STEP the projection: the CBF rows are a LINEARIZATION of the gap
        # at the plan pose, and a 0.06 rad joint step moves a wrist several cm —
        # enough second-order error to overshoot a margin by millimetres
        # (measured on the adversarial-sweep test). Capping each projected
        # sub-step at SUBSTEP_RAD keeps the linearization honest regardless of
        # the caller's tick size.
        s = self._scratch
        dofs = [self._dof[n] for n in names]
        n_sub = max(1, int(np.ceil(np.max(np.abs(step)) / self.SUBSTEP_RAD)))
        sub_dt = dt / n_sub
        self.min_dist = np.inf
        try:
            # Rows at the MEASURED pose (once per tick — the metal barely moves
            # within one): the physical corner-cut guard. Sliding tangentially
            # around a curved obstacle at the PLAN pose is a closing motion at
            # the lagging measured pose, so plan-only rows let servo lag eat the
            # margin (see WholeBodyIK._project_both).
            self.guard.data = self.data
            Gp, hp = self.guard.rows(dofs, braking=True,
                                     margin_pad=C.SC_LAG_PAD)
            for _ in range(n_sub):
                sub = np.clip(q_tgt - q_plan,
                              -self.vmax * sub_dt, self.vmax * sub_dt)
                if not np.any(np.abs(sub) > 1e-9):
                    break
                s.qpos[:] = self.data.qpos
                for n, v in zip(names, q_plan):
                    s.qpos[self._qadr[n]] = v
                for n in self._plan:               # other tracked joints too
                    if n not in guarded:
                        s.qpos[self._qadr[n]] = self._plan[n]
                mujoco.mj_kinematics(self.model, s)
                mujoco.mj_comPos(self.model, s)
                self.guard.data = s
                Gt, ht = self.guard.rows(dofs, braking=True)
                self.min_dist = min(self.min_dist, self.guard.min_dist)
                if len(Gt) and len(Gp):
                    rows = (np.vstack([Gt, Gp]), np.hstack([ht, hp]))
                elif len(Gt):
                    rows = (Gt, ht)
                else:
                    rows = (Gp, hp)
                dq = self.guard.project(sub / sub_dt, dofs, rows=rows) * sub_dt
                # NEVER INJECT MOTION THE OPERATOR DID NOT COMMAND: with zero
                # input, an inside-margin row's h<0 makes project(0) return a
                # nonzero "escape" — at a rest pose sitting on a (noisy)
                # boundary that is a per-tick oscillator (measured live: the
                # right arm shaking from torque-on). Cap the correction at the
                # commanded step plus a slow escape budget, so genuine
                # penetration still eases out without ever shaking the arm.
                n_in = float(np.linalg.norm(sub))
                n_out = float(np.linalg.norm(dq))
                cap = n_in + _ESCAPE_RATE * sub_dt
                if n_out > cap and n_out > 1e-12:
                    dq = dq * (cap / n_out)
                q_plan = q_plan + dq
        finally:
            self.guard.data = self.data

        out = dict(targets)
        for n, v in zip(names, q_plan):
            self._plan[n] = float(v)
            out[n] = float(v)
        return out

    # ── whole-path preview (tests / offline planners) ─────────────────────────
    def plan_path(self, targets: dict, dt: float = 0.05, max_steps: int = 400,
                  settle_eps: float = 1e-4):
        """Iterate the flow from the CURRENT pose to convergence and return the
        waypoint list [{actuator: rad}, ...] — the collision-avoided path a
        client would see previewing this target set. Leaves the live plan state
        untouched."""
        saved = dict(self._plan)
        self.reset()
        path, prev = [], None
        try:
            for _ in range(max_steps):
                cur = {n: v for n, v in self.step(targets, dt).items()
                       if n in self._qadr}
                path.append(cur)
                if prev is not None and all(
                        abs(cur[n] - prev[n]) < settle_eps for n in cur):
                    break
                prev = cur
        finally:
            self._plan = saved
        return path

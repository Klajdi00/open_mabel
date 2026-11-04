"""Safety layer: tracking-loss fail-safe (EngagementSupervisor) + clutch."""
import numpy as np
import mujoco

from .hyperparams import *
from .model_utils import position_actuators


class EngagementSupervisor:
    """Teleop fail-safe / tracking-loss state machine (fully modular — swap the
    safety policy without touching the retargeting). It watches the FRESHNESS of the
    operator signals (the VR stream itself, the head, the hands, the elbows) and
    eases the upper-body POSITION actuators + the lift toward a destination:

        engaged     -> the LIVE retargeted pose (tracked at full speed once there)
        disengaged  -> a safe HOME pose (rest arms/hands/neck/torso, lift lowered)

    STEP-FREE BY CONSTRUCTION (the real-robot requirement): every transition eases
    from an ANCHOR — the commanded ctrl captured at the moment the destination
    changed — so the command is always continuous:

        ctrl = (1-s)*anchor + s*dest,    s: 0 -> 1 over SAFE_ENGAGE/RETREAT_TIME

    The old law blended HOME <-> live by a global alpha, i.e. at alpha=0 it WROTE
    ctrl := HOME outright. In sim (robot starts at home) that was invisible; on the
    real robot — where the pose at engage-time is wherever the limp arms were left —
    it was a full-authority position step to HOME (a violent lurch), and every
    re-engage detoured THROUGH home-weighted blends. Anchoring removes both: boot /
    torque-enable eases gently from the measured pose, engage eases from wherever
    the body is straight to the operator's pose (which the re-anchored clutch has
    already made jump-free), loss eases from wherever it is to HOME.

    On LOSS — headset removed (the stream stalls), or a hand / arm drops out — the
    destination flips to HOME over SAFE_RETREAT_TIME and the base is frozen. On
    RE-ACQUIRE it flips to the live pose over SAFE_ENGAGE_TIME. `rebase()` re-captures
    the anchor from the CURRENT ctrl — hold_current/resume_hold call it so a torque
    enable adopts the measured pose instead of resuming a stale ease.

    States (for display):  ENGAGED · HOMING (easing to home) · HOME (safe) ·
    RE-ENGAGING (easing to live).  Tunables live in the SAFE_* constants."""

    def __init__(self, model, lift_act):
        pa = position_actuators(model)
        self.aids = np.array([a for a, _ in pa])
        self.home = np.array([q for _, q in pa], float)     # rest pose (qpos0)
        # SAFE-PARK OVERRIDE (see SAFE_HOME_* in hyperparams): the torso does NOT
        # rest at qpos0 (upright) but at its +15 deg lean-BACK stop, so a power cut
        # settles the chest into the mechanical stop instead of flopping it forward.
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Torso")
        if tid >= 0:
            sel = np.flatnonzero(self.aids == tid)
            if sel.size:
                lo, hi = model.actuator_ctrlrange[tid]
                self.home[sel[0]] = float(
                    np.clip(np.radians(SAFE_HOME_TORSO_DEG), lo, hi))
        self.lift_act = lift_act
        self.home_lift = float(np.clip(SAFE_HOME_LIFT, *LIFT_RANGE))
        self.reset()

    def reset(self):
        self.live = self.home.copy()
        self.live_lift = self.home_lift
        # Start a fresh ease toward HOME from wherever the commanded pose currently
        # is (s=0 + lazy anchor) — NOT at s=1, which would write home outright on
        # the first apply() (the boot/reset analogue of the engage lurch).
        self._s = 0.0                # segment progress toward the current destination
        self._anchor = None          # ctrl snapshot the ease departs from (lazy capture)
        self._anchor_lift = None
        self._out = None             # the ctrl apply() last COMMANDED (edge anchor)
        self._out_lift = None
        self.t_stream = self.t_head = self.t_hand = self.t_arm = -1e9
        self._was_engaged = False
        self.reengaged_edge = False
        # TRANSPARENT UNTIL FIRST ENGAGEMENT (2026-07-27): retreat-to-HOME is a
        # fail-safe for a LOST operator — it has no business running when no operator
        # has EVER engaged. Before this flag, boot/reset counted as "disengaged" and
        # apply() slowly eased the whole body (and the LIFT) from wherever it was to
        # HOME — observed on the real robot as "the lift moves by itself on real_run
        # startup" (13 mm/s creep to SAFE_HOME_LIFT with nobody connected). Until the
        # first genuine engagement the supervisor writes NOTHING: the robot holds
        # whatever the bring-up hold / user commands set, and only ever moves on a
        # new command.
        self._ever_engaged = False

    def rebase(self):
        """Re-anchor the ease at the CURRENT ctrl and restart the segment. Call after
        any external ctrl adoption (hold_current / resume_hold: ctrl := measured) so
        the next apply() eases from the measured pose instead of a stale anchor."""
        self._anchor = None
        self._anchor_lift = None
        self._out = None             # anchor from the freshly-held data.ctrl, not
        self._out_lift = None        # the pre-hold command
        self._s = 0.0

    def restart_ease(self):
        """Blend the NEXT destination in from the last COMMANDED pose (anchor :=
        last output, s := 0). Call when the retarget solution is about to change
        discontinuously for a legitimate reason — e.g. the world-ref re-anchor
        after an ARKit origin jump, where the origin-jump guard keeps engagement
        alive (no edge) yet the first post-anchor solution can sit far from the
        held pose. The new solution then eases in over SAFE_ENGAGE_TIME instead
        of stepping."""
        self._anchor = None if self._out is None else self._out.copy()
        self._anchor_lift = self._out_lift
        self._s = 0.0

    def _signals_fresh(self, now):
        hands = (now - self.t_hand) < SAFE_SIGNAL_TIMEOUT
        return ((now - self.t_stream) < SAFE_STREAM_TIMEOUT and
                (now - self.t_head) < SAFE_SIGNAL_TIMEOUT and hands and
                (now - self.t_arm) < SAFE_SIGNAL_TIMEOUT)

    def observe(self, payload, data, now):
        """On a fresh VR frame (AFTER retargeting wrote ctrl): refresh per-signal
        freshness and, only if fully tracked, snapshot the live target pose."""
        self.t_stream = now
        if payload.get("head_mat"):
            self.t_head = now
        r, l = payload.get("right_hand_mats"), payload.get("left_hand_mats")
        hands_ok = (r is not None and l is not None) if SAFE_REQUIRE_BOTH_HANDS else (r or l)
        if hands_ok:
            self.t_hand = now
        # NOTE: elbows are 4x4 ndarrays when present (webcam clients) — plain
        # truthiness raises on arrays, so test for presence explicitly.
        if payload.get("right_elbow") is not None or payload.get("left_elbow") is not None:
            self.t_arm = now
        if self._signals_fresh(now):                        # only trust a complete frame
            self.live = data.ctrl[self.aids].copy()
            self.live_lift = float(data.ctrl[self.lift_act])

    def step(self, now, wall_dt):
        """Advance the ease toward the current destination at the slow safe rate.
        Returns the engaged bool; sets reengaged_edge on the disengaged->engaged
        transition. A destination flip restarts the segment (the anchor is
        re-captured from the commanded ctrl on the next apply())."""
        eng = self._signals_fresh(now)
        if eng:
            self._ever_engaged = True
        elif not self._ever_engaged:
            # No operator has EVER engaged: stay parked (transparent apply, s=0) —
            # never start a retreat toward HOME from boot/reset.
            self.reengaged_edge = False
            self._was_engaged = False
            self._s = 0.0
            return False
        self.reengaged_edge = eng and not self._was_engaged
        if eng != self._was_engaged:
            # Ease FROM what apply() last COMMANDED — not from data.ctrl, which the
            # retargeter has already overwritten with the new live pose by the time
            # apply() runs (update() order: retarget -> step -> apply). Capturing
            # post-retarget ctrl re-created the exact engage snap this class exists
            # to prevent (caught by exp_real_retarget_chain: 0.40 rad neck step on
            # re-acquire). _out is None only before the first apply (boot): the
            # lazy data.ctrl capture there is correct (nothing has retargeted yet).
            self.restart_ease()
        self._was_engaged = eng
        rate = wall_dt / (SAFE_ENGAGE_TIME if eng else SAFE_RETREAT_TIME)
        self._s = float(np.clip(self._s + rate, 0.0, 1.0))
        return eng

    def apply(self, data):
        """Ease the upper body + lift from the anchor to the destination. Fully
        engaged (s=1, destination = live) this writes the live pose exactly — no
        slew — so tracking speed is never capped."""
        if not self._ever_engaged:
            return          # transparent until the first engagement: hold, never home
        if self._anchor is None:                             # lazy capture: the ctrl
            self._anchor = data.ctrl[self.aids].copy()       # the ease departs from
            self._anchor_lift = float(data.ctrl[self.lift_act])
        s = self._s
        dest = self.live if self._was_engaged else self.home
        dest_lift = self.live_lift if self._was_engaged else self.home_lift
        out = (1.0 - s) * self._anchor + s * dest
        out_lift = (1.0 - s) * self._anchor_lift + s * dest_lift
        data.ctrl[self.aids] = out
        data.ctrl[self.lift_act] = out_lift
        self._out = out                                      # edge anchor (see step)
        self._out_lift = float(out_lift)

    @property
    def alpha(self):
        """Back-compat engagement scalar: 1 = at the live pose, 0 = at home."""
        return self._s if self._was_engaged else 1.0 - self._s

    def state(self):
        if not self._ever_engaged:
            return "standby (hold)"     # no operator yet: supervisor is transparent
        if self._was_engaged:
            return "engaged" if self._s >= 0.999 else "re-engaging"
        return "HOME (safe)" if self._s >= 0.999 else "HOMING"


class ChannelPark:
    """Ease ONE switched-off tracking channel to its neutral pose.

    Switching a channel off cannot simply stop writing it: position actuators
    HOLD, so the limb freezes wherever the operator left it. A frozen arm reads
    as a crash and a frozen neck leaves the cameras pointed at the floor. So a
    disabled channel eases, over `PARK_TIME`, from wherever it was to a neutral
    pose it is safe and legible to sit in.

    Deliberately NOT the EngagementSupervisor: that one is a whole-body SAFETY
    retreat driven by loss of tracking, and it owns every position actuator at
    once. This is per-channel and operator-commanded, and the two must be able
    to coexist (the supervisor runs after, and wins, exactly as before).

    RE-ENABLING EASES TOO (2026-10-16). The live target a re-enabled channel
    resumes can be far from where it was held — a neck frozen forward while the
    operator looks 25° aside — and simply stopping the override handed that
    whole difference to the servo in ONE tick (a command STEP; the supervisor
    does not smooth it, contrary to what this note used to claim). So the output
    blends from the last held command to the live one over `release` seconds
    (smoothstep), the mirror image of the park. Measured by
    controller/experiments/exp_ablation_toggles.py (bumpless sweep).
    """

    def __init__(self, aids, neutral, seconds=None, release=None):
        self.aids = np.asarray(list(aids), int)
        self.neutral = np.asarray(list(neutral), float)
        self.seconds = float(seconds if seconds is not None else PARK_TIME)
        self.release = float(release if release is not None else PARK_RELEASE_TIME)
        self._held = None          # last command written while parked
        self._last_out = None      # last command THIS park wrote (park or release)
        self._r = 1.0              # release progress (1 = fully live)
        self.reset()

    def reset(self):
        self._s = 0.0
        self._anchor = None

    @property
    def parked(self):
        """True once the channel has fully reached neutral."""
        return self._anchor is not None and self._s >= 0.999

    def update(self, data, enabled, dt):
        """Call EVERY frame, AFTER the live command has been written.
        Disabled → ease to neutral and hold. Enabled → transparent, except for
        the release blend right after a park."""
        if self.aids.size == 0:
            return
        if enabled:
            if self._anchor is not None:             # just re-enabled: start the release
                self.reset()
                self._r = 0.0
            if self._held is not None and self._r < 1.0:
                self._r = float(np.clip(self._r + dt / max(self.release, 1e-3), 0.0, 1.0))
                w = self._r * self._r * (3.0 - 2.0 * self._r)      # smoothstep
                live = data.ctrl[self.aids].copy()
                out = (1.0 - w) * self._held + w * live
                data.ctrl[self.aids] = out
                self._last_out = out.copy()
                if self._r >= 1.0:
                    self._held = None
                    self._last_out = None
            return
        if self._anchor is None:
            # Park from wherever the OUTPUT is — a release in progress included
            # (anchoring on the old held value instead jumped the command).
            self._anchor = (self._last_out.copy() if self._last_out is not None
                            else data.ctrl[self.aids].copy())
            self._r = 1.0
        self._s = float(np.clip(self._s + dt / max(self.seconds, 1e-3), 0.0, 1.0))
        s = self._s
        out = (1.0 - s) * self._anchor + s * self.neutral
        data.ctrl[self.aids] = out
        self._held = out.copy()
        self._last_out = out.copy()


class ClutchController:
    """DOUBLE-PINCH CLUTCH (re-indexing). Pinching EITHER hand twice in quick
    succession (a double-click) toggles a whole-body FREEZE: the robot holds its pose
    and the base stops while the operator recenters; another double-pinch (either hand)
    RELEASES, and the caller re-anchors the world transform so the robot eases back from
    its held pose to the operator's CURRENT pose (no jump). A single pinch is left free
    for grasping. update() returns the TOGGLE edge; .clutched gates the retargeting.

    Detection: track EACH hand's pinch state with hysteresis; a rising edge is one
    'click'. Two clicks on the SAME hand within CLUTCH_DOUBLE_WINDOW toggle the clutch,
    then a global cooldown blocks re-triggering."""

    def __init__(self):
        self.clutched = False
        self._pinched = {"right": False, "left": False}   # per-hand state (hysteretic)
        self._click_t = {"right": -1e9, "left": -1e9}     # first click of a pending double
        self._lock_until = -1e9

    def _is_pinched(self, side, payload):
        p = payload.get(f"{side}_pinch")
        if p is None:
            return None              # no pinch signal for this hand
        if self._pinched[side]:      # currently pinched: release only past the OFF gap
            return not (p > CLUTCH_PINCH_OFF)
        return p < CLUTCH_PINCH_ON   # cross the ON threshold

    def update(self, payload, now):
        """Detect a single-hand double-pinch on either hand (hysteresis + cooldown).
        Returns True on the frame the clutch TOGGLES, so the caller can re-anchor."""
        if not CLUTCH_ENABLE:
            return False
        toggled = False
        for side in ("right", "left"):
            ip = self._is_pinched(side, payload)
            if ip is None:
                continue
            rising = ip and not self._pinched[side]       # this hand JUST pinched (a click)
            self._pinched[side] = ip
            if not rising or now < self._lock_until:
                continue
            if now - self._click_t[side] <= CLUTCH_DOUBLE_WINDOW:   # 2nd click -> toggle
                self.clutched = not self.clutched
                self._lock_until = now + CLUTCH_COOLDOWN
                self._click_t[side] = -1e9
                toggled = True
            else:
                self._click_t[side] = now                 # first click; await the second
        return toggled

    def reset(self):
        self.clutched = False
        self._pinched = {"right": False, "left": False}
        self._click_t = {"right": -1e9, "left": -1e9}
        self._lock_until = -1e9
        self._lock_until = -1e9

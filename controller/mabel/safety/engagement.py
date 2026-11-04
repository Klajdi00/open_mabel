"""Teleop safety layer: tracking-loss fail-safe + double-pinch clutch.

Two small, self-contained policies the production stack lacked. Both are pure of any
frontend: they consume already-parsed operator signals and act on the robot ctrl, so
they work identically in the desktop sim and on hardware.

  EngagementSupervisor -- eases the upper body to a safe HOME pose when tracking is
                          lost (headset off, hand/arm dropout) and eases it back to
                          the operator's CURRENT pose on re-acquire (no snap).
  ClutchController      -- a single-hand double-pinch toggles a whole-body freeze so
                          the operator can re-index (recenter) without moving the robot.
"""
import numpy as np
import mujoco

from .. import config as C

_SWERVE_ACTS = ("FL Steer", "FR Steer", "B Steer", "FL Drive", "FR Drive", "B Drive")


def position_actuators(model):
    """Position-servo actuator ids whose transmission is a single joint, paired with
    that joint's rest target (qpos0). Excludes the swerve steer/drive wheels. Used as
    the safe HOME pose for the fail-safe blend."""
    out = []
    for aid in range(model.nu):
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        if name in _SWERVE_ACTS:
            continue
        jid = model.actuator_trnid[aid, 0]
        out.append((aid, float(model.qpos0[model.jnt_qposadr[jid]])))
    return out


class EngagementSupervisor:
    """Tracking-loss fail-safe state machine. Watches the FRESHNESS of the operator
    signals (the stream, the head, the hands, the elbows) and drives one rate-limited
    blend scalar alpha in [0, 1]:

        alpha = 1  -> follow the LIVE retargeted pose
        alpha = 0  -> hold a safe HOME pose (rest arms/hands/neck/torso, lift lowered)

    On LOSS (headset removed -> stream stalls, or a hand/arm drops out) alpha ramps DOWN
    over SAFE_RETREAT_TIME so the whole upper body eases to HOME and the base is frozen.
    On RE-ACQUIRE, alpha ramps UP over SAFE_ENGAGE_TIME with the live target = the
    operator's CURRENT pose, so the robot eases back in rather than snapping. The blend
    is applied to the upper-body POSITION actuators + the lift every frame:
        ctrl = (1-alpha)*home + alpha*live

    States (for display): engaged (alpha=1) / HOMING (falling) / HOME (alpha=0) /
    re-engaging (rising).  Tunables live in config.SAFE_*.
    """

    def __init__(self, model, lift_act):
        pa = position_actuators(model)
        self.aids = np.array([a for a, _ in pa])
        self.home = np.array([q for _, q in pa], float)     # rest pose (qpos0)
        # SAFE-PARK OVERRIDE (see SAFE_HOME_* in mabel.config): the torso does NOT
        # rest at qpos0 (upright) but at its +15 deg lean-BACK stop, so a power cut
        # settles the chest into the mechanical stop instead of flopping it forward.
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Torso")
        if tid >= 0:
            sel = np.flatnonzero(self.aids == tid)
            if sel.size:
                lo, hi = model.actuator_ctrlrange[tid]
                self.home[sel[0]] = float(
                    np.clip(np.radians(C.SAFE_HOME_TORSO_DEG), lo, hi))
        self.lift_act = lift_act
        self.home_lift = float(np.clip(C.SAFE_HOME_LIFT, *C.LIFT_RANGE))
        self.reset()

    def reset(self):
        self.live = self.home.copy()
        self.live_lift = self.home_lift
        self.alpha = 0.0                                     # start AT home; ease out to first pose
        self.t_stream = self.t_head = self.t_hand = self.t_arm = -1e9
        self._arm_seen = False                               # have we EVER seen an elbow signal?
        self._was_engaged = False
        self.reengaged_edge = False
        self._ever_engaged = False   # transparent (hold, never home) until first engage
        self.require_hands = True                             # nav/drive mode clears this (see below)

    def _signals_fresh(self, now):
        # Hands are required in manipulation/retarget modes, but NOT while driving:
        # NAVIGATION streams head + joystick only (no hands), so requiring hands would
        # disengage the fail-safe and pin the upper body + LIFT at HOME (lift dead, base
        # still drives since swerve isn't supervisor-managed). The facade clears
        # require_hands in navigation mode; head + stream freshness still gate engagement
        # so a real link/headset drop still fail-safes.
        hands = (not self.require_hands) or ((now - self.t_hand) < C.SAFE_SIGNAL_TIMEOUT)
        # The elbow is OPTIONAL: only the ZMQ desktop AVP stream carries elbows; the
        # app wire path (iOS / Vision Pro / web) sends head + hands only. Gate on the
        # elbow's freshness ONLY once one has actually appeared, so a head+hands stream
        # can fully engage (otherwise the fail-safe would pin the robot at HOME forever).
        arm = (not self._arm_seen) or ((now - self.t_arm) < C.SAFE_SIGNAL_TIMEOUT)
        return ((now - self.t_stream) < C.SAFE_STREAM_TIMEOUT and
                (now - self.t_head) < C.SAFE_SIGNAL_TIMEOUT and hands and arm)

    def observe(self, signals, data, now):
        """On a fresh frame (AFTER retargeting wrote ctrl): refresh per-signal freshness
        and, only if fully tracked, snapshot the live target pose. `signals` is a dict
        with optional keys head_mat, right_hand_mats, left_hand_mats, right_elbow,
        left_elbow (truthy = present)."""
        self.t_stream = now
        if signals.get("head_mat") is not None:
            self.t_head = now
        r, l = signals.get("right_hand_mats"), signals.get("left_hand_mats")
        hands_ok = (r is not None and l is not None) if C.SAFE_REQUIRE_BOTH_HANDS else (r is not None or l is not None)
        if hands_ok:
            self.t_hand = now
        if signals.get("right_elbow") is not None or signals.get("left_elbow") is not None:
            self.t_arm = now
            self._arm_seen = True
        if self._signals_fresh(now):                        # only trust a complete frame
            self.live = data.ctrl[self.aids].copy()
            self.live_lift = float(data.ctrl[self.lift_act])

    def step(self, now, wall_dt):
        """Advance alpha toward 1 (engaged) or 0 (home) at the slow safe rate. Returns
        the engaged bool; sets reengaged_edge on the disengaged->engaged transition."""
        eng = self._signals_fresh(now)
        if eng:
            self._ever_engaged = True
        elif not getattr(self, "_ever_engaged", False):
            # TRANSPARENT UNTIL FIRST ENGAGEMENT (2026-07-27): retreat-to-HOME is a
            # fail-safe for a LOST operator. With no operator EVER engaged, apply()
            # must write nothing — the boot ease to HOME was "the lift moves by
            # itself on real_run startup" on the real robot.
            self.reengaged_edge = False
            self._was_engaged = False
            return False
        self.reengaged_edge = eng and not self._was_engaged
        self._was_engaged = eng
        rate = (wall_dt / C.SAFE_ENGAGE_TIME) if eng else -(wall_dt / C.SAFE_RETREAT_TIME)
        self.alpha = float(np.clip(self.alpha + rate, 0.0, 1.0))
        return eng

    def apply(self, data):
        """Blend the upper body + lift between HOME and the live target by alpha."""
        if not getattr(self, "_ever_engaged", False):
            return          # transparent until the first engagement: hold, never home
        a = self.alpha
        data.ctrl[self.aids] = (1.0 - a) * self.home + a * self.live
        data.ctrl[self.lift_act] = (1.0 - a) * self.home_lift + a * self.live_lift

    def state(self):
        if not getattr(self, "_ever_engaged", False):
            return "standby (hold)"     # no operator yet: supervisor is transparent
        if self.alpha >= 0.999:
            return "engaged"
        if self.alpha <= 0.001:
            return "HOME (safe)"
        return "re-engaging" if self._was_engaged else "HOMING"


class ClutchController:
    """Single-hand DOUBLE-PINCH clutch (re-indexing). Pinching EITHER hand twice in
    quick succession (a double-click) toggles a whole-body FREEZE: the robot holds its
    pose and the base stops while the operator recenters; another double-pinch (either
    hand) RELEASES, and the caller re-anchors the world transform so the robot eases
    back from its held pose to the operator's CURRENT pose (no jump). A single pinch is
    left free for grasping. update() returns the TOGGLE edge; .clutched gates retargeting.

    Detection: track EACH hand's pinch state with hysteresis; a rising edge is one
    'click'. Two clicks on the SAME hand within CLUTCH_DOUBLE_WINDOW toggle the clutch,
    then a global cooldown blocks re-triggering. Tunables live in config.CLUTCH_*.
    """

    def __init__(self):
        self.reset()

    def _is_pinched(self, side, pinch):
        """pinch = dict {'right': gap_m or None, 'left': gap_m or None}."""
        p = pinch.get(side)
        if p is None:
            return None              # no pinch signal for this hand
        if self._pinched[side]:      # currently pinched: release only past the OFF gap
            return not (p > C.CLUTCH_PINCH_OFF)
        return p < C.CLUTCH_PINCH_ON   # cross the ON threshold

    def update(self, pinch, now):
        """Detect a single-hand double-pinch on either hand (hysteresis + cooldown).
        `pinch` = {'right': gap_m or None, 'left': gap_m or None}. Returns True on the
        frame the clutch TOGGLES, so the caller can re-anchor."""
        if not C.CLUTCH_ENABLE:
            return False
        toggled = False
        for side in ("right", "left"):
            ip = self._is_pinched(side, pinch)
            if ip is None:
                continue
            rising = ip and not self._pinched[side]       # this hand JUST pinched (a click)
            self._pinched[side] = ip
            if not rising or now < self._lock_until:
                continue
            if now - self._click_t[side] <= C.CLUTCH_DOUBLE_WINDOW:   # 2nd click -> toggle
                self.clutched = not self.clutched
                self._lock_until = now + C.CLUTCH_COOLDOWN
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

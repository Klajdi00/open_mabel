"""schema.py — per-joint calibration records and the raw<->model anchor math.

Two anchoring families exist in the MABEL tree and BOTH must be preserved
verbatim (they are written by different wizards and consumed by different
mechanisms):

CENTER-anchored (arm + hand slices; keys: raw_min, raw_max, dir, model_mid,
reliability). The wizard sweeps both end-stops; the model position that
corresponds to the middle of the measured raw range is ``model_mid``:

    model = model_mid + dir * (raw - raw_mid),   raw_mid = (raw_min + raw_max)/2
    raw   = raw_mid   + dir * (model - model_mid)

``reliability`` (measured-range / model-range, ~1.0 when the sweep found the
full mechanical range) is carried as a quality metric — the mapping itself does
NOT scale by it, matching the upstream bridges.

NEUTRAL-anchored (neck motors; keys: raw_min, raw_max, dir, raw_neutral).
The RSSR neck is calibrated by capturing a LEVEL-HEAD pose, not end-stops, so
zero model position is anchored at that neutral capture:

    model = dir * (raw - raw_neutral)
    raw   = raw_neutral + dir * model

``to_raw`` clamps into [raw_min, raw_max] in BOTH families — the measured raw
end-stops are the hard safety envelope regardless of anchoring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..errors import CalibrationError

__all__ = ["JointCalib"]

CENTER = "center"
NEUTRAL = "neutral"
_TAU = 2.0 * math.pi          # one full motor turn (see JointCalib.unwrap)


@dataclass(frozen=True)
class JointCalib:
    raw_min: float
    raw_max: float
    dir: int                          # +1 / -1
    anchor: str                       # CENTER | NEUTRAL
    model_mid: float = 0.0            # center-anchored only
    raw_neutral: float = 0.0          # neutral-anchored only
    reliability: float = 1.0          # quality metric, not applied in the map

    @classmethod
    def from_dict(cls, d: Mapping, joint: str = "?") -> "JointCalib":
        """Build from one mabel.json entry; the anchor family is inferred from
        which key the wizard wrote (raw_neutral => neutral)."""
        try:
            raw_min, raw_max = float(d["raw_min"]), float(d["raw_max"])
            direction = int(d.get("dir", 1))
        except (KeyError, TypeError, ValueError) as e:
            raise CalibrationError(f"joint {joint!r}: bad calibration entry "
                                   f"{dict(d)!r}") from e
        if raw_max < raw_min:
            raise CalibrationError(f"joint {joint!r}: raw_max < raw_min "
                                   f"({raw_max} < {raw_min})")
        if direction not in (-1, 1):
            raise CalibrationError(f"joint {joint!r}: dir must be ±1, got {direction}")
        if "raw_neutral" in d:
            return cls(raw_min=raw_min, raw_max=raw_max, dir=direction,
                       anchor=NEUTRAL, raw_neutral=float(d["raw_neutral"]),
                       reliability=float(d.get("reliability", 1.0)))
        return cls(raw_min=raw_min, raw_max=raw_max, dir=direction,
                   anchor=CENTER, model_mid=float(d.get("model_mid", 0.0)),
                   reliability=float(d.get("reliability", 1.0)))

    # ── the map ───────────────────────────────────────────────────────────────
    @property
    def raw_mid(self) -> float:
        return 0.5 * (self.raw_min + self.raw_max)

    def model_from(self, x: float) -> float:
        """model from an ALREADY-continuous/unwrapped raw ``x`` — applies the anchor map
        with NO fold. For multi-turn DaMiao joints the caller rebuilds a continuous raw by
        delta-unwrapping the ±p_max-modular feedback (the encoder wraps at ±p_max ≈ 4 turns,
        so to_model()'s 2π fold is the wrong modulus); this maps that continuous value."""
        if self.anchor == NEUTRAL:
            return self.dir * (x - self.raw_neutral)
        return self.model_mid + self.dir * (x - self.raw_mid)

    def unwrap(self, raw: float) -> float:
        """Fold a MULTI-TURN raw reading back into the calibrated window.

        DaMiao motors report a multi-turn angle whose turn counter restarts on a
        power cycle, so a joint can come up a whole revolution away from where the
        wizard swept it. Measured 2026-07-13: l_j5 (left_arm_5) read +2.646 against
        a swept window of [-5.34, -2.19] — exactly one +2*pi turn out — and mapped
        to a nonsense 6.41 rad (367 deg) joint angle, which is what threw the left
        arm's pose in RViz.

        Shifting by whole turns is UNAMBIGUOUS only while the measured window is
        narrower than a full turn; for a wider window (or a value already inside)
        the reading is returned untouched. A shift is accepted only if it actually
        lands inside the window, so a genuinely out-of-envelope reading (a real
        fault) still surfaces instead of being silently "corrected"."""
        span = self.raw_max - self.raw_min
        if span >= _TAU or self.raw_min <= raw <= self.raw_max:
            return raw
        k = round((self.raw_mid - raw) / _TAU)
        if k == 0:
            return raw                      # just outside the stops, not a wrap
        cand = raw + k * _TAU
        margin = 0.1                        # sweeps rarely touch the exact stop
        if self.raw_min - margin <= cand <= self.raw_max + margin:
            return cand
        return raw

    def to_model(self, raw: float) -> float:
        if self.anchor == NEUTRAL:
            return self.dir * (self.unwrap(raw) - self.raw_neutral)
        return self.model_mid + self.dir * (self.unwrap(raw) - self.raw_mid)

    def raw_of(self, model: float) -> float:
        """model -> raw, WITHOUT the end-stop clamp. Callers that know the joint's
        CURRENT raw angle should use this + `clamp_raw`, so a joint sitting outside
        its (possibly stale) sweep window is never ordered to jump to the edge."""
        if self.anchor == NEUTRAL:
            return self.raw_neutral + self.dir * model
        return self.raw_mid + self.dir * (model - self.model_mid)

    def clamp_raw(self, raw: float, cur: float | None = None,
                  margin: float = 0.0) -> float:
        """Clamp a raw target to the calibrated end-stops — but NEVER order a move a
        joint isn't already making, and keep a safety MARGIN off the hard stops.

        ⚠ The window can be STALE. `raw_min/raw_max` come from a sweep; the motor
        ZEROS come from a later hang-pose capture. Re-zero after sweeping and the two
        disagree: l_j4's window is [+9.712, +12.186] and l_j5's is [-5.337, -2.186],
        neither of which even CONTAINS raw 0 (2026-07-14). A joint then sits legally
        outside its own window, and a plain clamp turns "hold where you are" into
        "slam to the window edge" — measured as an 87° snap on l_j5 the instant torque
        engaged, and the J6 snap the operator reported.

        So widen the window to include where the joint ACTUALLY is. The guard still
        holds in the direction that matters: you can never be driven FURTHER outside
        (the expanded bound is exactly `cur`), and you can always come back in.

        ⚠ MULTI-TURN BOOT (2026-07-15) — the widening must NOT confuse "resting past the
        window" with "a whole turn off the window". DaMiao's turn counter latches the
        absolute single-turn angle on power-up, so a joint whose sweep window drifted a
        full turn away (l_j4's was stranded at +11 rad while the motor boots near −1.5)
        comes up with `cur` a whole `2π` outside its window — NOT because it is past a
        stop, but because it is in another revolution. The naive `min(lo,cur)` then only
        widens ONE side, leaving a one-sided range that clamps every target back to `cur`:
        the joint reads correctly (unwrap) yet cannot be commanded anywhere — it pins and
        is dragged to its end-stop. So fold `cur` (and the value being clamped) into the
        window's revolution before widening, undone on the way out so the motor still gets
        a same-revolution command. Whole-turn folding is unambiguous only for a single-turn
        window (span < 2π; matches `unwrap`); a genuine sub-turn excursion past a stop
        gives k=0 and the widening behaves EXACTLY as before.

        `margin` (rad) pulls the *commandable* window in from each hard stop, so a target
        can never drive the motor into a mechanical end-stop and STALL there (the left
        elbow l_j4 rests only ~6° from its flex stop → any upward target hits it and the
        motor cooks). The margin is dropped on the side the joint is ALREADY past (via the
        cur-widening), so a joint resting inside the margin is held, never shoved inward."""
        lo, hi = self.raw_min, self.raw_max
        if margin > 0.0 and (hi - lo) > 2.0 * margin:
            lo, hi = lo + margin, hi - margin        # keep clear of the hard stops
        if cur is not None and math.isfinite(cur):
            # Fold a pure multi-turn offset out of cur (single-turn windows only), so only a
            # GENUINE past-the-window rest position widens the range — not a boot revolution.
            turns = (_TAU * round((self.raw_mid - cur) / _TAU)
                     if (self.raw_max - self.raw_min) < _TAU else 0.0)
            cur_w, raw_w = cur + turns, raw + turns
            lo, hi = min(lo, cur_w), max(hi, cur_w)  # ...but never push a joint that's past it
            return max(lo, min(hi, raw_w)) - turns   # command back in the motor's revolution
        return max(lo, min(hi, raw))

    def to_raw(self, model: float, cur: float | None = None,
               margin: float = 0.0) -> float:
        """model -> raw MOTOR angle, in the SAME REVOLUTION the motor is actually in.

        ⚠ THE 2π BUG (2026-07-14) — this is why an arm joint stalled and cooked itself.
        The DaMiao encoders are MULTI-TURN: a joint's raw angle may sit a full turn away
        from the window the sweep recorded. `to_model` handles that for READING — it
        calls `unwrap`, folding raw into the window — which is why the pose renders
        correctly in RViz/iOS. But the naive inverse emits the command in the WINDOW's
        revolution, not the MOTOR's:

            l_j5 measured raw = +0.884   (the motor's true angle)
                 to_model     -> -1.632  (folded into the window — correct, renders right)
                 to_raw       -> -5.333  (= raw_min: the SAME pose, one turn down)
                 the motor reads that as "rotate -356.2°"

        It cannot: it is against its mechanical stop. So it stalls at full PD torque
        (-10 N·m detuned, -22 N·m at true gains) and does not move a millimetre — a
        perfect heater. l_j5's ROTOR hit 80 °C in 13 s, and that is what latched the
        DaMiao drivers red (a latch only a power cycle clears).

        So: express the command in the revolution the MOTOR is actually in. Same
        physical pose, but where the motor lives — "stay put" instead of "spin a full
        turn". With `cur` unknown we cannot disambiguate the turn, so we fall back to
        the old behaviour (single-turn joints are unaffected either way).

        ⚠ FOLD BY THE WINDOW, NOT BY THE TARGET (2026-07-16). The shift must be the
        WINDOW's whole-turn offset from `cur`, not the TARGET's. Folding the target
        toward `cur` looks equivalent — and is, for any target within half a turn —
        but it silently inverts for a target further out than pi: `raw_of(+4.0)` on a
        [-1,+1] window folds to -2.28, which then clamps to raw_min and drives the
        joint to the NEGATIVE stop. An out-of-range target (a client bug, or the
        gateway's MuJoCo mirror flooding a clamped setpoint) must saturate at the
        NEAR stop; wrapping it to the FAR one parks the motor against a mechanical
        stop at full PD torque, which is exactly how a joint cooks. Folding by the
        window keeps the target monotonic in `model`, so the clamp always lands on
        the correct side. Ambiguous for a window wider than a turn (as in `unwrap`),
        so that case does not fold."""
        raw = self.raw_of(model)
        if cur is not None and math.isfinite(cur):
            if (self.raw_max - self.raw_min) < _TAU:
                raw -= _TAU * round((self.raw_mid - cur) / _TAU)
        return self.clamp_raw(raw, cur, margin)

    @property
    def model_min(self) -> float:
        return min(self.to_model(self.raw_min), self.to_model(self.raw_max))

    @property
    def model_max(self) -> float:
        return max(self.to_model(self.raw_min), self.to_model(self.raw_max))

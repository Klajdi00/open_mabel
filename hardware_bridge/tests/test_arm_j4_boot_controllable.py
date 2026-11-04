"""J4 (elbow) must be CONTROLLABLE from a fresh power-up — regression for the
'left J4 uncontrollable / slams to the extension stop on real_run.sh' bug.

Root cause: the DaMiao encoders are multi-turn and the counter latches the
ABSOLUTE single-turn angle (-pi, pi] on power-up, but the l_j4 calibration window
had drifted 2 turns up to ~+11 rad (turns accumulated during the calibration
SWEEP, not where the motor boots). `to_model` still reads correctly (it folds via
`unwrap`, so RViz/iOS looked fine), but on the COMMAND path `clamp_raw`'s
cur-widening — which assumes `cur` lives in the window's revolution — produced a
one-sided range and pinned the joint ~134 deg off any target. A pinned position
command drives the motor into its end-stop = the reported slam + overheat.

The right elbow r_j4 was calibrated with a single-turn window near zero
([+0.10, +2.63]) that already CONTAINS the boot reading, so it always worked. The
fix mirrors that: l_j4's window is relocated by -2 turns into the single-turn band,
keeping `dir` and `model_mid` so `to_model` and the model range are byte-identical
(RViz / iOS / MJCF ctrlrange unchanged).

These tests model the exact calibration math against the AS-DEPLOYED mabel.json,
so they fail if the window ever drifts back out of the boot band. The math under
test is mabel_hw.calibration.JointCalib, applied by
mabel_hw.subsystems.joint_group — the ONE place raw<->model happens.
"""
import json
import math
import os

import pytest

from mabel_hw.calibration import JointCalib

from conftest import BRIDGE_DIR

TAU = 2.0 * math.pi


def _calib():
    path = os.path.join(BRIDGE_DIR, "calibration_data", "mabel.json")
    with open(path) as f:
        return json.load(f)["arm"]["openarm-dual"]["calib"]


def _jc(name):
    return JointCalib.from_dict(_calib()[name], name)


def _single_turn(x):
    """What a DaMiao reports at power-up: the absolute angle folded into (-pi, pi]."""
    return x - TAU * round(x / TAU)


def _boot_controllable(jc, boot_turns=0):
    """Emulate JointGroupSubsystem's read + write paths on a fresh boot and return
    the worst error, in degrees, between the target pose and the pose actually
    commandable to the motor.

    `boot_turns` lets the encoder come up a whole turn off the true single-turn value
    (0 = the normal absolute-encoder power-up)."""
    mlo, mhi = jc.model_min, jc.model_max
    worst_cmd = worst_read = 0.0
    for k in range(41):
        m_phys = mlo + (mhi - mlo) * k / 40.0
        cur = _single_turn(jc.raw_of(m_phys)) + boot_turns * TAU   # motor's boot reading
        # READ path (subsystem.state -> _to_model): must recover the true pose
        # (this is what RViz / the apps show).
        worst_read = max(worst_read, abs(jc.to_model(cur) - m_phys))
        # WRITE path (subsystem.set_targets -> _to_raw(name, model, cur)): can we
        # actually reach a target? `cur` is what makes the multi-turn fold work.
        for m_tgt in (mlo + 0.15, 0.5 * (mlo + mhi), mhi - 0.15):
            want = _single_turn(jc.raw_of(m_tgt)) + boot_turns * TAU
            cmd = jc.to_raw(m_tgt, cur)
            err = cmd - want
            err -= TAU * round(err / TAU)          # compare within one revolution
            worst_cmd = max(worst_cmd, abs(err))
    return math.degrees(worst_cmd), math.degrees(worst_read)


# ── the mapping is preserved: RViz / iOS / MJCF limits do NOT move ──────────────

def test_l_j4_model_range_unchanged():
    """The relocation must keep to_model byte-identical: model range [-0.943,+1.542]
    (matches MJCF <joint 'Left Arm 4'> range and the position ctrlrange)."""
    jc = _jc("l_j4")
    assert jc.model_min == pytest.approx(-0.943, abs=2e-3)
    assert jc.model_max == pytest.approx(1.542, abs=2e-3)
    # a few interior points render exactly where they did with the +11 window
    assert jc.to_model(_single_turn(jc.raw_of(0.0))) == pytest.approx(0.0, abs=1e-9)
    assert jc.to_model(_single_turn(jc.raw_of(1.0))) == pytest.approx(1.0, abs=1e-9)


def test_l_j4_window_lives_in_the_motors_frame():
    """The window must CONTAIN the joint's live motor reading (~+10.5 rad), not be relocated a
    turn away from it.

    This supersedes the old "window must live in the (-pi, pi] boot band" rule. That relocation
    was a workaround for a stateless 2π fold; the studio bridge now unwraps the DaMiao's ±p_max
    feedback into a continuous signal, which needs the window in the MOTOR's own frame (the
    seed folds toward the window centre in ±2·p_max steps, not 2π). A window a turn away from the
    motor would read as out-of-range under that unwrap. l_j4 sits at ~+10.5, so its window is at
    ~+10..+12 by construction — dir / model_mid / span are preserved, so the MODEL range is
    unchanged (see test_l_j4_model_range_unchanged)."""
    jc = _jc("l_j4")
    motor = 10.475          # live l_j4 raw, measured 2026-07-17 (every other joint boots near 0)
    assert jc.raw_min <= motor <= jc.raw_max, (
        f"l_j4 window [{jc.raw_min:.2f}, {jc.raw_max:.2f}] must contain the motor at {motor}")


# ── controllability: l_j4 must behave like the known-good r_j4 ───────────────────

def test_r_j4_reference_is_controllable():
    """Sanity: the right elbow (the user's known-good reference) tracks its whole
    range from a fresh boot. If this ever fails the boot model itself is wrong."""
    cmd_err, read_err = _boot_controllable(_jc("r_j4"))
    assert read_err < 1e-6
    assert cmd_err < 1.0            # deg


def test_l_j4_controllable_after_fix():
    """The regression: left J4 must be reachable across its range on a fresh boot —
    NOT pinned ~134 deg off target (the pre-fix failure)."""
    cmd_err, read_err = _boot_controllable(_jc("l_j4"))
    assert read_err < 1e-6, "reading drifted — to_model no longer recovers the pose"
    assert cmd_err < 1.0, f"J4 not reachable: worst command error {cmd_err:.1f} deg (pinned)"


def test_l_j4_also_ok_if_encoder_boots_in_old_multiturn_frame():
    """Belt-and-braces: even if the counter comes up in the OLD +11 rad (2-turn)
    frame instead of single-turn, the fixed window + unwrap/to_raw fold still hold
    it — so the fix can never be worse than before, only better."""
    cmd_err, read_err = _boot_controllable(_jc("l_j4"), boot_turns=2)
    assert read_err < 1e-6
    assert cmd_err < 1.0


# ── the end-stop margin: J4 must never be COMMANDED into its mechanical stop ─────

def _ctrlranges():
    """{actuator: (lo, hi)} from the canonical MJCF — what the gateway clamps
    data.ctrl to, i.e. the most extreme model target the bridge can ever receive."""
    import re
    path = os.path.join(os.path.dirname(BRIDGE_DIR), "simulation", "mabel_mujoco",
                        "models", "mabel_full.xml")
    out = {}
    for m in re.finditer(
            r'<position[^>]*name="((?:Left|Right) Arm \d)"[^>]*ctrlrange="([^"]+)"',
            open(path).read()):
        lo, hi = m.group(2).split()
        out[m.group(1)] = (float(lo), float(hi))
    return out


def _act(fw):
    return f"{'Left' if fw[0] == 'l' else 'Right'} Arm {fw[3]}"


MARGIN = 0.05          # config.LEFT_ARM["end_stop_margin"]


def test_l_j4_ctrlrange_equals_its_mechanical_stop():
    """The premise of the bug, pinned so it stays visible: l_j4's MJCF ctrlrange IS
    its measured hard stop (the c42ccfdb hand-sweep wrote the same number to both).
    So the gateway WILL command the exact stop, and the HAL's clamp is the only
    thing that can stop short of it. Not a failure by itself — it is why the margin
    below is load-bearing for this joint and not for its neighbours."""
    jc = _jc("l_j4")
    lo, hi = _ctrlranges()[_act("l_j4")]
    assert lo == pytest.approx(jc.model_min, abs=1e-3)
    assert hi == pytest.approx(jc.model_max, abs=1e-3)


@pytest.mark.parametrize("fw", ["l_j4", "l_j7", "r_j6", "r_j4", "l_j1", "l_j2"])
def test_extreme_target_is_never_commanded_onto_the_stop(fw):
    """REGRESSION (2026-07-16, "J4 always hits the limit and stalls -> overheat +
    over-current"): commanding a joint to the END of its ctrlrange must NOT put the
    motor on its measured mechanical stop.

    The calibrated raw window IS the stop. With no margin, to_raw(ctrlrange_max)
    lands exactly ON raw_min/raw_max, so kp*error keeps pulling into the stop: it
    stalls, cooks, and trips the DaMiao over-current LATCH (physical power cycle to
    clear). The HAL has no stall or thermal guard, so this clamp is the only defence.
    l_j4 is the acute case (ctrlrange == stop, and it rests 1.3 deg from it); l_j7
    and r_j6 have ctrlranges that reach BEYOND their stops."""
    jc = _jc(fw)
    lo, hi = _ctrlranges()[_act(fw)]
    for target in (lo, hi):
        # commanded from a mid-range rest, so `cur` cannot widen the window
        cur = jc.raw_of(0.5 * (jc.model_min + jc.model_max))
        raw = jc.to_raw(target, cur, margin=MARGIN)
        d = min(abs(raw - jc.raw_min), abs(raw - jc.raw_max))
        assert d >= MARGIN - 1e-6, (
            f"{fw}: target {target:+.3f} -> raw {raw:+.4f} sits {d:.4f} rad from a "
            f"mechanical stop (want >= {MARGIN}) — it will stall there")


def test_j4_resting_inside_the_margin_is_held_not_shoved():
    """l_j4 RESTS ~1.3 deg from its stop — inside the margin. The margin must not
    push it inward (that would be motion nobody asked for); clamp_raw drops the
    margin on the side the joint is already past. Commanding the ctrlrange max must
    then command where it already IS: zero error, zero torque."""
    jc = _jc("l_j4")
    rest = 1.520
    rest_raw = jc.raw_of(rest)
    _, hi = _ctrlranges()[_act("l_j4")]
    raw = jc.to_raw(hi, rest_raw, margin=MARGIN)
    assert raw == pytest.approx(rest_raw, abs=1e-6), "the margin shoved a resting joint"

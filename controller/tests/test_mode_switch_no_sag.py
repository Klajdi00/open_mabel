"""The upper body must not sag when the operator changes CONTROL SPACE.

THE BUG (operator, 2026-07-24): "when I switch from drive to joint or task space
the robot arm will sag ... the underlying whole body controller should be the
same no matter the mode I am in".

In the Simulation Control Center those buttons change exactly one wire field
(web_gui/simulation_control_center/web/js/app.js:869):

    Drive -> method="navigation"   Task -> method="joint"   Joint -> method="joint"

Nothing about holding the arms up is supposed to change. But gravity
compensation was implemented TWICE, over two different joint sets:

  * TeleopEngine (navigation / retarget) fed all 22 actuated upper-body DoFs.
  * WBCEngine (joint / wrist) called plant.begin(), which zeroes the WHOLE
    qfrc_applied vector, then re-applied effort only to `self.managed` -- the 14
    arm joints. The ORCA wrists are outside the torque law by design
    (_managed_for), the torso is outside it in the "arm" region, the lift is
    outside it unless freed.

So leaving Drive dropped the feed-forward on 8 DoFs and they settled to their
servos' tau_g/kp error. Measured on the switch Drive -> Task/Joint before the
fix: wrists 1.61 deg, neck 0.28 deg, torso 0.18 deg -- and it RATCHETED, 1.17
deg of wrist drift and 2.4 mm of palm travel over 5 Drive<->Task cycles.

The fix is structural: ONE shared layer (mabel.dynamic_models.gravity.
UpperBodyGravityComp) held by both engines, with the torque engine feeding
exactly the complement of its managed set. The tests below pin the INVARIANT
(same DoFs compensated in every space), not just the symptom, because the
symptom is small in sim and large on hardware.

Full measurement harness + figures: controller/experiments/exp_mode_switch_sag.py
"""
import numpy as np
import mujoco
import pytest

from conftest import _load, needs_model

# The operator's Simulation Control Center defaults: Stiff, arms, slider 0.6.
STIFFNESS = 27.0
DT = 1.0 / 250.0

UPPER_BODY = (["Lift Lower", "Lift Upper", "Torso",
               "Neck Yaw", "Neck Pitch", "Neck Roll"]
              + [f"{S} Arm {i}" for S in ("Left", "Right") for i in range(1, 8)]
              + [f"{S} Wrist" for S in ("Left", "Right")])
#: The lift stages are PRISMATIC — their drift is millimetres, not degrees, and
#: mixing the two units into one "worst joint" number is how a 9 mm lift settle
#: hid behind a 0.5 "deg" reading.
SLIDES = ("Lift Lower", "Lift Upper")
HINGES = [j for j in UPPER_BODY if j not in SLIDES]

# The pre-fix failure is 1.6 deg at the wrists; post-fix is ~0.02 deg. 0.25 deg
# separates them with room for solver noise and is well inside what an operator
# could see (0.25 deg at the wrist is ~0.2 mm of palm travel).
SAG_LIMIT_DEG = 0.25
#: Prismatic limit. The lift is a heavy stage on a strong servo; 2 mm is below
#: what is visible and above the ~0.2 mm it actually settles.
SAG_LIMIT_MM = 2.0

# Every space the operator can select from Drive, as (method, type, region).
SPACES = [
    ("Drive", "navigation", "impedance", "arm"),
    ("Task (Stiff)", "joint", "impedance", "arm"),
    ("Task (Soft)", "joint", "compliance", "arm"),
    ("Wrist IK", "wrist", "impedance", "arm"),
    ("Joint upper_body", "joint", "impedance", "upper_body"),
    ("Joint whole_body (Soft)", "joint", "compliance", "whole_body"),
]


def _rig():
    model, data = _load()
    model.opt.timestep = DT
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    teleop.controller.init_hold()
    teleop.controller.settle_done()
    for _ in range(300):
        mujoco.mj_step(model, data)
    return model, data, teleop


def _run(model, data, teleop, seconds):
    for _ in range(int(seconds / DT)):
        try:
            teleop.update(None, None, DT)
        except Exception:
            pass
        mujoco.mj_step(model, data)


def _space(teleop, method, control_type, region):
    teleop.set_control_spec(method=method, control_type=control_type,
                            region=region, stiffness=STIFFNESS)


def _q(model, data):
    return {j: float(data.qpos[model.jnt_qposadr[model.joint(j).id]])
            for j in UPPER_BODY}


def _worst_deg(a, b):
    """Worst REVOLUTE drift, in degrees."""
    j = max(HINGES, key=lambda k: abs(b[k] - a[k]))
    return j, abs(np.degrees(b[j] - a[j]))


def _worst_mm(a, b):
    """Worst PRISMATIC drift, in millimetres."""
    j = max(SLIDES, key=lambda k: abs(b[k] - a[k]))
    return j, abs(b[j] - a[j]) * 1000.0


def _assert_held(before, after, what):
    j, deg = _worst_deg(before, after)
    assert deg < SAG_LIMIT_DEG, (
        f"{what}: {j} moved {deg:.2f} deg (limit {SAG_LIMIT_DEG}). The operator "
        "commanded nothing; only the reference generator was supposed to change.")
    j, mm = _worst_mm(before, after)
    assert mm < SAG_LIMIT_MM, (
        f"{what}: {j} moved {mm:.2f} mm (limit {SAG_LIMIT_MM})")


#: The two lift stages are rigidly synced by the MJCF "Sync Lift" joint equality
#: (q_upper = q_lower), so they are ONE degree of freedom and one gravity term
#: holds the pair. Both the kinematic path (which feeds both stages) and the
#: torque path's freed-lift float (which drives only "Lift Lower") satisfy that;
#: comparing the raw DoF sets would call them different for no physical reason.
_SYNCED = ("Lift Lower", "Lift Upper")


def _compensated_dofs(model, data):
    """Upper-body DoFs currently carrying an applied effort, with the synced lift
    pair collapsed to one entry."""
    live = set(np.nonzero(np.abs(data.qfrc_applied) > 1e-9)[0].tolist())
    out = {j for j in UPPER_BODY
           if int(model.jnt_dofadr[model.joint(j).id]) in live}
    if out & set(_SYNCED):
        out = (out - set(_SYNCED)) | {"lift"}
    return out


# ── the invariant ────────────────────────────────────────────────────────────
@needs_model
def test_gravity_compensated_dofs_are_the_same_in_every_control_space():
    """THE invariant the operator asked for: the whole-body controller holds the
    robot up identically no matter which space is selected.

    Asserted on the DoF SET rather than on drift, because that is the property
    that actually has to hold -- drift is just how it shows up, and it shows up
    ~60x larger on hardware (kp is finite there) than in this sim."""
    model, data, teleop = _rig()
    seen = {}
    for label, method, ctype, region in SPACES:
        _space(teleop, method, ctype, region)
        _run(model, data, teleop, 1.0)
        seen[label] = _compensated_dofs(model, data)

    full = (set(UPPER_BODY) - set(_SYNCED)) | {"lift"}
    drive = seen["Drive"]
    assert drive == full, (
        f"Drive itself compensates only {len(drive)}/{len(full)} upper-body DoFs; "
        f"missing {sorted(full - drive)}")
    for label, dofs in seen.items():
        assert dofs == drive, (
            f"{label} compensates a DIFFERENT set of joints than Drive.\n"
            f"  lost:   {sorted(drive - dofs)}\n"
            f"  gained: {sorted(dofs - drive)}\n"
            "Gravity compensation must not depend on the control space -- that is "
            "the mode-switch sag. See mabel/dynamic_models/gravity.py.")


@needs_model
def test_both_engines_share_one_gravity_layer():
    """Structural guard: the invariant above holds because there is ONE layer,
    not two that happen to agree today. A future edit that gives either engine
    its own gravity term must fail here, not silently re-open the bug."""
    model, data, teleop = _rig()
    kin = teleop.controller.gravcomp        # navigation / retarget
    tor = teleop.wbc.gravcomp               # joint / wrist
    assert type(kin) is type(tor)
    assert kin.joints == tor.joints, "the two paths cover different joints"
    assert kin.gain == tor.gain, "the two paths use different feed-forward gains"
    assert kin.hz == tor.hz, "the two paths refresh at different rates"


@needs_model
def test_torque_engine_feeds_exactly_what_its_law_does_not_own():
    """The complement rule: managed joints get the torque law's bias term, every
    other upper-body joint gets the shared feed-forward, nothing is counted
    twice and nothing is dropped."""
    model, data, teleop = _rig()
    for _, method, ctype, region in SPACES[1:]:
        _space(teleop, method, ctype, region)
        _run(model, data, teleop, 0.5)
        wbc = teleop.wbc
        if not wbc.active:
            continue
        managed, ff = set(wbc.managed), set(wbc._ff_joints)
        assert not (managed & ff), (
            f"{region}/{ctype}: {sorted(managed & ff)} would be compensated twice "
            "— once by the torque law's bias term and once by the shared layer")
        covered = (managed | ff) & set(UPPER_BODY)
        # A freed lift is held by the float's single b[LIFT_JOINT] term across the
        # synced pair, so neither stage appears in `ff` and that is correct.
        missing = set(UPPER_BODY) - covered - (set(SLIDES) if wbc._lift_freed
                                               else set())
        assert not missing, f"{region}/{ctype}: no gravity term for {sorted(missing)}"
        if wbc._lift_freed:
            assert not (ff & set(SLIDES)), (
                "the freed-lift float already carries the synced pair; a second "
                "term double-compensates it (measured 11 mm of lift rise)")


# ── the symptom ──────────────────────────────────────────────────────────────
@needs_model
@pytest.mark.parametrize("label,method,ctype,region", SPACES[1:])
def test_no_sag_leaving_drive(label, method, ctype, region):
    """Settle in Drive, switch space, hold. The robot must not move."""
    model, data, teleop = _rig()
    _space(teleop, "navigation", "impedance", "arm")
    _run(model, data, teleop, 3.0)
    before = _q(model, data)

    _space(teleop, method, ctype, region)
    _run(model, data, teleop, 4.0)
    _assert_held(before, _q(model, data), f"Drive -> {label}")


@needs_model
def test_no_sag_leaving_drive_with_stale_joint_targets():
    """The realistic case: the operator posed a joint earlier, so the app is
    re-sending an accumulated joint dict every tick when they switch space.

    This is the OTHER half of the same symptom, fixed separately: entering a
    torque mode used to spring the arms back to a minutes-old target dict
    (facade._reseed_targets_from_pose now re-seeds it from the measured pose).
    Both halves have to hold together, so exercise them together -- a gravity
    fix that only works with an empty target dict would not help the operator."""
    model, data, teleop = _rig()
    _space(teleop, "navigation", "impedance", "arm")
    _run(model, data, teleop, 2.0)

    # pose an arm, let it settle, then let the target go stale in Drive
    teleop.set_joint_targets({"right_arm_2": float(
        data.qpos[model.jnt_qposadr[model.joint("Right Arm 2").id]]) - 0.20})
    _run(model, data, teleop, 3.0)
    before = _q(model, data)

    _space(teleop, "joint", "impedance", "arm")
    _run(model, data, teleop, 4.0)
    _assert_held(before, _q(model, data), "Drive -> Task with a stale target")


@needs_model
def test_returning_to_drive_does_not_sag_either():
    """The switch must be reversible -- Task -> Drive is the same non-event."""
    model, data, teleop = _rig()
    _space(teleop, "navigation", "impedance", "arm")
    _run(model, data, teleop, 3.0)
    before = _q(model, data)
    _space(teleop, "joint", "impedance", "arm")
    _run(model, data, teleop, 3.0)
    _space(teleop, "navigation", "impedance", "arm")
    _run(model, data, teleop, 3.0)
    _assert_held(before, _q(model, data), "Drive -> Task -> Drive")


@needs_model
def test_space_switching_does_not_ratchet():
    """The operator-visible failure: flipping space repeatedly walks the arm
    down, because each entry re-seeds its set-point from the ALREADY-SAGGED
    measured pose. Before the fix this reached 1.17 deg of wrist drift and 2.4 mm
    of palm travel over 5 cycles."""
    model, data, teleop = _rig()
    _space(teleop, "navigation", "impedance", "arm")
    _run(model, data, teleop, 3.0)
    before = _q(model, data)
    palm0 = float(data.body("right_palm").xpos[2])

    for _ in range(5):
        _space(teleop, "joint", "impedance", "arm")
        _run(model, data, teleop, 1.5)
        _space(teleop, "navigation", "impedance", "arm")
        _run(model, data, teleop, 1.5)

    _assert_held(before, _q(model, data), "5x Drive<->Task")
    palm_mm = abs(palm0 - float(data.body("right_palm").xpos[2])) * 1000.0
    assert palm_mm < 2.0, (
        f"the right palm walked {palm_mm:.2f} mm over 5 Drive<->Task cycles")


@needs_model
def test_wrists_stay_inside_their_limits_under_the_feed_forward():
    """The wrists are the joints this fix newly feeds, and they are the ones the
    torque law had to give up (they rang through their MJCF limit under -D*qd
    across the RPC hop; see WBCEngine._managed_for). A STATIC g(q) has no
    velocity term, so it must not excite them -- verify, don't assume."""
    model, data, teleop = _rig()
    _space(teleop, "joint", "impedance", "arm")
    peak = 0.0
    for _ in range(int(10.0 / DT)):
        try:
            teleop.update(None, None, DT)
        except Exception:
            pass
        mujoco.mj_step(model, data)
        for w in ("Left Wrist", "Right Wrist"):
            dof = int(model.jnt_dofadr[model.joint(w).id])
            peak = max(peak, abs(float(data.qvel[dof])))
    for w in ("Left Wrist", "Right Wrist"):
        jid = model.joint(w).id
        q = float(data.qpos[model.jnt_qposadr[jid]])
        lo, hi = model.jnt_range[jid]
        assert lo - 1e-3 <= q <= hi + 1e-3, (
            f"{w} left its MJCF range [{lo:.3f},{hi:.3f}] at {q:.3f} rad")
    assert peak < 1.0, f"wrist velocity peaked at {peak:.2f} rad/s — ringing up"

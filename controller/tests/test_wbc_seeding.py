"""Every WBC-managed joint must be seeded from the MEASURED pose on enter().

THE BUG (2026-07-19): "when I do a reset the target joint position should be
reset as well — right now when I reset and switch from drive to task or joint
mode the robot will snap to some weird position... the hand is at weird
positions again."

`WBCEngine.enter()` seeds its set-point from the current pose, but only for
joints that resolve through `_MJ_TO_IOS`:

    for j in new_managed:
        ios = _MJ_TO_IOS.get(j)
        if ios:                       # <-- silent skip
            self._q_des[ios] = float(self.plant.q(j))

`_managed_for("arm")` returns the 14 arm joints AND `Left/Right Wrist`, but the
map contained no wrist entries at all. So the wrists were never seeded and the
torque law drove them to its default 0. Their MJCF ref is +/-0.15 rad, so
"default 0" is 8.6 deg of wrist rotation — measured 7-10 deg of snap on the
first Drive -> Task/Joint switch after a reset, which twists both hands because
the wrist carries the hand.

The guard below is deliberately structural: it asserts the MAP COVERS every
region's managed set, so the next joint added to a region cannot be silently
unseeded the same way. Listing wrists by name would only have caught this one.
"""
import numpy as np
import mujoco
import pytest

from conftest import _load, needs_model

from mabel.whole_body_controller import engine as E

REGIONS = ("arm", "upper_body", "whole_body")


@needs_model
@pytest.mark.parametrize("region", REGIONS)
def test_every_managed_joint_has_a_wire_name(region):
    """The structural invariant. A managed joint with no `_MJ_TO_IOS` entry is
    invisible to enter()'s seeding loop AND to the operator's joint targets."""
    model, data = _load()
    from mabel import MabelTeleop
    wbc = MabelTeleop(model, data).wbc
    managed = list(wbc._managed_for(region))
    assert managed, f"region {region!r} manages nothing"
    keys = {j: E._wbc_key(j) for j in managed}
    assert all(keys.values()), f"region {region!r}: unkeyed joints {keys}"
    dupes = [k for k in keys.values() if list(keys.values()).count(k) > 1]
    assert not dupes, (
        f"region {region!r}: two managed joints share a set-point key {dupes} — "
        "they would fight over one entry"
    )


@needs_model
@pytest.mark.parametrize("region", REGIONS)
def test_enter_seeds_every_managed_joint_from_the_measured_pose(region):
    """The behavioural invariant: after enter(), q_des == the pose we are in."""
    model, data = _load()
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    teleop.controller.init_hold()
    teleop.controller.settle_done()
    for _ in range(300):
        mujoco.mj_step(model, data)

    wbc = teleop.wbc
    wbc.enter("impedance", region, q_des={})
    for j in wbc._managed_for(region):
        ios = E._wbc_key(j)
        assert ios in wbc._q_des, f"{j} -> {ios} was never seeded"
        measured = float(wbc.plant.q(j))
        assert abs(wbc._q_des[ios] - measured) < 1e-6, (
            f"{j}: q_des={wbc._q_des[ios]:+.4f} but measured={measured:+.4f} — "
            "entering the torque engine must not move the robot"
        )


@needs_model
def test_reset_then_switching_space_does_not_snap_the_robot():
    """The operator's exact sequence, end to end.

    Drive -> command something -> Joint -> Drive -> RESET -> Joint.
    Nothing may move on that last switch: the reset cleared the targets, so the
    incoming law must hold the freshly-reset pose.
    """
    model, data = _load()
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    ctl = teleop.controller
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
             for a in range(model.nu)]

    def q(aid):
        return float(data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]])

    def pose():
        return {n: q(a) for a, n in enumerate(names)
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}

    def spin(n):
        for _ in range(n):
            try:
                teleop.update(None, None, 1.0 / 240.0)
            except Exception:
                pass
            mujoco.mj_step(model, data)

    ctl.init_hold(); ctl.settle_done(); spin(400)
    teleop.set_control_spec(method="navigation", control_type="impedance",
                            region="arm", stiffness=27.0)
    spin(300)
    # the operator drives things somewhere, so there ARE stale desired positions
    teleop.set_joint_targets({"left_arm_2": -0.9, "right_arm_4": 0.8,
                              "left_thumb_abd": 0.2})
    teleop.set_control_spec(method="joint", control_type="impedance",
                            region="arm", stiffness=27.0)
    spin(600)
    teleop.set_control_spec(method="navigation", control_type="impedance",
                            region="arm", stiffness=27.0)
    spin(300)

    mujoco.mj_resetData(model, data)
    teleop.reset()
    ctl.init_hold()
    spin(400)
    before = pose()

    teleop.set_control_spec(method="joint", control_type="impedance",
                            region="arm", stiffness=27.0)
    spin(900)
    after = pose()

    snapped = {n: (round(before[n], 4), round(after[n], 4),
                   round(np.degrees(after[n] - before[n]), 1))
               for n in before if abs(after[n] - before[n]) > 0.05}
    assert not snapped, (
        f"switching Drive -> Joint after a reset moved {len(snapped)} joints: "
        f"{snapped}. The reset cleared the targets, so nothing may move."
    )


@needs_model
def test_the_wrists_hold_their_nonzero_ref():
    """Pin the specific magnitude, so a regression is unmistakable in the log."""
    model, data = _load()
    from mabel import MabelTeleop
    teleop = MabelTeleop(model, data)
    teleop.controller.init_hold()
    teleop.controller.settle_done()
    for _ in range(300):
        mujoco.mj_step(model, data)
    teleop.set_control_spec(method="joint", control_type="impedance",
                            region="arm", stiffness=27.0)
    for _ in range(900):
        try:
            teleop.update(None, None, 1.0 / 240.0)
        except Exception:
            pass
        mujoco.mj_step(model, data)
    # TOLERANCE WIDENED 0.03 -> 0.06 (2026-07-19). The wrists are no longer
    # torque-managed — see WBCEngine._managed_for and
    # controller/tests/test_wbc_wrist_stability.py: across the gateway->bridge
    # process split the effort feedback rang them up and wrapped them ~221° past
    # their MJCF limit. They are held by their position servo instead, which has
    # finite stiffness, so the 1.74 kg ORCA hand now droops a STATIC ~0.046 rad
    # (2.6°) below the ref. That is the expected new resting value, not drift.
    #
    # The check this test exists for is unchanged and still sharp: the failure it
    # guards is the wrist heading for ZERO (0.15 rad away), which 0.06 still
    # catches with room to spare. Do NOT widen this past ~0.08 or it stops
    # distinguishing "sags a little on its servo" from "was never seeded".
    for name, sign in (("Left Wrist", +1), ("Right Wrist", -1)):
        aid = model.actuator(name).id
        v = float(data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]])
        assert abs(v - sign * 0.15) < 0.06, (
            f"{name} settled at {v:+.4f}, expected ~{sign * 0.15:+.2f} (its MJCF "
            "ref). Drifting toward 0 is the unseeded-wrist regression."
        )

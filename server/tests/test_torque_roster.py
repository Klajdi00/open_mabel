"""robot_state.torque vs the module roster (module_status.torque_report).

Pins the 2026-10-01 fix: with ONLY the left hand plugged into the Mac, the studio's
Torque ON button appeared dead because the absent right hand kept reporting torque
False and the studio's rule is "energised <=> every REPORTED module is on". Absent
modules must report null, and the reconcile must not re-assert torque at them.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from module_status import (ModuleTracker, TORQUE_MODULES, roster_ok,  # noqa: E402
                           torque_module_present, torque_report)


def _studio_energised(torque: dict) -> bool:
    """control_center app.js `torqueOn`: every non-null entry must be True."""
    v = [x for x in torque.values() if x is not None]
    return bool(v) and all(v)


STATE_ON = {"left_hand": True, "right_hand": False, "arms": True, "neck": True, "base": None}


def test_only_left_hand_connected_reports_absent_modules_as_null():
    tr = ModuleTracker(ttl=3.0)
    tr.mark("leftHand", "17 joints")                 # the one connected module
    ok = roster_ok(tr.snapshot())
    assert ok["leftHand"] and not ok["rightHand"] and not ok["leftArm"]
    rep = torque_report(STATE_ON, ok)
    assert rep == {"left_hand": True, "right_hand": None, "arms": None,
                   "neck": None, "base": None}
    assert _studio_energised(rep)                    # the studio flips to ENERGISED


def test_raw_state_would_have_pinned_the_studio_limp():
    # the pre-fix behaviour, kept as the counter-example
    assert not _studio_energised(STATE_ON)


def test_roster_not_live_passes_torque_through_unchanged():
    assert torque_report(STATE_ON, {}) == STATE_ON   # sim / early boot: no roster yet
    assert torque_module_present("right_hand", {}) is True


def test_arms_present_if_either_arm_module_is_up():
    assert torque_module_present("arms", {"leftArm": False, "rightArm": True})
    assert not torque_module_present("arms", {"leftArm": False, "rightArm": False})


def test_connected_module_keeps_its_real_state():
    tr = ModuleTracker(ttl=3.0)
    tr.mark("leftHand"); tr.mark("rightHand")
    rep = torque_report({"left_hand": True, "right_hand": False, "arms": None,
                         "neck": None, "base": None}, roster_ok(tr.snapshot()))
    assert rep["right_hand"] is False                # a REAL limp hand still reads False
    assert not _studio_energised(rep)


def test_every_torque_key_maps_to_known_roster_ids():
    ids = {m["id"] for m in ModuleTracker().snapshot()}
    for key, mods in TORQUE_MODULES.items():
        assert set(mods) <= ids, (key, mods)

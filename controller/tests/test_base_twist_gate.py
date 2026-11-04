"""ARCHITECTURE GUARD: no ungated publisher may drive the base.

`mabel/motion_model/__init__.py` states the rule this file enforces:

    "Every base twist, from every input device (joystick, iOS/VisionPro nav sticks,
     web GUI, ROS nav, admittance glide), MUST pass through `safe_base_step(...)`
     before it reaches the swerve IK"

The repo declared that rule and then had no way to notice it being broken. Nothing
here is new policy — this is the missing enforcement. `mabel.py drift` cannot catch
it: drift checks that *cited facts still exist*, it is a citation checker, not an
invariant checker. It passed clean while several ungated publishers shipped.

HOW IT WORKS
------------
There are TWO rungs a twist can reach the wheels on, and a guard that watches only
the top one is worthless:

  1. `/mabel_cmd`              — `MabelSwerveCmd`, consumed by swerve_node
  2. `/mabel/hw/base/cmd_vel`  — `Twist`, the hardware bridge's OWN intake, which
                                 swerve_node relays onto. Publishing here skips
                                 `/mabel_cmd` entirely, so scanning only for
                                 `MabelSwerveCmd` misses it (the GUI does exactly
                                 this). Rung 2's name is read from the bridge's
                                 frozen contract, not hardcoded.

Every Python file that publishes on either rung must

  * call `safe_base_step` (it gates its own twist), or
  * be listed in EXEMPT below with a REASON that says who gates it instead.

Adding a new base publisher therefore fails this test until you make a deliberate
choice. That is the point: the failure mode being prevented is a well-meaning new
teleop/experiment quietly acquiring a raw path to the wheels.

The `EXEMPT` shape mirrors `hardware_bridge/tests/contract/contract.json`'s
`source_needles` ({file, why}) so the repo has one idiom for "declared exception",
and rung 2's topic comes from that same file — add an actuator to the contract and
this test extends itself.
"""
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CONTRACT = REPO / "hardware_bridge" / "tests" / "contract" / "contract.json"

# Files that publish MabelSwerveCmd WITHOUT calling safe_base_step, each with the
# reason it is safe. A reason must name what gates the twist instead.
EXEMPT = {
    "ros/src/mabel_control/mabel_control/ros_plant.py":
        "THE plant — it publishes the RobotCommand the gateway's controller facade "
        "already produced, and the facade ran safe_base_step (wholebody.base_step). "
        "Gating here would apply the model twice to the same twist.",

    "ros/src/mabel_navigation/experiments/collect_localization.py":
        "sim-only experiment (requires /odom_ground_truth, which exists only in sim) "
        "that drives a deliberate trajectory to compare wheel vs visual odometry.",

    "hardware_bridge/tools/latency_test.py":
        "bench instrument: measures command round-trip latency, so it must publish a "
        "raw step — the motion model's ramp is exactly what it is trying to measure.",

    "hardware_bridge/tools/drive_speed_test.py":
        "bench instrument: measures how well the drive motors TRACK a commanded "
        "speed, so it must issue raw steps — gating would measure the model instead "
        "of the motors, which is the opposite of the point.",

    "ros/src/mabel_base_hardware/mabel_base_hardware/swerve_node.py":
        "pure relay: republishes /mabel_cmd onto the bridge's Twist intake and owns "
        "only the sim->real 180-degree yaw flip. Whoever published /mabel_cmd already "
        "gated it (rung 1 of this test enforces exactly that), so gating here would "
        "apply the model twice to one twist.",
}

# Test fixtures are allowed to publish anything; they drive no real base.
TEST_PATH_MARKERS = ("/test/", "/tests/", "test_")

# Rung 1: the canonical base command message.
PUB_RE = re.compile(r"create_publisher\(\s*MabelSwerveCmd")
# Rung 2: a Twist publisher. Not damning on its own — publishing /cmd_vel is the
# CORRECT gated path (cmd_vel_bridge applies the model downstream) — so it only
# counts when aimed at the bridge's base intake, per _aims_at_intake below.
TWIST_PUB_RE = re.compile(r"create_publisher\(\s*Twist")
# The intake is often reached through an indirection rather than the literal string
# (gui/bridge_link.py uses contract.BASE["cmd_vel"]; swerve_node uses a param named
# cmd_vel_topic). Matching only the literal silently missed the GUI — the exact
# blind spot that let an ungated publisher sit unnoticed.
INTAKE_ALIAS_RE = re.compile(r"""BASE\[['"]cmd_vel['"]\]|topic_cmd_vel|cmd_vel_topic""")
SKIP_DIRS = ("/build/", "/install/", "/node_modules/", "/.git/",
             "/site-packages/", "/__pycache__/", "/log/")


def _bridge_base_intake():
    """The bridge's own base topic, from its frozen contract (never hardcoded)."""
    topic = json.loads(CONTRACT.read_text())["topics"]["base_cmd_vel"]
    assert topic.startswith("/"), f"contract base_cmd_vel looks wrong: {topic!r}"
    return topic


def _base_publishers():
    """Every non-generated .py that can put a twist on the base, via either rung.

    Rung 2 is matched by the contract topic STRING rather than the message type,
    because `Twist` is published all over the repo for unrelated reasons — it is
    the destination that makes it a base command.
    """
    intake = _bridge_base_intake()
    hits = []
    for path in REPO.rglob("*.py"):
        p = str(path)
        if any(d in p for d in SKIP_DIRS):
            continue
        try:
            src = path.read_text(errors="ignore")
        except OSError:
            continue
        rung1 = bool(PUB_RE.search(src))
        # Rung 2: PUBLISHES a Twist and aims it at the bridge's base intake, by
        # literal topic or by a known alias. Requiring a Twist *publisher* keeps out
        # subscribers and monitors that merely name the topic (simulation_bridge/
        # ros_io.py only subscribes to it, and must not be flagged).
        rung2 = bool(TWIST_PUB_RE.search(src)) and (
            intake in src or bool(INTAKE_ALIAS_RE.search(src)))
        if rung1 or rung2:
            hits.append((path.relative_to(REPO).as_posix(), src))
    return hits


def _is_test(rel):
    return any(m in "/" + rel for m in TEST_PATH_MARKERS)


def test_the_scan_finds_the_known_publishers():
    """Guard the guard: if the scan silently matches nothing, it proves nothing."""
    found = {rel for rel, _ in _base_publishers()}
    assert found, "scan found no MabelSwerveCmd publishers — the regex has rotted"
    # The plant and the nav bridge must always be there; if these vanish the scan
    # is broken (or the architecture moved and this test needs a rethink).
    for expected in ("ros/src/mabel_control/mabel_control/ros_plant.py",
                     "ros/src/mabel_navigation/mabel_navigation/cmd_vel_bridge.py"):
        assert expected in found, f"scan missed a known base publisher: {expected}"


def test_every_base_publisher_is_gated_or_declared():
    """The invariant: raw twist -> wheels is impossible without a declared reason."""
    offenders = []
    for rel, src in _base_publishers():
        if _is_test(rel) or rel in EXEMPT:
            continue
        if "safe_base_step" not in src:
            offenders.append(rel)

    assert not offenders, (
        "these files publish MabelSwerveCmd (which reaches swerve_node -> the "
        "hardware bridge -> the wheels) without calling safe_base_step:\n  "
        + "\n  ".join(offenders)
        + "\n\nEvery base twist must pass the tip-safe motion model before the swerve "
          "IK (mabel/motion_model/__init__.py). Either call safe_base_step, publish "
          "/cmd_vel and let cmd_vel_bridge gate it, or add the file to EXEMPT in this "
          "test WITH a reason naming what gates it instead.")


def test_exempt_list_has_no_stale_entries():
    """An exemption for a file that no longer publishes is a lie — delete it."""
    found = {rel for rel, _ in _base_publishers()}
    stale = [rel for rel in EXEMPT if rel not in found]
    assert not stale, (
        f"EXEMPT lists files that no longer publish MabelSwerveCmd: {stale}. "
        f"Remove them so the list keeps meaning something.")


def test_exempt_reasons_are_substantive():
    """A reason must explain; 'legacy' or '' is not an argument."""
    for rel, why in EXEMPT.items():
        assert len(why) > 40, f"EXEMPT[{rel}] needs a real reason, got: {why!r}"


@pytest.mark.parametrize("rel", [
    "ros/src/mabel_teleop/mabel_teleop/keyboard_teleop.py",
    "ros/src/mabel_navigation/mabel_navigation/cmd_vel_bridge.py",
])
def test_known_teleop_paths_stay_gated(rel):
    """Pin the two fixed in this pass so they cannot silently regress."""
    src = (REPO / rel).read_text(errors="ignore")
    assert "safe_base_step" in src, (
        f"{rel} lost its safe_base_step call — its twist reaches the wheels ungated")

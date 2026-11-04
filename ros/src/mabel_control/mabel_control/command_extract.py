"""Translate the controller's MuJoCo output into the canonical hardware command contract.

The MABEL controller (`mabel.MabelTeleop`) is MuJoCo-coupled: one `update()` writes
the full actuator target vector into `data.ctrl` and returns a `RobotCommand` that also
carries the smoothed, tip-safe base twist. On the REAL robot the control node runs the
*same* controller against a MuJoCo MIRROR (encoders -> qpos, mj_forward, NO physics step),
then reads the controller's decision back out and republishes it on the canonical topic
contract that the hardware drivers (and the sim) consume:

    /mabel_cmd          MabelSwerveCmd  base body-twist (vx, vy, omega) + lift_pos
    /arms/command       JointState      urdf arm joint -> position [rad]
    /body/command       JointState      urdf torso/neck joint -> position [rad]
    /<side>_hand/command JointState     ORCA bare joint -> position [rad]

This module is the pure, MuJoCo-only half of that read-out (no ROS), so it is unit
testable offline. `control_node` wraps it with rclpy publishers.

The grouping + name translations come from the one contract definition,
`mabel_sim.interface_config` (the same file the sim consumes), so the extractor and the
sim can never disagree about which actuator belongs to which subsystem.
"""
from dataclasses import dataclass, field


def _default_cfg():
    # The canonical contract. Imported lazily so this module stays importable (and
    # testable) even when a caller injects its own cfg.
    from mabel_sim import interface_config as cfg
    return cfg


@dataclass
class MabelCommands:
    """One frame of hardware commands, grouped by subsystem / contract topic."""
    # base body-twist + lift -> /mabel_cmd (MabelSwerveCmd)
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    lift: float = 0.0
    # urdf joint name -> target position [rad]
    arms: dict = field(default_factory=dict)    # -> /arms/command
    body: dict = field(default_factory=dict)    # -> /body/command
    # side -> {orca bare joint name -> target position [rad]}
    hands: dict = field(default_factory=dict)   # -> /<side>_hand/command


def _ctrl_by_name(model, data, name):
    """data.ctrl for actuator `name`, or None if the model has no such actuator.

    The mirror model may legitimately lack some actuators (e.g. a base-only bench
    model), so a missing actuator is skipped, not an error."""
    try:
        aid = model.actuator(name).id
    except (KeyError, ValueError, IndexError):
        return None
    return float(data.ctrl[aid])


def extract(model, data, robot_cmd, cfg=None):
    """Read the controller's per-frame decision (a `RobotCommand` + the `data.ctrl`
    it wrote) and group it into the canonical contract commands.

    `robot_cmd.base` is the already-smoothed, tip-safe base twist (vx, vy, omega) the
    controller chose -- published verbatim, so the base motion model runs exactly once
    (in the controller) and is never re-applied downstream. The lift and every joint
    target are read from `data.ctrl` by their contract actuator names.
    """
    cfg = cfg or _default_cfg()
    vx, vy, omega = (robot_cmd.base if robot_cmd is not None
                     and getattr(robot_cmd, "base", None) is not None
                     else (0.0, 0.0, 0.0))

    lift = _ctrl_by_name(model, data, cfg.LIFT_ACTUATOR)

    arms = {}
    for mj_name in cfg.ARM_ACTUATORS:
        v = _ctrl_by_name(model, data, mj_name)
        if v is not None:
            arms[cfg.urdfize(mj_name)] = v

    body = {}
    for mj_name in cfg.BODY_ACTUATORS:
        v = _ctrl_by_name(model, data, mj_name)
        if v is not None:
            body[cfg.urdfize(mj_name)] = v

    hands = {}
    for side in cfg.HAND_SIDES:
        per = {}
        for orca_bare, mj_name in cfg.hand_joint_map(side).items():
            v = _ctrl_by_name(model, data, mj_name)
            if v is not None:
                per[orca_bare] = v
        if per:
            hands[side] = per

    return MabelCommands(
        vx=float(vx), vy=float(vy), omega=float(omega),
        lift=0.0 if lift is None else lift,
        arms=arms, body=body, hands=hands,
    )

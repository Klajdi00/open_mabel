"""Offline unit test for the contract command extractor (no ROS, no hardware).

Builds a tiny MuJoCo model whose actuators are named with REAL contract names from
`mabel_sim.interface_config`, writes targets into data.ctrl, and asserts the extractor
groups them onto the right subsystem with the right (urdf / ORCA) key names and values,
and that the base twist is taken verbatim from the RobotCommand. Actuators absent from
the (deliberately small) model must be skipped, not error.
"""
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mabel_sim.interface_config")

from mabel_control.command_extract import extract, MabelCommands  # noqa: E402


class _Cmd:
    """Stand-in for mabel.types.RobotCommand (only .base is read by extract)."""
    def __init__(self, base):
        self.base = base


def _model_with(actuators):
    """A minimal valid MJCF: one slide joint + position actuator per requested name."""
    bodies, acts = [], []
    for i, name in enumerate(actuators):
        bodies.append(
            f'<body name="b{i}" pos="0 0 {i*0.1}">'
            f'<joint name="{name}" type="slide" axis="0 0 1"/>'
            f'<geom type="box" size="0.01 0.01 0.01" mass="0.01"/></body>')
        acts.append(f'<position name="{name}" joint="{name}" ctrlrange="-3 3"/>')
    xml = (f'<mujoco><worldbody>{"".join(bodies)}</worldbody>'
           f'<actuator>{"".join(acts)}</actuator></mujoco>')
    return mujoco.MjModel.from_xml_string(xml)


def test_extract_groups_and_names():
    names = ["Lift", "Left Arm 1", "Right Arm 7", "Torso", "Neck Yaw",
             "Left Index MCP", "Right Wrist"]
    model = _model_with(names)
    data = mujoco.MjData(model)
    targets = {"Lift": 0.42, "Left Arm 1": 0.10, "Right Arm 7": -0.20,
               "Torso": 0.05, "Neck Yaw": 0.30,
               "Left Index MCP": 0.50, "Right Wrist": -0.10}
    for n, v in targets.items():
        data.ctrl[model.actuator(n).id] = v

    cmd = extract(model, data, _Cmd((0.3, -0.1, 0.7)))

    assert isinstance(cmd, MabelCommands)
    # base twist is verbatim from RobotCommand.base; lift from ctrl
    assert (cmd.vx, cmd.vy, cmd.omega) == (0.3, -0.1, 0.7)
    assert cmd.lift == pytest.approx(0.42)
    # arms keyed by urdf name (snake_case)
    assert cmd.arms["left_arm_1"] == pytest.approx(0.10)
    assert cmd.arms["right_arm_7"] == pytest.approx(-0.20)
    assert "torso" not in cmd.arms                       # torso is BODY, not arms
    # body group
    assert cmd.body["torso"] == pytest.approx(0.05)
    assert cmd.body["neck_yaw"] == pytest.approx(0.30)
    # hands grouped by side, keyed by ORCA bare names
    assert cmd.hands["left"]["index_mcp"] == pytest.approx(0.50)
    assert cmd.hands["right"]["wrist"] == pytest.approx(-0.10)
    # absent contract actuators are silently skipped (small model)
    assert "left_arm_2" not in cmd.arms


def test_missing_base_defaults_to_zero():
    model = _model_with(["Lift"])
    data = mujoco.MjData(model)
    cmd = extract(model, data, _Cmd(None))
    assert (cmd.vx, cmd.vy, cmd.omega) == (0.0, 0.0, 0.0)


def test_base_only_model_has_no_joint_groups():
    """A bench model with only the base/lift actuator yields empty arm/body/hand maps."""
    model = _model_with(["Lift"])
    data = mujoco.MjData(model)
    cmd = extract(model, data, _Cmd((0.1, 0.0, 0.0)))
    assert cmd.arms == {} and cmd.body == {} and cmd.hands == {}
    assert cmd.vx == pytest.approx(0.1)

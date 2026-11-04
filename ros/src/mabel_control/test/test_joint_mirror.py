"""Offline unit test for the MuJoCo state mirror (no ROS, no hardware).

Builds a tiny MuJoCo model with a free base joint + named 1-DOF joints, then checks that
JointMirror writes measured angles into data.qpos by URDF *or* MuJoCo name, and that
mirror_base writes the free-joint pose. Runs on the Mac (MuJoCo only).
"""
import pytest

mujoco = pytest.importorskip("mujoco")

from mabel_control.joint_mirror import JointMirror, mirror_base  # noqa: E402


def _model():
    xml = """
    <mujoco>
      <worldbody>
        <body name="base">
          <joint name="base_root" type="free"/>
          <geom type="box" size="0.1 0.1 0.05" mass="1"/>
          <body name="a" pos="0 0 0.1">
            <joint name="Left Arm 1" type="hinge" axis="0 0 1"/>
            <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
            <body name="b" pos="0 0 0.1">
              <joint name="Torso" type="slide" axis="0 0 1"/>
              <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def test_mirror_joints_by_urdf_and_mujoco_name():
    model = _model()
    data = mujoco.MjData(model)
    m = JointMirror(model)

    # URDF name for the hinge, MuJoCo name for the slide -> both must resolve.
    written = m.apply(data, ["left_arm_1", "Torso", "no_such_joint"], [0.7, 0.25, 9.9])
    assert written == 2                                   # unknown name skipped

    adr_arm = int(model.joint("Left Arm 1").qposadr[0])
    adr_torso = int(model.joint("Torso").qposadr[0])
    assert data.qpos[adr_arm] == pytest.approx(0.7)
    assert data.qpos[adr_torso] == pytest.approx(0.25)


def test_mirror_base_writes_free_joint_pose():
    model = _model()
    data = mujoco.MjData(model)
    ok = mirror_base(model, data, 1.0, 2.0, 0.3, (1.0, 0.0, 0.0, 0.0))
    assert ok
    adr = int(model.joint("base_root").qposadr[0])
    assert list(data.qpos[adr:adr + 7]) == pytest.approx([1.0, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0])


def test_mirror_base_noop_without_free_joint():
    # fixed-base bench model: mirror_base must no-op, not raise
    fixed = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="Torso" type="hinge"/>'
        '<geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(fixed)
    assert mirror_base(fixed, data, 1.0, 2.0, 0.3, (1.0, 0.0, 0.0, 0.0)) is False


def test_forward_kinematics_after_mirror():
    """After mirroring + mj_forward, the model reflects the mirrored state (the loop the
    real-robot host runs each tick: write qpos -> mj_forward -> read Jacobians/poses)."""
    model = _model()
    data = mujoco.MjData(model)
    JointMirror(model).apply(data, ["left_arm_1"], [1.0])
    mujoco.mj_forward(model, data)
    assert data.qpos[int(model.joint("Left Arm 1").qposadr[0])] == pytest.approx(1.0)

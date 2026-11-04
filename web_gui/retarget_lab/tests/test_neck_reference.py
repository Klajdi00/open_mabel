"""Head motion must not depend on the ARKit room's startup heading."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from mabel.teleop_engine.geometry import neck_angles


def transform(rotation):
    out = np.eye(4)
    out[:3, :3] = rotation
    return out


@pytest.mark.parametrize('heading', [-180, -135, -90, -45, 0, 45, 90, 135, 180])
@pytest.mark.parametrize('pitch', [-60, -40, 0, 25])
def test_nod_is_one_to_one_at_every_room_heading(heading, pitch):
    ref = Rotation.from_euler('y', heading, degrees=True).as_matrix()
    head = ref @ Rotation.from_euler('x', pitch, degrees=True).as_matrix()
    result = np.degrees(neck_angles(transform(head), transform(ref)))
    np.testing.assert_allclose(result, [0, -pitch, 0], atol=1e-10)


@pytest.mark.parametrize('heading', [-150, -90, 45, 170])
def test_combined_attitude_is_invariant_to_room_heading(heading):
    ref = Rotation.from_euler('y', heading, degrees=True).as_matrix()
    local = Rotation.from_euler('yxz', [30, -40, 15], degrees=True).as_matrix()
    expected = neck_angles(transform(local), np.eye(4))
    actual = neck_angles(transform(ref @ local), transform(ref))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('torso', [-.5, -.2, .1])
@pytest.mark.parametrize('yaw,pitch,roll', [(0, -30, 0), (35, -25, 10), (-40, 15, -15)])
def test_torso_compensation_preserves_full_viewing_attitude(torso, yaw, pitch, roll):
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT
    local = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=True).as_matrix()
    expected = AVP_TO_ROBOT @ local @ AVP_TO_ROBOT.T
    ny, npitch, nr = neck_angles(transform(local), np.eye(4), torso)
    actual = (Rotation.from_euler('y', torso).as_matrix()
              @ Rotation.from_euler('ZYX', [ny, -npitch, nr]).as_matrix())
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_camera_attitude_on_robot_forward_kinematics():
    import mujoco
    from retarget_lab.server import config
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT
    model = mujoco.MjModel.from_xml_path(config.SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    head = model.body('head').id
    neutral = data.xmat[head].reshape(3, 3).copy()
    for torso in [-.4, 0., .15]:
        for yaw, pitch, roll in [(0, -35, 0), (30, -20, 10), (-30, 20, -10)]:
            local = Rotation.from_euler('yxz', [yaw, pitch, roll], degrees=True).as_matrix()
            angles = neck_angles(transform(local), np.eye(4), torso)
            data.qpos[:] = model.qpos0
            for name, value in zip(['Torso', 'Neck Yaw', 'Neck Pitch', 'Neck Roll'], [torso, *angles]):
                data.qpos[model.joint(name).qposadr[0]] += value
            mujoco.mj_forward(model, data)
            actual = data.xmat[head].reshape(3, 3) @ neutral.T
            expected = AVP_TO_ROBOT @ local @ AVP_TO_ROBOT.T
            error = Rotation.from_matrix(actual @ expected.T).magnitude()
            # MJCF fixed mounts use rounded 1.57-radian rotations.
            assert np.degrees(error) < .1


@pytest.mark.parametrize('measured_torso', [-.4, 0., .15])
def test_hardware_camera_uses_actual_torso_despite_virtual_torso_lead(measured_torso):
    """Hardware may lag the lab servo; its actual torso sets the viewing axis."""
    import mujoco
    from types import SimpleNamespace
    from retarget_lab.server import config
    from mabel.teleop_engine.engine import TeleopEngine
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT

    model = mujoco.MjModel.from_xml_path(config.SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    head = model.body('head').id
    neutral = data.xmat[head].reshape(3, 3).copy()
    torso_qadr = model.joint('Torso').qposadr[0]
    eng = TeleopEngine.__new__(TeleopEngine)
    eng.data = SimpleNamespace(qpos=model.qpos0.copy())
    eng.torso_qadr = torso_qadr
    eng.torso_rest = float(model.qpos0[torso_qadr])
    # The simulated torso is already at its goal while the physical torso
    # remains elsewhere. Compensation must preserve the physical head view.
    eng.data.qpos[torso_qadr] += -.6
    eng.neck_torso_measured = eng.torso_rest + measured_torso

    local = Rotation.from_euler('yxz', [30, -20, 10], degrees=True).as_matrix()
    angles = neck_angles(transform(local), np.eye(4), eng._neck_torso_pitch())
    for name, value in zip(['Torso', 'Neck Yaw', 'Neck Pitch', 'Neck Roll'],
                           [measured_torso, *angles]):
        data.qpos[model.joint(name).qposadr[0]] += value
    mujoco.mj_forward(model, data)
    actual = data.xmat[head].reshape(3, 3) @ neutral.T
    expected = AVP_TO_ROBOT @ local @ AVP_TO_ROBOT.T
    error = Rotation.from_matrix(actual @ expected.T).magnitude()
    assert np.degrees(error) < .1


@pytest.mark.parametrize('feedback', [None, np.nan, np.inf, -np.inf])
def test_missing_torso_feedback_preserves_lab_compensation(feedback):
    from types import SimpleNamespace
    from mabel.teleop_engine.engine import TeleopEngine

    eng = TeleopEngine.__new__(TeleopEngine)
    eng.data = SimpleNamespace(qpos=np.array([.2]))
    eng.torso_qadr, eng.torso_rest = 0, .1
    eng.neck_torso_measured = feedback
    assert eng._neck_torso_pitch() == pytest.approx(.1)


@pytest.mark.parametrize('chassis_yaw', [-340, -45, 0, 60, 340])
@pytest.mark.parametrize('torso', [-.4, 0., .15])
def test_chassis_and_torso_share_the_target_camera_attitude(chassis_yaw, torso):
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT

    local = Rotation.from_euler('yxz', [30, -20, 10], degrees=True).as_matrix()
    base_turn = np.radians(chassis_yaw)
    angles = neck_angles(transform(local), np.eye(4), torso, base_turn)
    actual = (Rotation.from_euler('z', base_turn).as_matrix()
              @ Rotation.from_euler('y', torso).as_matrix()
              @ Rotation.from_euler('ZYX', [angles[0], -angles[1], angles[2]]).as_matrix())
    expected = AVP_TO_ROBOT @ local @ AVP_TO_ROBOT.T
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('initial_yaw,current_yaw', [(0, 35), (170, -170), (-170, 170)])
def test_hardware_camera_fk_while_base_turns_and_torso_lags(initial_yaw, current_yaw):
    import mujoco
    from types import SimpleNamespace
    from retarget_lab.server import config
    from mabel.teleop_engine.engine import TeleopEngine
    from mabel.teleop_engine.hyperparams import AVP_TO_ROBOT

    model = mujoco.MjModel.from_xml_path(config.SCENE)
    data = mujoco.MjData(model)
    base = model.body('mabel_chassis').id
    head = model.body('head').id
    root_qadr = model.joint('base_root').qposadr[0]
    torso_qadr = model.joint('Torso').qposadr[0]

    def turn_base(yaw):
        angle = .5 * np.radians(yaw)
        mujoco.mju_mulQuat(data.qpos[root_qadr + 3:root_qadr + 7],
                          np.array([np.cos(angle), 0., 0., np.sin(angle)]),
                          model.qpos0[root_qadr + 3:root_qadr + 7])
        mujoco.mj_forward(model, data)

    turn_base(initial_yaw)
    neutral = data.xmat[head].reshape(3, 3).copy()
    eng = TeleopEngine.__new__(TeleopEngine)
    eng.data = SimpleNamespace(qpos=data.qpos.copy(), xmat=data.xmat.copy())
    eng.base_bid, eng._neck_base_yaw0 = base, None
    eng.torso_qadr = torso_qadr
    eng.torso_rest = float(model.qpos0[torso_qadr])
    eng.data.qpos[torso_qadr] += -.6
    eng.neck_torso_measured = eng.torso_rest - .25
    assert eng._neck_chassis_yaw() == 0.

    turn_base(current_yaw)
    eng.data.xmat[:] = data.xmat
    local = Rotation.from_euler('yxz', [30, -20, 10], degrees=True).as_matrix()
    angles = neck_angles(transform(local), np.eye(4), eng._neck_torso_pitch(),
                         eng._neck_chassis_yaw())
    for name, value in zip(['Torso', 'Neck Yaw', 'Neck Pitch', 'Neck Roll'], [-.25, *angles]):
        data.qpos[model.joint(name).qposadr[0]] += value
    mujoco.mj_forward(model, data)
    actual = data.xmat[head].reshape(3, 3) @ neutral.T
    start = Rotation.from_euler('z', initial_yaw, degrees=True).as_matrix()
    expected = start @ AVP_TO_ROBOT @ local @ AVP_TO_ROBOT.T @ start.T
    error = Rotation.from_matrix(actual @ expected.T).magnitude()
    assert np.degrees(error) < .1


def test_set_forward_reanchors_the_camera_chassis_heading():
    from types import SimpleNamespace
    from mabel.teleop_engine.engine import TeleopEngine

    eng = TeleopEngine.__new__(TeleopEngine)
    eng.data = SimpleNamespace(xmat=Rotation.from_euler('z', 40, degrees=True).as_matrix()[None])
    eng.base_bid, eng._neck_base_yaw0 = 0, 0.
    eng._neck_ref = np.eye(3)
    assert eng._neck_chassis_yaw() == pytest.approx(np.radians(40))
    eng.set_forward()
    assert eng._neck_ref is None
    assert eng._neck_base_yaw0 is None
    assert eng._neck_chassis_yaw() == 0.

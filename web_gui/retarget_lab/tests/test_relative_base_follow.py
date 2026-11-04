"""The Lab and real adapter select one body controller through the robot preset.

Since 2026-10-29 NOVA's default is the 1:1 POSITION follow (tests at the bottom); the relative follower it replaced
is kept behind MABEL_NOVA_BASE_FOLLOW=relative, and the tests above them pin that path."""
import pytest
import math
import numpy as np

from retarget_lab.server.worker import KinematicRetargeter, pose_from_frame


@pytest.fixture(scope="module")
def model():
    return KinematicRetargeter("shipped_v4").model


@pytest.fixture
def relative(monkeypatch):
    monkeypatch.setenv("MABEL_NOVA_BASE_FOLLOW", "relative")


def test_nova_profile_uses_identified_relative_follow_and_survives_reset(model, relative):
    k=KinematicRetargeter("shipped_v4", model=model)
    k.apply_settings({"robot":"nova", "scope":"global", "ee":"gripper"})
    assert k.eng.wbc.tracker.follow_mode=="relative"
    assert k.base_pose()['followMode']=="relative"
    response=k.base_pose()['followResponse']
    assert response['response_s']>0
    k.reset()
    assert k.settings['robot']=='nova' and k.settings['scope']=='global'
    assert k.eng.wbc.tracker.follow_mode=='relative'
    assert k.base_pose()['followResponse']==response


def test_switching_robot_reanchors_instead_of_using_an_old_datum(model, relative):
    k=KinematicRetargeter("shipped_v4", model=model)
    for robot,mode in [('mabel','position'),('nova','relative'),('mabel','position')]:
        k.eng.wbc.base_anchored=True
        k.apply_settings({'robot':robot})
        assert k.eng.wbc.tracker.follow_mode==mode
        assert not k.eng.wbc.base_anchored


def test_stationary_operator_cannot_be_driven_by_arm_workspace_debt(model, relative):
    k=KinematicRetargeter("shipped_v4", model=model)
    k.apply_settings({'robot':'nova','scope':'global'})
    w=k.eng.wbc
    w.tracker.relative.reset([0.,0.],0.)
    assert not w.allow_base_assist
    w.tracker.relative.step([.1,0.],0.,0.,1/60)
    assert w.allow_base_assist
    w.tracker.relative.step([.1,0.],0.,0.,1/60)
    assert not w.allow_base_assist
    assert w.tracker.relative.velocity[0]>0 # bounded stop tail, not another debt command
    class Assist:
        def base_twist(self):
            raise AssertionError('stationary arm debt must not command the base')
    k.eng._ws_assist=Assist()
    k.eng.track_arms=True
    k.eng.nav_mode=False
    w.base_tgt=(0.,0.,0.)
    k._base_step(1/60)
    assert w.base_tgt==(0.,0.,0.)


@pytest.mark.skipif(not __import__("retarget_lab.server.physics", fromlist=["NOVA_ROOT"]).NOVA_ROOT.is_dir(),
                    reason="needs the second-robot checkout (MABEL_RTL_NOVA)")
def test_turn_then_hold_does_not_create_lazy_heading_catchup(model, relative):
    k=KinematicRetargeter('shipped_v4',model=model)
    k.apply_settings({'robot':'nova','scope':'global','ee':'gripper'})
    commands=[]
    for i in range(360):
        t=i/60
        angle=math.radians(60)*min(1.,max(0.,t-1.)/1.5)
        c,s=math.cos(angle),math.sin(angle)
        H=np.array([[c,0,s,0],[0,1,0,1.6],[-s,0,c,0],[0,0,0,1.]])
        frame={'kind':'wire','p':{'timestamp':t,'head':{
            'trackingState':'normal','transform':{'matrix':H.ravel().tolist()}}}}
        k.step(pose_from_frame(frame),t,collide=False,frame=frame)
        commands.append(k.base_pose()['cmd'][2])
    assert max(abs(v)for v in commands[90:150])>.1
    assert max(abs(v)for v in commands[240:])<1e-4


def test_untracked_head_drops_relative_motion_and_rebases_on_return(model, relative):
    k=KinematicRetargeter('shipped_v4',model=model)
    k.apply_settings({'robot':'nova','scope':'global'})
    w=k.eng.wbc
    w.tracker.relative.reset([0.,0.],0.)
    w.tracker.relative.step([.1,0.],.1,0.,1/60)
    w.update_targets(k.data,None,None,None,None,dt=1/60)
    assert not w.allow_base_assist
    assert w.base_tgt==(0.,0.,0.)
    assert w.tracker.relative.step([.6,0.],1.,0.,1/60)==(0.,0.,0.)


def test_follower_gets_the_true_frame_gap_through_the_worker(model, relative):
    """2026-10-28: KinematicRetargeter.step clips eng._frame_dt to 0.1 s; the base follower must still see the TRUE
    gap, or a 0.2 s Wi-Fi gap reads as 2x the operator's speed and a stall never reaches STALL_S."""
    from mabel.motion_model import relative_follow as RF
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    seen = []
    rel = k.eng.wbc.tracker.relative
    orig = rel.step
    def spy(position, yaw, measured_yaw, dt):
        seen.append(dt); return orig(position, yaw, measured_yaw, dt)
    rel.step = spy
    t = 0.
    for gap in [1 / 60] * 30 + [0.2] + [1 / 60] * 5 + [0.5] + [1 / 60] * 5:
        t += gap
        H = np.eye(4); H[:3, 3] = (0., 1.6, -0.3 * t)          # walking forward 0.3 m/s
        frame = {'kind': 'wire', 'p': {'timestamp': t, 'head': {
            'trackingState': 'normal', 'transform': {'matrix': H.ravel().tolist()}}}}
        k.step(pose_from_frame(frame), t, collide=False, frame=frame)
        if k.eng.wbc.tracker.relative is not rel:                 # follower rebuilt (settings): re-spy
            rel = k.eng.wbc.tracker.relative; orig = rel.step; rel.step = spy
    assert any(abs(d - 0.2) < 1e-6 for d in seen), seen[-15:]
    assert any(d >= RF.STALL_S for d in seen), seen[-15:]


# ── 1:1 position follow (NOVA default since 2026-10-29) ─────────────────────────────────────────────────────────

def _head(t, x=0., z=0., yaw=0.):
    """A tracked headset frame; ARKit: forward = -z, left = -x. yaw about the NECK pivot (9 cm behind the eyes)."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    piv = np.array([x, 1.5, z + .09])
    H = np.eye(4); H[:3, :3] = R; H[:3, 3] = piv + R @ np.array([0., .10, -.09])
    return {'kind': 'wire', 'p': {'timestamp': t, 'head': {'trackingState': 'normal',
                                                            'transform': {'matrix': H.ravel().tolist()}}}}


def _run(k, frames):
    for t, f in frames:
        k.step(pose_from_frame(f), t, collide=False, frame=f)
    return k.base_pose()


def test_nova_defaults_to_one_to_one_position_follow(model):
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    tr = k.eng.wbc.tracker
    assert tr.follow_mode == 'position' and tr.one_to_one and tr.kp == 1.5 and tr.err_band == .03
    assert not k.eng.wbc.allow_base_assist          # the arm-workspace debt never drives the base
    k.apply_settings({'robot': 'mabel'})
    assert not k.eng.wbc.tracker.one_to_one          # MABEL keeps its shipped follower


@pytest.mark.parametrize('axis,dx,dz,want', [('forward', 0., -1., (1., 0.)), ('back', 0., .6, (-.6, 0.)),
                                             ('left', -.5, 0., (0., .5)), ('right', .5, 0., (0., -.5))])
def test_one_to_one_walk_moves_the_base_the_same_distance(model, axis, dx, dz, want):
    """Operator: 'when human move forward 1 meters the robot needs to move forward 1 meter too'."""
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    frames = []
    for i in range(60 * 12):                         # still 1 s, walk 3 s, hold 8 s
        t = i / 60; u = min(1., max(0., (t - 1.) / 3.))
        frames.append((t, _head(t, x=dx * u, z=dz * u)))
    b0 = _run(k, frames[:1])
    b = _run(k, frames[1:])
    yaw0 = b0['yaw']                                  # engine chassis: front = -x, left = -y at yaw 0
    d = np.array([b['x'] - b0['x'], b['y'] - b0['y']])
    fwd = -(d @ [math.cos(yaw0), math.sin(yaw0)]); left = -(d @ [-math.sin(yaw0), math.cos(yaw0)])
    assert abs(fwd - want[0]) < .04 and abs(left - want[1]) < .04, (axis, fwd, left)


def test_one_to_one_looking_around_does_not_move_the_base(model):
    """A +-50 deg look swings the EYES ~7 cm; the base follows the neck pivot, so it stays."""
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    b0 = _run(k, [(0., _head(0.))])
    b = _run(k, [(i / 60, _head(i / 60, yaw=math.radians(50) * math.sin(2 * math.pi * .25 * i / 60)))
                 for i in range(1, 60 * 8)])
    assert math.hypot(b['x'] - b0['x'], b['y'] - b0['y']) < .01


def test_one_to_one_recloses_between_frames(model):
    """NOVA receives the headset in clumps: between frames retrack_base must re-close the loop on the measured pose
    (set_base_pose), so the command shrinks as the base arrives instead of coasting on the last frame's twist."""
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    _run(k, [(i / 60, _head(i / 60)) for i in range(30)])
    _run(k, [(.5 + i / 60, _head(.5 + i / 60, z=-.05 * i)) for i in range(1, 11)])   # 0.5 m ahead (5 cm/frame)
    speeds = []
    for _ in range(80):                              # 4 s of 20 Hz gateway ticks with NO new frame
        assert k.retrack_base(.05)
        speeds.append(math.hypot(*k.base_pose()['cmd'][:2]))
    assert max(speeds[:10]) > .1 and speeds[-1] < .02, speeds[::10]


def test_relative_switch_restores_the_old_follower(model, relative):
    k = KinematicRetargeter('shipped_v4', model=model)
    k.apply_settings({'robot': 'nova', 'scope': 'global', 'ee': 'gripper'})
    assert k.eng.wbc.tracker.follow_mode == 'relative' and not k.eng.wbc.tracker.one_to_one
    assert k.retrack_base(.05) is False

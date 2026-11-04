"""Real teleop stop behavior, jitter, latency and wheel-frame transforms."""
import json
import math

import numpy as np
import pytest

from mabel.motion_model.relative_follow import RelativeBaseFollow, identified_response


def test_static_operator_never_chases_a_robot_position_or_heading():
    follow = RelativeBaseFollow(response_s=.1)
    follow.reset([.3, -.2], .4)
    for i in range(600):
        # An arbitrary changing wheel heading must not generate a command.
        assert follow.step([.3, -.2], .4, i*.03, 1/60) == (0., 0., 0.)
        assert not follow.translation_active


@pytest.mark.parametrize("hz", [30, 60, 90])
def test_walking_then_stopping_has_no_return_or_position_debt(hz):
    follow = RelativeBaseFollow(response_s=.1)
    follow.reset([0., 0.], 0.)
    output=[]
    for i in range(4*hz):
        t=(i+1)/hz
        output.append(follow.step([.08*min(t, 1.), 0.], 0., 0., 1/hz))
    output=np.array(output)
    assert output[:hz, 0].max() > .06
    assert output[:, 0].min() >= -1e-12
    assert np.max(np.abs(output[-hz:])) < 1e-4
    assert not follow.translation_active
    # First-order smoothing preserves accepted travel, including the stop tail.
    assert output[:, 0].sum()/hz == pytest.approx(.08-follow.position_band, abs=.002)


def test_tracking_noise_does_not_start_the_base():
    f=RelativeBaseFollow();f.reset([0., 0.], 0.)
    for i in range(1000):
        assert f.step([.002*math.sin(i), .002*math.cos(i)],
                      .003*math.sin(i*.7), .4*math.sin(i*.1), 1/60) == (0., 0., 0.)


def test_slow_deliberate_walk_accumulates_beyond_spatial_hysteresis():
    f=RelativeBaseFollow(response_s=.1);f.reset([0., 0.], 0.)
    output=[f.step([i/60*.01, 0.], 0., 0., 1/60)[0] for i in range(300)]
    assert output[-1] == pytest.approx(.01, abs=.0005)


def test_measured_wheel_heading_rotates_velocity_but_cannot_add_position_correction():
    f=RelativeBaseFollow(response_s=.1);f.reset([0., 0.], 0.)
    for i in range(1, 61):v=f.step([i/60*.1, 0.], 0., math.pi/2, 1/60)
    assert abs(v[0])<1e-10 and v[1]<-.09


def test_gap_rebases_input_instead_of_replaying_missed_motion():
    f=RelativeBaseFollow();f.reset([0., 0.], 0.)
    for i in range(1, 40):f.step([i*.002, 0.], 0., 0., 1/60)
    assert f.step([.8, -.4], 2., .2, .3)==(0., 0., 0.)
    for _ in range(120):assert f.step([.8, -.4], 2., .2, 1/60)==(0., 0., 0.)


def test_yaw_wrap_is_short_and_stops_without_return():
    f=RelativeBaseFollow(response_s=.1);f.reset([0., 0.], math.pi-.05)
    outputs=[f.step([0., 0.], -math.pi+.05, 0., 1/60)[2] for _ in range(180)]
    assert min(outputs)>=0.
    assert max(outputs)<=f.wmax
    assert outputs[-1]==0.


def test_invalid_input_drops_velocity_and_requires_a_new_anchor():
    f=RelativeBaseFollow();f.reset([0., 0.], 0.)
    f.step([.1, 0.], 0., 0., 1/60)
    assert f.step([float('nan'), 0.], 0., 0., 1/60)==(0., 0., 0.)
    assert f.step([.5, 0.], 1., 0., 1/60)==(0., 0., 0.)


def test_identification_shapes_input_without_enabling_candidate_feedforward(tmp_path):
    p=tmp_path/'fit.json'
    p.write_text(json.dumps({'status':'CANDIDATE_NOT_DEPLOYED','motors':{
        k:{'parameters':{'tau':t,'delay':d}} for k,t,d in [('fl',.04,.17),('fr',.05,.16),('b',.15,.07)]}}))
    out=identified_response(p)
    assert out['response_s']==pytest.approx(.11)
    assert out['plant_response_s']==pytest.approx(.22)
    assert out['status']=='CANDIDATE_NOT_DEPLOYED'
    assert identified_response(tmp_path/'missing')['status']=='fallback'


def test_wifi_burst_gaps_keep_following_at_the_true_velocity():
    """Solved frames 51 ms apart at p50, up to 233 ms at p99 (NOVA auto takes 2026-10-27): a walk sampled that way is
    followed continuously — no rebase, no dropped travel — as long as each gap is under the stall bound."""
    f = RelativeBaseFollow(response_s=.1); f.reset([0., 0.], 0.)
    rng = np.random.default_rng(0)
    t, out, dts = 0., [], []
    while t < 2.:
        dt = float(rng.choice([.017, .05, .05, .08, .15, .22]))
        t += dt; dts.append(dt)
        out.append(f.step([.3*min(t, 1.5), 0.], 0., 0., dt)[0])
        assert f.position is not None
    travel = sum(v*d for v, d in zip(out, dts))
    assert max(out[len(out)//4:len(out)//2]) > .2          # the walk is followed, not pulsed through zero
    assert travel == pytest.approx(.45 - f.position_band, abs=.03)


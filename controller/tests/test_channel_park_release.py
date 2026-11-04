"""ChannelPark — the ablation switches' mechanism (neck / torso / lift / arms /
hands). A frozen channel eases to HOME and holds; a re-enabled one must blend
back to the live command, never STEP (2026-10-16: re-enabling the neck stepped
the command by the whole live-vs-home difference in one tick)."""
import numpy as np

from mabel.teleop_engine.safety import ChannelPark


class _Data:
    def __init__(self, n):
        self.ctrl = np.zeros(n)


def _tick(park, d, live, enabled, dt=0.01):
    d.ctrl[:] = live                     # the retargeter writes the live command…
    park.update(d, enabled, dt)          # …then the park acts on it
    return d.ctrl.copy()


def test_freeze_eases_to_home_and_holds():
    d = _Data(1)
    p = ChannelPark([0], [0.0], seconds=1.0, release=1.0)
    out = [_tick(p, d, 0.5, enabled=False)[0] for _ in range(150)]
    assert abs(out[-1]) < 1e-9                           # at home
    assert max(abs(np.diff(out))) <= 0.5 * 0.01 / 1.0 + 1e-9   # linear ease, no jump


def test_release_has_no_command_step_and_reaches_live():
    d = _Data(1)
    p = ChannelPark([0], [0.0], seconds=0.5, release=1.0)
    for _ in range(100):
        _tick(p, d, 0.4, enabled=False)                  # parked at 0
    out = [_tick(p, d, 0.4, enabled=True)[0] for _ in range(120)]
    steps = np.abs(np.diff([0.0] + out))
    # smoothstep over 1 s at dt 0.01: peak slope 1.5 × 0.4 / 1 s = 0.6/s → 0.006/tick
    assert steps.max() < 0.0065, f"command stepped by {steps.max():.4f} on release"
    assert abs(out[-1] - 0.4) < 1e-9                     # fully live after the release


def test_live_channel_is_untouched():
    d = _Data(2)
    p = ChannelPark([0, 1], [0.0, 0.0])
    for v in (0.1, -0.3, 0.7):
        assert np.allclose(_tick(p, d, v, enabled=True), v)


def test_refreeze_mid_release_starts_from_the_output():
    d = _Data(1)
    p = ChannelPark([0], [0.0], seconds=1.0, release=1.0)
    for _ in range(200):
        _tick(p, d, 0.8, enabled=False)
    for _ in range(30):
        held = _tick(p, d, 0.8, enabled=True)[0]       # mid-release, < 0.8
    nxt = _tick(p, d, 0.8, enabled=False)[0]
    assert abs(nxt - held) < 0.02, "re-freezing mid-release must not jump to the live value"

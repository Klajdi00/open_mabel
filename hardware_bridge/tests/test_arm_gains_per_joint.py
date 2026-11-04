"""Per-joint arm gains: the wrist must be soft while the shoulder stays stiff.

One kp for all seven joints is not physical. J1/J2 swing the whole arm; J5-J7 are
the wrist, and with no end effector their inertia is orders of magnitude smaller:
w ~ sqrt(kp/J) rises and the damping ratio kd/(2*sqrt(kp*J)) FALLS, so the same
gain that is calm on the shoulder makes the unloaded wrist ring against the 50 Hz
command rate. That was the "J5/J6/J7 jitter with no hand fitted" report
(2026-07-16). These pin the shape, not the exact numbers (which are tunable).
"""
import pytest

from mabel_hw import config
from mabel_hw.drivers.openarm import OpenArmGroup

WRIST = ("j5", "j6", "j7")


class FakeLink:
    def __init__(self):
        self.written = None
        self.sent = 0

    def write_slice(self, idx, pos, vel, kp, kd, tff):
        self.written = (list(kp), list(kd))

    def send(self):
        # command() pushes the merged vector immediately (2026-07-29 latency fix)
        self.sent += 1


def _group(side="LEFT"):
    cfg = getattr(config, f"{side}_ARM")
    names = tuple(cfg["joints"])
    g = OpenArmGroup(FakeLink(), names, kp=cfg["kp"], kd=cfg["kd"])
    g._resolved = True
    g._torque = True
    g._idx = tuple(range(len(names)))
    return g, names


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_wrist_is_softer_than_the_shoulder(side):
    g, names = _group(side)
    g.command({n: 0.0 for n in names})
    kp, kd = g._link.written
    by = dict(zip(names, kp))
    shoulder = [v for n, v in by.items() if n[-2:] not in WRIST]
    wrist = [v for n, v in by.items() if n[-2:] in WRIST]
    assert min(shoulder) > max(wrist), (
        f"the wrist must be SOFTER than the shoulder — got shoulder={shoulder} "
        f"wrist={wrist}. Equal gains are what made the unloaded wrist jitter.")


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_shoulder_and_elbow_gains_track_the_shared_params(side):
    """Softening the wrist must not quietly soften the joints that hold the arm
    up — a limp J2 is how the arm sags. Since the gravity-FF work the single
    source for those gains is controller/control_param.json (derived kp≈15
    shoulders, capped by the kd≤5 damping ceiling, with feedforward carrying
    gravity — NOT the old flat 30): what the HAL streams must match that file
    exactly, or the robot and the setup studio run different gains."""
    file_gains = (config._control_params().get("arm") or {}).get("gains") or {}
    g, names = _group(side)
    g.command({n: 0.0 for n in names})
    kp = dict(zip(names, g._link.written[0]))
    for n in names:
        if n[-2:] in WRIST:
            continue
        fg = file_gains.get(n)
        if isinstance(fg, dict) and "kp" in fg:
            assert kp[n] == pytest.approx(float(fg["kp"])), \
                f"{n} kp {kp[n]} drifted from the shared control_param.json {fg['kp']}"
        else:   # joint absent from the file → the historical built-in fallback
            assert kp[n] == 30.0, f"{n} kp drifted to {kp[n]}"


def test_a_scalar_still_works():
    """The map is additive: a plain float must still configure every joint (the
    CAN/teensy backends and every existing caller pass a scalar)."""
    names = ("a", "b", "c")
    g = OpenArmGroup(FakeLink(), names, kp=12.0, kd=0.5)
    g._resolved = True
    g._torque = True
    g._idx = (0, 1, 2)
    g.command({n: 0.0 for n in names})
    assert g._link.written == ([12.0] * 3, [0.5] * 3)


def test_map_default_key_covers_unnamed_joints():
    names = ("x_j1", "x_j5")
    g = OpenArmGroup(FakeLink(), names, kp={"default": 30.0, "x_j5": 8.0},
                     kd={"default": 1.0, "x_j5": 0.4})
    g._resolved = True
    g._torque = True
    g._idx = (0, 1)
    g.command({n: 0.0 for n in names})
    assert g._link.written == ([30.0, 8.0], [1.0, 0.4])


def test_a_per_command_kp_still_overrides_the_config():
    """The controller's compliance/impedance modes set kp per command; the config
    is only the DEFAULT for a joint the command does not name."""
    g, names = _group()
    g.command({n: 0.0 for n in names}, kp={names[4]: 25.0})
    assert g._link.written[0][4] == 25.0


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_gains_stay_within_the_actuator_envelope(side):
    """Sanity: a runaway gain would be a regression, and a negative one is
    positive feedback — it would run the joint away. Bounds reflect the DERIVED
    per-joint gains from controller/control_param.json (gravity-FF era: elbows
    ~30.5, kd up to the DaMiao's hard 5.0 encoding cap), not the old flat 30/1."""
    cfg = getattr(config, f"{side}_ARM")
    for name, v in cfg["kp"].items():
        assert 0.0 < v <= 35.0, f"{name} kp={v} outside (0, 35]"
    for name, v in cfg["kd"].items():
        assert 0.0 < v <= 5.0, f"{name} kd={v} outside (0, 5] (5.0 = the MIT wire's kd cap)"

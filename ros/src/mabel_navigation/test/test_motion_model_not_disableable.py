"""`use_motion_model:=false` must not be an accident away from an ungated base.

The tip-safe motion model is the only lift-aware speed/accel + ZMP gate on the Nav2
path. It was a plain ROS parameter, so a launch-file typo — or a benchmark config
copied onto the robot — silently turned the gate into a no-op with no signal.

It stays a knob (the stability harness legitimately needs an ungated baseline to
measure the gate against), but disabling it now takes an explicit, separate env
opt-in and is always logged loudly. Two independent mistakes, not one.

Skips cleanly without rclpy / the ROS overlay.
"""
import pytest

rclpy = pytest.importorskip("rclpy", reason="ROS 2 not on the path")
pytest.importorskip("mabel_interfaces", reason="ros overlay not sourced")

import rclpy.parameter  # noqa: E402

from mabel_navigation.cmd_vel_bridge import CmdVelBridge  # noqa: E402

ENV = "MABEL_ALLOW_UNGATED_BASE"


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def _bridge(**overrides):
    params = [rclpy.parameter.Parameter(k, value=v) for k, v in overrides.items()]
    return CmdVelBridge(parameter_overrides=params) if params else CmdVelBridge()


def test_on_by_default(ros):
    n = _bridge()
    try:
        assert n.use_motion_model is True
    finally:
        n.destroy_node()


def test_disable_is_ignored_without_the_env_optin(ros, monkeypatch):
    """The param alone must NOT be enough — that was the whole hazard."""
    monkeypatch.delenv(ENV, raising=False)
    n = _bridge(use_motion_model=False)
    try:
        assert n.use_motion_model is True, (
            "use_motion_model:=false disabled the tip gate with no second signal; "
            f"it must be ignored unless {ENV}=1 is also set")
    finally:
        n.destroy_node()


def test_disable_requires_both_the_param_and_the_env(ros, monkeypatch):
    """With both set, honour it — benchmarks need an ungated baseline."""
    monkeypatch.setenv(ENV, "1")
    n = _bridge(use_motion_model=False)
    try:
        assert n.use_motion_model is False
    finally:
        n.destroy_node()


def test_env_alone_does_not_disable_it(ros, monkeypatch):
    """The env is a permission, not an action — it must not gate-open by itself."""
    monkeypatch.setenv(ENV, "1")
    n = _bridge()
    try:
        assert n.use_motion_model is True
    finally:
        n.destroy_node()

"""mabel_hw purity contract:

  * ``import mabel_hw`` is CHEAP — no rclpy, no numpy, no device SDK at import;
  * rclpy may only ever be imported by mabel_hw.ros / mabel_hw.cameras (the two
    adapter subpackages) — one careless import elsewhere breaks every ROS-less
    consumer (Mac GUI, tests, experiments);
  * every lazy ``__all__`` symbol actually resolves.

(The forbid-hook pattern mirrors controller/tests' config-is-mujoco-free test.)
"""
import importlib
import pkgutil
import subprocess
import sys

import pytest

from conftest import BRIDGE_DIR

# subpackages that legitimately import ROS / device SDKs — everything else must not
ADAPTER_PREFIXES = ("mabel_hw.ros", "mabel_hw.cameras")
FORBIDDEN = ("rclpy",)


def test_import_mabel_hw_is_light():
    """Fresh interpreter: importing the package must not drag heavy deps."""
    code = ("import sys, mabel_hw; "
            "bad = [m for m in ('rclpy', 'numpy', 'serial', 'can') "
            "       if m in sys.modules]; "
            "assert not bad, f'import mabel_hw pulled {bad}'; "
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=BRIDGE_DIR,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_lazy_all_resolves():
    import mabel_hw
    for name in mabel_hw.__all__:
        obj = getattr(mabel_hw, name)
        assert obj is not None, name
    # identity: lazy resolution hands out the SAME objects as direct imports
    from mabel_hw.arbitration import LeaseBroker
    assert mabel_hw.LeaseBroker is LeaseBroker
    from mabel_hw.types import RobotState
    assert mabel_hw.RobotState is RobotState


def _walk_mabel_hw_modules():
    import mabel_hw
    for info in pkgutil.walk_packages(mabel_hw.__path__, prefix="mabel_hw."):
        yield info.name


def test_no_rclpy_outside_adapters():
    """Import every mabel_hw submodule (except the adapter subpackages) under a
    meta-path hook that refuses rclpy. Adapter modules are exempt by design."""

    class Forbid:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in FORBIDDEN:
                raise ImportError(
                    f"{fullname} imported by a non-adapter mabel_hw module — "
                    f"rclpy is only allowed under {ADAPTER_PREFIXES}")
            return None

    hook = Forbid()
    sys.meta_path.insert(0, hook)
    try:
        for name in _walk_mabel_hw_modules():
            if name.startswith(ADAPTER_PREFIXES):
                continue
            importlib.import_module(name)
    finally:
        sys.meta_path.remove(hook)


def test_mock_driver_is_deterministic():
    from mabel_hw.drivers.seam import MockJointGroupDriver

    def run():
        drv = MockJointGroupDriver(["a", "b"], alpha=0.5)
        drv.connect()
        drv.set_torque(True)
        drv.set_targets({"a": 1.0})
        for _ in range(5):
            drv.tick()
        return drv.read().position

    p1, p2 = run(), run()
    assert p1 == p2                       # bit-identical across runs
    assert abs(p1[0] - (1.0 - 0.5 ** 5)) < 1e-12
    assert p1[1] == 0.0


def test_mock_driver_respects_torque_off():
    from mabel_hw.drivers.seam import MockJointGroupDriver
    drv = MockJointGroupDriver(["a"])
    drv.connect()
    drv.set_targets({"a": 2.0})          # torque OFF: latches, must not move
    for _ in range(3):
        drv.tick()
    assert drv.read().position[0] == 0.0
    with pytest.raises(KeyError):
        drv.set_targets({"nope": 1.0})

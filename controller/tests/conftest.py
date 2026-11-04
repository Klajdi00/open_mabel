"""Shared fixtures for the mabel controller test suite.

All physics tests run headless MuJoCo against the canonical MJCF
(simulation/mabel_mujoco/models/mabel_full.xml). Tests that enter torque
modes mutate actuator gains, so they use the `fresh` factory (a private
model+data per test); read-only kinematic tests may share `kmodel`.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The bare-robot SCENE (robot + floor, no props): the robot model alone
# (models/mabel_full.xml) has no ground plane and free-falls.
MODEL_XML = os.path.join(_REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml")

needs_model = pytest.mark.skipif(not os.path.isfile(MODEL_XML),
                                 reason=f"canonical MJCF not found: {MODEL_XML}")


def _load():
    import mujoco
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def settle(model, data, steps=400):
    """Let the position servos settle the robot to its gravity equilibrium."""
    import mujoco
    for _ in range(steps):
        mujoco.mj_step(model, data)


@pytest.fixture()
def fresh():
    """Factory: a private (model, data) pair, safe to mutate gains/masses."""
    if not os.path.isfile(MODEL_XML):
        pytest.skip("canonical MJCF not available")
    return _load


@pytest.fixture(scope="session")
def kmodel():
    """Shared read-only (model, data) for kinematic checks (do NOT mutate)."""
    if not os.path.isfile(MODEL_XML):
        pytest.skip("canonical MJCF not available")
    return _load()


def arm_qpos(model, data, joints):
    return np.array([data.joint(j).qpos[0] for j in joints])


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: closed-loop physics sweeps (seconds, not milliseconds) — the "
        "collision guarantee is only meaningful measured this way, so they are "
        "marked rather than trimmed. Run a fast subset with -m 'not slow'.")

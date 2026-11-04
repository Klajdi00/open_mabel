"""MABEL_HW_* env-override plumbing — the systemd unit and every launch script
retune the bridge through these; they must survive the config move."""
import os
import subprocess
import sys

from conftest import BRIDGE_DIR


def test_env_helpers(monkeypatch):
    from mabel_hw import config
    monkeypatch.setenv("MABEL_HW_TESTF", "2.5")
    monkeypatch.setenv("MABEL_HW_TESTI", "7")
    monkeypatch.setenv("MABEL_HW_TESTS", "hello")
    assert config._envf("TESTF", 0.0) == 2.5
    assert config._envi("TESTI", 0) == 7
    assert config._envs("TESTS", "") == "hello"
    # bool accepts 1/true/yes/on, case-insensitive; anything else is False
    for v, expect in (("1", True), ("true", True), ("YES", True), ("on", True),
                      ("0", False), ("false", False), ("banana", False)):
        monkeypatch.setenv("MABEL_HW_TESTB", v)
        assert config._envb("TESTB", not expect) is expect, v
    # bad numeric falls back to default instead of raising
    monkeypatch.setenv("MABEL_HW_TESTF", "not-a-float")
    assert config._envf("TESTF", 1.25) == 1.25


def test_env_defaults(monkeypatch):
    from mabel_hw import config
    monkeypatch.delenv("MABEL_HW_NOPE", raising=False)
    assert config._envf("NOPE", 3.0) == 3.0
    assert config._envb("NOPE", True) is True


def test_override_reaches_config_dict():
    """End-to-end: an env var set before import lands in the subsystem dict
    (config reads env at import time, so probe in a subprocess)."""
    code = ("from mabel_hw import config; "
            "assert config.BASE['port'] == 5901, config.BASE['port']; "
            "assert config.HANDS['enabled'] is False; "
            "print('ok')")
    env = dict(os.environ,
               MABEL_HW_BASE_PORT="5901",
               MABEL_HW_HANDS="false")
    out = subprocess.run([sys.executable, "-c", code], cwd=BRIDGE_DIR,
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"

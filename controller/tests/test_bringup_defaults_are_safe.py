"""ARCHITECTURE GUARD: a bare bringup must not fight the bridge for a device.

`hardware_bridge` is the sole hardware owner. `hardware.launch.py` used to contradict
that in its DEFAULTS: `camera_node:=true` re-opened the D435i/ZED/D405 through the
legacy conda node (and re-bound :8080), and `lidar:=true` started the apt sllidar
driver against the CP2102 the bridge holds with an EXCLUSIVE open.

Nothing was broken in practice only because `real_run.sh` happened to pass both as
false. That is safe-by-accident: a bare `ros2 launch mabel_bringup hardware.launch.py`
— the documented way to bring the robot up — reintroduced the double-owners. The
defaults now match the architecture, and this test keeps them there.

`body` is the deliberate EXCEPTION and is asserted to stay true: body_node is the
neck's only owner today (the bridge cedes it via MABEL_HW_BODY=false), so flipping it
would leave the neck undriven. It flips only when the bridge gains a neck owner.

Parsed as text, not by importing launch, so it runs without a ROS install.
"""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
LAUNCH = (REPO / "ros" / "src" / "mabel_bringup" / "launch" / "hardware.launch.py")

# arg -> (required default, why)
MUST_DEFAULT = {
    "camera_node": ("false",
                    "the bridge opens the cameras via the vendor SDKs and serves "
                    ":8080 itself; the legacy conda node double-opens the USB devices "
                    "and double-binds the port"),
    "lidar": ("false",
              "the bridge's LIDAR subsystem holds the RPLIDAR's CP2102 with an "
              "exclusive open and publishes /scan; the apt sllidar node wedges it"),
    "body": ("true",
             "EXCEPTION: body_node is the neck's ONLY owner (the bridge cedes it via "
             "MABEL_HW_BODY=false) — false would leave the neck undriven. Flip only "
             "when the bridge gains a neck owner at parity"),
}


def _declared_default(arg):
    """The default_value of DeclareLaunchArgument("<arg>", ...), or None."""
    src = LAUNCH.read_text()
    m = re.search(
        r'DeclareLaunchArgument\(\s*["\']' + re.escape(arg)
        + r'["\']\s*,\s*default_value\s*=\s*["\']([^"\']+)["\']', src)
    return m.group(1) if m else None


def test_launch_file_exists():
    assert LAUNCH.is_file(), f"bringup launch not found at {LAUNCH}"


@pytest.mark.parametrize("arg", sorted(MUST_DEFAULT))
def test_default_matches_the_architecture(arg):
    want, why = MUST_DEFAULT[arg]
    got = _declared_default(arg)
    assert got is not None, (
        f"could not find DeclareLaunchArgument({arg!r}, default_value=...) — if the "
        f"arg was renamed or restructured, update this guard deliberately")
    assert got == want, (
        f"hardware.launch.py declares {arg}:={got}, expected {want}.\n\nWhy: {why}.\n\n"
        f"A bare `ros2 launch mabel_bringup hardware.launch.py` must not fight the "
        f"hardware bridge for a device — do not rely on real_run.sh's overrides to "
        f"make it safe.")

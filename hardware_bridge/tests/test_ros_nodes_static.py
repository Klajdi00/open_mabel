"""Static (AST-level) contract of the ROS adapter nodes — these modules need
rclpy to execute, so on the Mac we at least freeze their STRUCTURE: the
build(cfg, hw=None, **_) factory, the legacy class aliases the shims re-export,
the publisher watchdog on every command topic, and hw-mode + standalone
branches both present. thor's full pytest run exercises them for real."""
import ast
import os

import pytest

from conftest import BRIDGE_DIR

NODES = {
    "base_node": ("BaseNode", "BaseSubsystem"),
    "arms_node": ("ArmsNode", "ArmsSubsystem"),
    "body_node": ("BodyNode", "BodySubsystem"),
    "hands_node": ("HandsNode", "HandsSubsystem"),
    "lift_node": ("LiftNode", "LiftSubsystem"),
    "lidar_node": ("LidarNode", "LidarSubsystem"),
    "audio_node": ("AudioNode", "AudioSubsystem"),
}
# nodes that consume a command topic and must carry the >1-publisher watchdog
COMMANDED = ("base_node", "arms_node", "body_node", "lift_node",
             "audio_node")


def _tree(mod):
    path = os.path.join(BRIDGE_DIR, "mabel_hw", "ros", f"{mod}.py")
    return ast.parse(open(path, encoding="utf-8").read()), \
        open(path, encoding="utf-8").read()


@pytest.mark.parametrize("mod", sorted(NODES))
def test_build_factory_signature(mod):
    tree, _ = _tree(mod)
    builds = [n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build"]
    assert builds, f"{mod}: no build()"
    fn = builds[0]
    args = [a.arg for a in fn.args.args]
    assert args[0] == "cfg"
    assert any(a.arg == "hw" for a in fn.args.args + fn.args.kwonlyargs), \
        f"{mod}: build() must accept hw="
    assert fn.args.kwarg is not None, f"{mod}: build() must swallow **_"


@pytest.mark.parametrize("mod", sorted(NODES))
def test_legacy_class_alias(mod):
    tree, _ = _tree(mod)
    new_cls, legacy = NODES[mod]
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert new_cls in classes, f"{mod}: class {new_cls} missing"
    aliases = {t.id: getattr(n.value, "id", None)
               for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
    assert aliases.get(legacy) == new_cls, \
        f"{mod}: legacy alias {legacy} = {new_cls} missing (shims re-export it)"


@pytest.mark.parametrize("mod", COMMANDED)
def test_publisher_watchdog_present(mod):
    _, src = _tree(mod)
    assert "add_publisher_watchdog" in src, \
        f"{mod}: command topic without the >1-publisher watchdog"


@pytest.mark.parametrize("mod", sorted(NODES))
def test_hw_and_standalone_branches(mod):
    _, src = _tree(mod)
    assert "hw is not None" in src, f"{mod}: no hw-mode branch"


def test_bridge_registry_frozen():
    path = os.path.join(BRIDGE_DIR, "mabel_hw", "ros", "bridge.py")
    src = open(path, encoding="utf-8").read()
    for key, module in [("BASE", "mabel_hw.ros.base_node"),
                        ("ARMS", "mabel_hw.ros.arms_node"),
                        ("BODY", "mabel_hw.ros.body_node"),
                        ("HANDS", "mabel_hw.ros.hands_node"),
                        ("LIFT", "mabel_hw.ros.lift_node"),
                        ("LIDAR", "mabel_hw.ros.lidar_node"),
                        ("CAMERAS", "mabel_hw.cameras.manager")]:
        assert f'("{key}", "{module}")' in src, (key, module)


def test_launcher_shim_surface():
    src = open(os.path.join(BRIDGE_DIR, "hardware_bridge.py"),
               encoding="utf-8").read()
    for needle in ("_acquire_singleton", "_try_lock", "_REGISTRY", "main",
                   'if __name__ == "__main__":'):
        assert needle in src, needle


@pytest.mark.ros
def test_legacy_class_aliases_still_exported():
    """The `<Name>Subsystem` aliases are the node classes themselves — kept because
    the launch/registry surface names them. (The top-level `subsystems/` shim
    package that used to re-export these was deleted with the shim layer.)"""
    from mabel_hw.ros.base_node import BaseNode, BaseSubsystem
    assert BaseSubsystem is BaseNode
    from mabel_hw.ros.arms_node import ArmsNode, ArmsSubsystem
    assert ArmsSubsystem is ArmsNode

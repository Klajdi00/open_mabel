"""
DiT policy plugin — importing this package registers ``policy.type = "dit"`` with
LeRobot's config registry AND patches the factory's ``get_policy_class`` (a
hardcoded if/elif chain) so "dit" resolves to DiTPolicy. Import it before running
lerobot-train (the studio's run_train launcher does this).
"""
from .configuration_dit import DiTConfig  # noqa: F401  (registers the config)


def _register_policy_class():
    import lerobot.policies.factory as factory
    if getattr(factory, "_studio_dit_patched", False):
        return
    _orig = factory.get_policy_class

    def get_policy_class(name):
        if name == "dit":
            from .modeling_dit import DiTPolicy
            return DiTPolicy
        return _orig(name)

    factory.get_policy_class = get_policy_class
    factory._studio_dit_patched = True


_register_policy_class()

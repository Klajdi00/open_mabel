"""Self-collision avoidance: capsule bounding volumes + three velocity-domain laws
("project" CBF velocity-projection — keep tangential, remove the normal closing
excess — the default; "stop" uniform slow-down; "cbf" tracking-QP), the
torque-domain soft-wall barrier, and the manipulate-target CBF filter
(JointTargetProjector — the choke point every client's wrist/joint targets pass
through). This guard is the ONLY arm-arm defense — the MJCF disables arm-arm
contacts — so it is ON by default (config.SELF_COLLISION_ENABLE)."""
from .capsule_guard import SelfCollisionGuard, solve_qp, Capsule  # noqa: F401
from .target_filter import JointTargetProjector  # noqa: F401

__all__ = ["SelfCollisionGuard", "solve_qp", "Capsule", "JointTargetProjector"]

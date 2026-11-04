"""Vision Pro -> MABEL teleop: modular implementation package.

One responsibility per module:
  hyperparams  - all tunable parameters (edit here to tune)
  geometry     - frame math + operator-frame helpers + BodyFrame
  hand         - HandOptimizer (finger retargeting)
  arm          - ArmRetargeter (arm IK)
  collision    - ArmCollisionProjector (self-collision QP)
  motion_model - BaseMotionModel + LiftAwareReverseGuard (Layer 1)
  wholebody    - WholeBodyController + BaseTracker + TorsoController (Layer 2)
  safety       - EngagementSupervisor + ClutchController
  hud          - PerfMonitor
  overlay      - Overlay
  model_utils  - actuator/DoF maps, reach calibration
The runnable entry point is scripts/retarget_teleop.py, which just orchestrates these.
"""
from .hyperparams import *                      # noqa: F401,F403
from .logsetup import logger                    # noqa: F401
from .deps import (swerve_ik, MC, SelfCollisionGuard, solve_qp,   # noqa: F401
                   _HAVE_SWERVE, _HAVE_SELFCOL)
from .geometry import (quat_from_mat, operator_frame, neck_angles,   # noqa: F401
                       wrist_flexion, BodyFrame, clamp_in_front,
                       palm_axes, ElbowYawEstimator)
from .hand import HandOptimizer                 # noqa: F401
from .arm import ArmRetargeter                  # noqa: F401
from .collision import ArmCollisionProjector    # noqa: F401
from .motion_model import BaseMotionModel, LiftAwareReverseGuard   # noqa: F401
from .wholebody import (WholeBodyController, BaseTracker,    # noqa: F401
                        TorsoController)
from .safety import EngagementSupervisor, ClutchController   # noqa: F401
from .hud import PerfMonitor, _A                # noqa: F401
from .overlay import Overlay                    # noqa: F401
from .model_utils import (position_actuators, gravity_dofs,   # noqa: F401
                          robot_forward_reach, human_reach, calibrate_scale)
from .engine import TeleopEngine            # noqa: F401

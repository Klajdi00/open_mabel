"""
Lazy-base reactive follower.

When the operator's hand targets drift toward the edge of the arms' reachable
workspace, the mobile base creeps to recentre them. This is MABEL's analogue of
the root-translation handling in GMR / HoMMI: the swerve base supplies the
"root" motion so the arms stay in a comfortable configuration.

The comfort box is expressed in the robot's local frame. MABEL's front is -X
and its left is -Y, so we project onto the negated chassis axes.
"""
import math
import numpy as np

from .. import config as C


class LazyBase:
    def __init__(self):
        self.e_fwd = 0.0
        self.e_left = 0.0

    def command(self, p_l, p_r, data):
        """Return (vx, vy, omega) body-frame base velocity to recentre the hands."""
        base_pos = data.body(C.CHASSIS_BODY).xpos
        base_mat = data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
        fwd_vec  = -base_mat[:, 0]   # robot front (-X world at identity)
        left_vec = -base_mat[:, 1]   # robot left  (-Y world at identity)

        centroid = 0.5 * (p_l + p_r)
        e = centroid - base_pos
        self.e_fwd = float(np.dot(e, fwd_vec))
        self.e_left = float(np.dot(e, left_vec))

        v_fwd = v_left = omega = 0.0
        if self.e_fwd > C.LAZY_FWD_MAX:
            v_fwd = C.LAZY_KP_LIN * (self.e_fwd - C.LAZY_FWD_MAX)
        elif self.e_fwd < C.LAZY_FWD_MIN:
            v_fwd = C.LAZY_KP_LIN * (self.e_fwd - C.LAZY_FWD_MIN)

        if self.e_left > C.LAZY_LEFT_HALFSPAN:
            v_left = C.LAZY_KP_LIN * (self.e_left - C.LAZY_LEFT_HALFSPAN)
        elif self.e_left < -C.LAZY_LEFT_HALFSPAN:
            v_left = C.LAZY_KP_LIN * (self.e_left + C.LAZY_LEFT_HALFSPAN)

        yaw_err = math.atan2(self.e_left, self.e_fwd)
        if yaw_err > C.LAZY_YAW_DEADBAND:
            omega = C.LAZY_KP_ANG * (yaw_err - C.LAZY_YAW_DEADBAND)
        elif yaw_err < -C.LAZY_YAW_DEADBAND:
            omega = C.LAZY_KP_ANG * (yaw_err + C.LAZY_YAW_DEADBAND)

        # Re-express forward/left back into MABEL's -X / -Y motor convention.
        return -v_fwd, -v_left, omega


class LazyLift:
    """Vertical reach follower. Creeps the lift so the hand centroid stays near a
    comfortable height — the lift's analogue of LazyBase, used in manipulation
    mode where the lift is excluded from the fast 8-DOF arm IK.

    Returns a lift-travel velocity (m/s); the caller integrates it into the Lift
    actuator command and clamps to its ctrlrange.
    """

    def __init__(self):
        self.e_up = 0.0

    def command(self, p_l, p_r, data):
        centroid_z = 0.5 * (p_l[2] + p_r[2])
        ref = data.body(C.CHASSIS_BODY).xpos[2] + C.LAZY_LIFT_REF
        self.e_up = float(centroid_z - ref)

        v = 0.0
        if self.e_up > C.LAZY_LIFT_HALFSPAN:
            v = C.LAZY_LIFT_KP * (self.e_up - C.LAZY_LIFT_HALFSPAN)
        elif self.e_up < -C.LAZY_LIFT_HALFSPAN:
            v = C.LAZY_LIFT_KP * (self.e_up + C.LAZY_LIFT_HALFSPAN)
        return float(np.clip(v, -C.LAZY_LIFT_MAX_SPEED, C.LAZY_LIFT_MAX_SPEED))


class LazyTorso:
    """Forward-reach lean follower. Leans the torso hinge forward for moderate
    forward reach before the base has to translate. Returns a torso pitch rate
    (rad/s); negative leans forward (toward the joint's -angle limit).
    """

    def __init__(self):
        self.e_fwd = 0.0

    def command(self, p_l, p_r, data):
        base_pos = data.body(C.CHASSIS_BODY).xpos
        base_mat = data.body(C.CHASSIS_BODY).xmat.reshape(3, 3)
        fwd_vec = -base_mat[:, 0]                 # robot front (-X at identity)
        centroid = 0.5 * (p_l + p_r)
        self.e_fwd = float(np.dot(centroid - base_pos, fwd_vec))

        v = 0.0
        if self.e_fwd > C.LAZY_TORSO_FWD_MAX:
            # Reach forward -> lean forward -> drive the hinge toward -angle.
            v = -C.LAZY_TORSO_KP * (self.e_fwd - C.LAZY_TORSO_FWD_MAX)
        return float(np.clip(v, -C.LAZY_TORSO_MAX_SPEED, C.LAZY_TORSO_MAX_SPEED))

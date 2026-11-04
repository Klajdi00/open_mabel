"""
Swerve-drive inverse kinematics for the 3-module delta base.

Maps a body-frame velocity command (vx, vy, omega) to per-module steering
angles and wheel speeds, with shortest-path steering and a cosine anti-slip
choke so the wheels never scrub the floor.
"""
import math

from ... import config as C


def optimize_steering(target_angle, current_angle):
    """Return (steer_angle, drive_sign): take the shorter of steering vs.
    reversing the wheel, so the column never sweeps more than +/-90 deg."""
    target_angle = current_angle + (
        (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
    )
    if target_angle - current_angle > math.pi / 2:
        return target_angle - math.pi, -1.0
    if target_angle - current_angle < -math.pi / 2:
        return target_angle + math.pi, -1.0
    return target_angle, 1.0


def swerve_ik(vx, vy, omega, data):
    """Drive the base actuators to realise the body-frame twist (vx, vy, omega)."""
    for key, r in C.MODULE_POSITIONS.items():
        # Rigid-body kinematics: v_i = V + omega x r_i
        v_ix = vx - omega * r[1]
        v_iy = vy + omega * r[0]
        v_mag = math.hypot(v_ix, v_iy)

        prefix = key.upper()
        steer_joint = f"{prefix} Steer"
        steer_act   = f"{prefix} Steer"
        drive_act   = f"{prefix} Drive"
        current_theta = data.joint(steer_joint).qpos[0]

        if v_mag > 1e-3:
            raw_target  = math.atan2(v_iy, v_ix)
            local_target = raw_target - C.MODULE_YAW_OFFSETS[key]
            theta, dir_mult = optimize_steering(local_target, current_theta)

            err = abs(theta - current_theta)
            drive_scale = 0.0 if err > C.STEER_DEADBAND else math.cos(err)

            data.actuator(steer_act).ctrl[0] = theta
            data.actuator(drive_act).ctrl[0] = (v_mag / C.WHEEL_RADIUS) * dir_mult * drive_scale
        else:
            data.actuator(steer_act).ctrl[0] = current_theta
            data.actuator(drive_act).ctrl[0] = 0.0

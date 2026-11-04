"""base_ik.py — swerve inverse kinematics + per-module calibration for MABEL.

Extracted VERBATIM (geometry, conventions, constants) from the proven
``ps5_teleop.py`` so the bridge's wire path to the Teensy is byte-identical to the
joystick path. NOTHING here is re-derived — change it only by changing ps5_teleop and
re-extracting.

Pipeline (the bridge owns this whole chain):

    body twist (vx, vy, wz)  ->  inverse_kinematics  ->  (drive, steer) per module
                                                     ->  apply_calibration  ->  Teensy

Conventions (robot body frame, same as ps5_teleop): +x forward, +y left, +z up; yaw
wz CCW-positive (right-hand rule about +z). Drive command is a linear wheel speed in
m/s; steer is in rotations [0,1). We emit the absolute wheel heading + a positive
speed and let the firmware do the steer-travel minimization / wheel-flip.

Module index order (per hardware): 0 = back, 1 = front-right, 2 = front-left.
"""

import math

import numpy as np

# --- per-module calibration (mechanical steer zero + drive polarity) ----------
# From ps5_teleop.py. Index order matches MODULE_POS below: [back, front-right,
# front-left]. The bridge applies these host-side when APPLY_HOST_CALIB is set (see
# config.py) so the swerve behaves exactly like the proven ps5_teleop path.
STEER_OFFSET = np.array([-0.1, -0.17, 0.1])
DRIVE_DIR = np.array([-1.0, -1.0, -1.0])


def _module_positions():
    """Module [x, y] offsets (m) from the base origin (triangle incenter).

    Geometry: 34 cm front track, 36 cm front-axle -> back-wheel. Origin = incenter
    (intersection of the triangle's angle bisectors). Frame: +x forward, +y left.
    """
    fl = np.array([0.0,  0.17])   # front-left
    fr = np.array([0.0, -0.17])   # front-right
    bk = np.array([-0.36, 0.0])   # back
    a = np.linalg.norm(fr - bk)   # side opposite front-left
    b = np.linalg.norm(fl - bk)   # side opposite front-right
    c = np.linalg.norm(fl - fr)   # side opposite back
    incenter = (a * fl + b * fr + c * bk) / (a + b + c)
    fl, fr, bk = fl - incenter, fr - incenter, bk - incenter
    # Module index order (per hardware): 0=back, 1=front-right, 2=front-left.
    return np.array([bk, fr, fl])


MODULE_POS = _module_positions()   # (3, 2): [x, y] of each module from origin


def inverse_kinematics(vx, vy, wz, prev_steer):
    """Body twist (vx, vy m/s, wz rad/s) -> per-module (drive m/s, steer rotations).

    Each module velocity is v_i = v_body + wz x r_i, i.e.
        v_ix = vx - wz * y_i
        v_iy = vy + wz * x_i
    drive_i = |v_i|; steer_i = atan2(v_iy, v_ix) wrapped to [0, 1).
    When a module's speed is ~0 the heading is undefined, so we hold its last steer.
    """
    drive = np.zeros(3)
    steer = np.array(prev_steer, dtype=float)
    for i, (x, y) in enumerate(MODULE_POS):
        vix = vx - wz * y
        viy = vy + wz * x
        speed = math.hypot(vix, viy)
        drive[i] = speed
        if speed > 1e-6:
            steer[i] = (math.atan2(viy, vix) / (2.0 * math.pi)) % 1.0
    return drive, steer


def apply_calibration(drive, steer):
    """Fold in per-module drive polarity + steer zero (host-side calibration).

    Returns the (drive, steer) arrays to hand straight to SwerveClient.setCommand.
    Mirrors ps5_teleop.apply_calibrated. Only call this when APPLY_HOST_CALIB is set;
    if the Teensy firmware applies calibration, send the raw geometric values instead
    (double-applying would mis-steer the wheels).
    """
    return DRIVE_DIR * np.asarray(drive), (STEER_OFFSET + np.asarray(steer)) % 1.0

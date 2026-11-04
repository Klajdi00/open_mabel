"""Mirror real-robot state into a MuJoCo model -- the input half of the ROS plant.

On the real robot the control loop runs the unchanged controller against a MuJoCo
*mirror*: every tick it copies the measured joint angles (from `/joint_states`,
`/arms/joint_states`, ...) and the base pose (from `/odom`) into `data.qpos`, calls
`mj_forward` (kinematics/Jacobians/CoM, NO physics step), runs `teleop.update()`, and
publishes the result via `command_extract`. This module is the pure, MuJoCo-only state
write-in (no ROS), so it is unit testable offline; it is the inverse of `command_extract`.

Joint names are accepted leniently -- URDF snake_case (what the sim and the hardware
drivers publish) OR the MuJoCo Title-Case name -- because the resolver is built FROM the
model (every 1-DOF joint registers under both spellings). Unknown names are skipped.
"""


def _urdfize(mj_name):
    # Same transform as mabel_sim.interface_config.urdfize / mabel.config naming:
    # MuJoCo "Left Arm 1" -> URDF "left_arm_1". Inlined so this module needs no cfg.
    return mj_name.lower().replace(" ", "_")


class JointMirror:
    """Resolves incoming joint names (URDF or MuJoCo) to `data.qpos` addresses, once,
    from the model. Reuse one instance per model; `apply()` is called every tick."""

    # MuJoCo joint type ids for the 1-DOF joints we mirror by name (hinge=3, slide=2).
    _ONE_DOF_TYPES = (2, 3)

    def __init__(self, model):
        self.model = model
        self._adr = {}            # accepted name -> qpos address
        for j in range(model.njnt):
            if int(model.jnt_type[j]) not in self._ONE_DOF_TYPES:
                continue          # free/ball joints are mirrored via mirror_base
            name = model.joint(j).name
            if not name:
                continue
            adr = int(model.jnt_qposadr[j])
            self._adr[name] = adr
            self._adr[_urdfize(name)] = adr

    def apply(self, data, names, positions):
        """Write measured positions into data.qpos for every recognised joint name.
        Returns the count written (for diagnostics). Unknown names are skipped."""
        n = 0
        for name, pos in zip(names, positions):
            adr = self._adr.get(name)
            if adr is not None:
                data.qpos[adr] = float(pos)
                n += 1
        return n


def mirror_base(model, data, x, y, z, quat_wxyz, base_free_joint="base_root"):
    """Write the base pose from /odom into the model's free joint (7-DOF qpos:
    3 translation + 4 quaternion, MuJoCo wxyz order). No-op if the model has no such
    joint (e.g. a fixed-base bench model)."""
    try:
        adr = int(model.joint(base_free_joint).qposadr[0])
    except (KeyError, ValueError, IndexError):
        return False
    data.qpos[adr + 0] = float(x)
    data.qpos[adr + 1] = float(y)
    data.qpos[adr + 2] = float(z)
    w, qx, qy, qz = quat_wxyz
    data.qpos[adr + 3] = float(w)
    data.qpos[adr + 4] = float(qx)
    data.qpos[adr + 5] = float(qy)
    data.qpos[adr + 6] = float(qz)
    return True

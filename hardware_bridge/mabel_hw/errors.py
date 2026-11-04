"""errors.py — the mabel_hw exception taxonomy.

Everything raised by this package derives from :class:`HardwareError`, so a consumer
can `except HardwareError` at the top of a control loop and keep the robot alive
(graceful degradation is a bridge invariant — a sick subsystem must never sink the HAL).
"""


class HardwareError(RuntimeError):
    """Base class for every mabel_hw error."""


class SubsystemUnavailable(HardwareError):
    """The subsystem is disabled, disconnected, or its device never enumerated."""


class GroupBusy(HardwareError):
    """claim() found another live writer on the actuator group (see arbitration)."""


class NotLeaseHolder(HardwareError):
    """A write arrived with a token that no longer holds the group's lease
    (released, expired, or stolen). The write was NOT sent to hardware."""


class InfeasiblePose(HardwareError):
    """A task-space target is outside the mechanism's feasible set
    (e.g. the neck RSSR linkage cannot reach the requested gaze)."""


class CalibrationError(HardwareError):
    """Calibration data missing, unparsable, or inconsistent with the joint set."""

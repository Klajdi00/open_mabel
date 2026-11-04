"""
Plant abstraction — the ONE seam between the controllers and the robot.

Every torque-domain controller in mabel.whole_body_controller talks to a
`Plant`: read joint pos/vel, apply per-joint effort, neutralise the stiff
position servo on a torque-managed joint. Swap the plant and the identical
controller code drives simulation or hardware:

    SimPlant   --- MuJoCo: efforts go to data.qfrc_applied (simulation, or the
                   kinematic mirror on the real-robot bridge).
    RealPlant  --- (to implement on hardware) read encoders, send CAN torque
                   commands to the DAMIAO / REV drivers, flush on commit().

begin()/commit() bracket one control tick so a hardware plant can batch its
bus writes (e.g. one CAN flush per tick).
"""
from abc import ABC, abstractmethod

from .dynamic_models.model import _dof


class Plant(ABC):
    @abstractmethod
    def q(self, joint): ...
    @abstractmethod
    def qd(self, joint): ...
    @abstractmethod
    def apply_effort(self, joint, tau): ...
    @abstractmethod
    def disable_servo(self, actuator): ...
    def begin(self): ...   # called once per tick before efforts are applied
    def commit(self): ...  # called once per tick after  (real: flush CAN)


class SimPlant(Plant):
    """MuJoCo plant: efforts go to data.qfrc_applied at the joint DOF; the stiff
    position servo on a managed actuator is neutralized by zeroing its gains."""

    def __init__(self, model, data):
        self.model, self.data = model, data

    def q(self, joint):  return float(self.data.joint(joint).qpos[0])
    def qd(self, joint): return float(self.data.qvel[_dof(self.model, joint)])

    def begin(self):
        self.data.qfrc_applied[:] = 0.0

    def apply_effort(self, joint, tau):
        self.data.qfrc_applied[_dof(self.model, joint)] = tau

    def disable_servo(self, actuator):
        """Zero a position actuator's gain & bias so it produces no force; the
        joint is then driven only by our applied effort (true compliance)."""
        aid = self.model.actuator(actuator).id
        self.model.actuator_gainprm[aid, :] = 0.0
        self.model.actuator_biasprm[aid, :] = 0.0

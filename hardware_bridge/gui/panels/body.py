"""body.py - Body control panel (DaMiao torso + 3-DoF neck, /body/command)."""

import contract
from .joint_panel import JointControlPanel


class BodyPanel(JointControlPanel):
    def __init__(self, parent, link):
        super().__init__(parent, link,
                         title="Body", subtitle="Torso + 3-DoF neck (DaMiao MIT). Targets "
                         "in radians -> /body/command on the bridge.",
                         joints=contract.BODY["joints"], state_key="body",
                         send=link.body_command)

"""arms.py - Arms control panel (14-DoF DaMiao, /arms/command)."""

import contract
from .joint_panel import JointControlPanel


class ArmsPanel(JointControlPanel):
    def __init__(self, parent, link):
        super().__init__(parent, link,
                         title="Arms", subtitle="14-DoF DaMiao (MIT). Targets in radians "
                         "-> /arms/command on the bridge.",
                         joints=contract.ARMS["joints"], state_key="arms",
                         send=link.arms_command)

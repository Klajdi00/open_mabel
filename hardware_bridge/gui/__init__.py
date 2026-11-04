"""gui - MABEL Control Center: one iOS-dark studio for the whole hardware stack.

A single GUI that combines the per-subsystem GUIs (orca hand, swerve base, lift, arms /
body) and drives them ALL through the hardware bridge's DDS contract - never touching a
device directly. See app.py for the shell and bridge_link.py for the DDS client.
"""

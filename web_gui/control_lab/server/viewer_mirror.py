"""
Native MuJoCo viewer mirror for the Control Lab — run under **mjpython**
(macOS requires it for the passive viewer):

    mjpython -m control_lab.server.viewer_mirror

Mirrors the studio's qpos memmap (config.STATE_FILE — the repo's simmirror
pattern) into a passive viewer at ~30 Hz, and ships the user's Ctrl+drag
perturbation BACK through the same file: double-click a body, then
Ctrl+drag (force) / Ctrl+Shift+drag (torque). The passive viewer writes the
perturbation into this process's data.xfrc_applied; we forward
[active, bodyid, wrench6] to the studio, which applies it to the live
physics every control tick. Closing the window exits cleanly.
"""
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from control_lab.server import config as CFG                  # noqa: E402


def main():
    if not CFG.STATE_FILE:
        print("viewer_mirror: MABEL_CTS_STATE_FILE disabled — nothing to do")
        return
    model = mujoco.MjModel.from_xml_path(str(CFG.SCENE))
    data = mujoco.MjData(model)

    # wait for the studio to create the state file
    for _ in range(120):
        try:
            if os.path.getsize(CFG.STATE_FILE) >= 8 * (1 + model.nq + 8):
                break
        except OSError:
            pass
        time.sleep(0.5)
    else:
        print(f"viewer_mirror: no state file at {CFG.STATE_FILE} — "
              "is the studio running?")
        return
    mm = np.memmap(CFG.STATE_FILE, dtype=np.float64, mode="r+",
                   shape=(1 + model.nq + 8,))
    po = 1 + model.nq
    print("viewer_mirror: double-click a body, Ctrl+drag = force, "
          "Ctrl+Shift+drag = torque — applied to the LIVE studio physics")

    with mujoco.viewer.launch_passive(model, data,
                                      show_left_ui=False,
                                      show_right_ui=False) as v:
        while v.is_running():
            data.qpos[:] = mm[1:po]
            data.time = float(mm[0])
            mujoco.mj_forward(model, data)
            v.sync()
            # forward the viewer's perturbation to the studio, then clear it
            # locally so a released drag reads back as zero next frame
            rows = np.flatnonzero(np.abs(data.xfrc_applied).sum(axis=1) > 1e-9)
            if rows.size:
                bid = int(rows[0])
                mm[po + 1] = float(bid)
                mm[po + 2:po + 8] = data.xfrc_applied[bid]
                mm[po] = 1.0
                data.xfrc_applied[:] = 0.0
            else:
                mm[po] = 0.0
            time.sleep(1.0 / 30.0)
    mm[po] = 0.0


if __name__ == "__main__":
    main()

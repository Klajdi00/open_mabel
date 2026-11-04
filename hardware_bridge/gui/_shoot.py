"""_shoot.py - render the control center on :0 and screenshot every tab.

The :0 session is compositor-managed (root grabs read black), so we capture each tab by
reading the GUI window's OWN pixmap with python-Xlib XGetImage (works under the
compositor). Optionally starts the bridge (right hand, FREE-DRIVE) for live data when
$SHOOT_BRIDGE=1. Run under the mabel ROS env. Output dir via $OUT.
"""
import os, sys, time, signal, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.environ.get("OUT", "/tmp")
DISPLAY = os.environ.get("DISPLAY", ":0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
os.environ.setdefault("ROS_DOMAIN_ID", "0")

def log(*a): print(*a, flush=True)

sys.path.insert(0, BR)   # so `config` / `subsystems` / `hands` import

# In-process bridge: build the hand subsystem ROS node IN THIS PROCESS (no subprocess -
# detached/background children get killed in this env) so it publishes /<side>_hand/*
# that the GUI's BridgeLink reads back over DDS loopback. Right hand, FREE-DRIVE.
hands_node = bridge_exec = None
if os.environ.get("SHOOT_INPROC") == "1":
    import threading
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    if not rclpy.ok():
        rclpy.init()
    from mabel_hw import config
    from mabel_hw.ros import hands_node as hands_subsystem
    cfg = dict(config.HANDS)
    cfg.update(enabled=True, left_port="none", right_port="/dev/ttyACM0",
               auto_enable_torque=False)
    hands_node = hands_subsystem.build(cfg)
    bridge_exec = MultiThreadedExecutor()
    bridge_exec.add_node(hands_node)
    threading.Thread(target=bridge_exec.spin, daemon=True).start()
    log("in-process hand subsystem (right) spinning")
    time.sleep(3)

import app as A
from Xlib import X, display
from PIL import Image

_d = display.Display(DISPLAY)

root = A.App()
root.geometry("1180x820+40+40")
root.update(); root.update_idletasks()
root.lift(); root.update()
# wait for first hand telemetry if asked (bridge started externally as a & child)
_wait = float(os.environ.get("SHOOT_WAIT", "1.5"))
_t0 = time.time()
while time.time() - _t0 < _wait:
    root.update(); root.update_idletasks(); time.sleep(0.05)
    snap = root.link.snapshot()
    if snap["hands"]["right"]["joints"] or snap["hands"]["left"]["joints"]:
        log("telemetry arrived after", round(time.time() - _t0, 1), "s")
        break

def grab(wid, w, h, path):
    win = _d.create_resource_object("window", wid)
    raw = win.get_image(0, 0, w, h, X.ZPixmap, 0xffffffff)
    Image.frombytes("RGB", (w, h), raw.data, "raw", "BGRX").save(path)

def shoot(name):
    root._switch(name)
    if name == "Hands":                       # show the connected (right) hand
        try:
            hp = root._panels.get("Hands")
            view = hp._views.get(hp._active)
            view._side_seg.set("right"); view._switch_side("right")
        except Exception as e:
            log("side switch failed", repr(e))
    for _ in range(45):
        root.update(); root.update_idletasks(); time.sleep(0.04)
    root.update()
    w, h = root.winfo_width(), root.winfo_height()
    path = os.path.join(OUT, f"gui_{name.lower()}.png")
    grab(root.winfo_id(), w, h, path)
    log(f"shot {name:9} {w}x{h} -> {path}")

for t in ["Overview", "Base", "Arms", "Body", "Hands", "Lift", "Monitor"]:
    try:
        shoot(t)
    except Exception as e:
        log("shoot failed", t, repr(e))

try: root._on_close()
except Exception: pass
if hands_node is not None:
    try: bridge_exec.shutdown()
    except Exception: pass
    try: hands_node.destroy_node()   # releases the serial port (free-drive)
    except Exception: pass
log("DONE")

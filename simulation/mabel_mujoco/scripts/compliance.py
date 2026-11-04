"""
MABEL compliance — the "Soft" feel from the iOS / Vision Pro apps.

Drives the SAME WBCEngine the headset bridge uses, so testing here validates the
exact law that ships to the apps and the robot. Push ANYWHERE on the arms (Ctrl +
drag a body in the viewer): the managed joints yield (gravity-compensated, back-
drivable), and with --region whole the mobile base admits the net wrench and
glides, then coasts to rest when you let go.

This is one of MABEL's two controllers (the other is scripts/teleop.py, the
impedance wrist-tracking "Hold" feel). It mirrors the app's two pickers:

  Reach (--region)       how much of the body gives, lift always height-servoed:
    arm        arms only            (base + torso held)
    upper      arms + torso + lift  (base held)
    whole      arms + torso + base  (base glides under the net wrench)

  Live keyboard control is read from THIS TERMINAL (no need to click the 3D
  window): Tab or Space cycles the compliant region (arm -> upper -> whole ->
  arm) — the servos reconfigure on the fly and the arm holds where it is (no
  snap); C toggles the self-collision soft wall (same as --self-collision, but
  flippable while dragging); q quits. The viewer's left UI panel is kept on
  (Tab is mujoco's built-in toggle for it, so we re-assert it every frame).

  Estimator (--estimator)  where the base-driving wrench comes from:
    sim        read straight from the simulator (the viewer drag)   [default]
    torque     estimated from joint-torque feedback (the hardware path: a
               momentum observer -> J^T pseudo-inverse -> chassis wrench), so
               the SAME pipeline runs unchanged on the real robot.

--stiffness K springs back to the held pose with K [Nm/rad] (the app's stiffness
slider) instead of being fully back-drivable (K=0, the default).

Run (macOS needs mjpython for the viewer):
  mjpython scripts/compliance.py --region arm
  mjpython scripts/compliance.py --region whole
  mjpython scripts/compliance.py --region whole --estimator torque
"""
import os
import sys
import time
import math
import atexit
import argparse
import mujoco
import mujoco.viewer
try:
    import termios, tty, select        # POSIX raw-terminal keyboard input
    _HAS_TERMIOS = True
except ImportError:                     # non-POSIX (Windows): fall back to viewer keys
    _HAS_TERMIOS = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "controller"))  # mabel package lives here
from mabel import config as C
from mabel.wbc_runtime import WBCEngine
from mabel.admittance import external_base_wrench
from mabel.force_estimation import ContactEstimator
from mabel.dashboard import Dashboard

REGIONS = {"arm": "arm", "upper": "upper_body", "whole": "whole_body"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=REGIONS, default="arm",
                    help="how much of the robot is compliant (app's Reach picker)")
    ap.add_argument("--estimator", choices=("sim", "torque"), default="sim",
                    help="base wrench source: sim (viewer drag) or torque (hardware path)")
    ap.add_argument("--stiffness", type=float, default=0.0, metavar="K",
                    help="spring back to the held pose with K [Nm/rad] (0 = back-drivable)")
    ap.add_argument("--self-collision", action="store_true",
                    help="add the self-collision soft wall: a dragged arm stops at the "
                         "safety margin instead of pushing through itself (default OFF)")
    args = ap.parse_args()
    region = REGIONS[args.region]

    xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    # The scene's 360-beam rangefinder array (simulated LiDAR) ray-casts every step
    # and dominates the cost (~6.8x slower). Compliance reads joint state straight
    # from `data`, never sensordata, so disable sensors for a real-time viewer.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep

    eng = WBCEngine(model, data)
    ctrl_type = "impedance" if args.stiffness > 0 else "compliance"
    eng.set_stiffness(args.stiffness)
    eng.set_self_collision(args.self_collision)            # OFF unless --self-collision
    eng.enter(ctrl_type, region)
    # The hardware path estimates the base wrench from joint torque instead of
    # reading the simulator; the viewer drag is then just the physical push.
    est = ContactEstimator(model, data, gain=35.0) if args.estimator == "torque" else None
    dash = Dashboard(lines=12)

    # Keyboard control is read from THIS TERMINAL (stdin), not the viewer window,
    # so you can keep the dashboard in view and never click into the 3D scene.
    # Each key only FLAGS a request; the sim loop applies it via eng.enter() /
    # eng.toggle_self_collision() (which edit servo gains / build the guard) so the
    # mutation happens on the stepping thread, never mid-step.
    #   Tab / Space : cycle the compliant region (arm -> upper -> whole -> arm)
    #   C           : toggle the self-collision soft wall
    #   q           : quit
    region_keys = list(REGIONS)                  # ["arm", "upper", "whole"]
    cur = {"key": args.region}                   # the live region key
    pending = {"key": None}                       # next region requested by a key
    sc_pending = {"toggle": False}                # self-collision toggle requested
    quit_req = {"v": False}
    seen_keys = set()                             # report any other key once (diag)

    def cycle_region():
        i = region_keys.index(cur["key"])
        pending["key"] = region_keys[(i + 1) % len(region_keys)]

    # Viewer key fallback — used ONLY when stdin is not a TTY (piped / IDE run),
    # so a normal terminal session is driven purely by the stdin reader below.
    GLFW_KEY_TAB, GLFW_KEY_SPACE, GLFW_KEY_C = 258, 32, ord("C")
    def key_callback(keycode):
        if keycode in (GLFW_KEY_TAB, GLFW_KEY_SPACE):
            cycle_region()
        elif keycode == GLFW_KEY_C:
            sc_pending["toggle"] = True

    def region_blurb(key):
        return {"arm":   "arms only — base + torso + lift held",
                "upper": "arms + torso + lift free — base held",
                "whole": "arms + torso + lift free — base glides"}[key]

    # Put the terminal in cbreak mode: single keypresses arrive immediately (no
    # Enter, no echo into the dashboard); Ctrl-C still works (ISIG stays on).
    # atexit restores it on any exit path (normal, exception, Ctrl-C).
    stdin_fd = sys.stdin.fileno()
    tty_ok = _HAS_TERMIOS and sys.stdin.isatty()
    if tty_ok:
        _old_term = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
        atexit.register(termios.tcsetattr, stdin_fd, termios.TCSADRAIN, _old_term)

    def poll_terminal():
        while tty_ok and select.select([stdin_fd], [], [], 0)[0]:
            ch = os.read(stdin_fd, 1).decode(errors="ignore")
            if ch in ("\t", " "):
                cycle_region()
            elif ch in ("c", "C"):
                sc_pending["toggle"] = True
            elif ch == "q":
                quit_req["v"] = True
            elif ch.isprintable() and ch not in seen_keys:
                seen_keys.add(ch)
                print(f"[key {ch!r}] — not bound; Tab/Space = region, C = self-collision, q = quit")

    def force_left_panel(viewer):
        # Tab is mujoco's built-in toggle for the left UI panel; re-assert it ON
        # each frame so it never disappears (best-effort; private Handle field).
        try:
            sim = viewer._sim()
            if sim is not None:
                sim.ui0_enable = True
        except Exception:
            pass

    print(f"--- MABEL compliance [{ctrl_type} | region={args.region} | wrench={args.estimator}] ---")
    print(f"Ctrl + drag any arm body to push it.  ({region_blurb(args.region)})")
    print("Keys in THIS terminal:  Tab/Space = cycle region (arm -> upper -> whole),"
          "  C = self-collision,  q = quit.\n")

    fx = fy = mz = 0.0
    last_ui = 0.0
    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=None if tty_ok else key_callback,
                                      show_left_ui=True, show_right_ui=False) as viewer:
        t0 = time.time()
        while viewer.is_running():
            poll_terminal()
            if quit_req["v"]:
                break
            force_left_panel(viewer)
            if pending["key"] is not None:
                cur["key"] = pending["key"]
                pending["key"] = None
                eng.enter(ctrl_type, REGIONS[cur["key"]])   # reconfigures servos, no snap
                print(f"[region] -> {cur['key'].upper()}  ({region_blurb(cur['key'])})")
            if sc_pending["toggle"]:
                sc_pending["toggle"] = False
                on = eng.toggle_self_collision()            # builds the guard on first ON
                print(f"[self-collision] {'ON (soft wall)' if on else 'OFF'}")
            target = time.time() - t0
            if target - data.time > 0.1:
                t0 = time.time() - data.time
            while data.time < target:
                if est is not None:
                    # Observer fed last tick's applied torque (qfrc_applied set by
                    # the engine's efforts, qfrc_actuator by the lift/torso servos).
                    tau_cmd = data.qfrc_applied + data.qfrc_actuator
                    fx, fy, mz = est.update(tau_cmd, dt)
                    eng.step(dt, wrench=(fx, fy, mz))
                else:
                    eng.step(dt)
                mujoco.mj_step(model, data)

            if time.time() - last_ui > 0.1:
                if est is None:
                    fx, fy, mz = external_base_wrench(model, data, eng.arm_bodies)
                where = (", ".join(c["body"] for c in est.contacts) or "none") if est else "sim drag"
                dash.render(dash.header("MABEL COMPLIANCE — SOFT") + [
                    dash.line("FEEL", f"{ctrl_type.upper()} K={eng.stiffness:.0f}", "yellow"),
                    dash.line("REGION", f"{cur['key'].upper()}  (Tab/Space to cycle)", "cyan"),
                    dash.line("WRENCH |f|", f"{math.hypot(fx, fy):5.1f} N (fx {fx:5.1f} fy {fy:5.1f})", "cyan"),
                    dash.line("MOMENT mz", f"{mz:5.1f} Nm", "cyan"),
                    dash.line("CONTACT ON", where, "green"),
                    dash.line("BASE vx/vy/w", f"{eng.vx:5.2f} {eng.vy:5.2f} {eng.w:5.2f}"),
                    dash.line("SELF-COLLISION",
                              ("ON (soft wall)" if eng.self_collision else "OFF") + "  (C to toggle)",
                              "yellow" if eng.self_collision else "green"),
                ])
                last_ui = time.time()
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()

"""
MABEL teleop — the impedance wrist-tracking "Hold" feel from the iOS / Vision Pro apps.

6-DOF wrist targets -> bimanual SE(3) whole-body IK (8 DOF/arm) with lazy lift /
torso / base followers that creep to keep the hands reachable. This is one of
MABEL's two controllers (the other is scripts/compliance.py, the "Soft" feel).
Both consumers below drive the SAME shipped TeleopController, so testing here
validates the exact stack that ships to the apps and the robot.

Pick where the wrist targets come from (--source):

  keyboard   jog both wrists from the keyboard (sim stand-in, no headset)  [default]
  mock       synthetic wrists tracing a slow circle (smoke-test the stream)
  avp        live 6-DOF wrists + gaze from an Apple Vision Pro (the real bridge)

Four modes (the iOS app's NAV + Reach axis); TAB cycles them, 1/2/3/4 select directly:
  1 NAVIGATION  : drive the swerve base + lift; arms hold their pose
  2 ARM         : wrist tracking, arms only (lift/torso/base held)
  3 UPPER BODY  : wrist tracking + lift/torso follow to extend reach
  4 WHOLE BODY  : wrist tracking + lift/torso + mobile base follow

Keyboard controls
  NAV mode      : ↑/↓ = drive fwd/back | ←/→ = strafe | W/S = turn | R/F = lift up/down
  MANIP modes (ARM/UPPER/WHOLE):
    LEFT  wrist pos : W/S = ±X | A/D = ±Y | R/F = ±Z
    RIGHT wrist pos : ↑/↓ = ±X | ←/→ = ±Y | T/G = ±Z
    WRIST rotation  : Z/X = roll | C/V = pitch | B/N = yaw  (active wrist)
    SPACE           : toggle active wrist (LEFT / RIGHT)
  ARM STIFFNESS     : -/_ = softer (more back-drivable) | =/+ = stiffer
  BACKSPACE         : reset the simulation to the start pose (any mode)

The robot settles to gravity equilibrium on launch and holds perfectly still until a
key is pressed — the reach-assist engages only while you are actively commanding.

avp / mock controls
  C : (re)calibrate the clutch at the operator's current pose   TAB/1-4 : switch mode

Run (macOS needs mjpython for the viewer):
  mjpython scripts/teleop.py                       # keyboard
  mjpython scripts/teleop.py --source mock
  mjpython scripts/teleop.py --source avp --addr tcp://192.168.1.50:5556
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pynput import keyboard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "controller"))  # mabel package lives here
from mabel import config as C
from mabel.teleop_controller import TeleopController
from mabel.avp_receiver import AVPReceiver
from mabel.swerve import swerve_ik
from mabel.dashboard import Dashboard
from mabel import mock_avp

POS_SPEED = 0.5    # m/s   wrist translation rate at full key press
ROT_SPEED = 1.0    # rad/s wrist rotation rate at full key press
MOVE_CHARS = set("wsadrftgzxcvbn")              # any of these = active wrist command
ARROW_KEYS = {keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}
# The four operator modes (NAV + the three Reach regions), in TAB-cycle order.
MODES = ["navigation", "arm", "upper", "whole"]
MODE_LABEL = {"navigation": "NAVIGATION", "arm": "ARM CONTROL",
              "upper": "UPPER BODY", "whole": "WHOLE BODY"}

STIFF_MIN, STIFF_MAX = 0.25, 2.5                 # arm stiffness multiplier bounds
STIFF_STEP = 1.25                                # per-keypress multiplicative step

pressed = set()
active = "L"               # which wrist the rotation keys steer (keyboard source)
mode_idx = MODES.index("whole")                 # start in whole-body (full reach assist)
stiffness = 1.0           # arm position-servo stiffness multiplier (1.0 = nominal)
events = {"calibrate": False, "reset": False, "stiffness": True, "toggle_sc": False}


def on_press(key):
    global active, mode_idx, stiffness
    try:
        ch = key.char.lower()
        if ch == 'c': events["calibrate"] = True
        elif ch == 'o': events["toggle_sc"] = True          # self-collision on/off
        elif ch in "1234": mode_idx = int(ch) - 1          # direct mode select
        elif ch in "-_":                                    # softer arms
            stiffness = max(STIFF_MIN, stiffness / STIFF_STEP); events["stiffness"] = True
        elif ch in "=+":                                    # stiffer arms
            stiffness = min(STIFF_MAX, stiffness * STIFF_STEP); events["stiffness"] = True
        pressed.add(ch)
    except AttributeError:
        if key == keyboard.Key.tab and key not in pressed:
            mode_idx = (mode_idx + 1) % len(MODES)          # cycle modes
        elif key == keyboard.Key.space and key not in pressed:
            active = "R" if active == "L" else "L"          # toggle active wrist
        elif key == keyboard.Key.backspace and key not in pressed:
            events["reset"] = True                          # reset the simulation
        pressed.add(key)


def on_release(key):
    try: pressed.discard(key.char.lower())
    except AttributeError: pressed.discard(key)


def rotate_mat(mat, axis, angle):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)) @ mat


def keyboard_step(ctrl, dt, mode):
    """Dispatch the keyboard control for the active mode (the same IK + lazy
    followers the controller uses live; region gated by `mode`)."""
    if mode == "navigation":
        nav_step(ctrl, dt)
    else:
        manip_step(ctrl, dt, region=mode)
    ctrl._update_markers()


def nav_step(ctrl, dt):
    """NAVIGATION: drive the swerve base + jog the lift; arms hold their pose."""
    ctrl.ik.sync_to_current()                      # keep arms where they are
    tvx = tvy = tw = 0.0
    if keyboard.Key.up in pressed:    tvx -= C.MAX_LIN_SPEED
    if keyboard.Key.down in pressed:  tvx += C.MAX_LIN_SPEED
    if keyboard.Key.left in pressed:  tvy -= C.MAX_LIN_SPEED
    if keyboard.Key.right in pressed: tvy += C.MAX_LIN_SPEED
    if 'w' in pressed: tw += C.MAX_ANG_SPEED
    if 's' in pressed: tw -= C.MAX_ANG_SPEED
    ctrl.vx, ctrl.vy, ctrl.omega = C.safe_base_step(
        ctrl.vx, ctrl.vy, ctrl.omega, tvx, tvy, tw, dt, float(ctrl.data.actuator("Lift").ctrl[0]))
    swerve_ik(ctrl.vx, ctrl.vy, ctrl.omega, ctrl.data)
    if 'r' in pressed: _integrate(ctrl, "Lift",  C.LIFT_SPEED, dt)
    if 'f' in pressed: _integrate(ctrl, "Lift", -C.LIFT_SPEED, dt)


def manip_step(ctrl, dt, region):
    """ARM / UPPER / WHOLE: jog the wrist targets, solve the arms, and let the
    reach-assist follow per region (arm: none, upper: lift+torso, whole: +base)."""
    ik = ctrl.ik
    s, rs = POS_SPEED, ROT_SPEED
    if 'w' in pressed: ik.target_l_pos[0] -= s * dt
    if 's' in pressed: ik.target_l_pos[0] += s * dt
    if 'a' in pressed: ik.target_l_pos[1] -= s * dt
    if 'd' in pressed: ik.target_l_pos[1] += s * dt
    if 'r' in pressed: ik.target_l_pos[2] += s * dt
    if 'f' in pressed: ik.target_l_pos[2] -= s * dt
    if keyboard.Key.up in pressed:    ik.target_r_pos[0] -= s * dt
    if keyboard.Key.down in pressed:  ik.target_r_pos[0] += s * dt
    if keyboard.Key.left in pressed:  ik.target_r_pos[1] -= s * dt
    if keyboard.Key.right in pressed: ik.target_r_pos[1] += s * dt
    if 't' in pressed: ik.target_r_pos[2] += s * dt
    if 'g' in pressed: ik.target_r_pos[2] -= s * dt
    for k, (ax, sgn) in {'z': ([1, 0, 0], rs), 'x': ([1, 0, 0], -rs),
                         'c': ([0, 1, 0], rs), 'v': ([0, 1, 0], -rs),
                         'b': ([0, 0, 1], rs), 'n': ([0, 0, 1], -rs)}.items():
        if k in pressed:
            if active == 'L': ik.target_l_mat = rotate_mat(ik.target_l_mat, ax, sgn * dt)
            else:             ik.target_r_mat = rotate_mat(ik.target_r_mat, ax, sgn * dt)

    ik.apply(ik.solve(arms_only=True), dt, arms_only=True)
    # Reach-assist only while a key is held, scoped by region; idle => hold.
    commanding = bool((pressed & MOVE_CHARS) or (pressed & ARROW_KEYS))
    tvx = tvy = tw = 0.0
    if commanding and region in ("upper", "whole"):
        _integrate(ctrl, "Lift",  ctrl.lazy_lift.command(ik.target_l_pos, ik.target_r_pos, ctrl.data), dt)
        _integrate(ctrl, "Torso", ctrl.lazy_torso.command(ik.target_l_pos, ik.target_r_pos, ctrl.data), dt)
    if commanding and region == "whole":
        tvx, tvy, tw = ctrl.lazy_base.command(ik.target_l_pos, ik.target_r_pos, ctrl.data)
    ctrl.vx, ctrl.vy, ctrl.omega = C.safe_base_step(
        ctrl.vx, ctrl.vy, ctrl.omega, tvx, tvy, tw, dt, float(ctrl.data.actuator("Lift").ctrl[0]))
    swerve_ik(ctrl.vx, ctrl.vy, ctrl.omega, ctrl.data)


def _integrate(ctrl, name, vel, dt):
    if abs(vel) < 1e-9:
        return
    aid = ctrl.model.actuator(name).id
    lo, hi = ctrl.model.actuator_ctrlrange[aid]
    cur = float(ctrl.data.actuator(name).ctrl[0])
    ctrl.data.actuator(name).ctrl[0] = float(np.clip(cur + vel * dt, lo, hi))


def snapshot_arm_gains(model):
    """Nominal kp/kv of the arm + wrist position servos, for live stiffness scaling.
    The lift and torso keep their fixed (strong) gains; only the arms are tuned."""
    ids = np.array([model.actuator(n).id for n in C.IK_ACTUATORS[2:]])  # 7 arm + wrist/side
    return ids, model.actuator_gainprm[ids].copy(), model.actuator_biasprm[ids].copy()


def apply_stiffness(model, ids, gain0, bias0, factor):
    """Scale the arm servos: kp -> factor*kp, kv -> sqrt(factor)*kv (constant damping
    ratio, so it stays stable). factor<1 = softer/back-drivable, >1 = stiffer."""
    model.actuator_gainprm[ids, 0] = gain0[:, 0] * factor          # kp
    model.actuator_biasprm[ids, 1] = bias0[:, 1] * factor          # -kp (position)
    model.actuator_biasprm[ids, 2] = bias0[:, 2] * np.sqrt(factor) # -kv (damping)


def print_banner(source):
    """One-time controls legend printed above the live dashboard."""
    B, Y, C0, E = "\033[1m", "\033[93m", "\033[96m", "\033[0m"
    print(f"\n{B}{C0}MABEL teleop — keyboard controls{E}  (source: {source})")
    print(f"  {Y}SWITCH MODE{E}  TAB = cycle   |   1 NAVIGATION  2 ARM  3 UPPER BODY  4 WHOLE BODY")
    print(f"  {Y}SWITCH WRIST{E} SPACE = toggle which wrist the rotation keys steer (LEFT/RIGHT)")
    print(f"  {Y}NAVIGATION{E}   arrows = drive (↑↓ fwd/back, ←→ strafe) · W/S = turn · R/F = lift")
    print(f"  {Y}ARM/UPPER/WHOLE{E}  L wrist: W/S/A/D + R/F   |   R wrist: arrows + T/G   |   rotate: Z/X C/V B/N")
    print(f"  {Y}ARM STIFFNESS{E} -/_ = softer (back-drivable)   =/+ = stiffer")
    print(f"  {Y}SELF-COLLISION{E} O = toggle stop-at-margin avoidance (default OFF)")
    print(f"  {Y}RESET{E}        BACKSPACE = reset the simulation to the start pose")
    if source != "keyboard":
        print(f"  {Y}CALIBRATE{E}    C = anchor the clutch to the operator's current pose")
    print()


def control_rows(dash, mode, source):
    """Mode-aware control hints, always visible in the live dashboard."""
    rows = [dash.line("SWITCH MODE", "TAB cycle | 1 nav  2 arm  3 upper  4 whole", "blue")]
    if mode == "navigation":
        rows.append(dash.line("DRIVE", "arrows = move | W/S = turn | R/F = lift", "blue"))
    else:
        rows.append(dash.line("MOVE WRIST", "L: WASD+RF | R: arrows+TG | rot: ZXCVBN", "blue"))
        rows.append(dash.line("SWITCH WRIST", "SPACE  (which wrist the ZXCVBN keys rotate)", "blue"))
    rows.append(dash.line("ARM STIFFNESS", "-/_ softer   =/+ stiffer", "blue"))
    rows.append(dash.line("SELF-COLLISION", "O  (toggle stop-at-margin avoidance)", "blue"))
    rows.append(dash.line("RESET", "BACKSPACE  (reset the simulation)", "blue"))
    if source != "keyboard":
        rows.append(dash.line("CALIBRATE", "C  (anchor to operator pose)", "blue"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("keyboard", "mock", "avp"), default="keyboard",
                    help="where the wrist targets come from")
    ap.add_argument("--addr", default=C.AVP_ZMQ_ADDR, help="AVP ZMQ publisher address")
    ap.add_argument("--self-collision", action="store_true",
                    help="start with self-collision avoidance ON (toggle live with 'o'); "
                         "default OFF — plain teleop")
    args = ap.parse_args()

    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.SCENE_XML)
    model = mujoco.MjModel.from_xml_path(xml)
    # The scene's 360-beam rangefinder array (simulated LiDAR) ray-casts every step
    # and dominates the cost (~6.8x slower). Teleop reads joint state straight from
    # `data`, never sensordata, so disable sensors for a real-time viewer.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    ctrl = TeleopController(model, data, base_follow=True)
    ctrl.set_self_collision(args.self_collision)          # OFF unless --self-collision
    stiff_ids, gain0, bias0 = snapshot_arm_gains(model)   # nominal arm servo gains
    ctrl.settle()              # settle to gravity equilibrium so it doesn't creep on launch
    recv = AVPReceiver(args.addr) if args.source == "avp" else None
    print_banner(args.source)  # controls legend, stays above the live dashboard
    dash = Dashboard(lines=20)
    last_ui = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            frame = None
            if args.source == "mock":
                frame = mock_avp.mock_frame(time.time() - t0)
            elif args.source == "avp":
                frame = recv.poll()
            if events["calibrate"]:
                events["calibrate"] = False
                if frame is not None:                       # calibrate is avp/mock only
                    ctrl.calibrate(frame)
            if events["reset"]:                             # Backspace -> reset the sim
                events["reset"] = False
                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                ctrl.settle()                               # back to the clean settled pose
                t0 = time.time() - data.time                # resync the real-time clock
            if events["stiffness"]:                         # -/= -> rescale arm servos
                events["stiffness"] = False
                apply_stiffness(model, stiff_ids, gain0, bias0, stiffness)
            if events["toggle_sc"]:                         # 'o' -> self-collision on/off
                events["toggle_sc"] = False
                ctrl.toggle_self_collision()

            # Push the operator's selected mode onto the shared controller (so the
            # avp/mock path obeys the same NAV / region choice as the keyboard path).
            mode = MODES[mode_idx]
            if mode == "navigation":
                ctrl.set_mode(ctrl.MODE_NAV)
            else:
                ctrl.set_mode(ctrl.MODE_MANIP); ctrl.set_region(mode)

            if time.time() - last_ui > 0.1:
                st = ctrl.status()
                rows = dash.header(f"MABEL TELEOP — wrist tracking [{args.source}]") + [
                    dash.line("MODE", MODE_LABEL[mode], "yellow"),
                    dash.line("ARM STIFFNESS", f"{stiffness:.2f}x  ({int(stiffness*100)}% of nominal)",
                              "yellow" if abs(stiffness - 1.0) > 1e-3 else "green"),
                    dash.line("base vx/vy/w", f"{st['base'][0]:5.2f} {st['base'][1]:5.2f} {st['base'][2]:5.2f}"),
                    dash.line("LIFT", f"{float(data.actuator('Lift').ctrl[0]):.3f} m"),
                    dash.line("L/R wrist err", f"{st['palm_err_l']:.3f} / {st['palm_err_r']:.3f} m"),
                    dash.line("SELF-COLLISION", "ON (stop-at-margin)" if st["self_collision"] else "OFF",
                              "yellow" if st["self_collision"] else "green"),
                ]
                if args.source == "keyboard":
                    if mode != "navigation":
                        rows.append(dash.line("ACTIVE WRIST", "LEFT" if active == "L" else "RIGHT", "cyan"))
                else:
                    rows.append(dash.line("STATUS", "CALIBRATED" if st["calibrated"] else "press C to calibrate",
                                          "green" if st["calibrated"] else "yellow"))
                    rows.append(dash.line("STREAM", "ok" if frame is not None else "waiting...",
                                          "green" if frame is not None else "red"))
                rows += control_rows(dash, mode, args.source)   # always-visible key legend
                dash.render(rows)
                last_ui = time.time()

            target = time.time() - t0
            if target - data.time > 0.1:
                t0 = time.time() - data.time; target = data.time

            while data.time < target:
                dt = model.opt.timestep
                if args.source == "keyboard":
                    keyboard_step(ctrl, dt, mode)
                else:
                    ctrl.step(frame, None, dt)
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)

    if recv is not None:
        recv.close()


if __name__ == "__main__":
    main()

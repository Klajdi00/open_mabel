"""
MABEL MuJoCo Viewer
====================
Simple interactive viewer for the MABEL digital twin.
Supports keyboard shortcuts for common robot configurations.

Usage:
    cd simulation/mabel_mujoco
    python scripts/viewer.py                    # Full robot
    python scripts/viewer.py --model scenes/mabel_full.xml
    python scripts/viewer.py --model components/left_hand.xml

Controls (in MuJoCo viewer window):
    Space        : Pause / Resume simulation
    Backspace    : Reset to keyframe 0
    Tab          : Cycle through cameras
    Ctrl+R       : Revert to initial pose
    Q            : Quit

    Custom keyboard (via script):
    H            : Move robot to HOME pose (arms neutral)
    G            : Open both hands (fingers extended)
    F            : Close both hands (fist)
    L            : Raise lift to max
    D            : Lower lift to min

Requirements:
    pip install mujoco numpy
"""

import argparse
import time
import numpy as np
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("MuJoCo not installed. Run: pip install mujoco")
    raise

# ── Path resolution ────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO_DIR    = SCRIPT_DIR.parent
DEFAULT_XML = REPO_DIR / "scenes" / "mabel_sim.xml"


# ── Pose presets (actuator names match models/mabel_full.xml) ────────────────────

def set_home_pose(model, data):
    """Lift at mid-height, torso upright, neck centred."""
    data.ctrl[:] = 0.0
    _set_ctrl(model, data, "Lift",  0.20)
    _set_ctrl(model, data, "Torso", 0.0)
    _set_ctrl(model, data, "Neck Yaw",   0.0)
    _set_ctrl(model, data, "Neck Pitch", 0.0)
    _set_ctrl(model, data, "Neck Roll",  0.0)
    print("[POSE] HOME")


def open_hands(model, data):
    """Extend all fingers to open position."""
    for side in ("Left", "Right"):
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            _set_ctrl(model, data, f"{side} {finger} MCP", 0.0)
            _set_ctrl(model, data, f"{side} {finger} PIP", 0.0)
    print("[POSE] HANDS OPEN")


def close_hands(model, data):
    """Curl all fingers to a fist."""
    for side in ("Left", "Right"):
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            _set_ctrl(model, data, f"{side} {finger} MCP", 1.4)
            _set_ctrl(model, data, f"{side} {finger} PIP", 1.6)
    print("[POSE] HANDS CLOSED (fist)")


def lift_up(model, data):
    """Raise lift to max extension."""
    _set_ctrl(model, data, "Lift", 0.635)
    print("[POSE] LIFT UP")


def lift_down(model, data):
    """Lower lift to minimum."""
    _set_ctrl(model, data, "Lift", 0.00)
    print("[POSE] LIFT DOWN")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _set_ctrl(model, data, name: str, value: float):
    """Set a named actuator control, silently skipping if not found."""
    try:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx >= 0:
            lo, hi = model.actuator_ctrlrange[idx]
            data.ctrl[idx] = float(np.clip(value, lo, hi))
    except Exception:
        pass  # gracefully skip for partial models


def print_info(model):
    """Print robot model statistics."""
    print("\n╔══════════════════════════════════════╗")
    print(f"║  Model: {model.name[:30]:<30}  ║")
    print(f"║  DOF (nv):     {model.nv:<5}                   ║")
    print(f"║  Bodies (nb):  {model.nbody:<5}                   ║")
    print(f"║  Actuators:    {model.nu:<5}                   ║")
    print(f"║  Geoms:        {model.ngeom:<5}                   ║")
    print("╚══════════════════════════════════════╝")
    print("\nKeyboard shortcuts:")
    print("  H → Home pose     G → Open hands    F → Close hands")
    print("  L → Lift up       D → Lift down")
    print("  Space → Pause     Backspace → Reset\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MABEL MuJoCo Viewer")
    parser.add_argument("--model", default=str(DEFAULT_XML),
                        help="Path to MJCF XML file")
    args = parser.parse_args()

    xml_path = Path(args.model)
    if not xml_path.is_absolute():
        xml_path = REPO_DIR / xml_path
    if not xml_path.exists():
        raise FileNotFoundError(f"Model file not found: {xml_path}")

    print(f"\nLoading: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data  = mujoco.MjData(model)

    print_info(model)

    # Set initial home pose
    set_home_pose(model, data)
    open_hands(model, data)

    # Launch passive viewer with key callbacks
    key_map = {
        ord('h'): lambda m, d: set_home_pose(m, d),
        ord('H'): lambda m, d: set_home_pose(m, d),
        ord('g'): lambda m, d: open_hands(m, d),
        ord('G'): lambda m, d: open_hands(m, d),
        ord('f'): lambda m, d: close_hands(m, d),
        ord('F'): lambda m, d: close_hands(m, d),
        ord('l'): lambda m, d: lift_up(m, d),
        ord('L'): lambda m, d: lift_up(m, d),
        ord('d'): lambda m, d: lift_down(m, d),
        ord('D'): lambda m, d: lift_down(m, d),
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.cam.distance  = 3.5
        viewer.cam.lookat[:] = [0, 0, 0.9]

        print("Viewer open. Press keys in viewer window: H/G/F/L/D")
        print("Close the viewer window to exit.\n")

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep real-time pace
            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

"""
simulation_bridge configuration — the MABEL_SIM_* mirror of hardware_bridge's
MABEL_HW_* surface. Everything is env-overridable so a service unit retunes
without code edits; `--instance` scopes every path/port so one computer can
host several simulated robots side by side.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MABEL_ROOT", os.path.dirname(_HERE))


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return int(default)


def _envs(name, default):
    return os.environ.get(name, default)


def _envb(name, default):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return v.strip().lower() in ("1", "true", "yes", "on")


# ── instance identity (multi-robot: several bridges on one computer) ─────────
# The default instance keeps the LEGACY paths so every existing consumer
# (sim_camera_server, view_robot, the control-center mirror) works untouched.
INSTANCE = _envs("MABEL_SIM_INSTANCE", "sim0")


def state_file(instance=None):
    inst = instance or INSTANCE
    if inst == "sim0":
        return _envs("MABEL_STATE_FILE", "/tmp/mabel_teleop_state.f64")
    return _envs("MABEL_STATE_FILE", f"/tmp/mabel_sim_{inst}.f64")


def lock_file(instance=None):
    inst = instance or INSTANCE
    return _envs("MABEL_SIM_LOCK", f"/tmp/mabel_simulation_bridge_{inst}.lock")


def rpc_port(instance=None):
    inst = instance or INSTANCE
    base = _envi("MABEL_SIM_BRIDGE_PORT", 8788)      # hand bridge uses 8779
    if inst == "sim0":
        return base
    try:
        return base + int("".join(c for c in inst if c.isdigit()) or 1)
    except ValueError:
        return base + 1


RPC_HOST = _envs("MABEL_SIM_BRIDGE_HOST", "127.0.0.1")

# ── the world ────────────────────────────────────────────────────────────────
DEFAULT_SCENE = _envs(
    "MABEL_SCENE",
    os.path.join(REPO, "simulation", "mabel_mujoco", "scenes", "mabel_home.xml"))

# ── command semantics (mirroring hardware_bridge base/arms behavior) ─────────
CMD_TIMEOUT = _envf("MABEL_SIM_CMD_TIMEOUT", 0.5)     # stale twist → zero base
EFFORT_TIMEOUT = _envf("MABEL_SIM_EFFORT_TIMEOUT", 0.15)  # stale τ → zero
WRENCH_TIMEOUT = _envf("MABEL_SIM_WRENCH_TIMEOUT", 0.25)  # stale push → zero
VMAX = _envf("MABEL_SIM_VMAX", 1.0)                   # m/s cap (like HW base)
WMAX = _envf("MABEL_SIM_WMAX", 2.0)                   # rad/s cap
MAX_CATCHUP = _envf("MABEL_SIM_MAX_CATCHUP", 0.05)    # s of physics per wake

# ── optional DDS layer (Ubuntu): the hardware contract under one namespace ──
ROS = _envs("MABEL_SIM_ROS", "auto")                  # auto|1|0
ROS_NS = _envs("MABEL_SIM_NS", "/sim")                # /sim2 for a second robot
ROS_RATE = _envf("MABEL_SIM_ROS_RATE", 50.0)          # joint_states publish Hz

# sim "sensor health" analogs: honest ambient values so the modality exists
HAND_TEMP_C = _envf("MABEL_SIM_HAND_TEMP", 25.0)
HAND_VOLT_V = _envf("MABEL_SIM_HAND_VOLT", 7.4)

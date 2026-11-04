"""
Embodiment registry (design-doc §1.3): ONE record describing the MABEL rig, merged
from the episode schema (actuator names/groups, DOF counts, cameras, lidar), the
static camera rig (`mabel_camera_rig.json`), the ActionSpec (§1.1), and — best-effort
— joint control ranges + effort (force) limits + gripper joints read from the
compiled MJCF. Downstream checks (health, sync, retarget) read from here so an
episode is interpreted the same regardless of source.
"""
import json
import os

from . import h5pool, catalog


def registry():
    reg = {"robot": "mabel"}
    # ── schema (from any episode) ──────────────────────────────────────────────
    sample = _sample_episode()
    if sample:
        with h5pool.reading(sample) as f:
            sj = json.loads(f.attrs.get("schema_json", "{}"))
        reg["actuator_names"] = sj.get("actuator_names", [])
        reg["actuator_groups"] = sj.get("actuator_groups", {})
        reg["dof"] = {"nq": sj.get("nq"), "nv": sj.get("nv"), "nu": sj.get("nu")}
        reg["cameras"] = sj.get("cameras", [])
        reg["lidar"] = sj.get("lidar", {})
        reg["depth_encoding"] = sj.get("depth_encoding", {})
    # ── static camera rig (intrinsics + mounts) ───────────────────────────────
    try:
        rp = os.path.join(os.path.dirname(__file__), "mabel_camera_rig.json")
        reg["camera_rig"] = json.load(open(rp))
    except Exception:
        reg["camera_rig"] = []
    # ── ActionSpec (§1.1): MABEL action/ctrl = absolute joint-position targets ──
    reg["action_spec"] = {"space": "joint_pos", "frame": "base",
                          "gripper": "continuous", "is_delta": False}
    # ── control + effort limits + gripper geometry from the MJCF (best-effort) ─
    reg["limits"] = _limits_from_model()
    reg["has_limits"] = reg["limits"] is not None
    return reg


def _sample_episode():
    try:
        for ds in catalog.full_catalog()["datasets"]:
            for ep in ds["episodes"]:
                return catalog.episode_path(ds["name"], ep["file"])
    except Exception:
        pass
    return None


def _limits_from_model():
    """Per-actuator ctrlrange (position limits) + forcerange (effort limits) + the
    gripper/hand joints, read from the compiled scene (cached .mjb via camcloud)."""
    try:
        import mujoco
        from . import camcloud
        m = camcloud._model()
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        grip = [n for n in names if n and ("hand" in n.lower() or "thumb" in n.lower()
                                           or "finger" in n.lower() or "grip" in n.lower())]
        return {
            "actuator_names": names,
            "ctrlrange": [[round(float(a), 4), round(float(b), 4)] for a, b in m.actuator_ctrlrange],
            "forcerange": [[round(float(a), 3), round(float(b), 3)] for a, b in m.actuator_forcerange],
            "gripper_joints": grip,
        }
    except Exception:
        return None

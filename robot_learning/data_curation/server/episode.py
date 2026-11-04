"""
Episode reader: metadata, the dense whole-episode binary bundle (for instant
timeline scrubbing / plots), single-frame robot_state, and the lidar block.

Camera RGB/depth are NOT touched here — those stream per-frame from video.py so a
long episode never loads gigabytes of images. The dense numeric topics are tiny
(a 1000-frame episode is joint_pos[1000,59]f32 ~= 236 KB) so they ship in one
request the frontend caches.
"""
import json
import struct

import numpy as np

from . import h5pool, joint_map

# Dense topics shipped in the bundle: (hdf5 key, output name). All coerced f32.
_DENSE = [
    ("obs/joint_pos", "joint_pos"),
    ("obs/joint_vel", "joint_vel"),
    ("obs/joint_torque", "joint_torque"),
    ("action/ctrl", "action_ctrl"),
    ("obs/base_twist", "base_twist"),
    ("obs/lidar", "lidar"),
    ("obs/pose/chassis", "pose/chassis"),
    ("obs/pose/head", "pose/head"),
    ("obs/pose/left_palm", "pose/left_palm"),
    ("obs/pose/right_palm", "pose/right_palm"),
    ("teleop/left_palm_target", "teleop/left_palm_target"),
    ("teleop/right_palm_target", "teleop/right_palm_target"),
    ("teleop/grasp", "teleop/grasp"),
    ("teleop/base_cmd", "teleop/base_cmd"),
    ("teleop/look_at", "teleop/look_at"),
    ("teleop/mode", "teleop/mode"),
    ("time/sim", "time_sim"),
]


def _schema(f):
    if "schema_json" in f.attrs:
        try:
            return json.loads(f.attrs["schema_json"])
        except Exception:
            return {}
    return {}


def meta(path):
    with h5pool.reading(path) as f:
        a = f.attrs
        schema = _schema(f)
        num = int(a["num_frames"]) if "num_frames" in a else 0
        shapes = {}
        f.visititems(lambda n, o: shapes.__setitem__(
            n, {"shape": list(o.shape), "dtype": str(o.dtype)})
            if hasattr(o, "shape") else None)
        fps = 0.0
        t0 = t1 = 0.0
        if "time/sim" in f and num > 1:
            ts = f["time/sim"]
            t0, t1 = float(ts[0]), float(ts[-1])
            fps = (num - 1) / (t1 - t0) if t1 > t0 else 0.0
        return {
            "num_frames": num,
            "fps": round(fps, 4),
            "t0": t0, "t1": t1,
            "task": a["task"].item() if "task" in a and hasattr(a["task"], "item")
                    else (a["task"] if "task" in a else ""),
            "created": a["created"] if "created" in a else "",
            "schema": schema,
            "actuator_names": schema.get("actuator_names", []),
            "actuator_groups": schema.get("actuator_groups", {}),
            "cameras": schema.get("cameras", []),
            "lidar": schema.get("lidar", {}),
            "shapes": shapes,
        }


def dense_bundle(path):
    """Pack the dense numeric topics into one little-endian binary blob:

        u32 header_len | <header JSON utf-8> | f32 data blocks (in header order)

    header = {T, fps, topics:[{name, dim, offset, count}]}  (offset/count in floats)
    """
    with h5pool.reading(path) as f:
        num = int(f.attrs["num_frames"]) if "num_frames" in f.attrs else 0
        blocks = []
        topics = []
        offset = 0
        fps = 0.0
        if "time/sim" in f and num > 1:
            ts = f["time/sim"]
            dt = float(ts[-1]) - float(ts[0])
            fps = (num - 1) / dt if dt > 0 else 0.0
        for key, name in _DENSE:
            if key not in f:
                continue
            arr = np.asarray(f[key][:], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1)
            T, dim = arr.shape[0], int(np.prod(arr.shape[1:]))
            flat = np.ascontiguousarray(arr.reshape(T, dim), dtype=np.float32)
            blocks.append(flat.tobytes())
            topics.append({"name": name, "dim": dim,
                           "offset": offset, "count": T * dim})
            offset += T * dim
        header = {"T": num, "fps": round(fps, 4), "topics": topics}
    hb = json.dumps(header).encode("utf-8")
    # pad header to a 4-byte boundary so the data start (4 + len(hb)) is aligned
    # for the browser's Float32Array view (which requires a 4-aligned byteOffset)
    pad = (-(4 + len(hb))) % 4
    hb += b" " * pad
    return struct.pack("<I", len(hb)) + hb + b"".join(blocks)


def frame_state(path, i):
    """Single-frame robot_state JSON, shaped exactly like the teleop bridge's
    ROBOT_STATE payload so the existing rig viewer animates it unchanged."""
    with h5pool.reading(path) as f:
        num = int(f.attrs["num_frames"]) if "num_frames" in f.attrs else len(f["action/ctrl"])
        i = max(0, min(int(i), num - 1))
        schema = _schema(f)
        names = schema.get("actuator_names", [])
        jp = np.asarray(f["obs/joint_pos"][i], dtype=np.float32) if "obs/joint_pos" in f else None
        ctrl = np.asarray(f["action/ctrl"][i], dtype=np.float32) if "action/ctrl" in f else None
        src = jp if jp is not None else ctrl
        joints = joint_map.glb_from_actuators(src, names) if (src is not None and names) else {}
        base = {}
        if "obs/pose/chassis" in f:
            base = joint_map.base_from_chassis(f["obs/pose/chassis"][i])
        out = {
            "type": "robot_state",
            "frame": i,
            "t": float(f["time/sim"][i]) if "time/sim" in f else i,
            "jointPositions": joints,
            "base": base,
        }
        for key, name in (("obs/joint_pos", "joint_pos"),
                          ("obs/joint_vel", "joint_vel"),
                          ("obs/joint_torque", "joint_torque"),
                          ("action/ctrl", "ctrl"),
                          ("obs/base_twist", "base_twist")):
            if key in f:
                out[name] = [float(x) for x in np.asarray(f[key][i]).ravel()]
        poses = {}
        for b in ("chassis", "head", "left_palm", "right_palm"):
            k = f"obs/pose/{b}"
            if k in f:
                poses[b] = [float(x) for x in np.asarray(f[k][i]).ravel()]
        out["poses"] = poses
        tel = {}
        for key, name in (("teleop/left_palm_target", "left_palm_target"),
                          ("teleop/right_palm_target", "right_palm_target"),
                          ("teleop/grasp", "grasp"),
                          ("teleop/base_cmd", "base_cmd"),
                          ("teleop/look_at", "look_at"),
                          ("teleop/mode", "mode")):
            if key in f:
                tel[name] = [float(x) for x in np.asarray(f[key][i]).ravel()]
        out["teleop"] = tel
        return out, num


def trajectory(path, tf, stride=2):
    """Downsampled [x,y,z] path of a TF frame over the whole episode (pose arrays
    are tiny, so this is cheap even dataset-wide)."""
    key = "obs/pose/" + tf
    with h5pool.reading(path) as f:
        if key not in f:
            return []
        s = max(1, int(stride))
        arr = np.asarray(f[key][::s, :3], dtype=np.float32)
    return arr.tolist()


def lidar_bin(path, start=None, end=None):
    """Compact lidar: u32 T, u16 n_rays, f32 angle_res_deg, f32 scale, then
    uint16[T*n_rays] millimetres (0 = no-hit / cutoff)."""
    with h5pool.reading(path) as f:
        if "obs/lidar" not in f:
            return b""
        lid = f["obs/lidar"]
        T = lid.shape[0]
        s = 0 if start is None else max(0, int(start))
        e = T if end is None else min(T, int(end))
        arr = np.asarray(lid[s:e], dtype=np.float32)
        schema = _schema(f).get("lidar", {})
        n_rays = int(arr.shape[1]) if arr.ndim == 2 else 0
        res = float(schema.get("angle_res_deg", 1.0))
        scale = 1000.0
        # cutoff/no-hit -> 0; clip to uint16 range (65.535 m)
        mm = np.clip(arr, 0.0, 65.535) * scale
        mm[arr <= 0] = 0
        u16 = mm.astype(np.uint16)
        head = struct.pack("<IHff", e - s, n_rays, res, scale)
        return head + u16.tobytes()

"""
sim_sensor_source.py — MuJoCo sensor source for the episode recorder.

Renders every robot camera (RGB + head depth) and reads the planar LiDAR, the
full generalized state (qpos/qvel/ctrl), and the chassis odometry out of one
MuJoCo MjData. Hands the recorder a `sample` dict (see recorder.py).

It mirrors what the REAL robot would publish (real cameras, a real ZED depth
stream, a real 2-D LiDAR, joint encoders, wheel odometry), so the EpisodeRecorder
that consumes it is identical in sim and on hardware — only this source is swapped
for a hardware/ROS one.

NOTE (macOS): MuJoCo's off-screen render is a fixed ~per-frame GPU stall, so
rendering all cameras per recorded frame slows the sim while recording. The
recorder samples in SIM TIME, so the episode stays consistent; lower --record-fps
or drop cameras if you need the sim closer to real time during capture.
"""
from __future__ import annotations

import numpy as np
import mujoco

# All robot cameras present in the scene (verified against mabel_home.xml).
RGB_CAMERAS = ["zed_left", "zed_right",            # stereo head
               "l_wrist_cam", "r_wrist_cam",       # wrists
               "body_cam", "body_front_cam", "body_back_cam"]   # body / mobile base
DEPTH_CAMERAS = ["zed_left", "zed_right"]          # head stereo depth (the ZED gives depth)


class SimSensorSource:
    def __init__(self, model, data, chassis_body: str,
                 get_control, get_desired=None, rgb_res=(424, 240), depth_res=(424, 240)):
        self.model, self.data = model, data
        self.chassis_body = chassis_body
        self.get_control = get_control
        # Returns {mj_joint_name: desired_rad} — the controller set-points (the
        # command INTO the controller). None → no desired channel recorded.
        self.get_desired = get_desired
        self.rgb_cameras = [c for c in RGB_CAMERAS if self._has_cam(c)]
        self.depth_cameras = [c for c in DEPTH_CAMERAS if self._has_cam(c)]
        self.rgb_res, self.depth_res = tuple(rgb_res), tuple(depth_res)
        self._rgb = mujoco.Renderer(model, rgb_res[1], rgb_res[0])
        self._depth = mujoco.Renderer(model, depth_res[1], depth_res[0])
        self._depth.enable_depth_rendering()
        # Tracked body world poses (curation-format obs/pose/*): the same set
        # the IL data-collection pipeline logs. Missing bodies are skipped.
        try:
            from mabel import config as _C
            _tracked = {"chassis": chassis_body, "head": _C.HEAD_BODY,
                        "left_palm": _C.LEFT_PALM, "right_palm": _C.RIGHT_PALM}
        except Exception:
            _tracked = {"chassis": chassis_body}
        self._tracked = {}
        for key, name in _tracked.items():
            try:
                self._tracked[key] = int(model.body(name).id)
            except Exception:
                pass
        # LiDAR ray sensor addresses (lidar_range000 … lidar_rangeNNN), in order.
        self._lidar_adr = []
        i = 0
        while True:
            try:
                s = model.sensor(f"lidar_range{i:03d}")
            except Exception:
                break
            self._lidar_adr.append(int(s.adr[0]))
            i += 1
        self._lidar_adr = np.asarray(self._lidar_adr, dtype=int)

    def _has_cam(self, name: str) -> bool:
        try:
            self.model.camera(name); return True
        except Exception:
            return False

    def meta(self) -> dict:
        m = self.model
        return {
            "source": "sim",
            "rgb_cameras": self.rgb_cameras,
            "depth_cameras": self.depth_cameras,
            "depth_units": "uint16_millimetres",
            "lidar_rays": int(len(self._lidar_adr)),
            "rgb_res": list(self.rgb_res), "depth_res": list(self.depth_res),
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
            "joint_names": [m.joint(i).name for i in range(m.njnt)],
            "actuator_names": [m.actuator(i).name for i in range(m.nu)],
        }

    def sample(self) -> dict:
        d = self.data
        rgb = {}
        for cam in self.rgb_cameras:
            self._rgb.update_scene(d, camera=cam)
            rgb[cam] = self._rgb.render().copy()
        depth = {}
        for cam in self.depth_cameras:
            self._depth.update_scene(d, camera=cam)
            depth[cam] = self._depth.render().copy()
        lidar = d.sensordata[self._lidar_adr].copy() if len(self._lidar_adr) else None
        b = d.body(self.chassis_body)
        base = (float(b.xpos[0]), float(b.xpos[1]), float(b.xpos[2]),
                float(b.xquat[0]), float(b.xquat[1]), float(b.xquat[2]), float(b.xquat[3]))
        # Desired joint pose: a full qpos-length vector that defaults to the
        # realized pose and is overwritten at each commanded joint's qpos address,
        # so qpos_des and qpos are aligned per-joint for imitation learning.
        qpos_des = None
        if self.get_desired is not None:
            qpos_des = d.qpos.copy()
            for jname, val in (self.get_desired() or {}).items():
                try:
                    qpos_des[self.model.joint(jname).qposadr[0]] = float(val)
                except Exception:
                    pass
        # Curation-format extras: tracked body poses, actuator force, base twist.
        poses = {}
        for key, bid in self._tracked.items():
            poses[key] = np.concatenate([d.xpos[bid], d.xquat[bid]]).astype(np.float32)
        base_twist = None
        cid = self._tracked.get("chassis")
        if cid is not None:
            cvel = d.cvel[cid]                      # [ang(3), lin(3)] world
            base_twist = np.array([cvel[3], cvel[4], cvel[2]], dtype=np.float32)
        return {
            "rgb": rgb, "depth": depth, "lidar": lidar,
            "qpos": d.qpos.copy(), "qpos_des": qpos_des,
            "qvel": d.qvel.copy(), "ctrl": d.ctrl.copy(),
            "act_force": d.actuator_force.copy(),
            "poses": poses, "base_twist": base_twist,
            "base": base, "control": self.get_control(),
        }

    def close(self):
        for r in (self._rgb, self._depth):
            try: r.close()
            except Exception: pass
